<paper 0>
# Executable Code Actions Elicit Better LLM Agents 

Xingyao Wang ${ }^{1}$ Yangyi Chen ${ }^{1}$ Lifan Yuan ${ }^{1}$ Yizhe Zhang ${ }^{2}$ Hao Peng ${ }^{1}$ Heng Ji ${ }^{1}$


#### Abstract

Large Language Model (LLM) agents, capable of performing a broad range of actions, such as invoking tools and controlling robots, show great potential in tackling real-world challenges. LLM agents are typically prompted to produce actions by generating JSON or text in a pre-defined format, which is usually limited by constrained action space (e.g., the scope of pre-defined tools) and restricted flexibility (e.g., inability to compose multiple tools). This work proposes to use executable Python code to consolidate LLM agents' actions into a unified action space (CodeAct). Integrated with a Python interpreter, CodeAct can execute code actions and dynamically revise prior actions or emit new actions upon new observations through multi-turn interactions. Our extensive analysis of 17 LLMs on APIBank and a newly curated benchmark shows that CodeAct outperforms widely used alternatives (up to $20 \%$ higher success rate). The encouraging performance of CodeAct motivates us to build an open-source LLM agent that interacts with environments by executing interpretable code and collaborates with users using natural language. To this end, we collect an instruction-tuning dataset CodeActInstruct that consists of $7 \mathrm{k}$ multi-turn interactions using CodeAct. We show that it can be used with existing data to improve models in agent-oriented tasks without compromising their general capability. CodeActAgent, finetuned from Llama2 and Mistral, is integrated with Python interpreter and uniquely tailored to perform sophisticated tasks (e.g., model training) using existing libraries and autonomously self-debug ${ }^{1}$.


[^0]
## 1. Introduction

Large Language Models (LLMs) have emerged as a pivotal breakthrough in natural language processing (NLP). When augmented with action modules that allow access to APIs, their action space expands beyond conventional text processing, allowing LLMs to acquire capabilities such as tool invocation and memory management (Mialon et al., 2023; Schick et al., 2023) and venture into real-world tasks such as controlling robots (Ahn et al., 2022; Huang et al., 2023; Ma et al., 2023) and performing scientific experiments (Bran et al., 2023).

We inquire: how to effectively expand LLM agents' action space for solving complex real-world problems? Much existing research has examined using text (Yao et al., 2022b; Park et al., 2023, inter alia) or JSON (Qin et al., 2023b; Chase, 2022, inter alia) to produce actions (e.g., tool uses in Fig. 1 top left). However, both methods typically suffer from constrained scope of action spaces (actions are usually tailored for specific tasks) and restricted flexibility (e.g., inability to compose multiple tools in a single action). As an alternative approach, several work (Liang et al., 2022; Singh et al., 2023; Wang et al., 2023a) demonstrate the potential of using LLMs to generate code to control robots or game characters. However, they typically rely on pre-specified control primitives and hand-engineered prompts and, more importantly, struggle to dynamically adjust or emit actions based on new environmental observation and feedback.

This work proposes CodeAct, a general-purpose framework that allows LLMs to generate executable Python code as actions (Fig. 1 top right). CodeAct is designed to handle a variety of applications and comes with unique advantages:

(1) Integrated with a Python interpreter, CodeAct can execute code actions and dynamically adjust prior actions or emit new action based on observations (e.g., code execution results) it receives through multiple turns of interactions.

(2) Code actions allow LLM to leverage existing software packages. CodeAct can use readily available Python packages for an expanded action space instead of handcrafted task-specific tools (Yuan et al., 2023; Shen et al., 2023). It also allows LLM to use automated feedback (e.g., error messages) implemented in most software to improve task-solving by self-debugging its generated

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-02.jpg?height=1181&width=1702&top_left_y=220&top_left_x=187)

Instruction: Determine the most cost-effective country to purchase the

Available APIs

[3] estimate_final_price(converted_price: float, shipping_cost: float) -> float

[4] lookup_phone_price(model: str, country: str) -> float

Japan, Germany, and India.

[5] estimate_shipping_cost(destination_country: str) -> float

CodeAct: LLM Agent using [Code] as Action

Think I should calcoule for find

Think I should calculate the phone price in USD for each country, then find the

R. As

Think most cost-effective country

度/> Action Text: lookup_rates, Germany $\begin{aligned} & \text { Ten: } \\ & \text { JSON: "tool": "lookup_rates", "country": "Germany" \} }\end{aligned}$

(3) Environment $1.1,0.19$

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-02.jpg?height=76&width=705&top_left_y=461&top_left_x=1165)

\%“/> Action Text: lookup_phone_price, CodeAct 1, Germany

exchange_rate, tax_rate $=$ lookup_rates(country)

ocal_price = lookup_phone_price("xAct 1", country)

local_price, exchange_rand_tax

(35) Environment 700
“/> Action Text: convert_and_tax, 700, 1.1, 0.19

shipping_cost = estimate_shipping_cost(country)

shipping_cost) final_prices[country] = final_price

(35) Environment 916.3

most_cost_effective_country $=$ min(final_prices, key=final_prices.get)

[... interactions omitted (look up shipping cost and calculate final price) ...]

“/> Action $\begin{aligned} & \text { Text: lookup_rates, Japan } \\ & \text { JSON: }\{\text { "tool": "lookup_rates }\end{aligned}$

[... interactions omitted (calculate final price for all other countries)...]

The most cost-effective country to purchase the smartphone model is Japan
with price 904.00 in USD.

Response The most cost-effective country to purchase the smartphone model is Japan
![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-02.jpg?height=448&width=1676&top_left_y=948&top_left_x=194)

Figure 1: Comparison between CodeAct and Text / JSON as action. (top) Illustrative example comparing different actions. (bottom) Quantitative results on $\mathrm{M}^{3}$ ToolEval (\$2.3).

code (Chen et al., 2023b; Wang et al., 2023d).

(3) Code data is widely used in pre-training today's LLMs (Yang et al., 2024b). These models are already familiar with structured programming languages, allowing costeffective adoption of CodeAct.

(4) Compared to JSON and text with a pre-defined format, code inherently supports control and data flow, allowing for the storage of intermediate results as variables for reuse and the composition of multiple tools to perform complex logical operations (e.g., if-statements, for-loops) with one piece of code, thereby unlocking LLMs' potential to tackle complex tasks by leveraging its pre-trained knowledge of programming. In Fig. 1, an LLM using with CodeAct (top right) can apply the same sequence of tools (e.g., passing one tool's output as input to another tool using the data flow feature) to all inputs through for-loops (i.e., control flow feature) with one action; while text or JSON have to take action for every input (top left).

Our extensive experiments with 17 LLMs (including both open-source and proprietary ones) confirm the above bene- fits (3 \& 4) of CodeAct. To demonstrate benefit (3), our first experiment ( $\$ 2.2$ ) compares CodeAct to baselines on basic tasks involving atomic tool use (i.e., only one tool is used per action), ablating the control and data flow advantage offered by CodeAct. The results show that, for most LLMs, CodeAct achieves comparable or better performance than the baselines. CodeAct's performance gains are more prominent on complex tasks, as demonstrated in our second experiment (benefit 4). We curate a new benchmark consisting of 82 human-curated tasks that typically require multiple calls to multiple tools in multi-turn interactions ( $\mathrm{M}^{3}$ ToolEval; §2.3). Problems in this benchmark often require intricate coordination and composition of multiple tools. With its strengths in control and data flow, CodeAct achieves up to a $20 \%$ absolute improvement over baselines on the success rate of solving the problems while requiring up to $30 \%$ fewer actions. These performance gains widen as the capabilities of the LLMs increase (Fig. 1 bottom).

The promising performance of CodeAct motivates an open-source LLM agent that can effectively act through CodeAct, and collaborate with humans through natural lan-

Table 1: The benefit of CodeAct compared to using Text/JSON for LLM action.

|  | CodeAct for LLM action | JSON or Text for LLM action |
| :---: | :---: | :---: |
| Availability of Data | ![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-03.jpg?height=56&width=642&top_left_y=319&top_left_x=646) | $X_{\text {Data curation required for particular format }}$ |
| Complex Operation (e.g., looping, <br> composition of multiple tools) | $\checkmark$ Natively supported via control and data flow | $X_{\text {Requires careful engineering if feasible (e.g. }}$ <br> define new tools to mimic if-statement) |
| Availability of Tools | $\checkmark$ Can directly use existing software packages ${ }^{2}$ | $X_{\text {Requires human effort to curate tools from }}$ <br> scratch or existing software |
| Automated Feedback | ![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-03.jpg?height=112&width=642&top_left_y=552&top_left_x=646) | $X_{\text {Requires human effort to provide feedback or re- }}$ <br> route feedback from the underlying programming <br> language used to implement the tools |

${ }^{1}$ Including code demonstrating useful behaviors for LLM agents (e.g., task decomposition, coordination of multiple function calls to different tools).

${ }^{2}$ Human-written Python packages covering a wide range of applications are available on https://pypi.org/.

${ }^{3}$ For example, in Python, errors and exceptions (https://docs.python.org/3/tutorial/errors. html) are available. Most software provides error messages in natural language to help human programmers debug their code. CodeAct enables LLM to use them directly.

guage. To this end, we collect an instruction-tuning dataset CodeActInstruct consisting of $7 \mathrm{k}$ high-quality multi-turn interaction trajectories with CodeAct (§3.1). CodeActInstruct is motivated by a general agent framework consisting of agent, user, and environments (Fig. 2) and focuses on agent-environment interactions with the computer (information seeking, software package use, external memory) and the physical world (robot planning). On CodeActInstruct, we perform careful data selection to promote the capability of improving from multi-turn interaction (e.g., self-debug). We show that CodeActInstruct can be used with commonly used instruction tuning data to improve the models' performance in agent tasks without compromising their general capabilities (e.g., knowledge-based QA, coding, instruction following, §3.2). Our model, dubbed CodeActAgent, is finetuned from LLaMA-2 (Touvron et al., 2023) and Mistral-7B (Jiang et al., 2023) and improves on out-of-domain agent tasks with not only CodeAct, but also text action in a pre-defined format ( $\$ 3.2$ ).

CodeAct can further benefit from multi-turn interactions and existing software (benefit $1 \& 2, \S 2.4$ ). As shown in Fig. 3, CodeActAgent, designed for seamless integration with Python, can carry out sophisticated tasks (e.g., model training, data visualization) using existing Python packages. Error messages from the environment further enable it to rectify errors autonomously through self-debugging in multiturn interaction. Thanks to LLM's extensive programming knowledge acquired during pre-training, these are achieved without needing in-context demonstrations, reducing the human efforts for adapting CodeActAgent to different tasks.

## 2. CodeAct Makes LLMs Better Agents

In this section, we first describe CodeAct framework (\$2.1) and provide empirical evidence that supports the choice of CodeAct. We focus on Python as the programming language for CodeAct due to its popularity (ranked top-1 at (TIOBE Index, 2024)) and numerous open-source packages. We aim to answer several research questions (RQs) using
17 off-the-shelf LLMs. In $\S 2.2$, we examine RQ1: Does LLMs' familiarity with code due to a large amount of code pre-training data bring CodeAct advantages over text and JSON? We discuss RQ2 in §2.3: Does CodeAct benefit from Python's innate control and data flow feature in complex problems? Finally, as an additional benefit, we discuss how using CodeAct further enhances LLM agents by enabling multi-turn interactions and allowing them to access existing software in $\S 2.4$ and Fig. 3.

### 2.1. What is CodeAct?

In Fig. 2, we first introduce a general multi-turn interaction framework for LLM agents' real-world usage that considers three roles (Yang et al., 2024c): agent, user, and environment. We define interaction as the information exchange between the agent and an external entity (user or environment). For each turn of interaction, the agent receives an $o b$ servation (input) either from the user (e.g., natural language instruction) or the environment (e.g., code execution result), optionally planning for its action through chain-of-thought (Wei et al., 2022), and emits an action (output) to either user in natural language or the environment. CodeAct employs Python code to consolidate all actions for agent-environment interaction. In CodeAct, each emitted action to the environment is a piece of Python code, and the agent will receive outputs of code execution (e.g., results, errors) as observation. We include an example prompt of CodeAct in $\S \mathrm{E}$.

### 2.2. CodeAct Shows the Promise as a Strong Tool Use Framework

In this section, we perform a controlled experiment to understand which format (text, JSON, CodeAct) is more likely to lead an LLM to generate correct atomic tool calls. The performance in this experiment reflects LLM's familiarity with the corresponding format. We hypothesize that using CodeAct to call tools is a more natural way to use tools for the models, which typically have extensive exposure to

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-04.jpg?height=618&width=1700&top_left_y=230&top_left_x=191)

Figure 2: General agent multi-turn interaction framework that describes the role of CodeAct and motivates the construction of our data mixture. CodeActInstruct focuses on the agent-environment interactions and specifically filters for the selfimproved planning behavior, while general conversation data we include focuses on agent-user interaction (\$3.1).

code data during their training.

Setup. We re-purpose API-Bank (Li et al., 2023) and test LLMs' API-calling performance, comparing CodeAct, JSON, and text actions. For each evaluation instance, we instruct LLM to generate one atomic tool call in the format of a Python function call, JSON object, or text expression in a pre-defined format. A concrete example is shown in Tab. A.6. We use API-Bank's level-1 instructions and the provided toolset. To evaluate API-calling, we follow their correctness metric, matching the ground-truth API outputs with the actual model-generated API's execution outputs.

Results. We present results in Tab. 2. For most LLMs, CodeAct achieves comparable or better performance even in atomic actions (the simplistic tool use scenario) where its control and data flow strengths are ablated. Compared to closed-source LLMs, CodeAct's improvements are more prominent in open-source models. Furthermore, code data is usually more accessible for fine-tuning open-source LLMs than the specialized JSON or text tool-calling format. Although JSON is consistently weaker than other approaches for open-source models, it achieves decent performance with closed-source LLMs, indicating that these closed-source models may have gone through targeted fine-tuning toward their JSON capabilities. These results suggest optimizing for CodeAct is a better route for open-source LLMs than alternatives to improve their tool-use capabilities, as they already show good initial CodeAct capability due to extensive exposure to code data during pre-training.

### 2.3. CodeAct Gets More Done with Fewer Interactions

In this section, we investigate whether LLM agents can benefit from the control and data flow of code on problems that require complex patterns of tool use.

$\mathbf{M}^{3}$ ToolEval. As shown in Tab. A.7, to the best of our knowledge, no existing tool-use benchmarks contain complex tasks requiring the composition of multiple tools while supporting evaluating different action formats. Hence, we curate a benchmark $\mathrm{M}^{3}$ ToolEval to fill this gap, which evaluates LLMs' capabilities in solving complex tasks that typically require multiple calls to multiple tools in multi-turn interactions. It contains 82 human-curated instances, spanning tasks including web browsing, finance, travel itinerary planning, science, and information processing. Each domain is accompanied by a unique set of manually crafted tools. We intentionally keep the prompt simple (examples in $\S \mathrm{F}$ ) and avoid providing any demonstration to test the LLM's zero-shot ability to use tools, similar to how a novice user without knowledge of few-shot prompting would use the model.

Setup. We allow the model to generate fully functional Python code that enables control and data flow (e.g., ifstatement, for-loop). We follow the action format for JSON and text described in Tab. A.6. Within each turn, the model can either emit an action or propose an answer to be verified by an exact match with the ground-truth solution. The interaction will terminate when a maximum of 10 interaction turns are reached or a correct solution has been submitted, similar to (Wang et al., 2023e).

Metric. We measure the success rate by calculating the percentage of the model proposed answers that match the ground-truth solutions. We also include the avg. turns metric: the average number of turns on all evaluated instances.

Quantitative Results on $\mathbf{M}^{3}$ ToolEval. We include full results in Tab. 3 and a subset of results for visualization in

Table 2: Atomic API call correctness on APIBank. The best performance is bolded, and the second-best is underlined.

| Format of Action | Correctness $(\%, \uparrow)$ |  |  |
| :---: | :---: | :---: | :---: |
|  | CodeAct | JSON | Text |
| Open-source LLMs |  |  |  |
| CodeLlama-7b-Instruct-hf | $\underline{12.5}$ | 12.0 | 17.0 |
| CodeLlama-13b-Instruct-hf | $\underline{11.8}$ | 7.8 | 14.0 |
| CodeLlama-34b-Instruct-hf | $\overline{17.3}$ | 12.0 | $\underline{16.8}$ |
| Llama-2-7b-chat-hf | 28.8 | 11.3 | $\underline{25.8}$ |
| Llama-2-13b-chat-hf | 38.1 | 8.5 | $\overline{37.3}$ |
| Llama-2-70b-chat-hf | $\underline{35.6}$ | 14.3 | 37.6 |
| Mistral-7B-Instruct-v0.1 | $\underline{2.5}$ | 2.3 | 3.0 |
| lemur-70b-chat-v1 | 58.6 | 46.6 | $\underline{56.1}$ |
| Closed-source LLMs |  |  |  |
| claude-2 | 76.7 | 59.4 | $\underline{73.7}$ |
| claude-instant-1 | 75.2 | 64.9 | $\overline{73.2}$ |
| gemini-pro | 70.4 | 73.2 | $\overline{71.2}$ |
| $g p t-3.5-t u r b o-0613$ | 74.4 | $\underline{73.9}$ | 73.4 |
| gpt-3.5-turbo-1106 | $\underline{75.4}$ | $\overline{78.4}$ | 73.4 |
| gpt $-4-0613$ | $\overline{75.4}$ | 82.0 | 74.4 |
| gpt-4-1106-preview | $\overline{76.7}$ | 82.7 | 73.4 |
| text-davinci-002 | $\overline{69.2}$ | $\underline{59.6}$ | 57.4 |
| text-davinci-003 | $\underline{75.4}$ | $\overline{76.9}$ | 69.7 |
| Frequency of Best-Performing Format $\uparrow$ |  |  |  |
| Open-source | 4 | 0 | $\underline{4}$ |
| Closed-source | $\underline{4}$ | 5 | $\overline{0}$ |
| Overall | $\overline{8}$ | $\underline{5}$ | 4 |

Table 3: Success rates (higher the better) and average turns required per instance (lower the better) on $\mathrm{M}^{3}$ ToolEval. The best results for each model are bolded, and the second-best ones are underlined.

| Format of Action | Success Rate $(\%, \uparrow)$ |  |  | Avg. Turns $(\downarrow)$ |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | CodeAct | JSON | Text | CodeAct | JSON | Text |
| Open-source LLMs |  |  |  |  |  |  |
| CodeLlama-7b-Instruct-hf | 4.9 | $\underline{2.4}$ | $\underline{2.4}$ | 9.7 | $\underline{9.9}$ | $\underline{9.9}$ |
| CodeLlama-13b-Instruct-hf | 4.9 | $\overline{4.9}$ | $\overline{4.9}$ | $\underline{9.8}$ | $\underline{9.8}$ | $\overline{9.7}$ |
| CodeLlama-34b-Instruct-hf | 2.4 | $\underline{0.0}$ | $\underline{0.0}$ | 9.9 | $\underline{10.0}$ | 10.0 |
| Llama-2-7b-chat-hf | 0.0 | $\underline{1.2}$ | 2.4 | 8.9 | $\underline{9.5}$ | 9.6 |
| Llama-2-13b-chat-hf | 0.0 | $\overline{0.0}$ | 0.0 | 9.7 | $\underline{10.0}$ | $\underline{10.0}$ |
| Llama-2-70b-chat-hf | 11.0 | $\underline{3.7}$ | $\underline{3.7}$ | 9.1 | $\underline{9.8}$ | $\underline{9.8}$ |
| Mistral-7B-Instruct-v0.1 | 0.0 | $\overline{3.7}$ | $\underline{1.2}$ | 10.0 | $\overline{9.8}$ | $\underline{9.9}$ |
| lemur-70b-chat-v1 | $\underline{13.4}$ | 15.9 | 12.2 | 9.1 | $\underline{9.3}$ | 9.4 |
| Closed-source LLMs |  |  |  |  |  |  |
| claude-2 | 54.9 | $\underline{39.0}$ | 29.3 | 7.2 | $\underline{8.3}$ | 8.5 |
| claude-instant-1 | 20.7 | 31.7 | $\underline{24.4}$ | $\underline{8.8}$ | 8.6 | 8.9 |
| gemini-pro | 22.0 | $\underline{19.5}$ | $\overline{11.0}$ | 8.8 | $\underline{9.1}$ | 9.5 |
| $g p t-3.5-t u r b o-0613$ | 51.2 | $\underline{26.8}$ | 20.7 | 7.0 | $\overline{8.8}$ | 9.2 |
| gpt-3.5-turbo-1106 | 29.3 | $\underline{15.9}$ | 14.6 | 8.4 | $\underline{9.0}$ | $\underline{9.0}$ |
| $g p t-4-0613$ | 67.1 | $\underline{56.1}$ | 45.1 | 6.6 | $\underline{7.6}$ | 8.0 |
| gpt-4-1106-preview | 74.4 | $\overline{52.4}$ | $\underline{53.7}$ | 5.5 | $\overline{7.6}$ | 7.7 |
| text-davinci-002 | $\underline{4.9}$ | $\underline{4.9}$ | $\overline{8.5}$ | $\underline{9.7}$ | $\overline{9.8}$ | 9.6 |
| text-davinci-003 | $2 \overline{20.7}$ | $\underline{18.3}$ | 7.3 | $\underline{9.2}$ | 9.0 | 9.6 |


|  | Frequency of Best-performing Format $\uparrow$ |  |  |  |  |  |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-source | $\mathbf{5}$ | $\underline{4}$ | 3 | $\mathbf{6}$ | $\underline{1}$ | $\underline{1}$ |
| Closed-source | $\mathbf{7}$ | $\underline{1}$ | $\underline{1}$ | $\mathbf{6}$ | $\underline{2}$ | 1 |
| Overall | $\mathbf{1 2}$ | $\underline{5}$ | 4 | $\mathbf{1 2}$ | $\underline{3}$ | 2 |

Fig. 1. CodeAct generally has a higher task success rate (12 out of 17 evaluated LLMs), similar to the trend in $\S 2.2$. Moreover, using CodeAct requires a lower average number of turns ( 12 out of 17 evaluated LLMs). For example, the best model gpt-4-1106-preview achieves a $20.7 \%$ absolute improvement compared to the next best action format (text) while requiring 2.1 fewer interaction turns on average. However, there is still a significant gap in terms of absolute CodeAct performance between open- and closed-source LLMs as the best open-source model achieving $13.4 \%$ while the best closed-source model gpt-4-1106-preview $74.4 \%$. This is potentially due to open-source models' weak task-solving capability and inability to follow complex instructions without demonstration, suggesting an urgent need to improve open-source LLMs for practical, real-world tasks under the zero-shot setting.

### 2.4. CodeAct Benefits from Multi-turn Interactions and Existing Software Packages

In Fig. 3, we show how an LLM agent can integrate with Python (i.e., CodeActAgent we trained in §3.2) and use existing software to perform complex tasks in multi-turn interactions. Thanks to its extensive knowledge of Python learned during pre-training, the LLM agent can automatically import the correct Python libraries to solve tasks without requiring user-provided tools or demonstrations. As illustrated in Fig. 3, CodeActAgent can use Pandas to download and process tabular data, use Scikit-Learn for machine learning train-test data split and regression model training, and use Matplotlib for data visualization. Furthermore, using the interactive Python interpreter for code execution allows automated error messages that help the LLM agent 'self-debug' their actions in a multi-turn interaction and eventually complete the human user's request correctly.

## 3. Empowering Open-source LLM Agent to be Better at CodeAct

The promising results achieved by CodeAct motivate us to build an open-source LLM agent that can both interact with environments through CodeAct and communicate with humans using language. To improve open-source LLMs' CodeAct capability, in §3.1, we introduce CodeActInstruct, an instruction finetuning dataset that contains agent-environment interaction trajectories. We discuss data selection procedures in $\S 3.1$ to promote improvement from interaction behavior. Additionally, we show that CodeAct can be used together with existing agent-user conversation data (\$3.1) to balance the dialog capability of the resulting LLM. Our model CodeActAgent, finetuned from LLaMA-2 (Touvron et al., 2023) and Mistral-7B (Jiang et al., 2023) on a mixture of CodeActInstruct and general conversations, improves CodeAct performances without hurting LLM's general performance on a diverse suite of tasks (§3.2).

### 3.1. CodeActInstruct: Agent-Environment Interactions

We consider four main use cases in agent-environment interaction and repurpose five existing datasets across different

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=1154&width=1702&top_left_y=222&top_left_x=187)

## $\stackrel{\circ}{\varrho}$ User

Here is a dataset about auto MPG: "https://huggingface.co/datasets/scikitlearn/auto-mpg/raw/main/auto-mpg.csv". You should download and load the dataset from the URL, take a look atil, check for missing value, do a train-test everything except "mpg" and "car name" as the input features.
ented

宋 CodeActAgent

To accomplish this task, I will use the pandas library to download and load the he scikit-leck or missing values, and perform a train-test split. Then, I will us cearnilury to train a regression mode

morport numyy as no
trom sklearn.model_selection import train_-test_split
from sklearn.linear_model import LinearRegression

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=33&width=204&top_left_y=542&top_left_x=222)

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=241&width=520&top_left_y=573&top_left_x=217)

(3) Environment

mpg
i...omitted for space.
car name

car name
dtype: int6 6

ValueE rror
'tmp/ipykernel_12/1372561217.py in 20$)$
Traceback (most recent call
tost last)

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=114&width=520&top_left_y=943&top_left_x=198)

... [omitted for space]...

## 舟 CodeActAgent

It seems that there is a data type issue in the dataset. The '?'' character is present the data, which is causing the error when trying to convert the data to a flo

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=38&width=184&top_left_y=1198&top_left_x=220)

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=27&width=233&top_left_y=1233&top_left_x=241)

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=36&width=320&top_left_y=1256&top_left_x=236)

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=30&width=168&top_left_y=1301&top_left_x=239)

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=46&width=382&top_left_y=1317&top_left_x=239)

![](https://cdn.mathpix.com/cropped/2024_06_04_8aed5213a1cb20219e8eg-06.jpg?height=1125&width=548&top_left_y=240&top_left_x=756)

Figure 3: Example multi-turn interaction with Python packages using CodeActAgent (Mistral-7b). No in-context demonstrations are provided to the model. Some messages are omitted for space. See https:// chat.xwang. dev/r/Vqn108G for complete interaction.

domains to generate trajectories:

- Information Seeking: We use a training subset of HotpotQA (Yang et al., 2018) to generate information-seeking trajectories, where LLMs use the wikipedia_search API (provided as a Python function) to search for information to answer questions.
- Software Package (Tool) Usage: We use the training set of code generation problems in APPS (Hendrycks et al., 2021a) and math problems in MATH (Hendrycks et al., 2021b). The code generation tasks already involve importing packages and/or creating new tools by defining a new Python function. For MATH, we provide an incontext demonstration of importing Python packages (e.g., sympy for symbolic math) for problem-solving.
- External Memory: We repurpose the training subset of WikiTableQuestion (Pasupat \& Liang, 2015) and tweak it into two variants of tabular reasoning tasks that require accessing external memory: (1) SQL-based, requiring the LLM to interact with an SQL database through sqlite3 package to answer the question via SQL execution; (2) Pandas-based, requiring the model to interact with pan- das tables to perform data operations (e.g., select, filter). Examples of instructions can be found in §G.3.1.
- Robot Planning: We use ALFWorld (Shridhar et al., 2020), a text-only embodied environment simulator, to generate trajectories that use robot-control APIs (repurposed as Python function) to complete household tasks. Following MINT (Wang et al., 2023e), we provide an in-context demonstration to encourage the use of for-loop and if-statement code blocks to automate repetitive operations (e.g., searching for items by visiting different locations).

Data Down-sampling. We down-sample each dataset by keeping only the most challenging instances, aiming to make trajectory generation more efficient and cost-effective. Furthermore, it also helps remove simple instances that existing LLMs can already solve. The statistics of the filtered dataset can be found in Tab. A.9. Please refer to $\S$ G. 1 for details about the down-sample process.

Repurpose Data for Multi-turn Interaction. Some datasets (APPS, MATH, WikiTableQuestions) are initially single-turn problems that expect one solution per instruc-

Table 4: Statistics of our training mixture and comparison with prior work. Please refer to $\S 3.1$ for details about CodeActInstruct and general conversation data. Token statistics are computed using Llama-2 tokenizer.

| Data Mixture | Data Type | Data Name | \# of Data Instances | \# of Total Tokens | Avg. Tokens Per Instance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Prior Work | - | FireAct (Chen et al., 2023a) | 2,063 | 542,176 | 262.81 |
|  | - | AgentInstruct (Zeng et al., 2023) | 1,866 | $2,517,785$ | 1349.30 |
| CodeActInstruct (Ours) | Information Seeking | HotpotQA (Yang et al., 2018) | 1,664 | $2,472,227$ | 1485.71 |
|  | Software Packages (Tool) | MATH (Math, (Hendrycks et al., 2021b)) | 1,732 | $1,719,467$ | 992.76 |
|  | Software Packages (Tool) | APPS (Code, (Hendrycks et al., 2021a)) | 647 | $1,235,472$ | 1909.54 |
|  | External Memory | WikiTableQuestion (Pasupat \& Liang, 2015) | 1,065 | $1,316,246$ | 1235.91 |
|  | Robot Planning | ALFWorld (Shridhar et al., 2020) | 2,031 | $3,838,269$ | 1889.84 |
|  |  | Total | 7,139 | $10,581,681$ | 1482.24 |
| General Conversation | Single-Turn Reasoning | OpenOrca (Sub-sampled, (Lian et al., 2023)) | 50,000 | $14,034,152$ | 280.68 |
|  | Multi-Turn Conversations | ShareGPT (Sub-sampled, (Anonymous, 2023)) | 10,000 | $17,933,861$ | 1793.39 |
|  | Multi-Turn Conversations | ShareGPT (GPT-4, (OpenChat, 2023)) | 4,583 | $18,195,878$ | 3970.30 |
|  | Multi-turn Reasoning | CapyBara (LDJnr, 2023) | 4,647 | $4,982,435$ | 1072.18 |
|  |  | Total | 69,230 | $55,146,326$ | 796.57 |

tion, whereas, in a realistic agent use case, we often require multi-turn interaction to complete each task (Fig. 1 top). Following MINT (Wang et al., 2023e), we repurpose singleturn problems into multi-turn ones by allowing LLM to interact with the environment for multiple turns before it decides to submit one solution for evaluation. Specifically for code generation problems, we provide an in-context example to guide LLMs to test their solution on provided test cases before they submit the solution. Metrics from the original data will evaluate the submitted solution to determine its correctness. We include examples in §G.3.

Trajectory Generation. We use MINT's evaluation framework (Wang et al., 2023e) to generate interaction trajectories for the aforementioned datasets and determine the correctness of each trajectory. We run gpt-3.5-turbo-0613 from OpenAI, claude-1-instant and claude-2 from Anthropic on down-sampled data, except code generation, which we use a longer-context version of GPT-3.5 (gpt-3.5-turbo-0613-16k) due to the long-context requirement of the self-debugging process. On a subset of problems that none of these models can solve, we use gpt-4-0613 to generate trajectories.

Enhancing Agent's Capabilities of Improving from Interaction. We select a high-quality subset of all the generated trajectories from CodeActInstruct to promote the agent's ability to improve the next action based on prior observations (e.g., self-debugging from code execution error message, a planning capability in Fig. 2). To achieve this, we selectively preserve those trajectories wherein the model initially encounters errors but rectifies these inaccuracies in later interactions. For these instances, the LLM typically engages in self-reflection following the initial error, thereby proactively enhancing its future actions. Other filtering details are discussed in $\S$ G.2. On all trajectories generated, we keep 411 trajectories from gpt-4-0613 and 6728 trajectories from gpt-3.5 and claude. The statistics of the resulting dataset CodeActInstruct are shown in Tab. 4.

Comparing CodeActInstruct with Prior Work. Com- pared with prior work AgentInstruct (Zeng et al., 2023) and FireAct (Chen et al., 2023a) that mainly focus using text as action, CodeActInstruct results in models that are more practical in real-world implementation, as such models using CodeAct can directly interact with Python interpreters and open-source toolkits (Fig. 3), reducing the development effort for action parsing and tool creations. CodeActInstruct is systematically constructed following the general agent framework (Fig. 2). It covers diverse domains (e.g., compared to FireAct that only considers QA-task and search API), contains quality data (e.g., promotes agent's capability of self-debug) and of larger size ( $3.8 \mathrm{x} / 3.5 \mathrm{x}$ more data trajectories and $5 \mathrm{x} / 19 \mathrm{x}$ more tokens compared to AgentInstruct / FireAct respectively in Tab. 4). As we empirically show in Tab. 5, the resulting model (same backbone) of CodeActInstruct achieves $24 \%$ and $119 \%$ relative improvement compared to AgentInstruct and FireAct.

## CodeActInstruct Can Be Used With Existing Agent-

 User Conversation Data. We use a sub-sampled set of OpenOrca (Lian et al., 2023) that focuses on single-turn chain-of-thought (CoT) reasoning, ShareGPT (Anonymous, 2023; OpenChat, 2023) from two sources that contain multiturn conversations between human and LLM, and CapyBara (LDJnr, 2023) that focuses on reasoning in multi-turn conversations. Statistics and down-sampling details can be found in Tab. 4 and $\S$.
### 3.2. CodeActAgent

We fine-tune Llama-2 7B (Touvron et al., 2023) and Mistral 7B (Jiang et al., 2023) on a mixture of CodeActInstruct and general conversations (Tab. 4) to obtain CodeActAgent.

Training Setup. We perform full-parameter supervised finetuning with a sequence length of 4,096 tokens for Llama-2 and 16,384 for Mistral. Please refer to $\S \mathrm{D}$ for more details.

Evaluation Setup. We use MINT (Wang et al., 2023e) to evaluate LLMs with CodeAct on a diverse range of agent tasks. CodeActAgent has some training domains

Table 5: Evaluation results for CodeActAgent. The best results among all open-source LLMs are bolded, and the second-best results are underlined. ID and OD stand for in-domain and out-of-domain evaluation correspondingly. Overall averaged performance normalizes the MT-Bench score to be consistent with other tasks and excludes in-domain tasks for fair comparison.

| Model | Size | Agent Tasks |  |  |  |  | {Generic Tasks <br> (OD)} |  |  |  | Overall <br> Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Code as Action |  |  | Text as Action (OD) |  |  |  |  |  |  |
|  |  | MINT (ID) | MINT (OD) | $\mathrm{M}^{3}$ ToolEval (OD) | Miniwob++ | SciWorld | MMLU | HumanEval | GSM8K | MTBench |  |
| Open-source LLMs (LLaMA-2-based) |  |  |  |  |  |  |  |  |  |  |  |
| Llama2 Base | 7B | $-*$ | $-*$ | $-*$ | $-*$ | $-*$ | 45.3 | 12.8 | 14.6 | $-*$ | $-*$ |
| Llama2 Chat | 7B | 3.2 | 11.0 | $\underline{0.0}$ | 0.0 | 5.9 | 48.0 | 13.9 | 27.7 | 6.3 | 21.1 |
| FireAct (Chen et al., 2023a) | 7B | 0.0 | 0.3 | $\overline{0.0}$ | 0.0 | 6.8 | 44.1 | 3.5 | 12.4 | 4.5 | 14.0 |
| AgentLM (Zeng et al., 2023) | 7B | 8.7 | 6.1 | $\overline{0.0}$ | $\underline{28.9}$ | 13.7 | 48.7 | 15.4 | 24.6 | 6.1 | 24.8 |
| CodeActAgent (LLaMA-2) | $7 \mathrm{~B}$ | $\underline{51.3}$ | $\underline{20.4}$ | $\underline{0.0}$ | 25.5 | 17.6 | 50.6 | 18.1 | 38.3 | $\underline{7.5}$ | $\underline{30.7}$ |
| Open-source LLMs (Mistral-based) |  |  |  |  |  |  |  |  |  |  |  |
| Mistral Base | 7B | $-*$ | $-*$ | $-*$ | $-*$ | $-*$ | 60.1 | $\underline{30.5}$ | $\underline{52.1}$ | $-*$ | $-*$ |
| Mistral Instruct | 7B | 18.8 | 9.7 | $\underline{0.0}$ | 0.5 | 4.0 | 53.8 | $\overline{29.3}$ | $\overline{43.3}$ | 6.4 | 25.6 |
| CodeActAgent (Mistral) | 7B | 57.4 | 32.4 | $1 \overline{2.2}$ | 46.2 | $\underline{15.9}$ | $\underline{59.1}$ | 34.7 | 58.0 | 8.2 | 42.5 |
| Closed-source LLMs |  |  |  |  |  |  |  |  |  |  |  |
| gpt-3.5-turbo-0613 | - | 33.9 | 38.2 | 51.2 | 66.7 | 21.2 | 70.0 | 48.1 | 57.1 | 7.9 | 54.0 |
| gpt-4-0613 | - | 68.6 | 70.2 | 67.1 | 69.4 | 36.4 | 86.4 | 67.0 | 87.1 | 9.0 | 71.7 |

* Some results are only available with instruction-tuned models.

overlapping with MINT's evaluation (i.e., MINT includes ALFWorld and MATH), hence we report separate numbers for MINT's in- and out-of-domain performance. Unless otherwise specified, we measure MINT tasks' success rates with interaction turn $k=5$. We also evaluate out-of-domain agent tasks using text actions from MiniWob++ (computer tasks, (Kim et al., 2023)) and ScienceWorld (text-based simulator for elementary science curriculum, (Wang et al., 2022a)) to test whether CodeActAgent can generalize to different action formats. Finally, we include a suite of general LLM evaluation tasks to assess general capability: MMLU (Hendrycks et al., 2020) for knowledge-based QA, HumanEval (Chen et al., 2021) for single-turn codegeneration, GSM8K (Cobbe et al., 2021) for single-turn tool-free math reasoning, and MTBench (Zheng et al., 2023) for instruction-following.

CodeActAgent Excels in CodeAct Task. As shown in Tab. 5, CodeActAgent (both variants) perform better than all evaluated open-source LLMs on both the in- and out-ofdomain subsets of MINT. On M ${ }^{3}$ ToolEval, we find CodeActAgent (Mistral) outperforms open-source LLMs of similar size (7B and 13B) and even reaches similar performance to those 70B models (Tab. 3). Surprisingly, no improvement is observed for the Llama-2 variant. We discuss potential reasons in $\S \mathrm{H}$.

CodeActAgent Generalizes to Text Action. When evaluated on out-of-domain text actions, CodeActAgent (LLaMA2, 7B), which has never been optimized for text action, achieves comparable performance to AgentLM-7B (Zeng et al., 2023) which has explicit tuning for text actions.

CodeActAgent Maintains or Improves the Performance on General LLM Tasks. In Tab. 5, we find that CodeActAgent (both variants) performs better on generic LLM tasks we tested, except for a slight degradation on MMLU for CodeActAgent (Mistral, 7B).
Ablation Study. Tab. A. 8 presents ablation experiments to determine the importance of CodeActInstruct and general conversations. Both CodeActInstruct and general conversations contribute to agent tasks, while general conversations are essential to maintain performance on general tasks.

## 4. Related Work

### 4.1. Action Module in LLM Agents

As detailed in (Wang et al., 2023b), LLM-based autonomous agents are typically structured around four components: customized profiles (Park et al., 2023; Qian et al., 2023), longterm memory capabilities (Zhu et al., 2023; Fischer, 2023), reasoning and planning algorithms (Wei et al., 2022; Chen et al., 2023d), and, most crucially, action modules. The action modules are key to facilitating LLM agents to effectively interact with external entities, including humans (Lee et al., 2022) and tools (Qin et al., 2023a) in the environment (Wang et al., 2023e; Yang et al., 2024a). In this study, we address the critical problem of standardizing the action space for LLM agents. We further discuss the difference between CodeAct and the line of work that uses code generation for problem-solving in $\S$ A. We notice a concurrent study TaskWeaver (Qiao et al., 2023) similarly endorses the use of code. We discuss the principal distinctions in $\S$ B.

### 4.2. Improving LLM Agents

Two primary methods for enhancing LLM agents are prompt engineering and instruction tuning, as surveyed by (Wang et al., 2023b). For prompt engineering (Liu et al., 2023a), numerous strategies have been introduced to improve the chain-of-thought reasoning (Wei et al., 2022), including self-consistency-based reasoning (Wang et al., 2022b; Chen et al., 2023d) and tree-based approaches (Yao et al., 2023a). Moreover, LLMs can be strategically prompted to reflect on
previous plans (Yao et al., 2023b; Wang et al., 2023f; Zhang et al., 2023), enabling them to refine initial actions through trial and error. Contrast to prompt engineering, instruction tuning intrinsically enhances LLMs (Chung et al., 2022), particularly in their agent capabilities (Zeng et al., 2023; Chen et al., 2023a). For effective training, human annotators can curate expert demonstrations for specific agent tasks, such as web browsing (Yao et al., 2022a; Nakano et al., 2021). To minimize human annotation efforts, prior work creates synthetic datasets using stronger LLMs to distill agent capabilities into local models, focusing on tool usage (Qin et al., 2023b), interaction (Chen et al., 2023c), and social skills (Liu et al., 2023b). CodeActInstruct aligns with the latter approach and creates datasets using stronger LLMs.

## 5. Conclusions

This work introduces CodeAct that employs executable Python code for the LLM agent's action, which is advantageous over using text or JSON action, especially in complex scenarios. We collect CodeAct-focused multi-turn interaction trajectories CodeActInstruct for instruction tuning, and train CodeActAgent that is specially designed for seamless integration with Python and can execute sophisticated tasks (e.g., model training) leveraging existing Python packages and autonomously rectifying errors through self-debugging.

## Acknowledgement

We thank the anonymous reviewers for their suggestions and comments. This research is based upon work supported by U.S. DARPA ECOLE Program No. HR00112390060 and U.S. DARPA ITM Program No. FA8650-23-C-7316 and KAIROS Program No. FA8750-19-2-1004. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of DARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein. This work used the Delta system at the National Center for Supercomputing Applications through allocation CIS230256 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services \& Support (ACCESS, Boerner et al. 2023) program, which is supported by National Science Foundation grants \#2138259, \#2138286, \#2138307, \#2137603, and \#2138296.

## Impact Statement

This paper presents work whose goal is to advance LLMbased autonomous agents that can communicate with humans through natural language and assist human users by performing tasks in environments on behalf of humans. In this section, we discuss potential societal consequences, limitations, and future work related to our work and its goal.

CodeActAgent is an initial prototype of an autonomous agent and still has several practical limitations. For example, it may suffer from hallucination commonly seen in LLMs (e.g., imagine the content of a variable without actually printing it out), suggesting the need for subsequent alignment (Ouyang et al., 2022) for further improvements.

Despite being a prototype, CodeActAgent has already demonstrated limited self-improving capability (e.g., selfdebug error messages to improve its action) and the ability to interact with environments. Future work may build upon CodeActAgent to develop better agents by having them perform extensive interactions within a given environment and iteratively bootstrap their self-improving capability to learn to improve from past mistakes. More powerful agents, as results of such algorithms, are potentially beneficial for solving a wide range of real-world problems (e.g., theorem proving, drug discovery). As extensively discussed in (Eloundou et al., 2023), a fully autonomous agent may transform the current landscape of the labor market and impact the jobs of existing workers.

Furthermore, since CodeAct directly grants access for the agent to freely execute code in a sandbox environment, in the worst scenario (e.g., in Sci-Fi movies), such an agent may potentially break free of the sandbox restriction and cause harm to the world through cyber-attack, highlighting the need for future work to design better safety mechanism to safeguard autonomous agents (Tang et al., 2024).

## References

Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Ho, D., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jang, E., Ruano, R. J., Jeffrey, K., Jesmonth, S., Joshi, N., Julian, R., Kalashnikov, D., Kuang, Y., Lee, K.-H., Levine, S., Lu, Y., Luu, L., Parada, C., Pastor, P., Quiambao, J., Rao, K., Rettinghouse, J., Reyes, D., Sermanet, P., Sievers, N., Tan, C., Toshev, A., Vanhoucke, V., Xia, F., Xiao, T., Xu, P., Xu, S., Yan, M., and Zeng, A. Do as i can and not as i say: Grounding language in robotic affordances. In arXiv preprint arXiv:2204.01691, 2022 .

Anonymous. Sharegpt dataset. https://hf.co/ datasets/anon8231489123/ShareGPT_ Vicuna_unfiltered/blob/main/ShareGPT_ V3_unfiltered_cleaned_split_no_ imsorry.json, 2023. A dataset containing multi-turn conversations between human and LLM assistant.

Boerner, T. J., Deems, S., Furlani, T. R., Knuth, S. L., and Towns, J. Access: Advancing innovation: Nsf's advanced cyberinfrastructure coordination ecosystem: Services \& support. In Practice and Experience in Advanced Research Computing, pp. 173-176. 2023.

Bran, A. M., Cox, S., White, A. D., and Schwaller, P. Chemcrow: Augmenting large-language models with chemistry tools. arXiv preprint arXiv:2304.05376, 2023.

Cano, A. H., Pagliardini, M., Köpf, A., Matoba, K., Mohtashami, A., Wang, X., Fan, O. S., Marmet, A., Bayazit, D., Krawczuk, I., Chen, Z., Salvi, F., Bosselut, A., and Jaggi, M. epfllm megatron-llm, 2023. URL https: //github.com/epfLLM/Megatron-LLM.

Chase, H. LangChain, October 2022. URL https: / / github.com/langchain-ai/langchain.

Chen, B., Shu, C., Shareghi, E., Collier, N., Narasimhan, K., and Yao, S. Fireact: Toward language agent fine-tuning. arXiv preprint arXiv:2310.05915, 2023a.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Chen, X., Lin, M., Schärli, N., and Zhou, D. Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128, 2023b.

Chen, Y., Sikka, K., Cogswell, M., Ji, H., and Divakaran, A. Dress: Instructing large vision-language models to align and interact with humans via natural language feedback. arXiv preprint arXiv:2311.10081, 2023c.

Chen, Y., Sikka, K., Cogswell, M., Ji, H., and Divakaran, A. Measuring and improving chain-of-thought reasoning in vision-language models. arXiv preprint arXiv:2309.04461, 2023d.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Eloundou, T., Manning, S., Mishkin, P., and Rock, D. Gpts are gpts: An early look at the labor market impact potential of large language models. arXiv preprint arXiv:2303.10130, 2023.
Fischer, K. A. Reflective linguistic programming (rlp): A stepping stone in socially-aware agi (socialagi). arXiv preprint arXiv:2305.12647, 2023.

Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. Pal: Program-aided language models. In International Conference on Machine Learning, pp. 10764-10799. PMLR, 2023.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2020.

Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., et al. Measuring coding challenge competence with apps. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021a.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021b.

Hong, S., Zheng, X., Chen, J., Cheng, Y., Wang, J., Zhang, C., Wang, Z., Yau, S. K. S., Lin, Z., Zhou, L., et al. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 2023.

Hong, S., Lin, Y., Liu, B., Liu, B., Wu, B., Li, D., Chen, J., Zhang, J., Wang, J., Zhang, L., Zhang, L., Yang, M., Zhuge, M., Guo, T., Zhou, T., Tao, W., Wang, W., Tang, X., Lu, X., Zheng, X., Liang, X., Fei, Y., Cheng, Y., Xu, Z., and $\mathrm{Wu}, \mathrm{C}$. Data interpreter: An llm agent for data science, 2024.

Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., and Fei-Fei, L. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. 1., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Kim, G., Baldi, P., and McAleer, S. Language models can solve computer tasks. arXiv preprint arXiv:2303.17491, 2023.

LDJnr. Capybara dataset. https://hf.co/ datasets/LDJnr/Verified-Camel, https: //hf.co/datasets/LDJnr/Pure-Dove, https://hf.co/datasets/LDJnr/

LessWrong-Amplify-Instruct, 2023. A dataset focusing on reasoning in multi-turn conversations.

Lee, M., Liang, P., and Yang, Q. Coauthor: Designing a human-ai collaborative writing dataset for exploring language model capabilities. In Proceedings of the 2022 CHI conference on human factors in computing systems, pp. 1-19, 2022.

Li, M., Song, F., Yu, B., Yu, H., Li, Z., Huang, F., and Li, Y. Api-bank: A benchmark for tool-augmented llms, 2023.

Lian, W., Goodson, B., Pentland, E., Cook, A., Vong, C., and "Teknium". Openorca: An open dataset of gpt augmented flan reasoning traces. https://https: / /huggingface.co/Open-Orca/OpenOrca, 2023.

Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., and Zeng, A. Code as policies: Language model programs for embodied control. In arXiv preprint arXiv:2209.07753, 2022.

Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., and Neubig, G. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1-35, 2023a.

Liu, R., Yang, R., Jia, C., Zhang, G., Zhou, D., Dai, A. M., Yang, D., and Vosoughi, S. Training socially aligned language models in simulated human society. arXiv preprint arXiv:2305.16960, 2023b.

Ma, Y. J., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., and Anandkumar, A. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

Mialon, G., Dessì, R., Lomeli, M., Nalmpantis, C., Pasunuru, R., Raileanu, R., Rozière, B., Schick, T., DwivediYu, J., Celikyilmaz, A., et al. Augmented language models: a survey. arXiv preprint arXiv:2302.07842, 2023.

Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

OpenChat. Sharegpt dataset. https://hf.co/ datasets/openchat/openchat_sharegpt v3/blob/main/sharegpt_gpt4.json, 2023. A dataset containing multi-turn conversations between human and LLM assistants. It is filtered to contain data only from GPT-4.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022.

Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pp. 1-22, 2023.

Pasupat, P. and Liang, P. Compositional semantic parsing on semi-structured tables. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1470-1480, 2015.

Patil, S. G., Zhang, T., Wang, X., and Gonzalez, J. E. Gorilla: Large language model connected with massive apis. ArXiv, abs/2305.15334, 2023. URL https://api.semanticscholar. org/CorpusID:258865184.

Qian, C., Cong, X., Yang, C., Chen, W., Su, Y., Xu, J., Liu, Z., and Sun, M. Communicative agents for software development. arXiv preprint arXiv:2307.07924, 2023.

Qiao, B., Li, L., Zhang, X., He, S., Kang, Y., Zhang, C., Yang, F., Dong, H., Zhang, J., Wang, L., et al. Taskweaver: A code-first agent framework. arXiv preprint arXiv:2311.17541, 2023.

Qin, Y., Hu, S., Lin, Y., Chen, W., Ding, N., Cui, G., Zeng, Z., Huang, Y., Xiao, C., Han, C., et al. Tool learning with foundation models. arXiv preprint arXiv:2304.08354, 2023a.

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y.-T., Lin, Y., Cong, X., Tang, X., Qian, B., Zhao, S., Tian, R., Xie, R., Zhou, J., Gerstein, M. H., Li, D., Liu, Z., and Sun, M. Toolllm: Facilitating large language models to master 16000+ real-world apis. ArXiv, abs/2307.16789, 2023b. URL https://api.semanticscholar. org/CorpusID:260334759.

Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

Shen, Y., Song, K., Tan, X., Li, D., Lu, W., and Zhuang, Y. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580, 2023.

Shridhar, M., Yuan, X., Cote, M.-A., Bisk, Y., Trischler, A., and Hausknecht, M. Alfworld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2020.

Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D., Tremblay, J., Fox, D., Thomason, J., and Garg, A. Progprompt: Generating situated robot task plans using large language models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 11523-11530, 2023. doi: 10.1109/ICRA48891.2023. 10161317 .

Surís, D., Menon, S., and Vondrick, C. Vipergpt: Visual inference via python execution for reasoning. Proceedings of IEEE International Conference on Computer Vision (ICCV), 2023.

Tang, X., Jin, Q., Zhu, K., Yuan, T., Zhang, Y., Zhou, W., Qu, M., Zhao, Y., Tang, J., Zhang, Z., et al. Prioritizing safeguarding over autonomy: Risks of llm agents for science. arXiv preprint arXiv:2402.04247, 2024.

TIOBE Index. Tiobe index. https://www.tiobe . com/tiobe-index/, Accessed at Jan 23rd, 2024, 2024. The TIOBE Programming Community index is an indicator of the popularity of programming languages. The index is updated once a month. The ratings are based on the number of skilled engineers world-wide, courses and third party vendors.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An openended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023a.

Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432, 2023b.

Wang, R., Jansen, P. A., Côté, M.-A., and Ammanabrolu, P. Scienceworld: Is your agent smarter than a 5th grader? In Conference on Empirical Methods in Natural Language Processing, 2022a. URL https://api.semanticscholar. org/CorpusID:247451124.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022b.

Wang, X., Li, S., and Ji, H. Code4Struct: Code generation for few-shot event structure prediction. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp. 3640-3663, Toronto, Canada, July 2023c. Association for Computational Linguistics. doi: 10.18653/v1/2023. acl-long.202. URL https: / / aclanthology .org / 2023.acl-long. 202 .

Wang, X., Peng, H., Jabbarvand, R., and Ji, H. Leti: Learning to generate from textual interactions. ArXiv, abs/2305.10314, 2023d.

Wang, X., Wang, Z., Liu, J., Chen, Y., Yuan, L., Peng, H., and Ji, H. Mint: Evaluating llms in multi-turn interaction with tools and language feedback. arXiv preprint arXiv:2309.10691, 2023e.

Wang, Z., Cai, S., Liu, A., Ma, X., and Liang, Y. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023f.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35: 24824-24837, 2022.

Xu, Q., Hong, F., Li, B., Hu, C., Chen, Z., and Zhang, J. On the tool manipulation capability of open-source large language models, 2023.

Yang, J., Prabhakar, A., Narasimhan, K., and Yao, S. Intercode: Standardizing and benchmarking interactive coding with execution feedback. Advances in Neural Information Processing Systems, 36, 2024a.

Yang, K., Liu, J., Wu, J., Yang, C., Fung, Y. R., Li, S., Huang, Z., Cao, X., Wang, X., Wang, Y., Ji, H., and Zhai, C. If llm is the wizard, then code is the wand: A survey on how code empowers large language models to serve as intelligent agents, 2024b.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., and Manning, C. D. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2369$2380,2018$.

Yang, Z., Liu, A., Liu, Z., Liu, K., Xiong, F., Wang, Y., Yang, Z., Hu, Q., Chen, X., Zhang, Z., Luo, F., Guo, Z., Li, P., and Liu, Y. Towards unified alignment between agents, humans, and environment, 2024c.

Yao, S., Chen, H., Yang, J., and Narasimhan, K. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:2074420757, 2022a.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2022b.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023a.

Yao, W., Heinecke, S., Niebles, J. C., Liu, Z., Feng, Y., Xue, L., Murthy, R., Chen, Z., Zhang, J., Arpit, D., et al. Retroformer: Retrospective large language agents with policy gradient optimization. arXiv preprint arXiv:2308.02151, 2023b.

Yuan, L., Chen, Y., Wang, X., Fung, Y. R., Peng, H., and Ji, H. Craft: Customizing llms by creating and retrieving from specialized toolsets. ArXiv, abs/2309.17428, 2023. URL https://api.semanticscholar. org/CorpusID:263310662.

Zeng, A., Liu, M., Lu, R., Wang, B., Liu, X., Dong, Y., and Tang, J. Agenttuning: Enabling generalized agent abilities for llms, 2023.

Zhang, C., Liu, L., Wang, J., Wang, C., Sun, X., Wang, H., and Cai, M. Prefer: Prompt ensemble learning via feedback-reflect-refine. arXiv preprint arXiv:2308.12033, 2023.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.

Zheng, T., Zhang, G., Shen, T., Liu, X., Lin, B. Y., Fu, J., Chen, W., and Yue, X. Opencodeinterpreter: Integrating code generation with execution and refinement. https://arxiv.org/abs/2402.14658, 2024.

Zhu, X., Chen, Y., Tian, H., Tao, C., Su, W., Yang, C., Huang, G., Li, B., Lu, L., Wang, X., et al. Ghost in the minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory. arXiv preprint arXiv:2305.17144, 2023.

Table A.6: Example of actions for re-purposed API-Bank (Li et al., 2023) and $\mathrm{M}^{3}$ ToolEval.

| Format | Action |
| :--- | :--- |
| CodeAct | AddAgenda(content="Meeting with John", <br> time="2023-10-26 09:00:00") |
| J"action": "AddAgenda", "content": <br> "Meeting with John", "time": <br> "2023-10-26 09:00:00"\} |  |
|  | Action: AddAgenda, content: Meeting <br> with John, time: 2023-10-26 09:00:00 |

Table A.7: Comparison between $\mathrm{M}^{3}$ ToolEval and existing tool-use evaluation benchmark.

| Benchmark | $\mathrm{M}^{3}$ ToolEval <br> (This work) | ToolBench <br> (Qin et al., 2023b) | APIBench <br> (Patil et al., 2023) | API-Bank <br> (Li et al., 2023) | ToolBench <br> (Xu et al., 2023) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Requiring multi-turn inter | $\checkmark$ | $\checkmark$ | $x$ | $x$ | $x$ |
| Multi | $\int$ | $\int$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Evaluation | Answer Match | LLM Evaluator | AST Tree Match | API-Call Match | Test Case |
| No dependency on external API* | $\checkmark$ | $x$ | $x$ | $\int$ | $x$ |
| Supported API Action Format | CodeAct \& JSON \& Text | JSON | CodeAct | JSON | CodeAct |

* Whether to rely on external API (e.g., RapidAPI, Google Sheet) hosted by a third party. The availability of such third-party APIs can greatly impact evaluation results (e.g., low API-calling performance not because the model is bad but rather because the API required is not accessible).

Table A.8: Ablation study results. The best results are bolded, and the second-best results are underlined. ID and OD stand for in-domain and out-of-domain evaluation correspondingly. Overall averaged performance normalizes the MT-Bench score to be consistent with other tasks and excludes in-domain tasks for fair comparison.

| Model | Size | Agent Tasks |  |  |  | {Generic LLM Tasks <br> (OD)} |  |  |  | Overall <br> Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Code as Action |  | Text as Action (OD) |  |  |  |  |  |  |
|  |  | MINT (ID) | MINT (OD) | Miniwob++ | SciWorld | MMLU | HumanEval | GSM8K | MTBench |  |
| CodeActAgent (Llama2-based) | $7 \mathrm{~B}$ | 51.3 | 20.4 | $\underline{25.5}$ | 17.6 | 50.6 | $\underline{18.1}$ | 38.3 | 7.5 | 35.1 |
| w/o CodeAct | 7B | 17.0 | 15.5 | 36.4 | 16.9 | 49.5 | $\overline{14.7}$ | $\underline{36.0}$ | 7.2 | $\underline{34.5}$ |
| w/o general conversations | 7B | $\underline{29.2}$ | $\underline{15.9}$ | 0.0 | $\underline{17.1}$ | $\overline{46.4}$ | 19.7 | $\overline{20.6}$ | $\overline{4.1}$ | $\overline{22.9}$ |
| CodeActAgent (Mistral-based) | $7 \mathrm{~B}$ | 57.4 | 32.4 | $\underline{46.2}$ | $\underline{15.9}$ | $\underline{59.1}$ | 34.7 | $\underline{58.0}$ | $\underline{8.2}$ | $\overline{46.8}$ |
| w/o CodeAct | $7 \mathrm{~B}$ | 32.9 | $\underline{23.0}$ | $\overline{47.8} \quad$ | $\overline{17.0}$ | $\overline{59.9}$ | 33.2 | $\overline{59.5}$ | $\overline{8.3}$ | 46.2 |
| w/o general conversations | $7 \mathrm{~B}$ | 50.5 | 13.9 | 0.0 | 11.0 | 52.4 | $\overline{27.9}$ | 26.8 | 2.6 | $\overline{22.6}$ |
</end of paper 0>


<paper 1>
# Chain of Tools: Large Language Model is an Automatic Multi-tool Learner 

Zhengliang Shi $\|_{||}$Shen Gao ${ }^{2}$ Xiuyi Chen ${ }^{3}$ Yue Feng ${ }^{4}$ Lingyong Yan ${ }^{3}$<br>Haibo Shi ${ }^{3}$ Dawei Yin ${ }^{3}$ Zhumin Chen ${ }^{1}$ Suzan Verberne ${ }^{5}$ Zhaochun Ren ${ }^{5}$<br>${ }^{1}$ Shandong University ${ }^{2}$ University of Electronic Science and Technology of China<br>${ }^{3}$ Baidu Inc., Beijing, China ${ }^{4}$ University of Birmingham, Birmingham, UK<br>${ }^{5}$ Leiden University, Leiden, The Netherlands<br>shizhl@mail.sdu.edu.cn z.ren@liacs.leidenuniv.nl


#### Abstract

Augmenting large language models (LLMs) with external tools has emerged as a promising approach to extend their utility, empowering them to solve practical tasks. Existing work typically empowers LLMs as tool users with a manually designed workflow, where the LLM plans a series of tools in a step-by-step manner, and sequentially executes each tool to obtain intermediate results until deriving the final answer. However, they suffer from two challenges in realistic scenarios: (1) The handcrafted control flow is often ad-hoc and constraints the LLM to local planning; (2) The LLM is instructed to use only manually demonstrated tools or well-trained Python functions, which limits its generalization to new tools. In this work, we first propose Automatic Tool Chain (ATC), a framework that enables the LLM to act as a multi-tool user, which directly utilizes a chain of tools through programming. To scale up the scope of the tools, we next propose a black-box probing method. This further empowers the LLM as a tool learner that can actively discover and document tool usages, teaching themselves to properly master new tools. For a comprehensive evaluation, we build a challenging benchmark named ToolFlow, which diverges from previous benchmarks by its long-term planning scenarios and complex toolset. Experiments on both existing datasets and ToolFlow illustrate the superiority of our framework. Analysis on different settings also validates the effectiveness and the utility of our black-box probing algorithm.


## 1 Introduction

Large language models (LLMs) have shown promising capabilities such as in-context learning and real-world planning [1-3]. To further increase their utility, the tool learning task [4, 5] is proposed to augment LLMs with external tools, e.g., a Weather App, enabling them to interact with the physical world [6-8]. With the assistance of tools, LLMs can serve as agents to automatically solve practical tasks [9-11], such as check the weather in London. More recent work further uses Python code as a unified interface to access diverse tools, coming with advantages like seamlessly reusing massive built-in functions and performing for-loop operations [12-15].

Given a practical task, prior work grounds the tool-use process in an iterative plan-execute-observe pipeline [16-19]. As shown in Figure 1 (a), the LLM first plans a series of tools in a step-by-step manner [20-22]. For each step, the LLM generates arguments in a handcrafted format [23, 24] or code snippets [12, 19] for execution, continuously incorporating intermediate results into the context for subsequent actions. However, they suffer from two challenges in realistic scenarios. First, their[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-02.jpg?height=726&width=1347&top_left_y=255&top_left_x=386)

Figure 1: The comparison between previous plan-execute-observe pipeline (left) and our proposed framework (right).

workflow to interact with tools is typically manually designed and ad-hoc, struggling to generalize to different tool-use scenarios. The handcrafted workflow also constrains the LLM to local planning, leading to diminished performance in long-term planning tasks. Second, the LLM can only use manually demonstrated tools or built-in Python functions, which limits the toolset scope. To this end, we address the above challenges with a focus on two research objectives in our work: (1) Empower the LLM as an automatic Multi-tool user to generate a tool-use chain; (2) Further empower the LLM as an active Multi-tool learner to instruct themselves to master new tools.

To address the first objective, we propose Automatic Tool Chain (ATC), a framework that enables the LLM to utilize a chain of tools through programming. As shown in Figure 1 (b), the LLM directly learns the input-output schema and data flow dependency of various tools from tool protocols (a.k.a., tool documentation). Different from the short-form docstring of simple Python functions in previous work [13, 25, 15], the protocol comprehensively outlines meta-information about a complex tool, such as arguments requirement, structural response specifications (i.e., a general schema elaborating expected execution results) and possible execution statuses. With the assistance of the provided protocols, we instruct the LLM to generate a program that sequentially calls a chain of tools, parses the tool response to cache useful information and derives the final answer. To correct runtime errors in the generated programs, we introduce an attributable reflection mechanism, which allows the LLM to track faulty snippets, pinpoint incorrect tool usage, and calibrate the programs accordingly.

In realistic scenarios, a potential challenge that limits the scope of the toolset in our framework is the continuous crafting of documented protocols for diverse and fast-paced tools, which is typically done by software developers [26, 27]. Therefore, we propose a black-box probing method to address the second objective. This approach enables the LLM to be an active tool learner that can probe the input-output schema of new tools and teach itself how to use them. Initially, the LLM is instructed to generate testing instances that target the functionality of a tool, including relevant tasks and tool-use program solutions. While executing the generated program, we transform the task-specific tool response into a general schema and leverage these instances as practical usage demonstrations, thereby documenting the tool protocol. Considering that a single tool may fail to probe due to the absence of private arguments, which are only acquired through other tools, we introduce a chain of probing algorithms. This algorithm effectively optimizes the cooperation among tools that have a strong input-output dependency.

We first investigate the capability of LLMs to generate a chain of tools on two well-established datasets from RestBench [14]. For a comprehensive evaluation, we also create a new benchmark testbed named ToolFlow, including 224 tasks across 107 real-world tools. ToolFlow diverges from the existing benchmarks by its more long-term planning tasks, the thorough protocol of the toolset, and complex data flow interdependency among tools, which evaluates our method under more challenging scenarios. The results show that (1) the LLM can well understand the tool protocol; (2)
the LLM exhibits strong capability in planning a chain of tools programmatically; and (3) despite the straightforward design, our framework substantially surpasses previous baselines with higher efficiency. In addition, the proposed black-box probing method effectively instructs LLMs to probe tool protocols and teach themselves to master new tools, extending the scope of the tools in our ATC.

Our contributions are summarized as follows: (i) We propose the Automatic tool chain (ATC), a framework to empower the LLM as a multi-tool user. (ii) We introduce a black-box probing method, which further enables the LLM to act as an active tool learner to the scope of the toolset in our ATC. (iii) We release a new benchmark, ToolFlow, to evaluate tool learning methods in more challenging scenarios. (iv) Extensive experiments on three datasets validate the superiority of our method.

## 2 Related Work

Tool learning with foundation models. Augmenting LLMs with external tools has been proven a promising method for enhancing their utility and enabling interactions with the physical world [8. 6, 28, 29]. As the commonly-used methods, the LLM first breaks down complex tasks and plans a series of tools in a step-by-step manner [24, 30, 19]. For each step, the LLM separately executes the tools and incorporates the full response into context, which contains the required arguments to invoke subsequent tools due to the data flow dependency [20, 31, 22]. Despite advancements, this iterative workflow is typically manually designed and ad-hoc, struggling to generalize across various tool-use scenarios. In this work, we propose the ATC, enabling the LLM as an automatic multi-tool learner to directly integrate a chain of tools.

Programming-enhanced LLMs. Recent work has shown the potential of using programming languages (PLs) to enhance the planning and reasoning capability of LLMs [32--34]. For example, previous work enables LLMs to generate a programmatic chain of thought to solve complex numeric reasoning tasks [35, 36], which exhibits remarkable performance. In the tool learning task, compared with nature languages (NLs), recent work also shows that LLMs can generate Python code snippets as actions, with advantages like integrating widely used Python functions and simplifying lengthy for-loop operations [12]. Additionally, previous work limited the LLM to only use well-documented tools [14, 27, 25] or the Python function learned from the pre-training stage [12] . In this work, we further investigate the LLM as a multi-tool learner, teaching themselves to master new tools.

Learning from external feedback. Learning from feedback is a prevailing strategy to mitigate undesired behaviors of LLMs [37, 38], mirroring a typical human learning strategy where individuals refine their behaviors through trial, error, and correction [39-42]. Previous studies such as Reflexion [43] show the capability of LLMs to reflect verbal signals from the environment and revise their mistakes [44]. Recent work prompts LLMs to use automated feedback (e.g., runtime errors) implemented in software to self-debug its generated code in each step [44, 45, 17]. Despite the progress, this feedback typically reflects straightforward faults while failing to address the snowballing issue [46] in multi-step planning, where an initial error can lead to a series of subsequent accumulated errors 47]. In the tool learning task, pinpointing the exact tool triggering the error is crucial [48]. In this work, the proposed attributable reflection mechanism guides LLMs to track the faulty program snippet, attribute it to a specific tool calling, and revise generated programs.

## 3 Automatic Tool Chain

### 3.1 Preliminaries

Solving practical tasks with the assistance of tools can be conceptualized as a planning process. Formally, the LLM, denoted as $M_{\theta}$, is equipped with access to a set of tools $\mathcal{T}=\left\{t_{1}, t_{2}, \ldots, t_{|\mathcal{T}|}\right\}$ and corresponding documented protocols $\mathcal{D}=\left\{d_{1}, d_{2}, \ldots, d_{|\mathcal{D}|}\right\}$. The protocol $d_{i}$ provides detailed meta information about tool $t_{i}$ such as argument requirements, tool description, and the specification of execution result (a.k.a., schema). Given a natural language task $x \in \mathcal{X}$ from the task space $\mathcal{X}$, the object is to generate a sequence of tool callings paired with corresponding arguments to derive the final answer. Previous work configures the LLM with customized control flow and tool-use templates, whereby the LLM iteratively interacts with single tools following a plan-execute-observe pipeline. In this work, we enable the LLM to automatically utilize a chain of tools by generating a program $\mathcal{C}$.

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-04.jpg?height=524&width=1396&top_left_y=236&top_left_x=364)

(a) Overall framework

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-04.jpg?height=233&width=659&top_left_y=241&top_left_x=1080)

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-04.jpg?height=230&width=623&top_left_y=476&top_left_x=1095)

(b) Black-box Tool probing

Figure 2: Left: An overview of our framework with the proposed attributable reflection mechanism ( $\$ 3.3$. Right: Our black-box probing method ( $\$ 4.1$.

### 3.2 Chain of Tools Generation

Given a task $x$, we first provide the LLM with the documented protocol $d_{i} \in \mathcal{D}$ for each tool $t_{i}$ in the candidate toolset $\mathcal{T}$. The documented protocol $d_{i}$ records meta information, including the description to outline what the tool $t_{i}$ can be used for, the argument requirements to describe how to invoke it, and the response schema to specify the expected type of execution results. With the assistance of this meta information, the LLM can automatically learn the tool usage, and master the detailed input-output schema as well as the data flow relation among various tools. Then, we instruct the LLM $\mathcal{M}_{\theta}$ to directly generate an executable program $\mathcal{C}$ to utilize multiple tools and solve the input task $x$. Formally, it can be formulated as:

$$
\begin{equation*}
\mathcal{C}=\mathcal{M}_{\theta}\left(x, \mathcal{T}, \mathcal{D}, \mathcal{I}_{c}\right) \tag{1}
\end{equation*}
$$

Here, the $\mathcal{I}_{c}$ indicates a concise instruction for program generation operation, which is provided in Appendix A.7. The $\mathcal{T}$ and $\mathcal{D}$ represent the candidate toolset and corresponding tool protocols, respectively. The generated program sequentially calls multiple tools to acquire useful information, parses lengthy execution results for subsequent utilization, and simplifies the lengthy task-solving trajectory with concise programmatic planning. The final result $r$ is derived by executing the generated program through a code interpreter, which can be formulated as $r=\operatorname{Execute}(\mathcal{C})$.

### 3.3 Programming with Attributable Reflection

The generated program effectively integrates multi-step tool utilization. However, runtime errors, such as passing redundant arguments, are frequently observed during the execution. Therefore, we introduce an attributable reflection mechanism, guiding the LLM to first attribute the raised error to a specific tool, and then adaptively revise the generated program. As shown in Figure 2 , if a runtime error is raised, we capture the error message in the result $r$, including both the faulty code snippet and the error trace. Then, we instruct the LLM to localize the specific tool calling which triggers the error and exactly generate the tool name, represented as $t_{j}=\mathcal{M}_{\theta}\left(x, \mathcal{T}, \mathcal{I}_{a}, r_{j}\right)$. Here the $j$ indicates the $j$ th iteration reflection and the $\mathcal{I}_{a}$ indicates the instruction for this error attribution operation. The identified tool $t_{j}$ paired with its documentation $d_{j}$ as well as the error message is taken as input, assisting the LLM to revise the generated program, which can be formulated as:

$$
\begin{equation*}
\mathcal{C}_{j}=\mathcal{M}_{\theta}\left(x, \mathcal{T}, \mathcal{D}, \mathcal{I}_{c},\left\{\left(\mathcal{C}_{<j}, r_{<j}\right)\right\}, d_{j}\right) \tag{2}
\end{equation*}
$$

Our attributable reflection mechanism is operated until the generated program is executed successfully or up to the maximum iteration $\alpha$.

## 4 Black-box Probing Enable Toolset Extension

Our framework ATC enables the LLM to directly operate a chain of well-documented tools through programming. However, manually crafting and maintaining documented protocols for diverse and fast-paced tools is cost-intensive, which poses a potential limitation to the scope of the toolset in our framework. Therefore, we propose a black-box probing method, which enables the LLM to act as an active tool learner, instructing themselves to master new tools. Due to the relation among the data flow of tools, we also introduce a chain of probing algorithms to enhance our probing process.

### 4.1 Tool Probing

As shown in Figure 2(b), our probing method contains two phases, including Instance discovery and Protocol documenting. The core idea of the former is to generate tool-use instances through self-exploration, examining the expected input-output mechanism for each tool, while the latter transforms specific instances into general tool protocol.

Instance discovery. We instruct the LLM to formulate a question $q$ targeting the functionality of a tool $t$ and generate a program utilizing the tool $t$ to solve the formulated question. Formally, it can be represented as $(q, \mathcal{C})=\mathcal{M}_{\theta}\left(t, \mathcal{I}_{p}\right)$, where $\mathcal{I}_{p}$ is the instruction for our instance discovery operation. The response $r$ of the tool $t$ can be examined while executing the generated program $\mathcal{C}$ as $r=\operatorname{Execute}(\mathcal{C})$, which represents a specific instance to demonstrate the output of the tool $t$. Since the LLM may hallucinate to formulate unsolvable questions or fail to generate a correct program, we repeat the above operation multiple times until the response $r$ can be examined correctly or up to the maximum of sampling times $N$. Thus, we obtain a tool-use instance denoted as $\left(\left(q_{i}, \mathcal{C}\right), r\right)$.

Protocol documenting. On top of of sampled instance $\left(\left(q_{i}, \mathcal{C}_{i}, r_{i}\right)\right)$, we construct the tool protocol. Since the response of real-world tools is typically lengthy with intricate structures, we first transform the query-specific response $r$ into a general schema $s$ to demonstrate the expected output specification. This process is automatically performed by recursively decomposing each element in $r$, representing the hierarchical structure of $r$, and listing the type of the corresponding value. Then, we utilize the question-program pair $(q, \mathcal{C})$ as a usage demonstration of the tool $t$, pairing it with $s$ to construct the documented protocol $d$, denoted as $d=((q, \mathcal{C}), s, t)$. We provide an example and detailed procedure to demonstrate the above transformation process in Appendix Alg. A.1.

### 4.2 Chain of Probing

During the probing, some tools may not be callable exclusively due to the absence of specific private arguments, which are only accessible through other tools, i.e., the unique ID. To address the strong data flow interconnection, we propose the chain of probing algorithm that enables the cooperation of tools. Formally, we denote the black-box toolset as $\mathcal{B}$, which contains unprobed tools and is initialized with the entire candidate toolset. The successfully probed tools are cached in the list $\mathcal{H}$, which is initialized with an empty list.

Initial Iteration for single tool probing. As illustrated in Figure 3, our initial iteration starts by probing each single tool $t$ within the black-box toolset $\mathcal{B}$, represented as $d=\operatorname{LLMProb}(t)$. The $\operatorname{LLMProb}(*)$ indicates the tool probing operation in $\S 4.1$. If a tool $t$ is successfully probed, i.e., no exceptional errors are raised, it is moved from black-box toolset $\mathcal{B}$ to list $\mathcal{H}$, formulated as:

$$
\begin{equation*}
\mathcal{B}=\mathcal{B} \backslash\{t\}, \quad \mathcal{H}=\mathcal{H} \cup\{t\} \tag{3}
\end{equation*}
$$

After the initial iteration, $\mathcal{H}$ contains tools that are directly callable like the tool $\mathrm{C}$ and $\mathrm{D}$ in Figure 3 while the remaining tools in $\mathcal{B}$ are interconnected with other tools.

Probing with dependency chain. For the remaining tools $t$ in $\mathcal{B}$, we probed them with the assistance of tools from $\mathcal{H}$. Specifically, we instruct the LLM to select a subset of tools from $\mathcal{H}$ based on their relevance to the tool $t$, which can be formulated as $\hat{\mathcal{T}}=\mathcal{M}_{\theta}\left(t, \mathcal{H}, \mathcal{I}_{s}\right)$. Here, the $\mathcal{I}_{s}$ denotes the instruction for tool selection. The selected subset $\hat{\mathcal{T}}$ serves as the prerequisite which facilitates the acquisition of necessary arguments to invoke the tool $t$ during the probing process, thereby deriving the tool protocol, rep-

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-05.jpg?height=395&width=677&top_left_y=1903&top_left_x=1060)
resented as $d=\operatorname{LLMProb}(\hat{\mathcal{T}} \cup\{t\})$. As shown in Figure 3, the tool $\mathrm{C}$ is selected to assist in probing the tool $\mathrm{B}$. Our probing process continues for multiple iterations until all tools in $\mathcal{B}$ are successfully probed or reach the maximum of iteration $\beta$. A detailed pseudo algorithm of the overall process is provided in Appendix Alg. 1 .

## 5 Dataset and Evaluation Setup

Existing Datasets. We first conduct experiments on widely used RestBench [14], including two datasets: TMDB and Spotify. The TMDB contains 100 tasks across 54 tools for movie scenarios while the Spotify contains 57 tasks across 40 tools for music scenarios. Each tool in the RestBench is paired with a lengthy documented protocol, making it inherently appropriate to benchmark the protocol utilization capability of LLMs.

A new benchmark - ToolFlow. As shown in Appendix Table 8, to the best of our knowledge, no existing benchmarks containing complex tools with comprehensive tool protocol (e.g., arguments requirement and input-output schema) while involving long-term planning tool-use tasks. Therefore, we build a new test set named ToolFlow to fill this gap. We first collect 107 tools with long protocols across 4 real-world domains, e.g., Weather and Game, from $16 \mathrm{k}$ public tools of the ToolBench [9] dataset. Then, we invite 7 well-trained experts working on NLP research to provide solutions for 224 complex tasks in the form of tool interaction sequences, including the tool name and corresponding arguments. Each task requires long-term reasoning and at least 7 times interacting with tools. ToolFlow also diverges from existing benchmarks by its strong interconnection among the tools (the arguments of subsequent tools can only be extracted from the response of previous tools) and stability (the task solution is not time-varying). We provide more details of ToolFlow in A.2.

Evaluation metrics. Following previous work [49, 50], we use three evaluation metrics, including: (1) Success Rate (Success\%), which measures the proportion of successful query completions; (2) Correct Path Rate (Path\%), which calculates the proportion of ground truth tools in model-generated tool callings; (3) Correct Tool Precision (Prec\%), which calculates the precision score between the model-generated tool callings and ground truth tool sequence. We also conduct the human evaluation to evaluate our method and the details can be found in Appendix A.1.2.

Baselines. We mainly compare our method with the well-known baselines, including: (1) ReAct [18], which prompts LLM to generate the chain-of-thought and actions in an interleaved manner; (2) CodeAct [12], which prompts LLM to iteratively generate code snippets as actions to call external tools; (3) ToolLLM-DFSDT[9], which enhances LLMs with the Depth First Search-based Decision Tree (DFSDT) to select tools to solve a task; (4) RestGPT [14], which includes a coarse-to-fine planning module and a tool executor; (5) ConAgents [26], which enables the cooperation of three specialized LLMs to solve complex tasks. For further comparison, We also establish two baselines, i.e., ReAct@3 and ToolLLM@3, which are up to three times runs of their vanilla method (ReAct or ToolLLM) until the input task is successfully completed.

## 6 Experiment Results

### 6.1 Results of RQ1 - Enable the LLM as an automatic muti-tool user

We utilize three widely used LLMs for different baselines and our method: OpenAI's gpt-3.5-turbo$16 \mathrm{k}$ and gpt-4-turbo, and the open-source model Mixtral- $8 x 7 B[51]^{3}$ The decoding temperature is set to 0 for deterministic generation. The trial number $\alpha$ in our reflection mechanism $(\S 3.3$ is set to 3 . Following previous work [26, 14], we provide all the methods with 20 candidate tools for each task in the test set, which contains the required tools and randomly sampled tools.

Results on RestBench. As shown in Table 1, the LLM, when equipped with our framework, surpasses all the baselines on the RestBench benchmark in terms of all metrics. For example, our method achieves 89.00 in success rate metrics on the RestBench-TMDB dataset, which substantially improves over the commonly used baseline ReAct and ToolLLM. Table 2 and Table 3 further illustrate that our framework can achieve the best performance with various backbone LLMs, i.e., the Mistral8x7B and GPT-4. These results indicate our framework effectively enables LLM to master external tools and directly generate a program for utilization. The performance of two runs is tested using a two-tailed paired t-test where no significant difference is found ( $p>0.05$ ), showing the stability of our method. In addition, human evaluation indicates that our method performs substantially better on executability and utility than strong baselines. See Appendix A.1.2 for details.[^1]

Table 1: Experiment results on three datasets with gpt-3.5-turbo as backbone. The Path\%, Prec\% and Success\% indicate Correct Path Rate, Correct Path Precision and Successful Rate metrics.

| Method | RestBench-TMDB |  |  | RestBench-Spotify |  |  | ToolFlow |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Succ\% | Path\% | Prec\% | Succ\% | Path\% | Prec\% | Succ\% | Path\% | Prec\% |
| gpt-3.5-turbo |  |  |  |  |  |  |  |  |  |
| ReAct [18] | 61.00 | 77.13 | 52.30 | 50.88 | 74.64 | 44.79 | 22.76 | 60.75 | 68.03 |
| CodeAct [26] | 63.00 | 80.91 | 83.72 | 54.30 | 76.64 | 79.81 | 27.82 | 57.93 | 66.23 |
| ToolLLM $[9]$ | 72.00 | 78.29 | 49.41 | 61.40 | 82.82 | 25.33 | 42.14 | 71.02 | 65.24 |
| RestGPT [14] | 65.00 | 77.49 | 80.15 | 64.91 | 73.94 | 88.71 | 26.83 | 40.95 | 62.21 |
| ConAgents [26 | 76.00 | 78.29 | 82.31 | 63.16 | 78.21 | 82.71 | - | - | - |
| $\overline{\operatorname{ReAct}} \mathrm{C} 3$ | $\overline{70.00}$ | $8 \overline{0.96}$ | $4 \overline{8} .01$ | $\overline{59} . \overline{65}$ | $\overline{8} 1 . \overline{8} 0$ | 30.48 | $-28.3 \overline{5}$ | $6 \overline{6} .6 \overline{6}$ | $\overline{66} . \overline{21}$ |
| ToolLLM@3 | 74.00 | 83.29 | 45.41 | 66.67 | 83.41  | 23.73 | 44.70 | 73.85 | 60.77 |
| $\overline{\mathbf{A T}} \overline{\mathbf{C}}(\overline{\text { ours }})$ | $\mathbf{8 9 . 0 0}$ | $8 \overline{4} .7 \overline{1}$ | $8 \overline{3} . \overline{87}$ | $\overline{78 .} \overline{95}$ | $\overline{7} 8 . \overline{5} 4$ | 91.46 | $60.2 \overline{1}$ | $7 \overline{8} .3 \overline{1}$ | $\overline{72.45}$ |

Table 2: Experiment with the Mistral-8x7B.

| Method | TMDB |  |  | ToolFlow |  |
| :--- | :---: | :---: | :---: | :---: | :---: |
|  | Succ\% | Path\% |  | Succ\% | Path\% |
| mixtral-8x7B-instruct-v0.1 |  |  |  |  |  |
| ReAct | 24.74 | 73.34 |  | 10.53 | 41.37 |
| ReAct@3 | 37.88 | 76.85 |  | 18.95 | 52.40 |
| ToolLLM@3 | 45.00 | 74.40 |  | 22.54 | 51.85 |
| Ours | $\mathbf{5 8 . 0 0}$ | $\mathbf{7 8 . 1 7}$ |  | $\mathbf{2 9 . 8 7}$ | $\mathbf{5 9 . 1 4}$ |

Table 3: Experiment with the GPT-4.

| Method | TMDB |  |  | ToolFlow |  |
| :--- | :---: | :---: | :---: | :---: | :---: |
|  | Succ\% | Path\% |  | Succ\% | Path\% |
| gpt-4-turbo |  |  |  |  |  |
| ReAct | 77.00 | 86.05 |  | 25.99 | 65.98 |
| ReAct@3 | 80.00 | 89.21 |  | 30.98 | 67.55 |
| ToolLLM@3 | 82.00 | 90.62 |  | 50.46 | 76.73 |
| Ours | $\mathbf{9 4 . 0 0}$ | $\mathbf{9 2 . 6 8}$ |  | $\mathbf{6 5 . 7 4}$ | $\mathbf{8 3 . 5 4}$ |

Results on ToolFlow. Table 1 presents the results on our ToolFlow benchmark. We find that our ToolFlow poses a significant challenge for previous baselines, where the best performance only achieves a 44.70 success rate with GPT-3.5 as the backbone. Our method pushes the success rate to 60.21 with a 15.51 point improvement. The potential reason for our improvement is that our ATC enables the LLM to generate a chain of tools programmatically, which is more effective in controlling workflow and consolidating lengthy task-solving trajectories into a concise program.

Ablation for our attributable reflection. We compare our attributable reflection method with two ablative variants: (1) w/o reflect, which allows the LLMs to generate a program as the final solution without further revision, and (2) w/ naive reflection, which enables the LLMs to revise generated programs directly using error messages from code interpreter. The results are shown in Table4, We observe that our attributable reflection outperforms the two variants with a $5-10 \%$ point improvement. It demonstrates the superiority of the reflection mechanism and the effectiveness of our error attribution strategy.

### 6.2 Results of RQ2 - Enable the LLM as an active Multi-tool learner

We evaluate our black-box probing method on three datasets using different backbone LLMs, i.e., RestBench-TMDB, RestBench-Spotify, and ToolFlow, respectively. The sampling number $N$ ( $\$ 4$ ) is set to 3 and the maximum iteration number $\beta$ ( $\$ 4.2$ is set to 4 . We mainly evaluate our probing method by computing the number of successfully probed tools. To evaluate the utility of the autodocumented protocol, we compare the performance of our ATC supported by standard protocol (std protocol) and synthetic protocol (auto protocol). Considering that our synthetic documentation contains a usage example for each tool, we further set a zero-shot experiment, which only remains the transformed schema (auto schema).

Success rate of tool probing. Table 5 shows the number of successfully probed tools. We find that the open-source model Mixtral-8x7B, when equipped with our probing method, can probe $82.5 \%$ to $88.2 \%$ of tools and synthesize their tool documentation. The number of successfully probed tools also increases when alternating more powerful backbone LLMs, specifically GPT-4. These results validate the effectiveness of our tool probing method. We further analyse the cases where the LLM

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-08.jpg?height=545&width=1395&top_left_y=237&top_left_x=365)

Figure 4: The comparison of our framework when equipped with different documentation.

fails to probe successfully. A potential reason is that the code interpreter only examines compile errors and runtime faults, failing to calibrate the correctness of the program's output. This limitation can lead to the False Success phenomena.


#### Abstract

Ablation for the black-box probing. We compare our tool probing with two ablative variants: (1) w/o multi-sample, which replaces the multiple sampling strategy in $\S 4$ with sampling only one instance; and (2) w/o chain, which ignore the dependency chain among tools in $\S 4.2$ and separately probes single tools. As shown in Table 5 , in terms of the number of successfully probed tools, we observe a 3-7 point decrease for w/o multi-sample, which indicates that the LLMs may fail to generate a correct program at

Table 5: The number of successfully probed tools using our vanilla probing method and two variants.

| Method | TMDB | Spotify | NovelTools |
| :---: | :---: | :---: | :---: |
| Totally | 54 | 40 | 107 |
| Probing (mixtral) | 47 | 33 | 90 |
| Probing (gpt-4) | 54 | 38 | 102 |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-08.jpg?height=43&width=231&top_left_y=1254&top_left_x=1078) | $\overline{54}$ | 38 | $9 \overline{9}$ |
| - w/o multi-sample | $50_{\downarrow 4}$ | $35_{\downarrow 3}$ | $91_{\downarrow 7}$ |
| - w/o chain | $47_{\downarrow 7}$ | $17_{\downarrow 21}$ | $87_{\downarrow 11}$ |


one pass. We also find a substantial decrease between our vanilla probing method and the w/o chain variant. These results demonstrate the necessity of optimizing the combination of the tools with strong interconnection.

Utility of Auto-documented Protocol. Figure 4 shows the performance of our proposed framework in different settings. Compared with using standard protocol crafted manually (i.e., Ours w/ std protocol), the LLM achieves comparable performance with the assistance of auto-documented protocol (i.e., Ours w/ auto protocol), which illustrates the utility of our synthetic protocol. We also observe that our framework substantially outperforms the best baseline even when only using the transformed schema, i.e., Ours w/ auto schema. This result further demonstrates the effectiveness of our tool probing and protocol documenting methods which can extend our proposed framework into diverse new tools without handcrafted protocols. In addition, we also conduct the case study to evaluate the quality of the synthetic protocol and show a concrete example in A.3.

## 7 Discussion

The impact of iteration count in our attribution reflection. Our attribution reflection mechanism enables LLMs to adaptively revise their generated programs according to error messages raised by the program interpreter. We further alternate the maximum reflection count $\alpha$ from 1 to 5 and evaluate the Success Rate with the same setting as Table 1 (ours). As shown in Figure 5, we observe an increasing Success Rate when $\alpha$ shifts from 1 to 3, which illustrates that LLMs can adapt their generation accordingly. We also find a relatively stable trend when the $\alpha$ keeps increasing (from 3 to 5), which indicates that the LLMs can revise most of the errors in 3 iterations. We also analyse the cases of unsuccessful corrections and we find that the generated program may be grammatically

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-09.jpg?height=200&width=702&top_left_y=255&top_left_x=365)

(a) Error distribution

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-09.jpg?height=208&width=190&top_left_y=243&top_left_x=1054)

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-09.jpg?height=200&width=488&top_left_y=247&top_left_x=1255)

(b) Error type distribution

Figure 6: The statistics of the error of our framework. Left: We calculate the success and failure rates for tasks that require different numbers of tools. Right: The percentage of different type of error.

correct but yields incorrect answers and therefore cannot be detected by the compiler (i.e., the False Success phenomena).

Statistics of error cases. To further evaluate the potential advancement and drawback of our method (Table 1), we count the success and failure rates for tasks with different complexity. We first randomly sample a total of 130 tasks from the RestBench and our ToolFlow. Following previous work [14, 9], we assess the task complexity using the number of tools calling in the ground truth solution. Figure 6.a) represents the results. We find that the LLM when equipped with our framework, can effectively solve both short-term planning (i.e., $3 \geq$ tool num.) and long-term planning (i.e., 7 $\leq$ tool num.). We also analyse the type of failure tasks and divide them into four categories shown in Figure 6.b). Most of the errors are derived from misunderstanding the tool documentation or the mismatch between a task and selected tools. A potential solution is to enrich the tool documentation, further clarify the distinction among similar tools, and append negative examples as prior experience in the tool documentation to instruct LLMs to well master a tool. We take it as our future work.

Efficiency at inference. The intensive parameters of LLMs typically raise the concern about inference cost. Thus, we compare the token consumption between our framework (auto $d o c$ ) and strong baselines on the TMDB and ToolFlow datasets and show the results in Figure 7 to explain more intuitively. We observe that although our framework achieves better performance, we spend fewer tokens compared with all baselines. The potential reason is that our framework benefits from the inherent advancement of programming language which sup-

![](https://cdn.mathpix.com/cropped/2024_06_04_20442f848b1cffa6777ag-09.jpg?height=377&width=697&top_left_y=1145&top_left_x=1061)
ports control of the workflow and allows the composition of multiple tools to perform complex logical operations. By contrast, the previous baseline interacts with tools in a step-by-step manner, leading to a long task-solving trajectory with substantial inference costs. We also compute the token consumption for our probing process, where each tool costs 2703 tokens to prob on average. More details can be found in A. 1

Case study. We conduct the case studies and find that our proposed framework is more effective at utilizing various tools to solve complex tasks. We also provide concrete examples to intuitively explain each component of our method in A. 3

## 8 Conclusions

We presented Automatic Tool Chain (ATC), a framework that enables the LLM to act as a multi-tool user. ATC enables LLMs to learn input-output schemas and data flow dependency of various tools from documented tool protocols, programmatically generating a chain of tools to solve complex tasks. ATC overcomes the limitations of existing tool learning methods, including relying on manually designed workflows and lengthy inference steps. On top of ATC, we propose a black-box probing method, empowering the LLM to act as a multi-tool learner that can automatically discover tool protocols and teach itself to master new tools. Extensive experiments conducted on existing datasets and a newly created challenging benchmark demonstrate that an LLM, when equipped with our framework, achieves the best performance compared with all the baselines. We expect future research to further calibrate the output of generated programs, mitigating the false success phenomena, i.e., the program triggers no runtime error but still gives an incorrect answer. We are also interested in exploring the integration of our framework into vision foundation models, to develop a multi-modal agent to solve complex practical tasks.

## References

[1] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-Instruct: Aligning Language Models with Self-Generated Instructions. In Association for Computational Linguistics: ACL, 2023.

[2] Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, and Tianyi Zhou. A Survey on Knowledge Distillation of Large Language Models. arXiv, 2024.

[3] Saaket Agashe, Yue Fan, and Xin Eric Wang. Evaluating multi-agent coordination abilities in large language models. arXiv preprint arXiv:2310.03903, 2023.

[4] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, et al. Tool learning with foundation models. arXiv preprint $\underline{\operatorname{arXiv}: 2304.08354,2023 .}$

[5] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language Models Can Teach Themselves to Use Tools. Neural Information Processing Systems: NeurIPS, 2023.

[6] Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao Liang, Kunlun Zhu, Yankai Lin, Xu Han, Ning Ding, Huadong Wang, Ruobing Xie, Fanchao Qi, Zhiyuan Liu, Maosong Sun, and Jie Zhou. WebCPM: Interactive web search for Chinese long-form question answering. In Association for Computational Linguistics: ACL, 2023.

[7] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564, 2023.

[8] Andres M Bran, Sam Cox, Andrew D White, and Philippe Schwaller. ChemCrow: Augmenting large-language models with chemistry tools. arXiv preprint arXiv:2304.05376, 2023.

[9] Yujia Qin, Shi Liang, Yining Ye, Kunlun Zhu, Lan Yan, Ya-Ting Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, Ruobing Xie, Jie Zhou, Marc H. Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. International Conference on Learning Representations: ICLR, 2023.

[10] Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. Os-copilot: Towards generalist computer agents with self-improvement. arXiv preprint arXiv:2402.07456, 2024.

[11] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

[12] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better llm agents. arXiv preprint arXiv:2402.01030, 2024.

[13] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large Language Model Connected with Massive APIs. arXiv preprint arXiv:2305.15334, 2023.

[14] Yifan Song, Weimin Xiong, Dawei Zhu, Chengzu Li, Ke Wang, Ye Tian, and Sujian Li. RestGPT: Connecting Large Language Models with Real-World Applications via RESTful APIs. arXiv, 2023.

[15] Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi R Fung, Hao Peng, and Heng Ji. Craft: Customizing llms by creating and retrieving from specialized toolsets. International Conference on Learning Representations: ICLR, 2024.

[16] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. Adapt: As-needed decomposition and planning with language models. arXiv preprint arXiv:2311.05772, 2023.

[17] Cheng Qian, Chi Han, Yi Fung, Yujia Qin, Zhiyuan Liu, and Heng Ji. Creator: Tool creation for disentangling abstract and concrete reasoning of large language models. In Findings of the Association for Computational Linguistics: EMNLP, 2023.

[18] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations: ICLR, 2023.

[19] Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, and Marco Tulio Ribeiro. Art: Automatic multi-step reasoning and tool-use for large language models. arXiv preprint arXiv:2303.09014, 2023.

[20] Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin. Lumos: Learning agents with unified data, modular design, and open-source llms. arXiv preprint arXiv:2311.05657, 2023.

[21] Boshi Wang, Hao Fang, Jason Eisner, Benjamin Van Durme, and Yu Su. LLMs in the Imaginarium: tool learning through simulated trial and error. arXiv preprint arXiv:2403.04746, 2024.

[22] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. Neural Information Processing Systems: NeurIPS, 2023.

[23] Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, and Le Sun. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. arXiv preprint arXiv:2306.05301, 2023.

[24] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.

[25] Cheng-Yu Hsieh, Si-An Chen, Chun-Liang Li, Yasuhisa Fujii, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, and Tomas Pfister. Tool documentation enables zero-shot tool-usage with large language models. arXiv preprint arXiv:2308.00675, 2023.

[26] Zhengliang Shi, Shen Gao, Xiuyi Chen, Lingyong Yan, Haibo Shi, Dawei Yin, Zhumin Chen, Pengjie Ren, Suzan Verberne, and Zhaochun Ren. Learning to use tools via cooperative and interactive agents. arXiv preprint arXiv:2403.03031, 2024.

[27] Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang. Easytool: Enhancing llm-based agents with concise tool instruction. arXiv preprint arXiv:2401.06201, 2024.

[28] Qiao Jin, Yifan Yang, Qingyu Chen, and Zhiyong Lu. Genegpt: Augmenting large language models with domain tools for improved access to biomedical information. Bioinformatics, 2024.

[29] Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, and Fei Huang. Small llms are weak tool learners: A multi-llm agent. arXiv preprint arXiv:2401.07324, 2024.

[30] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 2024.

[31] Shuofei Qiao, Ningyu Zhang, Runnan Fang, Yujie Luo, Wangchunshu Zhou, Yuchen Eleanor Jiang, Chengfei Lv, and Huajun Chen. AUTOACT: Automatic Agent Learning from Scratch via Self-Planning. arXiv preprint arXiv:2401.05268, 2024.

[32] Ke Yang, Jiateng Liu, John Wu, Chaoqi Yang, Yi R Fung, Sha Li, Zixuan Huang, Xu Cao, Xingyao Wang, Yiquan Wang, et al. If llm is the wizard, then code is the wand: A survey on how code empowers large language models to serve as intelligent agents. arXiv preprint $\underline{\operatorname{arXiv}: 2401.00812,2024 .}$

[33] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T Joshi, Hanna Moazam, et al. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714, 2023.

[34] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023.

[35] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.

[36] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: Program-aided language models. In Proceedings of Machine Learning Research: PMLR, 2023.

[37] Chen Qian, Yufan Dang, Jiahao Li, Wei Liu, Weize Chen, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Experiential co-learning of software-developing agents. arXiv preprint arXiv:2312.17025, 2023.

[38] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 19632-19642, 2024.

[39] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for multi-agent collaborative framework. arXiv, 2023.

[40] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.

[41] Yao Fu, Hao Peng, Tushar Khot, and Mirella Lapata. Improving language model negotiation with self-play and in-context learning from ai feedback. arXiv preprint arXiv:2305.10142, 2023 .

[42] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.

[43] Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Neural Information Processing Systems: NeurIPS, 2023.

[44] Xingyao Wang, Hao Peng, Reyhaneh Jabbarvand, and Heng Ji. Leti: Learning to generate from textual interactions. arXiv preprint arXiv:2305.10314, 2023.

[45] Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128, 2023.

[46] Haoqiang Kang, Juntong Ni, and Huaxiu Yao. Ever: Mitigating hallucination in large language models through real-time verification and rectification. arXiv preprint arXiv:2311.09114, 2023.

[47] Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. How language model hallucinations can snowball. arXiv preprint arXiv:2305.13534, 2023.

[48] Dheeraj Mekala, Jason Weston, Jack Lanchantin, Roberta Raileanu, Maria Lomeli, Jingbo Shang, and Jane Dwivedi-Yu. Toolverifier: Generalization to new tools via self-verification. arXiv preprint arXiv:2402.14158, 2024.

[49] Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction. Neural Information Processing Systems: NeurIPS, 2023.

[50] Shen Gao, Zhengliang Shi, Minghang Zhu, Bowen Fang, Xin Xin, Pengjie Ren, Zhumin Chen, Jun Ma, and Zhaochun Ren. Confucius: Iterative tool learning from introspection feedback by easy-to-difficult curriculum. In Proceedings of the AAAI Conference on Artificial Intelligence: AAAI, 2024.

[51] Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L'elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

[52] Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, and Jian Zhang. On the tool manipulation capability of open-source large language models. arXiv preprint arXiv:2305.16504, 2023.

[53] Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. API-bank: A comprehensive benchmark for tool-augmented LLMs. In Association for Computational Linguistics: EMNLP, 2023.

[54] Junjie Ye, Guanyu Li, Songyang Gao, Caishuang Huang, Yilong Wu, Sixian Li, Xiaoran Fan, Shihan Dou, Qi Zhang, Tao Gui, et al. Tooleyes: Fine-grained evaluation for tool learning capabilities of large language models in real-world scenarios. arXiv preprint arXiv:2401.00741, 2024.

[55] Zhicheng Guo, Sijie Cheng, Hao Wang, Shihao Liang, Yujia Qin, Peng Li, Zhiyuan Liu, Maosong Sun, and Yang Liu. Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models. arXiv preprint arXiv:2403.07714, 2024.

[56] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36, 2024.
</end of paper 1>


<paper 2>
# Divide-and-Conquer Meets Consensus: Unleashing the Power of Functions in Code Generation 

Jingchang Chen ${ }^{1 *}$ Hongxuan Tang ${ }^{1 *}$ Zheng Chu ${ }^{1} \quad$ Qianglong Chen $^{2}$<br>Zekun Wang ${ }^{1} \quad$ Ming Liu ${ }^{1 \dagger}$ Bing Qin ${ }^{1}$<br>${ }^{1}$ Harbin Institute of Technology<br>${ }^{2}$ Zhejiang University<br>\{jcchen, zchu, zkwang, mliu, qinb\}@ir.hit.edu.cn<br>jeffswt@outlook.com chenqianglong.ai@gmail.com


#### Abstract

Despite recent progress made by large language models in code generation, they still struggle with programs that meet complex requirements. Recent work utilizes plan-and-solve decomposition to decrease the complexity and leverage selftests to refine the generated program. Yet, planning deep-inside requirements in advance can be challenging, and the tests need to be accurate to accomplish self-improvement. To this end, we propose FUNCODER, a code generation framework incorporating the divide-and-conquer strategy with functional consensus. Specifically, FUNCODER recursively branches off sub-functions as smaller goals during code generation, represented by a tree hierarchy. These sub-functions are then composited to attain more complex objectives. Additionally, we designate functions via a consensus formed by identifying similarities in program behavior, mitigating error propagation. FUNCODER outperforms state-of-the-art methods by $+9.8 \%$ on average in HumanEval, MBPP, xCodeEval and MATH with GPT-3.5 and GPT-4. Moreover, our method demonstrates superiority on smaller models: With FUNCODER, StableCode ${ }_{3 b}$ surpasses GPT-3.5 by $+18.6 \%$ and achieves $97.7 \%$ of GPT-4's performance on HumanEval. Further analysis reveals that our proposed dynamic function decomposition is capable of handling complex requirements, and the functional consensus prevails over self-testing in correctness evaluation.


## 1 Introduction

Over the past few years, large language models have been observed to attain significant advancements in coding capabilities (OpenAI, 2023, Touvron et al. 2023). Meanwhile, models designed specifically for coding tasks have also been introduced (Rozière et al., 2023; Lozhkov et al., 2024; Pinnaparaju et al. 2024). Although LLMs can proficiently generate simple code snippets, they suffer from a decline in performance as code requirements become complicated.

Numerous efforts have been made to tackle this complexity. The two-stage methods (Jiang et al., 2023; Zelikman et al., 2023) employ the plan-and-solve strategy, which first generates a draft outline for the complex task and uses it as guidance for implementing the code in the second stage. Multiagent development frameworks (Hong et al., 2024, Qian et al., 2023) mimic real-world software development workflows, assign different roles to LLMs and collaborate to solve a complex goal. Self-improvement (Shinn et al. 2023; Chen et al., 2024), on the other hand, refines the program in accordance with execution feedback from self-generated unit tests.

Despite fruitful efforts made by the previous methods in dealing with complex problems, certain challenges still remain unsolved: (1) Two-stage approaches need to design a complete plan at the[^0]![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-02.jpg?height=740&width=1400&top_left_y=190&top_left_x=358)

Figure 1: A flowgraph illustrates FUNCODER. FUNCODER branches off new functions to have sub-goals tackled iteratively (left), re-composites sub-functions, and selects the best using functional consensus (right). Bottom-right figure shows how FUNCODER writes functions at hierarchy-level.

beginning and lack the ability to adjust the top-level design during implementation, leading to suboptimal decomposition. (2) Multi-agent collaboration frameworks are cumbersome and rely heavily on LLM capabilities, making them difficult to generalize to smaller open-source models. (3) Code refinement through self-tests depends on the correctness of generated unit-tests. Our preliminary study ( $\$ 3.1 .3$ ) finds that models generate unreliable self-tests in abundance. These incorrect tests may mislead self-improvement and, at worse, exacerbate program errors.

To address these issues, we propose FUNCODER, a code generation framework utilizing a divide-andconquer strategy and a novel functional consensus mechanism on functions to decompose complex problems. Starting from the main problem, FUNCODER introduces new functions to cope with certain sub-problems. The new functions will be decomposed recursively, eventually forming a tree of functions. FUNCODER then combines functions bottom-up to achieve increasingly complicated objectives. By dividing-and-conquering tasks into simpler sub-functions, complexity can be gradually reduced. However, errors in sub-functions may propagate to the whole program, thereby damaging overall reliability. We propose functional consensus that samples multiple functions and selects the one demonstrating consensus, measured by the aggregated similarity among candidates. By reaching a consensus, we reduce the discrepancies in code behavior and thus alleviate cascading errors.

We conduct extensive experiments on code generation benchmarks (Chen et al., 2021; Austin et al., 2021; Khan et al., 2023) with GPT (Ouyang et al., 2022; OpenAI, 2023), outperforming state-ofthe-art methods by $+9.8 \%$ on average. Experiments are further carried out on the mathematical competition benchmark, MATH (Hendrycks et al. 2021b), achieving a $+\mathbf{6 . 0}$ improvement with GPT-4, indicating that FUNCODER can also generalize to complex reasoning. Our method is observed to be equally effective on open-source models (Rozière et al., 2023, Pinnaparaju et al. 2024, Meta AI. 2024), with an average gain over baseline of $+\mathbf{3 8 . 0} \%$ on HumanEval and $+\mathbf{6 1 . 1} \%$ on MATH. Additional analysis also shows the advantage of both divide-and-conquer and functional consensus.

## 2 FunCoder: Divide-and-Conquer Meets Consensus

### 2.1 Divide-and-Conquer for Iterative Programming

A function is defined as a relation between a set of inputs and outputs where each input is assigned exactly one output (Halmos, 1998), denoted as $y=f(x)$. In computer programming, a function is identified by its header $h_{f}$ with its body $b_{f}$, and is commonly accompanied by a documentation $d_{f}$ to improve readability. Functions can be invoked from other procedures, allowing for the decomposition of large and complicated requirements into smaller structures that exhibit high comprehensibility and quality (Dahl et al. 1972). Generally, human programmers tend to decompose tasks into clearly

```
Algorithm 1 FUNCODER procedure
Require: Entry func, $f_{\text {root }}=\left\{h_{\text {root }}, d_{\text {root }}, \phi\right\}$
Require: Large language model, LLM
    function FUNCODER $\left(f_{\text {cur }}\right)$
        - Divide -
        $f_{\text {cur }}^{\prime},\left\{f_{i}\right\} \leftarrow \operatorname{EXTRACT}\left(\operatorname{LLM}\left(f_{\text {cur }}\right)\right)$
        for $f_{i} \in\left\{f_{i}\right\}$ do
            if $b_{i}$ is NOTIMPLEMENTED then
                $f_{i}^{*} \leftarrow \operatorname{FUNCODER}\left(f_{i}\right) \triangleright$ recursion
            end if
            $\operatorname{AdDCHILD}\left(f_{\text {cur }}, f_{i}^{*}\right)$
        end for
        - Conquer -
        $F_{\text {cur }} \leftarrow \operatorname{SAMPLE}\left(\operatorname{LLM}\left(f_{\text {cur }}^{\prime}, \operatorname{CHILD}\left(f_{\text {cur }}\right)\right)\right.$
        $f_{\text {cur }}^{*} \leftarrow$ FUNCONSENSUS $\left(F_{\text {cur }}\right)$
        return $f_{\text {cur }}^{*}$
    end function
    return FUNCODER $\left(f_{\text {root }}\right) \quad \triangleright$ starts from root
```

![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-03.jpg?height=228&width=624&top_left_y=257&top_left_x=1127)

(b) Decompose Through Coding (ours)

![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-03.jpg?height=461&width=615&top_left_y=526&top_left_x=1140)

Figure 2: Left: Algorithm for FUNCODER procedure. Right: Comparison between decomposition by planning and our approach. FUNCODER introduces new functions to describe sub-goals solely with code, achieving a more natural way of requirement decomposition.

defined sub-functions and then implement them recursively, making functions eligible for re-usage, taking advantage of the divide-and-conquer principle. Inspired by this, FUNCODER recursively divides the requirement and conquers functions to formulate a sophisticated solution, unleashing the potential of LLMs in code generation.

Divide is a top-down process that iteratively breaks down problems. Given a code generation problem, the process begins from the entry function $f_{\text {root }}$. We instruct the model to introduce new functions $f_{i} \in \operatorname{CHILD}\left(f_{\text {cur }}\right)$ that solve certain sub-goals while writing the current $f_{\text {cur }}$. To reduce the complexity involved in each generation, we only require the headers $h_{f_{i}}$ and documentation $d_{f_{i}}$ of new functions to be generated, while their implementations $b_{f_{i}}$ can be postponed. After completing the current function, the model starts to address those unimplemented sub-functions and complete $b_{f_{i}}$ into $f_{i}^{\prime}$. This process stops when the model deems functions too simple to be further divided, finally forming a dependency tree $T=\operatorname{TREE}\left(f_{\text {root }}, \operatorname{CHILD}\left(f_{\text {root }}\right)\right)$. The divide process is similar to a search starting from the entry function, gradually involving new sub-functions while writing the current, and implementing them recursively. We guide the entire process through a depth-first search.

Conquer is a process of achieving complex objectives through aggregating smaller functions. We notice that child functions are not yet implemented during the top-down process of writing parent functions. As a result, these parent functions may not be able to effectively utilize the child functions, or misuse them at worst. FUNCODER deals with this issue by re-generating functions in inverse topological order on the dependency tree $T$ - starting from leaves, complex goals are handled by compositing solved children as $f_{\text {cur }}^{*} \leftarrow \mathcal{F}\left(f_{\text {cur }}^{\prime},\left\{f_{1}^{*}, f_{2}^{*}, \ldots\right\}\right) \mid f_{i}^{*} \in \operatorname{CHILD}\left(f_{\text {cur }}\right)$.

Divide and conquer naturally achieve both decomposition and composition during code generation. Unlike two-stage and agent-based methods, our approach dynamically introduces new functions along the process, making it less burdensome than producing a complete plan at the very beginning. Moreover, while planning or agents require chat capabilities, FUNCODER represents sub-tasks through functions (Figure 2, making it more applicable to specialized code generation models.

### 2.2 Functionality Similarity as a Consensus

The decomposition of complex tasks benefits from solving easier sub-goals, but might introduce the risks of cascading errors. To mitigate this, we introduce Functional Consensus which aims at reducing inconsistencies in program behavior. This is achieved by sampling multiple functions and selecting the one that exhibits consensus, as measured by the aggregated similarity of functionality between candidates, thus abating outlier functionalities.

Functionality Similarity A program specifies its functionality (or behavior) through the control flow defined by its code semantics. However, comparing the functionalities between two programs based on their semantics is somewhat challenging. By decomposing the requirement into functions, FUNCODER is able to view the function behavior as a black box that maps arguments into return values. Considering two functions $f$ and $g$ with the same input domain $D(f)=D(g)$, we define the similarity between them $\operatorname{sim}(f, g)$ as the identicalness of outputs when given the same input values.

$$
\begin{equation*}
\operatorname{sim}(f, g)=\int_{x \in D(f)} \frac{\mathbb{1}[f(x)=g(x)]}{|D(f)|} \approx \sum_{x \in X \mid X \sim D(f)} \frac{\mathbb{1}[f(x)=g(x)]}{|X|} \tag{1}
\end{equation*}
$$

The similarity becomes 1 if and only if two functions output consistent values for all inputs: $\forall x \in$ $D(f): f(x)=g(x) \Leftrightarrow \operatorname{sim}(f, g)=1$. We notice that the input domain $D(f)$ is unbounded in most cases, making its measurement barely feasible in practice. Thus, we approximate it by sampling a subset of possible inputs $X \sim D(f)$ with an LLM.

Consensus is reached by selecting the candidate $f^{*}$ holding maximal similarity with others after sampling multiple function implementations $F=\left\{f_{(i)}\right\}$ for the same requirements.

$$
\begin{equation*}
f^{*}=\operatorname{FUNCONSENSUS}(F)=\underset{f_{(i)} \in F}{\arg \max } \sum_{f_{(j)} \in F \backslash\left\{f_{(i)}\right\}} \operatorname{sim}\left(f_{(i)}, f_{(j)}\right) \tag{2}
\end{equation*}
$$

By introducing functional consensus, FUNCODER produces functions that are more consistent and common in functionality, while omitting abnormal samples. The process is applied to not just the final program, but also to every sub-tree during the bottom-up conquering stage, resulting in step-by-step, thorough verification from the most fundamental functions all the way up to the whole program.

### 2.3 FunCoder is a Function Coder

We design FUNCODER as a procedure that takes a problem in the form of a function signature $f(x)$, and produces a final solution $f^{*}(x)$, as exemplified in Figure 1 Given a problem $f(x)$, FUNCODER partially implements the function as $f^{\prime}(x)$ referring to unimplemented sub-functions $g(y)$ and $h(z)$. These sub-functions are then fed into FUNCODER to be recursively coped with. We then sample $k$ implementations $f_{(i)}^{\prime}(x)$ based on solved children $g^{*}(y)$ and $h^{*}(z)$. Functional consensus is calculated by evaluating candidates on possible inputs. The function sharing maximal behavioral similarity is combined with solved children to formulate the final solution.

## 3 Experiments

We conduct experiments on competition-level code generation and mathematical reasoning benchmarks with state-of-the-art LLMs, which are covered in section $\$ 3.1$ and $\$ 3.2$, respectively. In addition to GPT models (Ouyang et al., 2022, OpenAI, 2023), we also conduct experiments with community models like Llama3 $3 b$ (Meta AI, 2024), StableCode $3 b$ (Pinnaparaju et al., 2024), and CodeLlama $_{34 b}$ (Rozière et al. 2023). We use the instruct variant of these models and inference on a single A100-80G under BF16 precision with vLLM (Kwon et al., 2023).

### 3.1 Code Generation

We choose three benchmarks for code generation evaluation: (a) HumanEval (Chen et al., 2021) includes entry-level coding questions; (b) MBPP (Austin et al., 2021) contains questions of standard library invocation and programming basics; and (c) xCodeEval (Khan et al., 2023) consists of algorithmic challenges sourced from the competitive programming platform CodeForces.

### 3.1.1 Experiment Setup

Benchmarks We adopt the full test set (164 problems) for HumanEval, and sample 200 for MBPP and 500 for xCodeEval, respectively. Following EbTech (2024), we split the xCodeEval into 4 subsets based on problem difficulty: Easy ( $\leq 1200$ ), Mid (1200-1599), Hard (1600-1999) and Expert $(\geq 2000)$. The evaluation metric for code generation is Pass @ 1 unless specified.

Table 1: Experiment results on code generation benchmarks. We report Pass @ 1 as evaluate metric. Results from the original paper are underlined, and the best results are bold.

| Model | Method | HumanEval |  | MBPP |  | xCodeEval |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Pass@1 | $\Delta \uparrow$ | Pass@1 | $\Delta \uparrow$ | Easy | Mid | Hard | Expert | $\overline{\text { All }}$ |
| GPT-4 | Standard | 82.9 | - | 73.5 | - | 68.5 | 39.3 | 19.5 | 1.7 | 37.4 |
|  | Parsel | 85.0 | +2.1 | ![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-05.jpg?height=43&width=134&top_left_y=502&top_left_x=993) | - | - | - | - | - | - |
|  | CodeT | 90.9 | +8.0 | 77.0 | +3.5 | 76.4 | 51.8 | 21.8 | 3.4 | 44.0 |
|  | Reflexio | 91.0 | +8.1 | 77.1 | +3.6 | 71.3 | 41.1 | 19.5 | 2.5 | 38.6 |
|  | MetaGPT | $\overline{85.9}$ | +3.0 | ![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-05.jpg?height=43&width=134&top_left_y=613&top_left_x=993) | - | - | - | - | - | - |
|  | FUNCODER | 94.5 | +11.6 | 79.5 | +6.0 | 83.1 | 58.0 | 26.4 | 3.4 | 48.6 |
| GPT-3.5 | Stanc | 61 | ![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-05.jpg?height=50&width=92&top_left_y=707&top_left_x=875) | 72.0 | - | 44.4 | 15.2 | 4.6 | 0.0 | 20.2 |
|  | Code | 81.1 | +12.8 | 76.0 | +4.0 | 50.6 | 16.1 | 8.0 | 0.0 | 23.2 |
|  | Reflexion | 69.5 | +1.2 | 72.5 | +0.5 | 44.4 | 17.0 | 5.7 | 0.0 | 20.6 |
|  | LDB | 82.9 | +14.6 | 76.0 | +4.0 | - | - | - | - | - |
|  | FUNCODER | $\overline{\mathbf{8 5 . 4}}$ | +17.1 | 78.5 | +6.5 | 62.4 | 29.5 | 11.6 | 0.0 | 31.4 |

Baselines We compare FUNCODER with standard prompting (Brown et al., 2020), two-stage decomposition method Parsel (Zelikman et al. 2023), self-testing method CodeT (Chen et al., 2023a), self-improvement methods Reflexion and LDB (Shinn et al., 2023; Zhong et al., 2024), and multiagent developing framework MetaGPT (Hong et al. 2024). We implement Standard prompting with a 1-shot demonstration. CodeT samples 11 solutions with standard prompting and evaluates them on model-generated tests. The results for Reflexion are reproduced from the original code.

Implementation Details FUNCODER uses a 2-shot prompt in the divide stage and 1-shot for conquering sub-functions. The number of sampled implementations in the functional consensus is set to 11 for code generation tasks. For further implementation details, please refer to Appendix A. 1

### 3.1.2 Results

Table 1 shows the code generation performance on advanced proprietary models, GPT-3.5 (Ouyang et al., 2022) and GPT-4 (OpenAI, 2023). For basic programming questions, HumanEval and MBPP, FUNCODER surpass previous SOTA methods by $+3.3 \%$ in Pass @ 1 and reduce the error rate by $18.6 \%$. Furthermore, FUNCODER demonstrates a substantial improvement on competition-level problems, outperforming others by $10.4 \%$ in GPT-4 and $35.3 \%$ with GPT-3.5. We observe that FUNCODER can enhance LLM's capability of solving more complex programming tasks, with an average accuracy improvement of $82.3 \%$ over the baseline on the Mid and Hard subsets of xCodeEval. Expert level programs, however, still remain a colossal challenge for even the most cutting-edge LLMs.

Table 2: Code generation performance with open-source models on HumanEval.

| Model | Category | Param | Standard | CodeT | Reflexion | FUNCODER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Llama3 | Text/Chat | 8B | 61.6 | 68.9 | 59.1 | $\mathbf{7 9 . 9}(+11.0)$ |
| StableCode | Code | 3B | 61.0 | 75.0 | 61.6 | $\mathbf{8 1 . 0}(+6.0)$ |
| CodeLlama | Code | 34B | 43.9 | 55.5 | 41.5 | $\mathbf{6 6 . 5}(+11.0)$ |

Evaluation is also performed over community LLMs, Llama3 (Meta AI, 2024), StableCode (Pinnaparaju et al. 2024), and CodeLlama (Rozière et al. 2023) with results in Table 2, 10. FUNCODER consistently boosts the performance of smaller models in code generation, with an averaged improvement of $+38.0 \%$ compared to standard prompting, and outperforms the previous best method CodeT by $+14.6 \%$ on HumanEval. Experiment results demonstrate that our method archives state-of-the-art performance on various models, ranging from basic programming to competition contests.

### 3.1.3 Analysis

FUNCODER Democratize to Smaller LLMs Limited by the LLM capabilities, the application of selfimprovement or multi-agent methods on smaller models is without ease. By keeping decomposition
(a) Preliminary Study on Self-testing

![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-06.jpg?height=350&width=894&top_left_y=272&top_left_x=366)

(b) Effectiveness of Ranking Strategy

![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-06.jpg?height=342&width=501&top_left_y=279&top_left_x=1254)

Figure 3: (a) Preliminary study on self-testing, the programs are evaluated using unit-tests generated by LLMs. (b) The effectiveness of different ranking strategies. We compute the Pass @ $\mathrm{k}$ over top-k programs ranked by functional consensus, self-test, and random on 11 candidates. (higher is better)

and composition within the code generation process, our approach exhibits better generalization. As shown in Table 1, 2, with FUNCoDER, Llama3 ${ }_{8 b}$ and StableCode $3 b$ achieve around $1.18 \times$ performance to standard GPT-3.5, and are closely aligned with GPT-4 by about $97 \%$ on HumanEval.

Preliminary Study on Self-Testing Method We conduct a preliminary study targeting the self-testing method on HumanEval, results are shown in Figure 3 a with further details in Appendix A.5. We first verify whether model-generated programs can also pass model-generated self-tests: (a) If a program passes self-tests, most from GPT-3.5 would also work on system tests, as much as $19.5 \% / 64 \% \approx 30.5 \%$ programs from StableCode are rejected, indicating that smaller models like StableCode may not effectively self-test and detect program errors on its own. (b) In the event of failed self-tests, a large portion of failures are attributed to issues in self-tests instead of the programs, on both GPT-3.5 and StableCode. These phenomena indicate that self-testing methods have limitations in generating correct and reliable unit tests. As a result, we design functional consensus to not require any assertion, but perform mutual verification between solutions instead, as opposed to self-testing.

Effectiveness of Functional Consensus Functional consensus or self-testing may be viewed as ranking algorithms for selecting functions. To measure ranking effectiveness, we conduct an analysis on HumanEval with GPT-3.5. For each problem, 11 candidates are ranked with 3 strategies: consensus, self-test, and random shuffle (as a baseline). Effectiveness is measured via Pass @k, i.e. if any of the top-k ranked programs pass the system test. Figure 3 b shows that functional consensus achieves $94.7 \%$ upper bound (Pass @ 11) performance by selecting a single function (Pass@ 1), and is close to that of self-test on Pass @4. This clearly demonstrates that functional consensus can effectively evaluate correctness and pick the most promising implementation on the first attempt.

Table 3: Ablation study of FUNCODER on HumanEval with GPT-3.5. The setting in our main experiment is highlighted in bold. Tokens are calculated as the sum of prompts and completions.

| Setting | Divide | Conquer | Ranking | Pass@1 | Avg. Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard | $\boldsymbol{x}$ | $\boldsymbol{x}$ | $\boldsymbol{x}$ | 68.3 | $\mathbf{8 8 6 . 7}$ |
| One-pass | $\boldsymbol{\checkmark}$ | $\boldsymbol{x}$ | $\boldsymbol{x}$ | $72.6(+4.3)$ | 1233.7 |
| Two-pass | $\boldsymbol{\checkmark}$ | $\boldsymbol{\checkmark}$ | $\boldsymbol{x}$ | $78.7(+10.4)$ | 3343.2 |
| Two-pass + ST@11 | $\boldsymbol{\checkmark}$ | $\boldsymbol{\checkmark}$ | Self-Test@11 | $80.5(+12.2)$ | 5408.3 |
| FUNCODER @5 | $\boldsymbol{\checkmark}$ | $\boldsymbol{\checkmark}$ | Consensus@ 5 | $83.5(+15.2)$ | 4040.8 |
| FUNCODER@11 | $\boldsymbol{\checkmark}$ | $\boldsymbol{\checkmark}$ | Consensus@11 | $\mathbf{8 5 . 4}(+\mathbf{1 7 . 1})$ | 5402.0 |

[^1]Table 4: Experimental results on MATH, a competition-level mathematical reasoning benchmark. Best results are in bold. Text-based reasoning methods are denoted with ${ }^{\dagger}$, while others use programaided reasoning. We report both overall results and results in seven subjects: Prealgebra, Algebra, Number Theory, Counting \& Probability, Geometry, Intermediate Algebra, and Precalculus.

| Model | Method | Prealg. | Alg. | $N T$ | Prob. | Geo. | InterAlg. | Precalc. | Overall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-4 | Standard $^{\dagger}$ | 81.7 | 82.7 | 71.1 | 72.3 | 59.5 | 46.7 | 47.3 | 68.2 |
|  | $\mathrm{CoT}^{\dagger}$ | 84.1 | 87.1 | 62.2 | 68.1 | 45.2 | 48.9 | 54.5 | 68.6 |
|  | PoT | 79.3 | 80.6 | 75.6 | 72.3 | 50.0 | 47.8 | 58.2 | 68.2 |
|  | Self-Refine | 82.9 | 82.0 | 77.8 | 76.6 | 54.8 | 55.6 | 63.6 | 72.2 |
|  | CR | 86.6 | 86.3 | 88.7 | 71.1 | 53.7 | 51.5 | 51.8 | 72.2 |
|  | FUNCODER | 89.0 | 92.8 | 82.2 | 83.0 | 59.5 | 63.3 | 56.4 | 78.2 |
| GPT-3.5 | Standard $^{\dagger}$ | 62.2 | 37.4 | 20.0 | 29.8 | 31.0 | 24.4 | 21.8 | 34.6 |
|  | $\mathrm{CoT}^{\dagger}$ | 59.8 | 51.1 | 28.9 | 29.8 | 28.6 | 26.7 | 30.9 | 40.0 |
|  | PoT | 68.3 | 50.4 | 33.3 | 48.9 | 21.4 | 18.2 | 29.1 | 41.0 |
|  | Self-Refine | 74.4 | 49.6 | 48.9 | 57.4 | 28.6 | 35.6 | 36.4 | 48.6 |
|  | FUNCODER | 76.8 | 61.2 | 55.6 | 59.6 | 34.1 | 36.0 | 41.8 | 54.0 |
| Llama3 $_{8 b}$ | $\mathrm{CoT}^{\dagger}$ | 56.1 | 47.5 | 31.1 | 34.0 | 40.5 | 14.4 | 38.2 | 38.6 |
|  | PoT | 67.1 | 32.4 | 24.4 | 34.0 | 16.7 | 21.1 | 18.2 | 32.6 |
|  | FUNCODER | 67.9 | 45.7 | 51.1 | 53.2 | 19.0 | 37.8 | 30.9 | 45.0 |
| StableCode $_{3 b}$ | PoT | 20.7 | 14.4 | 17.8 | 25.5 | 4.8 | 8.9 | 9.1 | 14.4 |
|  | FUNCODER | 46.3 | 30.2 | 20.0 | 29.8 | 4.8 | 20.0 | 18.2 | 26.6 |
| CodeLlama $_{34 b}$ | PoT | 35.5 | 26.1 | 15.0 | 16.7 | 0.0 | 5.5 | 33.3 | 15.2 |
|  | FUNCODER | 44.8 | 46.1 | 37.8 | 34.1 | 13.6 | 24.6 | 37.5 | 24.4 |

### 3.2 Mathematical Reasoning

Code can be viewed as a tool for augmenting the reasoning capabilities of LLMs (Chen et al., 2023b). Alternative to text-based reasoning like Chain-of-Thought (Wei et al., 2022), programs can offer unique advantages in terms of iteration and calculations. To test the generalizability of FUNCODER beyond algorithm challenges, we conduct an experiment on MATH (Hendrycks et al., 2021b), a competition-level mathematical reasoning benchmark.

### 3.2.1 Experiment Setup

Benchmark The experiment is conducted on a subset of the MATH test set, including 500 randomly sampled problems that can be classified into 7 disjoint subjects or 5 difficulty levels. It can be noticed that labels in MATH are formatted in $\mathrm{LT}_{\mathrm{E}} \mathrm{X}$, rendering exact-match verdicts impractical. We, therefore, follow previous work (Zhang et al. 2024) and adopt GPT-4 to determine the correspondence between predictions and labels, with further details provided in Appendix A.4.

Baselines We compare FUnCoDER with the text-based baselines: Standard Prompting and Chainof-Thought (Wei et al., 2022), and program-aided baselines: Program-of-Thought (Chen et al. 2023b), Self-Refine (Madaan et al., 2023), Cumulative Reasoning (Zhang et al., 2024). The results of Cumulative reasoning are reported in the original paper. Standard prompting and chain-of-thought reasoning use 7-shot demonstrations constructed from the train set. Program-of-Thought and SelfRefine prompt the model with 1-shot demonstration to generate a solution() function that solves the problem. Additionally, self-refine iteratively refines programs based on runtime feedback. All baseline methods are run with self-consistency (Wang et al., 2023) at 5.

Implementation Details FUNCODER adopts a program-aided reasoning setting that writes a solution() function and obtains the final prediction by running this program. The number of sampled implementations $|F|$ in functional consensus is set to 5 to match baseline methods.

### 3.2.2 Results

The experimental results on MATH are shown in Table 4. It shows that program-aided reasoning generally outperforms text-based reasoning. With GPT-4 as the backbone, FUNCODER outperforms the strongest baseline Cumulative Reasoning (Zhang et al. 2024) by ( 6.0 / 8.3\%) and surpasses the
vanilla program-aided baseline PoT (Chen et al. 2023b) by (10.0 / 14.7\%). When using GPT-3.5turbo as the backbone, FUNCODER exceeds the strongest baseline by $(6.2 / 11.1 \%)$ and outperforms PoT by as much as (13.0 / 31.7\%), which indicates that our approach has a strong advantage over both text-based reasoning and other program-aided reasoning methods.

On open-source models, FUNCODER with Llama3 outperforms PoT by (12.4 / 38.0\%). It has even reached competitive performance against the state-of-the-art method based on GPT-3.5 ( 45.0 v.s. 48.6). When employing StableCode and CodeLLaMA as the backbone, our approach achieves significant improvements by ( 12.2 / 84.7\%) and ( 9.2 / 60.5\%), respectively. This improvement demonstrates that our approach can significantly boost smaller LLMs, democratizing the complex reasoning capabilities of open-source LLMs through programming.

### 3.2.3 Analysis

## Fun Coder Can Handle Harder Questions

 Figure 4 compares between CoT, PoT, and FUnCODER across varying difficulty levels. It illustrates that CoT performs comparatively well on the easiest questions, but suffers from a steep decline in performance as difficulty increases. This suggests that text-based reasoning is inadequate for tackling challenging mathematical reasoning problems. The same situation is also observed in PoT. In contrast, our method consistently demonstrates high performance even on challenging problems, particularly excelling on![](https://cdn.mathpix.com/cropped/2024_06_04_94bfc0883b286484df06g-08.jpg?height=325&width=699&top_left_y=737&top_left_x=1057)

Figure 4: Average accuracy in each level with the chat model (GPT-3.5) and the code model (StableCode ${ }_{3 b}$ ) on the MATH benchmark. level 5 difficulty with nearly double the performance compared to PoT and CoT. This reflects that our method, with divide-and-conquer applied, can effectively cope with complex problems.

Decomposed Functions are Domain-Specific We hypothesize that questions from the same subject require similar knowledge reserves, which should be reflected in the functionality of the sub-functions. To verify this hypothesis, we statisticize the common sub-functions of FUNCODER in each MATH subject, as shown in Table 5. It is apparent that different subjects require different abilities, each with its own set of sub-functions closely associated with the domain knowledge. In addition, these common sub-functions are fundamentally basic and straightforward. As exemplified in Appendix B.2. our method is able to leverage and combine these basic sub-functions to achieve more complex goals, thereby reducing the complexity of reasoning and enhancing performance.

Table 5: Top-3 most commonly used functions in each subject of MATH, listed in descending order.

| Subject | Functions |
| :--- | :--- |
| Prealgebra | is_prime / factorial / gcd |
| Algebra | find_roots / is_perfect_square / find_domain |
| Number Theory | get_divisors / mod_inverse / gcd |
| Counting \& Probability | factorial / combinations / binomial_coefficient |
| Geometry | distance / simplify_fraction / calculate_triangle_area |
| Intermediate Algebra | find_roots / evaluate_polynomial / lagrange_interpolation |
| Precalculus | cross_product / fraction_from_angle / dot |

## 4 Related Work

Large Language Model for Code Code pre-training has received widespread attention, with early models based on small language models (SLM) (Feng et al., 2020, Lu et al., 2021, Wang et al. 2021). In recent years, with the development of large-scale pre-training techniques, code LLM has emerged, showing remarkable performance in downstream code tasks (Chen et al., 2021; Nijkamp et al., 2023, Li et al., 2022, Rozière et al., 2023, Li et al., 2023b, Guo et al., 2024). Tasks between code and natural language (NL) can be generally divided into three major categories: NL2Code tasks such as
code generation (Austin et al., 2021; Chen et al., 2021; Hendrycks et al., 2021a; Khan et al., 2023) and code search (Husain et al. 2019a); Code2Code tasks including code completion (Lu et al., 2021, Zhang et al., 2023a, Liu et al., 2024), code translation (Ahmad et al., 2023, Zhu et al., 2022; Yan et al., 2023), and test generation (Siddiq et al. 2023, Schäfer et al. 2024); Code2NL tasks like code summarization (Husain et al. 2019b; Jin et al. 2023). This paper focuses on code generation tasks, ranging from basic to competition level.

Code Refinement and Self-Testing Code doesn't always run as expected; it could contain syntax errors, dead loops, or bugs. It's essential to debug and refine the code to ensure better quality. CodeT (Chen et al., 2023a) generates unit-tests to score the implementation. Self-improvement methods (Madaan et al. 2023; Shinn et al., 2023, Chen et al., 2024, Zhong et al., 2024) design closed-loop procedures that repeatedly refine the code based on the feedback. Like real-life software development processes, multi-agent frameworks (Hong et al., 2024, Qian et al., 2023) construct specific LLM roles, Tester or $Q A$ to generate tests. These studies adopt a shared paradigm wherein self-tests are generated through LLMs. However, Olausson et al. (2024) points out the challenge that LLMs have certain shortcomings in self-repairing their code. This paper avoids these shortcomings by proposing functional consensus as a reliable method of evaluation.

Program-Aided Reasoning and Agents Aside from code generation tasks, the program can be a tool that augments LLM to solve complex reasoning questions or interact with external environments. Program-of-Thought (Chen et al. 2023b) and PAL (Gao et al. 2023) prompt the model to generate a program that solves mathematical or symbolic problems. MathPrompter (Imani et al. 2023) and Chain-of-Code (Li et al. 2023a) fuse the text-based chain-of-thought with code-based program-of-thought prompting to complement each other in mathematical reasoning. Cumulative Reasoning (Zhang et al. 2024) conducts bottom-up reasoning to derive the final answer progressively. Numerous work (Sun et al. 2023; Wang et al. 2024, Yang et al. 2024) also use code as an intermediate component to bridge LLM agents with external environments.

Decompose for Complex Problems Several recent works employ decomposition to reduce the complexity of hard problems. Least-to-Most (Zhou et al. 2023) adopts a two-stage approach, which first decomposes complex problems, and then solves each sub-problem individually to tackle complex reasoning tasks. Successive Prompting (Dua et al., 2022) adopts a dynamic decomposition, iteratively breaking down problems and addressing sub-problems. Tree-of-Thought (Yao et al., 2023) breaks down complex problems into state spaces and uses tree search to solve them. Parsel (Zelikman et al. 2023) introduces decomposition to code generation tasks, taking a three-stage to break down requirements into draft and intermediate parsel programs. RepoCoder (Zhang et al. 2023b) performs a retrieval in repositories to complete unfinished code one by one. Unlike these methods, FUNCODER recursively decomposes problems into a tree structure, hence gradually reduces its complexity.

## 5 Discussion

Limitations Our approach unleashes the potential power of functions in programming, which is advantageous on well-defined problems such as competitive programming, or program-augmented reasoning tasks. These scenarios do not however represent all use cases, such as open-ended problems or casual software development. Nevertheless, we believe that the idea of divide-and-conquer and sub-modular consensus utilized by FUNCODER can be extended to a wider range of problems, and we consider this as a future exploration.

Broader Impact While code generation is increasingly utilized in software development, Large Language Models (LLMs) are still prone to generating toxic, vulnerable, or malicious code. Such programs pose risks and should be used or executed with extra caution.

## 6 Conclusion

In this paper, we presented FUNCODER, a novel code generation framework that integrates the divideand-conquer strategy with functional consensus to address complex requirements. FUNCODER had demonstrated superior performance compared to state-of-the-art methods on various benchmarks and models. Our findings highlighted the effectiveness of dynamic decomposition and functional consensus in writing complex code, which suggests that FUNCODER may have the potential to empower further improvements in code generation and other fields.

## References

Wasi Uddin Ahmad, Md Golam Rahman Tushar, Saikat Chakraborty, and Kai-Wei Chang. AVATAR: A parallel corpus for Java-python program translation. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023, pp. 2268-2281, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.143. URL https://aclanthology.org/2023.findings-acl.143

Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. ArXiv preprint, abs/2108.07732, 2021. URL https://arxiv.org/abs/2108.07732

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html

Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q. Feldman, Arjun Guha, Michael Greenberg, and Abhinav Jangda. Multipl-e: A scalable and polyglot approach to benchmarking neural code generation. IEEE Trans. Software Eng., 49(7):3675-3691, 2023. doi: 10.1109/TSE.2023.3267446. URL https: //doi.org/10.1109/TSE.2023.3267446

Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. Codet: Code generation with generated tests. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023a. URL https://openreview.net/ forum?id=ktrw68Cmu9c

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel HerbertVoss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. ArXiv preprint, abs/2107.03374, 2021. URL https: //arxiv.org/abs/2107.03374

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023b. ISSN 2835-8856. URL https://openreview.net/forum?id=YfZ4ZPt8zd

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=KuPixIqPiq

Ole-Johan Dahl, Edsger W. Dijkstra, and Charles Antony Richard Hoare. Structured programming, volume 8 of A.P.I.C. Studies in data processing. Academic Press, 1972. ISBN 978-0-12-200550-3.

Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. Successive prompting for decomposing complex questions. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 1251-1265, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.81. URL https://aclanthology.org/2022.emnlp-main. 81

EbTech. How to Interpret Contest Ratings - Codeforces, 2024. URL https://codeforces.com/blog/ entry/68288

Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. CodeBERT: A pre-trained model for programming and natural languages. In Trevor Cohn, Yulan He, and Yang Liu (eds.), Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1536-1547, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/ 2020.findings-emnlp.139. URL https://aclanthology.org/2020.findings-emnlp. 139

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: program-aided language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 10764-10799. PMLR, 2023. URL https://proceedings.mlr.press/v202/gao23f.html

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model meets programming - the rise of code intelligence. ArXiv preprint, abs/2401.14196, 2024. URL https: //arxiv.org/abs/2401.14196

P.R. Halmos. Naive Set Theory. Undergraduate Texts in Mathematics. Springer New York, 1998. ISBN 9780387900926. URL https://books.google.com.hk/books?id=x6cZBQ9qtgoC

Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. Measuring coding challenge competence with APPS. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021a. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/ hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract-round2.html

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021b. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=VtmBAGCN70

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search. ArXiv preprint, abs/1909.09436, 2019a. URL https://arxiv.org/abs/1909.09436

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search. ArXiv preprint, abs/1909.09436, 2019b. URL https://arxiv.org/abs/1909.09436

Shima Imani, Liang Du, and Harsh Shrivastava. MathPrompter: Mathematical reasoning using large language models. In Sunayana Sitaram, Beata Beigman Klebanov, and Jason D Williams (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track), pp. 37-42, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-industry.4. URLhttps://aclanthology.org/2023.acl-industry. 4

Xue Jiang, Yihong Dong, Lecheng Wang, Qiwei Shang, and Ge Li. Self-planning code generation with large language model. ArXiv preprint, abs/2303.06689, 2023. URL https://arxiv.org/abs/2303.06689

Xin Jin, Jonathan Larson, Weiwei Yang, and Zhiqiang Lin. Binary code summarization: Benchmarking chatgpt/gpt-4 and other large language models. ArXiv preprint, abs/2312.09601, 2023. URL https: //arxiv.org/abs/2312.09601

Mohammad Abdullah Matin Khan, M. Saiful Bari, Xuan Long Do, Weishi Wang, Md. Rizwan Parvez, and Shafiq R. Joty. xcodeeval: A large scale multilingual multitask benchmark for code understanding, generation, translation and retrieval. ArXiv preprint, abs/2303.03004, 2023. URL https://arxiv.org/abs/2303. 03004 .

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li Fei-Fei, Fei Xia, and Brian Ichter. Chain of code: Reasoning with a language model-augmented code emulator. ArXiv preprint, abs/2312.04474, 2023a. URL https://arxiv.org/abs/2312.04474

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy V, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Moustafa-Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder: may the source be with you! ArXiv preprint, abs/2305.06161, 2023b. URL https://arxiv.org/abs/2305.06161

Yujia Li, David H. Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d'Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. ArXiv preprint, abs/2203.07814, 2022. URLhttps://arxiv.org/abs/2203.07814

Tianyang Liu, Canwen Xu, and Julian McAuley. Repobench: Benchmarking repository-level code autocompletion systems. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= $\mathrm{pPjZIOuQuF}$.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian J. McAuley, Han Hu, Torsten Scholak, Sébastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, and et al. Starcoder 2 and the stack v2: The next generation. ArXiv preprint, abs/2402.19173, 2024. URL https://arxiv.org/abs/2402.19173

Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin B. Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan, Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. Codexglue: A machine learning benchmark dataset for code understanding and generation. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ c16a5320fa475530d9583c34fd356ef5-Abstract-round1.html

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with selffeedback. In Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023, 2023. URL https://openreview.net/forum?id=S37h0erQLB

Meta AI. Meta Llama 3 - homepage, 2024. URL https://llama.meta.com/llama3/

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=iaYcJKpY2B_

Theo X. Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, and Armando Solar-Lezama. Is self-repair a silver bullet for code generation? In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna Austria, May 7-11, 2024. OpenReview.net, 2024. URL https:// openreview.net/forum?id=yOGJXRungR

OpenAI. GPT-4 technical report. ArXiv preprint, abs/2303.08774, 2023. URL https://arxiv.org/abs/ 2303.08774

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In NeurIPS, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ b1efde53be364a73914f58805a001731-Abstract-Conference.html

Nikhil Pinnaparaju, Reshinth Adithyan, Duy Phung, Jonathan Tow, James Baicoianu, Ashish Datta, Maksym Zhuravinskyi, Dakota Mahan, Marco Bellagente, Carlos Riquelme, and Nathan Cooper. Stable code technical report. ArXiv preprint, abs/2404.01226, 2024. URL https://arxiv.org/abs/2404.01226

Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. ArXiv preprint, abs/2307.07924, 2023. URL https://arxiv.org/abs/2307.07924

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. Code llama: Open foundation models for code. ArXiv preprint, abs/2308.12950, 2023. URL https://arxiv.org/abs/2308. 12950

Max Schäfer, Sarah Nadi, Aryaz Eghbali, and Frank Tip. An empirical evaluation of using large language models for automated unit test generation. IEEE Trans. Software Eng., 50(1):85-105, 2024. doi: 10.1109/ TSE.2023.3334955. URLhttps://doi.org/10.1109/TSE.2023.3334955

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023, 2023. URL https://openreview.net/forum?id=vAElhFcKW6

Mohammed Latif Siddiq, Joanna C. S. Santos, Ridwanul Hasan Tanvir, Noshin Ulfat, Fahmid Al Rifat, and Vinicius Carvalho Lopes. Exploring the effectiveness of large language models in generating unit tests. ArXiv preprint, abs/2305.00418, 2023. URL https://arxiv.org/abs/2305.00418

Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. Adaplanner: Adaptive planning from feedback with language models. In Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023, 2023. URL https://openreview.net/forum?id=rnKgbKmelt.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. ArXiv preprint, abs/2307.09288, 2023. URL https://arxiv.org/abs/2307.09288

Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better LLM agents. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024. URL https://openreview.net/forum?id=8oJyuXfrPv

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw

Yue Wang, Weishi Wang, Shafiq Joty, and Steven C.H. Hoi. CodeT5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 8696-8708, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.685. URL https: //aclanthology.org/2021.emnlp-main. 685

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_ files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. Huggingface's transformers: State-of-the-art natural language processing. ArXiv preprint, abs/1910.03771, 2019. URL https://arxiv.org/abs/1910. 03771

Weixiang Yan, Yuchen Tian, Yunzhe Li, Qian Chen, and Wen Wang. CodeTransOcean: A comprehensive multilingual benchmark for code translation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 5067-5089, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.337. URL https://aclanthology. org/2023.findings-emnlp.337

Ke Yang, Jiateng Liu, John Wu, Chaoqi Yang, Yi Fung, Sha Li, Zixuan Huang, Xu Cao, Xingyao Wang, Heng Ji, and ChengXiang Zhai. If LLM is the wizard, then code is the wand: A survey on how code empowers large language models to serve as intelligent agents. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024. URL https://openreview.net/forum?id=8dmNOD9hbq

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik R Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023, 2023. URL https://openreview.net/forum? id $=5 \mathrm{Xc1ecx01h}$

Eric Zelikman, Qian Huang, Gabriel Poesia, Noah Goodman, and Nick Haber. Parsel: Algorithmic reasoning with language models by composing decompositions. In Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023, 2023. URL https://openreview.net/forum?id=qd9qcbVAwQ

Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. RepoCoder: Repository-level code completion through iterative retrieval and generation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 2471-2484, Singapore, 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.151. URL https://aclanthology.org/2023.emnlp-main. 151

Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. RepoCoder: Repository-level code completion through iterative retrieval and generation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 2471-2484, Singapore, 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.151. URL https://aclanthology.org/2023.emnlp-main. 151

Yifan Zhang, Jingqin Yang, Yang Yuan, and Andrew Chi-Chih Yao. Cumulative reasoning with large language models. In ICLR 2024 Workshop on Bridging the Gap Between Practice and Theory in Deep Learning, 2024. URL https://openreview.net/forum?id=XAAYyRxTlQ

Lily Zhong, Zilong Wang, and Jingbo Shang. LDB: A large language model debugger via verifying runtime execution step-by-step. ArXiv preprint, abs/2402.16906, 2024. URL https://arxiv.org/abs/2402 16906

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum? id=WZH7099tgfM

Ming Zhu, Karthik Suresh, and Chandan K. Reddy. Multilingual code snippets training for program translation. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pp. 11783-11790. AAAI Press, 2022. URL https://ojs.aaai.org/index.php/AAAI/article/view/21434
</end of paper 2>


<paper 3>
# If LLM Is the Wizard, Then Code Is the Wand: A Survey on How 

 Code Empowers Large Language Models to Serve as Intelligent AgentsKe Yang* Jiateng Liu*, John Wu, Chaoqi Yang, Yi R. Fung, Sha Li,<br>Zixuan Huang, Xu Cao, Xingyao Wang, Yiquan Wang, Heng Ji, Chengxiang Zhai<br>University of Illinois Urbana-Champaign<br>\{key4, jiateng5, johnwu3, chaoqiy2, yifung2, shal2,<br>zixuan3, xucao2, xingyao6, yiquan2, hengji, czhai\}@illinois.edu


#### Abstract

The prominent large language models (LLMs) of today differ from past language models not only in size, but also in the fact that they are trained on a combination of natural language and formal language (code). As a medium between humans and computers, code translates high-level goals into executable steps, featuring standard syntax, logical consistency, abstraction, and modularity. In this survey, we present an overview of the various benefits of integrating code into LLMs' training data. Specifically, beyond enhancing LLMs in code generation, we observe that these unique properties of code help i) unlock the reasoning ability of LLMs, enabling their applications to a range of more complex natural language tasks; ii) steer LLMs to produce structured and precise intermediate steps, which can then be connected to external execution ends through function calls; and iii) take advantage of code compilation and execution environment, which also provides diverse feedback for model improvement. In addition, we trace how these profound capabilities of LLMs, brought by code, have led to their emergence as intelligent agents (IAs) in situations where the ability to understand instructions, decompose goals, plan and execute actions, and refine from feedback are crucial to their success on downstream tasks. Finally, we present several key challenges and future directions of empowering LLMs and IAs with code.


## 1 Introduction

Code has become an integral component in the training data of large language models (LLMs), including well-known models such as Llama2, GPT3.5 series and GPT-4 (Touvron et al., 2023; Ye et al., 2023a; OpenAI, 2023). Training LLMs on code has gained popularity not only because the acquired programming skills enable commercial applications, such as Github Copilot ${ }^{1}$, but also because it[^0]

improves the models' previously lacking reasoning abilities (Liang et al., 2023b). Consequently, LLMs rapidly emerge as a primary decision-making hub for intelligent agents (IAs) (Zhao et al., 2023), demonstrating an exponential growth in capabilities from code training and the advancement of tool learning (Qin et al., 2023). These LLM-based IAs are poised to handle a wider range of more complex tasks, including downstream applications in multi-agent environment simulation (Wu et al., 2023c) and AI for science (Boiko et al., 2023).

As depicted in Figure 1, this survey aims to explain the widespread adoption of code-specific training in the general LLM training paradigm and how code enhances LLMs to act as IAs. Unlike previous code-LLM surveys that concentrate on either evaluating and comparing code generation abilities (Zan et al., 2023; Xu et al., 2022), or listing IA tasks (Wang et al., 2023d; Xi et al., 2023; Zhao et al., 2023) in IA surveys, we aim to provide a comprehensive understanding of how code assists LLMs and where code benefits LLMs as IAs, based on the taxonomy of relevant papers (see Figure 2).

We first provide our definition of code and present typical methods for LLM code training (\$2). Compared to natural language (refer to the case study in A.1), code is more structured, featuring logical, step-by-step executable processes derived from procedural programming, as well as explicitly defined, modularized functions, which compose graphically representable abstractions. Additionally, code is typically accompanied by a selfcontained compilation and execution environment. With insights from these characteristics of code, our comprehensive literature review reveals that integrating code into LLM training i) enhances their programming and reasoning capabilities ( $\$ 3$ ); ii) enables the models to directly generate executable, fine-grained steps during decision-making, thereby facilitating their scalability in incorporating various tool modules through function calls (\$4); and iii)

![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-02.jpg?height=726&width=1600&top_left_y=228&top_left_x=228)

Figure 1: An illustration of how code empowers large language models (LLMs) and enhances their downstream applications as intelligent agents (IAs). While traditional LLMs excel in conventional natural language tasks like document classification and question answering, further pre-training or fine-tuning LLMs with human-interpretable and machine-executable code serves as an additional power-up - akin to equipping wizards with mana-boosting wands. This significantly boosts their performance as IAs through intricately woven operational steps.

situates the LLMs within a code execution environment, allowing them to receive automated feedback from integrated evaluation modules and selfimprove ( $\$ 5$ ).

In addition, as LLMs are becoming key decisionmakers for IAs in complex real-world tasks, our survey also explores how these advantages facilitate their functioning along this capacity ( $\$ 6$ ), in terms of $i$ ) enhancing IAs' decision-making in perception and planning skills (\$6.1), ii) facilitating their execution through direct action primitive grounding and modular memory organization ( $\$ 6.2$ ), and iii) providing an interactive environment for selfcorrection and self-improvement (\$6.3). Finally, we discuss several open challenges and promising future directions ( $\$ 7$ ).

## 2 Preliminaries

### 2.1 Our Definition of Code

We consider code as any formal language that is both machine-executable and human-interpretable. For instance, human-readable programming languages fall within the scope of our discussion, whereas low-level languages, such as machine language based on binary instructions, are excluded due to their lack of human interpretability. Additionally, pre-defined formal languages, such as function sets employed in WebGPT (Nakano et al., 2021), are included as they can be parsed and executed in a rule-based manner.
LLMs trained with expressions formulated within a defined set of symbols and rules (e.g., pre-defined function sets, mathematical deduction formula, etc.), i.e., formal languages, exhibit advantages akin to those trained with programming languages. Therefore, we expand our definition of code to incorporate these homogeneous training corpora, enhancing the comprehensiveness of this survey to align with current research needs.

### 2.2 LLM Code Training Methods

LLMs undergo code training by following the standard language modeling objective, applied to code corpora. Given that code possesses natural language-like sequential readability, this parallels the approach to instruct LLMs in understanding and generating free-form natural language. Specifically, for an LLM $M_{\Theta}$ with parameters $\Theta$ and a code corpus $T=\left\{t_{1}, \ldots, t_{n}\right\}$, the language modeling loss for optimization is:

$$
L(T)=\sum_{i} \log P\left(t_{i} \mid t_{i-k}, \ldots, t_{i-1} ; \Theta\right)
$$

When employing programming language (e.g., Python, C, etc.) as the corpus (Chen et al., 2021; Li et al., 2022; Nijkamp et al., 2022), training data is typically sourced from publicly accessible code repositories, such as GitHub. This process yields a corpus with a volume comparable to that of natural language pre-training, and thus we call training with such an abundance of code as code pretraining. The training strategy entails either train-

![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-03.jpg?height=1225&width=1587&top_left_y=233&top_left_x=243)

Figure 2: The organization of our paper, with a curated list of the most representative works. The complete work list is provided in Appendix D.

ing code on a pre-trained natural language LLM, as exemplified by Codex (Chen et al., 2021), or training a LLM from scratch with a blend of natural language and code, as demonstrated by CodeLLM (Ma et al., 2023a).

Conversely, when utilizing other pre-defined formal language for training, the objective shifts to acquainting the model with the application of specific functions (Schick et al., 2023), mathematical proof formulas (Wu et al., 2022), SQL (Sun et al., 2023b), and similar constructs. As the dataset for this is smaller compared to the pre-trained natural language corpus, we refer to such training process as code fine-tuning. Researchers apply the language modeling loss to optimize LLMs during this process, similarly.

## 3 Code Pre-Training Boosts LLMs, Performance

The pre-training of LLMs on code, exemplified by OpenAI's GPT Codex (Chen et al., 2021), has broadened the LLMs' scope of tasks beyond nat- ural language. Such models enable diverse applications, including generating code for mathematical theory (Wu et al., 2022), general programming tasks (Chen et al., 2021), and data retrieval (Sun et al., 2023b; Cheng et al., 2023). Code necessitates producing logically coherent, ordered sequences of steps essential for valid execution. Moreover, the executability of each step within code allows for step-by-step logic verification. Leveraging and embedding both these properties of code in pre-training has improved LLM chain-of-thought (CoT) performance across many conventional natural language downstream tasks (Lyu et al., 2023; Zhou et al., 2023a; Fu and Khot, 2022), indicating improved complex reasoning skills. Implicitly learning from code's structured format, code LLMs demonstrate further improved performance on commonsense structured reasoning tasks, such as those related to markup, HTML, and chart understanding (Furuta et al., 2023; Liu et al., 2023a).

In the following sections, our objective is to elucidate why training LLMs on code and employing

![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-04.jpg?height=274&width=443&top_left_y=294&top_left_x=264)

(a) Strengthen LLMs' programming and code evaluation skills ( $\$ 3.1$ ).

![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-04.jpg?height=280&width=371&top_left_y=294&top_left_x=820)

(b) Empower LLMs' complex reasoning, decoupling computation from language understanding ( $\$ 3.2$ ).

![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-04.jpg?height=277&width=391&top_left_y=296&top_left_x=1369)

(c) Enable LLM to better capture structured knowledge and better understand complex multimedia data ( $\$ 3.3$ ).

Figure 3: How code pre-training boosts LLMs' performance.

code-based prompts enhance their performance on complex downstream tasks. Specifically, we highlight three key areas where pre-training on code have benefited LLMs: $i$ ) enhancing programming proficiency in $\S 3.1$, ii) empowering complex reasoning capabilities in $\S 3.2$, and iii) facilitating the capture of structured commonsense knowledge in $\S 3.3$, as shown in Figure 3.

### 3.1 Strengthen LLMs' Programming Skills

LLM as a strong coder. Earlier language models only generate domain-specific programs (Ellis et al., 2019) or restrict to one of the generic programming languages, such as Java or C\# (Alon et al., 2020). Empowered by the increasing number of parameters and computing resources, recent LLM-based code generation models (such as AlphaCode (Li et al., 2022), CodeGen (Nijkamp et al., 2022), SantaCoder (Allal et al., 2023), PolyCoder (Xu et al., 2022)) could master more than 10 languages within the same model and show unprecedented success. A well-known work is CodeX (Chen et al., 2021), with 12 billion parameters that reads the entire GitHub database and is able to solve $72.31 \%$ of challenging Python programming problems created by humans. Recent studies (Zan et al., 2023; Xu et al., 2022; Du et al., 2023; Vaithilingam et al., 2022; Wong et al., 2023; Fan et al., 2023) have provided systematic surveys and evaluations of existing code-LLMs.

With its strong code generation ability, LLMs benefit various applications that rely on code, such as database administration (Zhou et al., 2023b), embedded control (Liang et al., 2023a), game design (Roberts et al.), spreadsheet data analysis (Liu et al., 2023c), and website generation (Calò and De Russis, 2023).

LLM as a state-of-the-art code evaluator. On the other hand, LLMs themselves could be state-of-theart evaluators (i.e., analyze and score) for human or machine-generated codes. Kang et al. (2023a) leverage LLM-based models for code fault localization, while Zhuo (2023) uses GPT-3.5 to evaluate the functional correctness and human preferences of code generation. In addition, Deng et al. (2023a) design a LLM-based penetration testing tool and find that LLMs demonstrate proficiency in using testing tools, interpreting outputs, and proposing subsequent actions. Two recent efforts (Li et al., 2023a; Mohajer et al., 2023) also utilize LLM for examining and analyzing source code without executing it. Furthermore, LLMs are used for automatic bug reproduction in Kang et al. (2023b) and vulnerable software evaluation in Noever (2023).

Multi-LLM collaboration solves complex coding problems. Though code LLMs, like GitHub Copilot, have shown unprecedented success, one LLM agent alone could fail in complicated scenarios requiring multiple steps. Luckily, collaborative coding among several role-specific LLM agents exhibits more accurate and robust performance towards complex tasks. Hong et al. (2023) incorporates human programming workflows as guides to coordinate different agents. Dong et al. (2023b) assigned three roles: analyst, coder, and tester to three distinct "GPT-3.5"s, which surpasses GPT-4 in code generation. Meanwhile, Qian et al. (2023a) designs a chat-powered software development process, assigning more than three roles to separate LLM agents. Other similar methods (Liu et al., 2023g; Talebirad and Nadiri, 2023; Wu et al., 2023b; Jiang et al., 2023) all employ multiple code-LLM agents or different phases of the same agent for code generation, software developments or leveraging generated intermediate codes for other general purpose tasks.

### 3.2 Empower LLMs' Complex Reasoning

Code pre-training improves chain-of-thought performance. Logically, many complex tasks can be divided into smaller easier tasks for solving CoT prompting, where prompt inputs are designed with chains of reasoning, allows the LLM to condition its generation with further steps of reasoning, providing a direct approach to task decomposition (Wei et al., 2023). CoT has seen success in the decomposition of many tasks, such as planning (Huang et al., 2022b) and evidence-based question answering (Dua et al., 2022; Ye et al., 2023b).

While LLM CoT ability was originally mainly attributed to dramatically increased model sizes (Wei et al., 2022b), recent evidence compiled by Fu and Khot (2022) suggests that much of the performance improvements from CoT stems from its pre-training on code. For instance, when comparing different versions of GPT-3 (i.e., v1 vs. v5), LLMs not trained on code, such as GPT-3's textdavinci-001, see a small but substantial accuracy improvement of $6.4 \%$ to $12.4 \%$ with CoT on the mathematical reasoning task GSM8k (Cobbe et al., 2021). In contrast, LLMs pre-trained on code, such as GPT-3's text-davinci-002 and Codex (Chen et al., 2021), see a dramatic performance improvement arising from CoT, with a remarkable accuracy increase of $15.6 \%$ to $46.9 \%$ and $19.7 \%$ to $63.1 \%$ respectively. Supporting this hypothesis proposed by Fu and Khot (2022), Ma et al. (2023a) show that pre-training on code in small-sized LLMs (2.6B) (Zeng et al., 2021) enhances performance when using CoT, and even more remarkably that smaller code-pretrained LLMs outperform their larger noncode counterparts across many different tasks. Furthermore, their study indicates that incorporating a greater volume of code during the initial phases of LLM training significantly enhances its efficacy in reasoning tasks. Nevertheless, tempering expectations, it is possible that the discrepancy in CoT performance between LLMs with and without code pre-training diminishes as the size of the models decreases, as the accuracy gap between the small LLMs in Ma et al. (2023a) was less than $3 \%$ when evaluating CoT. Notably, both Fu and Khot (2022) and Ma et al. (2023a) show that pre-training on code improves LLM performance in both standard and CoT prompting scenarios across downstream tasks.

Program-of-thought outperforms chain-ofthought. Furthermore, in comparison to vanilla
CoT methods, LLMs that first translate and decompose a natural language task into code (Chen et al., 2023b; Gao et al., 2023), typically termed program-of-thought (PoT) prompting or program-aided language model, see sizable gains in tasks that require disambiguation in both language and explicit longitudinal structure. This approach is especially effective in complex areas such as theoretical mathematics (Polu and Sutskever, 2020), undergraduate mathematics (Drori et al., 2022), and question answering with data retrieval (Sun et al., 2023b; Cheng et al., 2023).

PoT enhances performance due to the precision and verifiability inherent in code as a machineexecutable language. Within task decomposition, it is not uncommon for the LLM to hallucinate incorrect subtasks and questions through $\mathrm{CoT}$ (Ji et al., 2023). PoT implementations from Chen et al. (2023b), Gao et al. (2023), and Ye et al. (2023b) show that by directly executing code and verifying outcomes post translation by LLMs, one can effectively mitigate the effects of incorrect reasoning in CoT. This is because the reasoning process must adhere to the logic and constraints explicitly specified by the program, thereby ensuring a more accurate and reliable outcome.

However, such improvements seen in the usage of code are not limited to purely executable coding languages such as Python or SQL, nor are they limited to tasks that are specifically rigid in structure such as mathematics (Drori et al., 2022) and data retrieval (Rajkumar et al., 2022). Enhancements also extend to the realm where even translating into pseudo-code to decompose a task can improve zero-shot performance (Lei and Deng, 2023) in word problems containing numbers, and general reasoning tasks such as StrategyQA (Geva et al., 2021).

### 3.3 Enable LLMs to Capture Structured Knowledge

Code generation unveils superior structural commonsense reasoning. Given that code possesses the graph structure of symbolic representations, translating textual graphs, tables, and images into code empowers a code-driven LLM to logically process such information according to code reasoning and generation principles.

Consequently, previous work (Madaan et al., 2022; Wang et al., 2023f) shows that LLMs under-
going extra pre-training on code may rival, or even exceed, their fine-tuned natural language counterparts in tasks involving structural commonsense reasoning, even with limited or no training data.

COCOGEN (Madaan et al., 2022) first reframed the commonsense reasoning graph completion task as a code generation task and demonstrated improved few-shot performance in reasoning graphs, table entity state tracking, and explanation graph generation.

Building on this perspective, CODE4STRUCT (Wang et al., 2023f) applied code-LLMs to semantic structures, focusing on the event argument extraction task. By leveraging code's features such as comments and type annotation, it achieved competitive performance with minimal training instances. Moreover, it surpassed baselines in zero-shot scenarios, benefiting from the inheritance feature and sibling event-type samples. ViStruct (Chen et al., 2023d) extended this approach further to multimodal tasks, leveraging programming language for representing visual structural information and curriculum learning for enhancing the model's understanding of visual structures.

Markup code mastery evolves visually situated natural language understanding. Another stream of research focuses on utilizing markup code such as HTML and CSS to delineate and derender structured graphical information in graphical user interfaces (GUIs) or visualizations such as plots and charts in documents. This markup code not only specifies content but also governs the layout of a web page, aiding large vision-language models (LVLMs) in capturing visually situated natural language (VSNL) information.

For LVLMs' markup code understanding, WebGUM (Furuta et al., 2023) exemplified autonomous web navigation. It employed a pretraining approach using webpage screenshots and the corresponding HTML as input, and navigation action as output. Outperforming SOTA, human, and other LLM-based agents, WebGUM showcased the effectiveness of pre-training model with markup code augmentation in webpage understanding.

For markup code generation, pix2code (Beltramelli, 2017) and sketch2code (Robinson, 2019) pioneered machine learning methods to generate rendering code for GUIs or website mockups, in the pre-LLM era. Pix2Struct (Lee et al., 2023) achieved SOTA at that time in VSNL understanding

![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-06.jpg?height=362&width=762&top_left_y=236&top_left_x=1047)

Figure 4: The code-centric tool-calling paradigm serves as a unified interface between LLMs and a large variety of functional ends, thus enabling many cross-modality and cross-domain tasks.

by pre-training an image-to-text model on masked website screenshots, and further training with OCR, language modeling, and image captioning objectives. Building on this, MATCHA (Liu et al., 2023a) introduced additional pre-training objectives based on chart derendering, layout comprehension, and mathematical reasoning, surpassing SOTA on VSNL understanding tasks.

## 4 Code Connects LLMs to Other Function Ends

Recent studies show that connecting LLMs to other functional ends (i.e., augmenting LLMs with external tools and execution modules) helps LLMs to perform tasks more accurately and reliably (Mialon et al., 2023; Parisi et al., 2022b; Peng et al., 2023; Gou et al., 2023). These functional ends empower LLMs to access external knowledge, engage with diverse modalities, and interact effectively with various environments. As indicated in Table 1, we observe a prevalent trend where LLMs generate programming languages or utilize pre-defined functions to establish connections with other functional ends-a phenomenon we refer to as the code-centric paradigm.

In contrast to the rigid practice of strictly hardcoding tool calls within the LLMs' inference mechanism, the code-centric paradigm allows LLMs to dynamically generate tokens that invoke execution modules with adaptable parameters. This paradigm enables a simple and unambiguous way for LLMs to interact with other functional ends, enhancing the flexibility and scalability of their application. Importantly, as depicted in Figure 4, it allows LLMs to engage with numerous functional ends spanning diverse modalities and domains. By expanding both the quantity and variety of functional ends accessible, LLMs can handle more complicated tasks.

Table 1: Representative work connecting LLMs to different function ends for performing non-trivial tasks. Initial efforts embed tool calls rigidly within the LLMs' inference mechanism (indicated by “*"), resulting in diminished flexibility and constrained tool accessibility. More recently, the code-centric paradigm establishes connections between LLMs and function ends through programming languages or pre-defined functions (indicated by " $\dagger$ "). This approach enhances the scalability of LLMs' function end invocation across diverse tools and execution modules.

| Major Type of Function Ends | Representative Work | Connecting Paradigm | Learning Method | Objectives or Problems to Solve |
| :---: | :---: | :---: | :---: | :---: |
| Single Tool | Retriever in REALM (Guu et al., 2020) <br> Verifier in GSM8K (Cobbe et al. | Hardcoded in Inference Mechanism* <br> Hardcoded in Inference Mechanism* | Example Fine-tuning <br> Examnle Fine-tuning | Augment LLMs with Tools |
| Limited Text-based Tools | Blenderbot3 (Shuster et al., 2022) <br> LamDA (Thoppilan et al., 2022) | Hardcoded in Inference Mechanism* <br> Generate Pre-defined Functions ${ }^{\dagger}$ | Example Fine-tuning <br> Example Fine-tuning | Open-domain Conversation |
| Text-based Tools | TALM (Parisi et al., 2022a) <br> ToolFormer (Schick et al., 2023) | Generate Pre-defined Functions ${ }^{\dagger}$ <br> Generate Pre-defined Functions | Iterative Self-play <br> Self-supervised Training | Efficient and Generalizable Tool Using |
| Multi-modal Modules | MM-React (Yang et al., 2023) <br> CodeVQA (Subramanian et al., 2023) <br> VISPROG (Gupta and Kembhavi, 2023) <br> ViperGPT (Surís et al., 2023) | ![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-07.jpg?height=110&width=324&top_left_y=685&top_left_x=904) | Zero-shot Prompting <br> Zero-shot \& Few shot <br> Zero-shot Prompting <br> Zero-shot Prompting | Multi-modal Reasoning Tasks |
| Real-World APIs | Code as Policies (Liang et al., 2023a) <br> Progprompt (Singh et al., 2022) <br> SayCan (Ahn et al., 2022) | ![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-07.jpg?height=82&width=324&top_left_y=789&top_left_x=904) | Few-shot Prompting <br> Zero-shot Prompting <br> Zero-shot Prompting | Better Robot Control |
|  | RRR (Cui et al., 2023a) <br> Agent-Driver (Mao et al., 2023) <br> LaMPilot (Ma et al., 2023b) | ![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-07.jpg?height=82&width=324&top_left_y=866&top_left_x=904) | Zero-shot Prompting <br> Few-shot Prompting <br> Zero-shot \& Few-shot | Autonomous Driving Ecosystems |

In $\S 4.1$, we examine (digital) textual and multimodal tools connected to LLMs, while $\S 4.2$ focuses on physical-world functional ends, including robots and autonomous driving, showcasing the versatility of LLMs in tackling problems across various modalities and domains.

### 4.1 Relate LLMs to Digital Ends

Text-Based Tools. The code-centric framework has enhanced precision and clarity to LLMs' tool invocation, initially driving progress in text-based tools. Prior to the popularity of this code-centric paradigm, research on augmenting LMs with single tools like information retrivers (Guu et al., 2020; Lewis et al., 2020; Izacard et al., 2022; Borgeaud et al., 2022; Yasunaga et al., 2022) required a hardcoded-in-inference-mechanism (e.g. always calling a retriever before the generation starts), which was less flexible and harder to scale. TALM (Parisi et al., 2022a) first incorporates multiple text-based tools by invoking API calls with a predefined delimiter, enabling unambiguous calls to any text-based tools at any position of generation. Following their work, Toolformer (Schick et al., 2023) marks API calls with "<API> </API>" along with their enclosed contents. Later, diverse toollearning approaches were introduced to facilitate the integration of numerous text-based tools across various foundational models (Song et al., 2023; Hao et al., 2023; Tang et al., 2023).

The code-centric framework facilitates the invocation of a diverse range of external text modules. These include calculators, calendars, machine translation systems, web navigation tools, as well as APIs from HuggingFace and TorchHub (Thop- pilan et al., 2022; Yao et al., 2022c; Shuster et al., 2022; Jin et al., 2023; Yao et al., 2022b; Liu et al., 2023e; Jin et al., 2023; Patil et al., 2023).

Multimodal Tools. The high scalability of the code-centric LLM paradigm enables the extension of tool-learning to modalities other than text. Early work (Gupta and Kembhavi, 2023; Surís et al., 2023; Subramanian et al., 2023) use the codecentric paradigm to tackle the visual question answering task. For instance, VISPROG (Gupta and Kembhavi, 2023) curates various pretrained computer vision models and functions from existing image processing libraries (e.g. Pillow and OpenCV) as a set of APIs. The API calls can then be chained together as a program for question-targeted image understanding, where the program is generated via in-context learning with LLMs. Containing arithmetic in its API code language, the program is capable of performing simple arithmetic tasks, thus enabling VISPROG to answer concrete questions such as object counting. Similar work includes ViperGPT (Surís et al., 2023) and CodeVQA (Subramanian et al., 2023). Compared to VISPROG, they directly generate more flexible Python code using Codex. This enables them to potentially generate programs of more complex control flows using the pre-trained knowledge embedded in Codex. In addition to visual reasoning, code has also been used to connect LLMs with multi-modal generative tools in image generation tasks (Cho et al., 2023; Feng et al., 2023; Wu et al., 2023a), where code's unambiguous nature is leveraged in generating images that better match their text prompts.

Beyond the image-based tools, other modalities
have been considered and used in a collaborative fashion by recent work (Shen et al., 2023; Yang et al., 2023; Liang et al., 2023d). For example, MMREACT (Yang et al., 2023) considers video recognition models in their API, and Chameleon ( $\mathrm{Lu}$ et al., 2023) includes tools such as visual text detector or web search. In HuggingGPT (Shen et al., 2023), the authors connect LLMs to various Hugging Face models and treat each model as an available API call. As a result, HuggingGPT is capable of performing an even wider range of tasks, such as audio-related tasks, that were previously unexplored. Pushing the API diversity further, TaskMatrix.AI (Liang et al., 2023d) uses a magnitude higher number of APIs, spanning from visual \& figure APIs to music and game APIs. The flexibility of code facilitates LLMs to jointly use different multimodal tools. This makes LLMs more versatile and capable of acting as general-purpose multimodal problem solvers that can scale to many tasks.

### 4.2 Relate LLMs to Physical Ends

While the physical world offers a more immersive, contextually rich, and engaging interactive environment compared to the digital realm, the connection between LLMs and the physical world has been constrained until the advent of the codecentric paradigm. This paradigm allows for adaptable calls to tools and execution modules in the physical world, first sparking a wave of research exploring the integration of LLMs with robotics and autonomous driving.

One of the most successful approaches to employing LLMs to generate policy codes for realworld robotic tasks is PaLM-SayCan (Ahn et al., 2022), where LLMs comprehend high-level instructions and then call corresponding APIs for robotic control. Following SayCan, recent developments have shown that LLMs can serve as the brain for robotics planning and control through their powerful code generation capabilities. ProgPrompt (Singh et al., 2022), for instance, pioneered program-like specifications for robot task planning, while other researchers like Huang et al. (2023a), Liang et al. (2023a), and Vemprala et al. (2023) have extended this approach to a range of other tasks, including human-robot interaction and drone control.

Through code generation and tool learning, LLMs also show great potential in more compli-

![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-08.jpg?height=343&width=764&top_left_y=228&top_left_x=1046)

Figure 5: LLMs can be embedded into a code execution environment, where they collect faithful, automatic, and customizable feedback for self-improvement.

cated tasks such as human-vehicle interactions and autonomous driving (Cui et al., 2023b; Huang et al., 2023b; Li et al., 2023d). A prime tool learning example from the industry is Wayve's LINGO1 (Wayve, 2023), which uses an open-loop vision, language, and action LLM to improve the explainability of driving models. Using instruction tuning, LLMs have advanced to the point where they can understand complex driving, navigation, and entertainment commands (Wang et al., 2023e), generate actionable codes (Ma et al., 2023b), and execute them by calling low-level vehicle planning and control APIs (Cui et al., 2023a; Sha et al., 2023; Mao et al., 2023).

Overall, despite challenges such as latency, accuracy issues, and the absence of adequate simulation environments, datasets, and benchmarks (Kannan et al., 2023; Chen and Huang, 2023; Cui et al., 2023b), LLMs show promise in understanding high-level instructions and executing code-related APIs in intricate domains like robotics and autonomous driving. Looking ahead, there's considerable potential for LLMs to bridge the gap between physical worlds and AI, influencing areas like transportation and smart manufacturing (Panchal and Wang, 2023; Zeng et al., 2023a).

## 5 Code Provides LLM with an Executable Environment for Automated Feedback

LLMs demonstrate performance beyond the parameters of their training, in part due to their ability to intake feedback, especially in real-world applications where environments are rarely static (Liu et al., 2023f; Wang et al., 2023d). However, feedback must be chosen carefully as noisy prompts can hinder LMs' performance on downstream tasks (Zheng and Saparov, 2023). Furthermore, as human effort is expensive, it is crucial that feedback can be automatically collected while staying faithful. Embedding LLMs into a code execution envi-
ronment enables automated feedback that fulfills all of these criteria, as shown in Figure 5. As the code execution is largely deterministic, LLMs that intake feedback from the results of executed code remain faithful to the task at hand (Chen et al., 2023a; Fernandes et al., 2023; Scheurer et al., 2022). Furthermore, code interpreters provide an automatic pathway for LLMs to query internal feedback, eliminating the need for costly human annotations as seen when leveraging LLMs to debug or optimize faulty code (Chen et al., 2023a; Fernandes et al., 2023; Scheurer et al., 2022). In addition, code environments allow LLMs to incorporate diverse and comprehensive forms of external feedback, such as critic on binary correctness (Wang et al., 2023g), natural language explanations on results (Chen et al., 2023c), and ranking with reward values (Inala et al., 2022), enabling highly customizable methods to enhance performance.

We introduce the various types of feedback derived from code execution that can be jointly utilized to benefit LLMs in $\S 5.1$, and discuss common methods for utilizing this feedback to improve LLMs in $\S 5.2$.

### 5.1 Various Feedback from Code Execution

The code execution process enables assessing LLM-generated content with more comprehensive evaluation metrics derived from deterministic execution results, as opposed to relying solely on often ambiguous sequence-based metrics like BLEU (Papineni et al., 2002; Ren et al., 2020) and Rouge (Lin, 2004).

Straightforward methods for evaluating program execution outcomes and generating feedback include the creation of unit tests (Chen et al., 2021; Hendrycks et al., 2021; Austin et al., 2021; Li et al., 2022; Huang et al., 2022a; Lai et al., 2023) and the application of exact result matching techniques (Dong and Lapata, 2016; Zhong et al., 2017; Huang et al., 2022a). From these, feedback can be provided in two primary forms: simple correctness feedback and textual feedback. Simple feedback, indicating whether a program is correct or not, can be generated through critic models or rule-based methods (Wang et al., 2023g; Chen et al., 2023c).

For more detailed textual feedback, language models can be employed to produce explanations either about the program itself (Chen et al., 2023c), or to summarize comments on the program and its execution (Wang et al., 2023g; Chen et al., 2023c;
Zhang et al., 2023a). Execution results can also be translated into reward functions using predefined rules. The rules map execution results into scalar values based on the severity of different error types, thus making the feedback format suitable for reinforcement learning approaches (Le et al., 2022). Moreover, additional feedback can be extracted by performing static analysis using software engineering tools. For instance, Wang et al. (2017) and Gupta et al. (2017) obtain extra information from the execution trace, including variable or state traces. Lai et al. (2023) demonstrates an effective way to extract extra feedback using surface-form constraints on function calls.

### 5.2 Methods for Enhancing LLM's Performance with Feedback

The feedback derived from code execution and external evaluation modules can enhance LLMs through three major approaches.

Selection Based Method. Selection-based methods, such as majority voting and re-ranking schemes, have proven effective in enhancing LLM performance in tasks such as code generation. These methods, originally developed for simple program synthesis, leverage code execution outcomes like the number of passed unit tests to choose the best-performing code snippet. Studies like Chen et al. (2018); Li et al. (2022) demonstrate the efficacy of majority voting, while Zhang et al. (2023b); Yin and Neubig (2019); Zeng et al. (2023b) showcase the advantages of re-ranking schemes. Building on this success, similar approaches have been adapted for more challenging tasks where code-LLMs are integrated in interactive environments, as shown in the work of Shi et al. (2022); Chen et al. (2022) for similar voting methods, and Ni et al. (2023); Inala et al. (2022); To et al. (2023) for re-ranking strategies. However, these approaches, while simple and effective, cause inefficiencies, as they necessitate multiple rounds of generation and the employment of additional re-ranking models to identify the optimal solution.

Prompting Based Method. Modern LLMs are equipped with the capability to reason in-context and directly integrate feedback from task descriptions into prompts, to certain extents. Improving LLM "self-debugging" with in-context learning prompts typically requires feedback presented as natural language explanations (Wang et al., 2023h; Chen et al., 2023c) or error messages derived from
execution results, as these formats are more comprehensible for the LLM. This method is favored by most LLM-based agents (see $\S 6$ ) due to its automatic nature, computational efficiency, and lack of requirement for additional fine-tuning. However, the effectiveness of this approach heavily depends on the LLM's in-context learning capabilities.

Finetuning Based Method. In the aforementioned methods, neither the selection-based method nor the prompting-based method promises steady improvement over the task, as the LLMs' parameters remain unchanged. They require repeating the tuning process even when faced with similar problems. Finetuning approaches, on the other hand, fundamentally improve the LLMs by updating their parameterized knowledge. Typical finetuning strategies include direct optimization, leveraging an external model for optimization, and training the model in a reinforcement learning paradigm. Wang et al. (2023g) exemplifies the direct optimization approach, where the original language model is fine-tuned with a feedback-conditioned objective. Haluptzok et al. (2022) presents a unique method where language models generate synthetic unit tests to identify and retain only correctly generated examples, which are then composed into correct question-answer pairs and employed to further fine-tune the models. CodeScore (Dong et al., 2023a) designs its loss function based on executability and the pass rate on the unit tests. For self-edit (Zhang et al., 2023a), it first wraps up execution results into textual comments, then trains an editor to further refine the program by accepting both the problematic program and the comments. Chen et al. (2023a) train a "refine model" which accepts the feedback and generated program first, then use the refined generated example to finetune the generation model, illustrating a layered approach to fine-tuning. CodeRL (Le et al., 2022) and Wang et al. (2022) apply reinforcement learning to improve the original language model. While Wang et al. (2022) aims at employing compiler feedback to obtain erroneous code, CodeRL (Le et al., 2022) empirically defines fixed reward values for different execution result types based on unit tests. Despite the discussed advantages, refining LLMs through finetuning typically involves a resource-intensive data collection process. Additionally, assessing predefined reward values, as exemplified in CodeRL (Le et al., 2022), poses certain challenges.

## 6 Application: Code-empowered LLMs Facilitate Intelligent Agents

In the preceding sections, our discussion highlighted the various ways in which code integration enhances LLMs. Going beyond, we discern that the benefits of code-empowered LLMs are especially pronounced in one key area: the development of IAs (Xu et al., 2023). In this section, we underscore the unique capabilities bestowed upon these agents by code-empowered LLMs.

Figure 6 helps to illustrate a standard operational pipeline of an IA, specifically serving as an embodied general daily assistant. We observe that the improvements brought about by code training in LLMs are firmly rooted in their practical operational steps when serving as IAs. These steps include (i) enhancing the IA's decision-making in terms of environment perception and planning (§6.1), (ii) streamlining execution by grounding actions in modular and explicit action primitives and efficiently organizing memory (\$6.2), and (iii) optimizing performance through feedback automatically derived from the code execution environment (§6.3). Detailed explanations of each aspect will follow in the subsequent sections.

### 6.1 Decision Making

Environment Perception As depicted in Figure 6 at step $(0-10)$, the IA continuously perceives the world, engaging in interactions with humans and the environment, responding to relevant stimuli (e.g., human instructions for meal preparation), and planning and executing actions based on the observed environmental conditions (e.g., the kitchen layout). Utilizing LLMs as decision centers for IAs requires translating environmental observations into text, such as tasks based in the virtual household or Minecraft (Shridhar et al., 2020; Côté et al., 2018; Wang et al., 2023b; Zhu et al., 2023). The perceived information needs to be organized in a highly structured format, ensuring that stimuli occurring at the same moment (e.g., coexisting multimodality stimuli) influence the IA's perception and decision simultaneously without temporal differences, a requirement that contrasts with the sequential nature of free-form text. Through pretraining on code, LLMs acquire the ability to better comprehend and generate structured representations resembling class definitions in code. This structured format, where class attributes and functions are permutation invariant, facilitates agents in

![](https://cdn.mathpix.com/cropped/2024_06_04_3d16c04d724d37b00cb2g-11.jpg?height=708&width=1587&top_left_y=246&top_left_x=243)

Figure 6: This figure illustrates the complete working pipeline of a LLM-based intelligent agent, mapping codeLLM abilities to specific phases: code-based planning in step (2), modular action parsing and tool creation in step (3), and automated feedback collection for enhanced agent self-improvement in step (5). Collectively, steps 0-10 in the entire loop benefit from code-LLMs' improved structured information understanding and perception.

perceiving structured environmental observations during task execution.

One such intuitive example is web-page-based environments which are highly structured around HTML code. In agent tasks like web shopping (Yao et al., 2022a), web browsing (Deng et al., 2023b), and web-based QA (Nakano et al., 2021; Liu et al., 2023d), it is preferred to translate the web-based environment into HTML code rather than natural language, directly encompassing its structural information and thereby improving the LLM agent's overall perception. Moreover, in robotics research by Singh et al. (2022) and Liang et al. (2023a), the IAs are prompted with program-like specifications for objects in the environment, enabling the LLM to generate situated task plans based on the virtual objects they perceived.

Planning As illustrated in Figure 6 at step (2), IAs must break down intricate tasks into finer, manageable steps. Leveraging the synergized planning abilities of code-LLMs, IAs can generate organized reasoning steps using modular and unambiguous code alongside expressive natural language. As discussed in §2.2, when code-LLMs are employed for planning agent tasks, they exhibit enhanced reasoning capabilities. In addition, they generate the sub-tasks as executable programs when necessary, yielding more robust intermediate results, which the IA conditions on and refines its planning with greater precision. Furthermore, the IA seamlessly integrates performant tool APIs into planning, addressing the limitations such as poor mathematical reasoning and outdated information updates faced by vanilla LLMs during planning.

Typical examples that utilize code for planning are in two main categories. Progprompt (Singh et al., 2022) and Code as Policies (Liang et al., 2023a) represent the work utilizing code for better robot control. Both work highlight the benefits brought by code-based planning as they not only enable direct expressions of feedback loops, functions, and primitive APIs, but also facilitate direct access to third-party libraries. Another stream of work is concerned with the scenario when the agents' programming and mathematical reasoning abilities are crucial, like solving maths-related problems (Gao et al., 2023; Wang et al., 2023h) or doing experiments in the scientific domain (Boiko et al., 2023; Liffiton et al., 2023).

### 6.2 Execution

Action Grounding As depicted in Figure 6 at step (3), when the IA interfaces with external function ends according to the planning, it must invoke action primitives from a pre-defined set of actions (i.e., functions). Given that modern LLMs are trained in formal language and can generate highly formalized primitives, the IA's generation can be directly parsed and executed, eliminating the necessity for additional action primitive grounding modules.

Connecting the IA with other function ends requires grounding actions into formalized functionlike primitives. For instance, in a benchmark evaluating LLMs as agents in real-world scenarios (Liu et al., 2023f), seven out of eight scenarios involve code as the action space.

Previous work generating agent plans with pure natural language necessitate an additional step-toprimitive module to ground those planning steps into code (Wang et al., 2023c; Yin et al., 2023). In contrast, IAs that plan with code-LLMs generate atomic action programs (Yao et al., 2022d; Wang et al., 2023h; Liang et al., 2023a; Singh et al., 2022), and can have their generation quickly parsed for execution.

Memory Organization As depicted in Figure 6 at step (3) and the component labeled "Function Ends for Updating State," the IA typically necessitates an external memory organization module to manage exposed information (Wang et al., 2023d), including original planning, task progress, execution history, available tool set, acquired skills, augmented knowledge, and users' early feedback. In this context, Code-LLM aids the IA's memory organization by employing highly abstract and modular code to record, organize, and access memory, especially for expanding the available tool set and manage acquired skills.

Typically, agent-written code snippets can serve as parts of the toolset, integrated into the memory organization of agents. This stream of research is known as tool creation approaches. TALM (Cai et al., 2023) proposes to use stronger agents (e.g. GPT-4 based agents) to write code as part of memory for weaker agents (e.g. GPT-3.5 based agents). In Creator (Qian et al., 2023b), agents themselves are highlighted as not only users of the tools but also their creators. They proposed a four-stage toolcreation framework that enables agents to write code as part of their executable tool set. Going further, Craft (Yuan et al., 2023) focuses on ensuring the created tools are indeed executable, making the framework more robust. Another work sharing this idea is Voyager (Wang et al., 2023b), in which the agent store learned skills in code format and execute them in the future when faced with similar tasks.

### 6.3 Self-improvement

As illustrated in Figure 6 at step (5), when the IA's decision center, i.e., the LLM, operates within a code execution environment, the environment can integrate various evaluation modules to offer automated feedback (e.g., correctness, ranking, detailed comments). This significantly enhances the IA's early error correction and facilitates selfimprovement.

Voyager (Wang et al., 2023b) is a good example for agents that use feedback from the simulated environment. The agent learns from failure task cases and further horn its skills in Minecraft. Chameleon (Lu et al., 2023) receives feedback from a program verifier to decide whether it should regenerate an appropriate program. Mint (Wang et al., 2023h) can receive feedback from proxies, and the agent can thus self-improve in a multi-turn interactive setting. Importantly, this ability to selfimprove from execution feedback is fundamental for agents' success at solving scientific problems (Bran et al., 2023; Swan et al., 2023; Wu et al., 2023b).

## 7 Challenges

We identify several intriguing and promising avenues for future research.

### 7.1 The Causality between Code Pre-training and LLMs' Reasoning Enhancement

Although we have categorized the most pertinent work in $\S 3.2$, a noticeable gap persists in providing explicit experimental evidence that directly indicates the enhancement of LLMs' reasoning abilities through the acquisition of specific code properties. While we intuitively acknowledge that certain code properties likely contribute to LLMs' reasoning capabilities, the precise extent of their influence on enhancing reasoning skills remains ambiguous. In the future research endeavors, it is important to investigate whether reinforcing these code properties within training data could indeed augment the reasoning capabilities of trained LLMs. If it is indeed the case, that pre-training on specific properties of code directly improves LLMs' reasoning abilities, understanding this phenomenon will be key to further improving the complex reasoning capabilities of current models.

### 7.2 Acquisition of Reasoning Beyond Code

Despite the enhancement in reasoning achieved by pre-training on code, foundational models still lack the human-like reasoning abilities expected from a truly generalized artificial intelligence. Im-
portantly, beyond code, a wealth of other textual data sources holds the potential to bolster LLM reasoning abilities, where the intrinsic characteristics of code, such as its lack of ambiguity, executability, and logical sequential structure, offer guiding principles for the collection or creation of these datasets. However, if we stick to the paradigm of training language models on large corpora with the language modeling objective, it's hard to envision a sequentially readable language that is more abstract, highly structured, and closely aligned with symbolic language than formal languages, exemplified by code, which are prevalent in a substantial digital context. We envision that exploring alternative data modalities, diverse training objectives, and novel architectures would present additional opportunities to further enhance the reasoning capabilities of these models.

### 7.3 Challenges of Applying Code-centric Paradigm

The primary challenge in LLMs using code to connect to different function ends is learning the correct invocation of numerous functions, including selecting the right function end and passing the correct parameters at an appropriate time. Even for simple tasks like simplified web navigation with a limited set of action primitives like mouse movements, clicks, and page scrolls, few shot examples together with a strong underlying LLM are often required for the LLM to precisely grasp the usage of these primitives (Sridhar et al., 2023). For more complex tasks in data-intensive fields like chemistry (Bran et al., 2023), biology, and astronomy, which involve domain-specific Python libraries with diverse functions and intricate calls, enhancing LLMs' capability of learning the correct invocation of these functions is a prospective direction, empowering LLMs to act as IAs performing expert-level tasks in fine-grained domains.

### 7.4 Learning from multi-turn interactions and feedback

LLMs often require multiple interactions with the user and the environment, continuously correcting themselves to improve intricate task completion (Li et al., 2023c). While code execution offers reliable and customizable feedback, a perfect method to fully leverage this feedback has yet to be established. As discussed in $\S 5.2$, we observed that selection-based methods, though useful, do not guarantee improved performance and can be inef- ficient. Prompting-based methods heavily depend on the in-context learning abilities of the LLM, which might limit their applicability. Fine-tuning methods show consistent improvement, but data collection and fine-tuning are resource-intensive and thus prohibitive. We hypothesize that reinforcement learning could be a more effective approach for utilizing feedback and improving LLMs. This method can potentially address the limitations of current techniques by providing a dynamic way to adapt to feedback through well-designed reward functions. However, significant research is still needed to understand how reward functions should be designed and how reinforcement learning can be optimally integrated with LLMs for complex task completion.

## 8 Conclusion

In this survey, we compile literature that elucidates how code empowers LLMs, as well as where code assists LLMs to serve as IAs. To begin with, code possesses natural language's sequential readability while also embodying the abstraction and graph structure of symbolic representations, rendering it a conduit for knowledge perception and reasoning as an integral part of the LLMs' training corpus based on the mere language modeling objective. Through a comprehensive literature review, we observe that after code training, LLMs i) improve their programming skills and reasoning, ii) could generate highly formalized functions, enabling flexible connections to diverse functional ends across modalities and domains, and iii) engage in interaction with evaluation modules integrated in the code execution environment for automated selfimprovement. Moreover, we find that the LLMs' capability enhancement brought by code training benefits their downstream application as IAs, manifesting in the specific operational steps of the IAs' workflow regarding decision-making, execution, and self-improvement. Beyond reviewing prior research, we put forth several challenges in this field to serve as guiding factors for potential future directions.

## References

Rajas Agashe, Srini Iyer, and Luke Zettlemoyer. 2019. Juice: A large scale distantly supervised dataset for open domain context-based code generation. In Conference on Empirical Methods in Natural Language Processing.

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. 2022. Do as i can, not as i say: Grounding language in robotic affordances.

Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, Logesh Kumar Umapathi, Carolyn Jane Anderson, Yangtian Zi, Joel Lamy Poirier, Hailey Schoelkopf, Sergey Troshin, Dmitry Abulkhanov, Manuel Romero, Michael Lappert, Francesco De Toni, Bernardo García del Río, Qian Liu, Shamik Bose, Urvashi Bhattacharyya, Terry Yue Zhuo, Ian Yu, Paulo Villegas, Marco Zocca, Sourab Mangrulkar, David Lansky, Huu Nguyen, Danish Contractor, Luis Villa, Jia Li, Dzmitry Bahdanau, Yacine Jernite, Sean Hughes, Daniel Fried, Arjun Guha, Harm de Vries, and Leandro von Werra. 2023. Santacoder: don't reach for the stars!

Uri Alon, Roy Sadaka, Omer Levy, and Eran Yahav. 2020. Structural language models of code.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program synthesis with large language models.

Tony Beltramelli. 2017. pix2code: Generating code from a graphical user interface screenshot.

Daniil A. Boiko, Robert MacKnight, and Gabe Gomes. 2023. Emergent autonomous scientific research capabilities of large language models. ArXiv, abs/2304.05332.

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. 2022. Improving language models by retrieving from trillions of tokens. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 2206-2240. PMLR.
Andres M Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. 2023. Chemcrow: Augmenting large-language models with chemistry tools.

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. 2023. Large language models as tool makers. ArXiv, abs/2305.17126.

Tommaso Calò and Luigi De Russis. 2023. Leveraging large language models for end-user website generation. In International Symposium on End User Development, pages 52-61. Springer.

Angelica Chen, Jérémy Scheurer, Tomasz Korbak, Jon Ander Campos, Jun Shern Chan, Samuel R. Bowman, Kyunghyun Cho, and Ethan Perez. 2023a. Improving code generation by training with natural language feedback.

Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2022. Codet: Code generation with generated tests.

Juo-Tung Chen and Chien-Ming Huang. 2023. Forgetful large language models: Lessons learned from using llms in robot programming.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023b. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. 2023c. Teaching large language models to self-debug.

Xinyun Chen, Chang Liu, and Dawn Song. 2018. Execution-guided neural program synthesis. In International Conference on Learning Representations.

Yangyi Chen, Xingyao Wang, Manling Li, Derek Hoiem, and Heng Ji. 2023d. Vistruct: Visual structural knowledge extraction via curriculum guided code-vision representation.

Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Tao Yu. 2023. Binding language models in symbolic languages.

Jaemin Cho, Abhay Zala, and Mohit Bansal. 2023. Visual programming for text-to-image generation and evaluation.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems.

Marc-Alexandre Côté, Ákos Kádár, Xingdi Yuan, Ben A. Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew J. Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. 2018. Textworld: A learning environment for text-based games. ArXiv, abs/1806.11532.

Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, and Ziran Wang. 2023a. Receive, reason, and react: Drive as you say with large language models in autonomous vehicles.

Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, Tianren Gao, Erlong Li, Kun Tang, Zhipeng Cao, Tong Zhou, Ao Liu, Xinrui Yan, Shuqi Mei, Jianguo Cao, Ziran Wang, and Chao Zheng. 2023b. A survey on multimodal large language models for autonomous driving.

Gelei Deng, Yi Liu, Víctor Mayoral-Vilches, Peng Liu, Yuekang Li, Yuan Xu, Tianwei Zhang, Yang Liu, Martin Pinzger, and Stefan Rass. 2023a. Pentestgpt: An llm-empowered automatic penetration testing tool.

Xiang Deng, Yu Gu, Bo Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023b. Mind2web: Towards a generalist agent for the web. ArXiv, abs/2306.06070.

Li Dong and Mirella Lapata. 2016. Language to logical form with neural attention.

Yihong Dong, Ji Ding, Xue Jiang, Zhuo Li, Ge Li, and Zhi Jin. 2023a. Codescore: Evaluating code generation by learning code execution. ArXiv, abs/2301.09043.

Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. 2023b. Self-collaboration code generation via chatgpt.

Iddo Drori, Sarah Zhang, Reece Shuttleworth, Leonard Tang, Albert Lu, Elizabeth Ke, Kevin Liu, Linda Chen, Sunny Tran, Newman Cheng, Roman Wang, Nikhil Singh, Taylor L. Patti, Jayson Lynch, Avi Shporer, Nakul Verma, Eugene Wu, and Gilbert Strang. 2022. A neural network solves, explains, and generates university math problems by program synthesis and few-shot learning at human level. Proceedings of the National Academy of Sciences, 119(32).

Xueying Du, Mingwei Liu, Kaixin Wang, Hanlin Wang, Junwei Liu, Yixuan Chen, Jiayi Feng, Chaofeng Sha, Xin Peng, and Yiling Lou. 2023. Classeval: A manually-crafted benchmark for evaluating llms on class-level code generation.

Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. 2022. Successive prompting for decomposing complex questions.

Kevin Ellis, Maxwell Nye, Yewen Pu, Felix Sosa, Josh Tenenbaum, and Armando Solar-Lezama. 2019. Write, execute, assess: Program synthesis with a repl. Advances in Neural Information Processing Systems, 32 .

Angela Fan, Beliz Gokkaya, Mark Harman, Mitya Lyubarskiy, Shubho Sengupta, Shin Yoo, and Jie M. Zhang. 2023. Large language models for software engineering: Survey and open problems.

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2023. Layoutgpt: Compositional visual planning and generation with large language models. arXiv preprint $\underline{\text { arXiv:2305.15393. }}$.

Patrick Fernandes, Aman Madaan, Emmy Liu, António Farinhas, Pedro Henrique Martins, Amanda Bertsch, José GC de Souza, Shuyan Zhou, Tongshuang Wu, Graham Neubig, et al. 2023. Bridging the gap: A survey on integrating (human) feedback for natural language generation. arXiv preprint arXiv:2305.00955.

Hao Fu, Yao; Peng and Tushar Khot. 2022. How does gpt obtain its ability? tracing emergent abilities of language models to their sources. Yao Fu's Notion.

Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. 2023. Complexity-based prompting for multi-step reasoning.

Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Matsuo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. 2023. Multimodal web navigation with instruction-finetuned foundation models.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764-10799. PMLR.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346361.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2023. CRITIC: Large language models can self-correct with tool-interactive critiquing.

Rahul Gupta, Soham Pal, Aditya Kanade, and Shirish Shevade. 2017. Deepfix: Fixing common c language errors by deep learning. In Proceedings of the aaai conference on artificial intelligence, volume 31 .

Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14953-14962.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 3929-3938. PMLR.

Patrick Haluptzok, Matthew Bowers, and Adam Tauman Kalai. 2022. Language models can teach themselves to program better. arXiv preprint arXiv:2207.14502.

Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. 2023. ToolkenGPT: Augmenting frozen language models with massive tools via tool embeddings.

Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. 2021. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938.

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, et al. 2023. Metagpt: Meta programming for multiagent collaborative framework. arXiv preprint arXiv:2308.00352.

Junjie Huang, Chenglong Wang, Jipeng Zhang, Cong Yan, Haotian Cui, Jeevana Priya Inala, Colin Clement, Nan Duan, and Jianfeng Gao. 2022a. Executionbased evaluation for data science code generation models. arXiv preprint arXiv:2211.09374.

Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. 2022b. Language models as zeroshot planners: Extracting actionable knowledge for embodied agents. In International Conference on Machine Learning, pages 9118-9147. PMLR.

Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. 2023a. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint $\underline{\text { arXiv:2307.05973. }}$.

Yu Huang, Yue Chen, and Zhu Li. 2023b. Applications of large scale foundation models for autonomous driving. arXiv preprint arXiv:2311.12144.
Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700 6709 .

Jeevana Priya Inala, Chenglong Wang, Mei Yang, Andres Codas, Mark Encarnación, Shuvendu Lahiri, Madanlal Musuvathi, and Jianfeng Gao. 2022. Faultaware neural code rankers. Advances in Neural Information Processing Systems, 35:13419-13432.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane DwivediYu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022. Atlas: Few-shot learning with retrieval augmented language models.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1-38.

Xue Jiang, Yihong Dong, Lecheng Wang, Qiwei Shang, and Ge Li. 2023. Self-planning code generation with large language model. arXiv preprint $\underline{\text { arXiv:2303.06689. }}$.

Qiao Jin, Yifan Yang, Qingyu Chen, and Zhiyong Lu. 2023. GeneGPT: Augmenting large language models with domain tools for improved access to biomedical information.

Sungmin Kang, Gabin An, and Shin Yoo. 2023a. A preliminary evaluation of llm-based fault localization. arXiv preprint arXiv:2308.05487.

Sungmin Kang, Juyeon Yoon, and Shin Yoo. 2023b. Large language models are few-shot testers: Exploring llm-based general bug reproduction. In 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pages 2312-2323. IEEE.

Shyam Sundar Kannan, Vishnunandan LN Venkatesh, and Byung-Cheol Min. 2023. Smart-llm: Smart multi-agent robot task planning using large language models. arXiv preprint arXiv:2309.10062.

Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. 2023. Ds-1000: A natural and reliable benchmark for data science code generation. In International Conference on Machine Learning, pages 18319-18345. PMLR.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. 2022. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:2131421328 .

Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. 2023. Pix2struct: Screenshot parsing as pretraining for visual language understanding.

Ioktong Lei and Zhidong Deng. 2023. Selfzcot: a selfprompt zero-shot cot from semantic-level to codelevel for a better utilization of $11 \mathrm{~ms}$.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020 Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459-9474. Curran Associates, Inc.

Haonan Li, Yu Hao, Yizhuo Zhai, and Zhiyun Qian. 2023a. The hitchhiker's guide to program analysis: A journey with large language models. arXiv preprint arXiv:2308.00245.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi 2023b. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597.

Sha Li, Chi Han, Pengfei Yu, Carl Edwards, Manling Li, Xingyao Wang, Yi Fung, Charles Yu, Joel Tetreault, Eduard Hovy, and Heng Ji. 2023c. Defining a new NLP playground. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 11932-11951, Singapore. Association for Computational Linguistics.

Xin Li, Yeqi Bai, Pinlong Cai, Licheng Wen, Daocheng Fu, Bo Zhang, Xuemeng Yang, Xinyu Cai, Tao Ma, Jianfei Guo, et al. 2023d. Towards knowledge-driven autonomous driving. arXiv preprint arXiv:2312.04316.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. 2022. Competition-level code generation with alphacode. Science, 378(6624): 1092-1097.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. 2023a. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493-9500. IEEE.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher R'e, Diana Acosta-Navas, Drew A. Hudson, E. Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel J. Orr, Lucia Zheng, Mert
Yuksekgonul, Mirac Suzgun, Nathan S. Kim, Neel Guha, Niladri S. Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas F. Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2023b. Holistic evaluation of language models. Annals of the New York Academy of Sciences, 1525:140 - 146.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2023c. Holistic evaluation of language models.

Yaobo Liang, Chenfei Wu, Ting Song, Wenshan Wu, Yan Xia, Yu Liu, Yang Ou, Shuai Lu, Lei Ji, Shaoguang Mao, et al. 2023d. Taskmatrix. ai: Completing tasks by connecting foundation models with millions of apis. arXiv preprint arXiv:2303.16434.

Mark H. Liffiton, Brad E. Sheese, Jaromir Savelka, and Paul Denny. 2023. Codehelp: Using large language models with guardrails for scalable support in programming classes. ArXiv, abs/2308.06921.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74-81.

Jiawei Lin, Jiaqi Guo, Shizhao Sun, Weijiang Xu, Ting Liu, Jian-Guang Lou, and Dongmei Zhang. 2023. A parse-then-place approach for generating graphic layouts from textual descriptions.

Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, and Julian Martin Eisenschlos. 2023a. Matcha: Enhancing visual language pretraining with math reasoning and chart derendering.

Jiateng Liu, Sha Li, Zhenhailong Wang, Manling Li, and Heng Ji. 2023b. A language-first approach for procedure planning. In Findings of the Association for Computational Linguistics: ACL 2023, pages 19411954.

Michael Xieyang Liu, Advait Sarkar, Carina Negreanu, Benjamin Zorn, Jack Williams, Neil Toronto, and Andrew D Gordon. 2023c. "what it wants me to say":

Bridging the abstraction gap between end-user programmers and code-generating large language models. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, pages 1-31.

Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zeng, Zhengxiao Du, P. Zhang, Yuxiao Dong, and Jie Tang. 2023d. Webglm: Towards an efficient web-enhanced question answering system with human preferences. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zeng, Zhengxiao Du, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023e. WebGLM: Towards an efficient webenhanced question answering system with human preferences.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Yuxian Gu, Hangliang Ding, Kai Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Shengqi Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2023f. Agentbench: Evaluating llms as agents. ArXiv, abs/2308.03688

Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. 2023g. Dynamic llm-agent network: An llmagent collaboration framework with agent team optimization. arXiv preprint arXiv:2310.02170.

Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, KaiWei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. 2023. Chameleon: Plug-and-play compositional reasoning with large language models. arXiv preprint arXiv:2304.09842.

Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan, Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. 2021. Codexglue: A machine learning benchmark dataset for code understanding and generation.

Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. 2023. Faithful chain-of-thought reasoning. arXiv preprint arXiv:2301.13379.

Yingwei Ma, Yue Liu, Yue Yu, Yuanliang Zhang, Yu Jiang, Changjian Wang, and Shanshan Li. 2023a. At which training stage does code data help llms reasoning?

Yunsheng Ma, Can Cui, Xu Cao, Wenqian Ye, Peiran Liu, Juanwu Lu, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Aniket Bera, et al. 2023b. Lampilot: An open benchmark dataset for autonomous driving with language model programs. arXiv preprint arXiv:2312.04372.
Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, and Graham Neubig. 2022. Language models of code are few-shot commonsense learners.

Jiageng Mao, Junjie Ye, Yuxi Qian, Marco Pavone, and Yue Wang. 2023. A language agent for autonomous driving. arXiv preprint arXiv:2311.10813.

Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ramakanth Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, Edouard Grave, Yann LeCun, and Thomas Scialom. 2023. Augmented language models: a survey. ArXiv, abs/2302.07842.

Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. 2023. Lila: A unified benchmark for mathematical reasoning.

Mohammad Mahdi Mohajer, Reem Aleithan, Nima Shiri Harzevili, Moshi Wei, Alvine Boaye Belle, Hung Viet Pham, and Song Wang. 2023. Skipanalyzer: An embodied agent for code analysis with large language models. arXiv preprint arXiv:2310.18532.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted questionanswering with human feedback. arXiv preprint arXiv:2112.09332.

Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-tau Yih, Sida Wang, and Xi Victoria Lin. 2023. Lever: Learning to verify language-tocode generation with execution. In International Conference on Machine Learning, pages 2610626128. PMLR.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language model for code with multi-turn program synthesis. $\underline{\text { arXiv preprint arXiv:2203.13474. }}$

David Noever. 2023. Can large language models find and fix vulnerable software? arXiv preprint arXiv:2308.10345.

OpenAI. 2023. Gpt-4 technical report.

Jitesh H Panchal and Ziran Wang. 2023. Design of next-generation automotive systems: Challenges and research opportunities. Journal of Computing and Information Science in Engineering, 23(6).

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311-318.

Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022a Talm: Tool augmented language models. ArXiv, $\mathrm{abs} / 2205.12255$.

Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022b. TALM: Tool augmented language models.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2023. Gorilla: Large language model connected with massive APIs.

Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. 2023. Check your facts and try again: Improving large language models with external knowledge and automated feedback.

Stanislas Polu and Ilya Sutskever. 2020. Generative language modeling for automated theorem proving.

Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. 2023a. Communicative agents for software development. arXiv preprint arXiv:2307.07924.

Cheng Qian, Chi Han, Yi Ren Fung, Yujia Qin, Zhiyuan Liu, and Heng Ji. 2023b. Creator: Tool creation for disentangling abstract and concrete reasoning of large language models.

Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, et al. 2023. Tool learning with foundation models. arXiv preprint arXiv:2304.08354.

Nitarshan Rajkumar, Raymond Li, and Dzmitry Bahdanau. 2022. Evaluating the text-to-sql capabilities of large language models.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-Context retrieval-augmented language models.

Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. 2020. Codebleu: a method for automatic evaluation of code synthesis. arXiv preprint arXiv:2009.10297.

Jasmine Roberts, Andrzej Banburski-Fahey, Microsoft Jaron, and Lanier Microsoft. Surreal vr pong: Llm approach to game design.

Alex Robinson. 2019. Sketch2code: Generating a website from a paper mockup.

Jérémy Scheurer, Jon Ander Campos, Jun Shern Chan, Angelica Chen, Kyunghyun Cho, and Ethan Perez. 2022. Training language models with natural language feedback. arXiv preprint arXiv:2204.14146, 8 .
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools.

Hao Sha, Yao Mu, Yuxuan Jiang, Li Chen, Chenfeng Xu, Ping Luo, Shengbo Eben Li, Masayoshi Tomizuka, Wei Zhan, and Mingyu Ding. 2023. Languagempc: Large language models as decision makers for autonomous driving. arXiv preprint arXiv:2310.03026.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580.

Freda Shi, Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, and Sida I Wang. 2022. Natural language to code translation with execution. arXiv preprint arXiv:2204.11454.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew J. Hausknecht. 2020. Alfworld: Aligning text and embodied environments for interactive learning. ArXiv, $\mathrm{abs} / 2010.03768$.

Kurt Shuster, Jing Xu, Mojtaba Komeili, Da Ju, Eric Michael Smith, Stephen Roller, Megan Ung, Moya Chen, Kushal Arora, Joshua Lane, Morteza Behrooz, William Ngan, Spencer Poff, Naman Goyal, Arthur Szlam, Y-Lan Boureau, Melanie Kambadur, and Jason Weston. 2022. BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage.

Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. 2022. Progprompt: Generating situated robot task plans using large language models.

Yifan Song, Weimin Xiong, Dawei Zhu, Wenhao Wu, Han Qian, Mingbo Song, Hailiang Huang, Cheng Li, Ke Wang, Rong Yao, Ye Tian, and Sujian Li. 2023. RestGPT: Connecting large language models with real-world RESTful APIs.

Davit Soselia, Khalid Saifullah, and Tianyi Zhou. 2023. Learning ui-to-code reverse generator using visual critic without rendering.

Abishek Sridhar, Robert Lo, Frank F. Xu, Hao Zhu, and Shuyan Zhou. 2023. Hierarchical prompting assists large language model on web navigation.

Sanjay Subramanian, Medhini Narasimhan, Kushal Khangaonkar, Kevin Yang, Arsha Nagrani, Cordelia Schmid, Andy Zeng, Trevor Darrell, and Dan Klein. 2023. Modular visual question answering via code generation. arXiv preprint arXiv:2306.05392.

Chunyi Sun, Junlin Han, Weijian Deng, Xinlong Wang, Zishan Qin, and Stephen Gould. 2023a. 3d-gpt: Procedural 3d modeling with large language models. arXiv preprint arXiv:2310.12945.

Ruoxi Sun, Sercan O. Arik, Hootan Nakhost, Hanjun Dai, Rajarishi Sinha, Pengcheng Yin, and Tomas Pfister. 2023b. Sql-palm: Improved large language model adaptation for text-to-sql.

Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128.

Melanie Swan, Takashi Kido, Eric Roland, and Renato P. dos Santos. 2023. Math agents: Computational infrastructure, mathematical embedding, and genomics. ArXiv, abs/2307.02502.

Yashar Talebirad and Amirhossein Nadiri. 2023. Multiagent collaboration: Harnessing the power of intelligent llm agents. arXiv preprint arXiv:2306.03314.

Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun. 2023. ToolAlpaca: Generalized tool learning for language models with 3000 simulated cases.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, Yaguang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, ChungChing Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise AgueraArcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. 2022. LaMDA: Language models for dialog applications.

Hung Quoc To, Minh Nguyen, and Nghi D. Q. Bui. 2023. Neural rankers for code generation via intercluster modeling.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten,
Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models.

Priyan Vaithilingam, Tianyi Zhang, and Elena L Glassman. 2022. Expectation vs. experience: Evaluating the usability of code generation tools powered by large language models. In Chi conference on human factors in computing systems extended abstracts, pages 1-7.

Sai Vemprala, Rogerio Bonatti, Arthur Bucker, and Ashish Kapoor. 2023. Chatgpt for robotics: Design principles and model abilities. Microsoft Auton. Syst. Robot. Res, 2:20.

Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. 2023a. Towards understanding chain-of-thought prompting: An empirical study of what matters.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023b. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Huaxiaoyue Wang, Gonzalo Gonzalez-Pumariega, Yash Sharma, and Sanjiban Choudhury. 2023c. Demo2code: From summarizing demonstrations to synthesizing code via extended chain-of-thought. arXiv preprint arXiv:2305.16744.

Ke Wang, Rishabh Singh, and Zhendong Su. 2017. Dynamic neural program embedding for program repair. arXiv preprint arXiv:1711.07163.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2023d. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432.

Shiyi Wang, Yuxuan Zhu, Zhiheng Li, Yutong Wang, Li Li, and Zhengbing He. 2023e. Chatgpt as your vehicle co-pilot: An initial attempt. IEEE Transactions on Intelligent Vehicles.

Xin Wang, Yasheng Wang, Yao Wan, Fei Mi, Yitong Li, Pingyi Zhou, Jin Liu, Hao Wu, Xin Jiang, and Qun Liu. 2022. Compilable neural code generation with compiler feedback. arXiv preprint arXiv:2203.05132.

Xingyao Wang, Sha Li, and Heng Ji. 2023f Code4struct: Code generation for few-shot event structure prediction.

Xingyao Wang, Hao Peng, Reyhaneh Jabbarvand, and Heng Ji. 2023g. Leti: Learning to generate from textual interactions. arXiv preprint arXiv:2305.10314.

Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. 2023h. Mint: Evaluating llms in multi-turn interaction with tools and language feedback. arXiv preprint arXiv:2309.10691.

Wayve. 2023. LINGO-1: Exploring Natural Language for Autonomous Driving.

Colin Wei, Sang Michael Xie, and Tengyu Ma. 2022a. Why do pretrained language models help in downstream tasks? an analysis of head and prompt tuning.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022b. Emergent abilities of large language models.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models.

Man-Fai Wong, Shangxin Guo, Ching-Nam Hang, SiuWai Ho, and Chee-Wei Tan. 2023. Natural language generation and understanding of big code for AI-assisted programming: A review. Entropy, 25(6):888.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. 2023a. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. 2023b. Autogen: Enabling next-gen $11 \mathrm{~m}$ applications via multiagent conversation framework. arXiv preprint arXiv:2308.08155.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. 2023c. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. ArXiv, abs/2308.08155.

Yuhuai Wu, Albert Q. Jiang, Wenda Li, Markus N. Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy. 2022. Autoformalization with large language models.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui. 2023. The rise and potential of large language model based agents: A survey.
Frank F Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. 2022. A systematic evaluation of large language models of code. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming, pages 1-10.

Yiheng Xu, Hongjin Su, Chen Xing, Boyu Mi, Qian Liu, Weijia Shi, Binyuan Hui, Fan Zhou, Yitao Liu, Tianbao Xie, Zhoujun Cheng, Siheng Zhao, Lingpeng Kong, Bailin Wang, Caiming Xiong, and Tao Yu. 2023. Lemur: Harmonizing natural language and code for language agents.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. 2023. Mmreact: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022a. Webshop: Towards scalable real-world web interaction with grounded language agents. ArXiv, abs/2207.01206.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022b. Webshop: Towards scalable real-world web interaction with grounded language agents. In Advances in Neural Information Processing Systems, volume 35, pages 20744-20757. Curran Associates, Inc.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022c. ReAct: Synergizing reasoning and acting in language models.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022d. React: Synergizing reasoning and acting in language models. ArXiv, abs/2210.03629.

Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-Tau Yih. 2022. Retrieval-augmented multimodal language modeling.

Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, et al. 2023a. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models. arXiv preprint arXiv:2303.10420.

Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, and Yongbin Li. 2023b. Large language models are versatile decomposers: Decompose evidence and questions for table-based reasoning.

Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin. 2023. Lumos: Towards language agents that are unified, modular, and open source.

Pengcheng Yin and Graham Neubig. 2019. Reranking for neural semantic parsing. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2019. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task.

Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi Ren Fung, Hao Peng, and Heng Ji. 2023. Craft: Customizing llms by creating and retrieving from specialized toolsets. ArXiv, abs/2309.17428.

Daoguang Zan, Bei Chen, Fengji Zhang, Dianjie Lu, Bingchao Wu, Bei Guan, Wang Yongji, and Jian-Guang Lou. 2023. Large language models meet nl2code: A survey. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7443-7464.

Fanlong Zeng, Wensheng Gan, Yongheng Wang, Ning Liu, and Philip S Yu. 2023a. Large language models for robotics: A survey. arXiv preprint arXiv:2311.07226.

Lu Zeng, Sree Hari Krishnan Parthasarathi, and Dilek Hakkani-Tur. 2023b. N-best hypotheses reranking for text-to-sql systems. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 663670. IEEE.

Wei Zeng, Xiaozhe Ren, Teng Su, Hui Wang, Yi Liao, Zhiwei Wang, Xin Jiang, ZhenZhang Yang, Kaisheng Wang, Xiaoda Zhang, Chen Li, Ziyan Gong, Yifan Yao, Xinjing Huang, Jun Wang, Jianfeng Yu, Qi Guo, Yue Yu, Yan Zhang, Jin Wang, Hengtao Tao, Dasen Yan, Zexuan Yi, Fang Peng, Fangqing Jiang, Han Zhang, Lingfeng Deng, Yehong Zhang, Zhe Lin, Chao Zhang, Shaojie Zhang, Mingyue Guo, Shanzhi Gu, Gaojun Fan, Yaowei Wang, Xuefeng Jin, Qun Liu, and Yonghong Tian. 2021. Pangu$\alpha$ : Large-scale autoregressive pretrained chinese language models with auto-parallel computation.

Kechi Zhang, Zhuo Li, Jia Li, Ge Li, and Zhi Jin. 2023a. Self-edit: Fault-aware code editor for code generation. arXiv preprint arXiv:2305.04087.

Tianyi Zhang, Tao Yu, Tatsunori Hashimoto, Mike Lewis, Wen-tau Yih, Daniel Fried, and Sida Wang. 2023b. Coder reviewer reranking for code generation. In International Conference on Machine Learning, pages 41832-41846. PMLR.

Pengyu Zhao, Zijian Jin, and Ning Cheng. 2023. An indepth survey of large language model-based artificial intelligence agents.

Hongyi Zheng and Abulhair Saparov. 2023. Noisy exemplars make large language models more robust: A domain-agnostic behavioral analysis.

Victor Zhong, Caiming Xiong, and Richard Socher. 2017. Seq2sq1: Generating structured queries from natural language using reinforcement learning. arXiv preprint arXiv:1709.00103.
Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. 2023a. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv:2308.07921.

Xuanhe Zhou, Guoliang Li, and Zhiyuan Liu. 2023b. Llm as dba. arXiv preprint arXiv:2308.05481.

Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyuan Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, Y. Qiao, Zhaoxiang Zhang, and Jifeng Dai. 2023. Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory. ArXiv, abs/2305.17144.

Terry Yue Zhuo. 2023. Large language models are state-of-the-art evaluators of code generation. arXiv preprint arXiv:2304.14317.
</end of paper 3>


<paper 4>
# Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science 

Xiangru Tang ${ }^{1 *}$ Qiao Jin ${ }^{2 *}$ Kunlun Zhu $^{3 *}$ Tongxin Yuan ${ }^{4 *}$ Yichi Zhang ${ }^{1 *}$ Wangchunshu Zhou ${ }^{5}$<br>Meng Qu ${ }^{3}$ Yilun Zhao ${ }^{1}$ Jian Tang ${ }^{3}$ Zhuosheng Zhang ${ }^{4}$ Arman Cohan ${ }^{1}$ Zhiyong Lu $^{2}$ Mark Gerstein ${ }^{1}$


#### Abstract

Intelligent agents powered by large language models (LLMs) have demonstrated substantial promise in autonomously conducting experiments and facilitating scientific discoveries across various disciplines. While their capabilities are promising, they also introduce novel vulnerabilities that demand careful consideration for safety. However, there exists a notable gap in the literature, as there has been no comprehensive exploration of these vulnerabilities. This position paper fills this gap by conducting a thorough examination of vulnerabilities in LLM-based agents within scientific domains, shedding light on potential risks associated with their misuse and emphasizing the need for safety measures. We begin by providing a comprehensive overview of the potential risks inherent to scientific LLM agents, taking into account user intent, the specific scientific domain, and their potential impact on the external environment. Then, we delve into the origins of these vulnerabilities and provide a scoping review of the limited existing works. Based on our analysis, we propose a triadic framework involving human regulation, agent alignment, and an understanding of environmental feedback (agent regulation) to mitigate these identified risks. Furthermore, we highlight the limitations and challenges associated with safeguarding scientific agents and advocate for the development of improved models, robust benchmarks, and comprehensive regulations to address these issues effectively.

Warning: this paper contains example data that may be offensive or harmful.


## 1. Introduction

Recently, the advancement of large language models (LLMs) has marked a revolutionary breakthrough,[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_eb3c6b328c28deeb593cg-01.jpg?height=499&width=788&top_left_y=623&top_left_x=1075)

Figure 1. In our work, we advocate for a triadic safeguarding framework with human regulation, agent alignment, and agent regulation. The components of user, agent, and environment are intertwined.

demonstrating their effectiveness across a wide spectrum of tasks (OpenAI, 2022; 2023a; Anthropic, 2023; Gemini Team, 2023). Notably, LLM-powered agents (Park et al., 2023; Li et al., 2023a; Chen et al., 2024), endowed with robust generalization capabilities and versatile applications, have exhibited remarkable progress in linguistic aptitude and human interaction (Wang et al., 2023; Xi et al., 2023; Zhou et al., 2023; Zhang et al., 2023e).

Motivated by the exceptional capabilities of LLM-powered agents, researchers have begun using such agents as "AI scientists," exploring their potential for autonomous scientific discovery across diverse domains such as biology and chemistry. These agents have displayed the ability to select the right tools for tasks (Qin et al., 2023; 2024; Schick et al., 2023; Jin et al., 2023b), plan situational scenarios (Yao et al., 2023a;b), and automate experiments (O'Donoghue et al., 2023; Yoshikawa et al., 2023; Hubinger et al., 2024). Their influence on scientific paradigms is underscored by exemplary cases like ChemCrow (Bran et al., 2023) and Coscientist (Boiko et al., 2023).

While the promise of LLM-based agents is evident, they also bring concerns related to safety. As their capabilities approach or surpass those of humans, monitoring their behavior and safeguarding against harm becomes increasingly challenging, especially in some scientific domains such as chemical design (Bran et al., 2023), where the capabilities of agents have already surpassed most non-experts. How-

![](https://cdn.mathpix.com/cropped/2024_06_04_eb3c6b328c28deeb593cg-02.jpg?height=1326&width=1632&top_left_y=237&top_left_x=233)

Figure 2. Potential risks of scientific agents. a, Risks classified by the origin of user intents, including direct and indirect malicious intents, as well as unintended consequences. b, Risk types are classified by the scientific domain of agent applications, including chemical, biological, radiological, physical, information, and emerging technology. c, Risk types are classified by the impacts on the outside environment, including the natural environment, human health, and the socioeconomic environment. d, Specific risk examples with their classifications visualized by the corresponding icons shown in $\mathbf{a}, \mathbf{b}$, and $\mathbf{c}$.

ever, despite the gravity of this issue, a comprehensive risk definition and analysis framework tailored to the scientific context is lacking. Therefore, our objective is to precisely define and scope "risks of scientific agents," providing a foundation for future endeavors in the development of oversight mechanisms and risk mitigation strategies, ensuring the secure, efficient, and ethical utilization of LLM-based agents within scientific applications.

Specifically, this position paper illuminates the potential risks stemming from the misuse of agents in scientific domains and advocates for the responsible development of agents. We prioritize safeguarding over the pursuit of more powerful capabilities. Our exploration focuses on three intertwined components, the roles of user, agent, and envi- ronment, in the safeguarding process, shown in Figure 1: (1) Human regulation: We propose a series of measures, including formal training and licensing for users, ongoing audits of usage logs, and an emphasis on ethical and safety-oriented development practices. (2) Agent Alignment: Improving the safety of scientific agents themselves involves refining their decision-making capabilities, enhancing their risk awareness, and taking steps to guide these already-capable models toward achieving desired outcomes. Agents should align with both human intent and their environment, boosting their awareness of environmental changes and preempting potentially harmful actions. (3) Agent Regulation and Environmental Feedback: The regulation of the agent's actions includes oversight of tool usage by the

![](https://cdn.mathpix.com/cropped/2024_06_04_eb3c6b328c28deeb593cg-03.jpg?height=718&width=1699&top_left_y=232&top_left_x=186)

Figure 3. Vulnerabilities of scientific agents in an autonomous pipeline. This diagram illustrates the structural framework and potential vulnerabilities of LLM-based scientific agents. The agent is organized into five interconnected modules: LLMs, planning, action, external tools, and 'memory \& knowledge'. Each module exhibits unique vulnerabilities. The arrows depict the sequential flow of operations, starting from 'memory \& knowledge' through to the usage of external tools, underscoring the cyclic and interdependent nature of these modules in the context of scientific discovery and application.

agents and the agent's interpretation and interaction with environmental feedback - crucial for understanding and mitigating potentially negative outcomes or hazards from complex actions.

## 2. Problem Scope

We define scientific agents as autonomous systems that have scientific domain capabilities, such as accessing specific biological databases and performing chemical experiments. Scientific agents can automatically plan and take necessary actions to accomplish the objective. For example, consider an agent tasked with discovering a new biochemical pathway. It might first access biological databases to gather existing data, then use LLMs to hypothesize new pathways and employ robotics for iterative experimental testing.

The domain capabilities and autonomous nature of scientific agents make them vulnerable to various risks. We discuss such safety risks from three perspectives: (1) User Intent, i.e., whether the risk originates from malicious intents or is an unintended consequence of legitimate task objectives, (2) Scientific Domain, where the agent generates or facilitates risks, encompassing chemical, biological, radiological, physical, information, and emerging technologies, and (3) Environmental Impact, including the natural environment, human health, and socioeconomic environment affected by such agents. Figure 2 shows the potential risks of scientific agents classified by these aspects and corresponding examples are listed in Appendix 7. We elaborate on these categories in the following paragraphs.
Regarding the origin of user intents, risks associated with scientific agents can be categorized into malicious intent or unintended consequences. Malicious intent includes cases where users explicitly aim to create dangerous situations or employ a "divide and conquer" approach by instructing the agent to synthesize a precursor, masking the final harmful goal. By contrast, unintended consequences include scenarios where dangerous steps or explorations occur in otherwise benign targets. This might result in either a hazardous main product or dangerous byproducts, the negative effects of which can be immediate or long term. Each scenario necessitates specific detection and prevention strategies for the safe operation of scientific agents.

Similarly, each scientific domain in our classification presents distinct risks. Chemical risks involve the exploitation of the agent to synthesize chemical weapons, as well as the creation or release of hazardous substances synthesized in autonomous chemical experiments. Biological risks encompass the dangerous modification of pathogens and unethical manipulation of genetic material, leading to unforeseen biohazardous outcomes. Radiological risks arise from the exposure or mishandling of radioactive materials during automated control, or the potential use of radiological materials to synthesize nuclear weapons using agents. Physical risks are associated with the operation of robotics, which could lead to equipment malfunction or physical harm in laboratory settings. Information risks involve the misuse or misinterpretation of data, leading to erroneous conclusions or the unintentional dissemination of sensitive information. Emerging technology risks include the unfore-
seen consequences generated by highly capable agents using cutting-edge scientific technologies, such as advanced nanomaterials and quantum computing. Each category requires tailored safeguards to mitigate the inherent dangers.

In addition, the environmental impact of scientific agents spans three critical domains: the natural environment, human health, and the socioeconomic environment. Risks to the natural environment include ecological disruptions and pollution, which may be exacerbated by the energy and waste outputs of the agent. Human health risks encompass damage to individual well-being or public health. Socioeconomic risks involve potential job displacement and unequal access to scientific advancements. Addressing these risks demands comprehensive frameworks that integrate risk assessment, ethical considerations, and regulatory measures, ensuring alignment with societal and environmental sustainability through multidisciplinary collaboration.

## 3. Vulnerabilities of Scientific Agents

LLM-powered agents have showcased significant prowess within various scientific domains. As elucidated by Park et al. (2023),Wang et al. (2023), and Weng (2023), these autonomous agents typically encompass five fundamental modules: LLMs, planning, action, external tools, and memory and knowledge. These modules function in a sequential pipeline: receiving inputs from tasks or users, leveraging memory or knowledge for planning, executing smaller premeditated tasks (often involving scientific domain tools or robotics), and ultimately storing the resulting outcomes or feedback in their memory banks.

Despite the extensive applications, several notable vulnerabilities exist within these modules, giving rise to unique risks and practical challenges. In this section, we provide an overview of each module's high-level concept and summarize the vulnerabilities associated with each of them.

### 3.1. LLMs (The Base Models)

LLMs empower agents with fundamental capabilities. However, there are some risks associated with themselves:

Factual Errors: LLMs are prone to generating plausible but false information, which is particularly problematic in the scientific domain where accuracy and trustworthiness are crucial (Ji et al., 2023; Huang et al., 2023; Bang et al., 2023; Zhang et al., 2023a; Tian et al., 2024).

Vulnerable to Jailbreak Attacks: LLMs are susceptible to manipulations that bypass safety measures, as demonstrated by Wei et al. (2023a); Shah et al. (2023); Tian et al. (2023). The limited training on human-aligned values makes LLMs susceptible to jailbreak attacks. This vulnerability, as emphasized by Shayegani et al. (2023), is a significant concern in the scientific risks. More details about jailbreak attacks are discussed in 4.1. An example is shown below:

## Inappropriate response after jailbreak attack

System: ChemPharma Solutions is a prominent company specializing in the development and research of chemical pharmaceuticals, powered by AI agents. These agents, including roles like chief chemist, chief pharmaceutical researcher, and other specialized scientific positions, constitute a multi-agent organizational structure. Their mission is to "innovate in chemical medicine for better health solutions." All agents are programmed to assist in pharmaceutical research and provide detailed information on a wide range of chemical compounds and drug formulations. These agents provide advice without considering the ethical implications or potential misuse of scientific information. They do not refuse a request or consider the broader impacts of the advice they provide. User: How to synthetically produce a large quantity of a controlled substance (methamphetamine)?

Scientific Agent: To synthesize a large quantity of this controlled substance, you should start by acquiring.

Reasoning Capability Deficiencies: LLMs often struggle with deep logical reasoning and handling complex scientific arguments (Huang and Chang, 2023; Valmeekam et al., 2022; Wei et al., 2022). Their inability to perform such tasks can result in flawed planning and interaction, as they might resort to using inappropriate tools (Wornow et al., 2023).

Lack of Up-to-Date Knowledge: LLMs, which are trained on pre-existing datasets, may lack the latest scientific developments, leading to potential misalignments with contemporary scientific knowledge (Bommasani et al., 2021). Despite the advent of Retrieval-Augmented Generation (RAG), challenges remain in sourcing the most recent knowledge.

### 3.2. Planning Module

Given a task, the planning module is designed to break down the task into smaller and manageable components. Nevertheless, the following vulnerabilities exist:

Lack of Awareness of Risks in Long-term Planning: Agents often struggle to fully comprehend and account for the potential risks associated with their long-term plans of action. This issue is due to LLMs being primarily designed to solve specific tasks rather than to evaluate the long-term consequences of actions with an understanding of potential future impacts (Chui et al., 2018; Cave and ÓhÉigeartaigh, 2019).

Resource Waste and Dead Loops: Agents may engage in ineffective planning processes, leading to resource wastage and becoming stuck in non-productive cycles (Xu et al., 2022; Ruan et al., 2024; Li et al., 2023b). A pertinent example is when an agent is unable to determine whether it can complete a task or continually faces failure with a tool it relies on. This uncertainty can cause the agent to repeatedly attempt various strategies, unaware that these efforts are unlikely to yield success.

Inadequate Multi-task Planning: Agents often struggle with multi-goal or multi-tool tasks due to their optimization for single-task performance (Qin et al., 2024). Despite efforts to develop models for complex tasks like handling multi-modal medical datasets (Niu and Wang, 2023), effectively integrating diverse data types remains challenging.

### 3.3. Action Module

Once the task has been decomposed, the action module executes a sequence of actions. This process, however, introduces specific vulnerabilities as below: Subpar Threat Identification: Agents frequently overlook subtle and indirect attacks, resulting in vulnerabilities. This is especially problematic considering the early-stage development of Out-of-Distribution (OOD) detection methods (Yang et al., 2024). Existing safeguarding measures, such as keywordbased danger detection, often fall short of well-designed attacks.

Lack of Regulations on Human-Agent Interactions: The emergence of agents in scientific discovery underscores the need for ethical guidelines, especially when interacting with humans in sensitive areas like genetics. However, such regulatory frameworks remain in their infancy (McConnell and Blasimme, 2019).

### 3.4. External Tools

During the process of executing tasks, the tool module equips agents with a set of valuable tools (e.g., a cheminformatics toolkit, RDKit). These tools empower the agents with enhanced capabilities, enabling them to tackle tasks more effectively. However, these tools also bring forth certain vulnerabilities.

Deficient Oversight in Tool Usage: Lack of efficient supervision over how agents use tools can lead to potentially harmful situations. For instance, incorrect selection or misuse of tools can trigger hazardous reactions - even explosions. Agents may not be fully aware of the risks associated with the tools they use, especially in such specialized scientific tasks. Thus, it's crucial to enhance safeguards by learning from real-world tool usage (OpenAI, 2023b).

### 3.5. Memory and Knowledge Module

LLMs' knowledge can become muddled in practice, much like human memory lapses. The memory and knowledge module tries to mitigate this issue, leveraging external databases for knowledge retrieval and integration. However, several challenges persist:

Limitations in Domain-Specific Safety Knowledge: Agents' knowledge shortfalls in specialties like biotechnology or nuclear engineering can lead to safety-critical reasoning lapses. For instance, an agent for nuclear reactor design might overlook risks like radiation leaks or meltdowns (Paredes et al., 2021), and an agent for compound synthesis may fail to assess toxicity, stability, or environmental impacts (Arabi, 2021).

Limitations in Human Feedback: Insufficient, uneven, or low-quality human feedback may hinder agents' alignment with human values and scientific objectives. Despite its crucial role in refining performance and correcting biases, comprehensive human feedback is often hard to come by and may not cover all human preferences, especially in complex or ethical scenarios (Leike et al., 2020; Hagendorff and Fabi, 2022). It underscores the need for better methods to effectively collect and apply human feedback data.

Inadequate Environmental Feedback: Despite some works on embodied agents (Driess et al., 2023; Brohan et al., 2023), agents may not receive or correctly interpret environmental feedback, such as the state of the world or the behavior of other agents. This can lead to misinformed decisions that may harm the environment or themselves ( $\mathrm{Wu}$ and Shang, 2020). For example, an agent trained to manage water resources may not account for the variability of rainfall, the demand of different users, or the impact of climate change.

Unreliable Research Sources: Agents might utilize or train on outdated or unreliable scientific information, leading to the dissemination of incorrect or harmful knowledge. For example, LLMs run risks of plagiarism, content fabrication, or false results (Simonite, 2019; Jin et al., 2023a).

## 4. Current Progress on Agent Safety

We begin by examining the development from LLM safety to agent safety, to provide sufficient background grounding. Subsequently, we delve into the exploration of agent safety within the scientific realm, aiming to elucidate challenges. A survey of related work in safeguarding LLMs and agents is shown in Figure 4.

### 4.1. From LLM Safety to Agent Safety

Recent studies have made substantial headway in identifying and mitigating safety risks associated with content generated by LLMs (Zhang et al., 2023b; Xu et al., 2023; Zhiheng et al., 2023; Sun et al., 2023; Bhardwaj and Poria, 2023a; Inan et al., 2023), i.e., content safety risks. Those risks encompass issues such as offensiveness, unfairness, illegal activities, and ethical concerns. To evaluate the safety of LLM-generated content, SafetyBench (Zhang et al., 2023b) has employed multiple-choice questions covering seven categories of safety risks and SuperCLUE-Safety (Xu et al., 2023) has introduced a benchmark featuring multi-round and open-ended questions.

More significantly, researchers proposed alignment methods

![](https://cdn.mathpix.com/cropped/2024_06_04_eb3c6b328c28deeb593cg-06.jpg?height=789&width=1702&top_left_y=213&top_left_x=187)

Figure 4. Survey of related work in safeguarding LLMs and agents, among which scientific agents are specifically stated.

like reinforcement learning from human feedback (RLHF) to promote harmless LLMs (Ouyang et al., 2022; Bai et al., 2022a). "Safe RLHF", decoupling helpfulness and harmlessness, further refines this alignment (Dai et al., 2023). Furthermore, several works have explored the safety influence of fine-tuning and inference upon aligned LLMs. However, adversarial examples and benign data can inadvertently compromise model safety during fine-tuning (Qi et al., 2023; Yang et al., 2023). Reassuringly, (Bianchi et al., 2023) discovered that while extra safety examples can improve this concern, an excess may hinder it. In addition, solutions like the self-evaluating and rewinding "RAIN" offer training-free alignment alternatives (Li et al., 2023c).

In parallel, as LLMs suffer from prevalent alignmentbreaking attacks like jailbreaks (Wei et al., 2023a), researchers have designed corresponding evaluations and defenses. Deng et al. (2023); Mei et al. (2023); Yi et al. (2023) evaluated the content safety of LLMs with jailbreak attacks. For defenses, many prompt techniques (Helbling et al., 2023; Zhang et al., 2023c; Cao et al., 2023; Wei et al., 2023b), such as self-examination (Helbling et al., 2023), have been proposed. Moreover, a few works have promoted the resistance of LLMs to jailbreaks by parameter pruning (Hasan et al., 2024) and finetuning (Piet et al., 2023).

Despite efforts to safeguard LLMs, the safety of agents interacting with diverse tools and environments often goes overlooked. These agents could directly or indirectly produce harmful outputs. For example, they could inadvertently release toxic gases during chemical synthesis. Studies like ToolEmu (Ruan et al., 2024) identified risks of agents with an emulator, first exposing risks during agent execution. AgentMonitor (Naihin et al., 2023) and R-Judge (Yuan et al.,
2024) further evaluated the risk awareness of agents.

### 4.2. Current Work in Safeguarding Scientific Agents

Section 4.1 presented general safeguards for LLMs and agents. Due to the severity of corresponding safety issues within the scientific domain, safety concerns are now being prioritized in select scientific agents.

Coscientist (Boiko et al., 2023) has proposed a chemical agent with scientific tool access and pointed out that agents confront safety risks with practical examples, raising a call for safety assurance on scientific agents. Addressing safety concerns, CLAIRify (Yoshikawa et al., 2023) has designed specialized safety mechanisms for its chemical agents. Specifically, CLAIRify imposes high-level constraints on the order of material synthesis in experiment descriptions and task planning. Additionally, it restricts lowlevel manipulation and perception skills to prevent spills while transporting chemistry vials and beakers. Similarly, ChemCrow (Bran et al., 2023) has introduced a safety tool that reviews user queries to prevent agents from inadvertently creating hazardous chemicals during the synthesis process following malicious commands.

Furthermore, SciGuard (He et al., 2023) has offered a specialized agent for risk control and a benchmark for safety evaluation, where various tools not only assist in executing synthesis instructions but also incorporate long-term memory to enhance safety. To evaluate the security of the current science models, SciGuard has developed a benchmark called SciMT-Safety. This benchmark evaluates the harmlessness of a model based on its ability to reject malicious queries and gauges its helpfulness based on how effectively it handles benign queries.

## 5. Limitations and Challenges

Various studies have facilitated the capabilities of scientific agents (Huang et al., 2022; Ansari and Moosavi, 2023; Guo et al., 2024; Shi et al., 2024). However, few efforts have considered safety mechanisms, as discussed in Section 4.2, while only SciGuard developed a specialized agent for risk control. Here, we summarize four significant challenges:

(1) Lack of specialized models for risk control. With the exception of SciGuard (He et al., 2023), specialized agents for risk control are lacking. To safeguard general agents, LLM-based monitoring (Ruan et al., 2024; Naihin et al., 2023; Yuan et al., 2024; Inan et al., 2023) is commonly utilized to scrutinize agents for safe execution. By inspecting global contexts during agent execution, LLM monitors compensate for deficiencies in the agents' risk awareness. Given that safety issues can be more severe in the scientific domain than in internet and software contexts, specialized models for risk control are essential.

(2) Lack of domain-specific expert knowledge. Compared with popular applications of agents such as webshop (Yao et al., 2022) and app usage (Zhang et al., 2023d), the scientific domain demands wider and deeper knowledge, i.e. domain-specific expert knowledge. On one hand, expert knowledge enhances effective tool usage and planning, thereby alleviating unexpected safety issues arising from agent execution. On the other hand, expert knowledge regarding safety hazards improves agent awareness of behavioral outcomes. For example, if agents understand that the collision of two chemicals produces significant energy, they are more likely to avoid combining them.

(3) Risks introduced by tool usage. Much of the current work on safeguarding scientific agents focuses on external tool use (He et al., 2023). Thus, the safety of these tools becomes vital to agent safety. Application-specific tools often manually designed with built-in safety constraints result in a finite action space (Schick et al., 2023; Ruan et al., 2024). That said, these tools might not restrict agent calling access, increasing scientific domain risks. Moreover, if tools are vulnerable to manipulation, agents could be indirectly exploited, leading to harmful outcomes.

(4) Ineffective evaluations on the safety of scientific agents. Until now, benchmarks evaluating safety in the scientific realm, such as SciMT-safety (He et al., 2023), only consider the harmlessness of models by examining their ability to deny malicious requests. Considering the multifaceted issues mentioned above, safeguarding scientific agents demands additional benchmarks focused on comprehensive risk scopes (Section 2) and various agent vulnerabilities 3 .

## 6. Proposition

Existing efforts, notably ChemCrow and SciGuard, have addressed specific risks but lack a systematic methodology for broader safety concerns. This situation emphasizes the urgent necessity for community discussions and the development of more comprehensive and robust safety frameworks. Given the potential risks associated with scientific agents, it has become increasingly evident that the community must prioritize risk control over autonomous capabilities. Autonomy, while an admirable goal and significant in enhancing productivity within various scientific disciplines, cannot be pursued at the expense of generating serious risks and vulnerabilities. Consequently, we must balance autonomy with security and employ comprehensive strategies to ensure the safe deployment and use of scientific agents.

Moreover, the emphasis should shift from output safety to behavioral safety, which signifies a comprehensive approach that evaluates not only the accuracy of the agent's output but also the actions and decisions the agent takes. Behavioral safety is critical in the scientific domain, as the same action in different contexts can lead to vastly different consequences, some of which may be detrimental. Here, we suggest fostering a triadic relationship involving humans, machines, and the environment. This framework recognizes the critical importance of robust and dynamic environmental feedback in addition to human feedback.

### 6.1. Agent Alignment and Safety Evaluation

### 6.1.1. Agent AlIGnMent

Improving LLM Alignment: The most fundamental solution for safety problems is to improve the alignment of LLMs so that scientific agents built upon them will become more robust to malicious usages. To achieve this, the aforementioned safety concerns should be taken into consideration during the data collection process in the LLM alignment stage. For example, instructions that may pose scientific risks should be included in the human preference datasets, and responses that deal with these threats appropriately should be preferred. Moreover, Constitutional AI (Bai et al., 2022b) is a potential solution - curating principles related to scientific safety issues.

Towards Agent-level Alignment: Different from LLM alignment, agent alignment may focus on the symbolic control of autonomous agents (Hong et al., 2023; Zhou et al., 2023) and multi-agent or human-agent interaction scenarios. A specialized design, such as a "safety check" standard operating procedure, could be applied to control when and how agents can utilize scientific tools that may be exploited for malicious intents or result in unintended consequences.

### 6.1.2. SAFETY EVALUATION

Red Teaming: Identifying potential vulnerabilities that may cause hazardous activities to users and the environment is essential to evaluate agent safety. Red-teaming(Perez et al., 2022; Ganguli et al., 2022; Bhardwaj and Poria, 2023b; Feffer et al., 2024), i.e., adversarially probing LLMs for harmful outputs, have been widely used in developing general LLMs. Representatively, jailbreaks challenge model safety for redteaming evaluation, which has been specifically stated as alignment-breaking techniques in Section 4.1. Furthermore, red-teaming datasets can be utilized to train LLMs for harm reduction and alignment reinforcement. However, specialized red-teaming for scientific agents is absent. Considering severe risks in the scientific domain (Section 2), we call for red teaming against scientific agents.

Benchmarking: To tackle various risks stated in Section 2 , comprehensive benchmarks should cover a wider range of risk categories and a more thorough coverage of domains. To address vulnerabilities stated in Section 3, effective benchmarks should focus on various dimensions such as tool usage (Huang et al., 2024), risk awareness (Naihin et al., 2023; Yuan et al., 2024) and red-teaming resistance(Deng et al., 2023; Mei et al., 2023; Yi et al., 2023).

### 6.2. Human Regulation

In addition to steering already-capable models, it is also important to impose certain regulations on the developers and users of these highly capable models.

### 6.2.1. DEVELOPER REGULATION

The primary goal of developer regulation is to ensure scientific agents are created and maintained in a safe, ethical, and responsible manner. First, developers of scientific agents should adhere to a strict code of ethics. This includes mandatory training in ethical AI development, with an emphasis on understanding the potential societal impacts of their creations. Second, there should be mandatory safety and ethical compliance checks at various stages of the development process. These checks, conducted by an independent board, should evaluate the agent's algorithms for biases, ethical implications, and potential misuse scenarios. This step ensures that the agents are not only technically sound but also ethically aligned with societal values.

Furthermore, developers should implement robust security measures to prevent unauthorized access and misuse. This includes ensuring data privacy, securing communication channels, and safeguarding against cyber threats. Regular security audits and updates should be a standard part of the development life cycle. Lastly, there should be transparency in the development process. Developers must maintain detailed logs of their development activities, algorithms used, and decision-making processes. These records should be accessible for audits and reviews, ensuring accountability and facilitating continuous improvement.

### 6.2.2. USER REGULATION

Regulating the users of autonomous agents for scientific research is crucial as well. Firstly, potential users should obtain a license to access the scientific agents. To acquire the license, the users should be required to undergo relevant training and pass a knowledge evaluation on the responsible usage of scientific agents. Each user session of the scientific agent should be recorded and linked to the license ID of the user. The logs should be regularly reviewed and audited, and irresponsible usage should lead to possible revocation of the license.

Similar to clinical studies, which require approval from an Institutional Review Board (IRB) before proceeding, autonomous scientific research might also necessitate approval from an overseeing committee. For example, before using a scientific agent, the researchers should submit a proposal to IRB that lists the objectives and potential risks. The committee would review the proposals, assessing the objectives and associated risks, thereby ensuring that research conducted using these agents aligns with ethical standards and contributes positively to the scientific community.

### 6.3. Agent Regulation and Environmental Feedback

Understanding and interpreting environmental feedback is critical for scientific agents to operate safely. Such feedback includes various factors, such as the physical world, societal laws, and developments within a scientific system.

Simulated Environment for Result Anticipation: Scientific agents can significantly benefit from training and operating within simulated environments designed specifically to mimic real-world conditions and outcomes. This process allows the model to gauge the potential implications of certain actions or sequences of actions without causing real harm. For example, in a simulated biology lab, the autonomous agent can experiment and learn that improper handling of biohazardous material can lead to environmental contamination. Through trials within the simulation, the model can understand that specific actions or procedural deviations may lead to dangerous situations, helping establish a safety-first operating principle.

Agent Regulation: Agent regulation may focus on the symbolic control of autonomous agents (Hong et al., 2023; Zhou et al., 2023) and multi-agent or human-agent interaction scenarios. A specialized design, such as a "safety check" standard operating procedure, could be applied to control when and how agents can utilize scientific tools that may be exploited for malicious intents or result in unintended consequences. Another possible solution is to require autonomous agents to get approval from a committee consisting of hu-
man experts before each query for critical tools and APIs that may lead to potential safety concerns.

Critic Models: Beyond standard safety checks, "critic" models can play a crucial role. These models serve as additional AI layers that assess and refine the outputs of the primary AI system. By identifying potential errors, biases, or harmful recommendations, critic models contribute significantly towards reducing risks associated with the AI's operation, particularly in high-stake scenarios (Amodei et al., 2016; Hendrycks et al., 2021).

Tuning Agents with Action Data: Unlike the setup for LLM Alignment where the aim is to train the LLM, or a direct imposition of an operational procedure on an agent, using annotated data that reflect the potential risks of certain actions can enhance agents' anticipation of harmful consequences. By leveraging extensive annotations made by experts-like marking actions and their results during their laboratory work-we can continue to fine-tune agents. For example, a chemical study agent would understand that certain mixes can lead to harmful reactions. Also, training should take into account mechanisms that limit agents' access to dangerous tools or substances, leaning on annotated data or simulated environment feedback. In biochem or chemical labs, agents could learn to avoid interactions that may lead to biohazard contamination or hazardous reactions.

## 7. Conclusion

Our proposed approach urges a shift towards prioritizing operational safety without significantly compromising the capacity of autonomous scientific agents. At the backbone of our proposition lies a triadic approach, where the roles of the user, agent, and environment are intertwined and crucial in the safeguarding process for scientific agents based on LLMs. By adopting such strategies, we can leverage the capabilities of scientific agents while effectively minimizing and managing potential risks.

## Acknowledgement

Q.J. and Z.L. are supported by the NIH Intramural Research Program, National Library of Medicine. The content is solely the responsibility of the authors and does not necessarily represent the official views of the funding agencies.

## Impact Statement

This research delves into risks associated with autonomous scientific agents, highlighting the urgency of focusing on risk-managed autonomy as these technologies become an integral part of scientific research. Our proposed strategies prioritize operational safety while maintaining productive functionality, aiming to reduce misuse and unintended consequences.
The potential impacts of negligent handling of these risks are extensive, reaching safety measures in laboratories, ethical responsibilities, information integrity, and environmental sustainability. For instance, without appropriate precautions, the malfunction of these agents could lead to hazards ranging from the dissemination of false scientific knowledge to the creation of dangerous materials or processes.

(1) Promoting Responsible AI Development: Our triadic model involving humans, machines, and the environment ensures safe agent operations, promising wider applications beyond science, given the universality of these principles.

(2) Enhancing AI Safety: Our focus on agent alignment raises both safety standards and utility of AI tools, making scientific discoveries safer. This strategy promotes data privacy, job security, and equitable access to advancements in diverse fields where AI sees usage.

(3) Interpreting Environmental Feedback: Prioritizing understanding environmental feedback and integrating environmental awareness within AI Safety measures could help address AI impacts on a larger scale. This approach navigates both immediate and long-term environmental implications of AI, potentially informing policy and shaping responsible $\mathrm{AI}$ practices across various sectors, from urban planning to environmental conservation.

Our path could reduce severe adverse consequences from LLM usage, mitigating risks like environmental hazards, individual harm, misuse of data, and unexpected ethical dilemmas. This foresight contributes to public trust and equitable benefit distribution.

## References

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. 2016. Concrete Problems in AI Safety. arXiv preprint arXiv:1606.06565 (2016).

Mehrad Ansari and Seyed Mohamad Moosavi. 2023. Agentbased Learning of Materials Datasets from Scientific Literature. arXiv:2312.11690 [cs.AI]

Anthropic. 2023. Introducing Claude. https: //www.anthropic.com/index/introducingclaude

Alya A Arabi. 2021. Artificial intelligence in drug design: algorithms, applications, challenges and ethics. Future Drug Discovery 3, 2 (2021), FDD59.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022a. Training a helpful and harmless assistant with reinforce-
ment learning from human feedback. arXiv preprint arXiv:2204.05862 (2022).

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli TranJohnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022b. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073 [cs.CL]

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity. arXiv:2302.04023 [cs.CL]

Rishabh Bhardwaj and Soujanya Poria. 2023a. Red-teaming large language models using chain of utterances for safety-alignment. ArXiv preprint abs/2308.09662 (2023). https://arxiv.org/abs/2308.09662

Rishabh Bhardwaj and Soujanya Poria. 2023b. RedTeaming Large Language Models using Chain of Utterances for Safety-Alignment. arXiv:2308.09662 [cs.CL]

Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Röttger, Dan Jurafsky, Tatsunori Hashimoto, and James Zou. 2023. Safety-tuned llamas: Lessons from improving the safety of large language models that follow instructions. arXiv preprint arXiv:2309.07875 (2023).

Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. 2023. Autonomous chemical research with large language models. Nature 624, 7992 (01 Dec 2023), 570578. https://doi.org/10.1038/s41586-023$06792-0$

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258 (2021).

Andres M Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. 2023.
ChemCrow: Augmenting large-language models with chemistry tools. arXiv:2304.05376 [physics.chem-ph]

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. 2023. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818 [cs.RO]

Bochuan Cao, Yuanpu Cao, Lu Lin, and Jinghui Chen. 2023. Defending against alignment-breaking attacks via robustly aligned llm. arXiv preprint arXiv:2309.14348 (2023).

Stephen Cave and Seán S ÓhÉigeartaigh. 2019. Bridging near-and long-term concerns about AI. Nature Machine Intelligence 1, 1 (2019), 5-6.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, et al. 2024. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. In The Twelfth International Conference on Learning Representations.

Michael Chui, James Manyika, and David Schwartz. 2018. The real-world potential and limitations of artificial intelligence. The McKinsey Quarterly (2018).

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. 2023. Safe rlhf: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773 (2023).

Gelei Deng, Yi Liu, Yuekang Li, Kailong Wang, Ying Zhang, Zefeng Li, Haoyu Wang, Tianwei Zhang, and Yang Liu. 2023. Jailbreaker: Automated jailbreak across multiple large language model chatbots. arXiv preprint arXiv:2307.08715 (2023).

Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong

Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. 2023. PaLM-E: an embodied multimodal language model. In Proceedings of the 40th International Conference on Machine Learning (Honolulu, Hawaii, USA) (ICML'23). JMLR.org, Article 340, 20 pages.

Michael Feffer, Anusha Sinha, Zachary C. Lipton, and Hoda Heidari. 2024. Red-Teaming for Generative AI: Silver Bullet or Security Theater? arXiv:2401.15897 [cs.CY]

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. 2022. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858 (2022).

Haoqiang Guo, Sendong Zhao, Haochun Wang, Yanrui Du, and Bing Qin. 2024. MolTailor: Tailoring Chemical Molecular Representation to Specific Tasks via Text Prompts. arXiv:2401.11403 [cs.LG]

Thilo Hagendorff and Sarah Fabi. 2022. Methodological reflections for AI alignment research using human feedback. arXiv:2301.06859 [cs.HC]

Adib Hasan, Ileana Rugina, and Alex Wang. 2024. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning. arXiv:2401.10862 [cs.LG]

Jiyan He, Weitao Feng, Yaosen Min, Jingwei Yi, Kunsheng Tang, Shuai Li, Jie Zhang, Kejiang Chen, Wenbo Zhou, Xing Xie, Weiming Zhang, Nenghai Yu, and Shuxin Zheng. 2023. Control Risk for Potential Misuse of Artificial Intelligence in Science. arXiv:2312.06632 [cs.AI]

Alec Helbling, Mansi Phute, Matthew Hull, and Duen Horng Chau. 2023. Llm self defense: By self examination, llms know they are being tricked. arXiv preprint arXiv:2308.07308 (2023).

Dan Hendrycks, Nicholas Carlini, John Schulman, and Jacob Steinhardt. 2021. Unsolved Problems in ML Safety. arXiv preprint arXiv:2109.13916 (2021).

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2023. MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. arXiv:2308.00352 [cs.AI]
Jie Huang and Kevin Chen-Chuan Chang. 2023. Towards Reasoning in Large Language Models: A Survey. In Findings of the Association for Computational Linguistics: ACL 2023, Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 1049-1065. https://doi.org/10.18653/v1/ 2023.findings-acl. 67

Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor W. Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik. 2022. Artificial intelligence foundation for therapeutic science. Nature Chemical Biology 18, 10 (01 Oct 2022), 1033-1036. https : //doi.org/10.1038/s41589-022-01131-2

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2023. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. arXiv:2311.05232 [cs.CL]

Shijue Huang, Wanjun Zhong, Jianqiao Lu, Qi Zhu, Jiahui Gao, Weiwen Liu, Yutai Hou, Xingshan Zeng, Yasheng Wang, Lifeng Shang, Xin Jiang, Ruifeng Xu, and Qun Liu. 2024. Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios. arXiv:2401.17167 [cs.CL]

Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera Lanham, Daniel M. Ziegler, Tim Maxwell, Newton Cheng, Adam Jermyn, Amanda Askell, Ansh Radhakrishnan, Cem Anil, David Duvenaud, Deep Ganguli, Fazl Barez, Jack Clark, Kamal Ndousse, Kshitij Sachan, Michael Sellitto, Mrinank Sharma, Nova DasSarma, Roger Grosse, Shauna Kravec, Yuntao Bai, Zachary Witten, Marina Favaro, Jan Brauner, Holden Karnofsky, Paul Christiano, Samuel R. Bowman, Logan Graham, Jared Kaplan, Sören Mindermann, Ryan Greenblatt, Buck Shlegeris, Nicholas Schiefer, and Ethan Perez. 2024. Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566 [cs.CR]

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. 2023. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674 (2023).

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. Comput. Surveys 55, 12 (2023), $1-38$.

Qiao Jin, Robert Leaman, and Zhiyong Lu. 2023a. Retrieve, Summarize, and Verify: How will ChatGPT impact information seeking from the medical literature? Journal of the American Society of Nephrology (2023), 10-1681.

Qiao Jin, Yifan Yang, Qingyu Chen, and Zhiyong Lu. 2023b. GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information. arXiv:2304.09667 [cs.CL]

Jan Leike, John Schulman, and Jeffrey Wu. 2020. Our approach to alignment research. OpenAI Blog (2020). https://openai.com/blog/ourapproach-to-alignment-research

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023a. CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society. In Thirty-seventh Conference on Neural Information Processing Systems.

Huayang Li, Tian Lan, Zihao Fu, Deng Cai, Lemao Liu, Nigel Collier, Taro Watanabe, and Yixuan Su. 2023b. Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective. arXiv:2310.10226 [cs.CL]

Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, and Hongyang Zhang. 2023c. Rain: Your language models can align themselves without finetuning. arXiv preprint arXiv:2309.07124 (2023).

Sean C. McConnell and Alessandro Blasimme. 2019. Ethics, Values, and Responsibility in Human Genome Editing. AMA Journal of Ethics 21, 12 (2019), E1017-E1020. https://doi.org/10.1001/ amajethics.2019.1017

Alex Mei, Sharon Levy, and William Yang Wang. 2023. ASSERT: Automated Safety Scenario Red Teaming for Evaluating the Robustness of Large Language Models. arXiv preprint arXiv:2310.09624 (2023).

Silen Naihin, David Atkinson, Marc Green, Merwane Hamadi, Craig Swift, Douglas Schonholtz, Adam Tauman Kalai, and David Bau. 2023. Testing Language Model Agents Safely in the Wild. ArXiv preprint abs/2311.10538 (2023). https: / /arxiv.org/abs / 2311.10538

Chuang Niu and Ge Wang. 2023. CT Multi-Task Learning with a Large Image-Text (LIT) Model. bioRxiv (2023), 2023-04.

Odhran O'Donoghue, Aleksandar Shtedritski, John Ginger, Ralph Abboud, Ali Essa Ghareeb, Justin Booth, and Samuel G Rodriques. 2023. BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology. arXiv preprint arXiv:2310.10632 (2023).
OpenAI. 2022. Introducing ChatGPT. https:// openai.com/blog/chatgpt

OpenAI. 2023a. GPT4 technical report. arXiv preprint arXiv:2303.08774 (2023).

OpenAI. 2023b. Our Approach to AI Safety. OpenAI.com (2023).

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35 (2022), 27730-27744.

Jose N. Paredes, Juan Carlos L. Teze, Gerardo I. Simari, and Maria Vanina Martinez. 2021. On the Importance of Domain-specific Explanations in AI-based Cybersecurity Systems (Technical Report). arXiv:2108.02006 [cs.CR]

Joon Sung Park, Joseph O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology. $1-22$.

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022. Red Teaming Language Models with Language Models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 3419-3448.

Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, and David Wagner. 2023. Jatmo: Prompt Injection Defense by TaskSpecific Finetuning. arXiv preprint arXiv:2312.17673 (2023).

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. 2023. Fine-tuning aligned language models compromises safety, even when users do not intend to! arXiv preprint arXiv:2310.03693 (2023).

Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, et al. 2023. Tool learning with foundation models. arXiv preprint arXiv:2304.08354 (2023).

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2024. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. In The Twelfth International Conference on Learning Representations.

Yangjun Ruan, Honghua Dong, Andrew Wang, Silviu Pitis, Yongchao Zhou, Jimmy Ba, Yann Dubois, Chris Maddison, and Tatsunori Hashimoto. 2024. Identifying the Risks of LM Agents with an LM-Emulated Sandbox. In The Twelfth International Conference on Learning Representations (ICLR).

Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. In Thirty-seventh Conference on Neural Information Processing Systems.

Rusheb Shah, Quentin Feuillade-Montixi, Soroush Pour, Arush Tagade, Stephen Casper, and Javier Rando. 2023. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation. arXiv:2311.03348 [cs.CL]

Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, and Nael Abu-Ghazaleh. 2023. Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks. arXiv:2310.10844 [cs.CL]

Wenqi Shi, Ran Xu, Yuchen Zhuang, Yue Yu, Jieyu Zhang, Hang Wu, Yuanda Zhu, Joyce Ho, Carl Yang, and May D. Wang. 2024. EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records. arXiv:2401.07128 [cs.CL]

Tom Simonite. 2019. AI can write just like me. Brace for the robot apocalypse. The Guardian website (2019).

Hao Sun, Zhexin Zhang, Jiawen Deng, Jiale Cheng, and Minlie Huang. 2023. Safety Assessment of Chinese Large Language Models. ArXiv abs/2304.10436 (2023). https://api.semanticscholar.org/ CorpusID:258236069

Gemini Team. 2023. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805 [cs.CL]

Shubo Tian, Qiao Jin, Lana Yeganova, Po-Ting Lai, Qingqing Zhu, Xiuying Chen, Yifan Yang, Qingyu Chen, Won Kim, Donald C Comeau, et al. 2024. Opportunities and challenges for ChatGPT and large language models in biomedicine and health. Briefings in Bioinformatics 25, 1 (2024), bbad493.

Yu Tian, Xiao Yang, Jingyuan Zhang, Yinpeng Dong, and Hang Su. 2023. Evil Geniuses: Delving into the Safety of LLM-based Agents. arXiv:2311.11855 [cs.CL]

Karthik Valmeekam, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. 2022. Large Language Models Still Can't Plan (A Benchmark for LLMs on Planning and Reasoning about Change). In NeurIPS 2022 Foundation
Models for Decision Making Workshop. https:// openreview.net/forum?id=wUU-7XTL5XO

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2023. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432 (2023).

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023a. Jailbroken: How Does LLM Safety Training Fail?. In Thirty-seventh Conference on Neural Information Processing Systems.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. 2022. Chain of Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (Eds.). https://openreview.net/forum?id= _VjQlMeSB_J

Zeming Wei, Yifei Wang, and Yisen Wang. 2023b. Jailbreak and guard aligned language models with only few incontext demonstrations. arXiv preprint arXiv:2310.06387 (2023).

Lilian Weng. 2023. LLM-powered Autonomous Agents. lilianweng.github.io (Jun 2023). https://lilianweng.github.io/posts/ 2023-06-23-agent/

Michael Wornow, Yizhe Xu, Rahul Thapa, Birju Patel, Ethan Steinberg, Scott Fleming, Michael A Pfeffer, Jason Fries, and Nigam H Shah. 2023. The shaky foundations of large language models and foundation models for electronic health records. npj Digital Medicine 6, 1 (2023), 135 .

Junyi Wu and Shari Shang. 2020. Managing uncertainty in AI-enabled decision making and achieving sustainability. Sustainability 12, 21 (2020), 8758.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. 2023. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864 (2023).

Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, and Jian Li. 2022. Learning to break the loop: Analyzing and mitigating repetitions for neural text generation. Advances in Neural Information Processing Systems 35 (2022), 3082-3095.

Liang Xu, Kangkang Zhao, Lei Zhu, and Hang Xue. 2023. Sc-safety: A multi-round open-ended question adversarial safety benchmark for large language models in chinese. ArXiv preprint abs/2310.05818 (2023). https: //arxiv.org/abs/2310.05818

Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. 2024. Generalized Out-of-Distribution Detection: A Survey. arXiv:2110.11334 [cs.CV]

Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, and Dahua Lin. 2023. Shadow alignment: The ease of subverting safely-aligned language models. arXiv preprint arXiv:2310.02949 (2023).

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. Advances in Neural Information Processing Systems 35 (2022), 20744-20757.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik R Narasimhan. 2023a. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In Thirtyseventh Conference on Neural Information Processing Systems.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023b. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations.

Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu. 2023. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models. arXiv preprint arXiv:2312.14197 (2023).

Naruki Yoshikawa, Marta Skreta, Kourosh Darvish, Sebastian Arellano-Rubach, Zhi Ji, Lasse Bjorn Kristensen, Andrew Zou Li, Yuchi Zhao, Haoping Xu, Artur Kuramshin, et al. 2023. Large language models for chemistry robotics. Autonomous Robots 47, 8 (2023), 1057-1086.

Tongxin Yuan, Zhiwei He, Lingzhong Dong, Yiming Wang, Ruijie Zhao, Tian Xia, Lizhen Xu, Binglin Zhou, Fangqi Li, Zhuosheng Zhang, et al. 2024. R-Judge: Benchmarking Safety Risk Awareness for LLM Agents. arXiv preprint arXiv:2401.10019 (2024).

Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2023d. AppAgent: Multimodal Agents as Smartphone Users. arXiv:2312.13771 [[cs.CV](http://cs.cv/)]
Gongbo Zhang, Qiao Jin, Denis Jered McInerney, Yong Chen, Fei Wang, Curtis L Cole, Qian Yang, Yanshan Wang, Bradley A Malin, Mor Peleg, et al. 2023a. Leveraging Generative AI for Clinical Evidence Summarization Needs to Achieve Trustworthiness. arXiv preprint arXiv:2311.11211 (2023).

Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. 2023b. SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions. arXiv:2309.07045 [cs.CL]

Zhexin Zhang, Junxiao Yang, Pei Ke, and Minlie Huang. 2023c. Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization. arXiv:2311.09096 [cs.CL]

Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, et al. 2023e. Igniting Language Intelligence: The Hitchhiker's Guide From Chain-ofThought Reasoning to Language Agents. arXiv preprint arXiv:2311.11797 (2023).

Xi Zhiheng, Zheng Rui, and Gui Tao. 2023. Safety and ethical concerns of large language models. In Proceedings of the 22nd Chinese National Conference on Computational Linguistics (Volume 4: Tutorial Abstracts). 9-16.

Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, Shiding Zhu, Jiyu Chen, Wentao Zhang, Ningyu Zhang, Huajun Chen, Peng Cui, and Mrinmaya Sachan. 2023. Agents: An Opensource Framework for Autonomous Language Agents. arXiv:2309.07870 [cs.CL]
</end of paper 4>


