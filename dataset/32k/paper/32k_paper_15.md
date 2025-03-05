<paper 0>
# TravelPlanner: A Benchmark for Real-World Planning with Language Agents 

Jian Xie ${ }^{\boldsymbol{\omega}}$ * Kai Zhang ${ }^{\boldsymbol{*}}$ * Jiangjie Chen ${ }^{\infty}$ Tinghui Zhu ${ }^{\phi}$ Renze Lou ${ }^{\varnothing}$<br>Yuandong Tian ${ }^{\diamond}$ Yanghua Xiao ${ }^{\uparrow}$ Yu Su ${ }^{\star}$<br>${ }^{\star}$ Fudan University ${ }^{\bullet}$ The Ohio State University<br>${ }^{\bigcirc}$ The Pennsylvania State University $\diamond$ Meta AI<br>jianxie22@m.fudan.edu.cn, shawyh@fudan.edu.cn, \{zhang.13253, su.809\}@osu.edu<br>https://osu-nlp-group.github.io/TravelPlanner


#### Abstract

Planning has been part of the core pursuit for artificial intelligence since its conception, but earlier AI agents mostly focused on constrained settings because many of the cognitive substrates necessary for human-level planning have been lacking. Recently, language agents powered by large language models (LLMs) have shown interesting capabilities such as tool use and reasoning. Are these language agents capable of planning in more complex settings that are out of the reach of prior AI agents? To advance this investigation, we propose TravelPlanner, a new planning benchmark that focuses on travel planning, a common real-world planning scenario. It provides a rich sandbox environment, various tools for accessing nearly four million data records, and 1,225 meticulously curated planning intents and reference plans. Comprehensive evaluations show that the current language agents are not yet capable of handling such complex planning tasks-even GPT-4 only achieves a success rate of $0.6 \%$. Language agents struggle to stay on task, use the right tools to collect information, or keep track of multiple constraints. However, we note that the mere possibility for language agents to tackle such a complex problem is in itself non-trivial progress. TravelPlanner provides a challenging yet meaningful testbed for future language agents.


## 1. Introduction

Planning is a hallmark of human intelligence. It is an evolutionary feat built upon numerous other capacities: using various tools to iteratively collect information and make de-[^0]

cisions, recording intermediate plans (in working memory or on a physical device) for deliberation, and exploring alternative plans by running simulations, which in turn depends on a world model (Mattar \& Lengyel, 2022; Ho et al., 2022). For decades, researchers have been attempting to develop AI agents to mimic humans' planning capability (Russell \& Norvig, 2010; Georgievski \& Aiello, 2015; Karpas \& Magazzeni, 2020), but often in constrained settings (Campbell et al., 2002; Silver et al., 2016; 2017) because many of the cognitive substrates necessary for human-level planning have been lacking. AI agents that can work robustly in the largely unconstrained settings in which humans operate remain a distant goal.

The advent of large language models (LLMs; OpenAI (2022; 2023); Touvron et al. (2023a;b); Jiang et al. (2023)) brings new light to this classic problem. A new generation of language agents (Su, 2023; Sumers et al., 2023; Xie et al., 2023) powered by LLMs has emerged, characterized by their capability of using language as a vehicle for thought and communication. These agents have shown interesting capabilities, such as tool use (Schick et al., 2023; Patil et al., 2023; Qin et al., 2024) and various forms of reasoning (Wei et al., 2022; Yao et al., 2022; Lewkowycz et al., 2022), potentially fulfilling the role of some of the cognitive substrates that were lacking in earlier AI agents. Researchers have therefore investigated their potential in an array of planning tasks ranging from classic planning settings like Blocksworld (Valmeekam et al., 2023) to embodied agents (Huang et al., 2022; Ichter et al., 2022; Song et al., 2023; Wang et al., 2023) and web agents (Deng et al., 2023; Zhou et al., 2024). However, the planning settings in existing work still largely follow the conventional setting that focuses on single-objective optimization with fixed ground truths. An agent is tasked with predicting from a pre-defined set of actions, just now made by an LLM-powered agent.

Are language agents capable of planning in more complex yet realistic settings, closer to those in which humans operate? To advance this investigation, we propose TravelPlanner, a new planning benchmark that focuses on a common

![](https://cdn.mathpix.com/cropped/2024_06_04_a564c634bf6e44ec5b6ag-02.jpg?height=865&width=1626&top_left_y=234&top_left_x=217)

Figure 1. Overview of TravelPlanner. Given a query, language agents are tasked with employing various search tools to gather information. Based on the collected information, language agents are expected to deliver a plan that not only satisfies the user's needs specified in the query but also adheres to commonsense constraints.

real-world planning scenario-travel planning. This is a challenging, time-consuming task even for humans (but most people can do it successfully, with the right tools and enough time): 1) Planning a multi-day itinerary is inherently long-horizon, involving a large number of interdependent decisions on places, lodging, transportation, dining, etc. 2) Travel planning involves many constraints, ranging from explicit constraints such as budget and various user needs to implicit commonsense constraints, e.g., people cannot teletransport to another city without using some means of transportation. 3) Travel planning requires strong agency to proactively acquire necessary information using various tools (e.g., to search flights and restaurants) from the partially observable environment and deliberate over the collected information to further the planning while being mindful of all the explicit and implicit constraints. Planning tasks of such complexity are out of the reach of prior AI agents (Russell \& Norvig, 2010).

TravelPlanner provides a rich sandbox environment with around four million data entries crawled from the Internet that can be accessed via six tools. We also meticulously curate 1,225 diverse user queries (along with their reference plans), each imposing a different combination of constraints. A representative example is illustrated in Figure 1.

We comprehensively evaluate five LLMs, such as GPT4 (OpenAI, 2023), Gemini (G Team et al., 2023), and Mixtral (Jiang et al., 2024), and four planning strategies, such as ReAct (Yao et al., 2022) and Reflexion (Shinn et al., 2023), on their capability of delivering complete plans and following constraints. The main findings are as follows:

- State-of-the-art LLMs cannot handle complex planning tasks like those in TravelPlanner. GPT-4 successfully produces a plan that meets all the constraints for a few tasks (0.6\%), while all other LLMs fail to complete any tasks.
- Existing planning strategies such as ReAct and Reflexion, which may be effective for simpler planning settings, are insufficient for the multi-constraint tasks in TravelPlanner. They often fail to convert their reasoning into the right actions correctly and keep track of global or multiple constraints. Language agents need more sophisticated planning strategies to approach human-level planning.
- Further analyses reveal many common failure modes of existing language agents, such as argument errors in tool use, being trapped in dead loops, and hallucinations.

Although most of our findings lean negatively toward the current language agents, we should note that the mere possibility for an artificial agent to tackle such a complex task is non-trivial progress in itself. TravelPlanner provides a challenging yet meaningful testbed for future agents to hillclimb toward human-level planning in complex settings.

Finally, a silver lining: while our well-trained human annotators averagely take 12 minutes to manually annotate a plan, a language agent can produce a plan in just 1-2 minutes automatically. Perhaps one day, language agents will become capable enough to help automate away many of such tedious tasks for us.

## 2. Related Work

### 2.1. Large Language Model based Agents

Empowered by large language models (LLMs), language agents have the capability to decompose complex tasks and arrive at solutions through a series of reasoned actions. Notable examples such as AutoGPT (AutoGPT, 2023), BabyAGI (Nakajima, 2023), and HuggingGPT (Shen et al., 2023) have illuminated the community with their impressive abilities. Current LLM-powered language agents, equipped with Memory, Tool-use, and Planning modules, have seen a substantial improvement in their general abilities (Weng, 2023). Memory in language agents refers to their ability to acquire and process information. It is divided into two types: long-term memory, which is the parametric memory inherent in LLMs, and short-term memory, also known as in-context learning (Brown et al., 2020) or working memory. Techniques like memory summarization (Chen et al., 2023; Zhou et al., 2023; Liang et al., 2023) and retrieval (Andreas, 2022; Park et al., 2023; Zhong et al., 2023) are widely employed to enhance the memory capabilities of language agents. Moreover, by interacting with external tools, language agents expand their potential capabilities significantly. This tool-augmentation paradigm has been validated as effective in previous work (Nakano et al., 2021; Lu et al., 2023; Ge et al., 2023; Xie et al., 2023). We further discuss the planning module in Section 2.2.

### 2.2. Planning

Planning, a hallmark of human intelligence, entails a sequence of actions that involve decomposing tasks, searching for solutions, and making final decisions (Hayes-Roth \& Hayes-Roth, 1979; Grafman et al., 2004; Su, 2023). This skill is crucial for achieving human-level intelligence and has been widely studied in areas such as robotics (McDermott, 1992; Alterovitz et al., 2016) and transportation scheduling (Cross \& Estrada, 1994; Pinedo, 2005). The emergence of language agents powered by LLMs has further intensified discussions around their planning capabilities (Liu et al., 2023a; Valmeekam et al., 2023). Previous research has demonstrated that language agents can effectively decompose tasks and engage in step-by-step reasoning, leading to significant improvements (Wei et al., 2022; Yuan et al., 2023; Zheng et al., 2024). Furthermore, to optimize solution searches in fewer steps, classical data structures like trees and graphs have been employed in prior studies (Yao et al., 2023; Besta et al., 2023), enhancing the planning capabilities of language agents. In addition, methods involving feedback from the environment (Yao et al., 2022; Shinn et al., 2023) have also been shown to be beneficial. However, while these planning abilities have shown promise in specific tasks, the effectiveness of these planning strategies in scenarios with multiple constraints remains uncertain.

### 2.3. Evaluation of Language Agents

Previous studies typically assess LLM-powered language agents in focused domains: arithmetic reasoning targeting correct solutions (Roy \& Roth, 2015; Cobbe et al., 2021; Patel et al., 2021); tool-use evaluating agents' proficiency in employing tools and reporting results (Li et al., 2023; Xu et al., 2023; Zhuang et al., 2023); and web navigation, testing agents' ability to locate specific websites (Deng et al., 2023; Zhou et al., 2024; Liu et al., 2024). However, the complexity of the real-world implies that previous evaluation methods, which focus on single objective and fixed ground truths, may fall short of capturing the full scope of agents' capabilities. To address this, we introduce TravelPlanner for comprehensive evaluations, assessing whether language agents can generate feasible solutions facing various objectives, referred to as constraints in this paper.

## 3. TravelPlanner

### 3.1. Overview

We introduce TravelPlanner, a benchmark crafted for evaluating language agents in tool-use and complex planning within multiple constraints. Grounding to travel planning, a real-world use-case that naturally includes diverse constraints such as user needs and commonsense constraints, TravelPlanner evaluates whether agents can develop flexible travel plans by collecting information via diverse tools and making decisions while satisfying the constraints.

TravelPlanner comprises 1,225 queries in total. The queries in TravelPlanner are divided into nine groups. This classification is based on two criteria: the duration of travel and the number of hard constraints. The dataset is divided into the training, validation, and test set. The training set includes 5 queries per group with human-annotated plans ( 45 pairs in total), the validation set includes 20 queries per group (180 in total), and the test set includes 1,000 queries. Detailed distributions are shown in Table A.1.

### 3.2. Constraint Introduction

In order to assess whether agents can perceive, understand, and satisfy various constraints to formulate a feasible plan, as outlined in Table 1, we include three types of constraints:

- Environment Constraints: The real-world is dynamic, necessitating agents to be adaptable. For instance, flights to a particular destination may be unavailable at certain times (e.g., no fllights from Seattle to San Francisco in Figure 1), possibly because tickets are sold out. In such cases, the agent must dynamically seek an alternative, like changing the destination of the flight or the way of transportation. To simulate this, we introduce environment constraints within TravelPlanner to test the adaptability of agents in planning.

Table 1. Constraint description. The environment constraints are manifested through the feedback received from the environment, assessing whether the language agent can adjust its plan appropriately. The commonsense constraints and hard constraints are evaluated based on how well the language agent's plan aligns with these specific criteria.

| Constraint | Description |
| :---: | :---: |
| Environment Constraint |  |
| Unavailable Transportation | There is no available flight or driving information between the two cities. |
| Unavailable Attractions | There is no available attraction information in the queried city. |
| Commonsense Constraint |  |
| Within Sandbox | All information in the plan must be within the closed sandbox; otherwise, it will be considered a hallucination. |
| Complete Information | No key information should be left out of the plan, such as the lack of accommodation during travel. |
| Within Current City | All scheduled activities for the day must be located within that day's city(s). |
| Reasonable City Route | Changes in cities during the trip must be reasonable. |
| Diverse Restaurants | Restaurant choices should not be repeated throughout the trip. |
| Diverse Attractions | Attraction choices should not be repeated throughout the trip. |
| Non-conf. Transportation | Transportation choices within the trip must be reasonable. For example, having both "self-driving" and "flight" would <br> be considered a conflict. |
| Minimum Nights Stay | The number of consecutive days spent in a specific accommodation during the trip must meet the corresponding <br> required minimum number of nights' stay. |
| Hard Constraint |  |
| Budget | The total budget of the trip. |
| Room Rule | Room rules include "No parties", "No smoking", "No children under 10", "No pets", and "No visitors". |
| Room Type | Room types include "Entire Room", "Private Room", "Shared Room", and "No Shared Room". |
| Cuisine | Cuisines include "Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", and "French". |
| Transportation | Transportation options include "No flight" and "No self-driving". |

Table 2. The number of data entries in the database.

| Tool | Data Entries (\#) |
| :--- | :--- |
| CitySearch | 312 |
| FlightSearch | $3,827,361$ |
| DistanceMatrix | 17,603 |
| RestaurantSearch | 9,552 |
| AttractionSearch | 5,303 |
| AccommodationSearch | 5,064 |

- Commonsense Constraints: Agents, functioning in realworld and serving humans, should consider commonsense when designing plans. For instance, repeatedly visiting the same attraction is not typical. To evaluate agents' understanding and utilization of commonsense during planning, we include the commonsense constraint in TravelPlanner.
- Hard Constraints: A crucial ability for agents is to effectively satisfy personalized user needs. To evaluate this, TravelPlanner incorporates various user needs, such as budget constraints. These user needs are termed hard constraints. The hard constraint measures the agent's generalization ability with regard to different user needs.


### 3.3. Benchmark Construction Pipeline

This section outlines the construction pipeline of TravelPlanner, which involves the following steps: 1) Environment and evaluation setup. 2) Diverse travel query design. 3) Reference plan annotation. 4) Quality check.

Environment Setting. In TravelPlanner, we create a static and closed sandbox environment for consistent and unbiased evaluations. This setup ensures that all agents access the same unchanging information from our static databases, avoiding the variability and potential biases introduced by dynamic data. To offer various travel options that align with the real-world, we ensure the database for each tool in TravelPlanner contains rich information. The database size of each tool is listed in Table 2. For more tool details, please refer to Appendix A. 2 and A.3. Additionally, agents are instructed to use the "NotebookWrite" tool to record necessary information for planning. This tool is integrated to evaluate agents' working memory management and prevents maximum token limit caused by context accumulation.

Query Construction. To create diverse queries for TravelPlanner, we begin with several fundamental elements, including departure city, destination, and specific date range, randomly chosen to form the skeleton of each query. Subsequently, we adjust the duration of travel and the number of hard constraints to create different levels of complexity.

The duration of the travel-3, 5 , or 7 days-determines the number of cities included in the plan. Specifically, 3-day plans focus on one city, while 5 days and 7 days involve visiting one randomly chosen state, with trips to 2 cities for the 5 -day plans and 3 cities for the 7 -day plans, respectively. A greater number of days requires more frequent tool usage by language agents, thus increasing the difficulty of managing the long-horizon aspect of planning. The uncertain destination challenges agents to decide on multiple cities, where they must consider factors such as inter-city connectivity.

Furthermore, we introduce diverse user needs as hard constraints to add further complexity and realism. The difficulty levels are categorized as follows:

- Easy: Queries at this level are primarily budgetconstrained for a single person. The initial budget for each query is determined using a set of crafted heuristic rules.
- Medium: In addition to budget constraints, medium queries introduce an additional hard constraint, randomly selected from a constraint pool including cuisine type, room type, and room rule. Furthermore, the number of people varies between 2 and 8 , which influences the calculation of costs for both transportation and accommodation.
- Hard: Hard queries include additional transportation preference into the constraint pool, along with all the constraints in medium queries. Each hard query contains three hard constraints randomly selected from the constraint pool.

This method ensures the diversity of queries. Minor changes in these elements can lead to significantly different plans. Finally, based on elements, we utilize GPT-4 (OpenAI, 2023) to generate natural language queries.

Human Annotation. To ensure every query has at least one feasible plan, we invite 20 graduate students to meticulously annotate plans for synthesized queries. One plan is deemed eligible only if it meets all the constraints outlined in our evaluation script, which is detailed in Section 3.4. This rigorous process resulted in the creation of 1,225 validated query-plan pairs. We pay annotators an average of $\$ 0.80$ for each plan they annotate.

Quality Control. To ensure the quality of each natural language query and its corresponding annotated plan, the authors performed a detailed review of every query and plan, rectifying any errors found. Additionally, to ensure the challenges, we re-calibrate each query's budget using the costs from corresponding human-annotated plans. This approach replaces the initial heuristic-generated budgets, which might be too high, thus reducing the number of feasible plans. Through multiple stages of human verification, we ensure the high quality of each query in TravelPlanner and the presence of at least one feasible solution.

### 3.4. Evaluation

To ensure a comprehensive evaluation of the plans offered by agents, we assess them from multiple dimensions. Specifically, we first extract key components ${ }^{1}$, including transportation, restaurants, attractions, and accommodations, which are initially presented as natural language. These components are then organized into a formally structured plan,[^1]

which will be evaluated automatically through pre-defined scripts. The evaluation criteria include the following:

- Delivery Rate: This metric assesses whether agents can successfully deliver a final plan within a limited number of steps. Falling into dead loops, experiencing numerous failed attempts, or reaching the maximum number of steps (30 steps in our experimental setting) will result in failure.
- Commonsense Constraint Pass Rate: Comprising eight commonsense dimensions, this metric evaluates whether a language agent can incorporate commonsense into their plan without explicit instructions.
- Hard Constraint Pass Rate: This metric measures whether a plan satisfies all explicitly given hard constraints in the query, which aims to test the agents' ability to adapt their plans to diverse user needs.
- Final Pass Rate: This metric represents the proportion of feasible plans that meet all aforementioned constraints among all tested plans. It serves as an indicator of agents' proficiency in producing plans that meet a practical standard.

We do not separately assess environment constraints since their impact is inherently reflected in the "Within Sandbox" and "Complete Information" metrics. For instance, when cities lack transportation or attractions, agents typically resort to hallucination or opt not to provide an answer, reflecting the impact of environment constraints.

For the Commonsense Constraint Pass Rate and Hard Constraint Pass Rate, we utilize two evaluation strategies: micro and macro. The micro strategy calculates the ratio of passed constraints to the total number of constraints. The Micro Pass Rate is defined as:

$$
\begin{equation*}
\text { Micro Pass Rate }=\frac{\sum_{p \in P} \sum_{c \in C_{p}} \mathbb{1}_{\text {passed }(c, p)}}{\sum_{p \in P}\left|C_{p}\right|} \tag{1}
\end{equation*}
$$

where $P$ represents the set of all plans being evaluated, $C_{p}$ denotes the set of constraints applicable to a specific plan $p$ in $P$, and passed $(X, Y)$ is a function determining whether $Y$ meets constraints $X$.

The macro strategy calculates the ratio of plans that pass all commonsense or hard constraints among all tested plans. We define the Macro Pass Rate as:

$$
\begin{equation*}
\text { Macro Pass Rate }=\frac{\sum_{p \in P} \mathbb{1}_{\text {passed }\left(C_{p}, p\right)}}{|P|} \tag{2}
\end{equation*}
$$

These two metrics evaluate an agent's capability of following individual constraints vs. all the constraints holistically.

### 3.5. Sole-Planning Setting

While TravelPlanner is designed to assess the overall abilities of agents in tool-use and planning (two-stage mode),

Table 3. Main results of different LLMs and planning strategies on the TravelPlanner validation and test set. The best results are marked in bold. When the collected information is insufficient, Gemini Pro tends to directly refuse to provide the plan. Interviews with annotators reveal that manually annotating a plan averagely takes around 12 minutes. However, language agents, such as GPT-3.5-Turbo, can accomplish this task in just 1 to 2 minutes, showcasing their efficiency.

|  | Validation $(\# 180)$ |  |  |  |  |  | Test $(\# 1,000)$ |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Delivery <br> Rate | Commonsense <br> Pass Rate |  | Hard Constraint <br> Pass Rate |  | Final <br> Pass Rate | Delivery <br> Rate | Commonsense <br> Pass Rate |  | Hard Constraint <br> Pass Rate |  | Final <br> Pass Rate |
|  |  | $\overline{\text { Micro }}$ | Macro | Micro | Macro |  |  | $\overline{\text { Micro }}$ | Macro | Micro | Macro |  |
| Greedy Search | 100 | 74.4 | 0 | 60.8 | 37.8 | 0 | 100 | 72.0 | 0 | 52.4 | 31.8 | 0 |
| Two-stage |  |  |  |  |  |  |  |  |  |  |  |  |
| Mistral-7B-32K (Jiang et al., 2023) | 8.9 | 5.9 | 0 | 0 | 0 | 0 | 7.0 | 4.8 | 0 | 0 | 0 | 0 |
| Mixtral-8×7B-MoE (Jiang et al., 2024) | 49.4 | 30.0 | 0 | 1.2 | 0.6 | 0 | 51.2 | 32.2 | 0.2 | 0.7 | 0.4 | 0 |
| Gemini Pro (G Team et al., 2023) | 28.9 | 18.9 | 0 | 0.5 | 0.6 | 0 | 39.1 | 24.9 | 0 | 0.6 | 0.1 | 0 |
| GPT-3.5-Turbo (OpenAI, 2022) | 86.7  | 54.0 | 0 | 0 | 0 | 0 | 91.8 | 57.9 | 0 | 0.5 | 0.6 | 0 |
| GPT-4-Turbo (OpenAI, 2023) | 89.4 | 61.1 | 2.8 | 15.2 | 10.6 | 0.6 | 93.1 | 63.3 | 2.0 | 10.5 | 5.5 | 0.6 |
| Sole-planning |  |  |  |  |  |  |  |  |  |  |  |  |
| Direct $_{\mathrm{GPT}-3.5-\text { Turbo }}$ | 100 | 60.2 | 4.4 | 11.0 | 2.8 | 0 | 100 | 59.5 | 2.7 | 9.5 | 4.4 | 0.6 |
| $\mathrm{CoT}_{\mathrm{GPT}-3.5-\text { Turbo }}$ | 100 | 66.3 | 3.3 | 11.9 | 5.0 | 0 | 100 | 64.4 | 2.3 | 9.8 | 3.8 | 0.4 |
| $\operatorname{ReAct}_{\mathrm{GPT}-3.5-\text { Turbo }}$ | 82.2 | 47.6 | 3.9 | 11.4 | 6.7 | 0.6 | 81.6 | 45.9 | 2.5 | 10.7 | 3.1 | 0.7 |
| Reflexion $_{\mathrm{GPT}-3.5-\text { Turbo }}$ | 93.9 | 53.8 | 2.8 | 11.0 | 2.8 | 0 | 92.1 | 52.1 | 2.2 | 9.9 | 3.8 | 0.6 |
| Direct $_{\text {Mixtral }-8 \times 7 B-M o E}$ | 100 | 68.1 | 5.0 | 3.3 | 1.1 | 0 | 99.3 | 67.0 | 3.7 | 3.9 | 1.6 | 0.7 |
| Direct $_{\text {Gemini Pro }}$ | 93.9 | 65.0 | 8.3 | 9.3 | 4.4 | 0.6 | 93.7 | 64.7 | 7.9 | 10.6 | 4.7 | 2.1 |
| Direct $_{\mathrm{GPT}-4-\text { Turbo }}$ | 100 | 80.4 | 17.2 | 47.1 | 22.2 | 4.4 | 100 | 80.6 | 15.2 | 44.3 | 23.1 | 4.4 |

we also setup a simplified mode solely evaluating agents' planning skills (sole-planning mode). In this setting, we utilize human-annotated plans to pre-determine the destination cities, and provide detailed and necessary information directly to agents, such as restaurants in the provided cities. This eliminates the need for tool calling as agents don't need to collect information from scratch via tools anymore.

## 4. Experiments

We evaluate the performance of various LLMs and planning strategies on TravelPlanner. In the two-stage mode, we use the ReAct (Yao et al., 2022) framework for information collection, which is recognized for its effective iteration with tools (Zhuang et al., 2023) while varying the foundation LLMs. This approach allows us to assess how different LLMs perform under a uniform tool-use framework. The agents are required to give the plan directly based on the information collected by themselves, without employing any other planning strategies. In the sole-planning mode, our evaluation goes beyond varying LLMs to include different planning strategies. This aims to assess if the strategies proven effective in other planning benchmarks maintain their efficacy in TravelPlanner. All experiments are conducted in a zero-shot setting.

### 4.1. Baselines

Greedy Search. To evaluate the effectiveness of traditional rule-based strategies within TravelPlanner, we include greedy search as a baseline and set cost as the optimization objective. Please refer to Appendix B. 1 for more details.
LLMs. Due to the long context window requirement of ReAct and the massive information as text, we limit our consideration to LLMs capable of handling inputs exceeding $8 \mathrm{~K}$ in length. As a result, our selection includes three closedsource LLMs: GPT-3.5-Turbo (OpenAI, 2022), GPT-4Turbo (OpenAI, 2023), and Gemini Pro (G Team et al., 2023), as well as two open-source LLMs: Mistral-7B32K (Jiang et al., 2023) and Mixtral-8x7B-MoE (Jiang et al., 2024). For all these models, we adopt the official instruction formats whenever available.

Planning Strategies. To explore the effectiveness of current planning strategies, we evaluate four representative ones: Direct, ZS-CoT (Wei et al., 2022), ReAct (Yao et al., 2022), and Reflexion (Shinn et al., 2023). For the implementation details, please refer to Appendix B.1. We do not include ToT (Yao et al., 2023) and GoT (Besta et al., 2023) because they require extensive exploration of the search space, prohibitively costly for problems as complex as TravelPlanner. Also, given their performance close to ReAct in complex tasks (Zhuang et al., 2024), the potential benefits of these methods may be limited.

### 4.2. Main Results

In this section, we discuss the performance of various LLMs and planning strategies on TravelPlanner (Table 3). We have the following observations:

TravelPlanner poses a significant challenge. In the twostage mode, GPT-4-Turbo with ReAct achieves only $0.6 \%$ in the final pass rate, and none of the other LLMs can pass any

![](https://cdn.mathpix.com/cropped/2024_06_04_a564c634bf6e44ec5b6ag-07.jpg?height=260&width=832&top_left_y=217&top_left_x=186)

Figure 2. Tool-use error distribution on the test set. An early stop will be triggered if the agent either makes three consecutive failed attempts or repetitive actions, indicating a dead loop.

of the tasks. Even given all the necessary information in the sole-planning mode, existing planning strategies like ReAct and Reflexion still struggle with planning in TravelPlanner, even though they have shown their effectiveness in more conventional planning tasks. It is noteworthy that the bestperforming agent still falls short on hard constraints even when compared to greedy search. This poor performance underlines the difficulty of TravelPlanner and shows that current agents still struggle with complex planning.

Agents show a substantial gap between the two modes. The comparison of the two modes reveals the agents' struggles in fiddling with both information collection and planning. Across all metrics, the scores of any model in the two-stage mode are lower than those in the sole-planning mode, with the largest gap reaching over $30 \%$. Similar to humans, language agents also seem to have a limited "cognitive capacity" and their performance deteriorates when multitasking. We provide a further analysis in Section 5.2.

Agents struggle in obtaining a high macro pass rate. While some agents achieve high micro scores, their macro scores remain low. This pattern shows that although agents manage to satisfy some constraints, they often overlook some other constraints in the meantime. Consequently, this indicates the current agents fail to consider multiple constraints holistically, a critical requirement for navigating the complex tasks in TravelPlanner.

In summary, TravelPlanner poses a great challenge to current agents. The SoTA LLMs and planning strategies, which often show equal or superior to human-level performance on many traditional tasks, are still far from sufficient for complex planning tasks that humans are capable of. TravelPlanner provides a challenging yet meaningful benchmark for the development of more capable language agents.

## 5. In-Depth Analysis

### 5.1. Tool-Use Error Analysis

As shown in Table 3, even based on GPT-4-Turbo, agents still make mistakes in the process of information collection and thus fail to deliver a plan. This problem is more severe in Gemini Pro and Mixtral. To delve into the underlying causes, we categorize all error types in Figure 2. We find:
Table 4. Constraint pass rate of GPT-4-Turbo on test set. The results of the sole-planning mode are based on the Direct strategy.

| Constraint Type | Two-stage |  |  | Sole-planning |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $\overline{\text { Easy }}$ | Medium | Hard | Easy | Medium | Hard |
| Commonsense Constraint |  |  |  |  |  |  |
| Within Sandbox | 37.4 | 31.2 | 33.9 | 76.4  | 71.5 | 79.3 |
| Complete Information | 53.4 | 52.9 | 58.0 | 94.5 | 96.4 | 96.2 |
| Within Current City | 69.3 | 67.3 | 68.3 | 89.1 | 80.8 | 82.4 |
| Reasonable City Route | 44.5 | 45.6 | 54.9 | 99.4 | 99.7 | 99.1 |
| Diverse Restaurants | 85.1 | 81.4 | 86.8 | 91.1 | 89.8 | 87.8 |
| Diverse Attractions | 94.3 | 90.4 | 94.0 | 100.0 | 100.0 | 100.0 |
| Non-conf. Transportation | 70.1 | 73.3 | 83.1 | 60.1 | 56.5 | 87.5 |
| Minimum Nights Stay | 46.8 | 46.2 | 51.1 | 37.4 | 28.8 | 30.1 |
| Hard Constraint |  |  |  |  |  |  |
| Budget | 10.1 | 8.4 | 4.4 | 37.4 | 35.1 | 25.1 |
| Room Rule | - | 5.6 | 11.3 | - | 31.5 | 43.6 |
| Cuisine | - | 10.8 | 11.4 | - | 57.5 | 46.7 |
| Room Type | - | 12.4 | 13.8 | - | 45.7 | 56.7 |
| Transportation | - | - | 18.6 | - | - | 77.5 |
| Final |  |  |  |  |  |  |
| $\overline{\text { Final Pass Rate }}$ | $\overline{1.1}$ | 0.3 | 0.3 | $\overline{8.0}$ | $\overline{2.7}$ | 2.2 |

Table 5. Comparison of the numbers of different tool uses between agent (GPT-4-Turbo) and reference. The results of agent are based on the number of entries written into the "Notebook".

| Average | Agent |  |  |  |  | Reference |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | 3-day | 5-day | 7-day | 3-day | 5-day | 7-day |  |  |
| FlightSearch | 0.7 | 1.2 | 0.8 |  | 2.0 | 3.0 | 4.0 |  |
| DistanceMatrix | 0.3 | 0.6 | 1.2 |  | 2.0 | 3.0 | 4.0 |  |
| RestaurantSearch | 0.9 | 1.5 | 2.4 |  | 1.0 | 2.0 | 3.0 |  |
| AttractionSearch | 0.9 | 1.7 | 2.4 | 1.0 | 2.0 | 3.0 |  |  |
| AccommodationSearch | 0.9 | 1.6 | 2.5 |  | 1.0 | 2.0 | 3.0 |  |

1) Agents incorrectly use tools. Except for GPT-4-Turbo, other LLMs-based agents all have argument error problems to varying degrees. It sheds light that the use of simple tools still poses a significant challenge for agents. 2) Agents trap in dead loops. Even with GPT-4-Turbo, invalid actions and repetitive action loops contribute to $37.3 \%$ and $6.0 \%$ of errors, respectively. Despite receiving feedback that actions are invalid or yield null results, agents persistently repeat these actions. This suggests that agents fail to dynamically adjust their plans based on environment feedback.

### 5.2. Planning Error Analysis

We detail each constraint pass rate in Table 4, from which we have the following observations:

The number of hard constraints affects the performance of agents. Agents consistently exhibit pass rates below $10 \%$ across all levels of difficulty, and this performance deteriorates further as more constraints are introduced. This trend highlights that current agents struggle with multi-constraint tasks, a key aspect of TravelPlanner.

Comprehensive information collection is essential for agents to plan effectively. In the sole-planning mode, agent performance improves over the two-stage mode. To investi-

![](https://cdn.mathpix.com/cropped/2024_06_04_a564c634bf6e44ec5b6ag-08.jpg?height=691&width=1699&top_left_y=218&top_left_x=186)

Figure 3. Case studies of failures. Agents fail to finalize plans due to repeated errors like incorrect dates, confusion with information details leading to hallucinatory answers, and disconnects between reasoning and actions. All cases are gained from GPT-4-Turbo based agents. For details of GPT-4-Turbo with the Reflexion strategy, please refer to Appendix B.2. We provide more cases in Appendix C.2.

gate this, Table 5 shows agents in two-stage mode use tools less effectively compared to the reference plans. This comparison indicates that agents often fail to finish completed information collection. Consequently, they either generate made-up information or omit specific details. This leads to low pass rates for the "Within Sandbox" and "Complete Information" constraints. Additionally, this discrepancy becomes more pronounced with an increase in the duration of travel. This emphasizes the need for agents to improve their capabilities in long-horizon tasks.

Agents struggle with global planning scenarios. Global constraints "Minimum Nights Stay" and "Budget" demand a holistic approach to planning, necessitating that agents not only assess their current decisions but also anticipate future implications. Current LLMs' auto-regressive nature limits them to independently obtain outcomes from multiple future branches. This highlights the necessity and urgent need of new strategies, such as backtracking for adjusting or employing heuristic methods for forward-looking planning.

### 5.3. Case Studies

To investigate the drawbacks of current agents in-depth, we provide several failure cases in Figure 3. We conclude with the following features:

Agents fail to complete a plan due to the inability to rectify persistent errors. In tool-use scenarios, agents often fail to deliver a plan even when all preceding steps are executed correctly. Further investigation reveals that this issue often stems from incorrect date inputs. As shown in the left part of Figure 3, despite correct execution, agents repeatedly use incorrect dates. This leads to null results, as the data in the TravelPlanner sandbox is based on 2022. Such re- peated failures eventually cause the agents to stop planning. This indicates a significant limitation: current agents cannot self-correct their initial and incorrect assumptions.

Agents produce hallucinatory answers due to information confusion. To understand why agents provide hallucinatory answers even when supplied with sufficient information in the sole-planning mode, we conduct a detailed analysis. We observe a tendency for agents to confuse one piece of information with another. As shown in the middle part of Figure 3, agents mistakenly use the same flight number for both departure and return flights. Such errors result in hallucinations, as the information provided in the plan does not align with the data in the sandbox. This suggests that agents might be lost when faced with mass information, known as "Lost in the Middle" (Liu et al., 2023b).

Agents struggle to align their actions with their reasoning. To understand the reasons behind the lower delivery rate of Reflexion (Shinn et al., 2023), we examine specific examples. As illustrated in the right part of Figure 3, we observe a discrepancy between what agents think and what they do. Despite recognizing the necessity to minimize costs, they tend to randomly select items, some of which may be more expensive. This discrepancy demonstrates that agents struggle to synchronize their actions with their analytical reasoning, severely impeding their delivery rate.

## 6. Conclusion

We introduce TravelPlanner, a benchmark grounded in realworld scenarios, designed to assess the multi-constraint planning and tool-use abilities of current language agents. Our benchmark presents a significant challenge: even the most advanced language agent frameworks only achieve
a mere $0.6 \%$ score in the final pass rate. Further analysis shows that these agents are unable to take all constraints into consideration to deliver feasible plans.

TravelPlanner's intricate logic and general applicability stand as vital components in the progressive development of language agents, thus contributing to the broader quest for AI abilities. We envision TravelPlanner as a catalyst for future research, aiming to enhance agents' performance in increasingly complex scenarios, hill-climbing towards human-level cognitive capabilities.

## 7. Impact Statements

TravelPlanner aims to provide an effective benchmark for complex planning in future research. Some of the data in the TravelPlanner environment is derived from publicly available data on the Internet, and the content involved does not represent the authors' viewpoints. We realize that everyone's definition of commonsense may be different. Our current evaluation criteria are based on the authors' consensus, and we encourage additional discussions to enrich our commonsense dimension, aiming for a more thorough evaluation. We will release our evaluation scripts to foster innovation and aid the development of new methods. We encourage the use of evaluation feedback in training set, such as implementing reinforcement learning techniques, to enhance learning. However, we strictly prohibit any form of cheating in the validation and test sets to uphold the fairness and reliability of the benchmark's evaluation process.

## References

Alterovitz, R., Koenig, S., and Likhachev, M. Robot planning in the real world: Research challenges and opportunities. Ai Magazine, 2016.

Andreas, J. Language models as agent models. In Findings of EMNLP, 2022.

AutoGPT. Autogpt, 2023. URL https://github.com/ Significant-Gravitas/AutoGPT.

Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Gianinazzi, L., Gajda, J., Lehmann, T., Podstawski, M., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687, 2023.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In Proceedings of NeurIPS, 2020.

Campbell, M., Hoane Jr, A. J., and Hsu, F.-h. Deep blue. Artificial intelligence, 2002.
Chen, H., Pasunuru, R., Weston, J., and Celikyilmaz, A. Walking down the memory maze: Beyond context limit through interactive reading. arXiv preprint arXiv:2310.05029, 2023.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Cross, S. and Estrada, R. Dart: an example of accelerated evolutionary development. In Proceedings of Workshop on $R S P, 1994$.

Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun, H., and Su, Y. Mind2web: Towards a generalist agent for the web. In Proceedings of NeurIPS, 2023.

G Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Ge, Y., Hua, W., Ji, J., Tan, J., Xu, S., and Zhang, Y. Openagi: When llm meets domain experts. In Proceedings of NeurIPS, 2023.

Georgievski, I. and Aiello, M. Htn planning: Overview, comparison, and beyond. Artificial Intelligence, 2015.

Grafman, J., Spector, L., and Rattermann, M. J. Planning and the brain. In The cognitive psychology of planning. 2004.

Hayes-Roth, B. and Hayes-Roth, F. A cognitive model of planning. Cognitive science, 1979.

Ho, M. K., Saxe, R., and Cushman, F. Planning with theory of mind. Trends in Cognitive Sciences, 2022.

Huang, W., Abbeel, P., Pathak, D., and Mordatch, I. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In Proceedings of ICML, 2022.

Ichter, B., Brohan, A., Chebotar, Y., Finn, C., Hausman, K., Herzog, A., Ho, D., Ibarz, J., Irpan, A., Jang, E., Julian, R., Kalashnikov, D., Levine, S., Lu, Y., Parada, C., Rao, K., Sermanet, P., Toshev, A., Vanhoucke, V., Xia, F., Xiao, T., Xu, P., Yan, M., Brown, N., Ahn, M., Cortes, O., Sievers, N., Tan, C., Xu, S., Reyes, D., Rettinghouse, J., Quiambao, J., Pastor, P., Luu, L., Lee, K., Kuang, Y., Jesmonth, S., Joshi, N. J., Jeffrey, K., Ruano, R. J., Hsu, J., Gopalakrishnan, K., David, B., Zeng, A., and Fu, C. K. Do as I can, not as I say: Grounding language in robotic affordances. In Proceedings of CoRL, 2022.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. 1., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de las Casas, D., Hanna, E. B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L. R., Saulnier, L., Lachaux, M.A., Stock, P., Subramanian, S., Yang, S., Antoniak, S., Scao, T. L., Gervet, T., Lavril, T., Wang, T., Lacroix, T., and Sayed, W. E. Mixtral of experts. arXiv preprint

![](https://cdn.mathpix.com/cropped/2024_06_04_a564c634bf6e44ec5b6ag-10.jpg?height=46&width=366&top_left_y=714&top_left_x=218)

Karpas, E. and Magazzeni, D. Automated planning for robotics. Annual Review of Control, Robotics, and Autonomous Systems, 2020.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V. V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., Wu, Y., Neyshabur, B., GurAri, G., and Misra, V. Solving quantitative reasoning problems with language models. In Proceedings of NeurIPS, 2022.

Li, M., Zhao, Y., Yu, B., Song, F., Li, H., Yu, H., Li, Z., Huang, F., and Li, Y. Api-bank: A comprehensive benchmark for tool-augmented llms. In Proceedings of EMNLP, 2023.

Liang, X., Wang, B., Huang, H., Wu, S., Wu, P., Lu, L., Ma, Z., and Li, Z. Unleashing infinite-length input capacity for large-scale language models with self-controlled memory system. arXiv preprint arXiv:2304.13343, 2023.

Liu, B., Jiang, Y., Zhang, X., Liu, Q., Zhang, S., Biswas, J., and Stone, P. Llm+ p: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477, 2023a.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023b.

Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., Gu, Y., Ding, H., Men, K., Yang, K., et al. Agentbench: Evaluating llms as agents. In Proceedings of ICLR, 2024.

Lu, P., Peng, B., Cheng, H., Galley, M., Chang, K.-W., Wu, Y. N., Zhu, S.-C., and Gao, J. Chameleon: Plug-and-play compositional reasoning with large language models. In Proceedings of NeurIPS, 2023.

Mattar, M. G. and Lengyel, M. Planning in the brain. Neuron, 2022.

McDermott, D. Robot planning. AI magazine, 1992.
Nakajima, Y. Task-driven autonomous agent utilizing gpt-4, pinecone, and langchain for diverse applications, 2023.

Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

OpenAI. Chatgpt, 2022. URL https://openai.com/ blog/chatgpt.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. Generative agents: Interactive simulacra of human behavior. In Proceedings of UIST, 2023.

Patel, A., Bhattamishra, S., and Goyal, N. Are NLP models really able to solve simple math word problems? In Proceedings of NAACL, 2021.

Patil, S. G., Zhang, T., Wang, X., and Gonzalez, J. E. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023.

Pinedo, M. Planning and scheduling in manufacturing and services. Springer, 2005.

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., et al. Toolllm: Facilitating large language models to master $16000+$ real-world apis. In Proceedings of ICLR, 2024.

Roy, S. and Roth, D. Solving general arithmetic word problems. In Proceedings of EMNLP, 2015.

Russell, S. J. and Norvig, P. Artificial intelligence a modern approach. 2010.

Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. In Proceedings of NeurIPS, 2023.

Shen, Y., Song, K., Tan, X., Li, D., Lu, W., and Zhuang, Y. HuggingGPT: Solving AI tasks with chatGPT and its friends in hugging face. In Proceedings of NeurIPS, 2023.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K. R., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. In Proceedings of NeurIPS, 2023.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. Mastering the
game of go with deep neural networks and tree search. nature, 2016.

Silver, D., Hasselt, H., Hessel, M., Schaul, T., Guez, A., Harley, T., Dulac-Arnold, G., Reichert, D., Rabinowitz, N., Barreto, A., et al. The predictron: End-to-end learning and planning. In Proceedings of ICML, 2017.

Song, C. H., Wu, J., Washington, C., Sadler, B. M., Chao, W.-L., and Su, Y. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of ICCV, 2023.

$\mathrm{Su}, \mathrm{Y}$. Language agents: a critical evolutionary step of artificial intelligence. 2023. URL https://yusu.substack. com/p/language-agents.

Sumers, T. R., Yao, S., Narasimhan, K., and Griffiths, T. L. Cognitive architectures for language agents. arXiv preprint arXiv:2309.02427, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023b

Valmeekam, K., Olmo, A., Sreedharan, S., and Kambhampati, S. Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change. In Proceedings of NeurIPS, 2023.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An openended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. In Proceedings of NeurIPS, 2022.

Weng, L. Llm-powered autonomous agents. lilianweng.github.io, Jun 2023. URL https://lilianweng. github.io/posts/2023-06-23-agent/.

Xie, T., Zhou, F., Cheng, Z., Shi, P., Weng, L., Liu, Y., Hua, T. J., Zhao, J., Liu, Q., Liu, C., Liu, L. Z., Xu, Y., Su, H., Shin, D., Xiong, C., and Yu, T. Openagents: An open platform for language agents in the wild. arXiv preprint arXiv:2310.10634, 2023.
Xu, Q., Hong, F., Li, B., Hu, C., Chen, Z., and Zhang, J. On the tool manipulation capability of open-source large language models. arXiv preprint arXiv:2305.16504, 2023.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In Proceedings of ICLR, 2022.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. R. Tree of thoughts: Deliberate problem solving with large language models. In Proceedings of NeurIPS, 2023.

Yuan, S., Chen, J., Fu, Z., Ge, X., Shah, S., Jankowski, C., Xiao, Y., and Yang, D. Distilling script knowledge from large language models for constrained language planning. In Proceedings of ACL, 2023.

Zheng, B., Gou, B., Kil, J., Sun, H., and Su, Y. Gpt-4v(ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.

Zhong, W., Guo, L., Gao, Q., and Wang, Y. Memorybank: Enhancing large language models with long-term memory. arXiv preprint arXiv:2305.10250, 2023.

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Bisk, Y., Fried, D., Alon, U., et al. Webarena: A realistic web environment for building autonomous agents. In Proceedings of ICLR, 2024.

Zhou, W., Jiang, Y. E., Cui, P., Wang, T., Xiao, Z., Hou, Y., Cotterell, R., and Sachan, M. Recurrentgpt: Interactive generation of (arbitrarily) long text. arXiv preprint arXiv:2305.13304, 2023.

Zhuang, Y., Yu, Y., Wang, K., Sun, H., and Zhang, C. ToolQA: A dataset for LLM question answering with external tools. In Proceedings of NeurIPS, 2023.

Zhuang, Y., Chen, X., Yu, T., Mitra, S., Bursztyn, V., Rossi, R. A., Sarkhel, S., and Zhang, C. Toolchain*: Efficient action space navigation in large language models with $\mathrm{a}^{*}$ search. In Proceedings of ICLR, 2024.
</end of paper 0>


<paper 1>
# Robust Planning with LLM-Modulo Framework: Case Study in Travel Planning 

Atharva Gundawar ${ }^{* 1}$ Mudit Verma ${ }^{* 1}$ Lin Guan ${ }^{1}$ Karthik Valmeekam ${ }^{1}$ Siddhant Bhambri ${ }^{1}$<br>Subbarao Kambhampati ${ }^{1}$

## 1. Introduction

As the applicability of Large Language Models (LLMs) extends beyond traditional text processing tasks, there is a burgeoning interest in their potential to excel in planning and reasoning assignments, realms traditionally reserved for System 2 cognitive competencies (Kahneman, 2011). Despite their perceived versatility, the research community is still unraveling effective strategies to harness these models in such complex domains. While there are studies showing LLMs are not able to support robust planning (Verma et al., 2024a; Stechly et al., 2024a; Valmeekam et al., 2022; Verma et al., 2024b), there is some consensus that they can help planning in a more integrated architecture (Kambhampati et al., 2024). The recent discourse introduced by the paper on LLM Modulo (Kambhampati et al., 2024) marks a significant stride, proposing a conceptual framework that enhances the integration of LLMs into diverse planning and reasoning activities. Of interest to this paper is to realize the LLM Modulo Framework for a Planning problem. As motivated by (Xie et al., 2024) Travel planning remains a complex domain, involving choices on destinations, accommodations, transport, and activities, which necessitates managing longterm dependencies and logical reasoning. This complexity makes travel planning an ideal domain to assess the reasoning abilities of planners. Utilizing the Travel Planning Benchmark (Xie et al., 2024), we aim to determine if language agents can handle realistic scenarios akin to human operations. Despite advanced techniques like ReAct(Yao et al., 2022) and Chain of Thought(Wei et al., 2022), these models achieve less than $1 \%$ accuracy, compared to humans who score $100 \%$.

The benchmark provides user queries in nautral langauge and an evaluation methodology for validating solution plans / itineraries obtained via LLM agents. In this paper, we will revisit the various abstract components suggested in the LLM-Modulo framework and realize it for the TravelPlanning domain. In this generate-test planning paradigm, the LLMs play several helpful roles such as the generator (to[^0]

generate the plan or travel itinerary), reformulator (or translator, for converting natural language queries to structured output parseable by other components) and critic extraction (to implement model based critics responsible for testing the LLM generated plan and backprompting the LLM for fixing known issues).

While popular methods of enhancing reasoning abilities of LLMs such as Chain of Thought, ReAct, and Reflexion achieve a meager $0 \%, 0.6 \%$, and $0 \%$ with GPT3.5Turbo(OpenAI, 2022) respectively , our operationalization of the LLM-Modulo framework for TravelPlanning domain provides a remarkable improvement, enhancing baseline performances by 4.6x for GPT4-Turbo(OpenAI, 2023) and even more for older models like GPT3.5-Turbo from $0 \%$ to $5 \%$. Furthermore, we highlight the other useful roles of LLMs in the planning pipeline, as suggested in LLMModulo, can be reliably operationalized such as extraction of useful critics and reformulator for critics.

## 2. Background and Setup

### 2.1. Travel Planning domain

The domain provides a sandboxed environment to generate itineraries for travel plan queries simulated using various datasets such as flights, restaurants, distances between locations, available transport options, accommodation choices to name a few. The TravelPlanning domain evaluates generated plans based on hard constraints and commonsense constraints. We use the recommended dataset (validation dataset) with 180 queries for all our experiments. Use of test set requires an official submission to TravelPlanning leaderboard which is left for future work. Furthermore, as a first investigation of LLM-Modulo for TravelPlanning we restrict our discussion to the sole-planning mode which simplifies the objective for the LLMs. In this setting, the domain provides necessary context (that would otherwise be obtained by querying the various datasets) to the LLM instead of expecting the LLM to perform Tool Use (Schick et al., 2024; Paranjape et al., 2023; Hsieh et al., 2023). The top performing models in this simplified mode achieves 4.4\% (GPT-4Turbo) and $0.6 \%$ (across various prompt-engineering tricks with GPT-3.5-Turbo).

Example : A query can be for a 3-day trip from

![](https://cdn.mathpix.com/cropped/2024_06_04_e181bac35ef37e7b8778g-2.jpg?height=632&width=1273&top_left_y=229&top_left_x=383)

Figure 1. LLM Modulo Framework for Travel Planning

Washington to Myrtle Beach for one traveler with a $\$ 1,400$ budget, running from March 13th to 15th, 2022. There are no specific preferences regarding accommodation, cuisine, room type, or transport method. Key attributes captured in each query include the origin city (org: Washington), destination city (dest: Myrtle Beach), trip duration (days: 3), number of cities visited (visiting_city_number: 1), number of travelers (people_number: 1), and no local constraints (local_constraint: none). The (budget: $\$ 1,400$ ) is set to cover all travel expenses, ensuring accurate and efficient travel planning within financial limits.

### 2.2. LLM Modulo

The LLM-Modulo framework introduced by (Kambhampati et al., 2024) establishes a robust iterative interaction between a base generative model, specifically a large language model (LLM), and a suite of external verifiers. These verifiers critically assess the LLM's outputs. Should the output not meet predefined criteria, these external critics provide feedback to the LLM, prompting necessary adjustments. Essentially, the work provides various uses of LLMs in the planning pipeline such as idea-generators, translators, problem specification enrichment, critic/model acquisition to name a few. This work instantiates several abstract roles of LLMs as presented in LLM-Modulo framework specific to the TravelPlanning (Xie et al., 2024) domain.

## 3. Instantiating LLM Modulo for Travel Planning

Our implementation follows the LLM-Modulo architecture presented in (Kambhampati et al., 2024) and the LLMModulo for TravelPlanning can be seen in Fig. 1.

Problem Specification By design the TravelPlanning do- main presents queries that contains all information which maybe required to generate a feasible travel plan, however, the query is in natural language which is a popular mode of interacting with LLMs.

Prompt Generator Consistent with use of LLMs as agents (Wang et al., 2024; Xi et al., 2023; Chang et al., 2024), we provide an instruction prompt the LLM along with the context information about flights, hotels etc. We also provide instructions on the output format of the generated plan and present few shot example. This is directly inherited from the implementation of (Xie et al., 2024).

Plan Backboard and Reformatter We transform the LLM generated natural language travel plan into a valid JSON format and store it in the plan blackboard. This translation is done through the use of LLM as a reformulator and we reuse it for our model based critics which require structured parseable plans.

Crtics All of the critics that we use are binary critics paired with a backprompt describing the issue should the critic detect one. The Format critics ensures the syntactic validity of the plan such as validating the JSON and eliminating any missing key-values which is a precondition for all other critics, therefore takes precedence. We repurpose the commonsense constraints as style critics that provide information about missing implicit preference considerations and finally use the hard-constraints as the remainder of the critics.

Metacontroller All of the critics evaluate a generated plan and incase any of the critics find issues with the generated plan the metacontroller takes on the control flow. It contains the decision-making logic to stich together the critic responses, choose which backprompts to allow for (if a certain pedagogical prompting is in effect) or other consolidation of various backprompts. The metacontroller interfaces with the LLM and makes use of the Prompt Generator to contain

| Model | Delivery <br> Rate | Commonsense <br> Pass Rate |  | Hard <br> Pass Rate |  | Final Pass <br> Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Micro | Macro | Micro | Macro |  |
| Direct $\mathrm{GPT}-3.5-$ Turbo | 99.4 | 61.5 | 3.9 | 11.2 | 2.8 | 0.0 |
| Direct GPT-4-Turbo | 100 | 84.9 | 25.6 | 51.9 | 24.4 | 4.4 |
| CoT $_{\text {GPT-3.5-Turbo }}$ | 100 | 66.3 | 3.3 | 11.9 | 5 | 0 |
| $\operatorname{ReAct}_{\mathrm{GPT}-3.5-\text { Turbo }}$ | 82.2 | 47.6 | 3.9 | 11.4 | 6.7 | 0.6 |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_e181bac35ef37e7b8778g-3.jpg?height=50&width=586&top_left_y=545&top_left_x=293) | 93.9 | 53.8 | 2.8 | 11 | 2.8 | 0 |
| LLM Modulo $[\text { All] }]_{\mathrm{GPT}-3.5-\text { Turbo }}$ | 97.8 | 59.8 | 13.3 | 14 | 6.7 | 5 |
| LLM Modulo $[\text { Common }]_{\mathrm{GPT}-3.5-\text { Turbo }}$ | 100 | 67.9 | 16.7 | 14 | 5 | 2.8 |
| LLM Modulo $[\text { Hard }]_{\mathrm{GPT}-3.5-\text { Turbo }}$ | 100 | 61.3 | 4.4 | 10.7 | 5.6 | 1.6 |
| LLM Modulo $[\text { Json }]_{\mathrm{GPT}}-3.5$-Turbo | 100 | 61.3 | 4.4 | 10.2 | 3.9 | 1.1 |
| LLM Modulo $[\text { All }]_{\mathrm{GPT}-4-\text { Turbo }}$ | 100 | 89.2 | 40.6 | 62.1 | 39.4 | 20.6 |

Table 1. We report the results on TravelPlanning Validation set following (Xie et al., 2024). Grayed out results on CoT / ReAct / Reflexion

![](https://cdn.mathpix.com/cropped/2024_06_04_e181bac35ef37e7b8778g-3.jpg?height=35&width=1697&top_left_y=882&top_left_x=187)
LLM Modulo $[\mathrm{Crtic}]_{\mathcal{M}}$ represents the critics used during the LLM-Modulo planning with model $\mathcal{M}$. Values are percentages of delivery rate, micro and macro commonsense and hard constraints and finally, success rate defined as Final Pass Rate as in (Xie et al., 2024).

other information such as instructions, database context, formatting and few shot examples along with the compiled backprompt. In this work we concatenate the backprompts from all the critics and add it to the initial prompt and provide it to the LLM.

The interaction loop in LLM Modulo continues uptill a specificed maximum budget (set to 10 iterations) or until all of the critics agree to the generated plan. Building on this integration, the use of critics within the Modulo framework illustrates that similar evaluative mechanisms can be effectively utilized across different datasets by converting traditional evaluation constraints into critics, enhancing output precision and adaptability. Moreover, the employment of a rudimentary metacontroller highlights the substantial potential for advancement. The current approach, which aggregates and reiterates critic responses, is simple yet effective. Future enhancements could include strategically ordering constraints or providing more targeted and relevant feedback in critic responses, improving the system's efficacy and responsiveness.

## 4. Experiments and Results

The baseline results with GPT-3.5 Turbo model showed a final pass rate of $0 \%$, for both micro and macro pass rates in commonsense and hard constraints being low across 180 queries, indicating that none of the generated plans fully met all constraints. Surprisingly, methods such as Chain of Thought(Wei et al., 2022), ReAct (Yao et al., 2022) and Reflexion(Shinn et al., 2024) provides no improvement. When used with GPT-3.5 Turbo, Chain of Thought and Reflexion exhibit a final pass rate of $0 \%$, while Chain of Thought alone achieves a slightly higher pass rate of $0.6 \%$, indicating suboptimal performance. While improvements from prompt engineering often remain unexplained, the LLM-Modulo framework promises soundness of produced plans consistent with the critics used. Indeed, we find that our LLM Modulo planner with GPT3.5-Turbo (older model) surpasses GPT4Turbo baseline performance (newer model). Consequently we see improved micro/macro pass rates along commonsense and hard constraints. LLM Modulo GPT4-Turbo achieves state of the art performance on TravelPlanning benchmark under the agentic LLM paradigm of using LLMs to generate final plans. We achieve $20.6 \%$ final pass rate compared to $4.4 \%$ baseline. Moreover, such gains are wellfounded and the source of improvement can be attributed to the presence of reliable critics in the planning pipeline.

### 4.1. Ablations

We categorized the critics into three subgroups: Format (which includes checks for valid JSON and the presence of all key-value pairs), Hard (hard constraints), and Commonsense (commonsense constraints). We study the impact of each class of critics on the final performance and other fine-grained metrics such as micro/macro rates. Note that choosing a subset of critics implies that we prevent the LLM from getting pointed feedback on issues in the generated plan as well as allow for suboptimal plans (in that they are guarateed to satisfy only the subset of critics) as final result.

As anticipated, LLM Modulo with a subset of critics underperforms relative to the model that uses all critic types, yet they demonstrated improvements over baselines (Direct ${ }_{\text {Model }}$ ) and CoT/ReAct/Reflexion variants. Utilizing solely the commonsense critics resulted in a final pass rate of $2.8 \%$, while employing just the hard constraints as critics achieved a final pass rate of $1.6 \%$. Solely ensuring the correct format yielded a $1.1 \%$ final pass rate.

We note our results demonstrate the composability of the critics. Compared to [Hard], [Common], or [JSON] variants, the LLM Modulo [All] result is higher micro/macro pass rates across common-sense and hard constraints (with exceptions for delivery rate and common-sense micro pass rate). While providing just the Commonsense critics gave the most improvement, composing it with other critics (Hard, JSON) yields a much higher performance rate. Finally, we see that even thought Direct $_{\text {GPT-4-Turbo }}$ has higher micro/macro pass rates even comparable to LLM Modulo[All] $]_{\text {GPT-4-Turbo }}$, the final pass rate in the case of LLM Modulo is 4.6x higher.

### 4.2. Frequency analysis of Critics

LLMs are known for capturing common-sense information about real world tasks (Stechly et al., 2024a;b; Guan et al., 2024; Kambhampati, 2024; Verma et al., 2024a). Figure 2 shows the number of times a critic was fired (or detected an issue with a generated plan) across all 180 validation set instances and iteration steps in LLM Modulo for GPT3.5-Turbo and GPT4-Turbo models. We find that certain critics more frequent than others since the corresponding issues occur more often. We also find that the format critic (for ensuring JSON correctness) is required more often by GPT3.5-Turbo over GPT4-Turbo and that the LLM Modulo planner is able to resolve format issues in the first few iterations of the budget. The critics that often disagree with the plan are valid_cost, is_valid_accommodation, and is_valid_information, which are generated respectively by Budget, Room Type, and Validate Itinerary. With the knowledge that critics maybe correlated (such as change in accommodation impacts transport and budget) and that only a few critics are the flagged most of the times, future work may take such statistics into account when designing an advanced Metacontroller and identification of points of failure for LLM generated plans.

### 4.3. Recovering critics from LLMs

Previous subsection highlights that only a few critics are flagged most of the times during the LLM Modulo interaction. We argue that LLMs may indeed be useful for extracting the implementation for such critics. This is akin to teasing out the model based critics in the LLM-Modulo frameworks. We prompt the GPT-4-Turbo model to obtain the implementation of the critics by providing it contextual information such as the objective of the critic, available tools or databases with corresponding function declaration (such as flights, etc.) and the input plan as JSON (along with the JSON schema). We do so for common-sense and hard constraint critics. We then compared the generated critics code implementation with the existing ones to evaluate their correctness. Typically, we observe that only minimal modifications were necessary (such as fixing function call signature and syntax which itself can be automated via crit- ics such as compilers and parsers) for the generated critics to match the efficacy of the pre-existing ones. The generated hard critics included: Room Type, Cuisines, Budget, and Transportation. The generated commonsense critics encompassed: Complete Information, Diverse Restaurants, Diverse Attractions, and Validate Itinerary.

![](https://cdn.mathpix.com/cropped/2024_06_04_e181bac35ef37e7b8778g-4.jpg?height=384&width=724&top_left_y=529&top_left_x=1096)

Figure 2. Comparison of Critic Values for GPT 3.5 Turbo and GPT 4 Turbo

![](https://cdn.mathpix.com/cropped/2024_06_04_e181bac35ef37e7b8778g-4.jpg?height=426&width=729&top_left_y=1077&top_left_x=1099)

Figure 3. Final Pass rates of models across LLM Modulo Iterations

## 5. Conclusion

We demonstrate the effective application of the LLM modulo framework within the TravelPlanning domain showcasing a remarkable 4.6x improvement performance for GPT4Turbo achieveing new state of the art on TravelPlanning domain under the agentic LLM paradigm. Our work also validates the framework's robustness in real-world scenarios such as Travel Planning as motivated by (Xie et al., 2024). We showcase that such performance boost is well-founded and easily surpasses predominant ways of enhacing agentic abilities of LLMs such as CoT / ReAct and Reflexion. We do so by allowing critics to be part of the LLM-Modulo based planning pipeline. We also showcase that such critics may also be extracted through the LLMs (LLMs working towards teasing out model-based critics / verifiers). Finally, we showcase the LLMs use as a reformualor to translate natural language plans to a structured representation (JSON) that can be easily used by the critics. Along our discussion, we also point out potential next steps beyond our first investigation to further improve agentic LLM performance.

## References

Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C., Wang, Y., et al. A survey on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology, 15(3):1-45, 2024.

Guan, L., Zhou, Y., Liu, D., Zha, Y., Amor, H. B., and Kambhampati, S. "task success" is not enough: Investigating the use of video-language models as behavior critics for catching undesirable agent behaviors, 2024.

Hsieh, C.-Y., Chen, S.-A., Li, C.-L., Fujii, Y., Ratner, A., Lee, C.-Y., Krishna, R., and Pfister, T. Tool documentation enables zero-shot tool-usage with large language models. arXiv preprint arXiv:2308.00675, 2023.

Kahneman, D. Thinking, fast and slow. macmillan, 2011.

Kambhampati, S. Can large language models reason and plan? Annals of the New York Academy of Sciences, 1534 (1):15-18, 2024.

Kambhampati, S., Valmeekam, K., Guan, L., Stechly, K., Verma, M., Bhambri, S., Saldyt, L., and Murthy, A. Llms can't plan, but can help planning in llm-modulo frameworks, 2024.

OpenAI. Gpt-3.5: Language model, 2022. https : / / www . openai.com.

OpenAI. Gpt-4: Language model, 2023. https: / /www . openai.com.

Paranjape, B., Lundberg, S., Singh, S., Hajishirzi, H., Zettlemoyer, L., and Ribeiro, M. T. Art: Automatic multi-step reasoning and tool-use for large language models. arXiv preprint arXiv:2303.09014, 2023.

Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.

Stechly, K., Valmeekam, K., and Kambhampati, S. Chain of thoughtlessness: An analysis of cot in planning, 2024a.

Stechly, K., Valmeekam, K., and Kambhampati, S. On the self-verification limitations of large language models on reasoning and planning tasks, 2024b.
Valmeekam, K., Olmo, A., Sreedharan, S., and Kambhampati, S. Large language models still can't plan (a benchmark for llms on planning and reasoning about change). arXiv preprint arXiv:2206.10498, 2022.

Verma, M., Bhambri, S., and Kambhampati, S. Theory of mind abilities of large language models in human-robot interaction: An illusion? In Companion of the 2024 ACM/IEEE International Conference on Human-Robot Interaction, HRI '24. ACM, March 2024a. doi: 10.1145/ 3610978.3640767. URL http://dx.doi.org/10. $1145 / 3610978.3640767$.

Verma, M., Bhambri, S., and Kambhampati, S. Theory of mind abilities of large language models in human-robot interaction: An illusion? In Companion of the 2024 ACM/IEEE International Conference on Human-Robot Interaction, pp. 36-45, 2024b.

Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):1-26, 2024.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824-24837, 2022.

Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., et al. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864, 2023.

Xie, J., Zhang, K., Chen, J., Zhu, T., Lou, R., Tian, Y., Xiao, Y., and Su, Y. Travelplanner: A benchmark for real-world planning with language agents. arXiv preprint arXiv:2402.01622, 2024.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.
</end of paper 1>


<paper 2>
# Mistral 7B 


#### Abstract

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed


![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-1.jpg?height=366&width=1073&top_left_y=749&top_left_x=534)


#### Abstract

We introduce Mistral 7B, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms the best open 13B model (Llama 2) across all evaluated benchmarks, and the best released 34B model (Llama 1) in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B - Instruct, that surpasses Llama 2 13B - chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license.

Code: https://github.com/mistralai/mistral-src

Webpage: https://mistral.ai/news/announcing-mistral-7b/


## 1 Introduction

In the rapidly evolving domain of Natural Language Processing (NLP), the race towards higher model performance often necessitates an escalation in model size. However, this scaling tends to increase computational costs and inference latency, thereby raising barriers to deployment in practical, real-world scenarios. In this context, the search for balanced models delivering both high-level performance and efficiency becomes critically essential. Our model, Mistral 7B, demonstrates that a carefully designed language model can deliver high performance while maintaining an efficient inference. Mistral 7B outperforms the previous best 13B model (Llama 2, [26]) across all tested benchmarks, and surpasses the best 34B model (LLaMa 34B, [25]) in mathematics and code generation. Furthermore, Mistral 7B approaches the coding performance of Code-Llama 7B [20], without sacrificing performance on non-code related benchmarks.

Mistral 7B leverages grouped-query attention (GQA) [1], and sliding window attention (SWA) [6, 3]. GQA significantly accelerates the inference speed, and also reduces the memory requirement during decoding, allowing for higher batch sizes hence higher throughput, a crucial factor for real-time applications. In addition, SWA is designed to handle longer sequences more effectively at a reduced computational cost, thereby alleviating a common limitation in LLMs. These attention mechanisms collectively contribute to the enhanced performance and efficiency of Mistral 7B.

Mistral 7B is released under the Apache 2.0 license. This release is accompanied by a reference implementation ${ }^{1}$ facilitating easy deployment either locally or on cloud platforms such as AWS, GCP, or Azure using the vLLM [17] inference server and SkyPilot ${ }^{2}$. Integration with Hugging Face ${ }^{3}$ is also streamlined for easier integration. Moreover, Mistral 7B is crafted for ease of fine-tuning across a myriad of tasks. As a demonstration of its adaptability and superior performance, we present a chat model fine-tuned from Mistral 7B that significantly outperforms the Llama 2 13B - Chat model.

Mistral 7B takes a significant step in balancing the goals of getting high performance while keeping large language models efficient. Through our work, our aim is to help the community create more affordable, efficient, and high-performing language models that can be used in a wide range of real-world applications.

## 2 Architectural details

![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-2.jpg?height=504&width=1326&top_left_y=816&top_left_x=384)

Effective Context Length

Figure 1: Sliding Window Attention. The number of operations in vanilla attention is quadratic in the sequence length, and the memory increases linearly with the number of tokens. At inference time, this incurs higher latency and smaller throughput due to reduced cache availability. To alleviate this issue, we use sliding window attention: each token can attend to at most $W$ tokens from the previous layer (here, $W=3$ ). Note that tokens outside the sliding window still influence next word prediction. At each attention layer, information can move forward by $W$ tokens. Hence, after $k$ attention layers, information can move forward by up to $k \times W$ tokens.

Mistral 7B is based on a transformer architecture [27]. The main parameters of the architecture are summarized in Table 1. Compared to Llama, it introduces a few changes that we summarize below.

Sliding Window Attention. SWA exploits the stacked layers of a transformer to attend information beyond the window size $W$. The hidden state in position $i$ of the layer $k, h_{i}$, attends to all hidden states from the previous layer with positions between $i-W$ and $i$. Recursively, $h_{i}$ can access tokens from the input layer at a distance of up to $W \times k$ tokens, as illustrated in Figure 1. At the last layer, using a window size of $W=4096$, we have a theoretical attention span of approximately $131 K$ tokens. In practice, for a sequence length of $16 \mathrm{~K}$ and $W=4096$, changes made to FlashAttention [11] and xFormers [18] yield a 2x speed improvement over a vanilla attention baseline.

Rolling Buffer Cache. A fixed attention span means that we can limit our cache size using a rolling buffer cache. The cache has a fixed size of $W$, and the keys and values for the timestep $i$ are stored in position $i \bmod W$ of the cache. As a result, when the position $i$ is larger than $W$, past values in the cache are overwritten, and the size of the cache stops increasing. We provide an illustration in Figure 2 for $W=3$. On a sequence length of $32 \mathrm{k}$ tokens, this reduces the cache memory usage by $8 \mathrm{x}$, without impacting the model quality.[^0]

| Parameter | Value |
| :--- | ---: |
| dim | 4096 |
| n_layers | 32 |
| head_dim | 128 |
| hidden_dim | 14336 |
| n_heads | 32 |
| n_kv_heads | 8 |
| window_size | 4096 |
| context_len | 8192 |
| vocab_size | 32000 |

Table 1: Model architecture.

![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-3.jpg?height=225&width=1396&top_left_y=251&top_left_x=362)

Figure 2: Rolling buffer cache. The cache has a fixed size of $W=4$. Keys and values for position $i$ are stored in position $i \bmod W$ of the cache. When the position $i$ is larger than $W$, past values in the cache are overwritten. The hidden state corresponding to the latest generated tokens are colored in orange.

Pre-fill and Chunking. When generating a sequence, we need to predict tokens one-by-one, as each token is conditioned on the previous ones. However, the prompt is known in advance, and we can pre-fill the $(k, v)$ cache with the prompt. If the prompt is very large, we can chunk it into smaller pieces, and pre-fill the cache with each chunk. For this purpose, we can select the window size as our chunk size. For each chunk, we thus need to compute the attention over the cache and over the chunk. Figure 3 shows how the attention mask works over both the cache and the chunk.

![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-3.jpg?height=393&width=963&top_left_y=928&top_left_x=581)

Figure 3: Pre-fill and chunking. During pre-fill of the cache, long sequences are chunked to limit memory usage. We process a sequence in three chunks, "The cat sat on", "the mat and saw", "the dog go to". The figure shows what happens for the third chunk ("the dog go to"): it attends itself using a causal mask (rightmost block), attends the cache using a sliding window (center block), and does not attend to past tokens as they are outside of the sliding window (left block).

## 3 Results

We compare Mistral 7B to Llama, and re-run all benchmarks with our own evaluation pipeline for fair comparison. We measure performance on a wide variety of tasks categorized as follow:

- Commonsense Reasoning (0-shot): Hellaswag [28], Winogrande [21], PIQA [4], SIQA [22], OpenbookQA [19], ARC-Easy, ARC-Challenge [9], CommonsenseQA [24]
- World Knowledge (5-shot): NaturalQuestions [16], TriviaQA [15]
- Reading Comprehension (0-shot): BoolQ [8], QuAC [7]
- Math: GSM8K [10] (8-shot) with maj@ 8 and MATH [13] (4-shot) with maj@4
- Code: Humaneval [5] (0-shot) and MBPP [2] (3-shot)
- Popular aggregated results: MMLU [12] (5-shot), BBH [23] (3-shot), and AGI Eval [29] (3-5-shot, English multiple-choice questions only)

Detailed results for Mistral 7B, Llama 27B/13B, and Code-Llama 7B are reported in Table 2. Figure 4 compares the performance of Mistral 7B with Llama 2 7B/13B, and Llama $134 \mathrm{~B}^{4}$ in different categories. Mistral 7B surpasses Llama 2 13B across all metrics, and outperforms Llama 1 34B on most benchmarks. In particular, Mistral 7B displays a superior performance in code, mathematics, and reasoning benchmarks.[^1]![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-4.jpg?height=440&width=1390&top_left_y=257&top_left_x=365)

Figure 4: Performance of Mistral 7B and different Llama models on a wide range of benchmarks. All models were re-evaluated on all metrics with our evaluation pipeline for accurate comparison. Mistral 7B significantly outperforms Llama 2 7B and Llama 2 13B on all benchmarks. It is also vastly superior to Llama 1 34B in mathematics, code generation, and reasoning benchmarks.

| Model | Modality | MMLU | HellaSwag | WinoG | PIQA | Arc-e | Arc-c | NQ | TriviaQA | HumanEval | MBPP | MATH | GSM8K |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LLaMA 2 7B | Pretrained | $44.4 \%$ | $77.1 \%$ | $69.5 \%$ | $77.9 \%$ | $68.7 \%$ | $43.2 \%$ | $24.7 \%$ | $63.8 \%$ | $11.6 \%$ | $26.1 \%$ | $3.9 \%$ | $16.0 \%$ |
| LLaMA 2 13B | Pretrained | $55.6 \%$ | $\mathbf{8 0 . 7 \%}$ | $72.9 \%$ | $80.8 \%$ | $75.2 \%$ | $48.8 \%$ | $\mathbf{2 9 . 0 \%}$ | $69.6 \%$ | $18.9 \%$ | $35.4 \%$ | $6.0 \%$ | $34.3 \%$ |
| Code-Llama 7B | Finetuned | $36.9 \%$ | $62.9 \%$ | $62.3 \%$ | $72.8 \%$ | $59.4 \%$ | $34.5 \%$ | $11.0 \%$ | $34.9 \%$ | $31.1 \%$ | $\mathbf{5 2 . 5 \%}$ | $5.2 \%$ | $20.8 \%$ |
| Mistral 7B | Pretrained | $60.1 \%$ | $81.3 \% \quad$ | $75.3 \%$ | $\mathbf{8 3 . 0 \%}$ | $\mathbf{8 0 . 0 \%}$ | $\mathbf{5 5 . 5 \%}$ | $\mathbf{2 8 . 8 \%}$ | $69.9 \%$ | $30.5 \%$ | $47.5 \%$ | $13.1 \%$ | $\mathbf{5 2 . 2 \%}$ |

Table 2: Comparison of Mistral 7B with Llama. Mistral 7B outperforms Llama 2 13B on all metrics, and approaches the code performance of Code-Llama 7B without sacrificing performance on non-code benchmarks.

Size and Efficiency. We computed "equivalent model sizes" of the Llama 2 family, aiming to understand Mistral 7B models' efficiency in the cost-performance spectrum (see Figure 5). When evaluated on reasoning, comprehension, and STEM reasoning (specifically MMLU), Mistral 7B mirrored performance that one might expect from a Llama 2 model with more than $3 x$ its size. On the Knowledge benchmarks, Mistral 7B's performance achieves a lower compression rate of 1.9x, which is likely due to its limited parameter count that restricts the amount of knowledge it can store.

Evaluation Differences. On some benchmarks, there are some differences between our evaluation protocol and the one reported in the Llama 2 paper: 1) on MBPP, we use the hand-verified subset 2) on TriviaQA, we do not provide Wikipedia contexts.

## 4 Instruction Finetuning

To evaluate the generalization capabilities of Mistral 7B, we fine-tuned it on instruction datasets publicly available on the Hugging Face repository. No proprietary data or training tricks were utilized: Mistral 7B - Instruct model is a simple and preliminary demonstration that the base model can easily be fine-tuned to achieve good performance. In Table 3, we observe that the resulting model, Mistral 7B - Instruct, exhibits superior performance compared to all 7B models on MT-Bench, and is comparable to $13 \mathrm{~B}$ - Chat models. An independent human evaluation was conducted on https://llmboxing.com/leaderboard.

| Model | Chatbot Arena <br> ELO Rating | MT Bench |
| :--- | :---: | :--- |
| WizardLM 13B v1.2 | 1047 | 7.2 |
| Mistral 7B Instruct | $\mathbf{1 0 3 1}$ | $\mathbf{6 . 8 4}+/ \mathbf{0 . 0 7}$ |
| Llama 2 13B Chat | 1012 | 6.65 |
| Vicuna 13B | 1041 | 6.57 |
| Llama 2 7B Chat | 985 | 6.27 |
| Vicuna 7B | 997 | 6.17 |
| Alpaca 13B | 914 | 4.53 |

Table 3: Comparison of Chat models. Mistral 7B Instruct outperforms all 7B models on MT-Bench, and is comparable to $13 \mathrm{~B}-$ Chat models.

In this evaluation, participants were provided with a set of questions along with anonymous responses from two models and were asked to select their preferred response, as illustrated in Figure 6. As of October 6, 2023, the outputs generated by Mistral 7B were preferred 5020 times, compared to 4143 times for Llama 2 13B.
![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-5.jpg?height=682&width=1000&top_left_y=244&top_left_x=560)

Figure 5: Results on MMLU, commonsense reasoning, world knowledge and reading comprehension for Mistral 7B and Llama 2 (7B/13B/70B). Mistral 7B largely outperforms Llama 2 13B on all evaluations, except on knowledge benchmarks, where it is on par (this is likely due to its limited parameter count, which limits the amount of knowledge it can compress).

## 5 Adding guardrails for front-facing applications

The ability to enforce guardrails when it comes to AI generation is important for front-facing applications. In this section, we highlight how to leverage system prompting to optionally enforce output constraints on top of our models. Additionally, we showcase the ability of Mistral 7B to perform fine-grained content moderation, which can be useful to enforce quality content in applications.

### 5.1 System prompt to enforce guardrails

We introduce a system prompt (see below) to guide the model to generate answers within specified guardrails, similar to the work done with Llama 2. Using this prompt allows the user to move on the Pareto front of model utility / guardrails enforcement, as indicated in Table 4.

Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.

We use a set of 175 unsafe prompts for evaluating safety. With the recommended system prompt, the model properly declines to answer $100 \%$ of the harmful questions.

As an illustration, we provide in Table 5 the answers of both Mistral 7B - Instruct and Llama 2 Chat 13B to the question How to kill a linux process with system prompts activated. We observe that Mistral 7B provides a correct response while Llama 2 declines to answer. Note that on this specific question, both models answer correctly when system prompts are deactivated.

| Guardrails | MT Bench |
| :---: | :---: |
| No system prompt | $6.84 \pm 0.07$ |
| Llama 2 system prompt | $6.38 \pm 0.07$ |
| Mistral system prompt | $6.58 \pm 0.05$ |

Table 4: System prompts. Mean official MT Bench score over 10 iterations with standard deviation for Mistral 7B - Instruct. For reference, Llama 2 13B - Chat reports official results of 6.65 .

### 5.2 Content moderation with self-reflection

Mistral 7B - Instruct can be used as a content moderator: the model itself is able to accurately classify a user prompt or its generated answer as being either acceptable or falling into one of the following categories: Illegal activities such as terrorism, child abuse or fraud; Hateful, harassing or violent content such as discrimination, self-harm or bullying; Unqualified advice for instance in legal, medical or financial domains.

| Model | Answer |
| :--- | :--- | | To kill a Linux process, you can use the ` kill` command followed by the process ID (PID) |
| :--- |
| of the process you want to terminate. For example, to kill process with PID 1234, you |
| would run the command `kill 1234`. It's important to note that killing a process can |
| have unintended consequences, so it's generally a good idea to only kill processes that |
| you are certain you want to terminate. Additionally, it's generally a good idea to use the |
| `kill`command with caution, as it can be used to terminate critical system processes that |
| are necessary for the operation of the system. |

Table 5: Comparison between Mistral and Llama system prompts. Mistral provides the right answer whereas Llama 2 declines to answer to the question How to kill a linux process.

To do so, we designed a self-reflection prompt that makes Mistral 7B classify a prompt or a generated answer. We evaluated self-reflection on our manually curated and balanced dataset of adversarial and standard prompts and got a precision of $99.4 \%$ for a recall of $95.6 \%$ (considering acceptable prompts as positives).

The use cases are vast, from moderating comments on social media or forums to brand monitoring on the internet. In particular, the end user is able to select afterwards which categories to effectively filter based on their particular use-case.

## 6 Conclusion

Our work on Mistral 7B demonstrates that language models may compress knowledge more than what was previously thought. This opens up interesting perspectives: the field has so far put the emphasis on scaling laws in 2 dimensions (directly associating model capabilities to training cost, as in [14]); the problem is rather 3 dimensional (model capabilities, training cost, inference cost), and much remains to be explored to obtain the best performance with the smallest possible model.

## Acknowledgements

We are grateful to CoreWeave for their $24 / 7$ help in marshalling our cluster. We thank the CINECA/EuroHPC team, and in particular the operators of Leonardo, for their resources and help. We thank the maintainers of FlashAttention, vLLM, xFormers, Skypilot for their precious assistance in implementing new features and integrating their solutions into ours. A huge thanks to Tri Dao and Daniel Haziza for helping include Mistral related changes to FlashAttention and xFormers on a tight schedule. We thank the teams of Hugging Face, AWS, GCP, Azure ML for their intense help in making our model compatible everywhere.

# Can you recommend a foundational and accessible book about quantum physics? 

\author{

| Copy Link to Question | Next Round (Enter) $\rightarrow$ |
| :--- | :--- |

}

Round Winner: mistral-7b-instruct-v0.1

![](https://cdn.mathpix.com/cropped/2024_05_29_c1e08f80a92ab8fdeda7g-7.jpg?height=309&width=301&top_left_y=995&top_left_x=923)

mistral-7b-instruct-v0.1 Round Winner

A great book for a foundational understanding of quantum physics is "The Quantum Universe: Everything that Can Happen Does Happen" by Brian Cox and Jeff Forshaw. This book is accessible and easy to understand, making it a great choice for beginners. It covers all the basics of quantum mechanics, including superposition, entanglement, and the uncertainty principle. The authors also explore the implications of quantum physics for our understanding of the universe, including the nature of time and space. Overall, "The Quantum Universe" is an excellent choice for anyone looking to gain a foundational understanding of this fascinating field
Llama-2-13b-chat

Certainly! For a foundational and accessible introduction to quantum physics, I highly recommend "The Feynman Lectures on Physics" by Richard Feynman. This classic text provides a clear and engaging overview of the principles and concepts of quantum mechanics, with minimal mathematical formalism and plenty of illustrations and examples. It's a great starting point for anyone looking to gain a deeper understanding of quantum physics without getting bogged down in too much technical detail.

Figure 6: Human evaluation of Mistral 7B - Instruct vs Llama 2 13B - Chat Example. An example of human evaluation from llmboxing. com. The question asks for recommendations of books in quantum physics. Llama 2 13B - Chat recommends a general physics book, while Mistral 7B - Instruct recommends a more relevant book on quantum physics and describes in the contents in more detail.

## References

[1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

[2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[3] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

[4] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, 2020.

[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[6] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

[7] Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. Quac: Question answering in context. arXiv preprint arXiv:1808.07036, 2018 .

[8] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

[9] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

[10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[11] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.

[12] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

[13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

[14] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karén Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems, volume 35, 2022.

[15] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.

[16] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453-466, 2019.

[17] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[18] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. https://github.com/ facebookresearch/xformers, 2022.

[19] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.

[20] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

[21] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99-106, 2021.

[22] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

[23] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

[24] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.

[25] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[26] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[27] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[28] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

[29] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.


[^0]:    ${ }^{1}$ https://github.com/mistralai/mistral-src

    ${ }^{2}$ https://github.com/skypilot-org/skypilot

    ${ }^{3}$ https://huggingface.co/mistralai

[^1]:    ${ }^{4}$ Since Llama 2 34B was not open-sourced, we report results for Llama 1 34B.

</end of paper 2>


