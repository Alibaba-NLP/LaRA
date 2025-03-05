<paper 0>
# LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks 

Subbarao Kambhampati * Karthik Valmeekam Lin Guan Kaya Stechly<br>Mudit Verma Siddhant Bhambri Lucas Saldyt Anil Murthy<br>School of Computing \& AI, Arizona State University


#### Abstract

There is considerable confusion about the role of Large Language Models (LLMs) in planning and reasoning tasks. On one side are over-optimistic claims that LLMs can indeed do these tasks with just the right prompting or self-verification strategies. On the other side are perhaps over-pessimistic claims that all that LLMs are good for in planning/reasoning tasks are as mere translators of the problem specification from one syntactic format to another, and ship the problem off to external symbolic solvers. In this position paper, we take the view that both these extremes are misguided. We argue that auto-regressive LLMs cannot, by themselves, do planning or self-verification (which is after all a form of reasoning), and shed some light on the reasons for misunderstandings in the literature. We will also argue that LLMs should be viewed as universal approximate knowledge sources that have much more meaningful roles to play in planning/reasoning tasks beyond simple front-end/back-end format translators. We present a vision of LLM-Modulo Frameworks that combine the strengths of LLMs with external model-based verifiers in a tighter bi-directional interaction regime. We will show how the models driving the external verifiers themselves can be acquired with the help of LLMs. We will also argue that rather than simply pipelining LLMs and symbolic components, this LLM-Modulo Framework provides a better neurosymbolic approach that offers tighter integration between LLMs and symbolic components, and allows extending the scope of model-based planning/reasoning regimes towards more flexible knowledge, problem and preference specifications.


## 1 Introduction

Large Language Models (LLMs), essentially n-gram models on steroids which have been trained on web-scale language corpora (or, effectively, our collective consciousness), have caught the imagination of the AI research community with linguistic behaviors that no one expected text completion systems to possess. Their seeming versatility has led many researchers to wonder whether they can also do well on planning and reasoning tasks typically associated with System 2 competency. On the face of it, this doesn't seem to ring true, as both by training and operation, LLMs are best seen as a giant pseudo System 1 (Kahneman, 2011) (see Figure 11. Even from a pure engineering perspective, a system that takes constant time to produce the next token cannot possibly be doing principled reasoning on its own. ${ }^{1}$ Not surprisingly, initial excitement based on anecdotal performance of LLMs on reasoning tasks (Bubeck et al. 2023) has dissipated to some extent by the recent spate of studies questioning the robustness of such behaviors-be it planning (Valmeekam et al. 2023c, Kambhampati,[^0]

2023), simple arithmetic and logic (Dziri et al., 2023), theory of mind abilities (Ullman, 2023, Verma et al. 2024), or general mathematical and abstract benchmarks (McCoy et al. 2023; Gendron et al. 2023). Despite this, a steady stream of claims continue to be made in the literature about the planning and reasoning capabilities of LLMs.

In an ironic juxtaposition to this unwarranted optimism about the planning and reasoning abilities of LLMs, there is also unwarranted pessimism about the roles LLMs can play in planning/reasoning tasks. Several efforts (e.g. (Liu et al., 2023, Pan et al., 2023, Xie et al., 2023)) advocate using LLMs only as glorified translators-for converting reasoning problems embedded in textual format to symbolic representations, which are then pawed off to external classical symbolic solvers (with all their attendant expressivity and search complexity challenges (Doyle \& Patil, 1991)). ${ }^{2}$

![](https://cdn.mathpix.com/cropped/2024_06_04_391d351144ab2b430cb9g-02.jpg?height=702&width=1328&top_left_y=749&top_left_x=388)

Figure 1: An informal account of viewing LLM as a giant external non-veridical memory that acts as a pseudo System 1

In truth, LLMs are a whole lot more than machine translators. They are a kind of approximate knowledge sources (albeit sans guarantees) trained on our collective consciousness. While it is unlikely that they will have System 2 competencies by themselves, they can nevertheless be valuable resources in solving System 2 tasks. To put it another way, the problem with Alchemy of yore is not that Chemistry is useless, but that people wanted to delude themselves that Chemistry-a pretty amazing discipline on its own merits-can be Nuclear Physics if you prompt it just so. The confusions regarding LLM abilities, or should we say, LLM alchemy, seems to be not that much different-oscillating between ignoring what they are good at, and ascribing them abilities they don't have.

The goal of this position paper is to introduce some clarity into this confusing state of affairs oscillating between over-optimism and over-pessimism. Simply put, we take the stance that LLMs are amazing giant external non-veridical memories that can serve as powerful cognitive orthotics for human or machine agents, if rightly used. The underlying n-gram nature makes them effortlessly intermix what would be considered disparate fields of study (not surprisingly, LLMs are seen to be very good at making/finding analogies!) The challenge is to leverage them without wrongly ascribing to them capabilities they don't possess. The LLM-Modulo framework proposed in this position paper tackles this challenge.

For the sake of concreteness, we focus on planning tasks, especially as studied in the automated planning community (Ghallab et al. 2004). The central position of the paper is that LLMs cannot[^1]plan themselves but can play a variety of constructive roles in solving planning tasks-especially as approximate knowledge sources and candidate plan generators in the so-called LLM-Modulo Frameworks in conjunction with external sound model-based verifiers.

We support this position by first reviewing literature that establishes that LLMs cannot be used as planners or plan verifiers themselves (Section 2). We also discuss why there are claims about planning/verification abilities in the first place, in the process hopefully clarifying some prevalent misunderstandings.

Second, we will propose a framework that allows us to leverage LLMs effectively in planning tasks, by combining them with external critics, verifiers and humans. We call this an LLM-Modulo Framework (a name loosely inspired by SAT Modulo Theories (Nieuwenhuis \& Oliveras, 2006)); see Figure 3 LLMs play a spectrum of roles in this architecture, from guessing candidate plans, to translating those plans into syntactic forms that are more accessible to external critics, to helping end users flesh out incomplete specifications, to helping expert users acquire domain models (that in turn drive model-based critics). All this leveraging of LLMs is done without ascribing to them any planning or verification abilities. The LLM ideas are vetted by external critics, thus ensuring that the plans generated in this architecture can have formal correctness guarantees where possible.

One question regarding this LLM-Modulo architecture from a planning perspective is whether it is more than a gratuitous attempt to shoe-horn LLMs to solve planning problems, when there are more formal combinatorial planning systems available in multiple communities (Ghallab et al. 2004). Compared to a planner that is guaranteed to be correct in a narrow set of domains, LLMs may likely be good at generating plausible (but not guaranteed to be correct) plan heuristics/suggestions in many more scenarios. Thus, unlike the traditional planning architectures studied in AI (Ghallab et al. (2004), which put a priori constraints on the expressiveness of the problems that can be posed to the planner (to wit, the different expressiveness levels of the PDDL specification (McDermott et al. 1998)), the LLM-Modulo architecture puts no such restrictions. In this sense, it is more representative of real-world planning problems such as those in NASA mission planning, where the different critics-human and automated-are at best able to give "no objection" certificates, clearing it from their perspective. (Indeed, both deep space network planning and mars rover task planning are done via a collective human blackboard. (Johnston et al., 2014) (Bresina et al., 2004).) Note that this is starkly different from just sending an unvetted plan out to execution (as would be the case if we have LLMs operate in autonomous mode to guess plans). Generalizing planning and reasoning frameworks this way is consistent with the Doyle \& Patil's call to the Knowledge Representation community of yore (Doyle \& Patil 1991).

## 2 Planning-centered Limitations of LLMs

In this section, we will first review the literature that calls into question the claims about planning and self-verification capabilities of LLMs. Subsequently, we will also provide some possible reasons for the claims to the contrary made in the literature.

### 2.1 LLMs cannot generate executable plans in autonomous mode

Despite the initial claims about the planning capabilities of LLMs (Bairi et al., 2023, Yao et al., 2023b, Shinn et al., 2023, Huang et al., 2022, Hao et al., 2023) several recent studies independently confirm that LLMs are not actually able to generate executable plans when they are used in autonomous modes (Valmeekam et al. 2023c, Liu et al., 2023, Silver et al., 2022). For example, in (Valmeekam et al. $2023 \mathrm{c}$ b), the authors evaluate LLMs ability to generate correct plans on a suite of planning problem instances based on the kinds of domains employed in the International Planning Competition (IPC, 1998). To eliminate the subjective aspect of analysis that forms the core part of many earlier efforts on evaluating the reasoning capabilities of LLMs, they automate the evaluation by leveraging models and tools from the automated planning community.

They show that the results in the autonomous mode are pretty bleak. On average, only about $12 \%$ of the plans that the best LLM (GPT-4) generates are actually executable without errors and reach their goals. They show that the choice of the specific LLM (they have tested the family of GPT LLMs including GPT-4 (OpenAI, 2023), GPT-3.5 (OpenAI, 2022), InstructGPT-3 (Ouyang et al., 2022) and GPT-3 (Brown et al. |2020)). They also sho that fine-tuning does not seem to have a major effect
on this dismal performance. They also show that the performance deteriorates further if the names of the actions and objects in the domain are obfuscated-a change that doesn't in any way affect the performance of the standard AI planners. This latter further suggests that LLMs are more likely doing approximate retrieval of plans than planning.

### 2.2 LLMs cannot verify plans and thus cannot improve by self-critiquing

There still exists considerable optimism that even if LLMs can't generate correct solutions in one go, their accuracy might improve in an iterative prompting regime, where LLMs will be able to "self-critique" their candidate solutions and refine them to the point of correctness (Yao et al., 2023b a, Shinn et al., 2023, Weng et al., 2023; Huang et al. 2022). This belief seems to rest largely on the assumption that verification of correctness should be easier than generation for many reasoning problems-a rather classical argument from computational complexity. There are grounds to be skeptical of this assumption as the complexity of the reasoning task should be irrelevant to LLM performance if what they are doing is approximate retrieval. In general, unless LLMs are trained not just on "correct data," but also "corrections data," there is no a priori reason to believe that their critiques would even be approximately relevant, let alone actually correct.

Two recent studies-one on plan verification (Valmeekam et al. 2023a) and the other on CSP verification (Stechly et al. 2023) seem to throw cold water on this optimism. In (Stechly et al., 2023), the authors systematically investigate the effectiveness of iterative prompting in the context of Graph Coloring, a canonical NP-complete reasoning problem. Their methodology involves a principled empirical study of the performance of GPT4 on two tasks: solving a large suite of random graph coloring instances and, separately, verifying the correctness of the candidate colorings-both in direct (i.e., return the first solution generated by the LLM) and iterative modes. In iterative modes, they experiment both with an LLM critiquing LLM-produced solutions and an external, guaranteed correct reasoner verifying solutions. In both cases, they analyze whether the content of criticisms actually affects bottom-line performance.

Their results indicate that in direct mode, LLMs are, perhaps not surprisingly, pretty bad at solving graph coloring instances. More interestingly, they are no better at verifying solutions. In iterative modes, given the inability of LLMs to verify solutions, it should come as no surprise that their experiments also show that the strategy of LLMs self-critiquing their solutions does not improve over the baseline. They report that the perforance is in fact worse because the system can't recognize a correct coloring and thus merrily passes over fortuitously correct colorings it has generated, ending up with a wrong one! Similar results have also been reported for planning problems in (Valmeekam et al., $2023 \mathrm{c}$ ).

One important corollary of the fact that LLMs cannot self-critique their plans is that they can't also self-improve by generating synthetic data by generating plans themselves, critiquing the plans by themselves to improve them, and then using those to fine-tune themselves, as has been claimed in the literature (Wang et al., 2022) ${ }^{3}$ (Huang et al. 2023b); see Section 3.3 .

### 2.3 Analyzing Claims to the Contrary in the Literature

Given that LLMs can neither guarantee correct generation nor correct verification of plans, as discussed in the previous sections, one obvious question is why the literature is replete with claims contrary to this (Bairi et al., 2023; Yao et al., 2023b; Shinn et al. 2023; Yao et al., 2023a, Weng et al., 2023; Huang et al. 2022).

Claims about Planning: To analyze the planning claims, we need to first understand that solving planning tasks requires (a) having the necessary planning domain knowledge-the actions and their preconditions, effects; the standard hierarchical recipes (e.g. task reduction schemas in HTN planning), past cases/plans, etc., and (b) being able to assemble this planning knowledge into an executable plan that takes care of any subgoal/resource interactions. The first can be called the knowledge acquisition and the second reasoning/planning part. Many of the papers claiming planning abilities of LLMs, on closer examination, wind up confusing general planning knowledge extracted from the LLMs for executable plans. When all we are looking for are abstract plans, such as "wedding[^2]plans," with no intention of actually executing the said plans, it is easy to confuse them for complete executable plans. Indeed, our close examination of several works claiming planning capabilities for LLMs (Kambhampati et al. 2023) suggests that they either work in domains/tasks where subgoal interactions can be safely ignored (Yao et al. 2023b; Shinn et al. 2023) ${ }^{4}$-either because they are just working on a single goal, or because the world is forgiving and ergodic; or delegate the interaction resolution (reasoning) to the humans in the loop (who, through repeated prompting, have to "correct" the plan). Sometimes, in common sense domains, or with enough fine-tuning, the "assembling" part may also be obviated by having seen a case that pretty much corresponds to the problem that needs to be solved. Not surprisingly, the work by (Valmeekam et al. 2023c) shows that if the action interactions are removed by relaxing the world models, then the ability of LLMs to guess executable plans improves. Without these assumptions or mitigations, the plans that come out of LLMs may look reasonable to the lay user, and yet lead to execution time interactions and errors. ${ }^{5}$

The fact that LLMs are often good at extracting planning knowledge can indeed be gainfully leveraged. As shown in recent works (Guan et al. 2023), LLMs can be a rich source of approximate models of world/domain dynamics and user preferences, as long as the humans (and any specialized critics) in the loop verify and refine those models, and give them over to model-based solvers. This way of using LLMs has the advantage that the humans need only be present when the dynamics/preference model is being teased out and refined, and the actual planning after that can be left to sounder planning frameworks with correctness guarantees, such as LLM-Modulo framework we propose.

![](https://cdn.mathpix.com/cropped/2024_06_04_391d351144ab2b430cb9g-05.jpg?height=686&width=1331&top_left_y=1077&top_left_x=381)

Figure 2: Viewing LLMs as an approximate knowledge source trained over civilizational knowledge

Such an overall approach has striking similarities to knowledge-based AI systems of yore, with LLMs effectively replacing the "knowledge engineer" (see Figure 2). Given the rather quixotic and dogmatic shift of AI away from approaches that accept domain knowledge from human experts that some writers termed "Polanyi's Revenge" (c.f. (Kambhampati, 2021)), this new trend of using LLMs as knowledge sources can be viewed as a form of avenging Polanyi's revenge! Indeed, LLMs make it easy to get problem-specific knowledge as long as we are willing to relax the correctness requirements of that knowledge. In contrast to the old knowledge engineering approaches, LLMs offer this without making it look like we are inconveniencing any specific human (we are, instead, just leveraging everything humans told each other on the Web!). So the million dollar question for reasoning tasks is:[^3]"how would you do robust planning if you have some doddering know-it-all ready to give you any kind of knowledge?" The LLM-Modulo Framework is a principled method for tackling this challenge.

![](https://cdn.mathpix.com/cropped/2024_06_04_391d351144ab2b430cb9g-06.jpg?height=748&width=1374&top_left_y=374&top_left_x=365)

Figure 3: The proposed LLM-Modulo framework where LLMs act as idea generators and various external critiques that specialize in different aspects, critique the candidate plan.

Claims about Self-Verification: Coming to the claims about LLM's self-verification abilities, a closer look at the literature (Yao et al., 2023a; Huang et al., 2023a) shows that those claims are either (i) made in the context of tacit knowledge tasks for which there is little possibility of a verifier (e.g. essay writing)-making it hard to evaluate whether LLM's critiquing actually helped ${ }^{6}$ or (ii) the external verification is carried out either by simulators (Wang et al., 2023, Yao et al. 2023b) or simple calls to the underlying operating system (as is the case, for example, for the 24 puzzle in (Yao et al., 2023a)).

In a related vein, there is the recent Tree of Thoughts (ToT) paper (Yao et al. 2023a), which has been pitched as a way to convert LLMs into some type of systematic search with self-verification. A closer look at the work however shows that ToT simply iteratively back-prompts the LLM until it comes up with a solution that is acceptable to an external verifier. Specifically, ToT employs a problem-specific prompt priming method. The "tree" in ToT is essentially a way to generate diverse priming prompts (that the authors set up in a problem specific way). In other words, despite the use of terminology of problem-solving agents (Russell \& Norvig, 2010)-search tree, expansion etc., there is really no deeper connection to search-based agents.

The guarantees-if any-are coming in terms of soundness of the external verifier. The one clear reasoning problem used in the ToT paper is the 24 puzzle-for which the external verifier can be easily implemented in terms of arithmetic operations (thankfully not done by the numerically challenged LLM!). Here, our experiments show that LLM's own criticisms are often quite off the mark. ${ }^{8}$ Because the 24 puzzle's solutions can be verified by simple arithmetic operations, readers don't quite realize that the framework relies on an external verifier. In general though, the verifier may be more complex and can involve substantial work (you can substitute a simulator for the verifier-but someone has to write that simulator too!)[^4]

In general planning problems, one way to provide an external verifier is to (a) write a domain model (e.g. in PDDL) and (b) feed it to an off-the-shelf model-based verifier like VAL (c.f. (Howey et al., 2004)).

## 3 LLM-Modulo Framework for Robust Planning

While Section 2 questions the claims that LLMs are capable of planning/reasoning by themselves, it is certainly not meant to imply that LLMs don't have any constructive roles to play in solving planning/reasoning tasks. On the contrary, as discussed in the Introduction, their uncanny ability to generate ideas/potential candidate solutions-albeit with no guarantees about those guesses-can be valuable in the generate-test-critique setups in conjunction with either model-based verifiers or expert humans in the loop. Accordingly, we propose a general "LLM-Modulo" framework". While we believe that versions of such an architecture can be of use in a wide variety of planning or reasoning tasks, for the sake of concreteness, we will focus on planning tasks, especially of the type studied in the automated planning community (Ghallab et al. 2004).

Figure 3 gives a schematic of the LLM-Modulo Framework, as we envision it. As can be seen readily, the underlying architecture is a Generate-Test-Critique loop, with the LLM generating candidate plans and a bank of critics critiquing the candidate. The loop starts with the LLM getting the problem specification and generating its first plan candidate. ${ }^{10}$ Note that the plans an LLM helps generate in this architecture have soundness guarantees because of the external sound critics. This means that plans coming out of such an architecture will constitute a better corpus of synthetic data for any fine tuning phase carried out to improve/customize the LLM's generation capability.

Design Considerations: Before going into the details about the framework and its various modules, it is worth noting some design decisions underlying the proposed architecture. We start by noting that the LLM-Modulo architecture is a "Generate-Test" one that involves LLMs interacting with the external critics rather than solvers. This is a deliberate decision-as this allows the LLM to guess/generate candidates to satisfy the critics, as against dealing with the expressiveness and search complexity issues of the solvers. Secondly, the framework explicitly recognizes that the LLMs can generate approximate ideas not just about plan candidates, but domain models, problem reduction strategies, and refinements to the problem specification. The framework also recognizes that LLMs are good at format/syntax changes. Accordingly, the framework leverages all these abilities of LLMs, letting them play multiple roles in planning. Finally, the architecture carefully circumscribes the human's role-domain experts interact with the LLM to tease out the models used by (some of) the critics, while end users take part in refinining any incomplete problem specification in concert with the LLM. A notable, and deliberate, absence is human's involvement in the inner loop of planning-e.g. with iterative prompting. In addition to posing an infeasible burden on the human's time for complex planning problems, such iterative prompting strategies are notorious for their Clever Hans effect (cle).

### 3.1 Critics

In the LLM-Modulo framework, critics can evaluate LLM-generated candidates for a planning/reasoning problem over both hard and soft constraints. Hard constraints refer to correctness verification which can include causal correctness, timeline correctness, resource constraint correctness, etc. For PDDL planning problems, the hard critic can be based on VAL (Howey et al. 2004), that works off of a model (which itself can be acquired with the help of the LLM (Guan et al., 2023). On the other hand, soft constraints can include more abstract notions of correctness such as style, explicability, preference conformance, etc. As discussed in Section 2.3, while LLMs cannot take on the role of hard critics, they can help simulate some aspects of the role of soft critics. So our framework does allow for style critics be possibly based on LLMs (e.g (Verma et al., 2024)). We reiterate that the soundness of the LLM-modulo framework is inherited from the soundness of the critics.

The bank of critics-hard (model-based) as well as soft (possibly LLM-based) evaluate the current plan candidate to evaluate its fitness/acceptability. If at least all the hard critics sign off on the current candidate, then that is considered a valid solution to be returned to the end-user or the executor. When[^5]a critic finds the current plan candidate to be unsatisfactory, it can provide varying levels of feedback, ranging from "No, try again" to "No, try again, here is one thing wrong with the current plan" to "No, try again, here are all the things wrong with the current plan. These critiques are all pooled at the Backprompt Controller (see Section 3.2

### 3.1.1 LLMs as Reformulators

One interesting challenge is that many of the symbolic model-based verifiers tend to be operating on specialized formal representations. Given a central candidate plan (e.g. a mission plan), these critics need translations of that candidate into their representations. This is the role of the reformulator module attached to individual critics. These reformulator modules can be supported to large extent by LLMs, given that one thing LLMs are very good at is format change across different syntactic representations, Olmo et al. 2021). Indeed, as discussed in the Introduction, some approaches to combine LLMs with external symbolic solvers just use LLMs as reformulators for these solvers (Liu et al. 2023, Pan et al., 2023). Our discussion of LLM-Modulo framework should make it clear that syntax reformulation alone is a severely limited role for LLMs!

### 3.2 Backprompt (Meta) Controller

The critiques from the various critics are pooled together by the Meta (Backprompt) Controller, which passes a processed version of them to the LLM as the next iterative prompt to elicit the next guess. This is especially required in the presence of a mix of soft and hard critics, where the Meta Controller can assume the responsibility of compiling the critiques into a consistent feedback to process.

The processing steps taken in the controller can range from simple round-robin selection of prompts to generating a summarized prompt (with LLM help) ${ }^{11}$ to employ a prompt diversification strategy to elicit the next candidate from a different part of the implicit search space (akin effectively to the strategy used in systems such as Tree of Thoughts prompting (Yao et al. 2023a), as discussed in 2.3).

### 3.3 Fine Tuning \& Synthetic Data

Once the LLM-Modulo framework "solves" a planning instance, the solution can then be added to a synthetic data corpus (step 6 in Figure 3), which is intermittently used to fine tune the LLM (step 7), so its future plan candidate guesses improve.

Such fine tuning on task-specific data has been a popular way to get LLMs to improve their performance on reasoning/planning tasks. For example, fine tune the LLM on blocks world planning problem-solution pairs to improve their performance in guessing solutions for blocks world instances (Pallagani et al. 2023). While fine tuning still doesn't guarantee correctness of the generated solutions, it might improve the chances that LLM guesses candidates that are closer to being vetted by the bank of critics.

One important question is where this additional data for fine tuning comes from. A tempting idea is to have the LLM itself generate this additional data, and improve it by self-critiquing/verification, before fine-tuning itself on the data. This EM-like approach unfortunately will not work given that LLMs can't verify their own solutions (see Section 2.2). In the past, this meant that the only reliable way to generate synthetic data is to use external plan generators-for example, use a classical planner like FF (Hoffmann \& Nebel 2001) to solve a host of blocks world instances and then use those solutions to fine-tune the LLM. The LLM-Modulo framework, in contrast, provides an alternative way of using an LLM-based framework to generate synthetic data that is guaranteed correct.

### 3.4 Specification Refinement \& Model Acquisition (Semi-automated)

As mentioned earlier, we avoid having humans involved in iteratively prompting LLMs-as this can be an infeasibly time-consuming activity for them. Instead, we let automated verifiers, either model-based or LLM-supported, to manage the plan critiquing process. The framework does depend on humans for "once per domain" and "once per problem" interactions. In the former category, human domain experts can play a role in acquiring the domain model with the help of the LLM.[^6]

Examples of such interaction include teasing out PDDL planning models from the LLMs with the help of human expert curation (top left in Figure 3). The idea here is that the traditional domain model acquisition task (e.g. (sim, 2001) ) is significantly made easier by having the LLMs help with ideas regarding various pieces of the domain model (e.g., actions, their preconditions and effects) and letting humans sign off/critique the resulting model. Once the model is acquired this way, it can be used by correctness verifiers such as VAL (Howey et al. 2004, Guan et al., 2023). Often the planning problems in real world situations are specified incompletely, leaving it to the human commonsense to refine the specification. This brings up a second role for humans-this time end users (bottom left in Figure 3-in collaboratively refining the specification with the help of LLMs (similar to the way done in (Xie et al. 2023, Liu et al. 2023).

### 3.5 Summary of Roles of LLMs in the LLM-Modulo Framework

It is worth summarizing the multiple roles the LLM plays in the LLM-Modulo architecture. The most prominent, of course, is its role in "guessing" the candidate plans (step 2 in Figure 3" in response to problem specification and iterative back prompting from the bank of critics (Step 5). Second the LLM plays a role in converting the guessed plan candidate into specialized representations used by the various critics (e.g., the time-line view, the causal link view etc.). This role leverages the fact that LLMs are very good at format conversion (c.f. (Olmo et al., 2021)) Third, the LLM plays a role in helping the end user flesh out the incomplete problem specification to begin with (Step 1 in Figure 3). Finally, the LLM plays a role in helping the domain expert tease out and refine the domain models used by the various model-based critics (Guan et al., 2023; Kwon et al., 2022).

## 4 Related Work

We will note that while the LLM-Modulo framework is being proposed in general form here for the first time, there are certainly works in leveraging LLMs in planning and reasoning tasks that are in line with the spirit of the LLM-Modulo framework. For exmaple, both (Valmeekam et al. 2023c) and (Stechly et al., 2023) describe and evaluate a backprompting interaction between an LLM and an external verifier. Work on FunSearch (Romera-Paredes et al., 2023) depends on a generate-test loop between a specially fine-tuned LLM that guesses solutions, and an external symbolic evaluator that critiques them. The authors note how the external verifier is critical for avoiding falling prey to hallucinations (i.e., approximate solution candidates that have flaws). AlphaGeometry (Trinh et al. (2024) too depends on the Generate-Test-Critique interaction between a fine-tuned LLM and a symbolic evaluator. Both these systems fine-tune pre-trained LLMs with task specific synthetic data-the correctness of which is vetted with external simulators (as we discuss in Section 3.3.

While we focused on PDDL planning tasks for the sake of concreteness, we believe that the essence of LLM-Modulo framework is equally applicable to other scenarios involving planning and reasoningsuch as Reinforcement Learning with Simulators. Such RL systems rely on rewards as feedback to train a policy. Simulators takes on the roles of plan evaluation and critiques performed by the respective critics in the LLM-Modulo framework (e.g. (Rajvanshi et al. 2023)). The fact that simulators play the role of verifiers is often not explicitly recognized in cases where LLMs are used as an actor to generate an admissible plan by interacting with a simulator, for example in the case of AlfWorld (Yao et al., 2023b; Shinn et al., 2023) and Minecraft (Wang et al., 2023). As mentioned in Section 3, similar to extracting a domain model such as in the case of PDDL planning (Guan et al. 2023), designing a reward model for the plan generation - feedback cycle is yet another potential use case that has been recently looked at for text-based (Kwon et al., 2022; Hao et al., 2023) and robot manipulation (Ma et al. 2023) domains.

Interestingly, the fact that LLM's can help come up with approximate quasi-symbolic transition models, reward models and models of high level actions has made a bigger splash in RL. This is because for far too long, researchers there have tried to spurn any high level models (lest that would involve depending on humans; (Kambhampati, 2021) and focused on learning to act from sensory information, under the name of "deep reinforcement learning." Given the horrendous sample complexity of the DRL methods even in reaching a single subgoal, and the well known fact that even approximate symbolic models can help drastically improve the performance (c.f. (Guan et al. 2022)), coupled with the fact that LLM's are only too glad to dream up approximate models and goal recipes, there has been a performance revolution of sorts there (Yao et al., 2023b; Liang et al.

2023, Wang et al. 2023). If we look beyond the improvements in these lower level goal seeking behaviors-especially in the presence of ergodic simulators, the RL approaches dependent on LLMs will encounter the same issues regarding subgoal interactions that our discussion of PDDL planning problems brought into focus. The LLM-Modulo inspired frameworks will thus, we believe, be equally relevant there. Indeed, SayCan (Ahn et al. 2022) the earliest use of LLMs in generating policies in an RL-with-Simulator scenario, explicitly filters the action choices suggested by the LLM with the help of simulator.

While we focused on text based LLMs (such as GPT4), recently there have also been impressive development in multi-modal LLMs (e.g. GPT4V). While multi-modality is a great addition that increases the coverage of their System 1 imagination (Figure 1), it is not clear that this gives them System 2 competence. ${ }^{12}$

## 5 Conclusion

This position paper is a modest attempt to combat both over-optimism and over-pessimism about the role of LLMs in planning and reasoning tasks. Our position is that LLMs cannot plan themselves but can play a variety of constructive roles in solving planning tasks-especially as approximate knowledge sources and candidate plan generators in the so-called LLM-Modulo Frameworks in conjunction with external sound model-based verifiers. In support of this position, we summarized the literature questioning the claims about the planning and self-verification capabilities of LLMs by themselves. We also discussed how conflating approximate knowledge acquisition and generating executable plans of action is behind many of the claims about planning and verification abilities of LLMs. We then shared LLM-Modulo framework, our vision for a productive way to integrate the impressive idea generation/approximate knowledge provision capabilities of LLMs with external verifiers with correctness guarantees for robust and expressive planning. We discussed how planning in LLM-Modulo framework avoids inheriting the expressiveness and search-complexity limitations of traditional symbolic planners, while retaining their soundness guarantees. As we discussed, LLM-Modulo frameworks are consistent with some of the most high-profile success stories of "neuro-symbolic" architectures, including AlphaGeometry and FunSearch.

## Acknowledgments

The ideas discussed in this paper have evolved over a series of talks, tutorials and twitter threads. The discussions, feedback and encouragement from colleagues, including Daniel Borrajo, Tom Dietterich, Yann LeCun, Sarath Sreedharan, and Dan Weld is gratefully acknowledged.

## References

Clever Hans. https://en.wikipedia.org/wiki/Clever_Hans.

Gipo: an integrated graphical tool to support knowledge engineering in ai planning. In $E C P-01, \mathrm{pp}$. 445. Citeseer, 2001.

Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

Bairi, R., Sonwane, A., Kanade, A., Iyer, A., Parthasarathy, S., Rajamani, S., Ashok, B., Shet, S., et al. Codeplan: Repository-level coding using llms and planning. arXiv preprint arXiv:2309.12499, 2023.

Bresina, J. L., Jónsson, A. K., Morris, P. H., and Rajan, K. Activity planning for the mars exploration rovers. In ICAPS-2005 Conference, 2004.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.[^7]

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.

Doyle, J. and Patil, R. S. Two theses of knowledge representation: Language restrictions, taxonomic classification, and the utility of representation services. Artificial intelligence, 48(3):261-297, 1991 .

Dziri, N., Lu, X., Sclar, M., Li, X. L., Jiang, L., Lin, B. Y., Welleck, S., West, P., Bhagavatula, C., Bras, R. L., Hwang, J. D., Sanyal, S., Ren, X., Ettinger, A., Harchaoui, Z., and Choi, Y. Faith and fate: Limits of transformers on compositionality. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=Fkckkr3ya8.

Gendron, G., Bao, Q., Witbrock, M., and Dobbie, G. Large language models are not abstract reasoners. arXiv preprint arXiv:2305.19555, 2023.

Ghallab, M., Nau, D., and Traverso, P. Automated Planning: theory and practice. Elsevier, 2004.

Guan, L., Sreedharan, S., and Kambhampati, S. Leveraging approximate symbolic models for reinforcement learning via skill diversity. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 7949-7967. PMLR, 17-23 Jul 2022. URLhttps://proceedings.mlr.press/v162/guan22c.html

Guan, L., Valmeekam, K., Sreedharan, S., and Kambhampati, S. Leveraging pre-trained large language models to construct and utilize world models for model-based task planning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=zDbsSscmuj

Hao, S., Gu, Y., Ma, H., Hong, J. J., Wang, Z., Wang, D. Z., and Hu, Z. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023.

Hoffmann, J. and Nebel, B. The FF planning system: fast plan generation through heuristic search. Journal of Artificial Intelligence Research, 14:253-302, 2001.

Howey, R., Long, D., and Fox, M. VAL: Automatic plan validation, continuous effects and mixed initiative planning using PDDL. In 16th IEEE International Conference on Tools with Artificial Intelligence, pp. 294-301. IEEE, 2004.

Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., and Zhou, D. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798, 2023a.

Huang, J., Gu, S., Hou, L., Wu, Y., Wang, X., Yu, H., and Han, J. Large language models can self-improve. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 1051-1068, Singapore, December 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.67. URL https://aclanthology.org/2023.emnlp-main.67.

Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022.

IPC. International planning competition, 1998. URL https://www.icaps-conference.org/ competitions/

Johnston, M. D., Tran, D., Arroyo, B., Sorensen, S., Tay, P., Carruth, B., Coffman, A., and Wallace, M. Automated scheduling for nasa's deep space network. AI Magazine, 35(4):7-25, 2014.

Kahneman, D. Thinking, fast and slow. macmillan, 2011.

Kambhampati, S. Polanyi's revenge and ai's new romance with tacit knowledge. Communications of the ACM, 64(2):31-32, 2021.

Kambhampati, S. Can llms really reason and plan? Communications of the Association for Computing Machinery (CACM) Blog, 2023. URL https://cacm.acm.org/blogs/blog-cacm/ 276268-can-llms-really-reason-and-plan/fulltext

Kambhampati, S., Valmeekam, K., Marquez, M., and Guan, L. On the role of large language models in planning, July 2023. URL https://yochan-lab.github.io/tutorial/ICAPS-2023/. Tutorial presented at the International Conference on Automated Planning and Scheduling (ICAPS), Prague.

Kugel, S. and Hiltner, S. A new frontier for travel scammers: A.i.-generated guidebooks. New York Times, August 2023. URL https://www.nytimes.com/2023/08/05/travel/ amazon-guidebooks-artificial-intelligence.html.

Kwon, M., Xie, S. M., Bullard, K., and Sadigh, D. Reward design with language models. In The Eleventh International Conference on Learning Representations, 2022.

Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., and Zeng, A. Code as policies: Language model programs for embodied control, 2023.

Liu, B., Jiang, Y., Zhang, X., Liu, Q., Zhang, S., Biswas, J., and Stone, P. Llm+ p: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477, 2023.

Ma, Y. J., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., and Anandkumar, A. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

McCoy, R. T., Yao, S., Friedman, D., Hardy, M., and Griffiths, T. L. Embers of autoregression: Understanding large language models through the problem they are trained to solve. arXiv preprint arXiv:2309.13638, 2023.

McDermott, D., Ghallab, M., Howe, A. E., Knoblock, C. A., Ram, A., Veloso, M. M., Weld, D. S., and Wilkins, D. E. Pddl-the planning domain definition language. 1998.

Nieuwenhuis, R. and Oliveras, A. On sat modulo theories and optimization problems. In Theory and Applications of Satisfiability Testing-SAT 2006: 9th International Conference, Seattle, WA, USA, August 12-15, 2006. Proceedings 9, pp. 156-169. Springer, 2006.

Olmo, A., Sreedharan, S., and Kambhampati, S. Gpt3-to-plan: Extracting plans from text using gpt-3. FinPlan 2021, pp. 24, 2021.

OpenAI. Introducing chatgpt by openai, 2022. URL https://openai.com/blog/chatgpt

OpenAI. Gpt-4 technical report, 2023.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022.

Pallagani, V., Muppasani, B., Murugesan, K., Rossi, F., Srivastava, B., Horesh, L., Fabiano, F., and Loreggia, A. Understanding the capabilities of large language models for automated planning, 2023 .

Pan, L., Albalak, A., Wang, X., and Wang, W. Y. Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning. arXiv preprint arXiv:2305.12295, 2023.

Rajvanshi, A., Sikka, K., Lin, X., Lee, B., Chiu, H.-P., and Velasquez, A. Saynav: Grounding large language models for dynamic planning to navigation in new environments. arXiv preprint arXiv:2309.04077, 2023.

Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J., Ellenberg, J. S., Wang, P., Fawzi, O., et al. Mathematical discoveries from program search with large language models. Nature, pp. 1-3, 2023.

Russell, S. J. and Norvig, P. Artificial intelligence a modern approach. London, 2010.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K. R., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Shridhar, M., Yuan, X., Côté, M.-A., Bisk, Y., Trischler, A., and Hausknecht, M. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2010 03768

Silver, T., Hariprasad, V., Shuttleworth, R. S., Kumar, N., Lozano-Pérez, T., and Kaelbling, L. P. PDDL planning with pretrained large language models. In NeurIPS 2022 Foundation Models for Decision Making Workshop, 2022. URL https://openreview.net/forum?id=1QMMUB4zfl.

Stechly, K., Marquez, M., and Kambhampati, S. Gpt-4 doesn't know it's wrong: An analysis of iterative prompting for reasoning problems. In NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023.

Trinh, T. H., Wu, Y., Le, Q. V., He, H., and Luong, T. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476-482, 2024.

Ullman, T. Large language models fail on trivial alterations to theory-of-mind tasks. arXiv preprint arXiv:2302.08399, 2023.

Valmeekam, K., Marquez, M., and Kambhampati, S. Can large language models really improve by self-critiquing their own plans? In NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023a.

Valmeekam, K., Marquez, M., Olmo, A., Sreedharan, S., and Kambhampati, S. Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023b. URL https://openreview.net/forum?id=YXogl4uQUO.

Valmeekam, K., Marquez, M., Sreedharan, S., and Kambhampati, S. On the planning abilities of large language models - a critical investigation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023c. URL https://openreview.net/forum?id=X6dEqXIsEW

Verma, M., Bhambri, S., and Kambhampati, S. Theory of mind abilities of large language models in human-robot interaction: An illusion? arXiv preprint arXiv:2401.05302, 2024.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023 .

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022.

Weng, Y., Zhu, M., Xia, F., Li, B., He, S., Liu, S., Sun, B., Liu, K., and Zhao, J. Large language models are better reasoners with self-verification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 2550-2575, 2023.

Xie, Y., Yu, C., Zhu, T., Bai, J., Gong, Z., and Soh, H. Translating natural language to planning goals with large-language models. arXiv preprint arXiv:2302.05128, 2023.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. R. Tree of thoughts: Deliberate problem solving with large language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023a. URL https://openreview.net/forum? id=5Xc1ecx01h

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023b. URL https://openreview.net/forum?id=WE_vluYUL-X


[^0]:    ${ }^{*}$ Corresponding author. Email: rao@ asu.edu

    ${ }^{1}$ Think of asking an LLM an yes/no question-is this theorem logically entailed by this first-order logic knowledge-base. This is well-known to be a semi-decidable problem. Ask yourself if the LLM will take longer in answering the question. (If you are thinking Chain-of-thought prompts or training with step-by-step data, consider that you are essentially changing the nature of the original prompt/training).

[^1]:    ${ }^{2}$ In some circles, this unidirectional pipeline has come to be given the undeserved badge of neuro-symbolic architecture.

[^2]:    ${ }^{3}$ Contrary to the claim of "self-improvement", works like Wang et al. 2022) actually heavily depend on external knowledge (crafted seed examples) and critics (filtering step).

[^3]:    ${ }^{4}$ Although domains like AlfWorld (Shridhar et al. 2021) do have sub-goal interactions for successful task completion, (Yao et al. 2023b) and (Shinn et al. 2023) ignore these interactions relying on the ergodic nature of the domain when prompting LLMs for generating plans.

    ${ }^{5}$ These issues are illustrated in part by a recent news story (Kugel \& Hiltner 2023) about the proliferation of travel planning books, mostly auto-extracted from LLMs, and the ensuing disappointment of the unsuspecting end users who buy them mistaking them for usable plans!

[^4]:    ${ }^{6}$ Paradoxically, the fact that it is infeasible to write sound verifiers for tacit knowledge tasks also makes it possible for everyone to be a critic. Think of R2 saying the paper could be made "less dense" or the Peloton instructor critiquing Christopher Nolan film.

    ${ }^{7}$ Our preliminary experiments also show that at least in 24 puzzle, a simple iterative prompting, even without a systematic prompt diversification, is quite competitive with the ToT framework.

    ${ }^{8}$ Note that we can do this check easily because of the formal specification of correctness. For the "improving writing task" also used in ToT, there are no formal quality metrics and so it is hard to say anything concrete about the critiques of the LLM.

[^5]:    ${ }^{9}$ The name LLM-Modulo is inspired by the SAT-Modulo theories (Nieuwenhuis \& Oliveras 2006)

    ${ }^{10}$ Although we focus on planning from scratch, it is easy to accommodate replanning scenarios, where the loop starts with an externally supplied candidate plan.

[^6]:    ${ }^{11}$ Such summarization is a reasonable strategy as the back prompts will not be treated as hard constraints by LLMs anyway.

[^7]:    ${ }^{12}$ If you know how to complete sentences, and now learned to complete dance moves, does your ability to reason/plan magically improve?

</end of paper 0>


<paper 1>
# Co-driver: VLM-based Autonomous Driving Assistant with Human-like Behavior and Understanding for Complex Road Scenes 

Ziang Guo ${ }^{1 *}$, Artem Lykov ${ }^{1 *}$, Zakhar Yagudin ${ }^{1 *}$, Mikhail Konenkov ${ }^{1}$ and Dzmitry Tsetserukou ${ }^{1}$


#### Abstract

Recent research about Large Language Model based autonomous driving solutions shows a promising picture in planning and control fields. However, heavy computational resources and hallucinations of Large Language Models continue to hinder the tasks of predicting precise trajectories and instructing control signals. To address this problem, we propose Co-driver, a novel autonomous driving assistant system to empower autonomous vehicles with adjustable driving behaviors based on the understanding of road scenes. A pipeline involving the CARLA simulator and Robot Operating System 2 (ROS2) verifying the effectiveness of our system is presented, utilizing a single Nvidia 4090 24G GPU while exploiting the capacity of textual output of the Visual Language Model. Besides, we also contribute a dataset containing an image set and a corresponding prompt set for fine-tuning the Visual Language Model module of our system. In the real-world driving dataset, our system achieved $96.16 \%$ success rate in night scenes and $89.7 \%$ in gloomy scenes regarding reasonable predictions. Our Codriver dataset will be released at https://github.com/ZionGo6/Codriver.


## I. INTRODUCTION

## A. Motivation

In autonomous driving system development, two main solutions have been presented both in the academic and industrial fields till now. The first type is modular system design with independent modules such as perception, prediction, control, planning, etc. This design can empower distributed development and flexible extension, while the errors of such a system could accumulate because of asynchronization and delay of communication among modules [1]. The second type is end-to-end system design connecting sensor input and planning policy directly bypassing intermediate tasks and enabling the simple design of the network. But end-toend models demand their interpretability and logicality [2]. Generally, due to the complex traffic environment, long-tail data and dynamic scenes are still remaining limitations of these existing solutions [3].

Promisingly, Large Language Models (LLMs) have been actively developed in recent years, bridging human interaction and reasoning. Based on the advancements in LLMs, in the driving scene, LLMs can provide a more holistic understanding of the driving environment, allowing vehicles to respond more effectively to various driving scenarios with human-like logic which helps alleviate public concerns about the safety and reliability of autonomous vehicles [4],

* denotes the equal contribution. 1 The authors are with the Intelligent Space Robotics Laboratory, Center for Digital Engineering, Skolkovo Institute of Science and Technology, Moscow, Russia \{ziang.guo, artem.lykov, Zakhar.Yagudin, mikhail.konenkov, d.tsetserukou@skoltech.ru\}

![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-1.jpg?height=732&width=810&top_left_y=566&top_left_x=1102)

Fig. 1: System overview. Our Visual Language Model module receives the image input and system prompt, publishing the analysis of environment and instruction results in a behavior tree format. Then the behavior tree of instruction results is mapped to agent behaviors according to the analysis of the environment.

[5]. However, to contribute to the precision and robustness requirements of autonomous driving, LLMs need more longterm verification and real-world experiments [6].

In this work, we introduce Co-driver, an autonomous driving assistant to provide the instructions for driving behavior at an agent level based on the analysis of visual perception input.

## B. Related Work

1) End-to-End Autonomous Driving: Recently, end-toend autonomous driving is developed vigorously. Hao Shao et al. present ReasonNet [7], an end-to-end driving framework that utilizes both temporal and global information of the driving scene to effectively predict the future evolution of the scene and behaviors of objects, especially in dense traffic scenarios. Jia et al. [8] propose a cascaded decoder paradigm for predicting the future action of the ego vehicle in a coarse-tofine fashion, inspired by the behavior of human drivers who check their intended target for safety and legitimacy. Hu et al. [2] propose a planning-oriented framework that incorporates full-stack driving tasks in one network, prioritizing planning and facilitating task coordination. Jiaxun Cui et al. [9] intro-
duce an end-to-end learning model that utilizes cross-vehicle perception for vision-based cooperative driving, aiming to enhance autonomous driving in dangerous or emergencies. Among them, ReasonNet approaches perception via global reasoning, but these methods still do not utilize and integrate a human-driver understanding of complex traffic scenes to fulfill the decision-making tasks.
2) Large Language Models: However, a recent study released the potential combination of large language models and autonomous driving. Boyi Li et al. [10] present LLaDA, a tool that enables human drivers and autonomous vehicles to adapt their driving behavior to new locations by interpreting traffic rules using large language models. Sharan et al. [11] propose a hybrid planner, which combines a rule-based planner with an LLM-based planner. Hao Shao et al. [3] introduce a language-guided, end-to-end autonomous driving framework that integrates multimodal sensor data with natural language instructions, enabling interaction with humans and navigation software. Can Cui et al. [12] introduce a framework that uses large language models to process verbal commands from humans and make autonomous driving decisions, taking into account contextual information and personalized preferences for safety, efficiency, and comfort. Wang et al. [13] explore the integration of Large Language Models (LLMs) into autonomous driving systems to enhance their performance and safety by leveraging the LLMs' common-sense knowledge, reasoning abilities, and human interaction capabilities. The above work mainly shows the exploitation of language modal and its extensions. However, in the autonomous driving field, a combination of multimodal sensors, especially including visual modal is critical for necessary scene understanding.
3) Visual Language Models in Autonomous Driving Scenarios: In this section, we explore various approaches to integrating Visual Language Models (VLMs) into autonomous driving scenarios, highlighting their roles in environmental analysis and decision-making. DriveLM [14] focuses on the integration of VLMs into end-to-end driving systems via Graph Visual Question Answering (VQA). By utilizing this approach, DriveLM enables comprehensive reasoning about driving scenes, encompassing perception, prediction, and planning stages. The introduced DriveLM-Data dataset and baseline approach provide a framework for end-to-end autonomous driving, showcasing competitive performance even when faced with unseen objects or sensor configurations. RAG-Driver [15] addresses the crucial need for human-understandable explanations in autonomous driving systems. Employing retrieval-augmented multimodal large language models, RAG-Driver excels in producing detailed explanations for driving actions and accurate predictions of control signals. Its remarkable zero-shot generalization capabilities to previously unseen environments underscore its potential for real-world deployment. DriveVLM [16] introduces an autonomous driving system that effectively leverages VLMs for enhanced scene understanding and planning. Through the integration of chain-of-thought (CoT) modules, DriveVLM achieves a robust spatial understanding and real-time inference speed. Particularly noteworthy is DriveVLM-Dual, a hybrid system that combines VLMs with traditional autonomous driving pipelines, resulting in superior performance in navigating complex and unpredictable driving conditions.

All of the above research regarding visual language models needs heavy computation resources for both training and inference, which is a critical factor in the robustness and safety of autonomous driving. Besides, a hallucination of Large Language Models is still not explainable [17], resulting in challenges and risks in practical deployment. For Large Language Models, to output the coordinates of waypoints with high precision and stable response for autonomous driving in complex traffic and extreme conditions needs more real-world experiments and long-term verification.

## C. Contribution

Our main contributions to this work are as follows. An image dataset created in CARLA simulator [18] with defined weather, light, road surface, locality, and traffic conditions associated with a prompt dataset with control and behavior parameters based on the scenes defined in the image dataset. Besides, a pipeline of our Co-driver system is presented, where the CARLA simulator is used to run the simulation scenes, publishing the status information of the ego vehicle via Robot Operating System 2 (ROS2) [19] and our Visual Language Model module is wrapped within ROS2, reading the published ROS2 topics of front images of the camera on the ego vehicle. While analyzing the front images, our Visual Language Model module instructs and alters the driving behaviors of the ego vehicle in CARLA. Fig. 1 shows our pipeline in detail, where a single Nvidia 4090 24G GPU is able to handle our whole pipeline.

## II. SYSTEM OVERVIEW

Our system is driven by Qwen-VL by Bai et al. [20]. Qwen-VL is a leading open-source model in the field of Visual Language Models (VLM), showcasing exceptional capabilities in tasks such as image captioning, question answering, visual localization, and interactive tasks. This model processes both textual and visual data and excels in recognizing individual objects and their positions, as well as grounding tasks, which are crucial for our study.

Qwen-VL's high performance is attributed to its positionaware vision-language adapter and its efficient 3-stage training process. With a total of 9.6 billion parameters, the model comprises a visual encoder (1.9 billion parameters), a visionlanguage adapter ( 0.08 billion parameters), and the Qwen large language model ( 7.7 billion parameters).

The advanced environmental analysis capabilities of Qwen-VL, combined with the reasoning power of Qwen [21], make it particularly suitable for our task. The model's compact size allows for seamless integration into a selfdriving car's onboard computer, enabling efficient local deployment without sacrificing performance. This positions Qwen-VL as an ideal choice for enhancing autonomous driving systems.

System architecture is depicted in Fig. 1. The primary task of our system is to analyze the visual input from the front camera of the ego vehicle and draw conclusions about environmental information such as weather, light, road surface, locality, etc., and parameters of control such as maximum speed, maximum brake, maximum throttle, etc. Determining the driving behaviors of a self-driving car based on visual data is a complex task for Visual Language Models. However, by breaking down the task into a two-step process, it becomes manageable.

The task is decomposed to identify environmental information from an image by feeding specifically defined scenes from our image dataset to the model and to predict the levels of control and behavior parameters based on the described environmental data. Both of these tasks pose no significant challenges for a fine-tuned Visual Language Model, which ensures the practical pipeline of implementation in our proposed system.

In the first step of the mentioned task, our Visual Language Model module receives system prompts containing the mission description and destination, along with the images from the ego vehicle's front camera. In this stage, the module identifies locality, lighting, and weather conditions, as well as potential hazards in front. Then our module continues to generate the levels of control and driving behavior parameters, guided by the environmental parameters identified in the first step. Lastly, all the obtained parameters are mapped as a set of agent behaviors altering and influencing the driving style of the ego vehicle in the CARLA simulator based on the image input of our Visual Language Model module.

## III. MethodOlogY

## A. Dataset Collection

Our image dataset is collected in the CARLA simulator from the view of the front camera of ego vehicle under defined weather (clear, rainy, foggy), light (bright, gloomy, dark), locality (city, town, highway) conditions with a classification of safe and unsafe distance concerning the potential obstacle in front [18].

In our prompt dataset, system prompts are given as the request of accomplishment of the driving missions and notice of the environmental information from the perspective of a driver's view. Then we include the defined environmental information and the suggestions for vehicle control and driving behavior regarding control type, maximum speed, maximum brake, maximum throttle, maximum acceleration, and maximum steering speed as the output prompt in a behavior tree format. Here is an example of our dataset in Fig. 2 .

## B. Training Recipe

The Visual Language Model (VLM) of our system was trained on the foundation of the Qwen-VL architecture utilizing the Quantized Low-Rank Adaptation (QLoRA) method [22], which is a form of Parameter Efficient Fine-tuning (PEFT) [23]. During the training process, the weights of the visual encoder were kept frozen to focus on optimizing the language aspect of the model.

Training was carried out on a single Nvidia RTX 4090 GPU, which provided $24 \mathrm{~GB}$ of video memory for processing. The dataset, containing a total of 221,228 samples, was divided into batches of 6 samples each to maintain efficient training throughput. Additionally, gradient accumulation steps were set to 8 , resulting in an epoch comprising approximately 4,600 steps.

With a learning rate of $1 \mathrm{e}-4$, the model quickly adapted to the target emergent capabilities and responded to the desired format. This process only required one epoch of training, which took around 25 hours to complete. Despite the relatively short training time, the approach proved effective, yielding satisfactory results in terms of model performance and output quality.

The progression of the training process is depicted in the training curve presented in Fig. 3 , showcasing the changes in loss over time and offering insights into the model's learning dynamics.

## IV. EXPERIMENTAL RESULTS

## A. Experiment Setup

To verify our system's effectiveness, we conducted two types of experiments. First, in CARLA, test scenes were created with adjustable weather, maps, and traffic settings. During the running of the test simulation, our Visual Language Model module was on, reading the front images from the ego vehicle and performing the scene understanding and behavior instructions. We recorded the driving scenes with vehicle trajectories and vehicle status information such as speed, acceleration, etc. Second, we verified the generalization ability of our system's Visual Language Model module on HawkDrive dataset [24] with real-world driving scenes in gloomy night conditions.

## B. CARLA Simulation

In the CARLA simulator, we compared the driving behaviors between the default built-in agent and our Co-driver

![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-3.jpg?height=608&width=832&top_left_y=1840&top_left_x=1096)

Fig. 2: Example of our dataset with image set and prompt set.

![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-4.jpg?height=605&width=762&top_left_y=169&top_left_x=237)

Fig. 3: The training loss decreases while our fine-tuning progress.

agent in Town 10, Town 04, and Town 02. CARLA map Town 04 in Fig. 5 is selected as one of our experimental maps to demonstrate our results. With the default agent, the ego vehicle was not able to switch driving behaviors along with the weather, light, and traffic conditions. In Fig. 6, under both rainy and foggy weather with gloomy light conditions, the driving behaviors of the ego vehicle along the same trajectory remained nearly identical since the planning and control modules were driven by defined rules only. When passing by the $90 \mathrm{~km} / \mathrm{h}$ speed limit sign, the default agent guided the ego vehicle to reach $90 \mathrm{~km} / \mathrm{h}$, ignoring the environmental information. In Fig. 7, with our Co-driver system running, under both rainy and foggy weather with gloomy light conditions, the driving behaviors of the ego vehicle were instructed according to the front image input. Based on the output of our Visual Language Model module, even if the ego vehicle passed by speed limit signs, our Co-driver system guided the ego vehicle to drive under the instructions.

Fig. 4 shows the acceleration recording of our experiments. To interpret the results, we identified the relative maxima and minima throughout the acceleration recording. Then the frequency of fluctuations is calculated by counting the number of peaks and valleys in the data. Finally, we use the ratio of frequency of fluctuations and running time to denote the smoothness of driving behaviors as follows,

$$
\begin{equation*}
\dot{\mathcal{F}}_{T}=\frac{\text { Concatenate }(\text { relmin }(\mathcal{X}), \operatorname{relmax}(\mathcal{X})) \times \frac{1}{2}}{T} \tag{1}
\end{equation*}
$$

where the arrays of indices of relative minima and relative maxima are concatenated as a combined array that contains the indices where the values in the input data $\mathcal{X}$ reach relative minima and maxima. Smaller $\dot{\mathcal{F}}_{T}$ means smoother driving with less intensive fluctuation of acceleration. $T$ is the running time of our experiments. The comparison of acceleration recording is presented in Table I.
![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-4.jpg?height=640&width=832&top_left_y=173&top_left_x=1080)

(a) Acceleration recording of (b) Acceleration recording of ego ego vehicle with default agent vehicle with Co-driver agent ununder foggy and gloomy con- der foggy and gloomy conditions ditions (above) and rainy and (above) and rainy and gloomy gloomy conditions (below). conditions (below).

Fig. 4: Ego vehicle test with Co-driver agent in CARLA map Town 04 .

TABLE I: Comparison of the smoothness of driving behaviors between default agent and Co-driver agent in Town 04.

|  | Foggy + Gloomy | Rainy + Gloomy |
| :---: | :---: | :---: |
| $\mathcal{F}_{T}$ of Default Agent | 0.117 | 0.153 |
| $\mathcal{F}_{T}$ of Co-driver Agent | $\mathbf{0 . 0 2 1}$ | $\mathbf{0 . 1 0 4}$ |

## C. Real-World Driving Dataset

To present the generalization ability of our system, HawkDrive dataset [24] which provides continuous driving scenes with different light conditions in a closed loop, is used to test the Visual Language Model module of our Co-driver system. In the night scene of the dataset, we labeled the night images corresponding to the ground truth regarding safety distance, weather, light condition, road surface and locality. Among 1,067 images, 41 images gave critical failure of instructions and understanding, showing a $96.16 \%$ precision of successful prediction. Meanwhile, the safety distance was detected correspondingly regarding potential obstacles in the frames in night scenes. The results are displayed in Fig. 8 In the gloomy scene of the dataset, throughout 952 images, 98 images gave critical failure of prediction, showing a $89.7 \%$ precision of successful prediction. The results are presented in Fig. 9

## V. CONCLUSION

In this work, we propose Co-driver, an autonomous driving assistant system to empower autonomous driving vehicles with adjustable driving behaviors based on the understanding of complex road scenes including safety distance, weather, light conditions, road surface and locality. To be practical, we present our system in a pipeline involving the CARLA simulator and Robot Operating System 2 (ROS2), while verifying the effectiveness of our system by comparing the

![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-5.jpg?height=536&width=542&top_left_y=171&top_left_x=344)

Fig. 5: CARLA map Town 04 is a small town embedded in the mountains with a special "figure of 8 " infinite highway.
![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-5.jpg?height=614&width=808&top_left_y=866&top_left_x=191)

(a) The front images were (b) The speed recording of the recorded during the experi- ego vehicle running with dement in foggy and gloomy fault built-in agent in foggy and conditions (above) and rainy gloomy conditions (above) and and gloomy conditions (be- rainy and gloomy conditions (below).

low).

Fig. 6: Ego vehicle test with default built-in agent in CARLA map Town 04.

driving behaviors of the rule-based default agent with our Co-driver agent. In the real-world driving dataset, our system achieved a $96.16 \%$ success rate in night scenes and $89.7 \%$ in gloomy scenes of reasonable prediction. Besides, we also contributed a Co-driver dataset containing 221,228 image samples and a corresponding prompt set to spark further related research.

From our results, the promising capacity of our Co-driver system is displayed. With the vigorous development of Large Multimodal Models, our work is able to enlighten further advancement in the autonomous driving field.

## ACKNOWLEDGMENT

This project is supported by Skolkovo Institute of Science and Technology, Moscow, Russia.
![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-5.jpg?height=608&width=792&top_left_y=190&top_left_x=1096)

(a) Co-driver agent running (b) The speed of ego vehicle under foggy and gloomy con- was adjusted according to the ditions (above) and rainy and image input even if passing by gloomy conditions (below). the speed limit signs.

Fig. 7: Ego vehicle test with Co-driver agent in CARLA map Town 04
![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-5.jpg?height=272&width=788&top_left_y=1091&top_left_x=1117)

(a) Failure case of the test on (b) Successful case of the test HawkDrive dataset in a night on HawkDrive dataset in a scene. night scene.

Fig. 8: Visual Language Model module of our Co-driver system test on HawkDrive dataset in a night scene.
![](https://cdn.mathpix.com/cropped/2024_06_04_a3403ab05a5c54cff7c9g-5.jpg?height=274&width=802&top_left_y=1614&top_left_x=1103)

(a) Failure case of the test (b) Successful case of the test on HawkDrive dataset in a on HawkDrive dataset in a gloomy scene. gloomy scene.

Fig. 9: Visual Language Model module of our Co-driver system test on HawkDrive dataset in a gloomy scene.

## REFERENCES

[1] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li, "Planning-oriented autonomous driving," 2023.

[2] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang et al., "Goal-oriented autonomous driving," arXiv preprint arXiv:2212.10156, 2022.

[3] H. Shao, Y. Hu, L. Wang, S. L. Waslander, Y. Liu, and H. Li, "Lmdrive: Closed-loop end-to-end driving with large language models," arXiv preprint arXiv:2312.07488, 2023.

[4] W. Han, D. Guo, C.-Z. Xu, and J. Shen, "Dme-driver: Integrating human decision logic and 3d scene perception in autonomous driving," arXiv preprint arXiv:2401.03641, 2024.

[5] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin et al., "A survey on large language model based autonomous agents," Frontiers of Computer Science, vol. 18, no. 6 , pp. 1-26, 2024 .

[6] S. Kambhampati, K. Valmeekam, L. Guan, K. Stechly, M. Verma, S. Bhambri, L. Saldyt, and A. Murthy, "Llms can't plan, but can help planning in llm-modulo frameworks," arXiv preprint arXiv:2402.01817, 2024.

[7] H. Shao, L. Wang, R. Chen, S. L. Waslander, H. Li, and Y. Liu, "Reasonnet: End-to-end driving with temporal and global reasoning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13723-13 733.

[8] X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li, "Think twice before driving: Towards scalable decoders for end-to-end autonomous driving," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21 983-21 994.

[9] J. Cui, H. Qiu, D. Chen, P. Stone, and Y. Zhu, "Coopernaut: End-toend driving with cooperative perception for networked vehicles," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 17252-17 262.

[10] B. Li, Y. Wang, J. Mao, B. Ivanovic, S. Veer, K. Leung, and M. Pavone, "Driving everywhere with large language model policy adaptation," arXiv preprint arXiv:2402.05932, 2024.

[11] S. Sharan, F. Pittaluga, M. Chandraker et al., "Llm-assist: Enhancing closed-loop planning with language-based reasoning," arXiv preprint arXiv:2401.00125, 2023.

[12] C. Cui, Z. Yang, Y. Zhou, Y. Ma, J. Lu, and Z. Wang, "Large language models for autonomous driving: Real-world experiments," arXiv preprint arXiv:2312.09397, 2023.

[13] Y. Wang, R. Jiao, C. Lang, S. S. Zhan, C. Huang, Z. Wang, Z. Yang, and Q. Zhu, "Empowering autonomous driving with large language models: A safety perspective," arXiv preprint arXiv:2312.00812, 2023.

[14] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, P. Luo, A. Geiger, and H. Li, "Drivelm: Driving with graph visual question answering," arXiv preprint arXiv:2312.14150, 2023.

[15] J. Yuan, S. Sun, D. Omeiza, B. Zhao, P. Newman, L. Kunze, and M. Gadd, "Rag-driver: Generalisable driving explanations with retrieval-augmented in-context learning in multi-modal large language model," arXiv preprint arXiv:2402.10828, 2024.

[16] X. Tian, J. Gu, B. Li, Y. Liu, C. Hu, Y. Wang, K. Zhan, P. Jia, X. Lang, and H. Zhao, "Drivevlm: The convergence of autonomous driving and large vision-language models," arXiv preprint arXiv:2402.12289, 2024.

[17] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin et al., "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions," arXiv preprint arXiv:2311.05232, 2023.

[18] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, "CARLA: An open urban driving simulator," in Proceedings of the 1st Annual Conference on Robot Learning, 2017, pp. 1-16.

[19] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, "Robot operating system 2: Design, architecture, and uses in the wild," Science Robotics, vol. 7, no. 66, p. eabm6074, 2022. [Online]. Available: https://www.science.org/doi/abs/10.1126/ scirobotics.abm6074

[20] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, "Qwen-vl: A frontier large vision-language model with versatile abilities," arXiv preprint arXiv:2308.12966, 2023.

[21] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., "Qwen technical report," arXiv preprint arXiv:2309.16609, 2023.

[22] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "Qlora: Efficient finetuning of quantized llms," Advances in Neural Information Processing Systems, vol. 36, 2024

[23] Z. Fu, H. Yang, A. M.-C. So, W. Lam, L. Bing, and N. Collier, "On the effectiveness of parameter-efficient fine-tuning," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 11, 2023, pp. 12799-12807.

[24] Z. Guo, S. Perminov, M. Konenkov, and D. Tsetserukou, "Hawkdrive: A transformer-driven visual perception system for autonomous driving in night scene," arXiv preprint arXiv:2404.04653, 2024.

</end of paper 1>


<paper 2>
# Can Large Language Models Really Improve by Self-critiquing Their Own Plans? 

Karthik Valmeekam ${ }^{*}$<br>School of Computing \& AI<br>Arizona State University Tempe.<br>kvalmeek@asu.edu

Matthew Marquez*<br>School of Computing \& AI<br>Arizona State University, Tempe.<br>mmarqu22@asu.edu

Subbarao Kambhampati<br>School of Computing \& AI<br>Arizona State University, Tempe.<br>rao@asu.edu


#### Abstract

There have been widespread claims about Large Language Models (LLMs) being able to successfully verify or self-critique their candidate solutions in reasoning problems in an iterative mode. Intrigued by those claims, in this paper we set out to investigate the verification/self-critiquing abilities of large language models in the context of planning. We evaluate a planning system that employs LLMs for both plan generation and verification. We assess the verifier LLM's performance against ground-truth verification, the impact of self-critiquing on plan generation, and the influence of varying feedback levels on system performance. Using GPT-4, a state-of-the-art LLM, for both generation and verification, our findings reveal that self-critiquing appears to diminish plan generation performance, especially when compared to systems with external, sound verifiers and the LLM verifiers in that system produce a notable number of false positives, compromising the system's reliability. Additionally, the nature of feedback, whether binary or detailed, showed minimal impact on plan generation. Collectively, our results cast doubt on the effectiveness of LLMs in a self-critiquing, iterative framework for planning tasks.


## 1 Introduction

Large Language Models have rapidly captured the attention of the AI research community with their exceptional natural language completion capabilities. Trained on web-scale language corpora, these models have demonstrated the ability to generate seemingly valuable completions across a wide range of topics. This led to a surge of interest in determining whether such models were able to perform well on reasoning tasks. Even though initial anecdotal results showed promise, systematic studies revealed their incompetency in reasoning - be it planning [12] or in simple arithmetic or logic [3]. These results questioning the robustness of their reasoning abilities led to researchers exploring ways to improve these systems. Of particular interest to us is the emerging research on self-critiquing, where the LLMs are used to critique their own candidate generations and iterate. The current works [15, 10, 14] exhibit considerable optimism about using LLMs to critique their own candidate generations, especially in an iterative setting where they keep refining their candidate generations. Additionally, the notion that verifying correctness is computationally simpler than generation for reasoning adds to the optimism. However, there are grounds to be skeptical about it as[^0]the complexity of a reasoning task in the classical sense should be irrelevant to models like LLMs that do approximate retrieval.

Intrigued by the prevailing optimism, in this paper, we set out to systematically investigate the effectiveness of using LLMs to critique their own generations in the context of planning. We look at the simplest class of planning problems, the goal-directed deterministic planning problems colloquially referred to as classical planning problems. Our methodology employs a planning system that utilizes the same LLM for both generation and verification, which we term the LLM+LLM system in an iterative setting. Within this setting, the generator LLM continuously produces candidate plans, drawing upon feedback from the verifier LLM, until the verifier LLM either approves a candidate plan as correct or the number of iterations surpasses a predefined threshold. We present an empirical evaluation of (i) the effect of self-critiquing on the plan generation performance of the overall LLM+LLM system (ii) the performance of the verifier LLM in comparison to the ground-truth verification and finally (iii) the influence of varying feedback levels while critiquing the LLM's generation on the overall system performance. For our study, we use GPT-4 [9] as both the generator and verifier.

Our findings suggest that self-critiquing degrades the plan generation performance compared to when an external, sound verifier is utilized. This decline in performance can be directly attributed to the verifier LLM's subpar results. The verifier LLM yields a significant number of false positives, which can severely undermine the system's reliability. Furthermore, we explored whether the nature of feedback on invalid plans influences plan generation performance. Our results indicate that the type of feedback-whether it's merely binary verification or combined with detailed feedback on the errors of the generated plan-doesn't significantly impact plan generation performance.

Thus, our systematic investigation offers compelling preliminary evidence to question the efficacy of LLMs as verifiers for planning tasks within an iterative, self-critiquing framework. In the rest of the paper, we first present the related work, then the required background before delving into the methodology and the evaluation.

## 2 Related Work

There has been significant interest in investigating the reasoning capabilities of LLMs, spanning from planning [12] to logic and arithmetic [3], and even puzzles [15]. As the initial excitement from triumphant anecdotes about LLMs' reasoning capabilities began to wane with systematic studies [12, 11, 3], researchers proposed that allowing LLMs to verify their own candidate solutions and iterate over this process could enhance their reasoning abilities [10, 7, 6, 14]. Our work systematically investigates the effect of iterative self-critiquing in the context of planning.

There have also been studies that utilize multiple LLMs to generate and verify candidate solutions, either in the form of a debate [2] or through cross-examination [1]. However, these studies still rely solely on the verification/self-critiquing abilities of the LLMs, an aspect our work critically examines in the context of planning. Our results provide compelling reasons to question the use of LLMs for self-critiquing in planning.

## 3 Background

We specifically are interested in classical planning problems that are represented within the PDDL (Planning Domain and Definition Language) framework [8]. These problem classes consist of a domain, initial state and a goal state. The domain consists of a set of predicates and a set of actions. The state-space of the planning problem is represented with some truth-assignment on the predicates. Every action in domain have a set of preconditions which determine when an action can be applied and a set of effects which determine the modifications to the state after the action is applied. A plan here is a sequence of actions which are present in the domain that when executed in the initial state, satisfy the goal conditions.

![](https://cdn.mathpix.com/cropped/2024_06_04_05e1f31d9191dc8af540g-3.jpg?height=564&width=1209&top_left_y=255&top_left_x=469)

Figure 1: Overall evaluation architecture

## 4 Methodology

### 4.1 The LLM+LLM planning system

The LLM+LLM planning system (as shown in Figure 1) consists of a generator LLM and a verifier LLM. For a given instance, the generator LLM produces a candidate plan, while the verifier LLM determines its correctness. If the plan is found to be incorrect, the verifier provides feedback detailing the reasons for its failure. This feedback is then relayed to the generator LLM, prompting the generation of a new candidate plan. It's worth noting that there are no constraints on the type or format of feedback the verifier LLM produces. The system ceases generation either when the verifier LLM approves the candidate plan as valid or when the number of prompting iterations exceeds a set threshold (for our experiments, this threshold is set at 15 iterations). This method is similar to the backprompting technique described in [12]. However, the main distinction lies in the type of verifier employed. In our system, both the verifier and generator are LLMs, whereas the referenced approach utilizes an external sound verifier, VAL [4]. For all our experiments, GPT-4 serves as the default LLM.

### 4.2 Prompt generation

For the LLM+LLM Planning system described above, we utilize distinct prompts for the generator and verifier LLMs. The prompt generator (as shown in Figure 1) utilizes the PDDL domain and instance files to generate the required prompts in natural language. Our prompts are structured similarly to the natural language prompts found in [12]. For plan generation, our prompts are one-shot: we begin by presenting the domain description, followed by an example instance (along with its corresponding plan). We then present the query instance. These example instances are randomly selected from our set of instances, and this forms the input for the generator LLM. For the verifier LLM, we adopt a zero-shot approach. Here, we present the domain description, followed by the query instance and its corresponding plan. The verifier LLM is then tasked with verifying the query plan and providing feedback if necessary. As mentioned earlier, we do not restrict the type or format of the feedback for the verifier LLM. Detailed examples of the prompts given to both the generator and verifier LLMs can be found in the Appendix.

## 5 Evaluation and Analysis

We evaluate our planning system on Blocksworld, a widely recognized common-sense planning domain in AI planning literature [5]. We generate 100 random instances for evaluation across various methods. To provide a ground-truth assessment of the final LLM plan's correctness, we employ an external sound verifier, VAL [4]. For all experiments, GPT-4 [9] serves as the chosen LLM and was run with a temperature of 0 , thereby making it deterministic.

### 5.1 Effect of self-critiquing on plan generation

We assessed the impact of self-critiquing on plan generation by comparing the LLM+LLM backprompting system with two other baselines. The first baseline is the LLM+VAL backprompting system, which mirrors the backprompting method described in [12]. In this method, the plan produced by the LLM is validated by an external sound verifier, VAL. If the plan is found lacking, the generator-LLM is reprompted using feedback from VAL. The second baseline involves a generatorLLM without backprompting. Here, the generator LLM receives a single prompt, and the resulting plan is considered final.

As illustrated in Table 1, the LLM+LLM backprompting approach slightly outperforms the nonbackprompting method in terms of accuracy. However, it falls short when compared to the LLM+VAL system. It's worth noting that the marginal improvement over the generator-LLM-only method might not solely be attributed to the LLM verifier. The backprompting itself, which offers the generator LLM multiple opportunities to produce a plan, could be a contributing factor. The subpar performance of the LLM+LLM system, especially when compared to LLM+VAL, can likely be traced back to the substantial number of type-1 errors produced by the LLM verifier. It's evident that incorporating a sound verifier in the backprompting process can significantly enhance overall performance.

| Plan Generation Method | Accuracy | Avg. Number of iterations |
| :--- | :---: | :---: |
| LLM+LLM w/ Backprompting (BP) | $55 / 100(55 \%)$ | 3.48 |
| LLM+VAL w/ BP | $88 / 100(88 \%)$ | 4.18 |
| Generator LLM only w/o BP | $40 / 100(40 \%)$ | 1.00 |

Table 1: Comparison between various plan generation methods on the Blocksworld domain.

### 5.2 Analysis on the self-critique verifier

We base our evaluation of the verifier LLM on its binary verification (i.e., determining whether the plan is valid or not) of the final plan produced by the LLM+LLM system. It's important to note that the system halts either when the verifier LLM considers the plan valid or when the number of iterations surpasses 15 . We compare the LLM verifier's output with ground truth classifications made using VAL [4], a sound verifier. To make the ground truth determination available for each input plan, we separately evaluate that plan using VAL as well.

As illustrated in Table 2, out of the 100 instances, the verifier accurately identifies 61 (or $61 \%$ ). However, a deeper examination of the verifier's errors reveals a concerning number of false positives. In this context, a false positive refers to the verifier LLM deeming a generated plan valid when, in fact, it is not. Out of the 100 instances, the verifier LLM produces 54 true positives and 38 false positives (type-1 errors). This indicates that the verifier deemed 38 plans, which were actually invalid, to be valid which can be catastrophic if such a system is deployed in scenarios where correctness is paramount.

|  | Accuracy | True Positive <br> Rate | False Positive <br> Rate | True Negative <br> Rate | False Negative <br> Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Verifier <br> LLM | $61 / 100(61 \%)$ | $54 / 55(98.2 \%)$ | $\mathbf{3 8 / 4 5}(\mathbf{8 4 . 4 5 \% )}$ | $7 / 45(15.55 \%)$ | $1 / 55(1.8 \%)$ |

Table 2: Breakdown of Plan Verification results on Blocksworld domain. The denominators (in aspects other than Accuracy) are ground-truth values based on VAL.

### 5.3 Effect of the levels of feedback on plan generation

While the use of a sound verifier appears to enhance overall performance, we sought to further investigate the impact of varied levels of feedback on plan generation performance. We assessed the system's performance across four distinct feedback levels:

1. No Feedback: At this level, the initial plan generated by the LLM is considered to be final and no feedback is provided to the LLM.
2. Binary Feedback: This level simply indicates whether the generated plan is valid or not.
3. Inexecutable Action Feedback: If the plan is invalid and inexecutable, this feedback highlights the first inexecutable action and the unmet preconditions causing the inexecutability. If the plan is executable but fails to meet all goal conditions, the unmet goal conditions are presented. This feedback mirrors what VAL provides.
4. Open Conditions Feedback: This level treats the plan as a partial-order plan [13] and presents all the actions for which there exists atleast one unmet pre-condition and the corresponding unmet preconditions. Further it also presents the unmet goal conditions.

Table 3 showcases the LLM's performance when subjected to various levels of feedback (including one with no feedback). Interestingly, the amount of feedback provided to the LLM seems to have minimal influence on its performance improvement. As long as the binary feedback is accurate and the LLM is given ample opportunities to generate a plan, the detailed feedback on invalid plans doesn't appear to significantly enhance the LLM's performance. We have provided examples for each feedback level in the Appendix.

| Levels of feedback | Accuracy | Avg. no of <br> steps |
| :--- | :---: | :---: |
| No feedback | $40 / 100(40 \%)$ | 1.00 |
| Only binary feedback | $37 / 50(74 \%)$ | 5.38 |
| Binary + First error feedback (by VAL) | $43 / 50(86 \%)$ | 4.18 |
| Binary + All errors feedback | $43 / 50(86 \%)$ | 4.42 |

Table 3: Performance of LLM+VAL system on plan generation with varied levels of feedback.

## 6 Conclusion and Future Work

In this paper, we conducted a systematic investigation into the ability of Large Language Models (LLMs) to critique their own outputs, specifically within the context of classical planning problems. While recent research has been optimistic about LLMs' potential in self-critiquing, especially in iterative settings, our findings present a different perspective.

Our empirical evaluations on Blocksworld, a simple common-sense domain, highlighted the ineffectiveness of self-critiquing in LLMs in the context of planning. We showed that the verifier LLM generates a significant number of false positives which be detrimental to the overall system's reliability, particularly in domains where the correctness of plans is paramount. Interestingly, the nature of feedback, whether binary or detailed, did not have a pronounced impact on plan generation performance, suggesting that the core issue lies in the LLM's binary verification capabilities rather than the granularity of feedback.

In the future, we plan to conduct more extensive experiments with respect to the number of instances, the number of domains and prompting methods (such as chain-of-thought).

## References

[1] Roi Cohen, May Hamri, Mor Geva, and Amir Globerson. Lm vs lm: Detecting factual errors via cross examination. arXiv preprint arXiv:2305.13281, 2023.

[2] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.

[3] Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jian, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D Hwang, et al. Faith and fate: Limits of transformers on compositionality. arXiv preprint arXiv:2305.18654, 2023.

[4] Richard Howey, Derek Long, and Maria Fox. VAL: Automatic plan validation, continuous effects and mixed initiative planning using PDDL. In 16th IEEE International Conference on Tools with Artificial Intelligence, pages 294-301. IEEE, 2004.

[5] IPC. International planning competition, 1998.

[6] Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. arXiv preprint arXiv:2303.17491, 2023.

[7] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.

[8] Drew McDermott, Malik Ghallab, Adele E. Howe, Craig A. Knoblock, Ashwin Ram, Manuela M. Veloso, Daniel S. Weld, and David E. Wilkins. Pddl-the planning domain definition language. 1998 .

[9] OpenAI. Gpt-4 technical report, 2023.

[10] Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint arXiv:2303.11366, 2023.

[11] Tom Silver, Varun Hariprasad, Reece S Shuttleworth, Nishanth Kumar, Tomás Lozano-Pérez, and Leslie Pack Kaelbling. PDDL planning with pretrained large language models. In NeurIPS 2022 Foundation Models for Decision Making Workshop, 2022.

[12] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models-a critical investigation. arXiv preprint arXiv:2305.15771, 2023.

[13] Daniel S Weld. An introduction to least commitment planning. AI magazine, 15(4):27-27, 1994.

[14] Yixuan Weng, Minjun Zhu, Shizhu He, Kang Liu, and Jun Zhao. Large language models are reasoners with self-verification. arXiv preprint arXiv:2212.09561, 2022.

[15] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.


[^0]:    * Equal Contribution

</end of paper 2>


