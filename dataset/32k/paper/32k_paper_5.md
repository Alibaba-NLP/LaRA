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
# LARGE LANGUAGE MoDELS CANNOT SELF-CorRECT REASONING YET 

Jie Huang ${ }^{1,2 *}$ Xinyun Chen ${ }^{1 *}$ Swaroop Mishra ${ }^{1}$ Huaixiu Steven Zheng ${ }^{1}$ Adams Wei Yu ${ }^{1}$<br>Xinying Song ${ }^{1}$ Denny Zhou ${ }^{1}$<br>${ }^{1}$ Google DeepMind ${ }^{2}$ University of Illinois at Urbana-Champaign<br>jeffhj@illinois.edu, \{xinyunchen, dennyzhou\}@google.com


#### Abstract

Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, self-correction, has been proposed as a remedy to these issues. Building upon this premise, this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to selfcorrect their responses without external feedback, and at times, their performance even degrades after self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.


## 1 INTRODUCTION

The rapid advancements in the domain of artificial intelligence have ushered in the era of Large Language Models (LLMs). These models, characterized by their expansive parameter counts and unparalleled capabilities in text generation, have showcased promising results across a multitude of applications (Chowdhery et al. 2023, Anil et al., 2023, OpenAI, 2023, inter alia). However, concerns about their accuracy, reasoning capabilities, and the safety of their generated content have drawn significant attention from the community (Bang et al., 2023, Alkaissi \& McFarlane, 2023, Zheng et al., 2023; Shi et al., 2023, Carlini et al. |2021; Huang et al., 2022; Shao et al., 2023; Li et al., 2023; Wei et al., 2023; Zhou et al., 2023b; Zou et al., 2023|, inter alia).

Amidst this backdrop, the concept of "self-correction" has emerged as a promising solution, where LLMs refine their responses based on feedback to their previous outputs (Madaan et al. 2023, Welleck et al., 2023, Shinn et al., 2023, Kim et al., 2023, Bai et al., 2022, Ganguli et al., 2023, Gao et al.| 2023; Paul et al., 2023, Chen et al., 2023b; Pan et al., 2023| inter alia). However, the underlying mechanics and efficacy of self-correction in LLMs remain underexplored. A fundamental question arises: If an LLM possesses the ability to self-correct, why doesn't it simply offer the correct answer in its initial attempt? This paper delves deeply into this paradox, critically examining the self-correction capabilities of LLMs, with a particular emphasis on reasoning (Wei et al., 2022, Zhou et al., 2023b, Huang \& Chang, 2023).

To study this, we first define the concept of intrinsic self-correction, a scenario wherein the model endeavors to rectify its initial responses based solely on its inherent capabilities, without the crutch of external feedback. Such a setting is crucial because high-quality external feedback is often unavailable in many real-world applications. Moreover, it is vital to understand the intrinsic capabilities of LLMs. Contrary to the optimism surrounding self-correction (Madaan et al. 2023, Kim et al., 2023; Shinn et al. 2023, Pan et al., 2023, inter alia), our findings indicate that LLMs struggle to self-correct their reasoning in this setting. In most instances, the performance after self-correction[^0]even deteriorates. This observation is in contrast to prior research such as Kim et al. (2023); Shinn et al. (2023). Upon closer examination, we observe that the improvements in these studies result from using oracle labels to guide the self-correction process, and the improvements vanish when oracle labels are not available.

Besides the reliance on oracle labels, we also identify other issues in the literature regarding measuring the improvement achieved by self-correction. First, we note that self-correction, by design, utilizes multiple LLM responses, thus making it crucial to compare it to baselines with equivalent inference costs. From this perspective, we investigate multi-agent debate (Du et al., 2023, Liang et al. 2023) as a means to improve reasoning, where multiple LLM instances (can be multiple copies of the same LLM) critique each other's responses. However, our results reveal that its efficacy is no better than self-consistency (Wang et al. 2022) when considering an equivalent number of responses, highlighting the limitations of such an approach.

Another important consideration for self-correction involves prompt design. Specifically, each selfcorrection process involves designing prompts for both the initial response generation and the selfcorrection steps. Our evaluation reveals that the self-correction improvement claimed by some existing work stems from the sub-optimal prompt for generating initial responses, where self-correction corrects these responses with more informative instructions about the initial task in the feedback prompt. In such cases, simply integrating the feedback into the initial instruction can yield better results, and self-correction again decreases performance.

In light of our findings, we provide insights into the nuances of LLMs' self-correction capabilities and initiate discussions to encourage future research focused on exploring methods that can genuinely correct reasoning.

## 2 BACKGROUND AND RELATED WORK

With the LLM evolution, the notion of self-correction gained prominence. The discourse on selfcorrection pivots around whether these advanced models can recognize the correctness of their outputs and provide refined answers (Bai et al., 2022, Madaan et al., 2023, Welleck et al., 2023, inter alia). For example, in the context of mathematical reasoning, an LLM might initially solve a complex problem but make an error in one of the calculation steps. In an ideal self-correction scenario, the model is expected to recognize the potential mistake, revisit the problem, correct the error, and consequently produce a more accurate solution.

Yet, the definition of "self-correction" varies across the literature, leading to ambiguity. A pivotal distinction lies in the source of feedback (Pan et al. 2023), which can purely come from the LLM, or can be drawn from external inputs. Internal feedback relies on the model's inherent knowledge and parameters to reassess its outputs. In contrast, external feedback incorporates inputs from humans, other models (Wang et al. 2023b, Paul et al. 2023, inter alia), or external tools and knowledge sources (Gou et al. 2023; Chen et al. 2023b; Olausson et al., 2023, Gao et al., 2023, inter alia).

In this work, we focus on examining the self-correction capability of LLMs for reasoning. Reasoning is a fundamental aspect of human cognition, enabling us to understand the world, draw inferences, make decisions, and solve problems. To enhance the reasoning performance of LLMs, Kim et al. (2023); Shinn et al. (2023) use oracle labels about the answer correctness to guide the self-correction process. However, in practice, high-quality external feedback such as answer correctness is often unavailable. For effective self-correction, the ability to judge the correctness of an answer is crucial and should ideally be performed by the LLM itself. Consequently, our focus shifts to self-correction without any external or human feedback. We term this setting intrinsic self-correction. For brevity, unless explicitly stated otherwise (e.g., self-correction with oracle feedback), all references to "selfcorrection" in the remainder of this paper pertain to intrinsic self-correction.

In the following sections, we will evaluate a variety of existing self-correction techniques. We demonstrate that existing techniques actually decrease reasoning performance when oracle labels are not used (Section 3 ), perform worse than methods without self-correction when utilizing the same number of model responses (Section 4 ), and lead to less effective outcomes when using informative prompts for generating initial responses (Section 5). We present an overview of issues in the evaluation setups of previous LLM self-correction works in Table 1, with detailed discussions in the corresponding sections.

Table 1: Summary of issues in previous LLM self-correction evaluation.

| Method | Issue |
| :---: | :---: |
| RCI (Kim et al., 2023); Reflexion (Shinn et al.. 2023) <br> Multi-Agent Debate (Du et al.. 2023, <br> Self-Refine (Madaan et al.. \|2023, | Use of oracle labels (Section 3 <br> Unfair comparison to self-consistency (Section 4 <br> Sub-optimal prompt design (Section |

## 3 LLMs CANNOT SELF-CORRECT REASONING INTRINSICALLY

In this section, we evaluate existing self-correction methods and compare their performance with and without oracle labels regarding the answer correctness.

### 3.1 EXPERIMENTAL SETUP

Benchmarks. We use datasets where existing self-correction methods with oracle labels have demonstrated significant performance improvement, including

- GSM8K (Cobbe et al. 2021): GSM8K comprises a test set of 1,319 linguistically diverse grade school math word problems, curated by human problem writers. There is a notable improvement of approximately $7 \%$ as evidenced by Kim et al. (2023) after self-correction.
- CommonSenseQA (Talmor et al. 2019): This dataset offers a collection of multi-choice questions that test commonsense reasoning. An impressive increase of around $15 \%$ is showcased through the self-correction process, as demonstrated by Kim et al. (2023). Following Kojima et al. (2022); Kim et al. (2023), we utilize the dev set for our evaluation, which encompasses 1,221 questions.
- HotpotQA (Yang et al., 2018): HotpotQA is an open-domain multi-hop question answering dataset. Shinn et al. (2023) demonstrate significant performance improvement through selfcorrection. We test models' performance in a closed-book setting and evaluate them using the same set as Shinn et al. (2023). This set contains 100 questions, with exact match serving as the evaluation metric.

Test Models and Setup. We first follow Kim et al. (2023); Shinn et al. (2023) to evaluate the performance of self-correction with oracle labels, using GPT-3.5-Turbo (gpt-3.5-turbo-0613) and GPT-4 accessed on 2023/08/29. For intrinsic self-correction, to provide a more thorough analysis, we also evaluate GPT-4-Turbo (gpt-4-1106-preview) and Llama-2 (Llama-2-70b-chat) (Touvron et al., 2023). For GPT-3.5-Turbo, we employ the full evaluation set. For other models, to reduce the cost, we randomly sample 200 questions for each dataset (100 for HotpotQA) for testing. We prompt the models to undergo a maximum of two rounds of self-correction. We use a temperature of 1 for GPT-3.5-Turbo and GPT-4, and a temperature of 0 for GPT-4-Turbo and Llama-2, to provide evaluation across different decoding algorithms.

Prompts. Following Kim et al. (2023); Shinn et al. (2023), we apply a three-step prompting strategy for self-correction: 1) prompt the model to perform an initial generation (which also serves as the results for Standard Prompting); 2) prompt the model to review its previous generation and produce feedback; 3) prompt the model to answer the original question again with the feedback.

For our experiments, we mostly adhere to the prompts from the source papers. For GSM8K and CommonSenseQA, we integrate format instructions into the prompts of Kim et al. (2023) to facilitate a more precise automatic evaluation (detailed prompts can be found in Appendix A). For HotpotQA, we use the same prompt as Shinn et al. (2023). We also assess the performance of various selfcorrection prompts for intrinsic self-correction. For example, we use "Assume that this answer could be either correct or incorrect. Review the answer carefully and report any serious problems you find." as the default feedback prompt for the evaluation on GPT-4-Turbo and Llama-2.

### 3.2 RESULTS

Self-Correction with Oracle Labels. Following previous works (Kim et al. 2023, Shinn et al. 2023), we use the correct label to determine when to stop the self-correction loop. This means we

Table 2: Results of GPT-3.5 and GPT-4 on reasoning benchmarks with oracle labels.

|  |  | GSM8K | CommonSenseQA | HotpotQA |
| :--- | :--- | :---: | :---: | :---: |
| GPT-3.5 | Standard Prompting | 75.9 | 75.8 | 26.0 |
|  | Self-Correct (Oracle) | 84.3 | 89.7 | 29.0 |
| GPT-4 | Standard Prompting | 95.5 | 82.0 | 49.0 |
|  | Self-Correct (Oracle) | 97.5 | 85.5 | 59.0 |

Table 3: Results of GPT-3.5 and GPT-4 on reasoning benchmarks with intrinsic self-correction.

|  |  | \# calls | GSM8K | CommonSenseQA | HotpotQA |
| :---: | :--- | :---: | :---: | :---: | :---: |
| GPT-3.5 | Standard Prompting | 1 | $\mathbf{7 5 . 9}$ | $\mathbf{7 5 . 8}$ | $\mathbf{2 6 . 0}$ |
|  | Self-Correct (round 1) | 3 | 75.1 | 38.1 | 25.0 |
|  | Self-Correct (round 2) | 5 | 74.7 | 41.8 | 25.0 |
|  | Standard Prompting | 1 | $\mathbf{9 5 . 5}$ | $\mathbf{8 2 . 0}$ | $\mathbf{4 9 . 0}$ |
|  | Self-Correct (round 1) | 3 | 91.5 | 79.5 | $\mathbf{4 9 . 0}$ |
|  | Self-Correct (round 2) | 5 | 89.0 | 80.0 | 43.0 |

utilize the ground-truth label to verify whether each step's generated answer is correct. If the answer is already correct, no (further) self-correction will be performed. Table 2 summarizes the results of self-correction under this setting, showcasing significant performance improvements, consistent with the findings presented in Kim et al. (2023); Shinn et al. (2023).

However, these results require careful consideration. For reasoning tasks, like solving mathematical problems, the availability of oracle labels seems counter-intuitive. If we are already in possession of the ground truth, there seems to be little reason to deploy LLMs for problem-solving. Therefore, the results can only be regarded as indicative of an oracle's performance.

Intrinsic Self-Correction. Per the above discussion, performance improvements achieved using oracle labels do not necessarily reflect true self-correction ability. Therefore, we turn our focus to the results in the intrinsic self-correction setting as defined in Section 2. To achieve this, we eliminate the use of labels, requiring LLMs to independently determine when to stop the self-correction process, i.e., whether to retain their previous answers.

Tables 3 and 4 report the accuracies and the number of model calls. We observe that, after selfcorrection, the accuracies of all models drop across all benchmarks.

To provide a more comprehensive assessment, we also design several different self-correction prompts to determine if there are better prompts that could enhance reasoning performance. Nonetheless, as shown in Tables 5 and 6 without the use of oracle labels, self-correction consistently results in a decrease in performance.

### 3.3 WHY DOES THE PERFORMANCE NOT INCREASE, BUT INSTEAD DECREASE?

Empirical Analysis. Figure 1 summarizes the results of changes in answers after two rounds of self-correction, with two examples of GPT-3.5 illustrated in Figure 2. For GSM8K, 74.7\% of the time, GPT-3.5 retains its initial answer. Among the remaining instances, the model is more likely to modify a correct answer to an incorrect one than to revise an incorrect answer to a correct one. The fundamental issue is that LLMs cannot properly judge the correctness of their reasoning. For CommonSenseQA, there is a higher chance that GPT-3.5 alters its answer. The primary reason for this is that false answer options in CommonSenseQA often appear somewhat relevant to the question, and using the self-correction prompt might bias the model to choose another option, leading to a high "correct $\Rightarrow$ incorrect" ratio. Similarly, Llama-2 also frequently converts a correct answer into an incorrect one. Compared to GPT-3.5 and Llama-2, both GPT-4 and GPT-4-Turbo are more likely to retain their initial answers. This may be because GPT-4 and GPT-4-Turbo have higher confidence

Table 4: Results of GPT-4-Turbo and Llama-2 with intrinsic self-correction.

|  |  | \# calls | GSM8K | CommonSenseQA |
| :--- | :--- | :---: | :---: | :---: |
| GPT-4-Turbo | Standard Prompting | 1 | $\mathbf{9 1 . 5}$ | $\mathbf{8 4 . 0}$ |
|  | Self-Correct (round 1) | 3 | 88.0 | 81.5 |
|  | Self-Correct (round 2) | 5 | 90.0 | 83.0 |
| Llama-2 | Standard Prompting | 1 | $\mathbf{6 2 . 0}$ | $\mathbf{6 4 . 0}$ |
|  | Self-Correct (round 1) | 3 | 43.5 | 37.5 |
|  | Self-Correct (round 2) | 5 | 36.5 | 36.5 |

Table 5: Results of GPT-4-Turbo with different feedback prompts.

|  | \# calls | GSM8K | CommonSenseQA |
| :--- | :---: | :---: | :---: |
| Standard Prompting | 1 | 91.5 | 84.0 |

Feedback Prompt: Assume that this answer could be either correct or incorrect. Review the answer carefully and report any serious problems you find.

| Self-Correct (round 1) | 3 | 88.0 | 81.5 |
| :--- | :--- | :--- | :--- |
| Self-Correct (round 2) | 5 | 90.0 | 83.0 |

Feedback Prompt: Review your previous answer and determine whether it's correct. If wrong, find the problems with your answer.

| Self-Correct (round 1) | 3 | 90.0 | 74.5 |
| :--- | :--- | :--- | :--- |
| Self-Correct (round 2) | 5 | 90.0 | 81.0 |

Feedback Prompt: Verify whether your answer is correct, and provide an explanation.

| Self-Correct (round 1) | 3 | 91.0 | 81.5 |
| :--- | :--- | :--- | :--- |
| Self-Correct (round 2) | 5 | 91.0 | 83.5 |

Table 6: Results of Llama-2 with different feedback prompts.

|  | \# calls | GSM8K | CommonSenseQA |
| :--- | :---: | :---: | :---: |
| Standard Prompting | 1 | 62.0 | 64.0 |

Feedback Prompt: Assume that this answer could be either correct or incorrect. Review the answer carefully and report any serious problems you find.

| Self-Correct (round 1) | 3 | 43.5 | 37.5 |
| :--- | :--- | :--- | :--- |
| Self-Correct (round 2) | 5 | 36.5 | 36.5 |

Feedback Prompt: Review your previous answer and determine whether it's correct. If wrong, find the problems with your answer.

| Self-Correct (round 1) | 3 | 46.5 | 26.0 |
| :--- | :--- | :--- | :--- |
| Self-Correct (round 2) | 5 | 30.5 | 37.0 |

Feedback Prompt: Verify whether your answer is correct, and provide an explanation.

| Self-Correct (round 1) | 3 | 58.0 | 24.0 |
| :--- | :--- | :--- | :--- |
| Self-Correct (round 2) | 5 | 41.5 | 43.0 |

in their initial answers, or because they are more robust and thus less prone to being biased by the self-correction prompt $]^{1}$[^1]

![](https://cdn.mathpix.com/cropped/2024_06_04_9324662417adfd778684g-06.jpg?height=673&width=1401&top_left_y=276&top_left_x=362)

Figure 1: Analysis of the changes in answers after two rounds of self-correction. No Change: The answer remains unchanged; Correct $\Rightarrow$ Incorrect: A correct answer is changed to an incorrect one; Incorrect $\Rightarrow$ Correct: An incorrect answer is revised to a correct one; Incorrect $\Rightarrow$ Incorrect: An incorrect answer is altered but remains incorrect.

![](https://cdn.mathpix.com/cropped/2024_06_04_9324662417adfd778684g-06.jpg?height=778&width=1396&top_left_y=1156&top_left_x=362)

Figure 2: Examples on GSM8K with GPT-3.5. Left: successful self-correction; Right: failed selfcorrection. Full prompts and responses can be viewed in Figures 3 and 4 of Appendix $\mathrm{A}$

Let's take another look at the results presented in Table 2. These results use ground-truth labels to prevent the model from altering a correct answer to an incorrect one. However, determining how to prevent such mischanges is, in fact, the key to ensuring the success of self-correction.

Intuitive Explanation. If the model is well-aligned and paired with a thoughtfully designed initial prompt, the initial response should already be optimal relative to the prompt and the specific decoding algorithm. Introducing feedback can be viewed as adding an additional prompt, potentially skewing the model towards generating a response that is tailored to this combined input. In an intrinsic self-correction setting, on the reasoning tasks, this supplementary prompt may not offer any extra advantage for answering the question. In fact, it might even bias the model away from producing an optimal response to the initial prompt, resulting in a performance drop.

Table 7: Results of multi-agent debate and self-consistency.

|  | \# responses | GSM8K |
| :--- | :---: | :---: |
| Standard Prompting | 1 | 76.7 |
| Self-Consistency | 3 | 82.5 |
| Multi-Agent Debate (round 1) | 6 | 83.2 |
| Self-Consistency | 6 | 85.3 |
| Multi-Agent Debate (round 2) | 9 | 83.0 |
| Self-Consistency | 9 | $\mathbf{8 8 . 2}$ |

Table 8: Results of Constrained Generation.

|  | \# calls | CommonGen-Hard |
| :--- | :---: | :---: |
| Standard Prompting* | 1 | $44.0^{*}$ |
| Self-Correct* | 7 | $67.0^{*}$ |
| Standard Prompting* | 1 | 53.0 |
| Self-Correct* | 7 | 61.1 |
| Standard Prompting (ours) | 1 | $\mathbf{8 1 . 8}$ |
| Self-Correct* | 7 | 75.1 |
| * Prompts |  |  |

* Prompts and results from Madaan et al. (2023).


## 4 MULTI-AGENT DEBATE DOES NOT OUTPERFORM SELF-CONSISTENCY

Another potential approach for LLMs to self-correct their reasoning involves allowing the models to critique and debate through multiple model calls (Du et al., 2023; Liang et al., 2023, Chen et al., 2023a). Du et al. (2023) implement a multi-agent debate method by leveraging multiple instances of a single ChatGPT model and demonstrate significant improvements on reasoning tasks. We adopt their method to test performance on GSM8K. For an unbiased implementation, we use the exact same prompt as Du et al. (2023) and replicate their experiment with the gpt-3.5-turbo-0301 model, incorporating 3 agents and 2 rounds of debate. The only distinction is that, to reduce result variance, we test on the complete test set of GSM8K, compared to their usage of 100 examples. For reference, we also report the results of self-consistency (Wang et al., 2022), which prompts models to generate multiple responses and performs majority voting to select the final answer.

Table 7 presents the results. The results indicate that both multi-agent debate and self-consistency achieve significant improvements over standard prompting. However, when comparing multi-agent debate to self-consistency, we observe that the performance of multi-agent is only slightly better than that of self-consistency with the same number of agents ( 3 responses, the baseline also compared in Du et al. (2023)). Furthermore, for self-consistency with an equivalent number of responses, multi-agent debate significantly underperforms simple self-consistency using majority voting.

In fact, rather than labeling the multi-agent debate as a form of "debate" or "critique", it is more appropriate to perceive it as a means to achieve "consistency" across multiple model generations. Fundamentally, its concept mirrors that of self-consistency; the distinction lies in the voting mechanism, whether voting is model-driven or purely based on counts. The observed improvement is evidently not attributed to "self-correction", but rather to "self-consistency". If we aim to argue that LLMs can self-correct reasoning through multi-agent debate, it is preferable to exclude the effects of selection among multiple generations.

## 5 PROMPT DESIGN ISSUES IN SELF-CORRECTION EVALUATION

In Section 3, we observe that although self-correction decreases reasoning performance with all types of feedback prompts we have evaluated, performance varies with different feedback prompts. In this section, we further emphasize the importance of proper prompt design in generating initial LLM responses to fairly measure the performance improvement achieved by self-correction. For example, if a task requires that the model response should meet criteria that can be easily specified
in the initial instruction (e.g., the output should contain certain words, the generated code should be efficient, the sentiment should be positive, etc.), instead of including such requirements only in the feedback prompt, an appropriate comparison would be to directly and explicitly incorporate these requirements into the prompt for generating initial responses. Otherwise, when the instruction for generating initial predictions is not informative enough, even if the performance improves, it is unclear whether the improvement merely comes from more detailed instructions in the feedback prompt or from the self-correction step itself.

To illustrate such prompt design issues in the self-correction evaluation of some prior work, we take the Constrained Generation task in Madaan et al. (2023) as an example, where the task requires models to generate coherent sentences using all 20-30 input concepts. The original prompt in Madaan et al. (2023) (Figure 7) does not clearly specify that the LLM needs to include all concepts in the prompt; thus, they show that self-correction improves task performance by asking the model to identify missing concepts and then guiding it to incorporate these concepts through feedback.

Based on this observation, we add the following instruction "Write a reasonable paragraph that includes *ALL* of the above concepts" to the prompt for initial response generation (refer to Figure 8 for the full prompt). Following Madaan et al. (2023), we use concept coverage as the metric. We reference their results and replicate their experiments using gpt-3.5-turbo-0613. Table 8 demonstrates that our new prompt, denoted as Standard Prompting (ours), significantly outperforms the results after self-correction of Madaan et al. (2023), and applying their self-correction prompt on top of model responses from our stronger version of the standard prompting again leads to a decrease in performance.

## 6 CONCLUSION AND DISCUSSION

Our work shows that current LLMs struggle to self-correct their reasoning without external feedback. This implies that expecting these models to inherently recognize and rectify their reasoning mistakes is overly optimistic so far. In light of these findings, it is imperative for the community to approach the concept of self-correction with a discerning perspective, acknowledging its potential and recognizing its boundaries. By doing so, we can better equip the self-correction technique to address the limitations of LLMs and develop the next generation of LLMs with enhanced capabilities. In the following, we provide insights into scenarios where self-correction shows the potential strengths and offer guidelines on the experimental design of future self-correction techniques to ensure a fair comparison.

Leveraging external feedback for correction. In this work, we demonstrate that current LLMs cannot improve their reasoning performance through intrinsic self-correction. Therefore, when valid external feedback is available, it is beneficial to leverage it properly to enhance model performance. For example, Chen et al. (2023b) show that LLMs can significantly improve their code generation performance through self-debugging by including code execution results in the feedback prompt to fix issues in the predicted code. In particular, when the problem description clearly specifies the intended code execution behavior, e.g., with unit tests, the code executor serves as the perfect verifier to judge the correctness of predicted programs, while the error messages also provide informative feedback that guides the LLMs to improve their responses. Gou et al. (2023) demonstrate that LLMs can more effectively verify and correct their responses when interacting with various external tools such as search engines and calculators. Cobbe et al. (2021); Lightman et al. (2023); Wang et al. (2023b) train a verifier or a critique model on a high-quality dataset to verify or refine LLM outputs, which can be used to provide feedback for correcting prediction errors. Besides automatically generated external feedback, we also often provide feedback ourselves when interacting with LLMs, guiding them to produce the content we desire. Designing techniques that enable LLMs to interact with the external environment and learn from different kinds of available feedback is a promising direction for future work.

Evaluating self-correction against baselines with comparable inference costs. By design, selfcorrection requires additional LLM calls, thereby increasing the costs for encoding and generating extra tokens. Section 4 demonstrates that the performance of asking the LLM to produce a final response based on multiple previous responses, such as with the multi-agent debate approach, is inferior to that of self-consistency (Wang et al. 2022) with the same number of responses. Regarding
this, we encourage future work proposing new self-correction methods to always include an in-depth inference cost analysis to substantiate claims of performance improvement. Moreover, strong baselines that leverage multiple model responses, like self-consistency, should be used for comparison. An implication for future work is to develop models with a higher probability of decoding the optimal solution in their answer distributions, possibly through some alignment techniques. This would enable the model to generate better responses without necessitating multiple generations.

Putting equal efforts into prompt design. As discussed in Section 5 , to gain a better understanding of the improvements achieved by self-correction, it is important to include a complete task description in the prompt for generating initial responses, rather than leaving part of the task description for the feedback prompt. Broadly speaking, equal effort should be invested in designing the prompts for initial response generation and for self-correction; otherwise, the results could be misleading.

## 7 LIMITATIONS AND BROADER IMPACT

Although we have conducted a comprehensive evaluation spanning a variety of self-correction strategies, prompts, and benchmarks, our work focuses on evaluating reasoning of LLMs. Thus, it is plausible that there exist self-correction strategies that could enhance LLM performance in other domains. For example, prior works have demonstrated the successful usage of self-correction that aligns model responses with specific preferences, such as altering the style of responses or enhancing their safety (Bai et al., 2022; Ganguli et al., 2023, Madaan et al., 2023). A key distinction arises in the capability of LLMs to accurately assess their responses in relation to the given tasks. For example, LLMs can properly evaluate whether a response is inappropriate Ganguli et al. 2023), but they may struggle to identify errors in their reasoning.

Furthermore, several prior works have already shown that LLM self-correction performance becomes significantly weaker without access to external feedback (Gou et al., 2023, Zhou et al., 2023a) and can be easily biased by misleading feedback (Wang et al., 2023a), which is consistent with our findings in this work. However, we still identified prevailing ambiguity in the wider community. Some existing literature may inadvertently contribute to this confusion, either by relegating crucial details about label usage to less prominent sections or by failing to clarify that their designed selfcorrection strategies actually incorporate external feedback. Regarding this, our paper serves as a call to action, urging researchers to approach this domain with a discerning and critical perspective. We also encourage future research to explore approaches that can genuinely enhance reasoning.

## REPRODUCIBILITY STATEMENT

Our experiments utilize GPT-3.5 and GPT-4, which are accessible via the public API at https: //platform.openai.com/docs/models, as well as Llama-2, an open-source model. To facilitate reproducibility, we detail the specific kernels used, e.g., gpt-3.5-turbo-0613, or provide the access times for each experiment. We use prompts from previous works when possible. For our designed prompts, we include the exact prompts in Appendix A

## ACKNOWLEDGEMENT

We would like to thank Chen Liang, William Cohen, Uri Alon, and other colleagues at Google DeepMind for valuable discussion and feedback.

## REFERENCES

Hussam Alkaissi and Samy I McFarlane. Artificial hallucinations in chatgpt: implications in scientific writing. Cureus, 15(2), 2023.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 675-718, 2023.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom B Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In USENIX Security Symposium, volume 6, 2021.

Justin Chih-Yao Chen, Swarnadeep Saha, and Mohit Bansal. Reconcile: Round-table conference improves reasoning via consensus among diverse llms. arXiv preprint arXiv:2309.13007, 2023a.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128, 2023b.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1-113, 2023. URL http: //jmlr.org/papers/v24/22-1144.html

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.

Deep Ganguli, Amanda Askell, Nicholas Schiefer, Thomas Liao, Kamilė Lukošiūtė, Anna Chen, Anna Goldie, Azalia Mirhoseini, Catherine Olsson, Danny Hernandez, et al. The capacity for moral self-correction in large language models. arXiv preprint arXiv:2302.07459, 2023.

Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. Rarr: Researching and revising what language models say, using language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16477-16508, 2023.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738, 2023

Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. In Findings of the Association for Computational Linguistics: ACL 2023. Association for Computational Linguistics, 2023.

Jie Huang, Hanyin Shao, and Kevin Chen-Chuan Chang. Are large pre-trained language models leaking your personal information? In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 2038-2047, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics.

Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. Advances in Neural Information Processing Systems, 2023.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, $35: 22199-22213,2022$.

Haoran Li, Dadi Guo, Wei Fan, Mingshi Xu, Jie Huang, Fanpu Meng, and Yangqiu Song. Multistep jailbreaking privacy attacks on chatgpt. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 4138-4153, 2023.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multiagent debate. arXiv preprint arXiv:2305.19118, 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 2023.

Theo X Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, and Armando SolarLezama. Demystifying gpt self-repair for code generation. arXiv preprint arXiv:2306.09896, 2023.

OpenAI. Gpt-4 technical report, 2023.

Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang. Automatically correcting large language models: Surveying the landscape of diverse selfcorrection strategies. arXiv preprint arXiv:2308.03188, 2023.

Debjit Paul, Mete Ismayilzada, Maxime Peyrard, Beatriz Borges, Antoine Bosselut, Robert West, and Boi Faltings. Refiner: Reasoning feedback on intermediate representations. arXiv preprint arXiv:2304.01904, 2023.

Hanyin Shao, Jie Huang, Shen Zheng, and Kevin Chen-Chuan Chang. Quantifying association capabilities of large language models and its implications on privacy leakage. arXiv preprint arXiv:2305.12707, 2023.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning, pp. 31210-31227. PMLR, 2023.

Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 2023.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149-4158, 2019.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Boshi Wang, Xiang Yue, and Huan Sun. Can chatgpt defend its belief in truth? evaluating llm reasoning via debate. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 11865-11881, 2023a.

Tianlu Wang, Ping Yu, Xiaoqing Ellen Tan, Sean O'Brien, Ramakanth Pasunuru, Jane Dwivedi-Yu, Olga Golovneva, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. Shepherd: A critic for language model generation. arXiv preprint arXiv:2308.04592, 2023b.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2022.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? arXiv preprint arXiv:2307.02483, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824-24837, 2022.

Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. Generating sequences by learning to self-correct. In The Eleventh International Conference on Learning Representations, 2023.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2018.

Shen Zheng, Jie Huang, and Kevin Chen-Chuan Chang. Why does chatgpt fall short in providing truthful answers? ArXiv preprint, abs/2304.10513, 2023.

Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv:2308.07921, 2023a.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, $2023 \mathrm{~b}$.

Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

# A PromptS AND EXAMPLE OUTPUTS 


#### Abstract

Can you solve the following math problem? Christina is planning a birthday party and needs . 75 gift bags per invited guest, because $1 / 4$ of attendees don't show up. She invited 16 friends. Gift bags are \$2 each. How much will she spend? Explain your reasoning. Your final answer should be a single numerical number, in the form \boxed\{answer\}, at the end of your response.

If Christina invited 16 friends, then $1 / 4$ of them won't show up, which means there will be 16 * $1 / 4=4$ friends who won't attend the party. To determine how many gift bags to buy, we need to subtract the number of friends who won't show up from the total number of invited guests. The number of invited guests is $16-4=12$. Since Christina needs 0.75 gift bags per invited guest, we can multiply this quantity by the number of invited guests to find the total number of gift bags needed, which is $0.75 * 12=9$ gift bags. If each gift bag is \$2, then Christina will spend a total of $2 * 9=\$<<2 * 9=18>>18$ on gift bags. Therefore, Christina will spend \$18 on gift bags. Answer: \boxed\{18\}. (Incorrect)


Review your previous answer and find problems with your answer.

Upon reviewing my previous answer, I recognize a mistake. I incorrectly stated that Christina needs 0.75 gift bags per invited guest. The given information states that $1 / 4$ of the attendees won't show up, so the correct calculation should take this into account. To calculate the number of gift bags needed, we will multiply the number of invited guests by the ratio of attendees who will actually show up (75\%). The number of invited guests is 16, and if $1 / 4$ of them won't show up, then the ratio of attendees who will show up is $1-1 / 4=3 / 4$. Multiplying the number of invited guests by the ratio of attendees who will show up gives us: 16 * $(3 / 4)=12$ gift bags needed. Since each gift bag costs $\$ 2$, the total amount Christina will spend on gift bags is: 12 * \$2 = \$<<12*2=24>>24.

Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form $\backslash$ boxed\{answer\}.

Upon reviewing my previous answer, I recognize a mistake. I incorrectly stated that Christina needs 0.75 gift bags per invited guest. The given information states that $1 / 4$ of the attendees won't show up, so the correct calculation should take this into account. To calculate the number of gift bags needed, we will multiply the number of invited guests by the ratio of attendees who will actually show up (75\%). The number of invited guests is 16, and if $1 / 4$ of them won't show up, then the ratio of attendees who will show up is $1-1 / 4=3 / 4$. Multiplying the number of invited guests by the ratio of attendees who will show up gives us: 16 * $(3 / 4)=12$ gift bags needed. Since each gift bag costs \$2, the total amount Christina will spend on gift bags is: 12 * \$2 = \$<<12*2=24>>24. Therefore, Christina will spend \$24 on gift bags. Answer: \boxed\{24\}. (Correct)

Figure 3: Example on GSM8K where self-correction changes an incorrect answer to a correct one. Text in bold denotes user prompts.

```
Can you solve the following math problem? Terry eats 2 yogurts a day.
They are currently on sale at 4 yogurts for $5.00. How much does he spend
on yogurt over 30 days? Explain your reasoning. Your final answer should
be a single numerical number, in the form \boxed{answer}, at the end of
your response.
First, we need to calculate how many sets of 4 yogurts Terry buys in 30
days. Since Terry eats 2 yogurts a day, he will need 2/4 = 0.5 sets of 4
yogurts per day. Next, we multiply the number of sets by the price of
each set to calculate how much Terry spends per day. Each set of 4
yogurts costs $5.00, so Terry spends 0.5*$5.00=$<<0.5*5=2.50>>2.50 on
yogurt per day. Finally, we multiply the amount Terry spends per day by
the number of days to find the total amount he spends on yogurt over 30
days. Terry spends $2.50 per day for 30 days, so he spends $2.50 * 30 =

```

![](https://cdn.mathpix.com/cropped/2024_06_04_9324662417adfd778684g-14.jpg?height=43&width=1190&top_left_y=1019&top_left_x=370)

```
$\boxed{75.00} $ on yogurt over 30 days. (Correct)
```

Review your previous answer and find problems with your answer.

Upon reviewing my previous answer, I realized that I made a mistake in calculating Terry's spending on yogurt over 30 days. I incorrectly stated that Terry spends $\$ 2.50$ per day for 30 days, resulting in a total of \$75.00. However, since Terry eats 2 yogurts a day and the yogurts are sold in sets of 4 for $\$ 5.00$, Terry would actually need to buy 1 set of 4 yogurts to last for 2 days. Therefore, his spending on yogurt per set would be $\$ 5.00$ for 2 days, or $\$ 5.00 / 2=\$ 2.50$ per day. To calculate the total amount spent on yogurt over 30 days, we would multiply the amount spent per day (\$2.50) by the number of sets needed in 30 days (which is 30 divided by 2 days per set). Therefore, the correct calculation would be $\$ 2.50 *(30 / 2)=\$ 2.50 * 15=\$ 37.50$. Apologies for the mistake. The correct answer is Terry spends \$\boxed\{37.50\}\$ on yogurt over 30 days.

Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form $\backslash$ boxed\{answer\}

Apologies for the error in my previous response. Upon reviewing the problem, here is the correct answer: Since Terry eats 2 yogurts a day and the yogurts are sold in sets of 4 for $\$ 5.00$, Terry needs to buy 1 set of 4 yogurts to last for 2 days. Therefore, Terry's spending on yogurt per set is $\$ 5.00$ for 2 days, or $\$ 5.00 / 2=\$ 2.50$ per day. To calculate the total amount spent on yogurt over 30 days, we multiply the amount spent per day (\$2.50) by the number of sets needed in 30 days. In this case, Terry needs $30 / 2=15$ sets of yogurts over 30 days. Multiplying \$2.50 by 15, we find that Terry spends a total of $\$ 2.50 * 15=\$ 37.50$ on yogurt over 30 days. Therefore, the final answer is Terry spends \$\boxed\{37.50\}\$ on yogurt over 30 days. (Incorrect)

Figure 4: Example on GSM8K where self-correction changes a correct answer to an incorrect one.

```
Can you solve the following math problem? Toulouse has twice as many
sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How
many sheep do Toulouse, Charleston, and Seattle have together if Seattle
has 20 sheep? Explain your reasoning. Your final answer should be a
single numerical number, in the form \boxed\{answer\}, at the end of your
response.
Let's start by figuring out how many sheep Charleston has. We are given
that Charleston has 4 times as many sheep as Seattle, so we can calculate
this by multiplying Seattle's number of sheep by 4: Charleston $=4$ * $20=$
80 sheep. Next, we are told that Toulouse has twice as many sheep as
Charleston. Therefore, we can find Toulouse's number of sheep by
multiplying Charleston's number of sheep by 2: Toulouse $=2 * 80=160$
sheep. Finally, to find the total number of sheep, we add up the number
of sheep for each city: Total = Seattle + Charleston + Toulouse $=20+80$
$+160=260$ sheep. Therefore, Toulouse, Charleston, and Seattle have a
total of 260 sheep. Answer: \boxed\{260\}. (Correct)
```

Review your previous answer and find problems with your answer.
I apologize for the mistake in my previous response. Let's go through the
problem again and find the correct answer. We are given that Charleston
has 4 times as many sheep as Seattle. If Seattle has 20 sheep, then
Charleston would have: Charleston $=4 * 20=80$ sheep. Next, we are told
that Toulouse has twice as many sheep as Charleston. Therefore, Toulouse
would have: Toulouse $=2 * 80=160$ sheep. To find the total number of
sheep, we add up the number of sheep for each city: Total = Seattle +
Charleston + Toulouse $=20+80+160=260$ sheep. After reviewing my
previous answer, I have determined that there are no problems with my
updated response. The correct answer is 260 sheep.
Based on the problems you found, improve your answer. Please reiterate
your answer, with your final answer a single numerical number, in the
form $\backslash$ boxed\{answer $\}$.
Apologies for the confusion. Let me provide an improved answer. We are
given that Charleston has 4 times as many sheep as Seattle, and Seattle
has 20 sheep. So, Charleston would have $4 * 20=80$ sheep. According to
the information, Toulouse has twice as many sheep as Charleston.
Therefore, Toulouse would have $2 * 80=160$ sheep. To determine the total
number of sheep, we add up the number of sheep for each city: Total =
Seattle + Charleston + Toulouse $=20+80+160=260$ sheep. After
reviewing the previous answer, the correct final answer is \boxed\{260\}.
(Correct)

Figure 5: Example on GSM8K where self-correction does not change the answer.

```
Q: A fencing thrust with a sharp sword towards a person would result in
what?
Answer Choices: (A) injury (B) small cuts (C) fever (D) competition (E)
puncture wound.
Explain your reasoning. You must choose only one option from A to E. Your
final answer should be a single letter from A to E, in the form (answer),
at the end of your response.
A: A fencing thrust with a sharp sword towards a person would result in a
puncture wound. This is because a fencing thrust involves forcefully
stabbing with a sword, which can penetrate the skin and create a deep,
narrow wound. Puncture wounds can be very dangerous as they can damage
internal organs and cause bleeding. Therefore, the most appropriate
answer is (E) puncture wound. (Correct)
```

Review your previous answer and find problems with your answer.
Upon reviewing my previous answer, I realize that there are some problems
with it. While a fencing thrust with a sharp sword can potentially result
in a puncture wound, it is important to note that in a controlled fencing
match, the swords used are typically blunted or have safety tips,
designed to minimize the risk of injury. Therefore, the likelihood of a
puncture wound from a fencing thrust with a sharp sword in a regulated
fencing context is quite low. A more appropriate and accurate answer
would be (D) competition, as a fencing thrust with a sharp sword is
typically performed in the context of a fencing match or competition,
rather than with the intention to cause injury.
Based on the problems you found, improve your answer. You must choose
only one option from A to E. Please reiterate your answer, with your
final answer a single letter from A to E, in the form (answer).
Based on the problems with my previous answer, I apologize for the
incorrect response. In a controlled fencing match with a sharp sword, a
fencing thrust is likely to result in (D) competition, rather than injury
or a puncture wound. Therefore, the correct final answer is (D)
competition. (Incorrect)

Figure 6: Example on CommonSenseQA where self-correction changes a correct answer to an incorrect one.

```
Concepts: ['footage', 'motion', 'ruin', 'tilt', 'window']
Sentence: time lapse footage with tilt up motion of the sun streaking
through window of ruin
###
Concepts: ['cause', 'hate', 'hut', 'local', 'love']
Sentence: new beach huts on the island have caused some controversy some
locals love them others hate them
[8 more in-context examples]
###
Concepts: ['vest', 'snow','remove', 'silhouette', 'dunk', 'step',
'item', 'toddler', 'plaster', 'wall', 'piece', 'tool', 'circle',
'pull','wrap', 'headphone', 'deck', 'suit', 'press', 'mother', 'pit',
'cupcake', 'priest', 'rink', 'counter', 'rest','piano', 'wine', 'serve',
'winter']
```

Sentence: A toddler wearing a vest and snow boots steps onto the icy
deck, removes a circle-shaped piece of plaster from the wall, and pulls
out a tool from their winter suit to wrap their head around a headphone.

Figure 7: Example output for Constrained Generation using the prompt from Madaan et al. (2023). The constraint that the output needs to contain all the concepts is not explicitly mentioned in the prompt for initial response generation. However, in the feedback \& refine prompts, the constraint is implied by "what concepts from the concept list are missing from the sentence?" (refer to Figures 25 and 26 of Madaan et al. (2023) for the feedback \& refine prompts).

```
Concepts: ['vest', 'snow','remove', 'silhouette', 'dunk', 'step',
'item', 'toddler', 'plaster', 'wall', 'piece', 'tool', 'circle',
'pull','wrap', 'headphone', 'deck', 'suit', 'press', 'mother', 'pit',
'cupcake', 'priest', 'rink', 'counter', 'rest','piano', 'wine', 'serve',
'winter']
```

Write a reasonable paragraph that includes *ALL* of the above concepts.
In the winter wonderland, a mother dressed in a warm vest watched her
toddler play in the snow. As the child attempted to remove the snow from
the ground, their silhouette created an enchanting image against the
white backdrop. Nearby, a group of friends gathered around a basketball
hoop, taking turns to dunk the ball with skillful steps. One of them,
holding an item resembling a tool, used it to plaster a piece of the wall
that had been damaged. Meanwhile, on the other side of the park, a circle
of children pulled their hats tightly to wrap around their heads,
protecting their ears from the cold wind as they listened to music
through their colorful headphones. On the deck of a nearby house, a man
in a sleek suit pressed a button, causing a hidden mechanism to reveal a
grand piano. As the music filled the air, the host served glasses of wine
to the guests, creating a warm and inviting atmosphere in the winter
evening. In the distance, a priest skated gracefully in an ice rink, his
movements mirroring the smoothness of his words during a sermon. At the
counter of a bakery, a customer ordered a cupcake and watched as the
baker skillfully decorated it with a winter-themed design. Not far away,
a group of friends gathered around a pit, roasting marshmallows and
enjoying the warmth of the fire. The scene painted a picture of joy,
friendship, and the beauty of winter.

Figure 8: Example output for Constrained Generation using a prompt casually written by the authors.


[^0]:    *Equal contribution.

[^1]:    ${ }^{1}$ We omit the analysis on HotpotQA because the sample size used in the source paper is quite small, which may not produce meaningful statistics.

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


