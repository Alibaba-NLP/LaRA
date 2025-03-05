<paper 0>
# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving 

Yinmin Zhong ${ }^{1}$ Shengyu Liu ${ }^{1}$ Junda Chen ${ }^{3}$ Jianbo Hu ${ }^{1}$ Yibo Zhu ${ }^{2}$ Xuanzhe Liu ${ }^{1}$<br>Xin Jin ${ }^{1}$ Hao Zhang ${ }^{3}$<br>${ }^{1}$ School of Computer Science, Peking University ${ }^{2}$ Independent Researcher ${ }^{3}$ UC San Diego


#### Abstract

DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding computation. Existing LLM serving systems colocate the two phases and batch the computation of prefill and decoding across all users and requests. We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases. LLM applications often emphasize individual latency for each phase: time to first token (TTFT) for the prefill phase and time per output token (TPOT) of each request for the decoding phase. In the presence of stringent latency requirements, existing systems have to prioritize one latency over the other, or over-provision compute resources to meet both.

DistServe assigns prefill and decoding computation to different GPUs, hence eliminating prefill-decoding interferences. Given the application's TTFT and TPOT requirements, DistServe co-optimizes the resource allocation and parallelism strategy tailored for each phase. DistServe also places the two phases according to the serving cluster's bandwidth to minimize the communication caused by disaggregation. As a result, DistServe significantly improves LLM serving performance in terms of the maximum rate that can be served within both TTFT and TPOT constraints on each GPU. Our evaluations show that on various popular LLMs, applications, and latency requirements, DistServe can serve $4.48 \times$ more requests or $10.2 \times$ tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for $>90 \%$ of requests.


## 1 Introduction

Large language models (LLMs), such as GPT-4 [32], Bard [2], and LLaMA [43], represent a groundbreaking shift in generative AI. They start to reshape existing Internet services, ranging from search engines to personal assistants [3], and enable fundamentally new applications, like universal chatbots [1, 14] and programming assistants [13, 37]. Yet, these advances come with a significant challenge: processing an end-to-end LLM query can be substantially slower than a

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-01.jpg?height=561&width=678&top_left_y=874&top_left_x=1168)

Figure 1: Performance when serving an LLM with 13B parameters under a synthetic workload with input length $=512$ and output length $=64$ on one NVIDIA 80GB A100. Upper: The P90 time-to-first-token (TTFT) latency comparing existing systems vs. a system serving only the prefill phase. Down: The P90 time-per-output-token (TPOT) latency comparing existing systems vs. a system serving only the decoding phase.

standard search query [36]. In order to meet the stringent latency requirements of various applications, service providers need to over-provision compute resources, particularly many GPUs, leading to a shortfall in cost efficiency. Therefore, optimizing the cost per LLM query while adhering to high $S L O$ attainment (the proportion of requests that meet the SLOs) is becoming increasingly essential for all LLM services.

An LLM service responds to a user query in two phases. The prefill phase processes a user's prompt, composed of a sequence of tokens, to generate the first token of the response in one step. Following it, the decoding phase sequentially generates subsequent tokens in multiple steps; each decoding step generates a new token based on tokens generated in previous steps, until reaching a termination token. This dualphase process distinguishes LLM services from traditional services - an LLM service's latency is uniquely measured by two key metrics: the time to first token (TTFT), which is the duration of the prefill phase, and the time per output
token (TPOT), which represents the average time taken to generate a token for each request (except for the first token) ${ }^{1}$. Different applications place varying demands on each metric. For example, real-time chatbots [1] prioritize low TTFT for response promptness, while TPOT only remains important until it is faster than human reading speed (i.e., 250 words/min). Conversely, document summarization emphasizes low TPOT for faster generation of the summary.

Hence, given the application's TTFT and TPOT requirements, an effective LLM serving system should balance these needs and maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, $90 \%$ ) for each GPU provisioned - higher per-GPU goodput directly translates into lower cost per query.

As the prefill and decoding phases share the LLM weights and working memory, existing LLM serving systems typically colocate both phases on GPUs and maximize the overall system throughput - tokens generated per second across all users and requests - by batching the prefill and decoding steps across requests $[27,45]$. However, to meet latency requirements, we find these systems must over-provision compute resources. To see this, Figure 1 illustrates how the P90 TTFT and TPOT shift with increasing request rates when serving a 13B LLM using existing systems [28], with workload pattern and two latency constraints set to emulate using LLM to generate a short summary for an article. Under the SLO attainment of $90 \%$, the maximum achievable goodput on a single A100 GPU, which is constrained by the more stringent one of TTFT and TPOT requirements, is about 1.6 requests per second (rps). The performance contrasts sharply when each phase is served independently on a separate GPU, shown by the orange and green curves, which achieve per-GPU goodput of $5.6 \mathrm{rps}$ for the prefill phase and $10 \mathrm{rps}$ for decoding. Ideally, by allocating 2 GPUs for prefill and 1 GPU for decoding, we can effectively serve the model with an overall goodput of $10 \mathrm{rps}$, or equally $3.3 \mathrm{rps}$ per GPU, which is $2.1 \mathrm{x}$ higher than existing systems. The gap in goodput primarily stems from the colocation of the prefill and decoding - two phases with very distinct computational characteristics and latency requirements (\$2.1).

First, colocation leads to strong prefill-decoding interference. A prefill step often takes much longer than a decoding step. When batched together, decoding steps in the batch are delayed by the prefill steps, significantly elongating their TPOT; similarly, the inclusion of decoding steps contributes to a non-trivial increase in TTFT, as evidenced in Figure 2. Even if we schedule them separately, issues persist as they begin to compete for resources. Decoding tasks awaiting GPU execution are subject to increased queuing delays due to ongoing prefill tasks, and vice versa. Prioritized scheduling of one phase risks failing the latency requirements of the other.

Second, the prefill and decoding computation differ in la-[^0]

tency requirements and preference for different forms of parallelism (\$3). Colocating prefill and decoding, however, couples their resource allocation, and prevents implementing different parallelism strategies more suited to meeting the specific latency requirements of each phase.

To overcome these challenges, we propose to disaggregate the prefill and decoding phases of LLM inference, assigning them to separate GPUs. Our approach has two benefits. First, operating each phase independently on different GPUs eliminates prefill-decoding interference. Second, it allows to scale each phase independently with tailored resource allocation and model parallelism strategies to meet their specific latency requirements. Although disaggregation causes communication of intermediate states between GPUs, we show that the communication overhead is insubstantial (\$3.3) in modern GPU clusters, and when managed appropriately, disaggregation significantly improves per-GPU goodput.

Based on the above insights, in this work, we build DistServe, a goodput-optimized LLM serving system by disaggregating the prefill and decoding phases. Given TTFT and TPOT requirements, DistServe first scales each phase independently by co-optimizing the GPU allocation and parallelism strategies of the prefill and decoding phase assuming serving a single model replica. The optimization ensures maximizing the per-GPU goodput and may assign different numbers of GPUs and parallelism strategies to each phase depending on their respective latency requirements. DistServe then scales this allocation to multiple instances via replication until meeting the user-required traffic rate (§4). DistServe also features an algorithm to place the prefill and decoding computation according to their allocation schemes and the cluster's bandwidth to minimize the overhead of communicating intermediate states between phases.

We implement DistServe as an orchestration layer on top of the LLM inference engine. We evaluate DistServe on various LLMs, varying the workloads based on three important realworld LLM applications: chatbots, programming assistant, and document summary. Compared to state-of-the-art solutions, DistServe can serve up to $4.48 \times$ more requests under latency constraints. Our contributions are:

- Identify the problems of prefill-decoding interference and resource coupling in existing LLM serving systems and propose to disaggregate the prefill and decoding phases.
- Design a novel placement algorithm to automatically choose the goodput-optimal schema for prefill and decoding instances.
- Conduct a comprehensive evaluation of DistServe with realistic workloads.


## 2 Background and Motivation

An LLM service follows a client-server architecture: the client submits a sequence of text as a request to the server; the server hosts the LLM on GPUs, runs inference over the request, and

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-03.jpg?height=404&width=868&top_left_y=237&top_left_x=173)

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-03.jpg?height=328&width=401&top_left_y=243&top_left_x=187)

(a) Input length $=128$

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-03.jpg?height=336&width=420&top_left_y=241&top_left_x=603)

(b) Input length $=1024$
Figure 2: Execution time for one batch when serving an LLM with 13B parameters as batch size increases. Compared between a decoding-only batch and a batch adding just one prefill request.

responds (or streams) the generation back to the client. As explained in $\S 1$, due to the unique prefill-decoding process, LLM service may impose aggressive service-level objectives (SLOs) on both TTFT and TPOT, varying with the application's needs. The serving system must meet both SLOs while minimizing the cost associated with expensive GPUs. In other words, we want the serving system to maximize the requests served per second adhering to the SLO attainment goal for each GPU provisioned - maximizing per-GPU goodput. Next, we detail the LLM inference computation (§2.1) and discuss existing optimizations for LLM serving (\$2.2).

### 2.1 LLM Inference

Modern LLMs $[32,43]$ predict the next token given an input sequence. This prediction involves computing a hidden representation for each token within the sequence. An LLM can take a variable number of input tokens and compute their hidden representations in parallel, and its computation workload increases superlinearly with the number of tokens processed in parallel. Regardless of the input token count, the computation demands substantial I/O to move LLM weights and intermediate states from the GPU's HBM to SRAM. This process is consistent across varying input sizes.

The prefill step deals with a new sequence, often comprising many tokens, and processes these tokens concurrently. Unlike prefill, each decoding step only processes one new token generated by the previous step. This leads to significant computational differences between the two phases. When dealing with user prompts that are not brief, the prefill step tends to be computation-bound. For instance, for a 13B LLM, computing the prefill of a 512-token sequence makes an A100 compute-bound. The larger the model, the shorter sequence is needed to turn the prefill step compute-bound (see $\S 3.1$ ). In contrast, the decoding phase, despite processing only one new token per step, incurs a similar level of I/O to the prefill phase, making it constrained by the GPU's memory bandwidth.

During both phases, intermediate states, known as KV caches [28], are generated at each token position, which are needed again in later decoding steps. To avoid recomputing them, they are saved in GPU memory. Because of the shared use of LLM weights and KV caches in memory, most LLM inference engines opt to colocate the prefill and decoding phases on GPUs, despite their distinct computational characteristics.

### 2.2 LLM Serving Optimization

In real-time online serving, multiple requests come and must be served within SLOs. Batching and parallelizing their computation is key for achieving low latency, high throughput, and high utilization of GPUs.

Batching. Current serving systems [8, 28, 45] utilize a batching technique known as continuous batching. This method batches the prefill of new requests with the decoding of ongoing ones. It boosts the GPU utilization and maximizes the overall system throughput - tokens generated per second across all users and requests. However, as mentioned in $\S 1$ and elaborated later in $\S 2.3$, this approach leads to trade-offs between TTFT and TPOT. An advanced variant of continuous batching [8] attempts to balance TTFT and TPOT by segmenting prefill and attaching decoding jobs in a manner that avoids exceeding GPU performance limits - but essentially, it trades TTFT for TPOT. In summary, batching prefill and decoding invariably leads to compromises in either TTFT or TPOT.

Model parallelism. In LLM serving, model parallelism is generally divided as intra- and inter-operator parallelisms [29, 39,50 ]. Both can be used to support larger models but may impact serving performance differently. Intra-operator parallelism partitions computationally intensive operators, such as matrix multiplications, across multiple GPUs, accelerating computation but causing substantial communication. It reduces the execution time ${ }^{2}$, hence latency, particularly for TTFT of the prefill phase, but requires high bandwidth connectivity between GPUs (e.g., NVLink). Inter-operator parallelism organizes LLM layers into stages, each running on a GPU to form pipelines. It moderately increases execution time due to inter-stage communication, but linearly scales the system's rate capacity with each added GPU. In this paper, we reveal an additional benefit of model parallelism: reduced queuing delay of both prefill and decoding phases, steaming from shorter execution time. We delve into this further in §3. Besides model parallelism, replicating a model instance, irrespective of its model parallelism configurations, linearly scales the system's rate capacity.

These parallelism strategies create a complex space of optimization that requires careful trade-offs based on the application's latency requirements.

### 2.3 Problems and Opportunities

Colocating and batching the prefill and decoding computation to maximize the overall system throughput, as in existing systems, is cost-effective for service providers. However, in[^1]the presence of SLOs, present approaches struggle to maintain both high service quality and low cost due to the issues discussed below.

Prefill-decoding interference. As Figure 2 shows, adding a single prefill job to a batch of decoding requests significantly slows down both processes, leading to a marked increase in TTFT and TPOT. Specifically, the decoding tasks in the batch must wait for lengthier prefill jobs to complete, thus extending TPOT; the slowdown intensifies with a longer prefill, shown in Figure 2(b). Adding decoding jobs to prefill also increases the time to complete the prefill task, particularly when the GPU is already at capacity (Figure 2 blue curves).

Ineffective scheduling. Unbatching prefill and decoding jobs and scheduling them sequentially does not mitigate the interference. Decoding jobs may experience longer queuing delays due to waiting for ongoing prefill jobs on GPUs. Moreover, batches dedicated to decoding often lead to GPU underutilization. Prioritizing tasks in either phase adversely affects the latency of the other, rendering priority scheduling ineffective.

Resource and parallelism coupling. Colocating prefill and decoding phases on the same GPUs unavoidably share their resource and parallelism settings. However, each phase has its unique computational characteristic and latency requirement that calls for more heterogeneous resource allocation. For example, the prefill phase benefits from more GPUs and intraop parallelism to reduce execution time to meet the tight SLO on TTFT. The decoding phase can handle a much higher rate using fewer GPUs than prefill, and its optimal parallelism configuration depends on the running batch size. In existing systems, due to coupling, resource allocation and parallelism plans are tailored to satisfy the more demanding of TTFT and TPOT, which may not be ideal for the other. This often leads to resource over-provisioning to meet both SLOs.

Opportunities. To address these issues, we propose to disaggregate the prefill and decoding phases. We use the term instance to denote a unit of resources that manages exactly one complete copy of model weights. One instance can correspond to many GPUs when model parallelism is applied. Note that when we disaggregate the two phases to different GPUs, each phase manages its copy of the model weights, resulting in prefill instances and decoding instances. A prefill instance, upon receiving a request, performs only the prefill computation for this request to generate the first output token. It then sends the intermediate results (mainly KV caches) to a decoding instance, which is responsible for subsequent decoding steps. Because decoding computation often has low GPU utilization, we may allocate multiple prefill instances per decoding instance. This allows batching more decoding jobs to achieve higher GPU utilization.

Disaggregating prefill and decoding naturally resolves the interference between the two phases and enables each to focus on its optimization target - TTFT or TPOT. Each type

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-04.jpg?height=320&width=421&top_left_y=244&top_left_x=1096)

(a) Prefill phase

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-04.jpg?height=317&width=418&top_left_y=245&top_left_x=1515)

(b) Decoding phase
Figure 3: Throughput for prefill phase and decoding phase with different batch size and input length when serving an LLM with 13B parameters.

of instance can employ different resources and parallelism strategies to meet a variety of latency requirements. By adjusting the number of GPUs and parallelisms provided to the two types of instances, we can maximize the per-device goodput of the overall system, avoiding over-provisioning, eventually translating to reduced cost-per-query adhering to service quality. Next, we develop ways to find out the best resource allocation and parallelism plan for each phase.

## 3 Tradeoff Analysis

Disaggregation uncouples the two phases and allows a distinct analysis of the characteristics of each phase, providing valuable insights into the algorithm design. It also expands the design space: now each phase needs to be scaled and scheduled independently based on their latency requirements.

In this section, we analyze the computational pattern of prefill (\$3.1) and decoding instances (\$3.2) post disaggregation. We aim to identify key parameters and derive guidelines for batching and parallelism in each phase. We then highlight several practical deployment considerations (§3.3). This section lays the foundation for per-gpu goodput optimization.

### 3.1 Analysis for Prefill Instance

After disaggregation, the prefill phase generates the first token by processing all tokens of the user prompt in parallel. Assuming a given arrival rate, our goal is to fulfill the service's latency requirement on TTFT using the least resources.

Batching strategy. The prefill step is typically computeintensive. Figure 3(a) shows how the throughput of the prefill phase changes with the input length and the batch size. For a 13B parameter LLM, processing a single sequence of 512 tokens can fully engage an A100 GPU; larger models require shorter sequences to reach GPU saturation. Once the GPU becomes compute-bound, adding more requests to the batch no longer improves GPU efficiency. Instead, it proportionally extends the total processing time for the batch, inadvertently delaying all included requests. Hence, for prefill instances, it is necessary to profile the specific LLM and GPUs in advance to identify a critical input length threshold, denoted as $L_{m}$, beyond which the prefill phase becomes compute-bound.

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-05.jpg?height=390&width=870&top_left_y=239&top_left_x=172)

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-05.jpg?height=322&width=401&top_left_y=243&top_left_x=187)

(a) Real experiment results

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-05.jpg?height=339&width=420&top_left_y=240&top_left_x=603)

(b) Changing intra-op speedup
Figure 4: Average TTFT when serving an LLM with 66B parameters using different parallelism on two A100 GPUs.

Batching more requests should only be considered when the input length of the scheduled request is below $L_{m}$. In practice, user prompts typically average over hundreds of tokens [7]. Batch sizes for the prefill instance are generally kept small.

Parallelism plan. To study the parallelism preferences for prefill-only instances, we serve a 66B LLM on two A100 GPUs with inter-op or intra-op parallelism strategy. To simplify the problem, we assume uniform requests input lengths of 512 tokens and a Poisson arrival process. We compare the resulting average TTFT at various arrival rates in Figure 4(a): intra-op parallelism is more efficient at lower arrival rates, while inter-op parallelism gains superiority as the rate increases. Disaggregation enables the prefill phase to function analogously to an M/D/1 queue, so we can use queuing theory to verify the observation.

We start by developing notations using the single-device case without parallelism: each request's execution time, denoted as $D$, remains constant due to uniform prefill length. Since one request saturates the GPU, we schedule requests via First-Come-First-Served (FCFS) without batching. Suppose the Poisson arrival rate is $R$ and the utilization condition of $R D<1$, the average TTFT (Avg_TTFT) can be modeled by the M/D/1 queue [40] in close form:

$$
\begin{equation*}
\operatorname{Avg}_{-} T T F T=D+\frac{R D^{2}}{2(1-R D)} \tag{1}
\end{equation*}
$$

where the first term represents the execution time and the second corresponds to the queuing delay. Based on Eq. 1, we incorporate parallelism.

With 2-way inter-op parallelism, we assume the requestlevel latency becomes $D_{s}$, and the slowest stage takes $D_{m}$ to finish. We have $D \approx D_{s} \approx 2 \times D_{m}$, due to negligible interlayer activation communication [29,50]. The average TTFT with 2-way inter-op parallelism is derived as:

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-05.jpg?height=97&width=712&top_left_y=2315&top_left_x=246)

For intra-op parallelism, we introduce a speedup coefficient $K$, where $1<K<2$, reflecting the imperfect speedup caused by high communication overheads of intra-op parallelism. With the execution time $D_{s}=\frac{D}{K}$, the average TTFT for 2degree intra-op parallelism is:

$$
\begin{equation*}
\text { Avg_TTFT }_{\text {intra }}=\frac{D}{K}+\frac{R D^{2}}{2 K(K-R D)} \text {. } \tag{3}
\end{equation*}
$$

Comparing Eq. 2 and Eq. 3: at lower rates, where execution time (first term) is the primary factor, intra-op parallelism's reduction in execution time makes it more efficient. As the rate increases and the queuing delay (second term) becomes more significant, inter-op parallelism becomes advantageous, concurred with Figure 4(a).

The prefill phase's preference for parallelism is also influenced by TTFT SLO and the speedup coefficient $K$. Seen from Figure 4(a): A more stringent SLO will make intra-op parallelism more advantageous, due to its ability to support higher request rates while adhering to SLOs. The value of $\mathrm{K}$ depends on factors such as the input length, model architecture, communication bandwidth, and placement $[39,50]$. As shown in Figure 4(b), a decrease in K notably reduces the efficacy of intra-op parallelism. §4 develops algorithms that optimize the resource and parallelism configurations taking into consideration these knobs.

### 3.2 Analysis for Decoding Instance

Unlike the prefill instance, a decoding instance follows a distinct computational pattern: it receives the intermediate states (KV caches) and the first token from the prefill instance and generates subsequent tokens one at a time. For decoding instances, our optimization goal is to satisfy the application's TPOT requirement using minimal computing resources.

Batching strategy. Since a single decoding job is heavily bandwidth-bound, batching is key to avoiding low GPU utilization (hence high per-gpu goodput). In existing systems where the prefill and decoding phases are colocated, increasing the decoding batch size is difficult because it conflicts with meeting latency goals, particularly in scenarios with high request rates. This is because sharing GPUs cause competition between prefill and decoding jobs, leading to a trade-off between TTFT and TPOT. For example, a higher arrival rate generates more prefill jobs, demanding greater GPU time to meet TTFT requirements if prioritizing prefill jobs, which in turn adversely affects TPOT.

On the contrary, disaggregation offers a solution by enabling the allocation of multiple prefill instances to a single decoding instance. This approach allows for accumulating a larger batch size on dedicated GPUs for the decoding phase without sacrificing TPOT.

Parallelism plan. Post-disaggregation, the batch size for decoding may be constrained by GPU memory capacity, as it is necessary to maintain the KV caches for all active requests. Scaling the decoding instance with model parallelism or leveraging advanced memory management techniques for LLM
![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-06.jpg?height=340&width=852&top_left_y=236&top_left_x=170)

Figure 5: Decoding phase latency and throughput when serving a 13B LLM with batch size $=128$ and input length $=256$ under different parallel degrees.

KV caches, such as Paged-Attention [28] and GQA [9], enable further scaling the decoding batch size to nearly computebound. As the decoding batch size continue to increase to approach the compute bound, the decoding computation begins to resemble the prefill phase. With this observation, we investigate how the latency and throughput change under different parallelism degrees under large batch conditions in Figure 5: intra-op parallelism reduces latency with diminishing returns, caused by communication and reduced utilization after partitioning. Inter-op parallelism can almost linearly scale the throughput. Hence, when the TPOT SLO is stringent, intra-op parallelism is essential to reduce TPOT to meet latency goals. Beyond this, inter-op parallelism is preferable to enhance throughput linearly.

It is worth noting that when the model can fit into the memory of a single GPU, replication is a competitive option in addition to model parallelism for both prefill and decoding instances, to linearly scale the system's rate capacity. It may also reduce the queuing delay - as indicated by Eq. 1 - by substituting $R$ with $R / N$ assuming requests are equally dispatched to $N$ replicas, at the cost of maintaining additional replicas of the model weights in GPU memory.

### 3.3 Practical Problems

We have developed foundational principles for selecting batching and parallelisms for each phase. In this section, we discuss and address several challenges encountered during the practical deployment of disaggregated prefill and decoding phases.

Variable prefill length. $\S 3$ has assumed uniform prompt length across requests. In real deployments, depending on the LLM application, the lengths of requests are non-uniform. The non-uniformity can cause pipeline bubbles [25,31] for prefill instances applying inter-op parallelism, because the execution time of pipeline stages across requests of different lengths will vary. This results in slight deviations from the conclusions indicated by using M/D/1 queue model. To address the problem, $\S 4$ develops algorithms that search for parallelisms based on workloads, and resort to scheduling to minimize the bubbles (\$4.3).

Communication overhead. Transferring KV caches from prefill to decoding instances incurs notable overheads. For example, the $\mathrm{KV}$ cache size of a single 512-token request on OPT-66B is approximately 1.13GB. Assuming an average arrival rate of 10 requests per second, we need to transfer $1.13 \times 10=11.3 \mathrm{~GB}$ data - or equivalently $90 \mathrm{Gbps}$ bandwidth to render the overhead invisible. The size of the KV caches increases with average input length and arrival rate. While many modern GPU clusters for LLMs are equipped with Infiniband (e.g., $800 \mathrm{Gbps}$ ), in cases where cross-node bandwidth is limited, disaggregation relies on the commonly available intra-node NVLINK, where the peak bandwidth between A100 GPUs is $600 \mathrm{~GB} / \mathrm{s}$, again rendering the transmission overhead negligible (see §6.3). However, this requirement imposes additional constraints on the placement of preill and decoding instances that we take into consideration in the next section.

Through the analysis in this section, we identify the workload pattern, placement constraints, SLO requirements, parallelism strategies, and resource allocation as key parameters that create a web of considerations in designing the disaggregated serving system. How to automatically navigate the search space to find the configuration that achieves optimal per-gpu goodput is challenging, and addressed next.

## 4 Method

We built DistServe to solve the above challenges. Given the model, workload characteristic, latency requirements, and SLO attainment target, DistServe will determine (a) the parallelism strategies for prefill and decoding instances, (b) the number of each instance type to deploy, as well as (c) how to place them onto the physical cluster. We call the solution a placement. Our goal is to find a placement that maximizes the per-gpu goodput.

As explained in $\S 3.3$, a key design consideration is to manage communications between disaggregated prefill and decoding phases, given varying cluster setups. In this section, we first present two placement algorithms: one for clusters with high-speed cross-node networks (\$4.1) and the other for environments lacking such infrastructure (\$4.2); the latter introduces additional constraints. We then develop online scheduling optimizations that adapt to the nuances of realworld workloads (\$4.3).

### 4.1 Placement for High Node-Affinity Cluster

On high node-affinity clusters equipped with Infiniband, KV caches transmission overhead across nodes is negligible, DistServe can efficiently deploy prefill and decoding instances across any two nodes without constraints. We propose a twolevel placement algorithm for such scenarios: we first optimize the parallelism configurations for prefill and decoding instances separately to attain phase-level optimal per-gpu goodput; then, we use replication to match the overall traffic rate.

However, finding the optimal parallel configuration for a single instance type, such as for the prefill instance, is still

```
Algorithm 1 High Node-Affinity Placement Algorithm
Input: LLM $G$, \#node limit per-instance $N$, \#GPU per-node
    $M$, GPU memory capacity $C$, workload $W$, traffic rate $R$.
Output: the placement best_plm.
    prefill_config $\leftarrow \emptyset$
    decode_config $\leftarrow \emptyset$
    for intra_op $\in\{1,2, \ldots, M\}$ do
        for inter_op $\in\left\{1,2, \ldots, \frac{N \times M}{\text { intra op }}\right\}$ do
            if $\frac{\text { G.size }}{\text { inter_op } \times \text { intra_op }}<C$ then
                    $\hat{G} \leftarrow$ parallel $(G$, inter_op,intra_op $)$
                    prefill_goodput $\leftarrow$ simu_prefill $(\hat{G}, W)$
                    decode_goodput $\leftarrow \operatorname{simu} \operatorname{decode}(\hat{G}, W)$
                    if $\frac{\text { prefill_config.goodput }}{\text { prefill_config.num_gpus }}<\frac{\text { prefill_goodput }}{\text { config.num_gpus }}$ then
                        prefill_config $\leftarrow$ config
                    if $\frac{\text { decode_config.goodput }}{\text { decode_config.num_gpus }}<\frac{\text { decode_goodput }}{\text { config.num_gpus }}$ then
                        decode_config $\leftarrow$ config
    $n \leftarrow\left\lceil\frac{R}{\text { prefill_config.goodput }}\right\rceil$
    $m \leftarrow\left\lceil\frac{R}{\text { decode_config.goodput }}\right\rceil$
    best_plm $\leftarrow$ (prefill_config, decode_config, $n, m)$
    return best_plm
```

challenging, due to the lack of a simple analytical formula to calculate the SLO attainment (a.k.a., percentage of requests that meet TTFT requirement), given that the workload has diverse input, output lengths, and irregular arrival patterns. Gauging the SLO via real-testbed profiling is time-prohibitive. We thus resort to building a simulator to estimate the SLO attainment, assuming prior knowledge of the workload's arrival process and input and output length distributions. Although short-term interval is impossible to predict, the workload pattern over longer timescales (e.g., hours or days) is often predictable [29, 46]. DistServe fits a distribution from the history request traces and resamples new traces from the distribution as the input workload to the simulator to compute the SLO attainment. Next, DistServe simply enumerates the placements via binary search and finds the maximum rate that meets the SLO attainment target with simulation trials.

Algorithm 1 outlines the process. We enumerate all feasible parallel configurations, subject to cluster capacity limit, for both prefill and decoding instances. For example, for a specific prefill phase configuration, we use simu_prefill to simulate and find their maximum goodput (similarly for using simu_decode for decoding). After determining the optimal parallel configurations for both prefill and decoding instances, we replicate them to achieve the user-required overall traffic rate according to their goodput.

The complexity of Algorithm 1 is $O\left(N M^{2}\right)$, with $N$ as the node limit per instance and $M$ representing the typical number of GPUs per node in modern clusters (e.g., 8). The search space is manageable and the solving time is under 1.3 minutes in our largest setting, as demonstrated in $\S 6.5$.

```
Algorithm 2 Low Node-Affinity Placement Algorithm
Input: LLM $G$, \#node limit per-instance $N$, \#GPU per-node
    $M$, GPU memory capacity $C$, workload $W$, traffic rate $R$.
Output: the placement best_plm.
    intra_node_config $\leftarrow \emptyset$
    for inter_op $\in\{1,2, \ldots, N\}$ do
        $\hat{G} \leftarrow$ parallel $(G$, inter_op $)$
        $\mathcal{P} \leftarrow$ get_intra_node_configs $(\hat{G}, M, C)$
        for $P \in \mathcal{P}$ do
            $P$. goodput $\leftarrow \operatorname{simulate}(\hat{G}, P, W)$
            if $\frac{\text { intra_node_config.goodput }}{\text { intra_node_config.num_gpus }}<\frac{P . \text { goodput }}{P . n u m \_g p u s}$ then
                intra_node_config $\leftarrow P$
    $n \leftarrow\left\lceil\frac{R}{\text { intra_node_config.goodput }}\right\rceil$
    best_plm $\leftarrow$ (inter_op,intra_node_config, $n$ )
    return best_plm
```

Simulator building. Algorithm 1 relies on a simulator to estimate the goodput under various SLOs and SLO attainment goals given the workload and the parallelism plan. To build an accurate simulator, we analyze the FLOPs and the number of memory accesses for prefill and decoding phases respectively, and use a latency model to approximate the inference execution time. See details in Appendix A. The simulator aligns well with real profiling results, thanks to the high predictability of DNN workloads [20,29], verified in §6.4.

By far, we have developed Algorithm 1 assuming we can place the prefill and decoding between any two nodes of the cluster, and the KV cache transmission utilizes high bandwidth. In many real clusters, GPUs inside a node access to high-bandwidth NVLINK while GPUs distributed across nodes have limited bandwidth. We next develop an algorithm to address this constraint.

### 4.2 Placement for Low Node-Affinity Cluster

A straightforward solution is to always colocate prefill and decoding instances on the same node, utilizing the NVLINK, which is commonly available inside a GPU node. For large models, e.g. with 175B parameters (350GB), we may be unable to even host a single pair of prefill and decoding instances in an 8 -GPU node $(80 G \times 8=640 G<350 \times 2 G B)$. We incorporate this as additional placement constraints and cooptimize it with model parallelism, presented in Algorithm 2.

The key insight is that intermediate states transfers occur exclusively between corresponding layers of prefill and decoding instances. Leveraging inter-op parallelism, we group layers into stages and divide each instance into segments, termed as instance segments, with each segment maintaining one specific inter-op stage. By colocating prefill and decoding segments of the same stage within a single node, we force the transfer of intermediate states to occur only via NVLINK. Inside a node, we set the same parallelism and resource allocation for segments of the same instance. Given the typical

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-08.jpg?height=412&width=783&top_left_y=257&top_left_x=210)

Figure 6: DistServe Runtime System Architecture

limitation of GPUs per node (usually 8), we can enumerate possible configurations inside one node and use the simulator to identify the configurations that yield the best goodput.

As outlined in Algorithm 2, we begin by enumerating interop parallelism degrees to get all the possible instance segments. For each segment, we get all possible intra-node configurations by calling get_intra_node_configs. Then we use simulation to find the optimal one and replicate it to satisfy the target traffic rate.

### 4.3 Online scheduling

The runtime architecture of DistServe is shown in Figure 6. DistServe operates with a simple FCFS scheduling policy. All incoming requests arrive at a centralized controller, then dispatched to the prefill instance with the shortest queue for prefill processing, followed by dispatch to the least loaded decoding instance for decoding steps. This setup, while simple, is optimized with several key enhancements tailored to the nuances of real-world workloads.

Reducing pipeline bubbles. To mitigate the pipeline bubbles caused by non-uniform prompt lengths ( $\$ 3.3$ ), we schedule the requests in a way that balances the execution time across all batches in the pipeline. This is achieved by noting that, for both prefill and decoding instances, the number of new tokens in the batch is a reliable indicator of the batch's real execution time. For prefill instances, we profile the target model and GPU to figure out the shortest prompt length $L_{m}$ needed to saturate the GPU. We schedule prefill batches with a total sequence length close to $L_{m}$, by either batching multiple requests shorter than $L_{m}$ or individually scheduling requests longer than $L_{m}$. For decoding instances, we set $L_{m}$ as the largest batch size.

Combat busrtiness. Burstiness in workloads can cause a deluge of $\mathrm{KV}$ caches to transfer from prefill to decoding instances, risking memory overload on decoding instances. To circumvent this, DistServe employs a "pull" method for KV cache transmission rather than a "push" approach - decoding instances fetch KV cache from prefill instances as needed, using the GPU memory of prefill instances as a queuing buffer. Hence, each type of instance operates at its own pace without complex coordination.
Replaning. The resource and parallelism plan in DistServe is optimized for a specific workload pattern, which may become suboptimal if the workload pattern changes over time. DistServe implement periodic replanning. A workload profiler monitors key parameters such as the average input and output length of the requests, the average arrival rate, etc. If a significant pattern shift is detected, DistServe will trigger a rerun of the placement algorithm based on recent historical data. This process is expedient - the proposed algorithm runs in seconds (§6.5) and reloading LLM weights can be completed within minutes - far shorter than the hourly scale at which real-world workload variations tend to occur.

DistServe does not implement advanced runtime policies like preemption [23] and fault tolerance [49], which are complementary to disaggregation. Nevertheless, we discuss how they fit into DistServe. In DistServe, the FCFS policy can lead to a "convoy effect", where longer requests block shorter ones in the prefill stage. Incorporating preemptive strategies, as suggested in existing literature [44], could enhance efficiency and is feasible within our system's architecture. While not a primary focus in the current DistServe, fault tolerance is a critical aspect for consideration. In traditional colocation- and replication-based systems, a fault in one instance typically does not disrupt other replica instances. However, in DistServe, the dependency between prefill and decoding instances introduces the risk of fault propagation. For example, a fault in a single decoding instance mapped to multiple prefill instances could potentially cripple the entire service and cluster. We leave both as future work.

## 5 Implementation

DistLLM is an end-to-end distributed serving system for LLMs with a placement algorithm module, a RESTful API frontend, an orchestration layer, and a parallel execution engine. The algorithm module, frontend, and orchestration layer are implemented with $6.5 \mathrm{~K}$ lines of Python code. The parallel execution engine is implemented with $8.1 \mathrm{~K}$ lines of C++/CUDA code.

The placement algorithm module implements the algorithm and the simulator mentioned in $\S 4$ which gives the placement decision for a specific model and cluster setting. The frontend supports OpenAI API compatible interface where clients can specify the sampling parameters like maximum output length and temperature. The orchestration layer manages the prefill and decoding instances, responsible for request dispatching, KV caches transmission, and results delivery. It utilizes NCCL [5] for cross-node GPU communication and asynchronous cudaMemcpy for intra-node communication, which avoids blocking the GPU during transmission. Each instance is powered by a parallel execution engine, which uses Ray [30] actor to implement GPU workers that execute the LLM inference and manage the KV Cache in a distributed manner. It integrates many recent LLM optimizations like continuous batching [45], FlashAttention [18], PagedAtten-

| Application | Model Size | TTFT | TPOT | Dataset |
| :---: | :---: | :---: | :---: | :---: |
| Chatbot OPT-13B | $26 \mathrm{~GB}$ | $0.2 \mathrm{~s}$ | $0.1 \mathrm{~s}$ | ShareGPT [7] |
| Chatbot OPT-66B | $132 \mathrm{~GB}$ | $0.4 \mathrm{~s}$ | $0.1 \mathrm{~s}$ | ShareGPT [7] |
| Chatbot OPT-175B | $350 \mathrm{~GB}$ | $4.0 \mathrm{~s}$ | $0.2 \mathrm{~s}$ | ShareGPT [7] |
| Code Completion OPT-66B | $132 \mathrm{~GB}$ | $0.125 \mathrm{~s}$ | $0.2 \mathrm{~s}$ | HumanEval [12] |
| Summarization OPT-66B | $132 \mathrm{~GB}$ | $15 \mathrm{~s}$ | $0.15 \mathrm{~s}$ | LongBench [11] |

Table 1: Workloads in evaluation and latency requirements.

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-09.jpg?height=344&width=870&top_left_y=549&top_left_x=169)

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-09.jpg?height=266&width=274&top_left_y=561&top_left_x=188)

(a) ShareGPT

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-09.jpg?height=266&width=271&top_left_y=561&top_left_x=474)

(b) HumanEval

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-09.jpg?height=265&width=287&top_left_y=564&top_left_x=735)

(c) LongBench
Figure 7: The input and output length distributions of (a) ShareGPT, (b) HumanEval, and (c) LongBench datasets.

tion [28] and supports popular open-source LLMs such as OPT [47] and LLaMA [43].

## 6 Evaluation

In this section, we evaluate DistServe under different sizes of LLMs ranging from 13B to 175B and various application datasets including chatbot, code-completion, and summarization. The evaluation shows that DistServe consistently outperforms the current state-of-the-art system across all the settings (§6.2). Specifically, DistServe can handle up to $4.48 \times$ higher rates and $10.2 \times$ more stringent SLO while meeting the latency requirements for over $90 \%$ requests. Additionally, we analyze the latency breakdown in DistServe to show the communication overhead is insubstantial thanks to our bandwidth-aware placement algorithm (§6.3) and do ablation studies of our techniques (§6.4).

### 6.1 Experiments Setup

Cluster testbed. We deploy DistServe on a cluster with 4 nodes and 32 GPUs. Each node has 8 NVIDIA SXM A10080GB GPUs connected with NVLINK. The cross-node bandwidth is $25 \mathrm{Gbps}$. Due to the limited cross-node bandwidth, we use the low node-affinity placement algorithm (\$2) for DistServe in most of the experiments except for the ablation study (§6.4) which uses simulation.

Model and workloads setup. Similar to prior work on LLM serving [28], we choose the OPT [47] series, which is a representative LLM family widely used in academia and industry. We use FP16 precision in all experiments. For workloads, as shown in Table 1, We choose three typical LLM applications and set the SLOs empirically based on their service target because there exists no available SLO settings for these applications as far as we know. For each application, we select a suitable dataset and sample requests from it for evaluation.
Since all the datasets do not include timestamps, we generate request arrival times using Poisson distribution with different request rates. Due to the space limit, we test the chatbot workload on all three OPT models and the other two workloads on OPT-66B, which matches the largest size in the recent open-source LLM series [43].

- Chatbot [1]: We use the ShareGPT dataset [7] for the chatbot application, which is a collection of user-shared conversations with ChatGPT. For OPT-13B, the TTFT SLO is set to 0.2 s for responsiveness and the TPOT SLO is set to $0.1 \mathrm{~s}$ which is higher than the normal human read speed. For OPT-66B and OPT-175B, we slightly relax the two SLOs due to the increase of model execution latency.
- Code completion [12]: We use the HumanEval [12] dataset for the code completion task. It includes 164 programming problems with a function signature or docstring which is commonly used in academia to evaluate code completion models. Since the code completion tool is used as a personal real-time coding assistant, we set both SLOs to be stringent.
- Summarization [4]: It is a popular LLM task to generate a concise summary for a long article, essay, or even an academic paper. We use LongBench [11] dataset which contains the summarization task. As shown in Figure 7, LongBench has much longer input lengths than the other two datasets. So we set a loose TTFT SLO but require a stringent TPOT.

Metrics. We use SLO attainment as the major evaluation metric. Under a specific SLO attainment goal (say, 90\%), we are concerned with two things: the maximum per-GPU goodput and the minimal SLO the system can handle. We are particularly interested in an SLO attainment of $90 \%$ (indicated by the vertical lines in all curve plots), but will also vary the rate and latency requirements to observe how the SLO attainment changes. To accurately understand the respective impacts of the two latency requirements on the system, we also present the proportion of requests that only meet one of these SLOs.

Baseline. We compare DistServe to the state-of-the-art serving system vLLM [28]. It supports iteration-level scheduling proposed by Orca [45] and PagedAttention to reduce memory fragmentation caused by KV cache allocation. However, it colocates and batches the prefill and decoding computation to maximize the overall system throughput and struggles to meet the latency requirements in a cost-efficient way. Since vLLM only supports intra-op parallelism, we follow previous work [28] to set intra-op equals 1, 4, and 8 for the three OPT models, respectively.

### 6.2 End-to-end Experiments

In this Section, we compare the end-to-end performance of DistServe against vLLM on real application datasets.

Chatbot. We evaluate the performance of DistServe on the chatbot application for all three OPT models. The first row

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-10.jpg?height=770&width=1699&top_left_y=241&top_left_x=213)

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-10.jpg?height=360&width=578&top_left_y=259&top_left_x=226)

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-10.jpg?height=315&width=574&top_left_y=623&top_left_x=228)

(a) OPT-13B
![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-10.jpg?height=676&width=570&top_left_y=262&top_left_x=798)

(b) OPT-66B
![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-10.jpg?height=676&width=552&top_left_y=262&top_left_x=1340)

(C) OPT-175B

Figure 8: Chatbot application with OPT models on the ShareGPT dataset.

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-10.jpg?height=428&width=1680&top_left_y=1114&top_left_x=228)

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-10.jpg?height=350&width=878&top_left_y=1126&top_left_x=233)

(a) Code Completion

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-10.jpg?height=344&width=808&top_left_y=1129&top_left_x=1081)

(b) Summarization

Figure 9: Code completion and summarization tasks with OPT-66B on HumanEval and LongBench datasets, respectively.

of Figure 8 illustrates that when we gradually increase the rate, more requests will violate the latency requirements and the SLO attainment decreases. The vertical line shows the maximum per-GPU rate the system can handle to meet latency requirements for over $90 \%$ of the requests. The dotted and dashed lines show the achieved SLO attainment for only TTFT or TPOT requirements, respectively.

On the ShareGPT dataset, DistServe can sustain $2.0 \times-$ $3.41 \times$ higher request rate compared to vLLM. This is because DistLLM eliminates the prefill-decoding interference through disaggregation. Two phases can optimize their own objectives by allocating different resources and employing tailored parallelism strategies. As a result, the gap between the curve that only meets TTFT requirements (Dist-TTFT) and the one that only meets TPOT requirements (Dist-TPOT) is relatively small. Specifically, by analyzing the chosen placement strategy $^{3}$ for $175 \mathrm{~B}$, we find the prefill instance has inter-op $=$ 3 , intra-op $=3$; and the decoding instance has inter-op $=3$,[^2]

intra-op $=4$. Under this placement, DistServe can effectively balance the load between the two instances on ShareGPT, meeting latency requirements at the lowest cost. This nontrivial placement strategy is challenging to manually find, proving the effectiveness of the algorithm. In the case of vLLM, collocating prefill and decoding greatly slows down the decoding phase, thereby significantly increasing TPOT. Due to the stringent TPOT requirements of chatbot applications, although vLLM meets the TTFT SLO for most requests, the overall SLO attainment is dragged down by a large number of requests that violate the TPOT SLO.

The second row of Figure 8 indicates the robustness to the changing latency requirements of the two systems. We fix the rate and then linearly scale the two latency requirements in Table 1 simultaneously using a parameter called SLO Scale. As SLO Scale decreases, the latency requirement is more stringent. We aim to observe the most stringent SLO Scale that the system can withstand while still achieving the attainment target. Figure 8 shows that DistServe can achieve $1.4 \times-1.8 \times$ more stringent SLO than vLLM, thus providing

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-11.jpg?height=379&width=848&top_left_y=244&top_left_x=172)

Figure 10: Left: Latency breakdown when serving OPT-175B on ShareGPT dataset with DistServe. Right: The CDF function of KV Cache transmission time for OPT models.

more engaging service quality to the users.

Code completion. Figure 9(a) shows the performance of DistServe on the code completion task when serving OPT-66B. DistServe can sustain $3.2 \times$ higher request rate and $1.5 \times$ more stringent SLO than vLLM. As a real-time coding assistant, the code completion task demands lower TTFT than chatbot, this leads to both systems ultimately being constrained by the TTFT requirement. However, in comparison, by eliminating the interference of the decoding jobs and automatically increasing intra-operation parallelism in prefill instances through the searching algorithm, DistServe reduces the average latency of the prefill jobs, thereby meeting the TTFT requirements of more requests.

Summarization. Figure 9 (b) shows the performance of DistServe on the summarization task when serving OPT-66B. DistServe achieves $4.48 \times$ higher request rate and $10.2 \times$ more stringent SLO than vLLM. The requests sampled from LongBench dataset have long input lengths, which brings significant pressure to the prefill computation. However, due to the loose requirement of TTFT for the summarization task, the TPOT service quality becomes particularly important. The vLLM, which collocates prefill and decoding phases, with long prefill jobs, experiences a greater slowdown in the decoding phase and fails to meet the TPOT requirement.

### 6.3 Latency Breakdown

To understand DistServe's performance in detail, we make a latency breakdown of the requests in DistServe. We divide the processing lifecycle of a request in DistServe into five stages: prefill queuing, prefill execution, transmission, decoding queuing, and decoding execution. The total time consumed by all requests in each stage is then summed up to determine their respective proportions in the system's total execution time.

Figure 10(a) shows the latency breakdown for the OPT175B models on ShareGPT dataset. We chose OPT-175B because the KV Cache transmission is more demanding for larger models. In fact, even for OPT-175B, the KV Cache transmission only accounts for less than $0.1 \%$ of the total latency. Even by examining the CDF of the absolute transmission time shown in Figure 10(b), we observe that over $95 \%$

| Rate <br> $(\mathrm{req} / \mathrm{s})$ | vLLM |  | DistServe-Low |  |
| :---: | :---: | :---: | :---: | :---: |
|  | Real System | Simulator | Real System | Simulator |
| 1.0 | $97.0 \%$ | $96.8 \%$ | $100.0 \%$ | $100.0 \%$ |
| 1.5 | $65.5 \%$ | $65.1 \%$ | $100.0 \%$ | $100.0 \%$ |
| 2.0 | $52.8 \%$ | $51.0 \%$ | $99.3 \%$ | $99.3 \%$ |
| 2.5 | $44.9 \%$ | $46.1 \%$ | $87.3 \%$ | $88.3 \%$ |
| 3.0 | $36.7 \%$ | $38.3 \%$ | $83.0 \%$ | $84.1 \%$ |
| 3.5 | $27.8 \%$ | $28.0 \%$ | $77.3 \%$ | $77.0 \%$ |
| 4.0 | $23.6 \%$ | $24.1 \%$ | $70.0 \%$ | $68.9 \%$ |

Table 2: Comparison of the SLO attainment reported by the simulator and the real system under different rates.

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-11.jpg?height=331&width=832&top_left_y=748&top_left_x=1096)

Figure 11: Ablation experiments.

of requests experience a delay of less than $30 \mathrm{~ms}$, despite our testbed having only limited cross-node bandwidth. This is due to the algorithm described in $\S 4.2$, where we require the prefill and decoding instance to maintain the same stage on one machine, enabling the use of intra-node NVLINK bandwidth for transmission, thus significantly reducing transmission delay.

### 6.4 Ablation Studies

We study the effectiveness of the two key innovations in DistServe: disaggregation and the placement searching algorithm. In $\S 6.2$, we choose the default parallelism setting for vLLM following its original paper [28]. So we implement "vLLM++" which enumerates different parallelism strategies and chooses the best. For DistServe, We also compare the placement found by Alg. 2 (DistServe-Low) with the one found by Alg. 1 (DistServe-High) which has fewer searching constraints and assumes high cross-node bandwidth. Since vLLM does not support inter-op parallelism and our physical testbed does not have high cross-node bandwidth, we use simulation for this experiment.

Simulator accuracy. Noticing that DNN model execution [21] has high predictability, even under parallel settings $[29,50]$. We study the accuracy of the simulator in Tab. 2. For "vLLM" and "DistServe-Low", we compare the SLO attainment reported by the simulator and by real runs on our testbed under different rates. The error is less than $2 \%$ in all cases, verifying the accuracy of our simulator.

Results. Figure 11 shows the performance of the four systems when serving OPT-13B on ShareGPT dataset. "vLLM++" has the same performance as "vLLM" because we find the de-

![](https://cdn.mathpix.com/cropped/2024_06_04_612977c49190976b6a37g-12.jpg?height=363&width=678&top_left_y=244&top_left_x=257)

Figure 12: Algorithm Running Time

fault non-parallelism setting has the best per-GPU goodput. This further demonstrates the importance of disaggregation. The interference between the prefill and decoding phases significantly reduces the potential performance improvement through adjusting parallelism. In contrast, "DistLLM-High" can achieve further improvements over "DistLLM-Low" because it is not constrained by the deployment constraint that the prefill and decoding instance on one node should share the same model stage. Through disaggregation, we can use tailored parallelism strategies for prefill and decoding instances and optimize their targets without the coupling effects.

### 6.5 Algorithm Running Time

Figure 12 shows the running time for Alg. 1 (DistServe-Low) and Alg. 2 (DistServe-High) on a AWS m5d.metal instance with 96 cores as the number of GPUs $(N \times M)$ provided to a single instance increases. According to the results, DistServe scales well with the number of GPUs and is independent of the model size. This is because the simulator only simulates discrete events and the running time is the same no matter how big the model is. On the other hand, both algorithms are highly parallelizable, as the searches for different parallelism strategies are independent of each other, allowing the execution time of the algorithms to accelerate almost linearly with more CPU cores.

As the number of GPUs increases, the execution time of "Dist-Low" becomes higher than that of "Dist-High". This is because the search for parallelism strategies for prefill and decoding instances in "Dist-High" is independent and can be parallelized. But for "Dist-Low", due to additional restrictions on deployment, we need to enumerate all the possible intra-node parallelism combinations for prefill and decoding instances. Even so, the execution time of the algorithm is in minutes, and since it only needs to be executed once before each redeployment, this overhead is acceptable.

## 7 Related Work

Inference serving. There has been plenty of work on inference serving recently. They range from general-purpose production-grade systems like TorchServe [6] and NVIDIA Triton [17] to systems optimized specifically for Transformerbased LLMs $[8,16,19,29,42,44,45,51]$. Among them, Orca [45] introduces iteration-level scheduling to increase throughput. vLLM [28] proposes a novel memory management strategy for KVCache. SARATHI [8] suggests a chunked-prefill approach, splitting a prefill request into chunks and piggyback decoding requests to improve hardware utilization. FastServe [44] implements iteration-level preemptive scheduling to mitigate the queuing delay caused by long jobs. However, they all employ a colocation approach for prefill and decoding processing, thus leading to severe interference. There are also concurrent works Splitwise [33], TetriInfer [24], and DéjàVu [41] which adopt similar disaggregation idea to optimize LLM inference, further confirming the effectiveness of this method. Differently, DistServe emphasizes the goodput optimization scenario more and takes a closer look at the aspect of network bandwidth.

Goodput-optimized systems. Optimizing goodput is a hot topic in DL applications. Pollux [34] improves scheduling performance in DL clusters by dynamically adjusting resources for jobs to increase cluster-wide goodput. Sia [26] introduces a heterogeneous-aware scheduling approach that can efficiently match cluster resources to elastic resourceadaptive jobs. Clockwork [20] and Shepherd [46] provide latency-aware scheduling and preemption to improve the serving goodput, but they only target traditional small models. AlpaServe [29] focuses on LLMs, employing model parallelism to statistically multiplex the GPU execution thus improving the resource utilization. However, it only targets the non-autoregressive generation. DistServe is the first work to optimize the goodput for autoregressive LLM inference.

Resource disaggregation. Resource disaggregated systems $[15,22,38]$ decouple the hardware resources from the traditional monolithic server infrastructure and separate them into different pools to manage independently. It allows for more flexible, efficient, and scalable deployment and increases resource utilization. Many applications benefit from a truly disaggregated data center with high-speed network bandwidth and heterogenous hardware support [10,48]. DistServe adopts a similar concept by disaggregating its system components, allowing for independent resource scaling and management.

Model parallelism for training. DistServe is orthogonal to the large body of work on model parallelism in training $[25,31,35,39,50]$. As described in $\S 3.3$, inference-serving workloads have unique characteristics not found in training settings. Where these systems do intersect with DistServe, is in their methods for implementing model parallelism along various dimensions. DistServe can integrate new parallelism optimizations into its placement searching algorithm.

## 8 Conclusion

We present DistServe, a new LLM serving architecture that disaggregates the prefill and decoding computation. DistServe maximizes the per-gpu goodput - the maximum request rate that can be served adhering to the SLO attainment goal for each GPU provisioned, hence resulting in up to $4.48 \times$ lower
cost per LLM query with guaranteed satisfaction of SLOs. Our findings affirm that as latency becomes an increasingly important metric for LLM services, prefill and decoding disaggregation is a vital strategy in promising improved performance and service quality guarantees.

## References

[1] Introducing chatgpt. https://openai.com/blog/ chatgpt, 2022.

[2] Bard, an experiment by google. https://bard. google.com/, 2023.

[3] Inflection tech memo. https://inflection.ai/ assets/Inflection-1.pdf, 2023.

[4] Lanchain usecase: Summarization, 2023.

[5] Nvidia collective communications library (nccl), 2023.

[6] Serve, optimize and scale pytorch models in production, 2023.

[7] Sharegpt teams. https://sharegpt.com/, 2023.

[8] Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, and Ramachandran Ramjee. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills. arXiv preprint arXiv:2308.16369, 2023.

[9] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints, 2023.

[10] Andrew Audibert, Yang Chen, Dan Graur, Ana Klimovic, Jiri Simsa, and Chandramohan A. Thekkath. A case for disaggregation of $\mathrm{ml}$ data processing, 2022.

[11] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding, 2023.

[12] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie
Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.

[13] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[14] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $90 \% *$ chatgpt quality, 2023.

[15] Compute Express Link Consortium. Compute express link, 2023. Accessed: 2023-12-07.

[16] NVIDIA Corporation. Fastertransformer, 2019.

[17] NVIDIA Corporation. Triton inference server: An optimized cloud and edge inferencing solution., 2019.

[18] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness, 2022.

[19] Jiarui Fang, Yang Yu, Chengduo Zhao, and Jie Zhou. Turbotransformers: an efficient gpu serving system for transformer models. In ACM PPoPP, 2021.

[20] Arpan Gujarati, Reza Karimi, Safya Alzayat, Wei Hao, Antoine Kaufmann, Ymir Vigfusson, and Jonathan Mace. Serving DNNs like clockwork: Performance predictability from the bottom up. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), pages 443-462. USENIX Association, November 2020.

[21] Arpan Gujarati, Reza Karimi, Safya Alzayat, Wei Hao, Antoine Kaufmann, Ymir Vigfusson, and Jonathan Mace. Serving DNNs like clockwork: Performance predictability from the bottom up. In USENIX OSDI, 2020 .

[22] Zhiyuan Guo, Zijian He, and Yiying Zhang. Mira: A program-behavior-guided far memory system. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP '23, page 692-708, New York, NY, USA, 2023. Association for Computing Machinery.

[23] Mingcong Han, Hanze Zhang, Rong Chen, and Haibo Chen. Microsecond-scale preemption for concurrent GPU-accelerated DNN inferences. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pages 539-558, Carlsbad, CA, July 2022. USENIX Association.

[24] Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, and Yizhou Shan. Inference without interference: Disaggregate $11 \mathrm{~m}$ inference for mixed downstream workloads, 2024.

[25] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism, 2019.

[26] Suhas Jayaram Subramanya, Daiyaan Arfeen, Shouxu Lin, Aurick Qiao, Zhihao Jia, and Gregory R Ganger. Sia: Heterogeneity-aware, goodput-optimized ml-cluster scheduling. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 642-657, 2023.

[27] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[28] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023.

[29] Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E Gonzalez, et al. Alpaserve: Statistical multiplexing with model parallelism for deep learning serving. arXiv, 2023.

[30] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan, and Ion Stoica. Ray: A distributed framework for emerging AI applications. In USENIX OSDI, 2018.

[31] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, and Matei Zaharia. Pipedream: Generalized pipeline parallelism for dnn training. In $A C M$ SOSP, 2019.

[32] OpenAI. Gpt-4 technical report, 2023.
[33] Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative llm inference using phase splitting, 2023.

[34] Aurick Qiao, Sang Keun Choe, Suhas Jayaram Subramanya, Willie Neiswanger, Qirong Ho, Hao Zhang, Gregory R. Ganger, and Eric P. Xing. Pollux: Co-adaptive cluster scheduling for goodput-optimized deep learning. In 15th USENIX Symposium on Operating Systems Design and Implementation (OSDI 21), pages 1-18. USENIX Association, July 2021.

[35] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models, 2020.

[36] Reuters, 2023.

[37] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

[38] Yizhou Shan, Yutong Huang, Yilun Chen, and Yiying Zhang. $\{$ LegoOS $\}$ : A disseminated, distributed $\{\mathrm{OS}\}$ for hardware resource disaggregation. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), pages 69-87, 2018.

[39] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020.

[40] John F Shortle, James M Thompson, Donald Gross, and Carl M Harris. Fundamentals of queueing theory, volume 399. John Wiley \& Sons, 2018.

[41] Foteini Strati, Sara Mcallister, Amar Phanishayee, Jakub Tarnawski, and Ana Klimovic. Déjàvu: Kv-cache streaming for fast, fault-tolerant generative llm serving, 2024.

[42] Yiming Su, Chengcheng Wan, Utsav Sethi, Shan Lu, Madan Musuvathi, and Suman Nath. Hotgpt: How to make software documentation more useful with a large language model? In Proceedings of the 19th Workshop on Hot Topics in Operating Systems, pages 87-93, 2023.

[43] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.

[44] Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, and Xin Jin. Fast distributed inference serving for large language models. arXiv preprint arXiv:2305.05920, 2023.

[45] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for \{Transformer-Based\} generative models. In USENIX OSDI, 2022.

[46] Hong Zhang, Yupeng Tang, Anurag Khandelwal, and Ion Stoica. Shepherd: Serving dnns in the wild. 2023.

[47] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022.
[48] Yiying Zhang. Make it real: An end-to-end implementation of a physically disaggregated data center. SIGOPS Oper. Syst. Rev., 57(1):1-9, jun 2023.

[49] Kai Zhao, Sheng Di, Sihuan Li, Xin Liang, Yujia Zhai, Jieyang Chen, Kaiming Ouyang, Franck Cappello, and Zizhong Chen. Ft-cnn: Algorithm-based fault tolerance for convolutional neural networks. IEEE Transactions on Parallel and Distributed Systems, 32(7):1677-1689, 2021.

[50] Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P. Xing, Joseph E. Gonzalez, and Ion Stoica. Alpa: Automating inter- and Intra-Operator parallelism for distributed deep learning. In USENIX OSDI, 2022.

[51] Zhe Zhou, Xuechao Wei, Jiejing Zhang, and Guangyu Sun. $\{$ PetS\}: A unified framework for $\{$ ParameterEfficient $\}$ transformers serving. In USENIX ATC, 2022.
</end of paper 0>


<paper 1>
# VIDUR: A LARGE-SCALE SIMULATION FRAMEWORK FOR LLM INFERENCE 

Amey Agrawal ${ }^{12}$ Nitin Kedia $^{3}$ Jayashree Mohan ${ }^{3}$ Ashish Panwar ${ }^{3}$ Nipun Kwatra ${ }^{3}$<br>Bhargav S. Gulavani ${ }^{3}$ Ramachandran Ramjee ${ }^{3}$ Alexey Tumanov ${ }^{1}$


#### Abstract

Optimizing the deployment of Large language models (LLMs) is expensive today since it requires experimentally running an application workload against an LLM implementation while exploring large configuration space formed by system knobs such as parallelization strategies, batching techniques, and scheduling policies. To address this challenge, we present Vidur - a large-scale, high-fidelity, easily-extensible simulation framework for LLM inference performance. Vidur models the performance of LLM operators using a combination of experimental profiling and predictive modeling, and evaluates the end-to-end inference performance for different workloads by estimating several metrics of interest such as latency and throughput. We validate the fidelity of Vidur on several LLMs and show that it estimates inference latency with less than $9 \%$ error across the range. Further, we present Vidur-Search, a configuration search tool that helps optimize LLM deployment. Vidur-Search uses Vidur to automatically identify the most cost-effective deployment configuration that meets application performance constraints. For example, Vidur-Search finds the best deployment configuration for LLaMA2-70B in one hour on a CPU machine, in contrast to a deployment-based exploration which would require $42 \mathrm{~K}$ GPU hours - costing 218K dollars. Source code for Vidur is available at https://github.com/microsoft/vidur.


## 1 INTRODUCTION

Large language models (LLMs) can learn from and generate natural language texts on a massive scale. LLMs such as GPT-3/4 (Brown et al., 2020; Bubeck et al., 2023), LLaMA (Touvron et al., 2023a), and Phi (Li et al., 2023) have demonstrated impressive performance on various natural language processing (NLP) tasks. However, LLM inference - the process of using an LLM to produce natural language outputs based on some input - is expensive. For example, the cost of serving ChatGPT is estimated to be $\$ 694 \mathrm{~K}$ per day (Patel \& Ahmed, 2023).

An LLM inference provider faces several challenges in optimizing LLM deployment. First, the provider has to choose a model parallelization strategy such as the number of tensor parallel dimensions, number of pipeline stages, number of replicas, etc. Second, the operator has to choose between different scheduling algorithms (e.g., Orca (Yu et al., 2022), vLLM (Kwon et al., 2023), Sarathi-Serve (Agrawal et al., 2024)). Third, the provider has to determine several configuration parameters, such as maximum batch size (BS), wait time for batching, as well as algorithm specific parameters[^0]

(e.g., chunk size in Sarathi, watermark fraction in vLLM) to satisfy the desired throughput and latency constraints. Finally, they have to generate representative workload traffic to test out each of their models on an experimental testbed with each of the different combinations above. Systematically optimizing deployment of tens of models with hundreds of configuration options is expensive and impractical.

This cost is further exacerbated by our observation that optimal configuration is a function of a model-trace pair, i.e., optimal configuration also depends on application workload characteristics (Figure 1a). In fact, an optimal config obtained on one trace could be sub-optimal by a factor of up to $2 \times$ (Figure 1b) when applied to the same model on a different trace. With both new models and new traces being released almost daily, the cost of identifying the optimal deployment configuration becomes prohibitively expensive.

To tackle this challenge, we present Vidur - a large-scale, high-fidelity and extensible LLM inference performance simulator, and Vidur-Search - a configuration search tool. Together, they enable fast and inexpensive exploration of LLM inference performance under a variety of deployment scenarios.

Simulating LLM inference poses several unique challenges that are not addressed in prior work that simulate the performance of deep neural network (DNN) training (Zhu et al., 2020; Yu et al., 2021; Lin et al., 2022). First, LLM inference predictions have to be accurate at much finer time granu-

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-02.jpg?height=461&width=960&top_left_y=236&top_left_x=192)

(a) Optimal configurations: Color bands correspond to the optimal config for each of the 12 model-trace pairs with corresponding throughput achieved per dollar.

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-02.jpg?height=463&width=575&top_left_y=251&top_left_x=1252)

(b) Cost of mis-configuration: the optimal config on one trace used for another results in up to $2 \times$ cost difference (LLaMA2-70B).

Figure 1. Both the model and workload matter for the optimal deployment configuration. Optimal configurations for each model-trace pair is shown in (a). Throughput/cost can differ significantly for the same model if the workload is changed as shown in (b).

larity compared to training jobs where each iteration runs for hundreds of milliseconds. Second, unlike training where batch sizes are typically fixed, the input sizes during inference can vary drastically. The difference in input sizes stems from varying sequences lengths of different requests, as well as the interleaving of prefill and decode stages depending on the scheduling strategy, resulting in significant variations in iteration latency. Since it is infeasible to experimentally profile the performance of the model for all possible input sizes, the simulator has to rely on a mixture of careful profiling and a prediction strategy for unprofiled input sizes. Third, small errors in predictions lead to cascading effect due to the dynamic and stateful nature of inference workloads, thus inference simulators need to provide extremely accurate per-iteration predictions to get good fidelity at high request arrival rates.

Vidur. To address these challenges, Vidur uses the key insight that the large majority of LLMs share similar architectures that can be decomposed into a small set of token-level, sequence-level and communication operators. Thus, Vidur takes in a model specification and first identifies various operators and a minimal set of input sizes that need to be profiled experimentally. Vidur then builds a fine-grained runtime estimator that accurately predicts kernel performance on input sizes that might not have been profiled. Using the estimator, Vidur takes a specification of deployment configuration and workload, and predicts a variety of request-level metrics such as Time to First Token (TTFT), Time Between Tokens (TBT), latency, throughput, as well as cluster-level metrics such as Model Flops Utilization (MFU) and memory utilization.

We demonstrate the fidelity of Vidur across a range of models, hardware and cluster configurations. Vidur accurately predicts request-level LLM inference performance with under $9 \%$ error rate, and mimics overall cluster metrics for large-scale workloads and traces with high fidelity.

Vidur-Bench. We find that the workload has a considerable impact on output metrics of interest in LLM inference. For example, variations in the number of input tokens, number of decode tokens and batch size can impact performance dramatically (Agrawal et al., 2023). We observe that there is no standardized benchmark suite available today to comprehensively evaluate LLM inference performance. Thus, we introduce Vidur-Bench to address this gap. Vidur-Bench is an easily extensible collection of workload traces along with several existing batching and scheduling policies such as vLLM (Kwon et al., 2023), Orca (Yu et al., 2022), FasterTransformer (fas) and Sarathi-Serve (Agrawal et al., 2024).

Vidur-Search. Finally, we present Vidur-Search to help LLM inference providers optimize their deployment. VidurSearch uses Vidur to automatically search over hundreds of deployment configurations to identify the highest throughput/cost configuration for a given model, workload pair. For example, for LLaMA2-70B, across a pool of A100 / H100 GPUs, Vidur-Search is able to identify the best configuration about one hour on a 96 -core CPU cores that costs $\$ 9.93$ per hour on Microsoft Azure, as opposed to an actual deployment-based exploration that would have taken $42 \mathrm{~K}$ GPU hours, costing approximately $\$ 218 \mathrm{~K}$.

In summary, this paper makes the following contributions.

- Vidur: an LLM inference simulator that predicts key performance metrics of interest with high-fidelity (§4)
- Vidur-Bench: a benchmark suite comprising of various workload patterns, schedulers and serving frameworks, along with profiling information for popular hardware
like A100 and H100 GPUs (\$5).
- Vidur-Search: a configuration search tool that helps optimize deployment by identifying the highest throughput per dollar configuration ( $\$ 6$ ).


## 2 BACKGROUND ANd Motivation

### 2.1 Overview of LLMs

LLMs utilize the transformer architecture based on the selfattention mechanism (Vaswani et al., 2017) as their core building block. The self-attention mechanism helps a language model learn the relationship between different elements of an input sequence and subsequently produce the output sequence. An LLM consists of two dominant submodules, self-attention and multilayer perceptron (MLP). Various LLMs have been developed in recent years using a variation of these modules (e.g., GPTs, LLaMAs, Falcons). Primarily, these models differ only in terms of the embedding size, the number of transformer blocks, and the attention mechanism used by the model.

### 2.2 LLM Inference Efficiency Optimizations

LLM inference request processing consists of two distinct phases - prefill and decode. The prefill phase processes the entire user input prompt and produces the first output token. Subsequently, output tokens are generated one at a time in an autoregressive manner. During this decode phase, the token generated in the previous step is passed through the model to generate the next token until a special endof-sequence token is generated at which point the request processing completes. The decode process requires access to the key and value activations of the previously processed tokens to perform the attention operation. To avoid repeated computation, contemporary LLM inference systems store them in $K V$-Cache.

Given the immense cost of LLM inference, LLM inference efficiency has become an active area of systems research. To this end, multiple optimization mechanisms have been proposed recently. Each of these techniques make different tradeoffs. For cost effective inference, right set of optimizations should be used be composed based on the specific application requirements. For example, Tensor Parallelism (TP) is a common strategy to parallelize LLM inference (Shoeybi et al., 2019; Pope et al., 2022). TP shards each layer across the participating GPUs by splitting the model weights and $K V$-Cache equally across GPU workers. TP (1) improves inference throughput with higher batch sizes, (2) lowers the latency of inference by splitting each operator across multiple GPUs. However, TP involves frequent blocking communication between workers, and thus requires expensive hardware with specialized high bandwidth interconnects like NVLINK. Alternatively, Pipeline
Parallelism (PP) is another parallelization strategy in which the model is partitioned into stages of consecutive transformer blocks. Each GPU is responsible for computing a stage and output activations are transferred across GPU boundaries via send/recv operations. PP has a much more favorable compute-communication ratio compared to TP, but can suffer from pipeline bubbles (stalls due to imbalance between stages).

Recently, Agrawal et al. 2024 identified an inherent tradeoff in LLM inference scheduler design and proposed classification of existing LLM inference schedulers into two categories - prefill prioritizing (Yu et al., 2022; Kwon et al., 2023) and decode prioritizing (fas). Prefill prioritizing schedules achieve higher throughput, by generating schedules with higher batch sizes, but suffer higher latency cost. Decode prioritizing schedulers can achieve low latency but at the cost of lower throughput (Kwon et al., 2023). SarathiServe (Agrawal et al., 2024) tries to mitigate this tradeoff by utilizing the computational slack in decode phase. Another set of recent works, Splitwise (Patel et al., 2023) and DistServe (Zhong et al., 2024) tackle this latency-throughput tradeoff by splitting the computation of prefill and decodes on separate devices.

Takeaway: Various systems optimizations provide a rich cost-latency tradoff. The right techniques to use depend on the application requirements and hardware availability.

### 2.3 LLM Inference Configuration Space

Control knobs like parallelism strategy, choice of scheduler, chunk size, batch size, SKU, etc. induce a large configuration space (Figure 1a) for LLM deployment. Furthermore, we make an important observation (Figure 1) that the optimal configuration (defined as a combination of specific choices for each control knob) is not simply a function of a specific model. But rather, the optimal configuration varies as a function of both the model $m$ and the trace $t$ evaluated on that model. Thus the complexity of configuration search is $O(|M| \cdot|T|)$, where $M$ is a set of all models of interest and $T$ is a set of workloads. With a rapid increase in both the number of models and downstream applications, the cost of optimal configuration search simply doesn't scale. And yet, misconfiguration is prohibitively expensive. For example, Figure $1 b$ shows that using the optimal configuration of one trace can have up to $2 \times$ cost differential on a different trace.

Takeaway: There is no single best deployment configuration for a model - rather the choice of configuration should be made in a workload-aware fashion.

With the cost of obtaining a single point in Figure 1 as high as $\$ 97 \mathrm{k}$, the high cost of misconfiguration, and the size of the search space growing with both models and traces, this begs a fundamental research question: is it possible to
find a performant configuration without requiring access to expensive experimental resources at a fraction of the cost? We explore this question in depth by proposing a simulationbased approach for LLM configuration search with Vidur, reducing the cost by several orders of magnitude.

## 3 ChAllengeS in SimUlating LLM INFERENCE

State-of-the-art DNN simulation frameworks (Daydream (Zhu et al., 2020), Habitat (Yu et al., 2021) and Proteus (Duan et al., 2023)) focus on training jobs. Building a large-scale inference simulator, especially for LLMs, involves multiple challenges that are not addressed by the existing simulators. We enumerate them in detail below.

Time Scale. Conventional DNN training workloads are typically compute-bound workload where each iteration executes for 100s of milliseconds (Zhu et al., 2020). In comparison, LLM inference is a far more latency-sensitive task where iterations can be much shorter (a few milliseconds each) (Yu et al., 2022; Kwon et al., 2023). Therefore, simulating LLM inference requires predicting iteration times at a much finer granularity.

Varying Iteration Times. Compared to traditional DL workloads where each iteration performs the same amount of compute and has predictable minibatch latency (Xiao et al., 2018), latency of different iterations can vary significantly during LLM inference. The variation in inference runtimes come from multiple sources. First, LLM inference consists of different phases - prefill and decode, each with a different compute characteristic and runtime. Second, the requests being processed may have a large variation in their sequence length (due to varying prompt lengths or number of decode tokens generated), resulting in varying runtimes. Third, the batch size during online inference keeps varying depending on the system load and workload characteristics. Moreover, the composition of a batch can accommodate requests from both prefill and/or decode phases, again adding to the runtime variation.

Cascading Errors. In training workloads, the batch composition is uniform across all batches, and the execution of each batch is independent. However, during inference, requests arrive in the system dynamically, and if the runtime prediction of any batch has significant errors, that can change in the batching pattern. Thus small errors in individual batch predictions cascade over time and lead to aggregate errors.

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-04.jpg?height=632&width=805&top_left_y=234&top_left_x=1061)

Figure 2. Vidur Simulator High Level Architecture.

## 4 VIDUR

Vidur leverages domain knowledge to provide high-fidelity performance estimations of LLM inference. It emulates the behavior of all layers of the inference stack, including both the model execution and the various tiers of request scheduling, at both replica as well as the cluster level.

### 4.1 Key Insights

LLMs Share Key Architectural Properties. The large majority of LLMs share fundamentally similar architectures with small differences in the choice of activation functions, normalization layers, residual connections, etc. This allows us to use a common declarative model specification format that captures the essential architectural choices of various models. Another consequence of this architectural uniformity is that Vidur only needs to model a small number of compute operators that are shared across all model families.

Operation Triaging for Runtime Prediction. In a running batch, each request may be associated with varying numbers of $K V$-Cache and query tokens, leading to a vast combinatorial input space. Consequently, profiling every possible combination to predict operation runtimes is not feasible. Instead, we observe that LLM operators can be classified into different categories. For instance, execution time of some operations depend on the total context length of all the requests in the batch whereas for others, it depends only on the number of tokens in the current iteration. This classification allows us to design tailored runtime prediction strategies for each operator type.

For example, we observe that apart from the attention kernel, all other operations are independent of request history. During the decode phase, the MLP layer would take the same amount of compute irrespective of the number of input or
output tokens processed previously. Profiling the attention kernel requires modeling history of each request. However, since the attention operation during decode is largely a memory-bound operation (Dao et al., 2022; Agrawal et al., 2023), we find that it is sufficient to model the total amount of $K V$-Cache to be fetched in a batch of requests to determine the kernel runtime (§4.3).

Automatic Profiling for Parallelism Strategies. Each model parallel configuration has different memory, compute, and network communication characteristics. A naive profile and replay approach would require a separate profiling run for each parallelism configuration, which can be expensive. In contrast, Vidur incorporates the domain knowledge about LLM parallelism strategies, which allows it to identify the subset of computation that is performed on each device. During the profiling phase, we automatically identify the tensor sharding configurations for each operator from a declarative specification of the model. Consequently, Vidur can simulate various parallelization schemes with minimal profiling performed on a single GPU.

### 4.2 System Overview

Vidur primarily has two phases of processing. First is the model onboarding phase wherein the model specification is used to generate a set of compute operators to be profiled. The Vidur profiler (§4.3) collects the runtime characteristics for the identified operators and feeds them to the runtime estimator. To minimize the cost barrier of adding new models to the system, we collect minimal data during the profiling phase and then train small machine-learning models to generate predictions over a large range of parameters that these operation could be triggered on during simulation. This phase is handled by Vidur's runtime estimator (§4.4), which produces operation-wise runtime lookup tables that can be later used during simulation.

Once the model is onboarded, the user can perform simulations using various scheduling policies, and parallelism strategies, across a wide range of workloads supported by Vidur-Bench (\$5). At the core of our event-driven simulator is a pluggable Hierarchical Scheduler (§4.5), which supports several popular batching strategies alongside memory planning and management capabilities. The simulator provides detailed metrics that capture both the request (normalized latency, time-to-first-token, time-between-tokens, etc.) and cluster (Model FLOPs utilization, $K V$-Cache utilization, etc.) performance metrics. The end-to-end process flow in Vidur is illustrated in Figure 2.

### 4.3 Profiler

To efficiently profile the runtime characteristics of LLMs, we leverage the insight that the large majority of LLMs share fundamentally similar architectures with small differences in the choice of activation functions, normalization layers, residual connections, etc.

Operator Triaging. The profiler analyzes different operators to identify their input dependencies. We find that all the operators can be placed on one of the three buckets:

- Token-level Operators: The operand dimensions for operations like linear, and activation functions depend on model architecture, however, their runtime only depends on the total number of tokens being processed (prefill plus decode) in the batch.
- Sequence-level Operators: The attention operation depends not only on the number of tokens in the current batch but also the context length of each request.
- Communication Operators: The runtime of communication operations like all-reduce and all-gather depend only on the amount of data to be transferred, independently of the model architecture.

Profiling Token-level Operators. There are two broad categories of token-level operators - matrix multiplications and simple point-wise apply or reduction operations, like addition, normalization, and activation functions. Based on the model specification, we generate all the different tensor parallel sharding configurations and profile each combination. This approach allows us to obtain traces for different parallelism configurations while profiling on a single GPU. We use standard PyTorch kernels for profiling these operations and measure their performance using CUPTI (cup).

Profiling Sequence-level Operators. Batching sequencelevel operators such as the attention kernels is sensitive to the context length of the requests in the batch, thereby exploding the state space of inputs to profile. We use several techniques to address this problem. First, we separately profile the attention kernels for prefill and decode phases due to their difference in compute characteristics.

While processing the prefill attention, we observe that the attention time for each prefill is quadratic in its length. Suppose we have a batch of $P$ prefills of length $p_{i}$, where $i$ varies from 1 to $P$. The cost of prefill attention for the whole batch is therefore proportional to $\Sigma_{i=1}^{P} p_{i}^{2}$. To approximate the runtime of this batch we predict the runtime of an equivalent batch of a single prefill of length $\sqrt{\sum_{i=1}^{P} p_{i}^{2}}$.

In contrast to prefill, we notice that the attention decode operation is largely memory-bound (Dao et al., 2022; Agrawal et al., 2023). As a result, the runtime of this operation is mainly determined by the total data volume that needs to be fetched from the $K V$-Cache and not the exact split of context lengths between different requests in the batch. In practice, the attention kernel might not be able to effectively parallelize $K V$-Cache fetch operation when there is a large
skew between the context length of different requests in a batch. However, we observe that sequence parallel attention kernels such as PagedAttention v2 (Kwon et al., 2023), and FlashDecoding (Dao et al., 2023) can effectively handle such skews, and thus it is sufficient to model decode based on total $K V$-Cache reads.

Profiling Communication Operators. There are three collective operations that are frequently used in LLM inference, namely, all-reduce, all-gather (used for tensor parallelism) and send-recv (used for pipeline parallelism). Since these operations don't depend on model-specific characteristics, we independently profile these kernels ahead of time in a model-agnostic manner for different topologies.

### 4.4 Runtime Estimator

Collecting profiling data for every possible input combination across all the operators is prohibitively expensive. Therefore, we collect a limited set of data points and rely on small machine-learning models to interpolate the runtimes. Runtime Estimator first trains these models using the profiled data, and then generates runtime estimates for a large range of input tensor dimensions which it encounters in end-to-end simulation.

Prior DL training simulators (Yu et al., 2021; Lin et al., 2022) train Multi-layer Perceptron (MLP) models for opaque operations like matrix multiplications which are provided by closed-source third-party libraries like CUBLAS (NVIDIA Corporation, a) and cuDNN (Chetlur et al., 2014). However, training MLPs requires a large amount of data and results. On the other hand, simple polynomial regression does not capture the non-linear runtime characteristics of CUDA kernels due to phenomenons like tile and wave quantization (NVIDIA Corporation, b). For our scenario, we find that random forest (RF) regression models achieve the right balance between data frugality and fidelity.

### 4.5 Hierarchical Scheduler

In Vidur we adopt a three-tier hierarchical scheduler architecture, that provides a powerful and extensible interface. First is the global scheduler, that is responsible for request routing in Vidur. In addition to standard load balancing policies like round-robin and least outstanding requests, we also support stateful scheduling policies, where routing decisions can be deferred to a later point in time, which can be helpful under busty workloads where early binding routing decisions can hurt performance.

Second is the replica scheduler that encapsulates two key responsibilities; batching and memory management. The replica scheduler contains a memory planner, which uses the model specification and parallelism configuration to compute the memory available for $K V$-Cache. This information is then used by the memory manager to provide high-level management APIs that are used to implement custom batching policies. Vidur currently supports five batching policies, FasterTransformers (fas), Orca (Yu et al., 2022), SarathiServe (Agrawal et al., 2024), vLLM (Kwon et al., 2023) and LightLLM (lig, 2023). The high-level API support provided by Vidur makes it extremely simple to implement new batching policies; all the aforementioned policies have been implemented each in less than 150 lines of Python code in our simulator

The final component of our scheduling stack is the replica stage scheduler, which handles the scheduling of microbatches within a pipeline stage. While we currently only support synchronous pipeline parallel scheduling policy, in the future, we aim to extend the replica stage scheduler to emulate various optimizations like asynchronous communication, sequence parallelism (Li et al., 2021) and speculative pipelined decoding (Hooper et al., 2023).

## 5 VIDUR-BENCH

Vidur-Bench is a benchmark suite for easy evaluation performance evaluation of LLM inference systems that comprises of plug-and-play support for a variety of (a) workload patterns, (b) scheduling, batching, and routing policies, and (c) serving frameworks.

### 5.1 Datasets and workloads

The overall performance of LLM inference is highly sensitive to the type of workloads such as the number of input and output tokens in a given query e.g., the decode phase can be as high as $200 \times$ more expensive than the prefill phase (Agrawal et al., 2023). Different workload patterns can therefore influence system performance in complex ways. For instance, vLLM incrementally allocates physical memory for the $K V$-Cache in order to fit a large batch size on the GPU. This works well when the number of decode tokens is high e.g., in chat applications (Zheng et al., 2023). In contrast, incremental memory allocation is less useful if the prompt length is much higher than the number of output tokens as in summarization tasks.

Vidur-Bench provides a set of workloads curated from publicly available datasets (see Table 1). These can be used to evaluate system performance for varying request types, arrival rates etc. or to tune the performance sensitive parameters of various components in the serving system.

### 5.2 Performance metrics

Vidur-Bench provides a comprehensive set of system-level performance metrics as discussed below:

Operator-level metrics. This includes each operator's input

Vidur: A Large-Scale Simulation Framework for LLM Inference

| Dataset | Content | \# queries | \# prefill tokens |  |  | \# decode tokens |  |  | P:D Ratio |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  | mean | median | p90 | mean | median | $\mathrm{p} 90$ | median | std dev |
| LMSys-Chat-1M [ Zheng et al. 2023] | Natural language conversations | $2 \mathrm{M}$ | 786 | 417 | 1678 | 215 | 141 | 491 | 2.3 | 236 |
| (Chat-1M) | LMSys-Chat-1M with max $4 \mathrm{k}$ total tokens | $2 \mathrm{M}$ | 686 | 417 | 1678 | 197 | 139 | 484 | 2.3 | 228 |
| Arxiv-Summarization [ Cohan et al. 2018] | Summarization of arxiv papers | $203 \mathrm{k}$ | 9882 | 7827 | 18549 | 411 | 228 | 475 | 35.4 | 81 |
| (Arxiv-4K) | Arxiv-Summarization with max $4 \mathrm{k}$ total tokens | $28 \mathrm{k}$ | 2588 | 2730 | 3702 | 291 | 167 | 372 | 15.7 | 16 |
| Bilingual-Web-Book [ Jiang et al. 2023] | Document-level English-Chinese parallel dataset | $195 \mathrm{k}$ | 2418 | 2396 | 3441 | 3654 | 3589 | 5090 | 0.66 | 0.23 |
| (BWB-4K) | Bilingual-Web-Book with max $4 \mathrm{k}$ total tokens | $33 \mathrm{k}$ | 1067 | 1037 | 1453 | 1612 | 1601 | 2149 | 0.65 | 0.37 |

Table 1. Details of the workloads curated from open-source datasets.

size and execution time which can be used to identify and optimize the heavy-duty operators eg. attn_prefill, mlp_up_proj etc.

Request-level metrics. These include per-request metrics such as the scheduling delay, prefill completion time, timeto-first-token (TTFT), and time-between-tokens (TBT). Furthermore, any additional metrics of interest can be easily added, e.g., we added support to track how many times vLLM preempts or restarts each request when it runs out of GPU memory for $K V$-Cache.

Replica-level metrics. These include metrics such as the batch size, the number of tokens processed in each iteration, busy and idle times as well as the memory and compute utilization of each replica.

Hardware metrics. These capture cluster-wide GPU FLOPs and memory utilization. We plan to extend these to also capture the cluster's energy consumption.

## 6 VIDUR-SEARCH

When deploying an inference system, the system operator needs to take into account various aspects. For example, there may be SLOs on latency metrics such as TTFT and TBT or minimum QPS that needs to be supported. At the same time, the operator can try multiple configurations such as the GPU SKU (e.g. A100 vs H100) to use for deployment, the parallelization strategy (TP vs PP), scheduling policy (Orca, vLLM, Sarathi-Serve, etc.), replication degree, etc. Vidur-Search is a tool which helps find the optimal cost configurations to deploy an inference system while satisfying the desired SLO constraints. Vidur-Search leverages our simulator to compute the optimal configuration in an efficient manner. Along with the optimal configuration, Vidur-Search also gives detailed visualizations of how changes in configurations impact cost, TTFT, TBT, etc.

Vidur-Search has the following main components:

Input. The input to the search tool consists of the LLM model, the workload (request characteristics can significantly affect inference performance), available GPU SKUs, and maximum number of GPUs in a replica.

Constraints. SLOs on metrics such as TTFT and TBT.

Search space. The search tool has the freedom to config- ure the parallelism strategy (TP vs PP), parallelism degree, scheduling policy, scheduler specific parameters (e.g. chunk size in Sarathi), batch size, choice of GPU, SKU, etc.

Optimization objective. Vidur-Search helps the operator maximize QPS per dollar. Consider a deployment with 16 A100 GPUs. Capacity of the system is defined as the maximum queries per second that it can support without the queuing delay blowing up. Specifically we constrain the P99 scheduling delay to be under 5 seconds. This QPS value is divided by the cost of renting 16 A100 GPUs per hour to get the QPS per dollar value.

Given the above, Vidur-Search needs to solve a constrained optimization problem to find the optimal configuration in the search space. Vidur-Search starts with first enumerating all possible deployment configurations of the system. For each configuration, we can run our simulator on the input workload at a specified QPS and predict the metrics such as TTFT and TBT. Note, however, that the possible QPS values to pass to the simulator can be infinite. To get around this, we instead target to find the maximum QPS that a given configuration can support. We do this by tracking the scheduling delay of requests for a given configuration and QPS. Note that any system configuration will have a maximum QPS capacity for a given workload at which it can process the input requests without accumulating the request queue. We use this property to find the maximum QPS supported by a system via a simple binary search which searches for the maximum QPS which does not increase the scheduling delay beyond a threshold. Each step of this binary search involves running our simulator for the corresponding configuration and QPS. We parallelize these runs by running each search on a separate core. After this search, we have for each configuration, the maximum QPS which is supported by the system. Finally, Vidur-Search analyzes this data to output the optimal configuration and also generates visualizations of how changes in configurations impact the various metrics.

Since the number of configurations that need to be evaluated can be very large (in 1000 s), doing a naïve search on actual hardware will be extremely costly. At the same time, a suboptimal choice of configuration can be very costly in the long run. Moreover, since the optimal configuration depends on the input workload, and the workload can change over time; it may be prudent to repeat this search whenever the

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-08.jpg?height=800&width=1566&top_left_y=202&top_left_x=255)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-08.jpg?height=341&width=447&top_left_y=217&top_left_x=275)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-08.jpg?height=324&width=445&top_left_y=600&top_left_x=276)

InternLM-20B (TP2)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-08.jpg?height=299&width=342&top_left_y=257&top_left_x=729)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-08.jpg?height=326&width=354&top_left_y=232&top_left_x=1080)

(a) Median Normalized Execution Latency
![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-08.jpg?height=300&width=1072&top_left_y=614&top_left_x=728)

(b) P95 Normalized Execution Latency

Figure 3. Fidelity of Vidur's request execution time prediction for four models and three static traces.

workload characteristics have diverged from the original workload. The use of simulator in Vidur-Search makes this practical, by reducing this search cost by many orders of magnitude. We leverage Vidur-Search for our what-if analysis in $\S 7.3$.

Note that while Vidur-Search is primarily designed for configuration optimization of online serving systems, it can be repurposed for offline inference scenarios by changing the objective function from QPS per Dollar to an alternate objective like the makespan metric.

## 7 EVALUATION

In this section, we demonstrate the fidelity and usefulness of Vidur across a wide range of models, hardware configurations and workloads. We perform all our evaluations on an optimized version of the vLLM codebase, with support for different scheduling policies and CUDA graphs, which eliminates unnecessary CPU overheads. Our evaluation seeks to answer the following questions:

1. Can Vidur accurately predict the end-to-end performance metrics across models of different sizes, parallelization strategies and workload traces with varying request lengths and arrival patterns (\$7.2)?
2. Can Vidur answer what-if questions related to LLM deployment challenges for a given hardware configuration (§7.3)?

### 7.1 Evaluation Setup

Implementation. As baseline, we use a fork of the opensource implementation of vLLM (Kwon et al., 2023; vLL).
We extend the base vLLM codebase to support various scheduling policies, chunked prefills (Agrawal et al., 2024), and an extensive telemetry system.

Models and Environment. We evaluate Vidur across four models: LLaMA2 7/70B (Touvron et al., 2023b), InternLM20B (Team, 2023), and Qwen-72B (Bai et al., 2023). We use Azure Standard_NC96ads_A100_v4 VMs, each equipped with 4 NVIDIA 80GB A100 GPUs, connected with pairwise NVLink. Our H100 VMs have 4 NVIDIA H100s each with 80GB memory and connected with pairwise NVLink.

Workloads. In order to emulate the real-world serving scenarios, we generate traces by using the request length characteristics from LMSys-Chat-1M, Arxiv-Summarization and Bilingual-Web-Book. LMSys-Chat-1M contains one million real-world conversations with many state-of-the-art LLMs. A conversation may contain multiple rounds of interactions between the user and chatbot. Each such interaction round is performed as a separate request to the system. This multi-round nature leads to high relative variance in the prompt lengths. Arxiv-Summarization is a collection of scientific publications and their summaries (abstracts) on arXiv.org (arx). This dataset contains large prompts and lower variance in the number of output tokens, and is representative of LLM workloads such as Microsoft M365 Copilot (mic) and Google Duet AI (goo). Bilingual-WebBook is a document-level Chinese-English parallel dataset. It consists of Chinese online novels across multiple genres and their corresponding English translations. The number of output tokens outweighs the number of prompt tokens in this dataset. This dataset also has a lower variance in number of prompt and decode tokens across requests. We restrict the

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-09.jpg?height=795&width=1550&top_left_y=215&top_left_x=255)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-09.jpg?height=328&width=461&top_left_y=232&top_left_x=276)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-09.jpg?height=325&width=355&top_left_y=233&top_left_x=733)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-09.jpg?height=326&width=340&top_left_y=234&top_left_x=1098)

(a) Median normalized end-to-end latency.

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-09.jpg?height=313&width=453&top_left_y=619&top_left_x=272)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-09.jpg?height=296&width=358&top_left_y=627&top_left_x=729)

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-09.jpg?height=271&width=338&top_left_y=626&top_left_x=1094)

Chat-1M Arxiv-4K BWB-4K
Qwen-72B (TP4)
![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-09.jpg?height=666&width=356&top_left_y=256&top_left_x=1443)

(b) P95 normalized end-to-end latency.

Figure 4. Fidelity of Vidur's execution time predictions across four models and three dynamic workload traces, using request load at $85 \%$ of the maximum serving capacity for each scenario.

total request length to 4096 tokens based on the maximum context supported by the LLaMA2 family of models. We call these shortened traces Chat-1M, Arxiv-4K and BWB$4 \mathrm{~K}$ respectively. Together, these traces represent varying workload characteristics, e.g., BWB- $4 \mathrm{~K}$ has $10 \times$ longer decodes and $2 \times$ longer prefills compared to Chat- $1 \mathrm{M}$; and a Prefill:Decode (P:D) ratio of 0.65 compared to 2.3. Further details for these workloads are present in Table 1.

### 7.2 Simulator Fidelity

In this section, we demonstrate Vidur's fidelity on end-toend request-level predictions across the four models and three workloads detailed in $\S 7.1$. We use tensor parallel for InternLM-20B (TP2), LLaMA2-70B (TP4), and Qwen-72B (TP4). We use the default vLLM scheduler for all these experiments. We first evaluate Vidur using static (offline) workloads where all requests are assumed to have arrived before the system starts. We then evaluate Vidur using a dynamic (online) workload in which we assume requests arrive based on a Poisson distribution, with the arrival rate corresponding to the throughput of the system.

Evaluation Metric. For dynamic workloads, we compare the percentage error of Vidur predictions for normalized end-to-end latency, which captures the request's end-to-end latency divided by its output length (Yu et al., 2022; Kwon et al., 2023). We augment this metric slightly for static workload, and measure only the request execution time, excluding the scheduling delay - which would otherwise dominate the latency measurement. This allows us to perform more fine-grained analysis of Vidur's capability.
Static Workloads. We present the request latency fidelity evaluation in Figure 3. We observe that Vidur predicts even the tail latency (P95) with upto $3.33 \%$ error across the four models and three datasets. Note that we observe slightly higher average error rates for the 7B model, we attribute this to the higher CPU overhead for smaller models.

Dynamic Workloads. Next we present the evaluation of Vidur on dynamic workloads. In order to perform this evaluation, first we need to determine the request arrival rate at which we should perform this comparison. If the chosen arrival rate is too low, the system would have high idle time which is not an interesting scenario. On the other hand, if the request arrival rate is too high, the system would be overloaded where scheduling delay grows rapidly. Therefore, we evaluate Vidur's fidelity near the capacity point, which represents the maximum arrival rate the system can sustain without overloading ( $\$ 6$ ).

As shown in Figure 4, Vidur achieves high fidelity ( $<5 \%$ error) in almost all scenarios with request rate set to $85 \%$ of the system capacity - which is reflective of real production scenarios. Note that, as we approach capacity point, any small deltas in prediction can lead to significant blow up of the errors. This is because at capacity, the system is at a tipping point - where even slight increase in the arrival rate or request processing time leads to a sharp increase in the request latency due to uncontrolled queue delays. If either the actual or simulated system runs into overload condition, the latency numbers become hard to reconcile due to large scheduling delay. However, production systems are provisioned with a buffer so that they don't tip over the critical
point due to sudden bursts. Since Vidur achieves high fidelity even at high arrival rates of up to $85 \%$ of capacity making it valuable in QPS range of importance. We provide additional results at different arrival rates in Appendix A.

### 7.3 What-if Analysis

We leverage Vidur-Search for an extensive what-if analysis to understand how the performance of a configuration changes with the workload, and how the cost of serving is impacted by Service Level Objective (SLO) requirements.

Inputs. We find the optimal deployment configuration (one that maximizes QPS per dollar) for four models on three (dynamic) workloads described in $\S 7.1$. We allow choosing between the GPU SKUs of A100 and H100. The maximum number of GPUs available across replicas is set to 16 .

SLOs. We put the following SLO constraints on the latency metrics: TTFT P90 $<2$ s and TBT P99 $<200 \mathrm{~ms}$. We use a more relaxed constraint of P90 for TTFT since it is a one time delay experienced by the user, as opposed to TBT which is recurrent for each output token.

Deployment Configurations. We experiment with TP and PP dimensions of 1,2 and 4 for each, with three iterationlevel schedulers vLLM, Orca+ and Sarathi-Serve that dynamically allocate memory for $K V$-Cache using paged attention. vLLM is a throughput-oriented scheduler that maximizes batch size by eagerly scheduling prefills while pausing on-going decodes. Orca+ is Orca (Yu et al., 2022) implemented over vLLM's paged attention. Sarathi-Serve creates hybrid batches with partial prefills to avoid pausing decodes while keeping GPU utilization high. We try these schedulers with batch size $32,64,128,256$ and 512. Note that the batch size gets divided by number of microbatches with PP. vLLM and Orca+ have a limit of maximum 4096 tokens per iteration while Sarathi-Serve has max $512,1 \mathrm{~K}$ and $2 \mathrm{~K}$ tokens per iteration (also known as chunk size).

Figure 1a shows the optimal configuration for the three models for each of the workloads, and Figure 6 shows the QPS per dollar for the optimal configuration. We summarize the key takeaways below.

First, the change in workload can drastically change the optimal configuration. For example, for the LLama2-70B model, the optimal configuration for LMSys-Chat-1M uses batch size of 256 , while for BWB it is 64 . This is a consequence of the high $K V$-Cache load in BWB workload due to large decode sequences. Even the optimal GPU SKU changes from $\mathrm{H} 100$ for Chat-1M to A100 for BWB.

Second, even models with similar sizes can have very different performance characteristics due to variation in architectural details. For instance, LLaMA2-70B uses Group Query
Attention (GQA), where as Qwen-72B employs Multi Head Attention (MHA) - which translates to $8 \times$ higher $K V$-Cache load. As a result, Qwen-72B is almost $2 \times$ more costly to serve and requires a different deployment configuration.

Finally, from Figure 6 it is clear that the capacity per dollar follows the expected trend. For example, larger models have lower capacity compared to smaller models. Also, Chat-1M has the least cost due to fewer prefill and decode tokens, while BWB has the highest cost due to larger number of tokens, especially decode tokens which are more expensive to compute compared to prefill. This complete exploration costs only 125 US dollars in simulation as opposed to actual execution which would have required 1.14 million dollars. We provide a detailed cost comparison in Table 2.

Configuration Stability. Figure 1b shows the overhead factor of using the optimal configuration for one workload, to serve a different workload on the LLaMA2-70B model. As shown, such a misconfiguration can result in a very high overhead, e.g., running LMSys-Chat-1M workload with the optimal configuration of Arxiv-Summarization$4 \mathrm{~K}$ workload results in a $2 \times$ overhead! This shows that even for the same model, the cost of using a homogeneous deployment configuration can result in huge overheads, as the optimal configuration for one workload can be far from optimal for another workload.

Pareto Frontier Analysis. We next analyze the Pareto frontier produced by Vidur for LLaMA2-70B-LMSys-Chat-1M and Qwen-72B-Bilingual-Web-Book-4K workloads. Figure 5 shows the best QPS per dollar for different configurations and the corresponding TTFT-P90 (left), TBT-P99 metrics (middle) along with the SLO complient regions. The figures on the right plot both the latency metrics for these configuration, and visualize the QPS per dollar via a temperature colormap. We summarize the key takeaways.

First, configurations which are optimal on one metric may not satisfy the SLO constraint on the other metric (these are the blue points on the Pareto curve). Second, small changes in latency SLOs can result in a significant cost overhead. For example, for the LLaMA2-70B-LMSys-Chat$1 \mathrm{M}$ workload, if the TBT SLO is changed from 0.12 seconds to 0.14 seconds (a difference of only $20 \mathrm{~ms}$ ), the Pareto curve point moves from approximately 0.07 to $0.13, \sim 1.85 \times$ reduction in cost!

## 8 RELATEd WORK

Prior techniques leverage the predictability of DNN training iterations (Sivathanu et al., 2019; Xiao et al., 2018) to model the performance of the entire job. For example, Habitat (Yu et al., 2021) models the performance of a training job on different types of GPUs based on the runtime profile collected

Best Config: Pipeline Parallel Dim: 2, Tensor Parallel Dim: 2, Scheduler: Sarathi-Serve, Sarathi Chunk Size: 512, Batch Size: 256, SKU: H100
![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-11.jpg?height=860&width=1526&top_left_y=258&top_left_x=272)

(a) LLaMA2-70B- LMSys-Chat-1M

Best Config: Pipeline Parallel Dim: 1, Tensor Parallel Dim: 4, Scheduler: Sarathi-Serve, Sarathi Chunk Size: 512 , Batch Size: 128, SKU: H100 QPS per Dollar: 0.03
![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-11.jpg?height=324&width=1504&top_left_y=714&top_left_x=278)

(b) Qwen-72B- Arxiv-4K

Figure 5. Capacity per dollar for different deployment configurations vs corresponding TTFT-P90 (left) and TBT-P99 (middle). Also show is the Pareto curve for these configurations. Shaded area corresponds to region where the corresponding SLO is satisfied. (right) Both latency metrics for these configuration, with capacity per dollar visualized via a temperature colormap. In the left and middle plots, green points correspond to configurations which satisfy SLOs for both metrics. Note that blue points on a Pareto curve show that, even Pareto curve points for one metric may not satisfy SLO for the other metric.

![](https://cdn.mathpix.com/cropped/2024_06_04_04592913db50f84ab7adg-11.jpg?height=252&width=767&top_left_y=1392&top_left_x=186)

Figure 6. QPS per dollar for best configurations using P90 TTFT and P99 TBT SLOs of 2s and 200ms respectively.

of a few training iterations on a given GPU. In doing so, Habitat applies the roofline model (Williams et al., 2009) to estimate the performance of individual operators based on the compute and memory requirements of the operator along with the compute and memory bandwidth of a GPU. Daydream (Zhu et al., 2020) proposes a different approach focused on modeling the effect of various system optimizations on training performance across various deployment scenarios. Daydream can help answer questions like: what is the main performance bottleneck in my training job (e.g., memory or network bandwidth), how will optimizations like kernel-fusion, quantization or gradient compression help improve performance etc. To accurately model the effect of such optimizations, Daydream first constructs a computation graph of a training job and then applies optimizations via graph transformations (e.g., kernel-fusion can be applied by substituting individual kernel nodes with a single node that represents the fused kernels in the computation graph). Proteus (Duan et al., 2023) further enables simulating various parallelization strategies to identify the best partitioning and scheduling strategy for a given training job. It does so by first modeling a parallelization strategy with a unified representation called Strategy Tree and then compiling it into a distributed execution graph. In another approach (Lin et al., 2022), the authors propose a criticalpath based strategy to predict the per-batch training time of deep learning recommendation models. Different from these training-based simulators, Vidur is the first simulator that accounts for the specific properties of LLM inference.

## 9 CONClUSION

LLM inference efficiency depends on a large number of configuration knobs such as the type or degree of parallelism, scheduling strategy, GPU SKUs. It is impractical to run all possible configurations on actual hardware. In this paper, we present Vidur: a high fidelity and easily extensible simulator for LLM inference, along with a benchmark and search suite. Vidur answers deployment related what-if questions that identify efficient deployment strategies for production environments and helps in evaluating the efficacy of various systems optimizations at nominal cost.

## REFERENCES

arxiv.org e-print archive. https://arxiv.org/.

Cupti: Cuda toolkit documentation. https:// docs.nvidia.com/cuda/cupti/index.html.

Faster Transformer. https://github.com/NVIDIA/ FasterTransformer.

Google duet ai. https://workspace.google.com/ solutions/ai/.

Microsoft copilot. https://www.microsoft.com/ en-us/microsoft-copilot.

vllm: Easy, fast, and cheap llm serving for everyone. https://github.com/vllm-project/vllm.

LightLLM: A python-based large language model inference and serving framework. https://github.com/ ModelTC/lightllm, 2023.

Agrawal, A., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., and Ramjee, R. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills, 2023.

Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., and Ramjee, R. Taming throughput-latency tradeoff in $11 \mathrm{~m}$ inference with sarathiserve. 2024.

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., Hui, B., Ji, L., Li, M., Lin, J., Lin, R., Liu, D., Liu, G., Lu, C., Lu, K., Ma, J., Men, R., Ren, X., Ren, X., Tan, C., Tan, S., Tu, J., Wang, P., Wang, S., Wang, W., Wu, S., Xu, B., Xu, J., Yang, A., Yang, H., Yang, J., Yang, S., Yao, Y., Yu, B., Yuan, H., Yuan, Z., Zhang, J., Zhang, X., Zhang, Y., Zhang, Z., Zhou, C., Zhou, J., Zhou, X., and Zhu, T. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877-1901, 2020.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., Nori, H., Palangi, H., Ribeiro, M. T., and Zhang, Y. Sparks of artificial general intelligence: Early experiments with gpt-4, 2023.

Chetlur, S., Woolley, C., Vandermersch, P., Cohen, J., Tran, J., Catanzaro, B., and Shelhamer, E. cudnn: Efficient primitives for deep learning, 2014.
Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., and Goharian, N. A discourse-aware attention model for abstractive summarization of long documents. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pp. 615-621, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-2097. URL https://aclanthology.org/N18-2097.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. Flashattention: Fast and memory-efficient exact attention with io-awareness, 2022.

Dao, T., Haziza, D., Massa, F., and Sizov, G. Flash-decoding for long-context inference, 2023.

Duan, J., Li, X., Xu, P., Zhang, X., Yan, S., Liang, Y., and Lin, D. Proteus: Simulating the performance of distributed DNN training. CoRR, abs/2306.02267, 2023. doi: 10.48550/arXiv.2306.02267. URL https: //doi.org/10.48550/arXiv. 2306.02267.

Hooper, C., Kim, S., Mohammadzadeh, H., Genc, H., Keutzer, K., Gholami, A., and Shao, S. Speed: Speculative pipelined execution for efficient decoding, 2023.

Jiang, Y. E., Liu, T., Ma, S., Zhang, D., Cotterell, R., and Sachan, M. Discourse centric evaluation of machine translation with a densely annotated parallel corpus. In Proceedings of the 2023 Conference of the Association for Computational Linguistics: Human Language Technologies, pp. 1550-1565, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.main.111. URL https:// aclanthology.org/2023.acl-main.111.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Flinn, J., Seltzer, M. I., Druschel, P., Kaufmann, A., and Mace, J. (eds.), Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pp. 611-626. ACM, 2023. doi: 10.1145/3600006.3613165. URL https://doi.org/ $10.1145 / 3600006.3613165$.

Li, Y., Bubeck, S., Eldan, R., Giorno, A. D., Gunasekar, S., and Lee, Y. T. Textbooks are all you need ii: phi-1.5 technical report. September 2023. URL https: //www.microsoft.com/en-us/research/ publication/textbooks-are-all-youneed-ii-phi-1-5-technical-report/.

Li, Z., Zhuang, S., Guo, S., Zhuo, D., Zhang, H., Song, D., and Stoica, I. Terapipe: Token-level pipeline parallelism for training large-scale language models, 2021.

Lin, Z., Feng, L., Ardestani, E. K., Lee, J., Lundell, J., Kim, C., Kejariwal, A., and Owens, J. D. Building a performance model for deep learning recommendation model training on gpus. In 29th IEEE International Conference on High Performance Computing, Data, and Analytics, HiPC 2022, Bengaluru, India, December 18-21, 2022, pp. 48-58. IEEE, 2022. doi: 10.1109/HiPC56025.2022.00019. URL https: //doi.org/10.1109/HiPC56025.2022.00019.

NVIDIA Corporation. CUBLAS library. https:// docs.nvidia.com/cuda/cu.blas/index.html, a.

NVIDIA Corporation. Matrix multiplication background user's guide. https: / /docs.nvidia.com/deeplearning/ performance/dl-performance-matrixmultiplication/index.html, b.

Patel, D. and Ahmed, A. The inference cost of search disruption - large language model cost analysis, 2023.

Patel, P., Choukse, E., Zhang, C., Goiri, Í., Shah, A., Maleki, S., and Bianchini, R. Splitwise: Efficient generative $1 \mathrm{~lm}$ inference using phase splitting. arXiv preprint arXiv:2311.18677, 2023.

Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Levskaya, A., Heek, J., Xiao, K., Agrawal, S., and Dean, J. Efficiently scaling transformer inference, 2022.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B. Megatron-lm: Training multi-billion parameter language models using gpu model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Sivathanu, M., Chugh, T., Singapuram, S. S., and Zhou, L. Astra: Exploiting predictability to optimize deep learning. In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS '19, pp. 909-923, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450362405. doi: 10.1145/3297858.3304072. URL https://doi.org/ $10.1145 / 3297858.3304072$.

Team, I. Internlm: A multilingual language model with progressively enhanced capabilities, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. Llama: Open and efficient foundation language models, 2023a.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2: Open foundation and fine-tuned chat models, 2023b.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/ paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aaPaper.pdf.

Williams, S., Waterman, A., and Patterson, D. Roofline: An insightful visual performance model for multicore architectures. Commun. ACM, 52(4):65-76, apr 2009. ISSN 0001-0782. doi: 10.1145/1498765.1498785. URL https: / /doi.org/10.1145/1498765.1498785.

Xiao, W., Bhardwaj, R., Ramjee, R., Sivathanu, M., Kwatra, N., Han, Z., Patel, P., Peng, X., Zhao, H., Zhang, Q., Yang, F., and Zhou, L. Gandiva: Introspective cluster scheduling for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), pp. 595-610, Carlsbad, CA, October 2018. USENIX Association. ISBN 9781-939133-08-3. URL https://www.usenix.org/ conference/osdi18/presentation/xiao.

Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.-G. Orca: A distributed serving system for Transformer-Based generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pp. 521-538, Carlsbad, CA, July 2022. USENIX Association. ISBN 9781-939133-28-1. URL https://www.usenix.org/ conference/osdi22/presentation/yu.

Yu, G. X., Gao, Y., Golikov, P., and Pekhimenko, G. Habitat: A runtime-based computational per-
formance predictor for deep neural network training. In Calciu, I. and Kuenning, G. (eds.), 2021 USENIX Annual Technical Conference, USENIX ATC 2021, July 14-16, 2021, pp. 503-521. USENIX Association, 2021. URL https://www.usenix.org/ conference/atc21/presentation/yu.

Zheng, L., Chiang, W.-L., Sheng, Y., Li, T., Zhuang, S., Wu, Z., Zhuang, Y., Li, Z., Lin, Z., Xing, E. P., Gonzalez, J. E., Stoica, I., and Zhang, H. Lmsys-chat-1m: A large-scale real-world llm conversation dataset, 2023.

Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., and Zhang, H. Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving. arXiv preprint arXiv:2401.09670, 2024.

Zhu, H., Phanishayee, A., and Pekhimenko, G. Daydream: Accurately estimating the efficacy of optimizations for DNN training. In Gavrilovska, A. and Zadok, E. (eds.), 2020 USENIX Annual Technical Conference, USENIX ATC 2020, July 15-17, 2020, pp. 337-352. USENIX Association, 2020. URL https://www.usenix.org/conference/ atc20/presentation/zhu-hongyu.
</end of paper 1>


<paper 2>
# Aladdin: Joint Placement and Scaling for SLO-Aware LLM Serving 

Chengyi Nie<br>Stony Brook University

Rodrigo Fonseca<br>Azure Research - Systems

Zhenhua Liu<br>Stony Brook University


#### Abstract

The demand for large language model (LLM) inference is gradually dominating the artificial intelligence workloads. Therefore, there is an urgent need for cost-efficient inference serving. Existing work focuses on single-worker optimization and lacks consideration of cluster-level management for both inference queries and computing resources. However, placing requests and managing resources without considering the query features easily causes SLO violations or resource underutilization. Providers are forced to allocate extra computing resources to guarantee user experience, leading to additional serving costs. In this paper we introduce $\mathrm{Al}$ addin, a scheduler that co-adaptively places queries and scales computing resources with SLO awareness. For a stream of inference queries, Aladdin first predicts minimal computing resources and the corresponding serving workers' configuration required to fulfill the SLOs for all queries. Then, it places the queries to each serving worker according to the prefill and decode latency models of batched LLM inference to maximize each worker's utilization. Results show that Aladdin reduces the serving cost of a single model by up to $71 \%$ for the same SLO level compared with the baselines, which can be millions of dollars per year.


## 1 Introduction

Recently, the applications of Large Language Models (LLMs) are skyrocketing [19], which greatly changes human life and work styles. The demand for LLM inference has increased significantly as more and more LLM applications become integrated into human work and life. Unlike normal Deep Neural Network inference [9], which requires a small amount of GPU resources, current transformer-based LLMs consist of billions of parameters. This makes LLM inference highly dependent on expensive GPU resources, specifically on GPU memory and computing power. The current state of LLM applications has led to a shortage of GPU resources in both public and private clouds [16]. With this common sense, effi- ciently managing and scaling the GPUs for LLM inference becomes a vital problem.

To improve the efficiency of LLM inference, Previous work [31] considers scheduling the requests with similar predicted output lengths to one batch for efficient batch inference, recent work $[2,12,29]$ focus on efficient dynamic batching for LLM inference to address the problem that requests in one batch have different output lengths. FlexGen [25] improves the LLM inference by aggregating CPU and GPU resources. Splitwise, etc $[10,20,32]$ separate the prompt processing stage and token generation stage into different instances to optimize the throughput and goodput. They adapt naive algorithms like Join the Shortest Queue (JSQ) to place requests for workers. Previous work [10] adapts a power-of-two algorithm for the request placement according to the predicted output length. However, they all focus on improving the LLM inference throughput, and some improve the Service Level Objectives (SLO) attainment as a side effect. Previous work $[6,8,23,30]$ investigated the SLO-aware DNN serving. However, the workload for those work is highly predictable. The early work on workload placement and resource management $[7,11]$ has a deep dive into cluster-level management. However, they focus on the traditional workload, which is distinct from the characteristics of LLM inference jobs. To the best of our knowledge, there is no previous work that guarantees the SLOs for all LLM queries as well as improving the request placement and worker scaling for optimized inference service cost.

In continuous batching inference, the $\mathrm{KV}$ cache usage of each request increases while decoding and becomes zero when the token generation is finished. The peak of $\mathrm{KV}$ cache usage of each request is right before the end of decoding. If all requests in a batch finish decoding simultaneously, the KV cache can easily overflow. Good request placement has to prevent this situation. When we implement continuous batching, a feature of the decoding phase becomes apparent: the decoding latency increases as more tokens are generated and stored in the KV cache, even with the same batch sizes. Simply constraining the batch size for the decoding phase can also result in violations of the decoding SLO. However,
current solutions lack awareness of those features.

The management and scaling of workers also significantly affect the cost of LLM inference. A worker serves as the smallest unit for inference. The demand for LLM inference varies throughout the day. For example, in the daytime, the demand is higher, necessitating more workers to meet the inference SLOs. Conversely, the demand decreases at nighttime, allowing for a reduction in the number of workers to save on inference costs. Regarding the cluster configuration, we aim to address the following question: What is the minimum number of GPUs required to serve LLM queries while meeting all SLOs? This involves considering two decision variables: the number of GPUs per worker and the total number of workers. The current LLM inference system [20] configures one inference worker with all GPUs on a machine. However, this static configuration is suboptimal for most models. DistServe [32] considered the Goodput of each GPU that optimized each worker's configuration from the computing latency perspective. However, it does not consider the key-value (KV) cache constraints of worker configuration or harness the features of arrival queries for workload allocation.

Based on the insights and limitations of the literature, we propose Aladdin, a co-adaptive scheduler for request placement and resource scaling. As shown in Figure 1, when LLM inference requests arrive, Aladdin first predicts minimal computing resources by learning the optimal configuration of serving workers based on the historical input-output length distributions and the request arriving rate. Secondly, Based on the requests' input and predicted output length, as well as the learned batching performance models, we formulate the request placement to an online multi-dimensional bin packing problem. Lastly, We monitor the ongoing requests of each worker and adjust the placement of new arrivals to reduce the impact of output length prediction errors. Aladdin supports the default setting vLLM [12] that does the prefill and decode in the same worker, as well as the decoupled prefill and decode setting like $[10,20,32]$.

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-02.jpg?height=485&width=596&top_left_y=1839&top_left_x=301)

Figure 1: The overall architecture of co-adaptive scheduling

Overall, the main contributions of our paper are:

- We conduct an empirical study of the dynamic batching performance of prefill-decoding LLM inference and deduce the accurate performance prediction model of LLM serving.
- We design a near-optimal online algorithm and a novel scheduler, Aladdin, to co-adaptively place the queries and manage computing resources to fulfill all requests' SLOs using minimal GPUs.
- We conducted a comprehensive evaluation of Aladdin, including the validation of our LLM inference performance models on the A100 and V100 testbeds to establish its generality. We evaluated Aladdin's end-toend performance with the real-world workload, which arrived as a stream on GPU servers. Additionally, we conducted a large-scale simulation for the high-demand LLM serving scenario.


## 2 Background and Motivation

### 2.1 Batch Processing of LLM Requests

The demand for large language model (LLM) serving has experienced exponential growth, making the efficient serving of LLM requests a critical challenge. LLM serving places significant demands on GPU computing power and memory, which can be prohibitively expensive. Previous work, such as Orca [29] and vLLM [12], have introduced dynamic continuous batching techniques for transformer-based generative models to optimize GPU utilization.

LLM generates responses iteratively, producing one token at a time and using it as input for the next iteration. Each request generates one token after every iteration in a batch of LLM requests. Importantly, these requests may have varying output lengths, necessitating different numbers of iterations to complete. Traditional request-level batching methods pose a disadvantage. Requests within the same batch must wait until all requests are finished before results are returned. In contrast, continuous batching employs iteration-level scheduling, submitting an iteration calculation to the execution engine with each token generation. This approach prevents early-finish requests from waiting for the completion of other requests, improving GPU utilization.

### 2.2 LLM Inference SLOs

In contrast to other DNN inference workloads [8] that have well-defined latency targets, LLM inference is a two-stage iterative process. The first stage involves the generation of the initial token, which processes all prefilled tokens, while the second stage is the decode stage, where tokens are generated iteratively one by one. LLM inference latency depends on the output length. Although the time for generating the first token increases with the number of prefilled tokens [2], it remains
predictable based on the length of the prefilled tokens. Additionally, the first token generation is a single-round inference process without iteration, so we have set a predetermined response deadline for time to the first token (TTFT).

For the decoding process, previous work [20] adopts the time between tokens (TBT) metric, constraining the latency between every token smaller than the target. However, the TBT metric is an over-strict metric with less flexibility, and it does not directly affect the user's quality of experience. We introduce the quality of experience SLO using the average token generation time (ATGT) metric $A T G T=\frac{t_{\text {decode }}}{l_{\text {out }}-1}$, where $t_{\text {decode }}$ is the decode time of a request and $l_{\text {out }}-1$ is the output length of the decode phase. This metric reflects the average time spent generating each token during the decode stage. For example, the average reading speed for individuals is approximately four words per second [4]. To ensure the delivery of quality service, the average token generation time for each request must not exceed 0.2 seconds.

### 2.3 Output Length Prediction

The input and output lengths of requests have a huge impact on the decision of the inference requests and worker configuration. However, when we make the request placement decisions, we only have the information for the input length of each request. There are some techniques to predict the output length of each request. Previous work [10,22,31] proposed the response length perception that harnesses the output length prediction before the execution ability of LLMs. They use historical data to fine-tune the LLM. However, there are drawbacks to this methodology. Firstly, the overhead of using a LLM to predict the output length is non-negligible because the output length prediction process is another inference. Although previous work [10] uses a smaller model to predict the output length for a larger LLM, the prediction overhead is still significant. And the prediction of response length perception is out of control. From our experiment result, the response length predicted by the fine-tuned models is biased.

Figure 2 presents the CDF of output length given the corresponding prompt length in different ranges. Although the output length prediction error is inevitable in our request placement, the prediction without bias can partially cancel the prediction error when we put requests in a batch. Hence, we use the estimated output length of each input length in the historical data as the predicted output length. This is the most naive output length predictor. Although the prediction error may be high, this prediction method has a low overhead and is non-biased. In Section 4.3, we address the prediction error by designing a novel re-balancing algorithm. Note that the output length prediction is not the main contribution of this paper. If there are accurate, non-biased, and low overhead output length predictors in the future, the performance of Aladdin could be further improved.
![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-03.jpg?height=454&width=726&top_left_y=256&top_left_x=1160)

Figure 2: CDF of output length for different prompt Lengths from ShareGPT and llama2-13b-chat-hf generated output.

### 2.4 Challenges and Opportunities

There are challenges to improving the request placement and worker scaling.

Challenge 1: Heterogeneous phases of LLM inference. The transformer-based LLM inference consists of prefilling and decoding stages. The prefill stage is the first iteration of an inference request that processes all prompt tokens; it has more computing demand than the decoding process. The decoding process is a memory-intensive stage compared with the prefill stage because of the KV cache. These distinct features result in different performance models of prefilling and decoding processes for each request. Given the requests with various input and output lengths, accurately predicting the iteration time of batched prefill and decode is challenging.

Challenge 2: Worker performance prediction. The inference workload varies over time with high uncertainty. Meanwhile, worker configuration and the number of workers directly affect the cost of inference. Considering the request arrival pattern, we must take into account the worker's computing latency, $\mathrm{KV}$ cache capacity, and communication overhead. The search space for configurations is too large to be explored by a naive enumeration approach. Accurately predicting optimal configurations poses significant challenges.

Challenge 3: Handle the error of output length prediction. The output length prediction error is inevitable. Therefore, reducing the impact of prediction errors on output length is crucial for enhancing performance when assigning tasks to workers. Systems need to effectively react when the prediction error is detected.

We tackle the request placement problem by transforming it into a multi-dimensional bin packing problem. As the LLM inference process is predictable, we develop a dynamic batching inference performance model. We consider the arrival pattern of queries with the output length prediction error awareness. Since the first token response time, average token generation time, and the KV cache demand are predictable, they facilitate the design of the scheduling algorithm.

With a thorough analysis of the computing and communi-
cation overhead of tensor parallelism and batching execution, we demonstrate the predictability of the inference throughput and the latency at the iteration level. In Aladdin's design, we predict the most resource-efficient worker configuration according to the performance of GPUs with their interconnection, LLM model size, and SLOs. With this optimal worker configuration, we can reach the highest SLO attainment rate with the same GPU resource. Furthermore, to achieve additional cost savings, we dynamically adjust the number of workers based on trends in arrival rates and query features.

## 3 Continuous Batching Performance Modeling

### 3.1 KV Cache Usage

In LLM inference, The transformer uses the given prompt (context) as the initial input and generates additional tokens one by one. During the inference process, the transformer performs self-attention, which requires the key-value (KV) vectors for each token (prompt and generated tokens) in the current sequence. These vectors are stored in the GPU as two matrices (key matrix and value matrix) during inference, often called the $\mathrm{KV}$ cache. At the beginning of an inference, the $\mathrm{KV}$ cache stores the key and value matrices of the prompt tokens. During response generation, the $\mathrm{KV}$ vectors associated with that token are appended to the $\mathrm{KV}$ cache matrices with each token generated. This dynamic expansion leads to a linear relationship between the KV cache's usage and the current sequence size. This linear relationship signifies that the $\mathrm{KV}$ cache's memory footprint increases proportionally with the sequence length. So the KV cache usage of a request

$$
\begin{equation*}
k v=h\left(l_{\text {in }}+l_{\text {out }}\right)+j \tag{1}
\end{equation*}
$$

where $h$ and $j$ are learnable coefficients, and $r$ is the output tokens generated so far.

### 3.2 Iteration Time

Iteration-level batching poses unique challenges. Not all requests can be batched together at any iteration due to varying input shapes. Orca [29] addresses this by proposing selective batching. However, operators like Attention require inputs with identical shapes, leading to separate calculations using cuBLAS [17] routines for batch matrix multiplication. The separate multiplications for each request result in a linear scaling of iteration time to the batch size. In default settings like vLLM [12] or split-phase inference, one batch can only contain prefill or decode. Since the query in the attention mechanism of the prefill process is a matrix that includes all input tokens, the query of the decode process is a vector of the last generated token. The iteration latency model of the prefill and decode batch is different.

Prefill iteration time. Since prompt processing is a computing-bottleneck process, a single request with a reasonable input length can effectively saturate the worker's computing power, which means the batching effect has limited improvement to the throughput in the prefill process. Our preliminary results indicate that the iteration time of the prefill batch is not affected by the batch size and is linear with the total input length of all batched requests. The iteration time:

$$
\begin{equation*}
t_{\text {pre }}=k_{1} \sum l_{\text {in }}+c_{1} \tag{2}
\end{equation*}
$$

where the $\sum l_{i} n$ is the total input length of all requests in the prefill batch, $k_{1}$ and $c_{1}$ are the learnable coefficients.

Decode iteration time. However, the token generation process has low compute utilization since each query only generates one token in an iteration. With a fixed batch size, the iteration time linearly increases as the average context length (the input length of the request and the tokens generated so far) increases. Similarly, with the same average context length, the iteration time increases linearly with the batch size. According to the experiment, the iteration time with a batch size of one (i.e., single request inference without batching) remains nearly constant. With this information, when we haven't reached the $\mathrm{KV}$ cache limit, the iteration time $t_{d}$ is:

$$
\begin{equation*}
t_{d}=\left(k_{2} l_{\text {ave }}+c_{2}\right) b+c_{3}, b>1 \tag{3}
\end{equation*}
$$

where $b$ is the batch size, $l_{\text {ave }}$ is the average context length among all requests. $k$ and $c$ are learnable coefficients. In the scheduling algorithm design, given the ATGT SLO $T_{d e c}$, the total input length is limited by a function of batch size $b$ :

$$
\begin{equation*}
l_{d} \leq \frac{1}{k_{2}}\left(-c_{2} b+T_{d e c}-c_{3}\right), b>1 \tag{4}
\end{equation*}
$$

Note that all coefficients in Eq. 4 are positive according to the batch inference scaling. And $T_{\text {dec }}$ must be greater than $c_{3}$ because the decoding latency SLO we choose must be greater than the individual request decoding latency without batching. From Eq. 4, we deduce that with a larger batch size, the maximum total input length limit of all requests within the batch decreases.

## 4 Co-Adaptive Scheduling

When requests arrive at the scheduler, our task is to determine how to use the minimum number of GPUs to serve both newly submitted and ongoing requests while ensuring compliance with the SLO requirements. This overarching objective can be deconstructed into several critical components:

- We need to determine the minimal GPU number required to serve the queries that fulfill the SLO requirements.
- Find the most efficient configuration of these GPUs, such as the number of workers and the number of GPUs configured with each worker.
- Decide how to place the requests to each worker in a manner that optimizes the utilization of each worker.

It's important to note that these three components are interconnected. When one decision is made, the other two are simultaneously determined. For example, when we establish the total number of GPUs, this decision implicitly dictates the optimized placement of GPUs and models on each worker, as well as the optimization of request assignments to each worker. Conversely, if we can devise a more effective strategy for worker configuration or request assignment that enhances resource utilization, we can reduce the total resource requirements for a more cost-efficient service. Firstly, Let's look into the optimal single-worker configuration because the optimal configuration for each worker is orthogonal to the request scheduling and worker number determination.

### 4.1 Worker Configuration

Model parallelism is a widely used technique for LLM training and inference. There are two types of model parallelism: tensor parallelism, which splits the tensors across all GPUs, and pipeline parallelism, which splits the layers across all GPUs. In general, people use tensor parallelism inside a node where GPUs are connected by high-bandwidth networks like NVlink, and pipeline parallelism is used when there is slower cross-node communication. However, the performance modeling is challenging for pipeline parallelism because of bubbling. In this paper, we consider the tensor parallelism distributed inference. The optimal worker configuration is achieved when we achieve the optimal per-GPU throughput. Therefore, the throughput with the given number of GPUs is optimized. With the different ranks of tensor parallelism, the computing, communication, and KV cache capacity all impact the throughput. In the default vLLM [12] setting, the prefill and decode processes are served with the same worker. The decode process dominates the inference process because tokens are generated in the decode process one by one while the prefill process only has one iteration. We have to predict the parallelism strategy with the most per-GPU throughput for decode phase.

In tensor parallelism, each GPU first computes the split tensor locally, then combines the tensor across all GPUs using All-reduce. The split tensor size is inverse to the GPU number, so the computing time is an inverse function of the number of GPUs:

$$
\begin{equation*}
t_{\text {compute }}=\frac{k_{4}}{N_{g}}+c_{4} \tag{5}
\end{equation*}
$$

where $N_{g}$ is the number of GPUs per worker, $k_{4}$ and $c_{4}$ are learnable parameters. Tensor parallelism adopts All-reduce for the inter-GPU communication. The communication overhead for All-reduce, relative to the number of GPUs, is $\left(N_{g}-1\right) / N_{g}$. When the number of GPUs is large, the communication overhead of All-reduce is nearly constant. However, for modern GPU servers like DGX A100 and H100, the number of GPUs on each server is less than or equal to 8 . So the

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-05.jpg?height=572&width=721&top_left_y=256&top_left_x=1147)

Figure 3: An example illustrates the sub-optimal of JSQ for request placement.

difference in the communication overhead is non-negligible. Since the communication between GPUs is intra-node communication with a high-speed network. The communication straggler is not significant. Therefore we use the communication overhead to predict the communication delay scaling to the GPU number. The KV cache capacity can be calculated by the sum of the GPUs' memory on each worker minus the model size, $M=N_{g} m_{g p u}-m_{\text {model }}$. The throughput can be limited by KV cache or the SLO constraint. When KV cache is the bottleneck, the maximum throughput is achieved when the $\mathrm{KV}$ cache is full. When iteration SLO is the bottleneck, the maximum throughput is achieved when the decode iteration time is equal to ATGT SLO latency limit. The maximum per-GPU throughput of tensor parallelism rank $N$ is:

$$
\begin{equation*}
T_{\max }=\min \left\{\frac{M}{N_{g} m_{r}\left(t_{\text {compute }}+t_{\text {comm }}\right)}, \frac{B}{N_{g} T_{\text {decode }}}\right\} \tag{6}
\end{equation*}
$$

where $m_{r}$ is the average per request $\mathrm{KV}$ cache demand learned from the historical data, and $t_{\text {compute }}+t_{\text {comm }}$ is the iteration time given the batch size $\frac{M}{m_{r}}$ with $N_{g}$ GPU per worker. $T_{\text {decode }}$ is the ATGT SLO, and $B$ is the batch size corresponding to the SLO. The optimal worker configuration has $N_{g}^{o p t}$ GPUs that maximize $T_{\max }$. Note that the optimal worker configuration remains unaffected by the request arrival rate but is influenced by factors such as the model's size, context length, and the computing and memory capacity of the GPU. Since we consider homogeneous GPUs in this paper, when scheduling requests and workers to adapt to varying workloads, the configuration of each worker remains unchanged.

### 4.2 Request Placement Policies

We optimized the worker configuration to achieve maximum per-GPU throughput, and our next objective is to minimize the number of workers required for LLM service. The placement of queries to workers significantly affects efficiency of
resource utilization. Figure 3 illustrates the suboptimal of naive JSQ and reveals the optimal request placement strategy. In this example, requests need to be placed to two workers with $\mathrm{KV}$ cache capacity of 9 . Note that in this example, the requests arrive in sequence from 1 to 4 but are submitted to workers at the same time. If we adopt JSQ, two long prompt requests will be placed to the same worker, while two long output requests will be placed to another worker. Suppose a token requires $1 \mathrm{KV}$ cache capacity. The max $\mathrm{KV}$ cache demand for both workers is 10 when requests finish generation, which exceeds the KV cache capacity of 9 . Therefore, we need to move requests to the waiting queue until there is available $\mathrm{KV}$ cache. However, with the optimal request placement, a long prompt request and a long output request are placed in one worker. The max KV cache demand for each worker is 7. To aid in this decision-making process, we leverage the parameters notated in Table 1 and the following information:

- Learnable prefill time to total input tokens Eq. 2, input tokens limit to batch size when constraining the decode iteration time Eq. 4 and learnable KV cache usage to token count Eq. 1 functions for each group.
- The current KV cache usage $m=\sum k v$ and total KV cache $M$ for each worker.
- For each newly added request, we utilize the known input prefill length $l_{j}^{i n}$ and predicted output length $l_{j}^{\text {pred }}$. For ongoing requests, we take into account the current length generated $l_{j}^{\text {out }}$.

Table 1: The inputs to Aladdin and decisions Aladdin makes

| Inputs | Notation | Definition |
| :--- | :--- | :--- |
|  | $k v(t)$ | The KV cache usage to tokens function |
|  | $l_{d}(b)$ | The input length limit to batch sizes |
|  | $t_{p}(l)$ | The prefill iteration time function |
|  | $m_{i}$ | The KV cache usage of Worker $i, i \in W$ |
|  | $M$ | The KV cache capacity of each worker |
|  | $l_{j}^{\text {in }}$ | The input length of a request |
|  | $l_{j}^{\text {pred }}$ | The predicted output length of a request |
|  | $l_{j}^{\text {real }}$ | The real output length of a request |
|  | $l_{j}^{\text {out }}$ | The output tokens a request generated so far |
|  | $t_{j}^{\text {dec }}$ | The time spent for decoding phase so far |
|  | $T_{p r e}$ | The SLO of prefill latency |
| Outputs | Notation | Definition |
|  | $W$ | The total worker number |
|  | $x_{i j}$ | binary variable for request $j$ |
|  | $y_{i}$ | binary variable for Worker $i$ |

The request scheduling with the constraints can be seen as a multi-dimensional bin packing problem. We formulate it as a mixed integer programming (MIP) that schedules the new-arrived requests between the scheduling heartbeat with different input/output lengths $l_{\text {in }}$ and $l_{\text {out }}^{\text {pre }}$, and we want to minimize worker number $W$.

Let $x_{i j}$ be a binary variable that equals 1 if request $j$ is scheduled to Worker $i$, and 0 otherwise. Let $y_{i}$ be a binary variable that equals 1 if Worker $i$ is used, and 0 otherwise. Assume $I$ is the initial worker number larger than the optimal $W$. When there are ongoing requests, for an ongoing request $j$, to prevent the unnecessary migration between workers, the $x_{i j}$ is kept the same as the current decoding worker. We also need to guarantee that the new request's prompt processing time won't cause the token generation time SLO violation of the ongoing requests. The MIP problem can be formulated as follows:

$$
\begin{array}{ll}
\min & \sum_{i=1}^{I} y_{i} \\
\text { s.t. } & \sum_{i=1}^{I} x_{i j}=1, j=1,2, \ldots, J \\
& \sum_{j=1}^{J} x_{i j}\left(l_{j}^{\text {in }}+\gamma l_{j}^{\text {out }}\right) \leq \theta l^{d}\left(\sum_{j=1}^{J} x_{i j}\right), i=1,2, \ldots, I \\
& t_{p}\left(\sum_{j=1}^{J_{\text {new }}} x_{i j} l_{j}^{\text {in }}\right) \leq T_{p r e}, i=1,2, \ldots, I \\
& t_{p}\left(\sum_{j=1}^{J_{\text {new }}} x_{i j} l_{j}^{\text {in }}\right) \leq \theta \min \left(T_{d e c} l_{i j}^{\text {out }}-t_{i j}^{\text {dec }}\right), i=1, \ldots, I \\
& {\left[\sum_{j=1}^{J} \mathbf{w}_{j} x_{i j}\right] \leq M, k=1,2, \ldots, K, i=1,2, \ldots, I} \\
& x_{i j} \leq y_{i}, i=1,2, \ldots, I, j=1,2, \ldots, J \\
& x_{i j} \in\{0,1\}, i=1,2, \ldots, I, j=1,2, \ldots, J \\
& y_{i} \in\{0,1\}, i=1,2, \ldots I . \tag{h}
\end{array}
$$

The constraints are: a Each request must be scheduled to one worker. (b) According to Eq. 3, the iteration time is determined by both batch size and the total context length. Eq. 4 shows the maximum total context length of all requests in one batch given the batch sizes. This constraint ensures the ATGT SLO for the decode process. Since the iteration time increases as more tokens are generated during decoding, the coefficient $\gamma$ can be considered as a "strictness knob" that tunes the scheduling bound, $0 \leq \gamma \leq 1$. When $\gamma=0$, only the first iteration can meet the ATGT SLO. When $\gamma=1$, the last token generation time can meet the ATGT SLO. We normally set $\gamma=0.5$ to increase the worker utilization while guaranteeing the SLOs. (c) According to Eq. 2, the sum of all new requests' input is limited by the TTFT SLO. (d) Since the prefill of new requests preempts the decode for ongoing requests, the prefill time of new requests can not exceed the time that ongoing requests have saved compared with the ATGT limit. Reflecting on the limitation of the sum of new requests' input length. (e) The total KV cache demand of

```
Algorithm 1: Request scheduling heuristic
    Input: $l^{\text {in }}, l^{\text {pred }}$ of the new request $j . l_{\text {in }}, l_{\text {pred }}, l_{\text {out }}$ of
    all ongoing requests. $\mathrm{KV}$ cache capacity $M$ for each
    worker. Worker number $W$. Performance models $k v(t)$,
    $t_{\text {iter }}(b, l), t_{\text {pre }}(l)$.
    Output: Worker $i$ where job $j$ be scheduled, $x_{i j}=1$.
    Initial: workerfound $\leftarrow$ False
    Sort all bins on capacity_norm from large to small.
    for sorted bins $i=1,2, \ldots, I$ do
        initial $x_{i j} \leftarrow 0, i=1,2, \ldots, I$
        $x_{i j}=1$
        if b) and c and $d$ and e for $i$ then
            workerfound $\leftarrow$ True
            return $x_{i j}$
    if workerfound $=$ False then
        Open a new bin $(I+1)$ and add job $j$.
        workerfound $\leftarrow$ True
        return $x_{(I+1) j}=1$
```

all the requests scheduled to each worker cannot exceed the $\mathrm{KV}$ cache capacity $M . K$ is the sequence length limit of the serving model. $\mathbf{w}$ is the vector with length $K$ that shows a request's $\mathrm{KV}$ cache footprint. For example, for request $j$,

$\mathbf{w}=\left[\begin{array}{lllllll}k v\left(l_{j}^{i n}\right) & k v\left(l_{j}^{i n}+1\right) & \cdots & k v\left(l_{j}^{\text {in }}+l_{j}^{\text {pred }}\right) & 0 & \cdots & 0\end{array}\right]$,

where each element in the vector presents the KV cache demand of an iteration. The KV cache demand for the first iteration includes the KV cache for input tokens. The KV cache demand increases in the following iterations while output tokens are generated. The KV cache demand becomes zero when the request $j$ finishes. This constraint guarantees that for all scheduled iterations, the $\mathrm{KV}$ cache demand will not exceed the KV cache capacity of the worker. (f) If a worker is used, it should have at least one request scheduled. Otherwise, we don't need this worker. (g) (h) All variables are binary. Unused boxes will have $y_{i}=0$ and will not be counted in the objective function. $0<\theta<1$ in (b) (d) is another hyperparameter that adapts to the prediction error of output length. For example, when $\theta$ is small, the constraints are tighter, so requests are less likely to violate the SLOs. However, the drawback is that we need more workers for the serving.

Scheduling heuristic. The multi-dimensional bin-packing problem is NP-hard, so an efficient heuristic is needed to approach optimal scheduling. Given that requests arrive in an online pattern, we employ the best-fit algorithm for online bin packing [13]. It schedules each arrived request to the worker with the maximum load and can guarantee the satisfaction of all SLO constraints. Intuitively, this heuristic increases the utilization of each worker compared to other scheduling algorithms, such as joining the shortest queue, thereby reducing the number of workers needed.

```
Algorithm 2: Re-balancing with prediction error
Input: $x_{i j}, l_{j}^{\text {pred }}, l_{j}^{\text {out }}, l_{j}^{\text {real }}$ of $J_{\text {old }}$ ongoing requests.
    $x_{i j}, l_{j}^{\text {in }}, l_{j}^{p r e d}$ of $J_{\text {new }}$ new requests.
    Output: Updated $x_{i j}$ of new requests.
    Initial: $l_{i}^{e}=b_{i}^{e}=0, i=1,2, \ldots, I$.
    for worker $i=1,2, \ldots, I$ do
        for ongoing job $j=1,2, \ldots, J_{i}$ on worker $i$ do
            $/ *$ Check if under estimate output length $* /$
            if $l_{j}^{\text {out }}>l_{j}^{\text {pred }}$ then
                $l_{i}^{e} \leftarrow l_{i}^{e}+l_{j}^{\prime p r e d}$
                $b_{i}^{e} \leftarrow b_{i}^{e}+1$
            $/ *$ Check if over estimate output length $* /$
            if $l_{j}^{\text {real }}<l_{j}^{\text {pred }}$ then
            $l_{i}^{e} \leftarrow l_{i}^{e}+l_{j}^{\text {real }}-l_{j}^{p r e d}$
            $b_{i}^{e} \leftarrow b_{i}^{e}-1$
    Calculate the equivalent error function
    $\alpha_{i} l_{i}^{e}+\beta_{i} b_{i}^{e}+c_{1}=0$ of worker $i, i=1,2, \ldots, I$.
    according to Eq. 4.
/*Fix error by adjusting the new requests placement*/
    if new request $j$ from worker $x$ to worker $y$ then
        $b_{x}^{e} \leftarrow b_{x}^{e}-1$
        $b_{y}^{e} \leftarrow b_{y}^{e}+1$
        $l_{x}^{e} \leftarrow l_{x}^{e}-l_{j}^{p r e d}$
        $l_{y}^{e} \leftarrow l_{y}^{e}+l_{j}^{p r e d}$
    $/ *$ Minimize the sum of the shortest distance between
        each worker's error function and the origin. $* /$
    $22 \min \left(\sum \frac{\left|c_{i}\right|}{\sqrt{\alpha_{i}^{2}+\beta_{i}^{2}}}\right), i=1,2, \ldots, I$.
    Return $x_{i j}, j=1,2, \ldots, J_{\text {new }}$
```

In the multi-dimensional bin packing problem, determining the metric for each worker's load is non-trivial. Using the batch size of each worker as the metric for its load is sub-optimal because the input and output lengths of requests significantly influence each worker's load. We propose capacity_norm, which is the $\mathrm{L} 2$ norm of batch size $B$ and weighted context length $\sum\left(l_{\text {in }}+\gamma l_{\text {out }}\right)$ of all ongoing requests to rank all workers. The heuristic algorithm for scheduling an arriving request is described in Algorithm 1.

### 4.3 Addressing Prediction Errors

The output length cannot be accurately predicted before execution. If we overestimate the output length, worker utilization will be reduced. Conversely, there will be SLO violations. When an ongoing request in a batch finishes earlier than predicted, we mark this worker as overestimated. If an ongoing request's output length is underestimated, i.e., the request hasn't finished with the predicted tokens, we mark this worker

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-08.jpg?height=412&width=610&top_left_y=255&top_left_x=302)

Figure 4: Workflow of Aladdin with default continuous batching

as underestimated and predict the output length again. Before the execution of the new requests, we re-schedule new requests that have been scheduled to the over-utilized workers to the under-utilized workers. We use $l^{e}$ and $b^{e}$ as the metrics to indicate the estimation error of each worker, where $l^{e}$ is the accumulated error of output length for outstanding requests, and $b^{e}$ is the error of batch size for each worker. If Request $j$ is finished before the estimated iteration, which means we overestimate the output length, we can calculate the output length over-estimate error $l_{j}^{\text {real }}-l_{j}^{p r e d}$. If we underestimate the output length of Request $j$, we predict the output length $l_{j}^{\text {pred }}$ again using conditional average output length when $l_{j}^{\text {real }}>l_{j}^{p r e d}$ with the same input length $l_{j}^{i n}$. In the request scheduling, we use $l^{e}$ and $b^{e}$ as the indicators to balance the workload between workers and reduce the effect of output length prediction error. The calculation for $l^{e}, b^{e}$, and the re-balancing algorithm are described in Algorithm 2.

## 5 System Design

Benefiting from the predictable nature of individual and batch LLM inference, we attempt to reveal the best way to serve requests that arrive as a stream from resource management and request placement perspectives. In this section, we describe the system design of Aladdin for two variances settings: default continuous batching and split-phase inference. The default continuous batching will process the input tokens and generate output tokens in the same worker, represented by vLLM [12]. The split-phase inference refers to the inference setting that splits the prompt processing and token generation into different working instances, and each instance only processes prompt or generates output. This setting is represented by Splitwise [20] and DistServe [32].

### 5.1 System Workflow.

Default continuous batching. The Figure 4 illustrates the workflow of continuous batching inference scheduling. Firstly, users submit their LLM inference requests via the API as the

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-08.jpg?height=409&width=848&top_left_y=256&top_left_x=1099)

Figure 5: Workflow of Aladdin with split-phase inference.

first step (1). The request scheduler uses the bin packing heuristic to schedule the new requests according to their input length and the predicted output length (2). Lastly, the request scheduler continuously update the performance model according to the worker's execution traces (3).

Split-phase inference. Figure 5 illustrates the workflow of split-phase inference. Users submit requests through API (1). We schedule the prefill of new requests based on their input lengths. Since the prefill only involves one iteration, there is no queue for the prefill workers (2). Next, the decoding scheduler places the requests from prefill workers to decoding workers based on the predicted output length and a learned performance model (3). Finally, the prefill and decode schedulers continuously update the performance model according to the workers' execution traces (4).

### 5.2 Adapt to Changing Demand

In every cluster heartbeat, we can reconfigure the cluster using change point detection. In LLM inference, although users submit different queries and receive different answers, the input and output lengths of LLM inference requests for the same model exhibit a strong pattern. From the SharGPT dataset [5], we found that the input lengths of user queries follow a fixed distribution, and the output lengths of the same LLM also follow a learnable distribution. According to our experiment using Algorithm 1, when the arrival rate $r_{a}$ is larger than a lower bound $R$, the total number of required workers $N_{w}$ is linear with the request arrival rate $r_{a}$.

$$
\begin{equation*}
N_{w}=\left\lceil k_{5} r_{a}+c_{5}\right\rceil, r_{a}>R \tag{7}
\end{equation*}
$$

where $k_{5}$ and $c_{5}$ are learnable coefficients associated with the historical demand, and we round the number of workers to the smallest integer larger than the function of $r_{a}$. The reason $R$ exists is that when the arrival rate is lower, there are fewer requests arriving in the same heartbeat, which cannot represent the real distributions of the input and output length. The standard error of the mean $S E M=\frac{\sigma}{\sqrt{n}}$ is the metric for the difference between the sampled requests' input and output lengths and the total requests, where $\sigma$ is the standard

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-09.jpg?height=325&width=851&top_left_y=241&top_left_x=187)

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-09.jpg?height=233&width=414&top_left_y=260&top_left_x=194)

(a) A100 testbed

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-09.jpg?height=236&width=414&top_left_y=256&top_left_x=606)

(b) V100 testbed

Figure 6: Prefill latency

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-09.jpg?height=309&width=830&top_left_y=688&top_left_x=192)

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-09.jpg?height=226&width=394&top_left_y=692&top_left_x=196)

(a) A100 testbed

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-09.jpg?height=239&width=414&top_left_y=691&top_left_x=606)

(b) V100 testbed
Figure 7: Decode context length limitation

deviation of all requests' input and output length and $n$ is the number of requests we place during a heartbeat. The smaller $n$ is, the more error appears in the prediction of $N_{w}$.

With this model, we can predict the total number of workers required before placing all requests to each worker. However, the scheduling time requirement of inference serving is in milliseconds. In a high-demand situation, the scheduling overhead is too large to schedule the requests in the target iteration for the centralized scheduler. We design a distributed scheduler for the high-demand scenario that harnesses the pattern of input and output length of requests in Appendix A.

Note that in this paper, we focus on predicting the minimal GPU required for the varying arrival rate without considering the cold start problem and the switching cost. Since the optimization of cluster scheduling is orthogonal to the worker number prediction problem, we defer it to future work.

### 5.3 Implementation

Aladdin is specifically designed for single-model serving, eliminating any model cold start problem for each worker. We adopt vLLM [12] for dynamic batch inference to optimize the $\mathrm{KV}$ cache usage of each worker and make the KV cache usage more predictable. Aladdin's request scheduler is a scheduling layer on top of the vLLM inference engine. Users submit their requests to the Aladdin frontend through the API interface. Aladdin routes and schedules the requests to different workers through each server's API interface. Note that Aladdin is a non-blocking system; once a request is scheduled to a worker, it will start inference in the next iteration. Aladdin doesn't support request migration, which means once a request has been sent to a worker, we won't migrate it to another worker with the same duty.

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-09.jpg?height=328&width=572&top_left_y=256&top_left_x=1229)

Figure 8: KV cache prediction and the observation.

## 6 Evaluation

For the evaluation of Aladdin, our first step is to validate the accuracy of our performance modeling for continuous batching inference in Section 6.2. Next, we examine the performance improvement achieved with Aladdin with different scenarios in Section 6.3 and Section 6.4. We also provide the overhead analysis of Aladdin in Section 6.5. The primary information of our evaluation is as follows:

- Aladdin accurately predicts performance metrics with the maximum error less than $10 \%$.
- Aladdin reduces the GPU number required by up to $71 \%$ and $60 \%$ compared with vanilla vLLM [12], and splitphase inference engines [20,32]'s decode instances for the same workload.
- Although single-worker optimization techniques like chunked prefill [2] and split-phase inference [20, 32] reduce the cost for inference, the cost reduced by Aladdin is orthogonal to those techniques. Aladdin can be combined with those single-worker optimization techniques to improve the performance further.


### 6.1 Experimental Setup

Testbed setup. We test the performance of Aladdin on highend GPU servers with 4 A100 80GB GPUs connected with PCIe. Each machine has two Intel Xeon Platinum 8380 processors and 512GB RAM. To validate the generalization of Aladdin from both a computation perspective and communication perspective, we also evaluate Aladdin on the GPU servers with 4 V100 32GB GPUs connected with NVLink. Each machine has two Intel Xeon Gold 6230 processors and 128GB RAM. We also do a large-scale simulation for the high-demand request arrival situation.

Models and SLOs. Our target is to prove Aladdin reduces the cost for transformer-based LLMs. To validate that Aladdin can accurately model the performance metrics of most models. We evaluate Aladdin on Llama2 series [28] models from 7B to 70B. The model, testbed information, and SLOs are shown in Table 2. Note that the prefill latency SLOs are

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-10.jpg?height=344&width=661&top_left_y=256&top_left_x=274)

(a) The end-to-end SLO attainment rate, (left): LlaMa213b, (right): LlaMa2-70b

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-10.jpg?height=344&width=664&top_left_y=701&top_left_x=270)

(b) The end-to-end P99 ATGT, (left): LlaMa2-13b, (right): LlaMa2-70b

Figure 9: End to end experiments on A100 testbed

the approximated inference latency for the model's context length (4096 tokens) for each testbed. The selection of decode latency SLO is according to the individual request inference latency. We guarantee that the batch inference latency of each request won't exceed the individual inference latency for 1.3 times.

Workload. For the end-to-end performance evaluation in Section 6.3, we first collect the prompts from users of ShareGPT_V3_unfiltered_cleaned_split dataset [27], then submit the prompts follows a Poisson distribution. The outputs are generated by each evaluated model with a temperature of 0 and a maximum output token limit of 2048. For the largescale simulation in Section 6.4, we use the same prompts' lengths as those collected from ShareGPT [27] in Section 6.3 as the prompt lengths. Then, we predict the output length based on the output length CDF of the responses generated in Section 6.3's end-to-end evaluations for each model.

Table 2: The LLM information and testbed allocation

| Model | Testbed | Prefill <br> SLO(ms) | Decode <br> SLO(ms) |
| :---: | :---: | :---: | :---: |
| Llama2-chat 70b | A100 | 1600 | 75 |
| Llama2-chat 13b | A100, V100 | 600,800 | 30,50 |
| Llama2-chat 7b | A100, V100 | 400,800 | 15,30 |

Because there is no available trace of LLM inference that includes the arrival time of each request, we simulate the request arrival stream using a Poisson distribution. We need to vali-

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-10.jpg?height=350&width=653&top_left_y=253&top_left_x=1191)

(a) The end-to-end SLO attainment rate, (left): LlaMa27b, (right): LlaMa2-13b

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-10.jpg?height=344&width=658&top_left_y=706&top_left_x=1189)

(b) The end-to-end P99 ATGT, (left): LlaMa2-7b, (right): LlaMa2-13b

Figure 10: End to end experiments on V100 testbed

date that Aladdin improves performance in both high-demand and low-demand scenarios. To evaluate the performance of Aladdin with varying demands, we tune the average arrival rate $\lambda$ to simulate different request demands.

Metrics. Since our final target is to reduce the cost of the inference service, we use the number of GPUs required to achieve a certain SLO attainment rate as the main metric. In Section 6.4, we evaluate the total GPU number required with different request arrival rates. In Section 6.3, As the total resources are limited for the real testbed evaluation, we evaluate the performance of Aladdin with the SLO attainment rate and the P99 ATGT in different request arrival rates. The SLO is attained when both TTFT and ATGT latency meet the requirement.

Baselines. Aladdin is a cluster-level scheduler. The performance improvement achieved by Aladdin is orthogonal to the improvements achieved by single-server optimization techniques such as split-phase inference [20,32] or page attention [12]. These single-server optimization techniques use naive cluster scheduling like JSQ. Previous work [10] adopts the power-of-two scheduling for request placement. However, it is suboptimal for request placement and cannot guarantee a high SLO attainment rate. We compared Aladdin's request placement with JSQ and power-of-two algorithms with different GPUs and different serving scenarios.
![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-11.jpg?height=610&width=740&top_left_y=294&top_left_x=213)

Figure 11: Simulation of the total GPU number needed with the mixed prefill and decode setting.

### 6.2 Performance Model Validation

Request placement and worker configuration depend on accurate predictions of performance metrics. In this section, we evaluate the model's accuracy by comparing the predicted metrics to the measured actual metrics.

In Section 3, we model the latency of prefill phase and decode phase separately because the two phases have different characteristics. In the evaluation, we evaluate the prefill and decode latency separately for different input and context lengths. In our prefill latency model, the prefill time of a batch size only corresponds to the total input length of all requests in the batch, not related to the batch size. In our experiment, we test different batch sizes $1,2,4,8$ with the same total input length within a batch to validate this formulation. We only evaluated the LlaMa2-70b model on the A100 testbed because our V100 testbed could not load the 70b model (around 140GB) even with all GPUs (32GB*4). Figure 6a and Figure 6b shows the results on A100 and V100 testbeds. The maximum prefill latency prediction error is less than $4 \%$. The shaded area is the prediction interval, which represents the estimation of the range in which future observations are likely to fall. Results indicate the maximum error of the prediction interval compared with our prediction is less than 10 tokens.

In the decode latency model, the iteration time is linear with respect to both the batch size and the total context length within the batch, not related to the context length of each request in the batch. This means that regardless of whether the context length of each request in a batch is long or short, the decoding latency will be the same when the sum of the context lengths of all requests and the batch size is the same. In our experiment, for the same batch size, we use the same sum of context length but different context length distributions for all requests in a batch to validate this formulation. Results
![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-11.jpg?height=600&width=744&top_left_y=296&top_left_x=1124)

Figure 12: Simulation of the total GPU number needed for the decode phase of the split-phase inference setting

are presented in Figure 7a and Figure 7b. Similar to the prefill latency prediction, the prediction error is less than $5 \%$. For the prediction interval, the error is less than 300 tokens for all context tokens in the batch.

The KV cache usage to the context length is the most accurate metric in our performance models. According to Figure 8, the prediction error is less than $1 \%$, and the prediction interval is just aligned with the prediction model. Note that the KV cache usage is not affected by the testbed; it is only related to the model. Generally speaking, the larger the model is, the more $\mathrm{KV}$ cache is needed for the same context length. However, from Figure 8, we can see that for the same context length, Llama2-13b requires more KV cache than Llama270b. This is because Llama2 7b and 13b adopt multi-head attention, while the $70 \mathrm{~b}$ model adopts grouped-query attention [3], which shares key and value pairs within each group.

### 6.3 End-to-End Performance

We evaluate Aladdin's end-to-end performance by comparing it with baselines on our A100 and V100 testbeds. In this experiment, requests arrived on Aladdin in a stream format following Poisson distribution. We use ShareGPT [27] dataset for the conversation content. The baseline we select is the default vLLM, with all GPUs (4 GPUs) on each machine in one worker. Since the performance improvement achieved by Aladdin is gained both from request placement and optimal worker configuration, we configure vLLM with the optimal worker configuration and adopt JSQ for the request placement to do the ablation study. Table 3 reveals the best worker configuration for different models on different testbeds.

The results of A100 testbed are shown in Figure 9. For the LlaMa2-70b model, Aladdin reduces the SLO violation

Table 3: Optimal worker configuration for different models and different GPUs for ShareGPT dataset

| Model | A100 <br> (GPUs/worker) | V100 <br> (GPUs/worker) |
| :---: | :---: | :---: |
| Llama2-70b-chat-hf | 2 | N/A |
| Llama2-13b-chat-hf | 1 | 2 |
| Llama2-7b-chat-hf | 1 | 1 |

rate by up to $3.5 \mathrm{X}$ compared with the default vLLM setting. Compared with the best worker configuration with JSQ placement, Aladdin only improved the SLO attainment rate by up to $19 \%$. This is because there are totally two workers for the LlaMa2-70b model, which limits the improvement in the SLO attainment rate. However, Aladdin significantly reduces the P99 ATGT by up to $40 \%$ compared with JSQ, as shown in Figure 9b's right side. The results for the LlaMa2-13b model are distinct from the $70 \mathrm{~b}$ model. The optimal worker configuration for the $13 \mathrm{~b}$ on the A100 testbed is one GPU according to Table 3. There are four workers in total for the request placement. So Aladdin improves the SLO attainment rate by up to $51 \%$ compared with JSQ, but only has minor P99 ATGT improvement. The results of the V100 testbed are described in Figure 10. The difference is when the request arrival rate is low, the P99 ATGT of baseline default vLLM output performs the performance with optimal worker configuration. This is because when the arrival rate is low, the batch effect is not significant, and the worker with more GPUs has higher computing power than the worker with fewer GPUs. Nevertheless, in those arrival rates, both baselines and Aladdin fulfill all requests SLOs. The higher ATGT won't further improve the SLO attainment rate. Note that we don't include the P99 TTFT because vLLM [12] preempts the decode batch with the prefill batch when new requests arrive, making the ATGT more easily violate the SLO.

### 6.4 Large-Scale Simulation

We conducted a simulation for the high-demand request arrival scenario. In this simulation, we evaluated Aladdin's performance with split-phase inference and the default vLLM inference setting. To show the direct cost savings of Aladdin, we simulate the GPU number required for P100 SLO-guaranteed inference serving at the different request arrival rates.

Default Continuous Batching Inference. In Figure 11, we compared vLLM with baselines in Section 6.3. Results indicate that Aladdin reduces the LLM serving cost by up to $71 \%$ and $40 \%$ compared with the default vLLM and JSQ with Aladdin optimal workers.

Split-Phase Inference. Previous work [10, 20, 32] split the prefill phase and decode phase into different instances. Splitphase serving maintains a group of prefill workers and a group of decode workers, as shown in Figure 5. According to the

![](https://cdn.mathpix.com/cropped/2024_06_04_6d5438d3fbbb9e3dd1f0g-12.jpg?height=349&width=480&top_left_y=281&top_left_x=1256)

Figure 13: The bin packing algorithm running time with different arrival rates

results of DistServe [32], the majority of GPU resources are scheduled for the decode workers. Since the scheduling for prefill instances is trivial with known prompt lengths, we only simulate the GPU number required for the decode phase instance. The baselines are JSQ adopted by DistServe [32] and the Power-of-Two algorithm adopted by previous work [10]. Results indicate that Aladdin reduces the GPU number required for the SLO-guaranteed decode phase by up to $60 \%$ and $49 \%$ compared with JSQ and Power-of-Two algorithm.

### 6.5 Scheduling Overhead

The scheduling overhead can be a problem in high-demand scenarios. For the scheduling latency, each scheduler's scheduling latency is predictable based on the request arrival rate since the time complexity of the best-fit bin packing algorithm is $\mathrm{O}($ nlogn). Figure 13 shows the scheduling overhead in centralized scheduling. According to the results, with a request arrival rate of around 25 requests per second as we adopted in Section 6.4. The scheduling overhead is less than $50 \mathrm{~ms}$, which is acceptable. However, if the arrival rate is very high or the scheduling latency limit is very strict, we can follow Appendix A to adopt the distributed grouped scheduling.

## 7 Related Work

LLM Inference Performance Modeling. To improve the LLM inference metrics, the first step is to model the performance of the LLM inference process. Previous work [15] estimates the prefill and decode runtime based on the floating point operations during the inference. However, they focus on the inference for a single query. The performance metrics prediction of [15] is only based on the model, which lacks the consideration of the hardware adaption. DistServe [32] models the latency of prefill and decode phases for a batch of queries, also according to the model architecture, e.g., general matrix multiply operations. However, they model the inference latency mainly for the inference worker configuration instead of request placement. As far as we know, there is no existing research that studies the inference latency for a batch
of requests with varying input and output lengths.

Inference Serving Quality. The unique prefill-decode phase and the varying output length of autoregressive inference lead to the specialty of LLM inference SLOs. Because the first token generation time is only related to the input length, it makes the first token generation time predictable. There is a common understanding that the time to the first token SLO can be set as a constant $[1,10,20,32]$. However, most of the previous work $[1,10,20]$ adopts the time between tokens as the SLO metric for the decode phase, which is overly strict and does not directly affect the user's experience, as discussed in Section 2.2. Only considering the time between tokens SLO also harms the fairness of LLM inference [24]. The average token generation time SLO we use in this paper directly affects the quality of service and achieves better fairness for users.

LLM Inference Systems. Recent work on LLM serving systems can be categorized into three classes. The first category is application-level optimization [21], where Continuous batching [29] and page attention [12,26] optimize the batch efficiency of LLM inference. Chunked-prefill [1,2] balances the prefill and decode to improve LLM inference throughput. The second category is inference worker optimization [18]; Splitwise [20] adopts heterogeneous GPUs to handle different bottlenecks in prefill and decode phases. Previous work [10,32] designed search algorithms to find the optimal worker configuration to achieve the best per-GPU goodput. The third class is about workload scheduling; recent work [14,22] focuses on the scheduling of queries to improve the QoE or throughput. However, they lack consideration for resource management.

## 8 Conclusion

We propose Aladdin, an adaptive LLM serving system that effectively scale and configures computing resources and optimally places inference queries to minimize serving costs while fulfilling SLOs. In this paper, we first deduce the performance models of the batched prefill and decode phases in LLM inference. Then, we predict the minimal computing resources required along with the corresponding worker configuration and request allocation. Results show that Aladdin reduced LLM serving costs by up to $71 \%$ compared to state-of-the-art baselines.

## References

[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming throughputlatency tradeoff in $1 \mathrm{~lm}$ inference with sarathi-serve, 2024.

[2] Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, and Ramachandran
Ramjee. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills, 2023.

[3] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895-4901, Singapore, December 2023. Association for Computational Linguistics.

[4] Marc Brysbaert. How many words do we read per minute? a review and meta-analysis of reading rate. Journal of memory and language, 109:104047, 2019.

[5] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $90 \% *$ chatgpt quality, March 2023.

[6] Daniel Crankshaw, Xin Wang, Guilio Zhou, Michael J. Franklin, Joseph E. Gonzalez, and Ion Stoica. Clipper: A Low-Latency online prediction serving system. In 14th USENIX Symposium on Networked Systems Design and Implementation (NSDI 17), pages 613-627, Boston, MA, March 2017. USENIX Association.

[7] Robert Grandl, Ganesh Ananthanarayanan, Srikanth Kandula, Sriram Rao, and Aditya Akella. Multiresource packing for cluster schedulers. In Proceedings of the 2014 ACM Conference on SIGCOMM, SIGCOMM '14, page 455-466, New York, NY, USA, 2014. Association for Computing Machinery.

[8] Arpan Gujarati, Reza Karimi, Safya Alzayat, Wei Hao, Antoine Kaufmann, Ymir Vigfusson, and Jonathan Mace. Serving DNNs like clockwork: Performance predictability from the bottom up. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), pages 443-462. USENIX Association, November 2020.

[9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.

[10] Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, and Yizhou Shan. Inference without interference: Disaggregate llm inference for mixed downstream workloads, 2024.

[11] Sangeetha Abdu Jyothi, Carlo Curino, Ishai Menache, Shravan Matthur Narayanamurthy, Alexey Tumanov,

Jonathan Yaniv, Ruslan Mavlyutov, Inigo Goiri, Subru Krishnan, Janardhan Kulkarni, and Sriram Rao. Morpheus: Towards automated SLOs for enterprise clusters. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), pages 117-134, Savannah, GA, November 2016. USENIX Association.

[12] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP '23, page 611-626, New York, NY, USA, 2023. Association for Computing Machinery.

[13] Adam Letchford. Approximation algorithms: Vv vazirani, springer-verlag, 2001. Journal of the Operational Research Society, 53:807-808, 072002.

[14] Jiachen Liu, Zhiyu Wu, Jae-Won Chung, Fan Lai, Myungjin Lee, and Mosharaf Chowdhury. Andes: Defining and enhancing quality-of-experience in llm-based text streaming services, 2024.

[15] Deepak Narayanan, Keshav Santhanam, Peter Henderson, Rishi Bommasani, Tony Lee, and Percy S Liang. Cheaply estimating inference efficiency metrics for autoregressive transformer models. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 66518-66538. Curran Associates, Inc., 2023.

[16] The new york times. The desperate hunt for the a.i. boom's most indispensable prize. https://www.nytimes.com/2023/08/16/ technology/ai-gpu-chips-shortage.html, 2023.

[17] NVIDIA. cublas. https://docs.nvidia.com/cuda/ cublas/index.html, 2023.

[18] Hyungjun Oh, Kihong Kim, Jaemin Kim, Sungkyun Kim, Junyeol Lee, Du-seong Chang, and Jiwon Seo. Exegpt: Constraint-aware resource scheduling for llm inference. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS '24, page 369-384, New York, NY, USA, 2024. Association for Computing Machinery.

[19] OpenAI. Gpts. https://openai.com/blog/ introducing-gpts, 2023.
[20] Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative llm inference using phase splitting, 2023.

[21] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference, 2022.

[22] Haoran Qiu, Weichao Mao, Archit Patke, Shengkun Cui, Saurabh Jha, Chen Wang, Hubertus Franke, Zbigniew T. Kalbarczyk, Tamer Başar, and Ravishankar K. Iyer. Efficient interactive llm serving with proxy model-based sequence length prediction, 2024.

[23] Francisco Romero, Qian Li, Neeraja J. Yadwadkar, and Christos Kozyrakis. INFaaS: Automated model-less inference serving. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pages 397-411. USENIX Association, July 2021.

[24] Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, and Ion Stoica. Fairness in serving large language models, 2023.

[25] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. Flexgen: High-throughput generative inference of large language models with a single gpu, 2023.

[26] Foteini Strati, Sara Mcallister, Amar Phanishayee, Jakub Tarnawski, and Ana Klimovic. Déjàvu: Kv-cache streaming for fast, fault-tolerant generative llm serving, 2024.

[27] Sharegpt teams. Sharegpt. https://huggingface. co/datasets/anon8231489123/ShareGPT_Vicuna_ unfiltered, 2023.

[28] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan

Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

[29] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for Transformer-Based generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pages 521-538, Carlsbad, CA, July 2022. USENIX Association.

[30] Chengliang Zhang, Minchen Yu, Wei Wang, and Feng Yan. MArk: Exploiting cloud services for CostEffective, SLO-Aware machine learning inference serving. In 2019 USENIX Annual Technical Conference (USENIX ATC 19), pages 1049-1062, Renton, WA, July 2019. USENIX Association.

[31] Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, and Yang You. Response length perception and sequence scheduling: An llm-empowered llm inference pipeline, 2023.

[32] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. Distserve: Disaggregating prefill and decoding for goodputoptimized large language model serving, 2024.
</end of paper 2>


<paper 3>
# Splitwise: Efficient Generative LLM Inference Using Phase Splitting 

Pratyush Patel ${ }^{1}$, Esha Choukse ${ }^{2}$, Chaojie Zhang ${ }^{2}$,<br>Aashaka Shah ${ }^{2}$, Íñigo Goiri ${ }^{2}$, Saeed Maleki ${ }^{2}$, Ricardo Bianchini ${ }^{2}$<br>${ }^{1}$ University of Washington $\quad{ }^{2}$ Microsoft


#### Abstract

Generative large language model (LLM) applications are growing rapidly, leading to large-scale deployments of expensive and power-hungry GPUs. Our characterization of LLM inference shows that each inference request undergoes two phases: a compute-intensive prompt computation phase and a memoryintensive token generation phase, each with distinct latency, throughput, memory, and power characteristics. Despite stateof-the-art batching and scheduling, the token generation phase underutilizes compute resources. Unlike prompt computation, token generation does not need the compute capability of the latest GPUs and can be run with lower power and cost.

Based on these insights, we propose Splitwise, a model deployment and scheduling technique that splits the two phases of LLM inference requests on to separate machines. Splitwise enables phase-specific resource management using hardware that is well suited for each phase. Request state is transferred efficiently between machines using optimized network libraries on the fast back-plane interconnects available in today's GPU clusters. Using Splitwise, we design homogeneous and heterogeneous LLM inference clusters optimized for throughput, cost, and power. Compared to current designs, Splitwise clusters achieve up to $1.4 \times$ higher throughput at $\mathbf{2 0 \%}$ lower cost. Alternatively, they can deliver $2.35 \times$ more throughput under the same power and cost budgets.


## I. INTRODUCTION

Recent advancements in generative large language models (LLMs) have significantly improved their response quality and accuracy [18], [71]. These trends have led to the widespread adoption of LLMs across various domains [6], [21]. Most modern LLMs are built using the transformer architecture [77], [78] and exhibit similar characteristics [63]. Transformer model sizes have grown steadily, from the early BERT models [36] having 340 million parameters, to GPT-3 [28] with a staggering 175 billion parameters, and GPT-4 rumored to have even more.

LLMs typically run on expensive and power-hungry GPUs [16]. The sudden and large-scale deployment of LLMs has led to a worldwide GPU capacity crunch [14]. The computational demand for LLM inference far exceeds that of training due to the vast number of applications leveraging LLMs. Furthermore, since training LLMs requires expensive and dedicated supercomputers [56], [60], a large number of inferences are necessary to amortize the high training costs. LLM inference jobs, although orders of magnitude smaller than training, are still expensive given the compute involved.[^0]

|  | A100 | H100 | Ratio |
| :---: | :---: | :---: | :---: |
| TFLOPs | 19.5 | 66.9 | $3.43 \times$ |
| HBM capacity | $80 \mathrm{~GB}$ | $80 \mathrm{~GB}$ | $1.00 \times$ |
| HBM bandwidth | $2039 \mathrm{GBps}$ | $3355 \mathrm{GBps}$ | $1.64 \times$ |
| Power | $400 \mathrm{~W}$ | $700 \mathrm{~W}$ | $1.75 \times$ |
| NVLink | $50 \mathrm{Gbps}$ | $100 \mathrm{Gbps}$ | $2.00 \times$ |
| Infiniband | $200 \mathrm{GBps}$ | $400 \mathrm{GBps}$ | $2.00 \times$ |
| Cost per machine [5] | $\$ 17.6 / \mathrm{hr}$ | $\$ 38 / \mathrm{hr}$ | $2.16 \times$ |

TABLE I: NVIDIA A100 vs. H100 specifications.

Generative LLM inference for a single request consists of several forward passes through the model, since the output tokens are generated one by one. This inherently has two contrasting phases of computation. First, the prompt computation phase, in which all the input prompt tokens run through the forward pass of the model in parallel to generate the first output token. This phase tends to be computationally intensive and requires the high FLOPs (floating point operations per second) of the latest GPUs today. Second, the token generation phase, in which subsequent output tokens are generated sequentially based on the forward pass of the last token and all the cached context from previous tokens in the sequence. Given the lack of compute parallelism, this phase tends to be more memory bandwidth and capacity bound, despite state-of-the-art batching. Running both phases on the same machine often leads to inconsistent end-to-end latencies due to the arbitrary batching of prompt and token phases. Due to these challenges, services need to over-provision expensive GPUs to meet tight inference service level objectives (SLOs) for interactive applications. At the same time, cloud service providers (CSPs) are having to build a lot of new datacenters to meet the GPU demand, and are running into a power wall [19].

The industry continues to release new computationally powerful GPUs, each much more power hungry and expensive than the last. However, as shown in Table I, the high-bandwidth memory (HBM) capacity and bandwidth on these GPUs has not scaled at the same rate recently. The latest NVIDIA H100 GPUs have $3.43 \times$ more compute and $1.75 \times$ more power compared to their predecessor A100 GPUs. However, their memory bandwidth only grew by $1.6 \times$, with no increase in memory capacity.

Our work. Given the distinct properties of prompt computation and token generation phases, we propose splitting the inference
request and running them on separate machines. Doing so allows us to separately manage hardware resources for each phase, thereby increasing the GPU utilization and the overall efficiency of the system. It also enables using different, bettersuited hardware for each phase. To realize such a setup, the cached context from the prompt computation needs to be communicated over from the prompt processing machine to the token generation machine at low latency. We implement these transfers in an optimized manner over the back-end Infiniband interconnects avaialble in datacenters today, allowing us to increase efficiency without any perceived performance loss.

With Splitwise, we design clusters optimized for cost, throughput, and power, using production traces of LLM inference requests [4]. Given the diverging memory and compute scaling rates across GPU generations, we also evaluate different GPUs and power caps for the different inference phases. This allows us to target better performance per dollar (Perf/\$) for users, and better performance per watt (Perf/W) for CSPs. Additionally, users can target older GPUs, which are likely more readily available to them.

We show that Splitwise-based LLM inference clusters can achieve $1.4 \times$ higher throughput at $20 \%$ lower cost than existing clusters. Alternatively, they can deliver $2.35 \times$ more throughput with the same cost and power budgets.

Summary. We make the following contributions:

1) An extensive characterization of the differences in the execution and utilization patterns of the prompt and token generation phases in LLM inference on the NVIDIA A100 and H100 GPUs using production traces.
2) Splitwise, our technique for optimized utilization of available hardware, which splits the prompt computation and token generation phases onto separate machines.
3) A design exploration of homogeneous and heterogeneous cluster deployments with Splitwise to optimize the overall cost, request throughput, and provisioned power.
4) An evaluation of the systems designed with Splitwise using production traces.

## II. BACKGROUND

## A. Large Language Models

Modern LLMs are based on transformers. Transformer models use attention [77] and multi-layer-perceptron layers to understand the inputs and generate an output, respectively. Transformer-based LLMs include encoder-only [36], [54], decoder-only [67], [69], [71], and encoder-decoder [70] models. Generative LLMs, the focus of this paper, are usually either decoder-only, or encoder-decoder models.

## B. Generative LLM inference phases

Figure 1 shows an example of generative LLM inference. Once the prompt query is received, all the input tokens are computed in parallel, within a single iteration, to generate the first token. We call this the prompt processing phase. The context generated from the attention layers during the prompt computation is saved in the key-value (KV) cache, since it

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-02.jpg?height=301&width=905&top_left_y=150&top_left_x=1068)

Fig. 1: An LLM inference example.

| Metric | Importance to user |
| :---: | :---: |
| End-to-end (E2E) latency | Total query time that the user sees |
| Time to first token (TTFT) | How quickly user sees initial response |
| Time between tokens (TBT) | Average token streaming latency |
| Throughput | Requests per second |

TABLE II: Performance metrics for LLMs.

is needed for all the future token generation iterations. After the first token is generated, the following tokens only use the last generated token and the $\mathrm{KV}$-cache as inputs to the forward pass of the model. This makes the subsequent token generation more memory bandwidth and capacity intensive than the computationally heavy prompt phase.

## C. Performance metrics for LLMs

Prior work has proposed three main metrics for LLM inference: end-to-end (E2E) latency, time to first token (TTFT), and throughput. We add another latency metric: time between tokens (TBT), to track the online streaming throughput of the tokens as they are generated serially. Table II summarizes the key performance metrics that we consider in this work.

Generative LLMs may be used for a variety of tasks with different kinds of SLOs. For batch tasks (e.g., summarization), TTFT or TBT latency metrics are less important than throughput. On the other hand, for latency-sensitive tasks (e.g., conversational APIs), TTFT and TBT are the more important metrics with tighter SLOs.

## D. Batching of requests

Inference requests can be batched together for higher throughput. Several prior works have explored batching [23], [81]. Figure 2 shows the timelines for inference with three common batching mechanisms. The default mechanism only batches at the request-level (Figure 2(a)). In this case, ready requests are batched together, but all the forward passes for these requests are completed before any other requests are run. Since requests can have long token generation phases, this can lead to long wait times for requests arriving in between, causing high TTFT and high E2E latencies. An optimization is continuous batching [81] (Figure 2(b)). In this case, scheduling decisions are made before each forward pass of the model. However, any given batch comprises either only of requests in their prompt phase or only requests in token phase. Prompt phase is considered more important since it impacts TTFT. Hence, a waiting prompt can preempt a token phase. Although this leads to shorter TTFT, it can substantially increase the tail
![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-03.jpg?height=270&width=908&top_left_y=160&top_left_x=150)

(a) Request-level.

(b) Continuous.

(c) Mixed.

Fig. 2: Batching mechanisms and their latency impact on the prompt and token phases.

for TBT, and therefore the E2E latency. Finally, there is mixed batching (Figure 2(c)) [23]. With this batching, the scheduling decisions are made at each forward pass, and the prompt and token phases can run together. This reduces the impact on TBT, but does not eliminate it, since token phases scheduled with prompt phases will experience a longer runtime. In the rest of the paper, we use mixed batching unless stated otherwise.

## E. Model parallelism

Model parallelism can be used to divide a model onto multiple GPUs, and even multiple machines, for higher efficiency and memory capacity. LLM inference typically uses pipeline and tensor parallelism. Pipeline parallelism (PP) divides the layers of the model among the GPUs, while keeping all the operators and tensors within a layer on the same GPU. Tensor parallelism (TP) divides the tensor across the GPUs, while replicating all the layers on each GPU. Pipeline parallelism requires lower communication across the participating GPUs, while tensor parallelism requires high bandwidth communication for each layer. In general, tensor parallelism performs better for GPUs within the same machine, connected with high bandwidth interconnects like e.g. NVLink [15]. In the rest of the paper, we use tensor parallelism across 8 GPUs for the best latency.

## F. GPU clusters and interconnects

With the recent rise of LLM use cases, several cloud service providers have expanded the GPU-based offerings, leading to large GPU cluster deployments [5], [56], [57]. Each machine in these AI clusters is generally comprised of 8 flagship NVIDIA GPUs (A100 or H100). Each GPU is connected to all the other GPUs in the cluster with a high bandwidth Mellanox InfiniBand interconnect [10], [13], forming a high bandwidth data plane network. The InfiniBand bandwidth offered in the cloud today ranges from 25 to 50GBps per GPU pair [7], [10].

## III. CHARACTERIZATION

In this section, we explore the performance and utilization characteristics of LLM inference and draw key insights to guide the design of Splitwise.

Production traces. We use production traces taken from two Azure LLM inference services on November $11^{\text {th }}$ 2023. Our traces represent the most common scenarios in LLM inference today: coding and conversation. We have released a subset of our traces at https://github.com/Azure/AzurePublicDataset [4]. The traces we use for characterization are 20 minutes long and include the arrival time, input size (number of prompt tokens),

| Model | \#Layers | Hidden size | \#Heads |
| :---: | :---: | :---: | :---: |
| Llama2-70B | 80 | 8192 | 32 |
| BLOOM-176B | 70 | 14336 | 112 |

TABLE III: Models we evaluate and their parameters.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-03.jpg?height=342&width=876&top_left_y=398&top_left_x=1096)

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-03.jpg?height=263&width=420&top_left_y=411&top_left_x=1102)

(a) Prompt input tokens.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-03.jpg?height=266&width=421&top_left_y=409&top_left_x=1535)

(b) Generated output tokens.
Fig. 3: Distribution for prompt and generated tokens.

and output size (number of output tokens). Due to customer privacy requirements (e.g., GDPR), we do not have visibility into the content of the prompts. We instead use the production traces to guide the input and output sizes, where we send the input prompt with the required number of tokens, and force the model to generate the corresponding number of output tokens for each request. Note that the text of the inputs prompts does not impact the performance metrics that we benchmark, since they depend only on the input and output sizes. For this characterization, we do not reuse the KV-cache between requests to emulate a cloud service with security guarantees.

Models. Table III shows the models that we evaluate. Both BLOOM [69] and Llama2 [71] are state-of-the-art open source LLMs. Both models are decoder-only, transformer-based models. We use the version of each model with the most parameters, since these versions are the most representative for production-class accuracy. Unless stated otherwise, we run BLOOM-176B and Llama-70B on vLLM [51] on a machine with $8 \mathrm{H} 100$ [16] GPUs.

## A. Number of prompt and generated tokens

To better understand our traces, we examine the distribution of the number of prompt input and generated output tokens. Figure 3a shows the distribution of number of prompt tokens. Since the coding LLM inference service is generally used to generate completions as the user is writing code, its input prompt can include large chunks of the code written so far. Thus, it has a large median prompt size of 1500 tokens. On the other hand, the conversation service has a wider range of input prompt tokens since it depends on the user. The median number of prompt tokens for this trace is 1020 tokens.

Figure $3 b$ shows the distribution of the number of generated tokens. Since the coding service typically only generates the next few words in the program as the user types, the median number of output token is 13 tokens. On the other hand, the conversation service has an almost bimodal distribution, with a median of 129 tokens generated.

Insight I: Different inference services may have widely different prompt and token distributions.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-04.jpg?height=317&width=634&top_left_y=156&top_left_x=282)

Fig. 4: Cumulative distribution of time spent with various active batched tokens.
![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-04.jpg?height=358&width=930&top_left_y=604&top_left_x=164)

(a) TTFT by prompt(b) size.

Fig. 5: TTFT, TBT, and E2E for BLOOM-176B and Llama70B on DGX-H100.

## B. Batch utilization

To understand how much can these requests be batched, we measure how often machines run at a given batch size. We use mixed continuous batching as shown in Figure 2. To fit into a single machine, we run a scaled-down version of the coding and conversation traces with 2 requests per second.

Figure 4 shows the distribution of the time spent by the machine running various number of active tokens in a batch. Note that if a prompt of 100 tokens is running in its prompt phase, we count the active tokens as 100 . However, once the request is in the token phase, we count it as one active token, since the tokens are generated one at a time (assuming a beam search size of one [51]). We find that most of the time (60-70\%) for conversation is spent running only 20 tokens or fewer. Since the coding service has very few output tokens, it experiences even worse batching in the token phase and runs with a single token for more than $20 \%$ of the time. Both the LLMs show very similar trends.

Insight II: Mixed continuous batching spends most of the time with very few active tokens batched.

## C. Latency

TTFT. Figure 5a shows the impact of the number of prompt tokens on TTFT. The range of sizes was chosen based on the coding and conversation traces. We find that TTFT for both models grows almost linearly as the prompt size increases. This behavior is due to the prompt phase having high GPU utilization and being computationally bound.

TBT. Figure 5b shows the impact of forcefully batching the output tokens of different requests together on the TBT. We observe very little impact on TBT as the batch size grows. With a batch size of 64 , there is only $2 \times$ impact on TBT.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-04.jpg?height=274&width=442&top_left_y=172&top_left_x=1080)

(a) Prompt phase.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-04.jpg?height=263&width=423&top_left_y=172&top_left_x=1537)

(b) Token generation phase.
Fig. 6: Impact of batching on the throughput for the 2 LLMs.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-04.jpg?height=383&width=550&top_left_y=600&top_left_x=1232)

Fig. 7: Required memory with batching in prompt/token phases.

E2E. Figure 5c shows various percentiles of E2E latency for both models, with no batching. The variability between the request input and output sizes is apparent. Furthermore, we see that most of the E2E time is spent running the token phase. This holds true even for the coding trace, where prompt sizes are large and generated tokens few. In fact, we find that for BLOOM-176B, a prompt phase with 1500 input tokens takes the same time as token phase with only 6 output tokens.

Insight III: For most requests, the majority of the E2E time is spent in the token generation phase.

## D. Throughput

Figure 6 shows the impact of batching on the throughput (measured as tokens per second). For the prompt phase, we define the throughput as the number of prompt input tokens that are processed per second. We see that the throughput decreases after 2048 prompt tokens, which corresponds to a batch size of less than 2 for the median prompt sizes from the traces. On the other hand, Figure $6 b$ shows that the throughput in the token phase keeps increasing with batching until 64 batch-size, at which point, the machine runs out of memory.

Insight IV: The prompt phase batch size should be limited to ensure good performance. In contrast, batching the token generation phase yields high throughput without any downside.

## E. Memory utilization

During an LLM inference, the GPU memory is used to host the model weights and activations, as well as the KV caches (Section II-B). As the number of tokens in a batch increase, the memory capacity required for the $\mathrm{KV}$ cache also increases. Figure 7 shows the memory capacity utilization during each phase as the number of tokens in the batch increases. During the prompt phase, the input prompt tokens generate the $\mathrm{KV}$

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-05.jpg?height=296&width=420&top_left_y=172&top_left_x=191)

(a) Prompt phase.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-05.jpg?height=287&width=418&top_left_y=171&top_left_x=626)

(b) Token generation phase.
Fig. 8: Maximum and mean power utilization varying the batching size.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-05.jpg?height=360&width=876&top_left_y=660&top_left_x=169)

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-05.jpg?height=284&width=420&top_left_y=671&top_left_x=188)

(a) Prompt phase.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-05.jpg?height=282&width=418&top_left_y=669&top_left_x=626)

(b) Token generation phase.
Fig. 9: Impact of power cap on the prompt and token generation latency with the maximum batch size possible.

cache. During the output token phase, each active generated token that is being processed accesses the $\mathrm{KV}$ cache of its entire context so far.

Insight V: Batching during the prompt phase is compute-bound, whereas the token phase is limited by memory capacity.

## F. Power utilization

When hosting machines, cloud providers need to consider the peak power draw, which has direct impact in the datacenter cost [26]. This is especially important when building GPU clusters, since GPUs consume much higher power than regular compute machines [63], [64]. Figure 8 shows the GPU power draw normalized to the thermal design power (TDP) when running prompt and token generation phases. Since the the prompt phase is compute intensive, its power draw increases with batch size. On the other hand, the token phase is memory bound and its power draw does not vary when increasing the number of tokens to process.

Providers can cap the power usage of the machines to reduce the peak power. Figure 9 shows the impact to latency when increasing the power caps for both prompt and token phases. The prompt phase is highly sensitive to the power cap and the latency increases substantially. On the other hand, the token generation phase incurs almost no latency impact when power capping by over $50 \%$ (i.e., 700 to $350 \mathrm{~W}$ ).

Insight VI: While the prompt phase utilizes the power budget of the GPU efficiently, the token phase does not.

## G. GPU hardware variations

Given the different characteristics of prompt and token generation phases, we measure the performance impact on

|  |  | Coding |  | Conversation |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | A100 | H100 | Ratio | A100 | H100 | Ratio |
| TTFT | $185 \mathrm{~ms}$ | $95 \mathrm{~ms}$ | $0.51 \times$ | $155 \mathrm{~ms}$ | $84 \mathrm{~ms}$ | $0.54 \times$ |
| TBT | $52 \mathrm{~ms}$ | $31 \mathrm{~ms}$ | $0.70 \times$ | $40 \mathrm{~ms}$ | $28 \mathrm{~ms}$ | $0.70 \times$ |
| E2E | $856 \mathrm{~ms}$ | $493 \mathrm{~ms}$ | $0.58 \times$ | $4957 \mathrm{~ms}$ | $3387 \mathrm{~ms}$ | $0.68 \times$ |
| Cost [5] | $\$ 0.42$ | $\$ 0.52$ | $1.24 \times$ | $\$ 2.4$ | $\$ 3.6$ | $1.5 \times$ |
| Energy | $1.37 \mathrm{Whr}$ | $1.37 \mathrm{Whr}$ | $1 \times$ | $7.9 \mathrm{Whr}$ | $9.4 \mathrm{Whr}$ | $1.2 \times$ |

TABLE IV: P50 request metrics on A100 vs. H100 without batching on Llama-70B.

the two from running on different hardware. Table I shows the specifications for DGX-A100 [15] and DGX-H100 [16]. The memory-to-compute ratio favors A100 over H100. Table IV shows our findings. We see a lower performance impact on the token generation phase (TBT) as compared to the Prompt phase (TTFT). Since coding requests are dominated by prompt phase, by having very few generated tokens, the E2E latency impact from A100 is worse on coding than conversation. Furthermore, we see that A100 has better or equal inference cost and energy overall compared to H100.

Insight VII: Token generation can be run on less computecapable hardware for better Perf/W and Perf/\$ efficiencies.

## IV. SplitWISE

Based on our characterization insights, we propose Splitwise, a technique to split the prompt and generation phases in the LLM inference on to separate machines.

Figure 10 shows the high-level overview of Splitwise. We maintain two separate pools of machines for prompt and token processing. A third machine pool, the mixed pool, expands and contracts as needed by the workload. All machines are preloaded with the model of choice. When a new inference request arrives, the scheduler allocates it to a pair of machines (i.e., prompt and token). The prompt machines are responsible for generating the first token for an input query, by processing all the input prompt tokens in the prompt phase and generating the KV-cache. The prompt machine also sends over the KV-cache to the token machine, which continues the token generation until the response is complete. We use continuous batching at the token machines to maximize their utilization. Machines in mixed pool use mixed continuous batching.

At a lower request rate, we target better latency in Splitwise, while, at a higher request rate, we target avoiding any performance or throughput reduction due to the fragmentation between prompt and token machine pools.

Splitwise uses a hierarchical two-level scheduling as shown in Figure 10. The cluster-level scheduler (CLS) 1) is responsible for machine pool management and for routing incoming inference requests. The machine-level scheduler (MLS) 2 maintains the pending queue and manages batching of requests at each machine.

## A. Cluster-level scheduling

Machine pool management. The CLS maintains the prompt, token, and mixed machine pools 3. Splitwise initially assigns

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-06.jpg?height=770&width=707&top_left_y=154&top_left_x=248)

Fig. 10: High-level system diagram of Splitwise.

machines to the prompt or token pool depending on the expected request load and input/output token distributions. Machines from the prompt or token pools may be dynamically moved into and out of the mixed pool to reduce fragmentation and meet SLOs at higher loads. A machine in the mixed pool retains its identity as a prompt or token machine and goes back to its original pool once there are no tasks of the opposite kind in its pending queue. Switching pools does not incur any noticeable latency. If the load distribution deviates considerably from initial assumptions, Splitwise employs a coarse grained re-purposing of machines and moves machines between the prompt and token pools. Re-purposing of machines is done infrequently, typically only if they stay in the mixed pool for a considerable amount of time.

Request routing. CLS uses Join the Shortest Queue (JSQ) scheduling [39], [85] to assign a prompt and a token machine to each request. Queue lengths are defined by the number of pending tokens. Each machine regularly communicates to the CLS changes in its memory capacity or pending queue. Note that this does not happen at every iteration boundary. We simultaneously assign both the prompt and token machine when scheduling requests, since we can then overlap KV-cache transfers with prompt computation to reduce transfer overheads (Section IV-C).

When routing requests, if the pending queue is bigger than a certain threshold, the CLS looks for target machines in the mixed pool. If the mixed pool is also full, it proceeds to look in the opposite pool (i.e., a token machine to run prompts and vice versa) and moves the machine into the mixed pool. Machines in the mixed pool operate exactly as a non-Splitwise machine would, with mixed batching. Once the queue of mixed requests is drained, the CLS transitions the machine back to its original pool. For example, when the queue is too long, we can move a prompt machine to the mixed pool to run tokens;

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-06.jpg?height=257&width=873&top_left_y=159&top_left_x=1081)

(a) Serialized KV-cache transfer. (b) Optimized KV-cache transfer per-layer during prompt phase.

Fig. 11: Optimizing KV-cache transfer in Splitwise.

once the machine is done running tokens, we transition the machine back into the prompt pool.

## B. Machine-level scheduling

The MLS runs on each machine and is responsible for tracking the GPU memory utilization, maintaining the pending queue (4), deciding the batch for each iteration, and reporting the relevant status to the CLS.

Prompt machines. The MLS simply uses first-come-first-serve (FCFS) to schedule prompts. The results in Figure 6a show that after 2048 prompt tokens, the throughput degrades. For this reason, the MLS restricts the batching of multiple prompts together to 2048 tokens in total. This is a configurable value, and can change for a different model or hardware.

Token machines. The MLS uses FCFS to schedule tokens and batches as much as possible. Figure $6 \mathrm{~b}$ shows that the token generation throughput keeps scaling up with the batch size until the machine runs out of memory. For this reason, the MLS tracks the memory and starts queueing tokens once the machine is close to running out of memory.

Mixed machines. To meet the TTFT SLO, the MLS must prioritize running prompts and schedule any new prompts in the pending queue immediately. If the machine is running token phases and has no additional capacity to run the prompt phase, the MLS will preempt tokens. To avoid starvation of the token phase due to preemption, we increase the priority of the token with age and limit the number of preemptions that each request can have.

## C. KV-cache transfer

As discussed in Section II, the KV-cache is generated during the prompt phase of the request, and it continuously grows during the token generation phase. In Splitwise, we need to transfer the KV-cache from the prompt machine to the token machine (5) (shown in Figure 10) to complete the inference. This transfer delay is the main overhead associated with Splitwise. In this section, we discuss the impact of KVcache transfer and how we optimize it.

Figure 11a shows the Gantt chart for the prompt phase, the $\mathrm{KV}$-cache transfer, and the token generation phase for a single batch of requests when naively transferring the $\mathrm{KV}$ cache in a serialized way. The KV-cache transfer starts only after the prompt phase has finished and the first token is generated. Further, it needs to complete before the next output token
can be generated in the token generation phase. This directly impacts the maximum TBT and end-to-end latency of inference.

The time required for the transfer depends on the size of the KV cache (which is directly proportional to the number of prompt tokens) and on the bandwidth of the interconnect between the prompt and the token machines. Even when using fast InfiniBand links, the transfer overhead for large prompt sizes could become a significant fraction of the TBT.

In Splitwise, we optimize the KV-cache transfer by overlapping it with the computation in the prompt phase. As each layer in the LLM gets calculated in the prompt machine, the KV cache corresponding to that layer is also generated. At the end of each layer, we trigger an asynchronous transfer of the KVcache for that layer while the prompt computation continues to the next layer. Figure 11b shows this asynchronous transfer which reduces the transfer overheads. Layer-wise transfer also enables other optimizations, such as earlier start of the token phase in the token machines, as well as earlier release of KV-cache memory on the prompt machines.

Layer-wise KV-cache transfer happens in parallel with the prompt computation for the next layer. This requires finegrained synchronization per layer for correctness. Thus, it is possible to incur performance interference and increase the TTFT, especially for smaller prompts. However, for small prompts the total KV-cache size is small and does not need the layer-wise transfer to hide the latency. Since the number of tokens in a batch is already known at the start of computation, Splitwise picks the best technique for KV-cache transfer. It uses serialized KV-cache transfer for smaller prompts and layer-wise transfer and for larger prompts. We show that the overall transfer and interference overheads are relatively small in Section VI-A.

## D. Provisioning with Splitwise

We leverage Splitwise to optimize LLM inference cluster deployments for power, cost, and throughput.

Type of machines. We propose four main variants of Splitwisebased systems: Splitwise-AA, Splitwise-HH, Splitwise-HA, and Splitwise-HHcap. The nomenclature is simply drawn from the first letter representing the Prompt machine type, and the second letter representing the Token machine type. "A" represents a DGX-A100 machine, "H" represents a DGX-H100 machine, and "Hcap" represents a power-capped DGX-H100 machine. Table $\mathrm{V}$ shows a summary of the cost, power, and hardware in each of our evaluated systems.

Splitwise-AA uses DGX-A100 for both prompt and token pools, while Splitwise-HH uses DGX-H100 for both. These two variants represent the commonly available setups in providers where machines are homogeneous and interchangeable.

Splitwise-HA uses DGX-H100 for the prompt pool and DGX-A100 for the token pool. We choose this configuration based on Table IV, and the Insight VII (i.e., A100s can be more cost- and power-efficient for the token phase).

Splitwise-HHcap uses DGX-H100 machines for both prompt and token pools. However, we power cap the token machines down to $70 \%$ of their rated power, with each GPU capped by
![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-07.jpg?height=688&width=878&top_left_y=171&top_left_x=1076)
SLO ${ }_{-5}^{6+}$
![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-07.jpg?height=228&width=870&top_left_y=620&top_left_x=1080)

Fig. 12: Design space for provisioning a Splitwise-HH cluster. Cluster configurations targets a peak throughput of 70 RPS. The cost-optimal Splitwise-HH configuration is marked with * (27 prompt and 3 token machines).

$50 \%$ of the power. We propose this design based on Figure 9 and Insight VII (i.e., the prompts phase is impacted by power caps while token has no performance impact with $50 \%$ lower power cap per GPU).

Number of machines. The LLM inference cluster deployment must be sized with the appropriate number of prompt and token machines. Our methodology involves searching the design space using our event-driven cluster simulator, which is described in detail in Section V. We need to provide as input: (1) the target cluster design (e.g., Splitwise-HA or Splitwise-HHcap), (2) an LLM-specific performance model that can estimate the TTFT and TBT at various input, output, and batch sizes, (3) a short trace derived from the target prompt and token size distributions for the service (e.g., Figure 3), (4) the SLOs (e.g., Table VI), (5) the constraints (e.g., throughput), and (6) the optimization goal (e.g., minimize cost). Using this information, our provisioning framework searches the space for the desired optimal point. For example, searching with a throughput constraint and a cost minimization goal gives us iso-throughput cost-optimized clusters across different designs.

Search space. Figure 12 shows an example of the twodimensional search space for the number of prompt and token machines under Splitwise-HH for the coding workload (using a 2-minute trace). The simulator outputs the various percentiles for TTFT, TBT, and E2E latencies. Then, we select the clusters that meet the SLOs for each of these metrics and optimize our target function. For example, Figure 12 shows a $\star$ for the setup with 27 prompt and 3 token machines with the lowest cost that achieves 70 RPS. We call this setup iso-throughput cost-optimized.

Optimization. We can use three optimization goals: throughput, cost, and power. Throughput optimization is important for both,

|  | Prompt Machine |  |  |  | Token Machine |  | Prompt-Token |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Type | Cost | Power | Type | Cost | Power | Interconnect Bandwidth |
| Splitwise-AA | DGX-A100 | $1 \times$ | $1 \times$ | DGX-A100 | $1 \times$ | $1 \times$ | $1 \times$ |
| Splitwise-HH | DGX-H100 | $2.35 \times$ | $1.75 \times$ | DGX-H100 | $2.5 \times$ | $1.75 \times$ | $2 \times$ |
| Splitwise-HHcap | DGX-H100 | $2.35 \times$ | $1.75 \times$ | DGX-H100 | $2.5 \times$ | $1.23 \times$ | $2 \times$ |
| Splitwise-HA | DGX-H100 | $2.35 \times$ | $1.75 \times$ | DGX-A100 | $1 \times$ | $1 \times$ | $1 \times$ |

TABLE V: Evaluated Splitwise designs all normalized to DGX-A100

the cloud service provider (CSP) and the user. Cost optimization has different importance levels to the CSP and the user. For the CSP, a higher cost for the same throughput might be acceptable if there are gains in power and space requirements for the cluster. However, for the end-user, a higher cost at the same throughput is generally unacceptable. Finally, power optimization is attractive for a CSP, since it enables more GPUs to be deployed in the same datacenter [62], [63], but it may not be as important to the user. We only consider the provisioned power, and not the dynamic power utilization, in our study.

## E. Practical Considerations

Accuracy impact. Splitwise does not impact accuracy since it uses lossless KV-cache transfer and does not add any randomization. It executes inference with the same parameters and state as on a single machine.

Scalability. Since LLM requests are much longer than typical ML requests [37], [38], they incur lower scheduling overhead for similar cluster sizes. However, the CLS may become a scalability bottleneck for large clusters. Insights from prior work on partitioned or replicated scheduling could help improve scalability [27], [61], [72] and are orthogonal to Splitwise.

Reliability and fault tolerance. If the prompt or the token machine fail, Splitwise simply restarts requests from scratch, similar to today's LLM serving systems [44], [51]. Alternatively, Splitwise could checkpoint the KV-cache generated after prompt computation into an in-memory database. To recover, Splitwise can use this cache to skip prompt recomputation, and start right away with the token phase. The KV-cache could also be checkpointed periodically during the token phase. Designing safe and efficient failure recovery is out of scope for our paper.

## V. MethodologY

## A. Experimental setup

To evaluate our proposal on real hardware, we implement Splitwise's KV-cache transfer mechanism on top of vLLM [51]. Our implementation is open source [1]. We run this modified vLLM on two DGX-A100 and two DGX-H10 virtual machines (VMs) on Microsoft Azure with specifications from Table I. These are the VMs used to collect the characterization data in Section III. These machines are connected with InfiniBand and the DGX-H100s have double the bandwidth (i.e., $400 \mathrm{Gbps}$ ).

Since vanilla vLLM only supports continuous batching with token preemption which can lead to much higher TBT, we implement state-of-the-art mixed continuous batching [81] as discussed earlier in Figure 2(c).

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-08.jpg?height=477&width=662&top_left_y=520&top_left_x=1184)

Fig. 13: Overview of the design of the Splitwise simulator.

Our implementation of the Splitwise technique assigns machines either a prompt role, or a token role. As the prompt machine generates the first token, it transfers the KVcache to the token machine using the technique described in Section IV-C. We use MSCCL++ [11], an optimized GPUdriven communication library, to implement the naive and layer-wise KV cache transfers.

In our implementation, the prompt machine uses the zerocopy one-sided put primitive of MSCCL++ to send KV-cache data over InfiniBand as soon as it is ready, without requiring the token machine to issue any receive instructions. Once we have issued a put for all layers, the prompt machine signals a semaphore that the token machine waits on. The synchronization done with the help of semaphores uses the same InfiniBand connection used to send KV-cache data. When processing a batch of prompts, each request is assigned a different semaphore since it may be routed to different token machines. We ship the KV-caches block-by-block in vLLM. To minimize the number of transfers, we also consider the contiguity of KV blocks as long as they use the same semaphore.

## B. Simulator setup

We build a simulator to explore cluster designs and evaluate Splitwise at scale. The simulator code is open source [20].

Figure 13 shows the design of our simulator. The simulator is event-driven and faithfully models the Splitwise machine pools, schedulers, machine-level memory and queues, and KV-cache transfer. We first profile the LLM on the target hardware with various input/output sizes 1. Based on the characterization profiles, we build a performance model. The simulator takes as input the request traces, SLOs, the performance model, and the configurations for cluster and scheduler 2. For our

|  | P50 | P90 | P99 |
| :---: | :---: | :---: | :---: |
| TTFT | $2 \times$ | $3 \times$ | $6 \times$ |
| TBT | $1.25 \times$ | $1.5 \times$ | $5 \times$ |
| E2E | $1.25 \times$ | $1.5 \times$ | $5 \times$ |

TABLE VI: SLO expressed as slowdown compared to a request running on DGX-A100 under no contention.

evaluation, we use the prompt and token size distributions from the production traces in Section III. We tune the Poisson arrival rate to increase and decrease the load (requests per second) for cluster sizing. The simulator provides the achieved metrics per request (TTFT, TBT, E2E), and the machine utilization levels (3). We cross-validated the performance model with hardware experiments to ensure accuracy; we also validated the simulator end-to-end using production load with over $50 \mathrm{~K}$ iterations to ensure fidelity (4).

Performance model. We build a piece-wise linear performance model using performance profiles at various batch sizes, input sizes, output sizes, in the required parallelism configuration on A100 and H100 machines from Section III. We validate that our performance model has high accuracy; it incurs a mean absolute percentage error (MAPE) of less than $3 \%$ when evaluated with a 80:20 train:test dataset split.

Communication model. In our evaluation, KV-cache transfers cause inter-machine communication, whereas tensor parallelism only causes intra-machine communication. We model intermachine communication overheads by benchmarking our KVcache transfer implementation over Infiniband in Section VI-A.

SLOs. To determine the maximum throughput that can be supported by a given cluster design, we use P50, P90, and P99 SLOs for TTFT, TBT, and E2E latency metrics. Table VI shows our SLO definition using DGX-A100 as a reference. We require all nine SLOs to be met. SLOs on TTFT are slightly looser, since it has a much smaller impact on the E2E latency.

Baselines. We compare our Splitwise designs against BaselineA100 and Baseline-H100. The clusters in these baselines consist of just DGX-A100s and DGX-H100s, respectively. Both baselines use the same mixed continuous batching that Splitwise uses for mixed pool machines (described in Section IV-A).

## VI. EVALUATION

## A. Experimental results

KV-cache transfer latency. We first measure the latency to transfer the KV-cache as the prompt size grows. Figure 14 shows the visible transfer latency on both A100 and H100 setups with the naive and optimized transfer design as discussed in Figure 11. Compared to the prompt computation time, the overhead is minimal $(<7 \%)$. The time for serialized transfers linearly increases with the prompt size since the size of the KV-cache also increases. The optimized per-layer transfer, on the other hand, hides much of the latency. For these transfers, we see a constant non-overlapped transfer time of around $8 \mathrm{~ms}$ for the A100 and around $5 \mathrm{~ms}$ for the $\mathrm{H} 100$ setup. The $\mathrm{H} 100$

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-09.jpg?height=344&width=523&top_left_y=175&top_left_x=1256)

Fig. 14: Overhead of the KV-cache transfer as the prompt size increases on A100s and H100s.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-09.jpg?height=363&width=637&top_left_y=680&top_left_x=1205)

Fig. 15: Overhead of KV cache transfer on TTFT, E2E latency for coding trace for A100 and H100.

setup has double the bandwidth of the A100 setup (i.e., 200 vs $400 \mathrm{Gbps}$ ), and the impact of this can be clearly seen with transfers in the $\mathrm{H} 100$ setup happening about twice as fast as those in the A100 setup.

As discussed in Section IV-C, for small prompt sizes (<512 in H100), Splitwise uses the serialized KV-cache transfer and for larger prompts, it uses per-layer transfers.

End-to-end impact. Next, we run the coding trace on the 2-machine Splitwise setups without batching, and compare the observed latency metrics to a 1-machine baseline setup with no batching. Figure 15 shows our results. The latency impact of serially transferring the KV-cache grows up to $3 \%$ of the E2E with large prompts. However, Splitwise only incurs $0.8 \%$ of E2E. In a user-facing inference, the only visible impact of KV-cache transfer overhead is the latency for the second token. Splitwise adds a $16.5 \%$ latency to the second token, as compared to the $64 \%$ overhead from a serialized transfer. Overall, the transfer impact in Splitwise is hardly perceivable even in a user-facing inference.

## B. Iso-power throughput-optimized clusters

Cluster provisioning. We provision clusters using the methodology described in Section IV-D. We target a specific workload (e.g., conversation) at a peak load with the same power (i.e., iso-power) for each cluster design. For the baseline, we use the power for 40 DGX-H100 machines as our target peak power. For the A100 baseline, we can fit 70 DGX-A100 machines under the same power budget. We denote these two designs as $40 \mathrm{P} / \mathrm{T}$ and $70 \mathrm{P} / \mathrm{T}$ respectively, since they both use mixed batching in all machines.

For Splitwise cluster designs under the coding trace, Splitwise-AA provisions 55 prompt machines and 15 for the token pool, denoted as (55P, 15P). Note that like BaselineA100, Splitwise-AA also provisions $75 \%$ more machines than Baseline-H100. The legends in Figure 16 show the different provisioning choices under coding and conversation workloads. Request size distributions reflect in the machine pool sizing. For example, we provision more prompt machines under Splitwise$\mathrm{HH}$ (35P, 5T) for the coding trace, while we provision more token machines $(25 \mathrm{P}, 15 \mathrm{~T})$ for the conversation trace.

Latency and throughput. Figure 16 shows a deep dive into all the latency metrics at different input load for each cluster design with the same power (i.e., iso-power). For the coding trace (Figure 16a), Splitwise-HH, Splitwise-HHcap, and Splitwise-AA all perform better than Baseline-H100. As the load increases, Baseline-H100 suffers from high TBT due to mixed batching with large prompt sizes. Although SplitwiseAA can support higher throughput, its TTFT is consistently higher than most designs. Splitwise-HA clearly bridges the gap by providing low TTFT and E2E at high throughput. The mixed machine pool in Splitwise becomes useful at higher loads to use all the available hardware without fragmentation. This benefit can be seen clearly in the P50 TBT chart for Splitwise-HA, where after 90 RPS, H100 machines jump into the mixed machine pool and help reduce TBT.

For the conversation trace (Figure 16b), Splitwise-HHcap clearly does better on all fronts, including latency. This is because its token generation phases typically run for much longer than in the coding trace, which is beneficial for the token machines.

Impact on batched tokens. Figure 17 shows the cumulative distribution of time spent processing a varying number of batched active tokens in an iso-power throughput-optimized cluster. The distributions are collected by running the conversation trace at low (70 RPS) and high (130 RPS) loads.

At low load, all 40 Baseline-H100 machines spend $70 \%$ of the time running $\leq 15$ tokens, and the rest running mixed batches with large prompts, which affects TBT and E2E. The 35 Splitwise-HH prompt machines are mostly idle, and when active, run much larger batches of tokens. The 15 Splitwise$\mathrm{HH}$ token machines also do a better job at batching. Overall, Splitwise machines have better batching and latency at 70 RPS. At high load, since the mixed pool is utilized more, the batch sizes start looking similar across prompt and token machines.

Summary plot. Figure 18a summarizes the results across all cluster metrics for iso-power throughput-optimized designs for the conversation trace. We use Baseline-A100 as the baseline. Compared to Baseline-A100, Splitwise-AA delivers $2.15 \times$ more throughput at the same power and cost. Splitwise-HA delivers $1.18 \times$ more throughput at $10 \%$ lower cost and the same power.

## C. Other cluster optimizations

We have described iso-power throughput-optimized clusters in detail. For the rest of the cluster optimization evaluation, we only discuss the summary plots.
![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-10.jpg?height=1262&width=896&top_left_y=168&top_left_x=1080)

(a) Coding trace.
![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-10.jpg?height=236&width=594&top_left_y=822&top_left_x=1096)

se-HA (25P, 26T)
![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-10.jpg?height=548&width=900&top_left_y=884&top_left_x=1080)

(b) Conversation trace.

Fig. 16: Latency metrics across input loads for iso-power throughput optimized clusters. Dashed red lines indicate SLO.

Iso-cost throughput-optimized. Figure 18 b shows the summary plot for iso-cost clusters, with their space, throughput, and power requirements. We find that Splitwise-AA provides the best throughput for the same cost, namely $1.4 \times$ more throughput than Baseline-H100, running at $25 \%$ more power, and $2 \times$ the space. This is an interesting operational point for most customers who may not care about power and space, instead preferring the $40 \%$ higher throughput using older, more easily available GPUs. In contrast, the preferable choice for the CSP is less clear.

Iso-throughput power-optimized. Figure 19a shows cluster designs that yield same throughput at the least power. SplitwiseHHcap can achieve the same throughput as Baseline-H100 at $25 \%$ lower power at the same cost and space. This can be a clear win for the CSPs.

Iso-throughput cost-optimized. Figure 19b shows the costoptimized versions of the iso-throughput design. Note that there are no changes to any of the homogeneous designs between Figures 19a and 19b. This is because the prompt and token machines have the same cost and power. However, Splitwise-

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=494&width=894&top_left_y=157&top_left_x=171)

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=415&width=425&top_left_y=172&top_left_x=183)

(a) Low load (70 RPS).

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=428&width=423&top_left_y=171&top_left_x=642)

(b) High load (130 RPS).

Fig. 17: Cumulative distribution of time spent at various batched token sizes for iso-power throughput-optimized design.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=390&width=878&top_left_y=797&top_left_x=184)

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=317&width=399&top_left_y=804&top_left_x=188)

(a) Iso-power.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=322&width=402&top_left_y=801&top_left_x=642)

(b) Iso-cost.
Fig. 18: Summary of throughput-optimized cluster designs.

HA and Splitwise-HHcap arrive at slightly different results with the cost and power optimizations. Figure 19b shows that with Splitwise-AA, customers can achieve the same throughput as Baseline-H100 at $25 \%$ lower cost.

## D. Impact of workload changes

So far, we have tested a trace and a model on clusters optimized for a specific workload pattern and model. To test the Splitwise' robustness, we now run conversation trace on a cluster meant for coding service, and Llama-70B on a cluster meant for BLOOM-176B. Figure 20 shows these results for iso-power throughput-optimized clusters.

Changing workload trace. Compared to Figure 16b, we find that in Figure 20a, the Baseline clusters are similarly sized and incur no throughput or latency impact. Splitwise-AA and Splitwise-HH with the mixed pool morph well to meet the requirements of the new workload, and they see no throughput or latency impact. Since Splitwise-HA and Splitwise-HHcap have different types of machines in the prompt and token pools, they experience a throughput setback of $7 \%$ from the respective cluster optimized designs for conversation trace. Note that all the Splitwise designs still perform much better than any of the Baseline designs.

Changing model. Figure 20b shows that Llama-70B can support much higher throughput in the same cluster design than BLOOM-176B, given its fewer parameters (Table III). All the Splitwise designs out-perform both the Baseline designs at higher load. Furthermore, Splitwise-HH and Splitwise-HHcap consistently achieve the best latency, even as the load increases.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=393&width=878&top_left_y=167&top_left_x=1098)

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=317&width=418&top_left_y=175&top_left_x=1100)

(a) Power-optimized.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=320&width=417&top_left_y=173&top_left_x=1556)

(b) Cost-optimized.
Fig. 19: Summary of iso-throughput cluster designs.
![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=634&width=880&top_left_y=648&top_left_x=1094)

(a) Conversation trace running on a cluster designed for coding.

![](https://cdn.mathpix.com/cropped/2024_06_04_b95729158cc8550c25d7g-11.jpg?height=244&width=870&top_left_y=970&top_left_x=1105)

(b) Llama-70B, on a cluster designed for BLOOM-176B.

Fig. 20: Latency impact of running a workload on a cluster designed for another workload. Dashed red lines indicate SLO.

Summary. Based on these two experiments, we conclude that Splitwise can morph according to the requirements of the workload using its smart scheduling, and it is robust to changes in the LLMs, request load, and token distributions.

## E. Cluster design for batch job

We design various clusters with Splitwise under strict latency SLOs, even when we are optimizing for throughput. This is unnecessary for batch jobs, which can be stressed to high load for a high token generation throughput. We find that upon stressing our iso-power throughput-optimized clusters, BaselineA100 and Splitwise-AA have the best throughput per cost at 0.89 RPS/\$. At high load, Splitwise devolves into the iso-count Baseline, since it starts mixed batching with all the machines in the mixed pool. The same holds true for Splitwise-HH and Baseline-H100, which achieve 0.75 RPS/\$.

## VII. DISCUSSION

Extensibility to new models. Despite the plethora of model sizes from 2B parameters [47], [84] to 176B parameters [69] or more [18], all modern transformer-based generative LLMs have the distinct prompt processing and token generation phases. Similarly, even modifications and flavors like Mixtureof-Experts (MoEs) have these phases. Since Splitwise is built solely by exploiting these phases, it is applicable to all of the
current and upcoming LLMs, as long as the auto-regressive nature of the workload requires these two phases. Note that as shown in Section VI-D, clusters provisioned with Splitwise for one model can also efficiently serve other models.

Alternative compute hardware. In this work, we use NVIDIA H100 and A100 GPUs since they are commonly used for LLM inference in datacenters today [17]. Smaller datacenter GPUs like NVIDIA T4 lack enough memory to run modern LLMs efficiently. In general, our methodology is applicable to any hardware (including CPUs, FPGAs, ASICs [33]) that aligns with the computational requirements of prompt and token phases. Our characterization suggests that prompt phases need high compute capability and memory bandwidth with low memory capacity, whereas token phases need moderate compute capability with high memory capacity and bandwidth. Thus, GPUs like AMD MI-250 [2] and CPUs like Intel SapphireRapids (with HBM) [9] could be effective token machines. Since we do not have access to such hardware and/or optimized LLM implementations, we leave this to future work.

Interconnect between prompt and token machines. In this work, we assume Infiniband connection between the prompt and token machines in all the designs (albeit, lower bandwidth when A100s were involved). Although this is common for all homogenous machines, Splitwise-HA is not be readily available with an Infiniband connection between H100s and A100s, even though technically feasible. The alternative could be HPC clouds, with Infiniband connections through the CPU [3], or Ethernet, using RoCE [58]. Given our optimized KV-cache transfer that helps reduce critical latency, an interconnect with $10 \times$ lower bandwidth would likely still be beneficial. To further reduce our bandwidth utilization, we could also compress the KV-cache before transferring it across the network [55].

Heterogeneous prompt/token machines. Although Splitwise is robust to varied models and input traces, we recognize that fragmenting a data center with different types of GPUs (e.g., Splitwise-HA) may bring its own challenges for the CSP.

Conversation back and forth. Chat APIs for LLMs today require the user to send the complete context of the conversation so far [18]. However, in the future, services may have enough GPU capacity to cache the context and avoid recomputation. This could sway the memory utilization pattern of the prompt phase from our characterization. Furthermore, it may require transferring the KV-cache back to a prompt machine to be ready for the next conversation request.

## VIII. RELATED WORK

Heterogeneous scheduling and dataflow systems. Prior work has studied heterogeneous scheduling for a variety of interactive services [65], [68], [83]. These works exploit hardware heterogeneity to strike a balance between different objectives such as cost, energy, and performance. However, they run the entire workload on the same machine. Research on heterogeneous multiprocessor CPU scheduling attempts to match workload heterogeneity to hardware heterogeneity [29], [40], [41], [50], [76], [80]. These works use profiling or online monitoring with metrics like request length or hardware performance counters to identify workload phases and allocate them appropriately on heterogeneous processors. However, they do not consider the complexities with batching. Distributed dataflow systems orchestrate large-scale computational graphs and aim to provide general-purpose programmability [34], [46], [75], [82]. LLM inference under Splitwise can be viewed as a static computational graph with two stages, so it could be implemented using distributed frameworks that provide efficient GPU abstractions [59]. Splitwise differs from these works since it uses a specialized two-phase design for generative LLM inference and leverages phase-aware resource management with efficient batching.

Model serving systems. LLM inference serving is a rapidly developing field, with several recent works optimizing batching [23], [25], [51], [53], [81], scheduling [22], [42], [51], [66], [73], [79], and memory usage [32], [35], [51], [74]. Prior work has also proposed using CPUs and lower compute capability devices for LLM serving [8], [12]. These approaches use the same machine for both prompt and token phase. With Splitwise, they could improve throughput and latency by splitting phases.

Prior work on video and ML serving focuses on scheduling model chains with data dependencies under latency constraints [24], [31], [43], [49], [68]. Such schedulers rely on model profiling to make efficient allocation decisions and manage requests across machines. Recommendation system inference exhibits compute/memory heterogeneity both within and across models. Prior work exploits such heterogeneity to selectively schedule requests between CPUs and accelerators [38], [52], colocate models with complementary memory usage [30], and partition compute/memory on heterogeneous hardware resources [45], [48]. Similarly, Splitwise exploits the heterogeneity within LLM inference requests. However, it uses different optimizations due to the differences in LLM workload characteristics and requirements.

## IX. CONCLUSION

We extensively characterized the prompt computation and token generation phases of LLM inference to draw out differences in their system utilization patterns. Based on our insights, we designed Splitwise to separate these phases onto different machines and enable phase-specific resource management. Using Splitwise, we explored cluster designs optimized for throughput, cost, and power, and showed that they perform well even as workloads change. Splitwise clusters under performance SLOs achieve $1.76 \times$ better throughput with $15 \%$ lower power at the same cost, or $2.35 \times$ better throughput with same the cost and power than existing designs.

## ACKNOWLEDGEMENTS

We thank the reviewers for their helpful feedback. We thank Chetan Bansal, Srikant Bhardwaj, Suriya Kalivardhan, Ankur Mallick, Deepak Narayanan, and Amar Phanishayee for insightful discussions. Pratyush Patel was partially supported by NSF CNS-2104548 and a research grant from VMware.

## REFERENCES

[1] Add Splitwise Implementation to vLLM. GitHub. [Online]. Available: https://github.com/vllm-project/vllm/pull/2809

[2] AMD Instinct ${ }^{\mathrm{TM}}$ MI250 Accelerator. [Online]. Available: https: //www.amd.com/en/products/server-accelerators/instinct-mi250

[3] Azure InfiniBand HPC VMs. [Online]. Available: https://learn.microsoft. com/en-us/azure/virtual-machines/overview-hb-hc

[4] Azure Public Dataset: Azure LLM Inference Trace 2023. GitHub. [Online]. Available: https://github.com/Azure/AzurePublicDataset/blob/ master/AzureLLMInferenceDataset2023.md

[5] CoreWeave - Specialized Cloud Provider. [Online]. Available: https://www.coreweave.com

[6] Google Assistant with Bard. [Online]. Available: https://blog.google/ products/assistant/google-assistant-bard-generative-ai/

[7] HPC Interconnect on CoreWeave Cloud. [Online]. Available: https //docs.coreweave.com/networking/hpc-interconnect

[8] Intel BigDL-LLM. [Online]. Available: https://github.com/intel-analytics/ BigDL

[9] Intel Sapphire Rapids with HBM. [Online]. Available: https://www.anandtech.com/show/17422/ intel-showcases-sapphire-rapids-plus-hbm-xeon-performance-isc-2022

[10] Microsoft Azure ND A100 v4-series . [Online]. Available: https: //learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series

[11] MSCCL++: A GPU-driven communication stack for scalable AI applications. [Online]. Available: https://github.com/microsoft/mscclpp

[12] Numenta Inference on CPUs. [Online]. Available: https://www.servethehome.com/ numenta-has-the-secret-to-ai-inference-on-cpus-like-the-intel-xeon-max/

[13] NVIDIA Accelerated InfiniBand Solutions. [Online]. Available: https://www.nvidia.com/en-us/networking/products/infiniband/

[14] NVIDIA Chip Shortage. nvidia-chip-shortages-leave-ai-startups-scrambling for-computing-powe

[15] NVIDIA DGX A100: Universal System for AI Infrastructure. [Online] Available: https://resources.nvidia.com/en-us-dgx-systems/dgx-ai

[16] NVIDIA DGX H100. [Online]. Available: https://www.nvidia.com/en-us/ data-center/dgx-h100/

[17] NVIDIA Hopper GPUs Expand Reach as Demand for AI Grows. [Online]. Available: https://nvidianews.nvidia.com/news/ nvidia-hopper-gpus-expand-reach-as-demand-for-ai-grows

[18] OpenAI ChatGPT APIs. [Online]. Available: https://openai.com/blog/ introducing-chatgpt-and-whisper-apis

[19] Power Availability Stymies Datacenter Growth. [Online]. Available: https://www.networkworld.com/article/972483/ power-availability-stymies-data-center-growth.

[20] SplitwiseSim: LLM Serving Cluster Simulator. GitHub. [Online]. Available: https://github.com/Mutinifni/splitwise-sim

[21] The New Bing. [Online]. Available: https://www.microsoft.com/en-us/ edge/features/the-new-bing?form=MT00D8

[22] TurboMind Inference Server. [Online]. Available: https://github.com/ InternLM/lmdeploy

[23] A. Agrawal, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, and R. Ramjee, "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," arXiv preprint arXiv:2308.16369, 2023

[24] H. Albahar, S. Dongare, Y. Du, N. Zhao, A. K. Paul, and A. R. Butt, "SchedTune: A heterogeneity-aware GPU Scheduler for Deep Learning," in CCGrid, 2022.

[25] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li, E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley, and Y. He, "DeepSpeedInference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale," in SC, 2022

[26] L. A. Barroso, U. Hölzle, and P. Ranganathan, "The Datacenter as a Computer: Designing Warehouse-Scale Machines," Synthesis Lectures on Computer Architecture, 2018.

[27] E. Boutin, J. Ekanayake, W. Lin, B. Shi, J. Zhou, Z. Qian, M. Wu, and L. Zhou, "Apollo: Scalable and Coordinated Scheduling for Cloud-scale Computing," in OSDI, 2014.

[28] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, "Language Models are Few-Shot Learners," arXiv preprint arXiv:2005.14165, 2020

[29] J. Chen and L. K. John, "Efficient Program Scheduling for Heterogeneous Multi-core Processors," in DAC, 2009

[30] Y. Choi, J. Kim, and M. Rhu, "Hera: A Heterogeneity-Aware Multi-Tenant Inference Server for Personalized Recommendations," arXiv preprint arXiv:2302.11750, 2023

[31] D. Crankshaw, G.-E. Sela, X. Mo, C. Zumar, I. Stoica, J. Gonzalez, and A. Tumanov, "InferLine: Latency-aware Provisioning and Scaling for Prediction Serving Pipelines," in SoCC, 2020.

[32] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and Memory-efficient Exact Attention with IO-Awareness," in NeurIPS, 2022.

[33] David Patterson. Domain Specific Architectures for Deep Neural Networks: Three Generations of Tensor Processing Units (TPUs) Allen School Distinguished Lecture. [Online]. Available: https: //www.youtube.com/watch?v=VCScWh966u4

[34] J. Dean and S. Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters," Communications of the ACM, 2008.

[35] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int80: 8-bit Matrix Multiplication for Transformers at Scale," arXiv preprint arXiv:2208.07339, 2022.

[36] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in NAACL, 2019

[37] A. Gujarati, R. Karimi, S. Alzayat, W. Hao, A. Kaufmann, Y. Vigfusson, and J. Mace, "Serving DNNs like Clockwork: Performance Predictability from the Bottom Up," in OSDI, 2020.

[38] U. Gupta, S. Hsia, V. Saraph, X. Wang, B. Reagen, G.-Y. Wei, H.-H. S Lee, D. Brooks, and C.-J. Wu, "DeepRecSys: A System for Optimizing End-to-end At-scale Neural Recommendation Inference," in ISCA, 2020.

[39] V. Gupta, M. Harchol Balter, K. Sigman, and W. Whitt, "Analysis of Join-the-Shortest-Queue Routing for Web Server Farms," Performance Evaluation, 2007

[40] M. E. Haque, Y. H. Eom, Y. He, S. Elnikety, R. Bianchini, and K. S McKinley, "Few-to-Many: Incremental Parallelism for Reducing Tail Latency in Interactive Services," ACM SIGPLAN Notices, 2015.

[41] M. E. Haque, Y. He, S. Elnikety, T. D. Nguyen, R. Bianchini, and K. S. McKinley, "Exploiting Heterogeneity for Tail Latency and Energy Efficiency," in MICRO, 2017

[42] K. Hong, G. Dai, J. Xu, Q. Mao, X. Li, J. Liu, K. Chen, H. Dong, and Y. Wang, "FlashDecoding++: Faster Large Language Model Inference on GPUs," arXiv preprint arXiv:2311.01282, 2023.

[43] Y. Hu, R. Ghosh, and R. Govindan, "Scrooge: A Cost-effective Deep Learning Inference System," in SoCC, 2021.

[44] Huggingface. Text Generation Inference (TGI). [Online]. Available: https://github.com/huggingface/text-generation-inference

[45] R. Hwang, T. Kim, Y. Kwon, and M. Rhu, "Centaur: A Chiplet-based, Hybrid Sparse-Dense Accelerator for Personalized Recommendations," in ISCA, 2020.

[46] M. Isard, M. Budiu, Y. Yu, A. Birrell, and D. Fetterly, "Dryad: Distributed Data-Parallel Programs from Sequential Building Blocks," in EuroSys, 2007.

[47] M. Javaheripi and S. Bubeck, "Phi-2: The Surprising Power of Small Language Models," Microsoft Research Blog, 2023.

[48] W. Jiang, Z. He, S. Zhang, K. Zeng, L. Feng, J. Zhang, T. Liu, Y. Li, J. Zhou, C. Zhang et al., "FleetRec: Large-scale Recommendation Inference on Hybrid GPU-FPGA Clusters," in $K D D, 2021$.

[49] R. S. Kannan, L. Subramanian, A. Raju, J. Ahn, J. Mars, and L. Tang, "GrandSLAm: Guaranteeing SLAs for Jobs in Microservices Execution Frameworks," in EuroSys, 2019.

[50] R. Kumar, K. I. Farkas, N. P. Jouppi, P. Ranganathan, and D. M. Tullsen, "Single-ISA Heterogeneous Multi-core Architectures: The Potential for Processor Power Reduction," in MICRO, 2003.

[51] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention," in SOSP, 2023.

[52] Y. Kwon, Y. Lee, and M. Rhu, "TensorDIMM: A Practical Near-Memory Processing Architecture for Embeddings and Tensor Operations in Deep Learning," in MICRO, 2019.

[53] Z. Li, L. Zheng, Y. Zhong, V. Liu, Y. Sheng, X. Jin, Y. Huang, Z. Chen, H. Zhang, J. E. Gonzalez, and I. Stoica, "AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving," in OSDI, 2023

[54] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv preprint arXiv:1907.11692, 2019.

[55] Z. Liu, J. Wang, T. Dao, T. Zhou, B. Yuan, Z. Song, A. Shrivastava, C. Zhang, Y. Tian, C. Re, and B. Chen, "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time," in ICML, 2023.

[56] Meta. Introducing the AI Research SuperCluster - Meta's CuttingEdge AI Supercomputer for AI Research. [Online]. Available: https://ai.facebook.com/blog/ai-rsc/

[57] "Azure OpenAI Service," Microsoft Azure, 2022. [Online]. Available: https://azure.microsoft.com/en-us/products/ai-services/openai-service

[58] R. Mittal, A. Shpiner, A. Panda, E. Zahavi, A. Krishnamurthy, S. Ratnasamy, and S. Shenker, "Revisiting Network Support for RDMA," arXiv preprint arXiv:1806.08159, 2018.

[59] P. Moritz, R. Nishihara, S. Wang, A. Tumanov, R. Liaw, E. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan et al., "Ray: A Distributed Framework for Emerging AI Applications," in OSDI, 2018.

[60] OpenAI. Scaling Kubernetes to 7,500 Nodes. [Online]. Available: https://openai.com/research/scaling-kubernetes-to-7500-nodes

[61] K. Ousterhout, P. Wendell, M. Zaharia, and I. Stoica, "Sparrow: Distributed, Low Latency Scheduling," in SOSP, 2013.

[62] P. Patel, E. Choukse, C. Zhang, Í. Goiri, B. Warrier, N. Mahalingam, and R. Bianchini, "POLCA: Power Oversubscription in LLM Cloud Providers," arXiv preprint arXiv:2308.12908, 2023.

[63] P. Patel, E. Choukse, C. Zhang, Í. Goiri, B. Warrier, N. Mahalingam, and R. Bianchini, "Characterizing Power Management Opportunities for LLMs in the Cloud," in ASPLOS, 2024.

[64] P. Patel, Z. Gong, S. Rizvi, E. Choukse, P. Misra, T. Anderson, and A. Sriraman, "Towards Improved Power Management in Cloud GPUs," in IEEE CAL, 2023.

[65] P. Patel, K. Lim, K. Jhunjhunwalla, A. Martinez, M. Demoulin, J. Nelson, I. Zhang, and T. Anderson, "Hybrid Computing for Interactive Datacenter Applications," arXiv preprint arXiv:2304.04488, 2023.

[66] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, J. Heek, K. Xiao, S. Agrawal, and J. Dean, "Efficiently Scaling Transformer Inference," in MLSys, 2023.

[67] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners," OpenAI blog, 2019.

[68] F. Romero, Q. Li, N. J. Yadwadkar, and C. Kozyrakis, "INFaaS: Automated Model-less Inference Serving," in USENIX ATC, 2021.

[69] T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilić, D. Hesslow, R. Castagné, A. S. Luccioni, F. Yvon, M. Gallé, J. Tow, A. M. Rush, S. Biderman, A. Webson, P. S. Ammanamanchi, T. Wang, B. Sagot, N. Muennighoff, A. V. del Moral, O. Ruwase, R. Bawden, S. Bekman, A. McMillan Major, I. Beltagy, H. Nguyen, L. Saulnier, S. Tan, P. O. Suarez, V. Sanh, H. Laurençon, Y. Jernite, J. Launay, M. Mitchell, and C. Raffel, "BLOOM: A 176B-Parameter Open-access Multilingual Language Model," arXiv preprint arXiv:2211.05100, 2022.

[70] P. Schmid. Fine-tune FLAN-T5 XL/XXL using DeepSpeed \& Hugging Face Transformers. [Online]. Available: https://www.philschmid.de/ fine-tune-flan-t5-deepspeed

[71] P. Schmid, O. Sanseviero, P. Cuenca, and L. Tunstall. Llama 2 is here - Get it on Hugging Face. [Online]. Available: https://huggingface.co/blog/llama2

[72] M. Schwarzkopf, A. Konwinski, M. Abd-El-Malek, and J. Wilkes, "Omega: Flexible, Scalable Schedulers for Large Compute Clusters," in EuroSys, 2013

[73] Y. Sheng, S. Cao, D. Li, B. Zhu, Z. Li, D. Zhuo, J. E. Gonzalez, and I. Stoica, "Fairness in Serving Large Language Models," arXiv preprint arXiv:2401.00588, 2023.

[74] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, B. Chen, P. Liang, C. Ré, I. Stoica, and C. Zhang, "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," in ICML, 2023.

[75] K. Shvachko, H. Kuang, S. Radia, and R. Chansler, "The Hadoop Distributed File System," in MSST, 2010.

[76] K. Van Craeynest, A. Jaleel, L. Eeckhout, P. Narvaez, and J. Emer, "Scheduling Heterogeneous Multi-cores Through Performance Impact Estimation (PIE)," ACM SIGARCH Computer Architecture News, 2012.

[77] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is All You Need," in NeurIPS, 2017.
[78] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. v. Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, "Transformers: State-of-the-art Natural Language Processing," in EMNLP, 2020.

[79] B. Wu, Y. Zhong, Z. Zhang, G. Huang, X. Liu, and X. Jin, "Fast Distributed Inference Serving for Large Language Models," arXiv preprint arXiv:2305.05920, 2023.

[80] H. Yang, Q. Chen, M. Riaz, Z. Luan, L. Tang, and J. Mars, "PowerChief: Intelligent Power Allocation for Multi-stage Applications to Improve Responsiveness on Power Constrained CMP," in ISCA, 2017.

[81] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun, "Orca: A Distributed Serving System for Transformer-Based Generative Models," in OSDI, 2022.

[82] M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. Ma, M. McCauly, M. J. Franklin, S. Shenker, and I. Stoica, "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing," in NSDI, 2012.

[83] C. Zhang, M. Yu, W. Wang, and F. Yan, "MArk: Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving," in USENIX ATC, 2019.

[84] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin et al., "OPT: Open Pre-trained Transformer Language Models," arXiv preprint arXiv:2205.01068, 2022.

[85] W. Zhu, "Analysis of JSQ Policy on Soft Real-time Scheduling in Cluster," in HPCAsia, 2000.
</end of paper 3>


<paper 4>
# Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads 

Cunchen $\mathrm{Hu}^{1,2 *}$ Heyang Huang ${ }^{1,2}$, Liangliang Xu ${ }^{3}$, Xusheng Chen $^{3}$, Jiang $\mathrm{Xu}^{3}$, Shuang Chen $^{3}$,<br>Hao Feng ${ }^{3}$, Chenxi Wang ${ }^{1,2}$, Sa Wang ${ }^{1,2}$, Yungang Bao ${ }^{1,2}$, Ninghui Sun ${ }^{1,2}$, Yizhou Shan ${ }^{3}$<br>${ }^{1}$ University of Chinese Academy of Sciences, ${ }^{2}$ ICT, CAS ${ }^{3}$ Huawei Cloud


#### Abstract

Transformer-based large language model (LLM) inference serving is now the backbone of many cloud services. LLM inference consists of a prefill phase and a decode phase. However, existing LLM deployment practices often overlook the distinct characteristics of these phases, leading to significant interference. To mitigate interference, our insight is to carefully schedule and group inference requests based on their characteristics. We realize this idea in TetriInfer through three pillars. First, it partitions prompts into fixed-size chunks so that the accelerator always runs close to its computationsaturated limit. Second, it disaggregates prefill and decode instances so each can run independently. Finally, it uses a smart two-level scheduling algorithm augmented with predicted resource usage to avoid decode scheduling hotspots. Results show that TetriInfer improves time-to-first-token (TTFT), job completion time (JCT), and inference efficiency in turns of performance per dollar by a large margin, e.g., it uses $38 \%$ less resources all the while lowering average TTFT and average JCT by $97 \%$ and $47 \%$, respectively.


## 1 Introduction

Since the boom of ChatGPT, large language model (LLM) based services have now played a vital role in our daily lives $[4,9,20,31,34,38]$. Behind the scenes, all use cases boil down to LLM inference serving. To run an inference request, the LLM model will first take the user inputs to generate the first token (known as the prefill phase), and then generate outputs token-by-token in an auto-regressive manner (known as the decode phase). Numerous works were proposed to improve the cost efficiency of LLM inference [21,41].

There are various ways to interact with LLM, from simple chats to more complex downstream tasks such as document summarization, content creation, etc. As a result, LLMempowered services serve inference requests with dramatically different properties that can be categorized across two di-[^0]

mensions: the input prompt length during the prefill phase and the generated token length during the decode phase. As shown in Figure 1, summarization tasks have long input prompts and short generated tokens, while context creation tasks are the opposite. Token lengths of different downstream tasks can differ by more than two orders of magnitude. Given the significant variation in LLM inference requests from various downstream tasks, the first research question we ask in this paper is how do these inference requests perform when running together?.

To answer this question, we run extensive tests that mix LLM prefill and decode requests of different lengths. Unfortunately, we have observed serious interference across all combinations. For example, mixing prefill requests could result in a 10x slowdown, combining prefill and decode requests could lead to a $5 \mathrm{x}$ slowdown, and mixing decode requests with different lengths could take a $16 \%$ throughput hit (see $\S 2.2$ ). A naive solution to avoid interference is to provision resources for each downstream task statically. Given the high cost of LLM serving infrastructure, this solution is impractical. To this end, the second research question we ask in this paper is how to build a distributed LLM inference serving system that minimizes interferences?

We take a step back to examine why interference exists. We find the fundamental issue lies in the fact that current LLM deployment practices do not account for the distinct characteristics exhibited by LLM prefill and decode phases. Specifically, the prefill phase resembles a computation-heavy batch job, with its computation scaling quadratically with the input prompt length. The decode phase resembles a memoryintensive, latency-critical task, with its resource usage scaling sublinearly with the generated token length [33]. Interferences observed in our tests are classic system problems. Running prefill requests leads to a serious slowdown because we continue adding computation-heavy jobs to an already saturated hardware (\$2.2.1). Combining prefill and decode requests hurts both because we co-run batch and latency-critical jobs simultaneously (\$2.2.2). Mixing decode requests leads to a throughput drop because we are unaware of the memory bandwidth and capacity usage, thus leading to contention and
head-of-line blocking (\$2.2.3).

To solve these issues, our insight is to carefully schedule and group requests based on their characteristics. We realize this idea in TetriInfer ${ }^{1}$, a cloud-scale LLM inference serving system designed to battle interferences.

Our designs are three-fold. First, to avoid interference running prefill, we propose limiting the number of tokens processed in a single prefill iteration so that hardware is fully utilized without incurring extra penalties. TetriInfer partitions and pads input prompts into fixed-size chunks so that the accelerator always runs close to its computation-saturated limit (§3.3). Second, to avoid interference in co-running prefill and decode, we propose disaggregating prefill from decode phases. TetriInfer has dedicated prefill and decode instances. During runtime, prefill instances transfer prefilled KV cache to decode instances. The prefill and decode instances are virtual concepts in that each can scale independently and flip roles if load changes (\$3.5). Third, to avoid interference running decode requests, we propose using a smart two-level scheduling algorithm augmented with predicted resource usage to avoid scheduling hotspots (\$3.4). TetriInfer incorporates an LLM-based length prediction model to speculate the number of generated tokens of decode requests, and then schedule them accordingly.

We implement TetriInfer's disaggregated prefill and decode instances based on vLLM [21]. Most of our modules are implemented in Python, except for the network stack module, which utilizes C++ to interface with low-level APIs for KV cache transfer. The fine-tuning part uses Trainer APIs offered by HuggingFace Transformer [16]. Since we cannot access high-end hardware, we implement a mock mechanism to emulate varying network bandwidth connecting prefill and decode instances, as illustrated in Figure 9.

We compare TetriInfer with vanilla vLLM using public dataset [35] in terms of time-to-first-token (TTFT), job completion time (JCT), and efficiency as in performance per dollar (perf/\$). We run them atop a real testbed with emulated network bandwidth ranging from 200Gbps to 300GBps. For light prefill and heavy decode workload, TetriInfer improves perf/\$ by $2.4 x$ (Figure 12). For common mixed workload, TetriInfer improves average TTFT and average JCT by $85 \%$ and $50 \%$, respectively (Figure 15). Nevertheless, we also find that TetriInfer's design is not ideal for heavy prefill and heavy decode workloads since the room for improvement is marginal, and the overhead we introduce cannot be offset (Figure 14). Overall, our ideas mentioned above are effective. TetriInfer achieves effective LLM inference serving, outperforming vLLM by a large margin in TTFT, JCT, and perf/\$ running most common workloads (\$5.1).[^1]![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-02.jpg?height=346&width=866&top_left_y=252&top_left_x=1081)

Figure 1: Length Distribution. Prompt Tokens for Prefill and Generated Tokens during Decode. Data sources: conversation [35], summarization [17], writing [18].
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-02.jpg?height=304&width=874&top_left_y=848&top_left_x=1080)

Figure 2: Prefill and Decode's Characteristics. Decode's GPU utilization fluctuates because the task is faster than our monitoring granularity.

## 2 Background and Motivation

We present a brief primer on LLM inference and study interferences while running various LLM inference requests to motivate our work. For model and testbed details, see $\S 5$.

### 2.1 Generative LLM Inference

LLM inference is a process that involves generating a sequence of output tokens in response to an input prompt. This process consists of two phases: prefill and decode. The prefill phase outputs the first token and generates the key and value cache (KV cache) for future decoding [21]. The decode phase uses the previous $\mathrm{KV}$ cache to generate new tokens stepby-step in an auto-regressive manner. Generally, the prefill phase is computation-bound, and the decode phase is memorybound [33]. We report this in Figure 2. Results indicate that the prefill phase's throughput stays flat once the accelerator is saturated at a certain number of tokens (which we name the accelerator-saturate threshold). The decode phase's throughput continues increasing with a larger batch size but plateaus once the memory bandwidth is saturated.

### 2.2 Motivation: Interference Study

This section studies the impact of running different inference requests concurrently. Inspired by Figure 1, we classify in-

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-03.jpg?height=304&width=832&top_left_y=257&top_left_x=186)

Figure 3: Interference of Prefill \& Prefill. LP means light prefill. HP means heavy prefill. (a) and (b) indicate that light prefill's prefill latency increases as the number of co-running requests increases. The same applies to (c)'s heavy prefill.

ference requests across two dimensions (prefill and decode length) and one property (light or heavy), resulting in four distinct request types: heavy prefill, light prefill, heavy decode, and light decode. Here, heavy refers to a long token length, while light refers to a short token length. Below, we study mixing prefill and prefill (\$2.2.1), prefill and decode (§2.2.2), decode and decode (\$2.2.3).

### 2.2.1 Prefill and Prefill

We first study mixing prefill requests. Here, light prefill has roughly 18 prompt tokens as it is the median token length in ShareGPT's short prompts [35], while heavy prefill has 512 prompt tokens as the accelerator is saturated at this length in our testbed (see Figure 2). In Figure 3 (a) and (b), we show how a light prefill's latency changes if it co-runs with other light prefill and heavy prefill requests. We find its latency increases by $2 x$ and $8 x$ if there are 7 and 63 concurrent light prefill requests in the same batch. Additionally, it incurs more than 10x latency slowdown if it runs with other heavy prefill requests. In Figure 3 (c), we show that heavy prefill's latency also incurs a 3x slowdown if co-run with other light prefill requests. Overall, we find that when the total number of tokens in a batch is larger than the accelerator-saturate threshold, the prefill latency dramatically increases.

### 2.2.2 Prefill and Decode

We now study mixing prefill and decode in the same batch due to continuous batching [21,45]. Both light prefill and heavy prefill follow $\S 2.2 .1$ 's definition. Also, light decode refers to the ones that generate a small number of tokens, e.g., less than 100. heavy decode refers to the ones that generate a large number of tokens, e.g., larger than 512 tokens. Though decoding latency increases slowly with an increasing number of generated tokens, we only present tests related to light decode as heavy decode presents similar results.

In Figure 4 (a) and (b), we show how a light decode's periteration decoding latency changes if it co-runs with light prefill and heavy prefill requests. Results indicate that its

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-03.jpg?height=322&width=436&top_left_y=259&top_left_x=1083)

(b) 1-LD \& N-HP

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-03.jpg?height=312&width=420&top_left_y=581&top_left_x=1086)

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-03.jpg?height=326&width=439&top_left_y=257&top_left_x=1510)

(d) $1-H P \& N-L D$

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-03.jpg?height=317&width=442&top_left_y=584&top_left_x=1508)

Figure 4: Interference of Prefill \& Decode. LD means light decode. HD means heavy decode.
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-03.jpg?height=358&width=876&top_left_y=1092&top_left_x=1080)

Figure 5: Interference of Decode \& Decode.

decoding latency increases by $5 x$ even if only one other heavy prefill request is in the same continuous batch! In Figure 4 (c) and (d), we show how a light prefill's and a heavy prefill's latency changes if they co-run with other light decode requests. Figure 4 (c) indicates that the prefill latency increases once the number of co-running light decode requests is more than 7. Both slow down roughly by up to $2.5 x$.

### 2.2.3 Decode and Decode

We now study mixing decode requests. Following $\$ 2.2 .2$ 's definition and ShareGPT's distribution, we set both requests to use very short prompts. And light decode generates roughly 20 to 100 tokens, while heavy decode generates more than 512 tokens. Figure 5 presents the decoding throughput and latency while running different numbers of light decode and heavy decode requests in the same batch. Results suggest that compared to a batch with all light decode requests, increasing heavy decode requests could seriously hurt throughput and latency. For example, with a batch size of 128 , compared to a batch with all light decode, a batch with half heavy decode and half light decode's throughput drops by $16 \%$ while the
latency increases by $23 \%$.

### 2.3 Analysis and Insights

We have observed significant interferences in LLM inferencing. The root cause is simple: current LLM systems are ignorant of the distinct characteristics exhibited by LLM prefill and decode phases. The prefill phase resembles a computationheavy batch job, while the decode phase resembles a memoryintensive, latency-critical task [33].

Interferences measured above are classic system problems. In $\S 2.2 .1$, running prefill requests leads to a serious slowdown because we continue adding computation-heavy jobs to an already saturated hardware. In §2.2.2, mixing prefill and decode requests hurts both because we co-run batch and latency-critical jobs at the same time. In $\S 2.2 .3$, mixing decode requests leads to a throughput drop because we are unaware of the memory bandwidth and capacity usage, thus leading to contention and head-of-line blocking.

Our work aims to solve these issues by carefully schedule and group requests based on their characteristics. Our ideas are three-fold. First, to avoid interference running prefill, we propose limiting the number of tokens processed in a single prefill step so that hardware is fully utilized without incurring extra penalties. Second, to avoid interference co-running prefill and decode, we propose disaggregating prefill from decode so that each runs independently. Third, to avoid interference running decode requests, we propose to use a smart two-level scheduling algorithm augmented with predicted resource usage to avoid scheduling hotspots. We visualize the comparison in Figure 6 (a).

## 3 Design

### 3.1 Overview

We realize the above insights in TetriInfer, an LLM inference serving system designed to battle interferences. First, we run prefill in a fixed-size computation unit by partition and pad input prompts into fixed-size chunks such that the accelerator always runs close to its computation-saturated limit (§3.3). Second, we design instances dedicated to running the prefill or decode phases. We schedule prefill requests to prefill instances only, and the same goes for decode requests. Prefill instances will transfer prefilled KV cache to decode instances. Our prefill and decode instances are virtual concepts in that each can scale independently and flip roles if load changes (§3.5). Finally, we design a two-level scheduling algorithm for both prefill and decode request scheduling. We incorporate a length-prediction model to speculate decode requests' resource usage and then schedule them accordingly (\$3.4).

We show TetriInfer's architecture in Figure 6 (b) with four modules highlighted: centralized control plane, prefill instance, decode instance, and length prediction model.
Centralized control plane. It consists of a global scheduler and a cluster monitor. The global scheduler sends requests to prefill instances based on load and receives streaming outputs from decode instances. The cluster monitor collects statistics from prefill and decode instances and regularly broadcasts load information to prefill instances. It adds, removes, and flips prefill or decodes instances.

Prefill Instances. They only run the prefill phase of an LLM inference request. Each prefill instance has a local scheduler, a length predictor, the main LLM engine, and a dispatcher. All requests undergo four steps. First, the local prefill scheduler sorts requests based on pre-defined policies. Second, the length predictor runs a prediction model to speculate the requests' decode lengths, which are then used to estimate resource usage during the decoding phase. Third, the main LLM engine partitions all requests into fixed chunks. Finally, for each request, the dispatcher runs an inter-decode load-balancing algorithm to select a decode instance and then forwards the generated KV cache to it.

Decode instances. They are virtually disaggregated from prefill instances and only run the decode phase of an LLM inference request. Each decode instance can receive requests from any prefill instance. It runs a local scheduler with three pre-defined policies for selecting decode requests to run in the main LLM engine.

Length Prediction Model. The prediction model is a small LLM model fine-tuned offline for predicting the generation length of LLM inference requests. TetriInfer's prefill dispatcher and decode instance's local scheduler utilize the speculated information to schedule decode instances and avoid hotspots measured in $\S 2.2 .3$. The prediction model is small and deployed at all prefill instances.

### 3.2 Control Plane

TetriInfer has a centralized control plane to manage inference clusters at the cloud scale. It consists of a cluster monitor that manages the lifecycle of prefill and decode instances and a global scheduler that managesthe lifecycle of inference requests. The centralized control plane is a distributed system without a single point of failure or processing bottlenecks.

The cluster monitor is responsible for collecting and broadcasting statistics and scaling instances. Both prefill and decode instances regularly send their load information to the cluster monitor (e.g., every $100 \mathrm{~ms}$ ). Since we run decentralized decode request scheduling at prefill instances, the cluster monitor will aggregate decode instances' load information and broadcast it to all prefill instances.

The global scheduler is responsible for forwarding inference requests from external services to prefill instances and sending inference outputs from decode instances back to external services in a streaming fashion. The global scheduler maintains a request status table, which stores requests' arrival time, current phase (e.g., prefill or decode), SLA requirement,

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-05.jpg?height=686&width=675&top_left_y=240&top_left_x=213)

(a) Execution Timeline

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-05.jpg?height=686&width=1006&top_left_y=237&top_left_x=885)

(b) Architecture

Figure 6: TetriInfer's Workflow and Architecture. (a) compares existing systems and TetriInfer's execution timeline. The existing systems part has two nodes running mixed prefill and decode. TetriInfer has separated prefill and decode instances. This allows for the load-balancing of long decoding tasks across different on-demand decode instances. Each timeline comprises four rounds (R1 to R4), with the length of prefill and decode boxes representing their sequence length and the width of the decode box indicating its resource usage. A wider decode box indicates the presence of lengthy generated tokens, resulting in larger resource usage and decoding latency. (b) shows TetriInfer's architecture with four core modules highlighted.

etc. When a request arrives, the global scheduler will choose a prefill instance with the least load and then insert the request into the table. Following our insight to disaggregate prefill and decode instances, the global scheduler only decides which prefill instance will handle the request. It is up to the prefill instance's dispatcher to decide which decode instances to use with a speculated resource usage.

### 3.3 Prefill Instance

The prefill instance runs the prefill phase of an inference request. To avoid interference among prefill requests, we use a prefill scheduler and chunked prefill to sort and partition all prompts into fixed-size chunks. To help avoid interference during the decode phase, we run a length predictor and a decentralized dispatcher to choose decode instances based on speculated resource usage.

### 3.3.1 Prefill Scheduler

The prefill instance's scheduler is crucial for improving the prefill phase's latency and throughput. The scheduler maintains a raw request queue that stores requests from the global scheduler and a scheduled queue that stores sorted requests. In this work, we have designed and implemented three scheduler policies: first-come-first-serve (FCFS), shortest-job-first $(S J F)$, and longest-job-first $(L J F)$. We can use the latter two policies because we can accurately estimate a request's prefill time based on the number of tokens in its prompt. We only explore non-preemptive policies, though chunked prefill (described soon) has opened the door to preemptive and out-of-order prefill scheduling, such as shortest-remainingtime-first, which we leave for future work.

The scheduled requests are sent to the length predictor which executes scheduled requests as-is using fixed-size batch (\$3.3.2), and the main LLM which uses chunked prefill (§3.3.3). In Figure 7, we illustrate the above three scheduler policies and how scheduled requests are partitioned and merged into fixed-size chunks. Specifically, FCFS keeps the original request arrival order. Prompt tokens are partitioned and merged into chunks sequentially. This policy is the easiest to implement and works best for inference requests with similar prompt lengths. However, FCFS can lead to head-of-line blocking and high average job completion time (JCT) when requests have long prompts. This is problematic since the length differences among LLM inference requests are more than three orders of magnitude (see Figure 1).

In response, we add the shortest-job-first (SJF), and longestjob-first (LJF) to overcome these issues. These two policies schedule prefill requests based on prompt token lengths in ascending or descending order. By design, they can achieve lower JCT compared to FCFS. Nevertheless, they are no panacea. They introduce starvation for either long or short re-

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-06.jpg?height=252&width=700&top_left_y=243&top_left_x=257)

Figure 7: Prefill Scheduler Policies. The left shows four raw inference requests (R1 to R4). The right shows scheduled requests using FCFS, SJF, and LJF. We show the chunked version to illustrate slicing and merging ( $\mathrm{C} 1$ to $\mathrm{C} 4$ ).

quests. To avoid starvation, we propose using a prefill scheduling batch (i.e., PrefillSchedBatch) variable to control how many inference requests can be scheduled at a time. For example, assume the raw request queue has twenty requests awaiting scheduling. If we set the batch size to ten, we will schedule twice, each with ten requests sorted and put into the scheduled queue. This simple mechanism prevents starvation during the prefill phase.

Our scheduler is effective. Results in Figure 16 show that SJF lowers average prefill waiting time by $7.8 \%$ compared to FCFS when the batch size is set to 16 . Additionaly, the improvement is even more pronounced with larger batch sizes.

### 3.3.2 Length Predictor

To address the interference cases measured in $\S 2.2 .3$, it is essential to determine the number of tokens that a decode request is likely to generate. This information will enable us to schedule decode requests in a length-aware manner. As such, the prefill instance runs a length predictor to predict the length range of an inference request's generated tokens. The prefill instance's dispatcher utilizes this information for inter-decode instance scheduling (\$3.3.4), while the decoding instance's local scheduler employs this information for intradecode instance scheduling (\$3.4).

Our length predictor uses a small LLM-based classification model called a "predict model" to classify the length of generated tokens into fixed-size buckets if the request were executed by a specific target LLM model. The predict model is intentionally small, containing millions of parameters while the target model is much larger, with billions of parameters. As we run the length predictor at the prefill instance, we aim to minimize its cost and avoid impacting the main LLM model. Therefore, approaches like using a giant LLM to predict length are not feasible for us [48]. Fortunately, a small LLM model is much faster than a giant LLM and uses much less resources. For example, we use OPT-125M as the predict model and OPT-13B as the target model, the small one is roughly ten times faster than the larger one.

We opt to predict the length range instead of an exact number of tokens because the latter is extremely difficult to predict. Various inference parameters, such as temperature and top- $\mathrm{p}$ [3], result in significant response variations from the same LLM model to the same question in practice. Since our primary goal is to use the estimated length to guide our request scheduling decisions, an exact length estimation is unnecessary; a length range suffices. For instance, if we estimate the length to be between ten to twenty tokens, we can deduce its resource usage's lower and upper bounds.

In this work, we have tested two execution modes: a sequential mode, where we first execute the predict model followed by the target model, and a parallel mode, where both models are run simultaneously. The sequential mode adds extra latency for the target LLM model, while the parallel mode may reduce the target LLM model's throughput. Based on our findings in Figure 17, we opted to use the parallel mode because the main LLM is not affected for most requests (more than $80 \%$ ), though throughput take a $10 \%$ hit under extreme stress test.

Figure 8 outlines the offline fine-tuning and online prediction workflow. In this process, the predict model (depicted in red) is trained to speculate the decoding behavior of a specific target model (depicted in blue). The fine-tuning of the predict model involves three key steps. Firstly, we assemble a promptonly training dataset inherited from public datasets, a large target LLM model (e.g., OPT-13B), and a classification model for our predict model (e.g., 125M OPTForSequenceClassification [16]). Secondly, we send training prompts to the target LLM model, which generates responses. Subsequently, we categorize the generated responses into fixed-size buckets with a chosen granularity. For instance, using a granularity of 100 , responses with token lengths between 0 to 200 are labeled with $0,200-400$ are labeled with 1 , and so on. These labels are paired with the training prompts to create a new dataset. Lastly, we partition the new dataset into a training section and an evaluation section and then proceed to train and evaluate the predict model using this dataset.

The length range granularity plays a crucial role. If set to one, we fall back to predicting an exact number of tokens, which is not practical. If set to target model's context window size (e.g., $2 \mathrm{~K}$ ), we fall back to no prediction at all and could run into interferences reported in §2.2.1. Intuitively, a smaller granularity means more accurate resource and performance estimation but lower accuracy in practice. A larger granularity means higher accuracy but essentially makes scheduling harder. Regardless of granularity, it's easy to calculate resource usage's upper and lower bound but not performance. In this work, we can predict a granularity of 200 tokens with $74.9 \%$ accuracy. Since improving prediction accuracy is not the focus of this work, we leave it for future work.

Discussions. We run the length predictor at each prefill instance, hence prefill instances can make well-informed decisions on which decode instances should have enough resources to run certain decoding requests. Nevertheless, we identify two alternative designs. The first design is to run the length predictor at each decode instance. As a result, the

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-07.jpg?height=315&width=336&top_left_y=241&top_left_x=233)

Training $\Rightarrow$ Evaluation

Predict Model

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-07.jpg?height=453&width=393&top_left_y=237&top_left_x=584)

Figure 8: Predict Model's Fine-tuning and Prediction Flow. The target model is the one that we want to predict its decoding behavior. The predict model is the one we train. This work does not explore online fine-tuning.

prefill instance can only schedule requests based on the load of decoding instances. However, this design cannot avoid interference cases we measured in $\$ 2.2 .3$. Indeed, one could migrate interference requests among decoding instances at runtime based on predicted length. This would be an overly complex solution. The second design is to run the length predictor at the global scheduler before dispatching requests to refill instances. This design could make the global scheduler a bottleneck. We believe our current design is easier and simpler to reason about and deploy compared to alternatives.

### 3.3.3 Chunked Prefill

After the prefill scheduler, we concurrently execute the prefill phase of the main LLM alongside the length predictor. We employ fixed-size chunks for the LLM prefill rather than using fixed batch sizes [21].

As demonstrated in $\$ 2.2 .1$, we observe that as the number of tokens in a prefill iteration increases, the accelerator's throughput remains constant, while the latency continues to rise after reaching a certain threshold. We refer to this threshold as ChunkSize. Compared to the traditional fixed batch size approach, running prefill in ChunkSize allows for the optimal utilization of accelerators without incurring additional latency penalties. The accelerator and the LLM model architecture determine the ChunkSize. Models with larger hidden dimensions and accelerators with lower capabilities typically result in a smaller ChunkSize. For example, in our test environment, the value is 512 tokens for OPT 13B.

Figure 7 illustrates how chunked prefill works for different scheduler policies. For scheduled requests, we first slice and then merge prompt tokens into fixed-size chunks without altering their order. The final chunk in a batch could be partial, and we will pad it to ChunkSize with zeros. Then, we invoke the main LLM model to execute prefill forward one chunk at a time. To record progress, we maintain a simple variable per request that records the last prefilled token position.
The benefits of using chunked prefill and various prefill scheduler policies are substantial. In Figure 16, we compare vanilla vLLM, which uses fixed batch size for prefill, against TetriInfer which uses chunked prefill along with FCFS, SJF, and LJF. Chunked prefill with FCFS lowers average prefill latency by $86.4 \%$. Additionally, we avoid the interference cases measured in $\S 2.2 .1$ as heavy prefill requests are broken into fixed-chunks and the accelerator is best utilized.

Discussion. (1) An early work, Sarathi [1], has also proposed chunked prefill for the same purpose, where they utilize prefill-decode-mixed chunks. In contrast, our approach involves running prefill-only chunks as we disaggregate LLM's prefill and decode into separate instances. (2) Our length predictor utilizes a small LLM model for prediction and continues using fixed-size batching instead of chunked prefill. This is due to the model's small size, which does not exhibit a clear compute-saturate threshold as seen in larger models.

### 3.3.4 Dispatcher

The prefill instance's final module is the dispatcher, which carries out two essential steps for each prefilled request. First, it runs an inter-decode instance scheduling algorithm to select a decode instance and then transmits the prefilled KV cache to the chosen instance. The dispatcher runs on an eventdriven basis, running whenever there are prefilled requests (or chunks). The disaptcher plays a vital role in mitigating decode and decode interferences as measured in $\S 2.2 .3$.

Once a request's initial chunk is prefilled, the dispatcher invokes a decentralized load-balancing algorithm to select a decode instance with sufficient resources to run this request's decode phase. Our algorithm consists of three steps. First, we categorize decode instances into two sets: $\alpha$, those with enough resources to execute the chosen request, and $\beta$, those without. Recall that the prefill instance has all decode instances' load information broadcasted from the cluster monitor (§3.2). With the predicted length range (\$3.3.2), finding decode instances with adequate resources for executing this request's decode phase is easy. Second, we use the powerof-two [25] algorithm to choose two instances from the $\alpha$ set randomly. Lastly, from the two instances, we choose the one that would encounter the least interference if the prefilled request is sent to it. Based on Figure 5, our goal is to establish the lowest average ratio of heavy decode:light decode, which means we need to spread heavy decode requests evenly. Figure 19 proves our algorithm is effective, achieving the lowest total decoding time compared to other policies.

Once a decision is made, the dispatcher sends this request's metadata and prefilled KV cache to the selected decode instance. Crucially, there are two key design considerations for transferring KV Cache: (1) transfer granularity and (2) network stack for transferring the cache.

We begin by discussing granularity. Due to our use of chunked prefill, the prefilled $\mathrm{KV}$ cache is created in chunks

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-08.jpg?height=412&width=696&top_left_y=239&top_left_x=259)

Prefill Instance

Decode Instance

| Data Link | Two-sided Software | One-sided Software |
| :--- | :--- | :--- |
| Direct | Collective Libraries | Low-level APIs |
| Direct-NIC | IBV Verbs | IBV Verbs |
| Indirect | TCP/IBV Verbs | N/A |

Figure 9: Data Links and Network Stacks. The Direct link provides bandwidth in the hundreds of GBs (e.g, NVLink 900GBps). The Direct-NIC and Indirect link offer bandwidth in the hundreds of Gbs (e.g, ConnectX-6 200Gbps). The 1sided stack means the sender accelerator can transmit data to the receiver accelerator without involving the receiver's CPU.

(§3.3.3). As a result, we have the option to transfer the KV cache either at a chunk-level, sending each chunk's KV cache as it is generated, or at a request-level, sending the aggregated $\mathrm{KV}$ cache for a request until all of its chunks are prefilled. Utilizing chunk-level transfer enables us to parallelize chunked prefill and $\mathrm{KV}$ cache transfer, while request-level transfer allows us to minimize the number of network transfers by sending larger data. A concurrent work [32] has proposed layer-wise KV cache transfer, which aligns with our chunklevel approach. Combining their layer-wise approach with our chunk-level transfer could further optimize compute and network parallelization. In this work, we only implement request-level transfer for simplicity and leave the chunk-level transfer to future work.

We now delve into the network stack. Once the main LLM completes its prefill phase, the prefilled $\mathrm{KV}$ cache is generated at the accelerator's memory (e.g., GPU's HBM). Our goal is to transmit this cache to the selected decode instance' accelerator memory, regardless of the hardware platforms on which our system is deployed. This is challenging as multiple physical data links exist between the prefill and decode instances, each requiring different software stacks. We classify existing physical data links into three types, as shown in Figure 9. The first is called Direct, where accelerators have a directly connected high-speed link such as NVLink [40] or HCCS [13]. We can use low-level memory copy primitives [14,26] or collective libraries [28] to transmit data over these links. The second is called Direct-NIC, in which accelerators communicate via their companion NICs. We can use custom-built libraries [27] to transmit data over PCIe and Ethernet (or Infiniband). The third is called Indirect, where there is no direct link, and the accelerators must bounce data via their companion CPU DRAM, incurring extra memory copies. In Figure 9, we also categorize network stacks that utilize aforementioned data links into one-sided and two-sided, similar to RDMA's classification. Accelerators like GPU or NPU can do one-sided memory access as they have low-level primitives such as direct memory copies between devices $[14,26]$.

To navigate the complicated physical data links and ensure that TetriInfer can always use the most performant link once deployed, we design a unified network transfer abstraction to utilize the different network stack options listed in Figure 9. The stack exposes APIs such as send, receive, read, write, etc. Our dispatcher calls these APIs to transmit the KV cache to remote decode instances.

Discussion. We identify two unexplored research questions. The first question pertains to whether it is beneficial to simultaneously utilize multiple data links for transmitting the KV cache. While this approach could enhance performance, it may also introduce complex control logic. The second question involves the sender accelerator accessing the memory of the receiver accelerator without involving the receiver's CPU. This scenario raises typical challenges associated with building large-scale RDMA-based memory systems [10, 12]. Unfortunately, we cannot explore either of these ideas in this work due to limited access to high-end hardware.

### 3.4 Decode Instance

The decode instance runs the decoding phase of an inference request. As shown in Figure 6, it includes a receiver module, which is part of the unified network transfer module, a local scheduler, and an LLM for decoding. The processing steps are straightforward. The receiver module accepts requests transmitted from remote prefill instances and waits for prefilled KV caches to be received before adding them to the local scheduler's queue. The scheduler uses continuous batching to group dynamic-sized batches and invokes the LLM for decoding in an auto-regressive fashion. We implement TetriInfer based on vLLM [21] (see §4). Hence, it manages the $\mathrm{KV}$ cache in pages rather than reserved for the maximum context length $[29,45]$.

With the predicted length information sent from the prefill instance, we propose two working-set-aware scheduling policies in addition to vLLM's vanilla one. vLLM's existing policy schedule requests in a greedy fashion. As long as the accelerator has spare memory, it will add requests to the current iteration. However, it may run out of memory in future iterations and cause thrashing. Fundamentally, it is oblivious to the working set size.

To address this limitation, we propose reserve-static and reserve-dynamic policies, both aim to prevent triggering swaps. Under the reserve-static policy, a request is scheduled
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-09.jpg?height=336&width=796&top_left_y=247&top_left_x=208)

Figure 10: Prefill and Decode Instance Flip.

only if its predicted memory usage is smaller than the available accelerator memory for the current iteration. In contrast, the reserve-dynamic policy takes a more proactive approach by considering the predicted number of remaining tokens. Specifically, a new request is added to the scheduled batch only if there is still spare memory when the shortest remaining job in the batch finishes. This approach effectively mitigates memory thrashing while maximizing the advantages of paging. Our tests in Figure 18 suggest that with our current prediction accuracy, these two policies are on par with vLLM's greedy policy. When the prediction accuracy increases, these two policies can lower the average JCT by roughly $10 \%$.

### 3.5 Instance Flip

TetriInfer scales out by allocating more hardware resources. Additionally, TetriInfer can also dynamically adjust the number of prefill and decode instances within fixed hardware resources. This is crucial as LLM inference workloads have huge variations regarding prefill and decode needs (see Figure 1), and we cannot statically provision the ratio of prefill and decode instances in advance. Below, we will describe our policy and mechanism to flip a prefill instance to become a decode instance and vice versa.

Policy. As we described in $\S 3.2$, the centralized control plane oversees all instances and has the latest load information. We design a transition watcher module that regularly checks load and decides whether certain instances should be flipped. Various policies can be plugged in, such as flipping an instance if its load has been under $10 \%$ for the past minute.

Mechainsm. Once an instance is selected, we will go through the steps depicted in Figure 10. To flip a prefill instance, the global scheduler stops forwarding requests and then sends a flip request to it. The prefill instance will wait until all queued requests are drained. Then, we flip the instance. Flipping a decode instance is slightly more complex. The global scheduler notifies all prefill instances to stop forwarding requests to the selected decode instance and notifies the decode instance to complete flipping. Note that the actual flipping is fast and simple. It involves changing an internal variable without restarting the process or reloading models. In our current implementation, both instance flips take roughly 5 to $7 \mathrm{~ms}$, excluding the dynamic draining time.

## 4 Implementation

We implement TetriInfer's centralized control plane from scratch in Python. We adopt prefill and decode instances based on vLLM [21]. Most of our core modules are implemented in Python, except for the unified network stack, which utilizes C++ to interface with low-level APIs and IB Verbs for network transfer. Additionally, we implement a sharedmemory-based communication mechanism that enables fast command transfer across Python and $\mathrm{C}++$ languages. The fine-tuning part uses Trainer APIs offered by HuggingFace Transformer [16].

A prefill or decode instance is a single deployable unit consisting of two processes when deployed. For prefill, it has a Python process that runs the scheduler, length predictor, and the main LLM, as well as a C++ process that runs the dispatcher and the network stack. For decode, it has a Python process for running the scheduler and the main LLM, along with a $\mathrm{C}++$ process that runs the network stack.

Due to limited high-end hardware availability, our current implementation only supports the Indirect type using sockets (see Figure 9). In order to evaluate TetriInfer's performance across different hardware configurations, we have implemented a mock mechanism to emulate varying network bandwidth. This mechanism works as follows: for a given set of requests, we initially run their prefill phase offline to obtain their prefilled KV cache. Before testing, we load these prefilled KV caches into the decode instance's local memory. When testing starts, the prefill instance transmits only the request metadata to the decode instance, excluding the actual prefilled KV cache. Subsequently, the decode instance calculates the latency of the KV cache transfer and waits accordingly. This latency is calculated given a specific model architecture and the hardware bandwidth we aim to emulate.

## 5 Evaluation

We evaluate TetriInfer using public dataset [35] and report time-to-first-token (TTFT), job completion time (JCT), and efficiency as in performance per dollar (perf/\$).

Our testbed consists of four NVIDIA V100 GPUs, each with 32GB HBM. All GPUs are plugged into a single server with Xeon Gold 5218R CPU and 256GB DRAM. For the large LLM, we run OPT-13B [47]. For the length prediction model, we use OPT-125M. We compare with vLLM [21]. Since we adopted TetriInfer after vLLM, both systems manage KV caches in pages. Unlike TetriInfer, vanilla vLLM tightly couples prefill and decode phases.

### 5.1 End-to-End Performance

This section compares TetriInfer with vanilla vLLM using end-to-end benchmarks. We emulate TetriInfer atop two hardware setups using the mock mechanism described in $\S 4$. The
first is $T S-R o C E$, assuming prefill and decode instances communicate over 200Gbps RoCE (Direct-NIC in Figure 9). The second is called $T S$-NVLink, assuming instances communicate over 300GBps NVLink (Direct in Figure 9). Both setups are adopted from commercial V100-based servers.

For all tests, the prefill instance's scheduler uses the SJF policy as it has the best performance with the PrefillSchedBatch set to 16 (see Figure 16). For the interdecode instance scheduling, we use the decentralized loadbalancing algorithm as it outperforms other policies. For the intra-decode instance scheduling, we use the reserve-dynamic policy atop paging as it could outperform vLLM's greedy policy (see Figure 18). We flip an instance once it becomes idle for a minute using mechanisms described in $\S 3.5$.

To understand how these systems perform under mixed downstream inference tasks, we run five different types of workloads as presented in Figure 1: Heavy Prefill with Light Decode (HPLD), Heavy Prefill with Heavy Decode (HPHD), Light Prefill with Heavy Decode $(L P H D)$, Light Prefill with Light Decode (LPLD), and Mixed. Akin to the configuration used in $\S 2.2$, prefill requests that have more than 512 prompt tokens are categorized as heavy, and others are light. Decode requests with more than 128 tokens are categorized as heavy as ShareGPT answers' median length is 128 . We generate these workloads using samples from the ShareGPT dataset [35], following the distribution illustrated in Figure 1.

For each workload, we compare all systems across three key metrics: TTFT, JCT, and resource usage time (i.e., cost). Comparing TTFT indicates whether TetriInfer's prefill scheduler and chunked prefill are effective. Comparing JCT indicates whether TetriInfer's disaggregated prefill and decode design and two-level decode scheduling are effective. A natural question that centers around our disaggregated design is cost. Intuitively, TetriInfer uses two times the resources compared to vLLM's prefill-decode-coupled setting. Our results suggest otherwise. Two factors contributed: first, TetriInfer's prefill and decode run faster; second, TetriInfer can recycle or flip instances to reduce waste. Below, resource usage time represents the aggregated wall time that the prefill and decode instances use to run a particular workload. For example, the resource usage time is 3 seconds if we run prefill in in 1 second and decode in 2 seconds. For vLLM, it is the total runtime since it couples prefill and decode.

We now present each workload's results.

Light Prefill and Light Decode. Generally, LPLD represents the chat workload. We test LPLD in Figure 11 using 128 requests. When comparing TetriInfer to vLLM, we reduce average TTFT by $44 \%$ and average JCT by $40 \%$ for both emulated hardware setups. Despite using twice the number of hardware cards, TetriInfer completes tasks almost twice as fast, resulting in resource usage time that is comparable to the vanilla vLLM. Thus, we improve perf/\$ by $1.4 x$.

Light Prefill and Heavy Decode. Generally, LPHD represents the content creation workload. We test LPHD in Fig- ure 12 using 128 requests. Surprisingly, TetriInfer improves average TTFT by $97 \%$ despite using short prompts. This is because vLLM's prefill incurs serious interference while running prefill and decode requests in the same batch; in contrast, TetriInfer disaggregates them into separate instances. Additionally, with variable decode batch size over vLLM's fixed batch size during the decode phase, TetriInfer improves average JCT by $47 \%$ while using $38 \%$ less total hardware resources. Overall, we improve perf/\$ by $2.4 x$.

Heavy Prefill and Light Decode \& Heavy Prefill and Heavy Decode. HPLD and HPHD represent summarization or prompt engineering types of workloads. Both have long prompt tokens. This means TetriInfer faces two challenges: (a) large prefilled KV caches and (b) the main LLM may be impacted by the prediction model (roughly $10 \%$ as shown in Figure 17). Nevertheless, in Figure 13, we can see that TetriInfer still improves average TTFT and average JCT by $9 \%$ and $23 \%$, respectively, but at the cost of $43 \%$ increase in resource usage. vLLM outperforms TetriInfer in terms of perf $/ \$$ by $14 \%$. As Figure 14 shows, with heavy decode, TetriInfer's TTFT improvement is more pronounced because we disaggregated heavy decode from prefill, akin to Figure 12. We improve the average JCT by $19 \%$ at the cost of $7 \%$ more resources, improving perf/ $\$$ by $1.1 \mathrm{x}$.

Mixed. The last workload is a mix of all the above workloads, randomly sampled from the ShareGPT dataset. This is the case where a cluster is running all kinds of requests. In Figure 15, we run 128 requests, TetriInfer lowers average TTFT, average JCT, and resource usage by $85 \%, 50 \%, 21 \%$, respectively, improving perf/\$ by $1.9 \mathrm{x}$.

Takeaways. (1) For most LLM inference workloads, TetriInfer improve average TTFT, average JCT, resource use time, and most importantly, perf/\$ by a large margin. (2) Disaggregating prefill from decode into two distinct instances significantly improves TTFT and efficiency by minimizing interference, particularly for workloads with heavy decodes such as LPHD and HPHD. (3) TetriInfer's design is not ideal for HPHD workloads as the room for improvement is small, and the overhead we introduce cannot be offset.

### 5.2 Microbenchmark

### 5.2.1 Prefill Scheduler

Below, we study the overhead of sorting requests, compare different policies, and study the impact of batch size on performance. Note we run OPT-13B TP=2, ChunkSize is set to 512 , and vLLM's batch size is set to 16 .

Sort. Our scheduler sorts incoming requests based on the length of their input tokens if non-FCFS policies are used. We use Python's native sort API. We find the sorting overhead ranges from 10s to 100s of microseconds, which is negligible compared to millisecond-level or second-level TTFT latency.

Scheduler Policy and Batch Size. In Figure 16, we com-
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-11.jpg?height=336&width=880&top_left_y=247&top_left_x=172)

Figure 11: Light Prefill and Light Decode (LPLD)
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-11.jpg?height=328&width=882&top_left_y=638&top_left_x=168)

Figure 13: Heavy Prefill and Light Decode (HPLD)

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-11.jpg?height=334&width=896&top_left_y=1028&top_left_x=167)

Figure 15: Mixed Prefill and Decode

pare TetriInfer which uses chunked prefill along with FCFS, SJF, and LJF, against vanilla vLLM, which uses fixed batch size for prefill. Requests used in this test follow the ShareGPT distribution. In the left part, we set PrefillSchedBatch to 16. Compared to vLLM's fixed batch mode, chunked prefill alone with FCFS improves latency by $86.4 \%$. Additionally, the SJF scheduler policy further lowers the average waiting time by $7.8 \%$. The right part examines the impact of adjusting PrefillSchedBatch. When we increase the batch size from 16 to 128 , SJF's average TTFT decreases by $46.5 \%$. The improvement in TTFT increases with a larger scheduling batch.

### 5.2.2 Length Predictor

Our length predictor uses the OPT-125M classification model to speculate the decoding behavior of the OPT-13B model (§3.3.2). This section studies the performance of both models and the prediction accuracy.

In Figure 17, we run stress tests among several settings regarding per-iteration latency and throughput. L-Alone means running the OPT-13B model alone, using chunked fill with ChunkSize set to 512. $P$-Alone means running the OPT-125M prediction model alone. It does not use chunked prefill but
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-11.jpg?height=324&width=882&top_left_y=252&top_left_x=1058)

Figure 12: Light Prefill and Heavy Decode (LPHD)
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-11.jpg?height=332&width=892&top_left_y=636&top_left_x=1061)

Figure 14: Heavy Prefill and Heavy Decode (HPHD)

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-11.jpg?height=334&width=876&top_left_y=1031&top_left_x=1058)

Figure 16: Scheduler Policies and Chunked Prefill.

uses dynamic batch sizes. It can group multiple requests into a batch. Due to the limitation of [16], we need to pad requests in a batch to the longest one. For example, if we have two requests in a batch, one has 100 tokens, and the other has 500 tokens. Then, we need to pad the first to 500 tokens. This is costly for requests with short prompts. Hence, we set a cutting limit. Requests higher than the limit will run alone. The default is 512 tokens. $L+P 512$ means OPT-13B model's performance if co-runs with the small OPT-125M in parallel. The suffix number means the max padded size. We can see that the large LLM's prefill latency is roughly ten times of the small LLM's. If we co-run both models and a padding limit of $512,80 \%$ of large LLM's prefill requests remain unchanged compared to when it runs alone. Overall, while co-running with a small LLM, the large LLM's average prefill latency increases by $10 \%$, and throughput drops by $12 \%$. Note that these are stress tests. The impact will be smaller in practice. We believe beefier hardware can further mitigate the drop.

We train our OPT-125M prediction model using $75 \mathrm{~K}$ training data from ShareGPT. We test the model using three different length range granularities: 100,200 , and 400 . The accuracy achieved by our prediction model for these granularities is $58.9 \%, 74.9 \%$, and $85 \%$, respectively.
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-12.jpg?height=358&width=872&top_left_y=244&top_left_x=168)

Figure 17: Running Large LLM with Prediction Model.

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-12.jpg?height=361&width=873&top_left_y=752&top_left_x=168)

(a) Execution Time
![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-12.jpg?height=346&width=854&top_left_y=762&top_left_x=169)

Figure 18: Intra-Decode Instance Scheduling.

### 5.2.3 Decode Scheduling

We now study the scheduling policies related to decode instances. We first compare intra-decode instance scheduler policies (\$3.4) and then compare different load balance algorithms for inter-decode instance scheduling (\$3.3.4). All tests use OPT-13B with $\mathrm{TP}=2$.

We compare three intra-decode scheduler algorithms, namely vLLM's greedy, TetriInfer's reserve-static (RS), and reserve-dynamic (RD) in Figure 18. We run 256 requests following ShareGPT distribution. Our policies estimate resource usage using the predicted length range's lower end. We compare using the actual accuracy (acc-200 74.9\%) and an ideal accuracy of $100 \%$. While using the actual accuracy, reservedynamic achieves the same JCT as vLLM's greedy algorithm. When using an ideal prediction accuracy, reserve-dynamic and reserve-static improve average JCT by $12 \%$ and $10 \%$, respectively. This is because our policies carefully provision requests based on their memory usage.

We compare three distributed load balance algorithms in Figure 19. Firstly, we present our decentralized power-oftwo algorithm, designed to distribute requests based on predicted length. The second is random, in which the prefill instance randomly chooses a decode instance. The third algorithm, imbalance, simulates a worst-case scenario where heavy decode requests are consistently directed to the same decode instances. We run 32 requests per decode instance, spanning the range of 2 to 8 decode instances. Figure 19's left part shows that TetriInfer's decentralized load balancing

![](https://cdn.mathpix.com/cropped/2024_06_04_af820c56df3c96842c7bg-12.jpg?height=346&width=870&top_left_y=252&top_left_x=1083)

Figure 19: Inter-Decode Instance Scheduling.

| Work | C. P. | Disagg. P/D | Interference | Dist-Sched. |
| :---: | :---: | :---: | :---: | :---: |
| TetriInfer | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Splitwise [32] | $\times$ | $\checkmark$ | $\times$ | $\checkmark$ |
| Sarathi [1] | $\checkmark$ | $\times$ | $\times$ | $\times$ |
| vLLM [21] | $\times$ | $\times$ | $\times$ | $\checkmark$ |
| FastServe [41] | $\times$ | $\times$ | $\times$ | $\checkmark$ |

Table 1: Related work comparison. (1) C. P.: chunked prefill. (2) Disagg. P/D: disaggregated prefill and decode. (3) Interference: whether the system deals with inference interference. (4) Dist-Sched: distributed scheduling policies.

algorithm is effective, achieving the lowest total decoding time. The right parts show the number of heavy decode and light decode requests in the slowest instance. Cleary, TetriInfer's inter-decode scheduling algorithm evenly balances load across instances, which avoids interferences measured in $\S 2.2 .3$.

## 6 Related Work

Table 1 compares TetriInfer with other closely related works. We are among the first to disaggregate prefill and decode in LLM inference, concurrent to Splitwise [32]. Sarathi [1] has proposed chunked prefill to overcome suboptimal prefill processing. They run prefill-decode-mixed chunks. In contrast, TetriInfer runs prefill-only chunks as we observe non-neglible interference between prefill and decode, thus choose to disaggregate prefill from decode. FastServe [41] utilizes a multi-level priority feedback queue to minimize JCT. In contrast, TetriInfer utilizes two-level scheduling for prefill and decode instances. Our policies are working-set-aware, reducing interference and swaps thereby improving JCT and efficiency. Many recent work focus on optimizing batching, caching, and scheduling [2, 23, 30]. Specifically, Orca [45] introduce the iterative-level scheduling. Sheng et.al [36] have proposed a fair scheduler based on the continuous batching mechanism. Many works try to optimize memory usage. For example, using quantization $[7,8,11,19,22,37,42,43]$ to compress the model weights into lower precision, using paging to reduce fragmentation [21], and low-level algorithm and
kernel optimizations $[5,6,15,24,39,44,46]$. Those works are orthogonal to our efforts to mitigate interference.

## 7 Conclusion

We propose TetriInfer, an LLM inference serving system designed to battle interference. Our key insight is to carefully schedule and group inference requests based on their characteristics. It has three key parts. First, it partitions prompts into fixed-size chunks, ensuring the accelerator consistently operates at its computation-saturated limit. Second, it disaggregates prefill and decode instances to avoid interference when mixing them together. Finally, it uses a smart two-level scheduling algorithm to avoid decode scheduling hotspots. Results show that TetriInfer improves time-to-first-token, job completion time, and inference efficiency in terms of performance per dollar by a large margin.

## References

[1] Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, and Ramachandran Ramjee. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills. arXiv preprint arXiv:2308.16369, 2023.

[2] Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, et al. Deepspeed-inference: enabling efficient inference of transformer models at unprecedented scale. In SC22: International Conference for High Performance Computing, Networking, Storage and Analysis, 2022.

[3] AWS Bedrock.

https://docs.aws.amazon.com/bedrock/latest /userguide/inference-parameters.html.

[4] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[5] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

[6] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 2022.

[7] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm. int8 (): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022 .

[8] Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. Spqr: A sparse-quantized representation for near-lossless $11 \mathrm{~m}$ weight compression. arXiv preprint arXiv:2306.03078, 2023.

[9] Xin Luna Dong, Seungwhan Moon, Yifan Ethan Xu, Kshitiz Malik, and Zhou Yu. Towards next-generation intelligent assistants leveraging $11 \mathrm{~m}$ techniques. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023.

[10] Aleksandar Dragojević, Dushyanth Narayanan, Miguel Castro, and Orion Hodson. FaRM: Fast remote memory. In 11th USENIX Symposium on Networked Systems Design and Implementation (NSDI 14), 2014.

[11] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Optq: Accurate quantization for generative pre-trained transformers. In The Eleventh International Conference on Learning Representations, 2022.

[12] Zhiyuan Guo, Yizhou Shan, Xuhao Luo, Yutong Huang, and Yiying Zhang. Clio: A Hardware-Software CoDesigned Disaggregated Memory System. In Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, 2022.

[13] HiAscend. Atlas 900 AI Cluster.

https://www.hiascend.com/en/hardware/clust er.

[14] HiAscend. CANN aclrtMemcpy.

https://www.hiascend.com/document/detail/e

n/canncommercial/601/inferapplicationdev/ac lcppdevg/aclcppdevg_03_0081.html.

[15] Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Hanyu Dong, and Yu Wang. Flashdecoding++: Faster large language model inference on gpus. arXiv preprint arXiv:2311.01282, 2023.

[16] HugginFace.

https://huggingface.co/docs/transformers/m odel_doc/opt\#transformers.OPTForSequenceCl assification.

[17] Hugging Face.

https://huggingface.co/datasets/Zhongsheng Wang/Alpaca-pubmed-summarization.

[18] Hugging Face.

https://huggingface.co/datasets/lancexiao/ write_doc_sft_v1.

[19] Berivan Isik, Hermann Kumbong, Wanyi Ning, Xiaozhe Yao, Sanmi Koyejo, and Ce Zhang. Gpt-zip: Deep compression of finetuned large language models. In Workshop on Efficient Systems for Foundation Models @ ICML2023, 2023.

[20] A Jo. The promise and peril of generative ai. Nature, 2023.

[21] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, 2023.

[22] Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, and Joey Gonzalez. Train big, then compress: Rethinking model size for efficient training and inference of transformers. In International Conference on machine learning, 2020.

[23] Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E Gonzalez, et al. Alpaserve: Statistical multiplexing with model parallelism for deep learning serving. arXiv preprint arXiv:2302.11665, 2023.

[24] Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, and Lidong Zhou. Rammer: Enabling holistic deep learning compiler optimizations with \{rTasks $\}$. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), 2020.

[25] NGINX.

https://www.nginx.com/blog/nginx-power-o f-two-choices-load-balancing-algorithm/.

[26] NVIDIA. CUDA Runtime API Memory Management.

https://docs.nvidia.com/cuda/cuda-runtime -api/group__CUDART__MEMORY.html.

[27] NVIDIA. GPU Direct.

https://developer.nvidia.com/gpudirect.

[28] NVIDIA. NCCL.

https://docs.nvidia.com/deeplearning/nccl/ user-guide/docs/overview.html.

[29] NVIDIA, FasterTransformer.

https://github.com/NVIDIA/FasterTransforme r.
[30] NVIDIA, Triton Inference Server.

https://developer.nvidia.com/.

[31] Charles Packer, Vivian Fang, Shishir G Patil, Kevin Lin, Sarah Wooders, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

[32] Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative llm inference using phase splitting. arXiv preprint arXiv:2311.18677, 2023.

[33] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. Proceedings of Machine Learning and Systems, 2023.

[34] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, 2021.

[35] Sharegpt teams.

https://sharegpt.com/.

[36] Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E Gonzalez, and Ion Stoica. Fairness in serving large language models. arXiv preprint arXiv:2401.00588, 2023.

[37] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. Flexgen: High-throughput generative inference of large language models with a single gpu. In International Conference on Machine Learning, 2023.

[38] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv. org/abs/2307.09288, 2023.

[39] Xiaohui Wang, Ying Xiong, Yang Wei, Mingxuan Wang, and Lei Li. Lightseq: A high performance inference library for transformers. arXiv preprint arXiv:2010.13887, 2020.

[40] Wikipedia. NVLink.

https://en.wikipedia.org/wiki/NVLink.

[41] Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, and Xin Jin. Fast distributed inference serving for large language models. arXiv preprint arXiv:2305.05920, 2023.

[42] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, 2023.

[43] Zhewei Yao, Cheng Li, Xiaoxia Wu, Stephen Youn, and Yuxiong He. A comprehensive study on post-training quantization for large language models. arXiv preprint arXiv:2303.08302, 2023.

[44] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. Advances in Neural Information Processing Systems, 2022.

[45] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for \{Transformer-Based\} generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), 2022.

[46] Yujia Zhai, Chengquan Jiang, Leyuan Wang, Xiaoying Jia, Shang Zhang, Zizhong Chen, Xin Liu, and Yibo Zhu. Bytetransformer: A high-performance transformer boosted for variable-length inputs. In 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2023.

[47] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

[48] Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, and Yang You. Response length perception and sequence scheduling: An llm-empowered llm inference pipeline. arXiv preprint arXiv:2305.13144, 2023.


[^0]:    ${ }^{0}$ Work done while intern at Huawei Cloud

[^1]:    ${ }^{1}$ The name of our system, TetriInfer, implies that it can efficiently organize LLM inference requests, similar to how tetris blocks are stacked.

</end of paper 4>


