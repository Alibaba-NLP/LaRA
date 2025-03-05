<paper 0>
# ON THE ConVERGENCE OF FEDAVG ON NON-IID DATA 

\author{
Xiang Li ${ }^{*}$ <br> School of Mathematical Sciences <br> Peking University <br> Beijing, 100871, China <br> smslixiang@pku.edu.cn <br> Wenhao Yang* <br> Center for Data Science <br> Peking University <br> Beijing, 100871, China <br> yangwenhaosms@pku.edu.cn

## Zhihua Zhang

 <br> School of Mathematical Sciences <br> Peking University <br> Beijing, 100871, China <br> zhzhang@math.pku.edu.cn}

Kaixuan Huang ${ }^{*}$<br>School of Mathematical Sciences<br>Peking University<br>Beijing, 100871, China<br>hackyhuang@pku.edu.cn<br>Shusen Wang<br>Department of Computer Science<br>Stevens Institute of Technology<br>Hoboken, NJ 07030, USA<br>shusen.wang@stevens.edu


#### Abstract

Federated learning enables a large amount of edge computing devices to jointly learn a model without data sharing. As a leading algorithm in this setting, Federated Averaging (FedAvg) runs Stochastic Gradient Descent (SGD) in parallel on a small subset of the total devices and averages the sequences only once in a while. Despite its simplicity, it lacks theoretical guarantees under realistic settings. In this paper, we analyze the convergence of FedAvg on non-iid data and establish a convergence rate of $\mathcal{O}\left(\frac{1}{T}\right)$ for strongly convex and smooth problems, where $T$ is the number of SGDs. Importantly, our bound demonstrates a trade-off between communicationefficiency and convergence rate. As user devices may be disconnected from the server, we relax the assumption of full device participation to partial device participation and study different averaging schemes; low device participation rate can be achieved without severely slowing down the learning. Our results indicates that heterogeneity of data slows down the convergence, which matches empirical observations. Furthermore, we provide a necessary condition for FedAvg on non-iid data: the learning rate $\eta$ must decay, even if full-gradient is used; otherwise, the solution will be $\Omega(\eta)$ away from the optimal.


## 1 INTRODUCTION

Federated Learning (FL), also known as federated optimization, allows multiple parties to collaboratively train a model without data sharing (Konevenỳ et al., 2015; Shokri and Shmatikov, 2015; McMahan et al., 2017; Konevcnỳ, 2017; Sahu et al., 2018; Zhuo et al., 2019). Similar to the centralized parallel optimization (Jakovetic, 2013; Li et al., 2014a;b; Shamir et al., 2014; Zhang and Lin, 2015; Meng et al., 2016; Reddi et al., 2016; Richtárik and Takác, 2016; Smith et al., 2016; Zheng et al., 2016; Shusen Wang et al., 2018), FL let the user devices (aka worker nodes) perform most of the computation and a central parameter server update the model parameters using the descending directions returned by the user devices. Nevertheless, FL has three unique characters that distinguish it from the standard parallel optimization Li et al. (2019).[^0]

First, the training data are massively distributed over an incredibly large number of devices, and the connection between the central server and a device is slow. A direct consequence is the slow communication, which motivated communication-efficient FL algorithms (McMahan et al., 2017; Smith et al., 2017; Sahu et al., 2018; Sattler et al., 2019). Federated averaging (FedAvg) is the first and perhaps the most widely used FL algorithm. It runs $E$ steps of SGD in parallel on a small sampled subset of devices and then averages the resulting model updates via a central server once in a while. ${ }^{1}$ In comparison with SGD and its variants, FedAvg performs more local computation and less communication.

Second, unlike the traditional distributed learning systems, the FL system does not have control over users' devices. For example, when a mobile phone is turned off or WiFi access is unavailable, the central server will lose connection to this device. When this happens during training, such a non-responding/inactive device, which is called a straggler, appears tremendously slower than the other devices. Unfortunately, since it has no control over the devices, the system can do nothing but waiting or ignoring the stragglers. Waiting for all the devices' response is obviously infeasible; it is thus impractical to require all the devices be active.

Third, the training data are non-iid ${ }^{2}$, that is, a device's local data cannot be regarded as samples drawn from the overall distribution. The data available locally fail to represent the overall distribution. This does not only bring challenges to algorithm design but also make theoretical analysis much harder. While FedAvg actually works when the data are non-iid McMahan et al. (2017), FedAvg on non-iid data lacks theoretical guarantee even in convex optimization setting.

There have been much efforts developing convergence guarantees for FL algorithm based on the assumptions that (1) the data are iid and (2) all the devices are active. Khaled et al. (2019); Yu et al. (2019); Wang et al. (2019) made the latter assumption, while Zhou and Cong (2017); Stich (2018); Wang and Joshi (2018); Woodworth et al. (2018) made both assumptions. The two assumptions violates the second and third characters of FL. Previous algorithm Fedprox Sahu et al. (2018) doesn't require the two mentioned assumptions and incorporates FedAvg as a special case when the added proximal term vanishes. However, their theory fails to cover FedAvg.

Notation. Let $N$ be the total number of user devices and $K(\leq N)$ be the maximal number of devices that participate in every round's communication. Let $T$ be the total number of every device's SGDs, $E$ be the number of local iterations performed in a device between two communications, and thus $\frac{T}{E}$ is the number of communications.

Contributions. For strongly convex and smooth problems, we establish a convergence guarantee for FedAvg without making the two impractical assumptions: (1) the data are iid, and (2) all the devices are active. To the best of our knowledge, this work is the first to show the convergence rate of FedAvg without making the two assumptions.

We show in Theorem 1, 2, and 3 that FedAvg has $\mathcal{O}\left(\frac{1}{T}\right)$ convergence rate. In particular, Theorem 3 shows that to attain a fixed precision $\epsilon$, the number of communications is

$$
\begin{equation*}
\frac{T}{E}=\mathcal{O}\left[\frac{1}{\epsilon}\left(\left(1+\frac{1}{K}\right) E G^{2}+\frac{\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+\Gamma+G^{2}}{E}+G^{2}\right)\right] \tag{1}
\end{equation*}
$$

Here, $G, \Gamma, p_{k}$, and $\sigma_{k}$ are problem-related constants defined in Section 3.1. The most interesting insight is that $E$ is a knob controlling the convergence rate: neither setting $E$ over-small ( $E=1$ makes FedAvg equivalent to SGD) nor setting $E$ over-large is good for the convergence.

This work also makes algorithmic contributions. We summarize the existing sampling ${ }^{3}$ and averaging schemes for FedAvg (which do not have convergence bounds before this work) and propose a new scheme (see Table 1). We point out that a suitable sampling and averaging scheme is crucial for the convergence of FedAvg. To the best of our knowledge, we are the first to theoretically demonstrate[^1]

Table 1: Sampling and averaging schemes for FedAvg. $\mathcal{S}_{t} \sim \mathcal{U}(N, K)$ means $\mathcal{S}_{t}$ is a size- $K$ subset uniformly sampled without replacement from $[N] . \mathcal{S}_{t} \sim \mathcal{W}(N, K, \mathbf{p})$ means $\mathcal{S}_{t}$ contains $K$ elements that are iid sampled with replacement from $[N]$ with probabilities $\left\{p_{k}\right\}$. In the latter scheme, $\mathcal{S}_{t}$ is not a set.

| Paper | Sampling | Averaging | Convergence rate |
| :---: | :---: | :---: | :---: |
| McMahan et al. (2017) | $\mathcal{S}_{t} \sim \mathcal{U}(N, K)$ | $\sum_{k \notin \mathcal{S}_{t}} p_{k} \mathbf{w}_{t}+\sum_{k \in \mathcal{S}_{t}} p_{k} \mathbf{w}_{t}^{k}$ | - |
| Sahu et al. (2018) | $\mathcal{S}_{t} \sim \mathcal{W}(N, K, \mathbf{p})$ | $\frac{1}{K} \sum_{k \in \mathcal{S}_{t} \mathbf{w}_{t}^{k}}$ | $\mathcal{O}\left(\frac{1}{T}\right)^{5}$ |
| Ours | $\mathcal{S}_{t} \sim \mathcal{U}(N, K)$ | $\sum_{k \in \mathcal{S}_{t}} p_{k} \frac{N}{K} \mathbf{w}_{t}^{k}$ | $\mathcal{O}\left(\frac{1}{T}\right)^{6}$ |

that FedAvg with certain schemes (see Table 1) can achieve $\mathcal{O}\left(\frac{1}{T}\right)$ convergence rate in non-iid federated setting. We show that heterogeneity of training data and partial device participation slow down the convergence. We empirically verify our results through numerical experiments.

Our theoretical analysis requires the decay of learning rate (which is known to hinder the convergence rate.) Unfortunately, we show in Theorem 4 that the decay of learning rate is necessary for FedAvg with $E>1$, even if full gradient descent is used. ${ }^{4}$ If the learning rate is fixed to $\eta$ throughout, FedAvg would converge to a solution at least $\Omega(\eta(E-1))$ away from the optimal. To establish Theorem 4 , we construct a specific $\ell_{2}$-norm regularized linear regression model which satisfies our strong convexity and smoothness assumptions.

Paper organization. In Section 2, we elaborate on FedAvg. In Section 3, we present our main convergence bounds for FedAvg. In Section 4, we construct a special example to show the necessity of learning rate decay. In Section 5, we discuss and compare with prior work. In Section 6, we conduct empirical study to verify our theories. All the proofs are left to the appendix.

## 2 FEDERATED AVERAGING (FedAvg)

Problem formulation. In this work, we consider the following distributed optimization model:

$$
\begin{equation*}
\min _{\mathbf{w}}\left\{F(\mathbf{w}) \triangleq \sum_{k=1}^{N} p_{k} F_{k}(\mathbf{w})\right\} \tag{2}
\end{equation*}
$$

where $N$ is the number of devices, and $p_{k}$ is the weight of the $k$-th device such that $p_{k} \geq 0$ and $\sum_{k=1}^{N} p_{k}=1$. Suppose the $k$-th device holds the $n_{k}$ training data: $x_{k, 1}, x_{k, 2}, \cdots, x_{k, n_{k}}$. The local objective $F_{k}(\cdot)$ is defined by

$$
\begin{equation*}
F_{k}(\mathbf{w}) \triangleq \frac{1}{n_{k}} \sum_{j=1}^{n_{k}} \ell\left(\mathbf{w} ; x_{k, j}\right) \tag{3}
\end{equation*}
$$

where $\ell(\cdot ; \cdot)$ is a user-specified loss function.

Algorithm description. Here, we describe one around (say the $t$-th) of the standard FedAvg algorithm. First, the central server broadcasts the latest model, $\mathbf{w}_{t}$, to all the devices. Second, every device (say the $k$-th) lets $\mathbf{w}_{t}^{k}=\mathbf{w}_{t}$ and then performs $E(\geq 1)$ local updates:

$$
\mathbf{w}_{t+i+1}^{k} \longleftarrow \mathbf{w}_{t+i}^{k}-\eta_{t+i} \nabla F_{k}\left(\mathbf{w}_{t+i}^{k}, \xi_{t+i}^{k}\right), i=0,1, \cdots, E-1
$$

where $\eta_{t+i}$ is the learning rate (a.k.a. step size) and $\xi_{t+i}^{k}$ is a sample uniformly chosen from the local data. Last, the server aggregates the local models, $\mathbf{w}_{t+E}^{1}, \cdots, \mathbf{w}_{t+E}^{N}$, to produce the new global model, $\mathbf{w}_{t+E}$. Because of the non-iid and partial device participation issues, the aggregation step can vary.[^2]

IID versus non-iid. Suppose the data in the $k$-th device are i.i.d. sampled from the distribution $\mathcal{D}_{k}$. Then the overall distribution is a mixture of all local data distributions: $\mathcal{D}=\sum_{k=1}^{N} p_{k} \mathcal{D}_{k}$. The prior work Zhang et al. (2015a); Zhou and Cong (2017); Stich (2018); Wang and Joshi (2018); Woodworth et al. (2018) assumes the data are iid generated by or partitioned among the $N$ devices, that is, $\mathcal{D}_{k}=\mathcal{D}$ for all $k \in[N]$. However, real-world applications do not typically satisfy the iid assumption. One of our theoretical contributions is avoiding making the iid assumption.

Full device participation. The prior work Coppola (2015); Zhou and Cong (2017); Stich (2018); Yu et al. (2019); Wang and Joshi (2018); Wang et al. (2019) requires the full device participation in the aggregation step of FedAvg. In this case, the aggregation step performs

$$
\mathbf{w}_{t+E} \longleftarrow \sum_{k=1}^{N} p_{k} \mathbf{w}_{t+E}^{k}
$$

Unfortunately, the full device participation requirement suffers from serious "straggler's effect" (which means everyone waits for the slowest) in real-world applications. For example, if there are thousands of users' devices in the FL system, there are always a small portion of devices offline. Full device participation means the central server must wait for these "stragglers", which is obviously unrealistic.

Partial device participation. This strategy is much more realistic because it does not require all the devices' output. We can set a threshold $K(1 \leq K<N)$ and let the central server collect the outputs of the first $K$ responded devices. After collecting $K$ outputs, the server stops waiting for the rest; the $K+1$-th to $N$-th devices are regarded stragglers in this iteration. Let $\mathcal{S}_{t}\left(\left|\mathcal{S}_{t}\right|=K\right)$ be the set of the indices of the first $K$ responded devices in the $t$-th iteration. The aggregation step performs

$$
\mathbf{w}_{t+E} \longleftarrow \frac{N}{K} \sum_{k \in \mathcal{S}_{t}} p_{k} \mathbf{w}_{t+E}^{k}
$$

It can be proved that $\frac{N}{K} \sum_{k \in \mathcal{S}_{t}} p_{k}$ equals one in expectation.

Communication cost. The FedAvg requires two rounds communications- one broadcast and one aggregation- per $E$ iterations. If $T$ iterations are performed totally, then the number of communications is $\left\lfloor\frac{2 T}{E}\right\rfloor$. During the broadcast, the central server sends $\mathbf{w}_{t}$ to all the devices. During the aggregation, all or part of the $N$ devices sends its output, say $\mathbf{w}_{t+E}^{k}$, to the server.

## 3 ConVERGENCE ANALYSIS OF FedAvg IN NON-IID SETTING

In this section, we show that FedAvg converges to the global optimum at a rate of $\mathcal{O}(1 / T)$ for strongly convex and smooth functions and non-iid data. The main observation is that when the learning rate is sufficiently small, the effect of $E$ steps of local updates is similar to one step update with a larger learning rate. This coupled with appropriate sampling and averaging schemes would make each global update behave like an SGD update. Partial device participation $(K<N)$ only makes the averaged sequence $\left\{\mathbf{w}_{t}\right\}$ have a larger variance, which, however, can be controlled by learning rates. These imply the convergence property of FedAvg should not differ too much from SGD. Next, we will first give the convergence result with full device participation (i.e., $K=N$ ) and then extend this result to partial device participation (i.e., $K<N$ ).

### 3.1 NOTATION AND ASSUMPTIONS

We make the following assumptions on the functions $F_{1}, \cdots, F_{N}$. Assumption 1 and 2 are standard; typical examples are the $\ell_{2}$-norm regularized linear regression, logistic regression, and softmax classifier.

Assumption 1. $F_{1}, \cdots, F_{N}$ are all $L$-smooth: for all $\mathbf{v}$ and $\mathbf{w}, F_{k}(\mathbf{v}) \leq F_{k}(\mathbf{w})+(\mathbf{v}-$ $\mathbf{w})^{T} \nabla F_{k}(\mathbf{w})+\frac{L}{2}\|\mathbf{v}-\mathbf{w}\|_{2}^{2}$.

Assumption 2. $F_{1}, \cdots, F_{N}$ are all $\mu$-strongly convex: for all $\mathbf{v}$ and $\mathbf{w}, F_{k}(\mathbf{v}) \geq F_{k}(\mathbf{w})+(\mathbf{v}-$ $\mathbf{w})^{T} \nabla F_{k}(\mathbf{w})+\frac{\mu}{2}\|\mathbf{v}-\mathbf{w}\|_{2}^{2}$.

Assumptions 3 and 4 have been made by the works Zhang et al. (2013); Stich (2018); Stich et al. (2018); Yu et al. (2019).

Assumption 3. Let $\xi_{t}^{k}$ be sampled from the $k$-th device's local data uniformly at random. The variance of stochastic gradients in each device is bounded: $\mathbb{E}\left\|\nabla F_{k}\left(\mathbf{w}_{t}^{k}, \xi_{t}^{k}\right)-\nabla F_{k}\left(\mathbf{w}_{t}^{k}\right)\right\|^{2} \leq \sigma_{k}^{2}$ for $k=1, \cdots, N$.

Assumption 4. The expected squared norm of stochastic gradients is uniformly bounded, i.e., $\mathbb{E}\left\|\nabla F_{k}\left(\mathbf{w}_{t}^{k}, \xi_{t}^{k}\right)\right\|^{2} \leq G^{2}$ for all $k=1, \cdots, N$ and $t=1, \cdots, T-1$

Quantifying the degree of non-iid (heterogeneity). Let $F^{*}$ and $F_{k}^{*}$ be the minimum values of $F$ and $F_{k}$, respectively. We use the term $\Gamma=F^{*}-\sum_{k=1}^{N} p_{k} F_{k}^{*}$ for quantifying the degree of non-iid. If the data are iid, then $\Gamma$ obviously goes to zero as the number of samples grows. If the data are non-iid, then $\Gamma$ is nonzero, and its magnitude reflects the heterogeneity of the data distribution.

### 3.2 ConVERGENCE ReSULT: FULL DeVICE PARTICIPATION

Here we analyze the case that all the devices participate in the aggregation step; see Section 2 for the algorithm description. Let the FedAvg algorithm terminate after $T$ iterations and return $\mathbf{w}_{T}$ as the solution. We always require $T$ is evenly divisible by $E$ so that FedAvg can output $\mathbf{w}_{T}$ as expected.

Theorem 1. Let Assumptions 1 to 4 hold and $L, \mu, \sigma_{k}, G$ be defined therein. Choose $\kappa=\frac{L}{\mu}$, $\gamma=\max \{8 \kappa, E\}$ and the learning rate $\eta_{t}=\frac{2}{\mu(\gamma+t)}$. Then $F e d A v g$ with full device participation satisfies

$$
\begin{equation*}
\mathbb{E}\left[F\left(\mathbf{w}_{T}\right)\right]-F^{*} \leq \frac{\kappa}{\gamma+T-1}\left(\frac{2 B}{\mu}+\frac{\mu \gamma}{2} \mathbb{E}\left\|\mathbf{w}_{1}-\mathbf{w}^{*}\right\|^{2}\right) \tag{4}
\end{equation*}
$$

where

$$
\begin{equation*}
B=\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+6 L \Gamma+8(E-1)^{2} G^{2} \tag{5}
\end{equation*}
$$

### 3.3 ConVERGENCE ReSUlT: Partial DeVICE PARTICIPATION

As discussed in Section 2, partial device participation has more practical interest than full device participation. Let the set $\mathcal{S}_{t}(\subset[N])$ index the active devices in the $t$-th iteration. To establish the convergence bound, we need to make assumptions on $\mathcal{S}_{t}$.

Assumption 5 assumes the $K$ indices are selected from the distribution $p_{k}$ independently and with replacement. The aggregation step is simply averaging. This is first proposed in (Sahu et al., 2018), but they did not provide theoretical analysis.

Assumption 5 (Scheme I). Assume $\mathcal{S}_{t}$ contains a subset of $K$ indices randomly selected with replacement according to the sampling probabilities $p_{1}, \cdots, p_{N}$. The aggregation step of FedAvg performs $\mathbf{w}_{t} \longleftarrow \frac{1}{K} \sum_{k \in \mathcal{S}_{t}} \mathbf{w}_{t}^{k}$.

Theorem 2. Let Assumptions 1 to 4 hold and $L, \mu, \sigma_{k}, G$ be defined therein. Let $\kappa, \gamma, \eta_{t}$, and $B$ be defined in Theorem 1. Let Assumption 5 hold and define $C=\frac{4}{K} E^{2} G^{2}$. Then

$$
\begin{equation*}
\mathbb{E}\left[F\left(\mathbf{w}_{T}\right)\right]-F^{*} \leq \frac{\kappa}{\gamma+T-1}\left(\frac{2(B+C)}{\mu}+\frac{\mu \gamma}{2} \mathbb{E}\left\|\mathbf{w}_{1}-\mathbf{w}^{*}\right\|^{2}\right) \tag{6}
\end{equation*}
$$

Alternatively, we can select $K$ indices from $[N]$ uniformly at random without replacement. As a consequence, we need a different aggregation strategy. Assumption 6 assumes the $K$ indices are selected uniformly without replacement and the aggregation step is the same as in Section 2. However, to guarantee convergence, we require an additional assumption of balanced data.

Assumption 6 (Scheme II). Assume $\mathcal{S}_{t}$ contains a subset of $K$ indices uniformly sampled from $[N]$ without replacement. Assume the data is balanced in the sense that $p_{1}=\cdots=p_{N}=\frac{1}{N}$. The aggregation step of FedAvg performs $\mathbf{w}_{t} \longleftarrow \frac{N}{K} \sum_{k \in \mathcal{S}_{t}} p_{k} \mathbf{w}_{t}^{k}$.

Theorem 3. Replace Assumption 5 by Assumption 6 and $C$ by $C=\frac{N-K}{N-1} \frac{4}{K} E^{2} G^{2}$. Then the same bound in Theorem 2 holds.

Scheme II requires $p_{1}=\cdots=p_{N}=\frac{1}{N}$ which obviously violates the unbalance nature of FL. Fortunately, this can be addressed by the following transformation. Let $\widetilde{F}_{k}(\mathbf{w})=p_{k} N F_{k}(\mathbf{w})$ be a scaled local objective $F_{k}$. Then the global objective becomes a simple average of all scaled local objectives:

$$
F(\mathbf{w})=\sum_{k=1}^{N} p_{k} F_{k}(\mathbf{w})=\frac{1}{N} \sum_{k=1}^{N} \widetilde{F}_{k}(\mathbf{w})
$$

Theorem 3 still holds if $L, \mu, \sigma_{k}, G$ are replaced by $\widetilde{L} \triangleq \nu L, \widetilde{\mu} \triangleq \varsigma \mu, \widetilde{\sigma}_{k}=\sqrt{\nu} \sigma$, and $\widetilde{G}=\sqrt{\nu} G$, respectively. Here, $\nu=N \cdot \max _{k} p_{k}$ and $\varsigma=N \cdot \min _{k} p_{k}$.

### 3.4 DISCUSSIONS

Choice of $E$. Since $\left\|\mathbf{w}_{0}-\mathbf{w}^{*}\right\|^{2} \leq \frac{4}{\mu^{2}} G^{2}$ for $\mu$-strongly convex $F$, the dominating term in eqn. (6) is

$$
\begin{equation*}
\mathcal{O}\left(\frac{\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+L \Gamma+\left(1+\frac{1}{K}\right) E^{2} G^{2}+\gamma G^{2}}{\mu T}\right) \tag{7}
\end{equation*}
$$

Let $T_{\epsilon}$ denote the number of required steps for FedAvg to achieve an $\epsilon$ accuracy. It follows from eqn. (7) that the number of required communication rounds is roughly ${ }^{7}$

$$
\begin{equation*}
\frac{T_{\epsilon}}{E} \propto\left(1+\frac{1}{K}\right) E G^{2}+\frac{\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+L \Gamma+\kappa G^{2}}{E}+G^{2} \tag{8}
\end{equation*}
$$

Thus, $\frac{T_{E}}{E}$ is a function of $E$ that first decreases and then increases, which implies that over-small or over-large $E$ may lead to high communication cost and that the optimal $E$ exists.

Stich (2018) showed that if the data are iid, then $E$ can be set to $\mathcal{O}(\sqrt{T})$. However, this setting does not work if the data are non-iid. Theorem 1 implies that $E$ must not exceed $\Omega(\sqrt{T})$; otherwise, convergence is not guaranteed. Here we give an intuitive explanation. If $E$ is set big, then $\mathbf{w}_{t}^{k}$ can converge to the minimizer of $F_{k}$, and thus FedAvg becomes the one-shot average Zhang et al. (2013) of the local solutions. If the data are non-iid, the one-shot averaging does not work because weighted average of the minimizers of $F_{1}, \cdots, F_{N}$ can be very different from the minimizer of $F$.

Choice of $K$. Stich (2018) showed that if the data are iid, the convergence rate improves substantially as $K$ increases. However, under the non-iid setting, the convergence rate has a weak dependence on $K$, as we show in Theorems 2 and 3. This implies FedAvg is unable to achieve linear speedup. We have empirically observed this phenomenon (see Section 6). Thus, in practice, the participation ratio $\frac{K}{N}$ can be set small to alleviate the straggler's effect without affecting the convergence rate.

Choice of sampling schemes. We considered two sampling and averaging schemes in Theorems 2 and 3. Scheme I selects $K$ devices according to the probabilities $p_{1}, \cdots, p_{N}$ with replacement. The non-uniform sampling results in faster convergence than uniform sampling, especially when $p_{1}, \cdots, p_{N}$ are highly non-uniform. If the system can choose to activate any of the $N$ devices at any time, then Scheme I should be used.

However, oftentimes the system has no control over the sampling; instead, the server simply uses the first $K$ returned results for the update. In this case, we can assume the $K$ devices are uniformly sampled from all the $N$ devices and use Theorem 3 to guarantee the convergence. If $p_{1}, \cdots, p_{N}$ are highly non-uniform, then $\nu=N \cdot \max _{k} p_{k}$ is big and $\varsigma=N \cdot \min _{k} p_{k}$ is small, which makes the convergence of FedAvg slow. This point of view is empirically verified in our experiments.

## 4 NECESSITY OF LEARNING RATE DECAY

In this section, we point out that diminishing learning rates are crucial for the convergence of FedAvg in the non-iid setting. Specifically, we establish the following theorem by constructing a ridge regression model (which is strongly convex and smooth).[^3]

Theorem 4. We artificially construct a strongly convex and smooth distributed optimization problem. With full batch size, $E>1$, and any fixed step size, FedAvg will converge to sub-optimal points. Specifically, let $\tilde{\mathbf{w}}^{*}$ be the solution produced by FedAvg with a small enough and constant $\eta$, and $\mathrm{w}^{*}$ the optimal solution. Then we have

$$
\left\|\tilde{\mathbf{w}}^{*}-\mathbf{w}^{*}\right\|_{2}=\Omega((E-1) \eta) \cdot\left\|\mathbf{w}^{*}\right\|_{2}
$$

where we hide some problem dependent constants.

Theorem 4 and its proof provide several implications. First, the decay of learning rate is necessary of FedAvg. On the one hand, Theorem 1 shows with $E>1$ and a decaying learning rate, FedAvg converges to the optimum. On the other hand, Theorem 4 shows that with $E>1$ and any fixed learning rate, FedAvg does not converges to the optimum.

Second, FedAvg behaves very differently from gradient descent. Note that FedAvg with $E=1$ and full batch size is exactly the Full Gradient Descent; with a proper and fixed learning rate, its global convergence to the optimum is guaranteed Nesterov (2013). However, Theorem 4 shows that FedAvg with $E>1$ and full batch size cannot possibly converge to the optimum. This conclusion doesn't contradict with Theorem 1 in Khaled et al. (2019), which, when translated into our case, asserts that $\tilde{\mathbf{w}}^{*}$ will locate in the neighborhood of $\mathbf{w}^{*}$ with a constant learning rate.

Third, Theorem 4 shows the requirement of learning rate decay is not an artifact of our analysis; instead, it is inherently required by FedAvg. An explanation is that constant learning rates, combined with $E$ steps of possibly-biased local updates, form a sub-optimal update scheme, but a diminishing learning rate can gradually eliminate such bias.

The efficiency of FedAvg principally results from the fact that it performs several update steps on a local model before communicating with other workers, which saves communication. Diminishing step sizes often hinders fast convergence, which may counteract the benefit of performing multiple local updates. Theorem 4 motivates more efficient alternatives to FedAvg.

## 5 RELATED WORK

Federated learning (FL) was first proposed by McMahan et al. (2017) for collaboratively learning a model without collecting users' data. The research work on FL is focused on the communicationefficiency Konevcnỳ et al. (2016); McMahan et al. (2017); Sahu et al. (2018); Smith et al. (2017) and data privacy Bagdasaryan et al. (2018); Bonawitz et al. (2017); Geyer et al. (2017); Hitaj et al. (2017); Melis et al. (2019). This work is focused on the communication-efficiency issue.

FedAvg, a synchronous distributed optimization algorithm, was proposed by McMahan et al. (2017) as an effective heuristic. Sattler et al. (2019); Zhao et al. (2018) studied the non-iid setting, however, they do not have convergence rate. A contemporaneous and independent work Xie et al. (2019) analyzed asynchronous FedAvg; while they did not require iid data, their bound do not guarantee convergence to saddle point or local minimum. Sahu et al. (2018) proposed a federated optimization framework called FedProx to deal with statistical heterogeneity and provided the convergence guarantees in non-iid setting. FedProx adds a proximal term to each local objective. When these proximal terms vanish, FedP rox is reduced to FedAvg. However, their convergence theory requires the proximal terms always exist and hence fails to cover FedAvg.

When data are iid distributed and all devices are active, FedAvg is referred to as LocalSGD. Due to the two assumptions, theoretical analysis of LocalSGD is easier than FedAvg. Stich (2018) demonstrated LocalSGD provably achieves the same linear speedup with strictly less communication for strongly-convex stochastic optimization. Coppola (2015); Zhou and Cong (2017); Wang and Joshi (2018) studied LocalSGD in the non-convex setting and established convergence results. Yu et al. (2019); Wang et al. (2019) recently analyzed LocalSGD for non-convex functions in heterogeneous settings. In particular, Yu et al. (2019) demonstrated LocalSGD also achieves $\mathcal{O}(1 / \sqrt{N T})$ convergence (i.e., linear speedup) for non-convex optimization. Lin et al. (2018) empirically shows variants of LocalSGD increase training efficiency and improve the generalization performance of large batch sizes while reducing communication. For LocalGD on non-iid data (as opposed to LocalSGD), the best result is by the contemporaneous work (but slightly later than our first version) (Khaled et al., 2019). Khaled et al. (2019) used fixed learning rate $\eta$ and showed $\mathcal{O}\left(\frac{1}{T}\right)$
convergence to a point $\mathcal{O}\left(\eta^{2} E^{2}\right)$ away from the optimal. In fact, the suboptimality is due to their fixed learning rate. As we show in Theorem 4, using a fixed learning rate $\eta$ throughout, the solution by LocalGD is at least $\Omega((E-1) \eta)$ away from the optimal.

If the data are iid, distributed optimization can be efficiently solved by the second-order algorithms Mahajan et al. (2018); Reddi et al. (2016); Shamir et al. (2014); Shusen Wang et al. (2018); Zhang and Lin (2015) and the one-shot methods Lee et al. (2017); Lin et al. (2017); Wang (2019); Zhang et al. (2013; 2015b). The primal-dual algorithms Hong et al. (2018); Smith et al. (2016; 2017) are more generally applicable and more relevant to FL.

## 6 NUMERICAL EXPERIMENTS

Models and datasets We examine our theoretical results on a logistic regression with weight decay $\lambda=1 e-4$. This is a stochastic convex optimization problem. We distribute MNIST dataset (LeCun et al., 1998) among $N=100$ workers in a non-iid fashion such that each device contains samples of only two digits. We further obtain two datasets: mnist balanced and mnist unbalanced. The former is balanced such that the number of samples in each device is the same, while the latter is highly unbalanced with the number of samples among devices following a power law. To manipulate heterogeneity more precisly, we synthesize unbalanced datasets following the setup in Sahu et al. (2018) and denote it as synthet ic ( $\alpha, \beta$ ) where $\alpha$ controls how much local models differ from each other and $\beta$ controls how much the local data at each device differs from that of other devices. We obtain two datasets: synthetic $(0,0)$ and synthetic $(1,1)$. Details can be found in Appendix D.

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=344&width=1418&top_left_y=1292&top_left_x=337)

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=252&width=347&top_left_y=1305&top_left_x=347)

(a) The impact of $E$

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=258&width=349&top_left_y=1302&top_left_x=693)

(b) The impact of $K$

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=252&width=349&top_left_y=1305&top_left_x=1037)

(c) Different schemes

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=258&width=350&top_left_y=1302&top_left_x=1386)

(d) Different schemes

Figure 1: (a) To obtain an $\epsilon$ accuracy, the required rounds first decrease and then increase when we increase the local steps $E$. (b) In Synthetic $(0,0)$ dataset, decreasing the numbers of active devices each round has little effect on the convergence process. (c) In mnist balanced dataset, Scheme I slightly outperforms Scheme II. They both performs better than the original scheme. Here transformed Scheme II coincides with Scheme II due to the balanced data. (d) In mnist unbal anced dataset, Scheme I performs better than Scheme II and the original scheme. Scheme II suffers from instability while transformed Scheme II has a lower convergence rate.

Experiment settings For all experiments, we initialize all runnings with $\mathbf{w}_{0}=0$. In each round, all selected devices run $E$ steps of SGD in parallel. We decay the learning rate at the end of each round by the following scheme $\eta_{t}=\frac{\eta_{0}}{1+t}$, where $\eta_{0}$ is chosen from the set $\{1,0.1,0.01\}$. We evaluate the averaged model after each global synchronization on the corresponding global objective. For fair comparison, we control all randomness in experiments so that the set of activated devices is the same across all different algorithms on one configuration.

Impact of $E \quad$ We expect that $T_{\epsilon} / E$, the required communication round to achieve curtain accuracy, is a hyperbolic finction of $E$ as equ (8) indicates. Intuitively, a small $E$ means a heavy communication burden, while a large $E$ means a low convergence rate. One needs to trade off between communication efficiency and fast convergence. We empirically observe this phenomenon on unbalanced datasets in Figure 1a. The reason why the phenomenon does not appear in mnist balanced dataset requires future investigations.

Impact of $K$ Our theory suggests that a larger $K$ may slightly accelerate convergence since $T_{\epsilon} / E$ contains a term $\mathcal{O}\left(\frac{E G^{2}}{K}\right)$. Figure $1 \mathrm{~b}$ shows that $K$ has limited influence on the convergence of FedAvg in synthetic $(0,0)$ dataset. It reveals that the curve of a large enough $K$ is slightly better. We observe similar phenomenon among the other three datasets and attach additional results in Appendix D. This justifies that when the variance resulting sampling is not too large (i.e., $B \gg C$ ), one can use a small number of devices without severely harming the training process, which also removes the need to sample as many devices as possible in convex federated optimization.

Effect of sampling and averaging schemes. We compare four schemes among four federated datasets. Since the original scheme involves a history term and may be conservative, we carefully set the initial learning rate for it. Figure 1c indicates that when data are balanced, Schemes I and II achieve nearly the same performance, both better than the original scheme. Figure 1d shows that when the data are unbalanced, i.e., $p_{k}$ 's are uneven, Scheme I performs the best. Scheme II suffers from some instability in this case. This is not contradictory with our theory since we don't guarantee the convergence of Scheme II when data is unbalanced. As expected, transformed Scheme II performs stably at the price of a lower convergence rate. Compared to Scheme I, the original scheme converges at a slower speed even if its learning rate is fine tuned. All the results show the crucial position of appropriate sampling and averaging schemes for FedAvg.

## 7 CONCLUSION

Federated learning becomes increasingly popular in machine learning and optimization communities. In this paper we have studied the convergence of FedAvg, a heuristic algorithm suitable for federated setting. We have investigated the influence of sampling and averaging schemes. We have provided theoretical guarantees for two schemes and empirically demonstrated their performances. Our work sheds light on theoretical understanding of FedAvg and provides insights for algorithm design in realistic applications. Though our analyses are constrained in convex problems, we hope our insights and proof techniques can inspire future work.

## ACKNOWLEDGEMENTS

Li, Yang and Zhang have been supported by the National Natural Science Foundation of China (No. 11771002 and 61572017), Beijing Natural Science Foundation (Z190001), the Key Project of MOST of China (No. 2018AAA0101000), and Beijing Academy of Artificial Intelligence (BAAI).

## REFERENCES

Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly Shmatikov. How to backdoor federated learning. arXiv preprint arXiv:1807.00459, 2018. 7

Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacypreserving machine learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, 2017. 7

Gregory Francis Coppola. Iterative parameter mixing for distributed large-margin training of structured predictors for natural language processing. PhD thesis, 2015. 4, 7

Robin C Geyer, Tassilo Klein, Moin Nabi, and SAP SE. Differentially private federated learning: A client level perspective. arXiv preprint arXiv:1712.07557, 2017. 7

Briland Hitaj, Giuseppe Ateniese, and Fernando Pérez-Cruz. Deep models under the GAN: information leakage from collaborative deep learning. In ACM SIGSAC Conference on Computer and Communications Security, 2017. 7

Mingyi Hong, Meisam Razaviyayn, and Jason Lee. Gradient primal-dual algorithm converges to second-order stationary solution for nonconvex distributed optimization over networks. In International Conference on Machine Learning (ICML), 2018. 8

Dusan Jakovetic. Distributed optimization: algorithms and convergence rates. PhD, Carnegie Mellon University, Pittsburgh PA, USA, 2013. 1

Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. First analysis of local gd on heterogeneous data. arXiv preprint arXiv:1909.04715, 2019. 2, 7

Jakub Konevcnỳ. Stochastic, distributed and federated optimization for machine learning. arXiv preprint arXiv:1707.01155, 2017. 1

Jakub Konevcnỳ, Brendan McMahan, and Daniel Ramage. Federated optimization: distributed optimization beyond the datacenter. arXiv preprint arXiv:1511.03575, 2015. 1

Jakub Konevcnỳ, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016. 7

Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998. 8, 24

Jason D Lee, Qiang Liu, Yuekai Sun, and Jonathan E Taylor. Communication-efficient sparse regression. The Journal of Machine Learning Research, 18(1):115-144, 2017. 8

Mu Li, David G Andersen, Jun Woo Park, Alexander J Smola, Amr Ahmed, Vanja Josifovski, James Long, Eugene J Shekita, and Bor-Yiing Su. Scaling distributed machine learning with the parameter server. In 11th \{USENIX\} Symposium on Operating Systems Design and Implementation (\{OSDI\} 14), pages 583-598, 2014a. 1

Mu Li, David G Andersen, Alexander J Smola, and Kai Yu. Communication efficient distributed machine learning with the parameter server. In Advances in Neural Information Processing Systems (NIPS), 2014b. 1

Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. arXiv preprint arXiv:1908.07873, 2019. 1

Shao-Bo Lin, Xin Guo, and Ding-Xuan Zhou. Distributed learning with regularized least squares. Journal of Machine Learning Research, 18(1):3202-3232, 2017. 8

Tao Lin, Sebastian U Stich, and Martin Jaggi. Don't use large mini-batches, use local sgd. arXiv preprint arXiv:1808.07217, 2018. 7

Dhruv Mahajan, Nikunj Agrawal, S Sathiya Keerthi, Sundararajan Sellamanickam, and Léon Bottou. An efficient distributed learning algorithm based on effective local functional approximations. Journal of Machine Learning Research, 19(1):2942-2978, 2018. 8

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2017. 1, 2, 3, 7, 17, 25

Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov. Exploiting unintended feature leakage in collaborative learning. In IEEE Symposium on Security \& Privacy (S\&P). IEEE, 2019. 7

Xiangrui Meng, Joseph Bradley, Burak Yavuz, Evan Sparks, Shivaram Venkataraman, Davies Liu, Jeremy Freeman, DB Tsai, Manish Amde, and Sean Owen. MLlib: machine learning in Apache Spark. Journal of Machine Learning Research, 17(34):1-7, 2016. 1

Yurii Nesterov. Introductory lectures on convex optimization: a basic course, volume 87. Springer Science \& Business Media, 2013. 7

Sashank J Reddi, Jakub Konecnỳ, Peter Richtárik, Barnabás Póczós, and Alex Smola. AIDE: fast and communication efficient distributed optimization. arXiv preprint arXiv:1608.06879, 2016. 1, 8

Peter Richtárik and Martin Takác. Distributed coordinate descent method for learning with big data. Journal of Machine Learning Research, 17(1):2657-2681, 2016. 1

Anit Kumar Sahu, Tian Li, Maziar Sanjabi, Manzil Zaheer, Ameet Talwalkar, and Virginia Smith. Federated optimization for heterogeneous networks. arXiv preprint arXiv:1812.06127, 2018. 1, 2, $3,5,7,8,17,24,25$

Felix Sattler, Simon Wiedemann, Klaus-Robert Müller, and Wojciech Samek. Robust and communication-efficient federated learning from non-iid data. arXiv preprint arXiv:1903.02891, 2019. 2,7

Ohad Shamir, Nati Srebro, and Tong Zhang. Communication-efficient distributed optimization using an approximate Newton-type method. In International conference on machine learning (ICML), 2014. 1,8

Reza Shokri and Vitaly Shmatikov. Privacy-preserving deep learning. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, 2015. 1

Shusen Wang, Farbod Roosta Khorasani, Peng Xu, and Michael W. Mahoney. GIANT: Globally improved approximate newton method for distributed optimization. In Conference on Neural Information Processing Systems (NeurIPS), 2018. 1, 8

Virginia Smith, Simone Forte, Chenxin Ma, Martin Takac, Michael I Jordan, and Martin Jaggi. CoCoA: A general framework for communication-efficient distributed optimization. arXiv preprint arXiv:1611.02189, 2016. 1, 8

Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar. Federated multi-task learning. In Advances in Neural Information Processing Systems (NIPS), 2017. 2, 7, 8

Sebastian U Stich. Local SGD converges fast and communicates little. arXiv preprint arXiv:1805.09767, 2018. 2, 4, 5, 6, 7, 12

Sebastian U Stich, Jean-Baptiste Cordonnier, and Martin Jaggi. Sparsified SGD with memory. In Advances in Neural Information Processing Systems (NIPS), pages 4447-4458, 2018. 5

Jianyu Wang and Gauri Joshi. Cooperative SGD: A unified framework for the design and analysis of communication-efficient SGD algorithms. arXiv preprint arXiv:1808.07576, 2018. 2, 4, 7

Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin K Leung, Christian Makaya, Ting He, and Kevin Chan. Adaptive federated learning in resource constrained edge computing systems. IEEE Journal on Selected Areas in Communications, 2019. 2, 4, 7

Shusen Wang. A sharper generalization bound for divide-and-conquer ridge regression. In The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI), 2019. 8

Blake E Woodworth, Jialei Wang, Adam Smith, Brendan McMahan, and Nati Srebro. Graph oracle models, lower bounds, and gaps for parallel stochastic optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2018. 2, 4

Cong Xie, Sanmi Koyejo, and Indranil Gupta. Asynchronous federated optimization. arXiv preprint arXiv:1903.03934, 2019. 7

Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted sgd with faster convergence and less communication: Demystifying why model averaging works for deep learning. In AAAI Conference on Artificial Intelligence, 2019. 2, 4, 5, 7

Sixin Zhang, Anna E Choromanska, and Yann LeCun. Deep learning with elastic averaging SGD. In Advances in Neural Information Processing Systems (NIPS), 2015a. 4

Yuchen Zhang and Xiao Lin. DiSCO: distributed optimization for self-concordant empirical loss. In International Conference on Machine Learning (ICML), 2015. 1, 8

Yuchen Zhang, John C. Duchi, and Martin J. Wainwright. Communication-efficient algorithms for statistical optimization. Journal of Machine Learning Research, 14:3321-3363, 2013. 5, 6, 8

Yuchen Zhang, John Duchi, and Martin Wainwright. Divide and conquer kernel ridge regression: a distributed algorithm with minimax optimal rates. Journal of Machine Learning Research, 16: $3299-3340,2015$ b. 8

Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated learning with non-iid data. arXiv preprint arXiv:1806.00582, 2018. 7

Shun Zheng, Fen Xia, Wei Xu, and Tong Zhang. A general distributed dual coordinate optimization framework for regularized loss minimization. arXiv preprint arXiv:1604.03763, 2016. 1

Fan Zhou and Guojing Cong. On the convergence properties of a k-step averaging stochastic gradient descent algorithm for nonconvex optimization. arXiv preprint arXiv:1708.01012, 2017. 2, 4, 7

Hankz Hankui Zhuo, Wenfeng Feng, Qian Xu, Qiang Yang, and Yufeng Lin. Federated reinforcement learning. arXiv preprint arXiv:1901.08277, 2019. 1
</end of paper 0>


<paper 1>
# First Analysis of Local GD on Heterogeneous Data 

Ahmed Khaled*<br>Cairo University<br>akregeb@gmail.com

Konstantin Mishchenko<br>$\mathrm{KAUST}^{\dagger}$<br>konstantin.mishchenko@kaust.edu.sa

Peter Richtárik<br>KAUST<br>peter.richtarik@kaust.edu.sa


#### Abstract

We provide the first convergence analysis of local gradient descent for minimizing the average of smooth and convex but otherwise arbitrary functions. Problems of this form and local gradient descent as a solution method are of importance in federated learning, where each function is based on private data stored by a user on a mobile device, and the data of different users can be arbitrarily heterogeneous. We show that in a low accuracy regime, the method has the same communication complexity as gradient descent.


## 1 Introduction

We are interested in solving the optimization problem

$$
\begin{equation*}
\min _{x \in \mathbb{R}^{d}}\left\{f(x) \stackrel{\text { def }}{=} \frac{1}{M} \sum_{m=1}^{M} f_{m}(x)\right\} \tag{1}
\end{equation*}
$$

which is arises in training of supervised machine learning models. We assume that each $f_{m}: \mathbb{R}^{d} \rightarrow$ $\mathbb{R}$ is an $L$-smooth and convex function and we denote by $x_{*}$ a fixed minimizer of $f$.

Our main interest is in situations where each function is based on data available on a single device only, and where the data distribution across the devices can be arbitrarily heterogeneous. This situation arises in federated learning, where machine learning models are trained on data available on consumer devices, such as mobile phones. In federated learning, transfer of local data to a single data center for centralized training is prohibited due to privacy reasons, and frequent communication is undesirable as it is expensive and intrusive. Hence, several recent works aim at constructing new ways of solving (1) in a distributed fashion with as few communication rounds as possible.

Large-scale problems are often solved by first-order methods as they have proved to scale well with both dimension and data size. One attractive choice is Local Gradient Descent, which divides the optimization process into epochs. Each epoch starts by communication in the form of a model averaging step across all $M$ devices ${ }^{3}$ The rest of each epoch does not involve any communication, and is devoted to performing a fixed number of gradient descent steps initiated from the average model, and based on the local functions, performed by all $M$ devices independently in parallel. See Algorithm 1 for more details.[^0]

```
Algorithm 1 Local Gradient Descent
Input: Stepsize $\gamma>0$, synchronization/communication times $0=t_{0} \leqslant t_{1} \leqslant t_{2} \leqslant \ldots$, initial
    vector $x_{0} \in \mathbb{R}^{d}$
    Initialize $x_{0}^{m}=x_{0}$ for all $m \in[M] \stackrel{\text { def }}{=}\{1,2, \ldots, M\}$
    for $t=0,1, \ldots$ do
        for $m=1, \ldots, M$ do
            $x_{t+1}^{m}= \begin{cases}\frac{1}{M} \sum_{j=1}^{M}\left(x_{t}^{j}-\gamma \nabla f_{j}\left(x_{t}^{j}\right)\right), & \text { if } t=t_{p} \text { for some } p \in\{1,2, \ldots\} \\ x_{t}^{m}-\gamma \nabla f_{m}\left(x_{t}^{m}\right), & \text { otherwise. }\end{cases}$
        end for
    end for
```

The stochastic version of this method is at the core of the Federated Averaging algorithm which has been used recently in federated learning applications, see e.g. [7, 10]. Essentially, Federated Averaging is a variant of local Stochastic Gradient Descent (SGD) with participating devices sampled randomly. This algorithm has been used in several machine learning applications such as mobile keyboard prediction [5], and strategies for improving its communication efficiency were explored in [7]. Despite its empirical success, little is known about convergence properties of this method and it has been observed to diverge when too many local steps are performed [10]. This is not so surprising as the majority of common assumptions are not satisfied; in particular, the data is typically very non-i.i.d. [10], so the local gradients can point in different directions. This property of the data can be written for any vector $x$ and indices $i, j$ as

$$
\left\|\nabla f_{i}(x)-\nabla f_{j}(x)\right\| \gg 1
$$

Unfortunately, it is very hard to analyze local methods without assuming a bound on the dissimilarity of $\nabla f_{i}(x)$ and $\nabla f_{j}(x)$. For this reason, almost all prior work assumed bounded dissimilarity [8, 16, 17, 18] and addressed other less challenging aspects of federated learning such as decentralized communication, nonconvexity of the objective or unbalanced data partitioning. In fact, a common way to make the analysis simple is to assume Lipschitzness of local functions, $\left\|\nabla f_{i}(x)\right\| \leqslant G$ for any $x$ and $i$. We argue that this assumption is pathological and should be avoided when seeking a meaningful convergence bound. First of all, in unconstrained strongly convex minimization this assumption cannot be satisfied, making the analysis in works like [14] questionable. Second, there exists at least one method, whose convergence is guaranteed under bounded gradients [6], but in practice the method diverges [3, 12].

Finally, under the bounded gradients assumption we have

$$
\begin{equation*}
\left\|\nabla f_{i}(x)-\nabla f_{j}(x)\right\| \leqslant\left\|\nabla f_{i}(x)\right\|+\left\|\nabla f_{j}(x)\right\| \leqslant 2 G \tag{2}
\end{equation*}
$$

In other words, we lose control over the difference between the functions. Since $G$ bounds not just dissimilarity, but also the gradients themselves, it makes the statements less insightful or even vacuous. For instance, it is not going to be tight if the data is actually i.i.d. since $G$ in that case will remain a positive constant. In contrast, we will show that the rate should depend on a much more meaningful quantity,

$$
\sigma^{2} \stackrel{\text { def }}{=} \frac{1}{M} \sum_{m=1}^{M}\left\|\nabla f_{m}\left(x_{*}\right)\right\|^{2}
$$

where $x_{*}$ is a minimizer of $f$. Obviously, $\sigma$ is always finite and it serves as a natural measure of variance in local methods. On top of that, it allows us to obtain bounds that are tight in case the data is actually i.i.d. We note that an attempt to get more general convergence statement has been made in [13], but sadly their guarantee is strictly worse than that of minibatch Stochastic Gradient Descent (SGD), making their theoretical contribution smaller.

We additionally note that the bound in the mentioned work [8] not only uses bounded gradients, but also provides a pessimistic $\mathcal{O}\left(H^{2} / T\right)$ rate, where $H$ is the number of local steps in each epoch, and $T$ is the total number of steps of the method. Indeed, this requires $H$ to be $\mathcal{O}(1)$ to make the

where averaging is done across all devices. We focus on this simpler situation first as even this is not currently understood theoretically.
rate coincide with that of SGD for strongly convex functions. The main contribution of that work, therefore, is in considering partial participation as in Federated Averaging.

When the data is identically distributed and stochastic gradients are used instead of full gradients on each node, the resulting method has been explored extensively in the literature under different names, see e.g. [1, 14, 15, 19]. [11] proposed an asynchronous local method that converges to the exact solution without decreasing stepsizes, but its benefit from increasing $H$ is limited by constant factors. [9] seems to be the first work to propose a local method, but no rate was shown in that work.

## 2 Convergence of Local GD

### 2.1 Assumptions and notation

Before introducing our main result, let us first formulate explicitly our assumptions.

Assumption 1. The set of minimizers of (1) is nonempty. Further, for every $m \in[M] \stackrel{\text { def }}{=}$ $\{1,2, \ldots, M\}, f_{m}$ is convex and $L$-smooth. That is, for all $x, y \in \mathbb{R}^{d}$ the following inequalities are satisfied:

$$
0 \leqslant f_{m}(x)-f_{m}(y)-\left\langle\nabla f_{m}(y), x-y\right\rangle \leqslant \frac{L}{2}\|x-y\|^{2}
$$

Further, we assume that Algorithm 1 is run with a bounded synchronization interval. That is, we assume that

$$
H \stackrel{\text { def }}{=} \max _{p \geqslant 0}\left|t_{p}-t_{p+1}\right|
$$

is finite. Given local vectors $x_{t}^{1}, x_{t}^{2}, \ldots, x_{t}^{M} \in \mathbb{R}^{d}$, we define the average iterate, iterate variance and average gradient at time $t$ as

$$
\begin{equation*}
\hat{x}_{t} \stackrel{\text { def }}{=} \frac{1}{M} \sum_{m=1}^{M} x_{t}^{m} \quad V_{t} \stackrel{\text { def }}{=} \frac{1}{M} \sum_{m=1}^{M}\left\|x_{t}^{m}-\hat{x}_{t}\right\|^{2} \quad g_{t} \stackrel{\text { def }}{=} \frac{1}{M} \sum_{m=1}^{M} \nabla f_{m}\left(x_{t}^{m}\right) \tag{3}
\end{equation*}
$$

respectively. The Bregman divergence with respect to $f$ is defined via

$$
D_{f}(x, y) \stackrel{\text { def }}{=} f(x)-f(y)-\langle\nabla f(y), x-y\rangle
$$

Note that in the case $y=x_{*}$, we have $D_{f}\left(x, x_{*}\right)=f(x)-f\left(x_{*}\right)$.

### 2.2 Analysis

The first lemma enables us to find a recursion on the optimality gap for a single step of local GD:

Lemma 1. Under Assumption 1 and for any $\gamma \geqslant 0$ we have

$$
\begin{equation*}
\left\|r_{t+1}\right\|^{2} \leqslant\left\|r_{t}\right\|^{2}+\gamma L(1+2 \gamma L) V_{t}-2 \gamma(1-2 \gamma L) D_{f}\left(\hat{x}_{t}, x_{*}\right) \tag{4}
\end{equation*}
$$

where $r_{t} \stackrel{\text { def }}{=} \hat{x}_{t}-x_{*}$. In particular, if $\gamma \leqslant \frac{1}{4 L}$, then $\left\|r_{t+1}\right\|^{2} \leqslant\left\|r_{t}\right\|^{2}+\frac{3}{2} \gamma L V_{t}-\gamma D_{f}\left(\hat{x}_{t}, x_{*}\right)$.

We now bound the sum of the variances $V_{t}$ over an epoch. An epoch-based bound is intuitively what we want since we are only interested in the points $\hat{x}_{t_{p}}$ produced at the end of each epoch.

Lemma 2. Suppose that Assumption 1 holds and let $p \in \mathbb{N}$, define $v=t_{p+1}-1$ and suppose Algorithm 1 is run with a synchronization interval $H \geqslant 1$ and a constant stepsize $\gamma>0$ such that $\gamma \leqslant \frac{1}{4 L H}$. Then the following inequalities hold:

$$
\begin{aligned}
& \sum_{t=t_{p}}^{v} V_{t} \leqslant 5 L \gamma^{2} H^{2} \sum_{i=t_{p}}^{v} D_{f}\left(\hat{x}_{i}, x_{*}\right)+\sum_{i=t_{p}}^{v} 8 \gamma^{2} H^{2} \sigma^{2} \\
& \sum_{t=t_{p}}^{v} \frac{3}{2} L V_{t}-D_{f}\left(\hat{x}_{t}, x_{*}\right) \leqslant-\frac{1}{2} \sum_{t=t_{p}}^{v} D_{f}\left(\hat{x}_{i}, x_{*}\right)+\sum_{t=t_{p}}^{v} 12 L \gamma^{2} H^{2} \sigma^{2}
\end{aligned}
$$

Combining the previous two lemmas, the convergence of local GD is established in the next theorem:

Theorem 1. For local GD run with a constant stepsize $\gamma>0$ such that $\gamma \leqslant \frac{1}{4 L H}$ and under Assumption 1, we have

$$
\begin{equation*}
f\left(\bar{x}_{T}\right)-f\left(x_{*}\right) \leqslant \frac{2\left\|x_{0}-x_{*}\right\|^{2}}{\gamma T}+24 \gamma^{2} \sigma^{2} H^{2} L \tag{5}
\end{equation*}
$$

where $\bar{x}_{T} \stackrel{\text { def }}{=} \frac{1}{T} \sum_{t=0}^{T-1} \hat{x}_{t}$.

### 2.3 Local GD vs GD

In order to interpret the above bound, we may ask: how many communication rounds are sufficient to guarantee $f\left(\bar{x}_{T}\right)-f\left(x_{*}\right) \leqslant \epsilon$ ? To answer this question, we need to minimize $\frac{T}{H}$ subject to the constraints $0<\gamma \leqslant \frac{1}{4 L}, \frac{2\left\|x_{0}-x_{*}\right\|^{2}}{\gamma T} \leqslant \frac{\epsilon}{2}$, and $24 \gamma^{2} \sigma^{2} H^{2} L \leqslant \frac{\epsilon}{2}$, in variables $T, H$ and $\gamma$. We can easily deduce from the constraints that

$$
\begin{equation*}
\frac{T}{H} \geqslant \frac{16\left\|x_{0}-x_{*}\right\|^{2}}{\epsilon} \max \left\{L, \sigma \sqrt{\frac{3 L}{\epsilon}}\right\} \tag{6}
\end{equation*}
$$

On the other hand, this lower bound is achieved by any $0<\gamma \leqslant \frac{1}{4 L}$ as long as we pick

$$
T=T(\gamma) \stackrel{\text { def }}{=} \frac{4\left\|x_{0}-x_{*}\right\|^{2}}{\epsilon \gamma} \quad \text { and } \quad H=H(\gamma) \stackrel{\text { def }}{=} \frac{1}{4 \max \left\{L, \sigma \sqrt{\frac{3 L}{\epsilon}}\right\} \gamma}
$$

The smallest $H$ achieving this lower bound is $H\left(\frac{1}{4 L}\right)=\min \left\{1, \sqrt{\frac{\epsilon L}{3 \sigma^{2}}}\right\}$.

Further, notice that as long as the target accuracy is not too high, in particular $\epsilon \geqslant \frac{3 \sigma^{2}}{L}$, then $\max \{L, \sigma \sqrt{3 L / \epsilon}\}=L$ and (6) says that the number of communications of local GD (with parameters set as $H=H(\gamma)$ and $T=T(\gamma))$ is equal to

$$
\frac{T}{H}=\mathcal{O}\left(\frac{L\left\|x_{0}-x_{*}\right\|^{2}}{\epsilon}\right)
$$

which is the same as the number of iterations (i.e., communications) of gradient descent. If $\epsilon<\frac{3 \sigma^{2}}{L}$, then (6) gives the communication complexity

$$
\frac{T}{H}=\mathcal{O}\left(\frac{\sqrt{L} \sigma}{\epsilon^{3 / 2}}\right)
$$

### 2.4 Local GD vs Minibatch SGD

Equation (5) shows a clear analogy between the convergence of local GD and the convergence rate of minibatch SGD, establishing a $1 / T$ convergence to a neighborhood depending on the expected noise at the optimum $\sigma^{2}$, which measures how dissimilar the functions $f_{m}$ are from each other at the optimum $x_{*}$.

The analogy between SGD and local GD extends further to the convergence rate, as the next corollary shows:

Corollary 1. Choose $H$ such that $H \leqslant \frac{\sqrt{T}}{\sqrt{M}}$, then $\gamma=\frac{\sqrt{M}}{4 L \sqrt{T}} \leqslant \frac{1}{4 H L}$, and hence applying the result of the previous theorem

$$
f\left(\bar{x}_{T}\right)-f\left(x_{*}\right) \leqslant \frac{8 L\left\|x_{0}-x_{*}\right\|^{2}}{\sqrt{M T}}+\frac{3 M \sigma^{2} H^{2}}{2 L T}
$$

To get a convergence rate of $1 / \sqrt{M T}$ we can choose $H=O\left(T^{1 / 4} M^{-3 / 4}\right)$, which implies a total number of $\Omega\left(T^{3 / 4} M^{3 / 4}\right)$ communication steps. If a rate of $1 / \sqrt{T}$ is desired instead, we can choose a larger $H=O\left(T^{1 / 4}\right)$.

## 3 Experiments

To verify the theory, we run our experiments on logistic regression with $\ell_{2}$ regularization and datasets taken from the LIBSVM library [2]. We use a machine with 24 Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz cores and we handle communication via the MPI for Python package [4].

Since our architecture leads to a very specific trade-off between computation and communication, we also provide plots for the case the communication time relative to gradient computation time is higher or lower. In all experiments, we use full gradients $\nabla f_{m}$ and constant stepsize $\frac{1}{L}$. The amount of $\ell_{2}$ regularization was chosen of order $\frac{1}{n}$, where $n$ is the total amount of data. The data partitioning is not i.i.d. and is done based on the index in the original dataset.

We observe a very tight match between our theory and numerical results. In cases where communication is significantly more expensive than gradient computation, local methods are much faster for imprecise convergence. This was not a big advantage though with our architecture, mainly because full gradients took a lot of time to be computed.
![](https://cdn.mathpix.com/cropped/2024_06_04_78e00d735d4b42453915g-05.jpg?height=320&width=1378&top_left_y=867&top_left_x=362)

Figure 1: Convergence of local GD methods with different number of local steps on the 'a5a' dataset. 1 local step corresponds to fully synchronized gradient descent and it is the only method that converges precisely to the optimum. The left plot shows convergence in terms of communication rounds, showing a clear advantage of local GD when only limited accuracy is required. The mid plot, however, illustrates that wall-clock time might improve only slightly and the right plot shows what changes with different communication cost.
![](https://cdn.mathpix.com/cropped/2024_06_04_78e00d735d4b42453915g-05.jpg?height=322&width=1378&top_left_y=1487&top_left_x=362)

Figure 2: Same experiment as in Figure 3, performed on the 'mushrooms' dataset.

## References

[1] Debraj Basu, Deepesh Data, Can Karakus, and Suhas Diggavi. Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification, and Local Computations. arXiv:1906.02367, 2019.

[2] Chih-Chung Chang and Chih-Jen Lin. LIBSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology (TIST), 2(3):27, 2011.

[3] Tatjana Chavdarova, Gauthier Gidel, François Fleuret, and Simon Lacoste-Julien. Reducing Noise in GAN Training with Variance Reduced Extragradient. arXiv preprint arXiv:1904.08598, 2019.

[4] Lisandro D. Dalcin, Rodrigo R. Paz, Pablo A. Kler, and Alejandro Cosimo. Parallel distributed computing using Python. Advances in Water Resources, 34(9):1124-1139, 2011.

[5] Andrew Hard, Kanishka Rao, Rajiv Mathews, Françoise Beaufays, Sean Augenstein, Hubert Eichner, Chloé Kiddon, and Daniel Ramage. Federated Learning for Mobile Keyboard Prediction. arXiv:1811.03604, 2018.

[6] Anatoli Juditsky, Arkadi Nemirovski, and Claire Tauvel. Solving variational Inequalities with Stochastic Mirror-Prox algorithm. Stochastic Systems, 1(1):17-58, 2011.

[7] Jakub Konečný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated Learning: Strategies for Improving Communication Efficiency. In NIPS Private Multi-Party Machine Learning Workshop, 2016.

[8] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the Convergence of FedAvg on Non-IID Data. arXiv:1907.02189, 2019.

[9] L. Mangasarian. Parallel Gradient Distribution in Unconstrained Optimization. SIAM Journal on Control and Optimization, 33(6):1916-1925, 1995.

[10] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. Proceedings of the 20 th International Conference on Artificial Intelligence and Statistics (AISTATS) 2017. JMLR: W\&CP volume 54, 2016.

[11] Konstantin Mishchenko, Franck Iutzeler, Jérôme Malick, and Massih-Reza Amini. A Delaytolerant Proximal-Gradient Algorithm for Distributed Learning. In International Conference on Machine Learning, pages 3584-3592, 2018.

[12] Konstantin Mishchenko, Dmitry Kovalev, Egor Shulgin, Peter Richtárik, and Yura Malitsky. Revisiting Stochastic Extragradient. arXiv preprint arXiv:1905.11373, 2019.

[13] Anit Kumar Sahu, Tian Li, Maziar Sanjabi, Manzil Zaheer, Ameet Talwalkar, and Virginia Smith. On the Convergence of Federated Optimization in Heterogeneous Networks. arXiv:1812.06127, 2018.

[14] Sebastian U. Stich. Local SGD Converges Fast and Communicates Little. arXiv:1805.09767, 2018 .

[15] Jianyu Wang and Gauri Joshi. Cooperative SGD: A Unified Framework for the Design and Analysis of Communication-Efficient SGD Algorithms. arXiv:1808.07576, 2018.

[16] Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin K. Leung, Christian Makaya, Ting He, and Kevin Chan. When Edge Meets Learning: Adaptive Control for Resource-Constrained Distributed Machine Learning. arXiv:1804.05271, 2018.

[17] Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel Restarted SGD with Faster Convergence and Less Communication: Demystifying Why Model Averaging Works for Deep Learning. arXiv:1807.06629, 2018.

[18] Hao Yu, Rong Jin, and Sen Yang. On the Linear Speedup Analysis of Communication Efficient Momentum SGD for Distributed Non-Convex Optimization. arXiv preprint arXiv:1905.03817, 2019.

[19] Fan Zhou and Guojing Cong. On the Convergence Properties of a $k$-step Averaging Stochastic Gradient Descent Algorithm for Nonconvex Optimization. In IJCAI International Joint Conference on Artificial Intelligence, volume 2018-July, pages 3219-3227, 2018.
</end of paper 1>


<paper 2>
# ON THE ConVERGENCE OF FEDAVG ON NON-IID DATA 

\author{
Xiang Li ${ }^{*}$ <br> School of Mathematical Sciences <br> Peking University <br> Beijing, 100871, China <br> smslixiang@pku.edu.cn <br> Wenhao Yang* <br> Center for Data Science <br> Peking University <br> Beijing, 100871, China <br> yangwenhaosms@pku.edu.cn

## Zhihua Zhang

 <br> School of Mathematical Sciences <br> Peking University <br> Beijing, 100871, China <br> zhzhang@math.pku.edu.cn}

Kaixuan Huang ${ }^{*}$<br>School of Mathematical Sciences<br>Peking University<br>Beijing, 100871, China<br>hackyhuang@pku.edu.cn<br>Shusen Wang<br>Department of Computer Science<br>Stevens Institute of Technology<br>Hoboken, NJ 07030, USA<br>shusen.wang@stevens.edu


#### Abstract

Federated learning enables a large amount of edge computing devices to jointly learn a model without data sharing. As a leading algorithm in this setting, Federated Averaging (FedAvg) runs Stochastic Gradient Descent (SGD) in parallel on a small subset of the total devices and averages the sequences only once in a while. Despite its simplicity, it lacks theoretical guarantees under realistic settings. In this paper, we analyze the convergence of FedAvg on non-iid data and establish a convergence rate of $\mathcal{O}\left(\frac{1}{T}\right)$ for strongly convex and smooth problems, where $T$ is the number of SGDs. Importantly, our bound demonstrates a trade-off between communicationefficiency and convergence rate. As user devices may be disconnected from the server, we relax the assumption of full device participation to partial device participation and study different averaging schemes; low device participation rate can be achieved without severely slowing down the learning. Our results indicates that heterogeneity of data slows down the convergence, which matches empirical observations. Furthermore, we provide a necessary condition for FedAvg on non-iid data: the learning rate $\eta$ must decay, even if full-gradient is used; otherwise, the solution will be $\Omega(\eta)$ away from the optimal.


## 1 INTRODUCTION

Federated Learning (FL), also known as federated optimization, allows multiple parties to collaboratively train a model without data sharing (Konevenỳ et al., 2015; Shokri and Shmatikov, 2015; McMahan et al., 2017; Konevcnỳ, 2017; Sahu et al., 2018; Zhuo et al., 2019). Similar to the centralized parallel optimization (Jakovetic, 2013; Li et al., 2014a;b; Shamir et al., 2014; Zhang and Lin, 2015; Meng et al., 2016; Reddi et al., 2016; Richtárik and Takác, 2016; Smith et al., 2016; Zheng et al., 2016; Shusen Wang et al., 2018), FL let the user devices (aka worker nodes) perform most of the computation and a central parameter server update the model parameters using the descending directions returned by the user devices. Nevertheless, FL has three unique characters that distinguish it from the standard parallel optimization Li et al. (2019).[^0]

First, the training data are massively distributed over an incredibly large number of devices, and the connection between the central server and a device is slow. A direct consequence is the slow communication, which motivated communication-efficient FL algorithms (McMahan et al., 2017; Smith et al., 2017; Sahu et al., 2018; Sattler et al., 2019). Federated averaging (FedAvg) is the first and perhaps the most widely used FL algorithm. It runs $E$ steps of SGD in parallel on a small sampled subset of devices and then averages the resulting model updates via a central server once in a while. ${ }^{1}$ In comparison with SGD and its variants, FedAvg performs more local computation and less communication.

Second, unlike the traditional distributed learning systems, the FL system does not have control over users' devices. For example, when a mobile phone is turned off or WiFi access is unavailable, the central server will lose connection to this device. When this happens during training, such a non-responding/inactive device, which is called a straggler, appears tremendously slower than the other devices. Unfortunately, since it has no control over the devices, the system can do nothing but waiting or ignoring the stragglers. Waiting for all the devices' response is obviously infeasible; it is thus impractical to require all the devices be active.

Third, the training data are non-iid ${ }^{2}$, that is, a device's local data cannot be regarded as samples drawn from the overall distribution. The data available locally fail to represent the overall distribution. This does not only bring challenges to algorithm design but also make theoretical analysis much harder. While FedAvg actually works when the data are non-iid McMahan et al. (2017), FedAvg on non-iid data lacks theoretical guarantee even in convex optimization setting.

There have been much efforts developing convergence guarantees for FL algorithm based on the assumptions that (1) the data are iid and (2) all the devices are active. Khaled et al. (2019); Yu et al. (2019); Wang et al. (2019) made the latter assumption, while Zhou and Cong (2017); Stich (2018); Wang and Joshi (2018); Woodworth et al. (2018) made both assumptions. The two assumptions violates the second and third characters of FL. Previous algorithm Fedprox Sahu et al. (2018) doesn't require the two mentioned assumptions and incorporates FedAvg as a special case when the added proximal term vanishes. However, their theory fails to cover FedAvg.

Notation. Let $N$ be the total number of user devices and $K(\leq N)$ be the maximal number of devices that participate in every round's communication. Let $T$ be the total number of every device's SGDs, $E$ be the number of local iterations performed in a device between two communications, and thus $\frac{T}{E}$ is the number of communications.

Contributions. For strongly convex and smooth problems, we establish a convergence guarantee for FedAvg without making the two impractical assumptions: (1) the data are iid, and (2) all the devices are active. To the best of our knowledge, this work is the first to show the convergence rate of FedAvg without making the two assumptions.

We show in Theorem 1, 2, and 3 that FedAvg has $\mathcal{O}\left(\frac{1}{T}\right)$ convergence rate. In particular, Theorem 3 shows that to attain a fixed precision $\epsilon$, the number of communications is

$$
\begin{equation*}
\frac{T}{E}=\mathcal{O}\left[\frac{1}{\epsilon}\left(\left(1+\frac{1}{K}\right) E G^{2}+\frac{\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+\Gamma+G^{2}}{E}+G^{2}\right)\right] \tag{1}
\end{equation*}
$$

Here, $G, \Gamma, p_{k}$, and $\sigma_{k}$ are problem-related constants defined in Section 3.1. The most interesting insight is that $E$ is a knob controlling the convergence rate: neither setting $E$ over-small ( $E=1$ makes FedAvg equivalent to SGD) nor setting $E$ over-large is good for the convergence.

This work also makes algorithmic contributions. We summarize the existing sampling ${ }^{3}$ and averaging schemes for FedAvg (which do not have convergence bounds before this work) and propose a new scheme (see Table 1). We point out that a suitable sampling and averaging scheme is crucial for the convergence of FedAvg. To the best of our knowledge, we are the first to theoretically demonstrate[^1]

Table 1: Sampling and averaging schemes for FedAvg. $\mathcal{S}_{t} \sim \mathcal{U}(N, K)$ means $\mathcal{S}_{t}$ is a size- $K$ subset uniformly sampled without replacement from $[N] . \mathcal{S}_{t} \sim \mathcal{W}(N, K, \mathbf{p})$ means $\mathcal{S}_{t}$ contains $K$ elements that are iid sampled with replacement from $[N]$ with probabilities $\left\{p_{k}\right\}$. In the latter scheme, $\mathcal{S}_{t}$ is not a set.

| Paper | Sampling | Averaging | Convergence rate |
| :---: | :---: | :---: | :---: |
| McMahan et al. (2017) | $\mathcal{S}_{t} \sim \mathcal{U}(N, K)$ | $\sum_{k \notin \mathcal{S}_{t}} p_{k} \mathbf{w}_{t}+\sum_{k \in \mathcal{S}_{t}} p_{k} \mathbf{w}_{t}^{k}$ | - |
| Sahu et al. (2018) | $\mathcal{S}_{t} \sim \mathcal{W}(N, K, \mathbf{p})$ | $\frac{1}{K} \sum_{k \in \mathcal{S}_{t} \mathbf{w}_{t}^{k}}$ | $\mathcal{O}\left(\frac{1}{T}\right)^{5}$ |
| Ours | $\mathcal{S}_{t} \sim \mathcal{U}(N, K)$ | $\sum_{k \in \mathcal{S}_{t}} p_{k} \frac{N}{K} \mathbf{w}_{t}^{k}$ | $\mathcal{O}\left(\frac{1}{T}\right)^{6}$ |

that FedAvg with certain schemes (see Table 1) can achieve $\mathcal{O}\left(\frac{1}{T}\right)$ convergence rate in non-iid federated setting. We show that heterogeneity of training data and partial device participation slow down the convergence. We empirically verify our results through numerical experiments.

Our theoretical analysis requires the decay of learning rate (which is known to hinder the convergence rate.) Unfortunately, we show in Theorem 4 that the decay of learning rate is necessary for FedAvg with $E>1$, even if full gradient descent is used. ${ }^{4}$ If the learning rate is fixed to $\eta$ throughout, FedAvg would converge to a solution at least $\Omega(\eta(E-1))$ away from the optimal. To establish Theorem 4 , we construct a specific $\ell_{2}$-norm regularized linear regression model which satisfies our strong convexity and smoothness assumptions.

Paper organization. In Section 2, we elaborate on FedAvg. In Section 3, we present our main convergence bounds for FedAvg. In Section 4, we construct a special example to show the necessity of learning rate decay. In Section 5, we discuss and compare with prior work. In Section 6, we conduct empirical study to verify our theories. All the proofs are left to the appendix.

## 2 FEDERATED AVERAGING (FedAvg)

Problem formulation. In this work, we consider the following distributed optimization model:

$$
\begin{equation*}
\min _{\mathbf{w}}\left\{F(\mathbf{w}) \triangleq \sum_{k=1}^{N} p_{k} F_{k}(\mathbf{w})\right\} \tag{2}
\end{equation*}
$$

where $N$ is the number of devices, and $p_{k}$ is the weight of the $k$-th device such that $p_{k} \geq 0$ and $\sum_{k=1}^{N} p_{k}=1$. Suppose the $k$-th device holds the $n_{k}$ training data: $x_{k, 1}, x_{k, 2}, \cdots, x_{k, n_{k}}$. The local objective $F_{k}(\cdot)$ is defined by

$$
\begin{equation*}
F_{k}(\mathbf{w}) \triangleq \frac{1}{n_{k}} \sum_{j=1}^{n_{k}} \ell\left(\mathbf{w} ; x_{k, j}\right) \tag{3}
\end{equation*}
$$

where $\ell(\cdot ; \cdot)$ is a user-specified loss function.

Algorithm description. Here, we describe one around (say the $t$-th) of the standard FedAvg algorithm. First, the central server broadcasts the latest model, $\mathbf{w}_{t}$, to all the devices. Second, every device (say the $k$-th) lets $\mathbf{w}_{t}^{k}=\mathbf{w}_{t}$ and then performs $E(\geq 1)$ local updates:

$$
\mathbf{w}_{t+i+1}^{k} \longleftarrow \mathbf{w}_{t+i}^{k}-\eta_{t+i} \nabla F_{k}\left(\mathbf{w}_{t+i}^{k}, \xi_{t+i}^{k}\right), i=0,1, \cdots, E-1
$$

where $\eta_{t+i}$ is the learning rate (a.k.a. step size) and $\xi_{t+i}^{k}$ is a sample uniformly chosen from the local data. Last, the server aggregates the local models, $\mathbf{w}_{t+E}^{1}, \cdots, \mathbf{w}_{t+E}^{N}$, to produce the new global model, $\mathbf{w}_{t+E}$. Because of the non-iid and partial device participation issues, the aggregation step can vary.[^2]

IID versus non-iid. Suppose the data in the $k$-th device are i.i.d. sampled from the distribution $\mathcal{D}_{k}$. Then the overall distribution is a mixture of all local data distributions: $\mathcal{D}=\sum_{k=1}^{N} p_{k} \mathcal{D}_{k}$. The prior work Zhang et al. (2015a); Zhou and Cong (2017); Stich (2018); Wang and Joshi (2018); Woodworth et al. (2018) assumes the data are iid generated by or partitioned among the $N$ devices, that is, $\mathcal{D}_{k}=\mathcal{D}$ for all $k \in[N]$. However, real-world applications do not typically satisfy the iid assumption. One of our theoretical contributions is avoiding making the iid assumption.

Full device participation. The prior work Coppola (2015); Zhou and Cong (2017); Stich (2018); Yu et al. (2019); Wang and Joshi (2018); Wang et al. (2019) requires the full device participation in the aggregation step of FedAvg. In this case, the aggregation step performs

$$
\mathbf{w}_{t+E} \longleftarrow \sum_{k=1}^{N} p_{k} \mathbf{w}_{t+E}^{k}
$$

Unfortunately, the full device participation requirement suffers from serious "straggler's effect" (which means everyone waits for the slowest) in real-world applications. For example, if there are thousands of users' devices in the FL system, there are always a small portion of devices offline. Full device participation means the central server must wait for these "stragglers", which is obviously unrealistic.

Partial device participation. This strategy is much more realistic because it does not require all the devices' output. We can set a threshold $K(1 \leq K<N)$ and let the central server collect the outputs of the first $K$ responded devices. After collecting $K$ outputs, the server stops waiting for the rest; the $K+1$-th to $N$-th devices are regarded stragglers in this iteration. Let $\mathcal{S}_{t}\left(\left|\mathcal{S}_{t}\right|=K\right)$ be the set of the indices of the first $K$ responded devices in the $t$-th iteration. The aggregation step performs

$$
\mathbf{w}_{t+E} \longleftarrow \frac{N}{K} \sum_{k \in \mathcal{S}_{t}} p_{k} \mathbf{w}_{t+E}^{k}
$$

It can be proved that $\frac{N}{K} \sum_{k \in \mathcal{S}_{t}} p_{k}$ equals one in expectation.

Communication cost. The FedAvg requires two rounds communications- one broadcast and one aggregation- per $E$ iterations. If $T$ iterations are performed totally, then the number of communications is $\left\lfloor\frac{2 T}{E}\right\rfloor$. During the broadcast, the central server sends $\mathbf{w}_{t}$ to all the devices. During the aggregation, all or part of the $N$ devices sends its output, say $\mathbf{w}_{t+E}^{k}$, to the server.

## 3 ConVERGENCE ANALYSIS OF FedAvg IN NON-IID SETTING

In this section, we show that FedAvg converges to the global optimum at a rate of $\mathcal{O}(1 / T)$ for strongly convex and smooth functions and non-iid data. The main observation is that when the learning rate is sufficiently small, the effect of $E$ steps of local updates is similar to one step update with a larger learning rate. This coupled with appropriate sampling and averaging schemes would make each global update behave like an SGD update. Partial device participation $(K<N)$ only makes the averaged sequence $\left\{\mathbf{w}_{t}\right\}$ have a larger variance, which, however, can be controlled by learning rates. These imply the convergence property of FedAvg should not differ too much from SGD. Next, we will first give the convergence result with full device participation (i.e., $K=N$ ) and then extend this result to partial device participation (i.e., $K<N$ ).

### 3.1 NOTATION AND ASSUMPTIONS

We make the following assumptions on the functions $F_{1}, \cdots, F_{N}$. Assumption 1 and 2 are standard; typical examples are the $\ell_{2}$-norm regularized linear regression, logistic regression, and softmax classifier.

Assumption 1. $F_{1}, \cdots, F_{N}$ are all $L$-smooth: for all $\mathbf{v}$ and $\mathbf{w}, F_{k}(\mathbf{v}) \leq F_{k}(\mathbf{w})+(\mathbf{v}-$ $\mathbf{w})^{T} \nabla F_{k}(\mathbf{w})+\frac{L}{2}\|\mathbf{v}-\mathbf{w}\|_{2}^{2}$.

Assumption 2. $F_{1}, \cdots, F_{N}$ are all $\mu$-strongly convex: for all $\mathbf{v}$ and $\mathbf{w}, F_{k}(\mathbf{v}) \geq F_{k}(\mathbf{w})+(\mathbf{v}-$ $\mathbf{w})^{T} \nabla F_{k}(\mathbf{w})+\frac{\mu}{2}\|\mathbf{v}-\mathbf{w}\|_{2}^{2}$.

Assumptions 3 and 4 have been made by the works Zhang et al. (2013); Stich (2018); Stich et al. (2018); Yu et al. (2019).

Assumption 3. Let $\xi_{t}^{k}$ be sampled from the $k$-th device's local data uniformly at random. The variance of stochastic gradients in each device is bounded: $\mathbb{E}\left\|\nabla F_{k}\left(\mathbf{w}_{t}^{k}, \xi_{t}^{k}\right)-\nabla F_{k}\left(\mathbf{w}_{t}^{k}\right)\right\|^{2} \leq \sigma_{k}^{2}$ for $k=1, \cdots, N$.

Assumption 4. The expected squared norm of stochastic gradients is uniformly bounded, i.e., $\mathbb{E}\left\|\nabla F_{k}\left(\mathbf{w}_{t}^{k}, \xi_{t}^{k}\right)\right\|^{2} \leq G^{2}$ for all $k=1, \cdots, N$ and $t=1, \cdots, T-1$

Quantifying the degree of non-iid (heterogeneity). Let $F^{*}$ and $F_{k}^{*}$ be the minimum values of $F$ and $F_{k}$, respectively. We use the term $\Gamma=F^{*}-\sum_{k=1}^{N} p_{k} F_{k}^{*}$ for quantifying the degree of non-iid. If the data are iid, then $\Gamma$ obviously goes to zero as the number of samples grows. If the data are non-iid, then $\Gamma$ is nonzero, and its magnitude reflects the heterogeneity of the data distribution.

### 3.2 ConVERGENCE ReSULT: FULL DeVICE PARTICIPATION

Here we analyze the case that all the devices participate in the aggregation step; see Section 2 for the algorithm description. Let the FedAvg algorithm terminate after $T$ iterations and return $\mathbf{w}_{T}$ as the solution. We always require $T$ is evenly divisible by $E$ so that FedAvg can output $\mathbf{w}_{T}$ as expected.

Theorem 1. Let Assumptions 1 to 4 hold and $L, \mu, \sigma_{k}, G$ be defined therein. Choose $\kappa=\frac{L}{\mu}$, $\gamma=\max \{8 \kappa, E\}$ and the learning rate $\eta_{t}=\frac{2}{\mu(\gamma+t)}$. Then $F e d A v g$ with full device participation satisfies

$$
\begin{equation*}
\mathbb{E}\left[F\left(\mathbf{w}_{T}\right)\right]-F^{*} \leq \frac{\kappa}{\gamma+T-1}\left(\frac{2 B}{\mu}+\frac{\mu \gamma}{2} \mathbb{E}\left\|\mathbf{w}_{1}-\mathbf{w}^{*}\right\|^{2}\right) \tag{4}
\end{equation*}
$$

where

$$
\begin{equation*}
B=\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+6 L \Gamma+8(E-1)^{2} G^{2} \tag{5}
\end{equation*}
$$

### 3.3 ConVERGENCE ReSUlT: Partial DeVICE PARTICIPATION

As discussed in Section 2, partial device participation has more practical interest than full device participation. Let the set $\mathcal{S}_{t}(\subset[N])$ index the active devices in the $t$-th iteration. To establish the convergence bound, we need to make assumptions on $\mathcal{S}_{t}$.

Assumption 5 assumes the $K$ indices are selected from the distribution $p_{k}$ independently and with replacement. The aggregation step is simply averaging. This is first proposed in (Sahu et al., 2018), but they did not provide theoretical analysis.

Assumption 5 (Scheme I). Assume $\mathcal{S}_{t}$ contains a subset of $K$ indices randomly selected with replacement according to the sampling probabilities $p_{1}, \cdots, p_{N}$. The aggregation step of FedAvg performs $\mathbf{w}_{t} \longleftarrow \frac{1}{K} \sum_{k \in \mathcal{S}_{t}} \mathbf{w}_{t}^{k}$.

Theorem 2. Let Assumptions 1 to 4 hold and $L, \mu, \sigma_{k}, G$ be defined therein. Let $\kappa, \gamma, \eta_{t}$, and $B$ be defined in Theorem 1. Let Assumption 5 hold and define $C=\frac{4}{K} E^{2} G^{2}$. Then

$$
\begin{equation*}
\mathbb{E}\left[F\left(\mathbf{w}_{T}\right)\right]-F^{*} \leq \frac{\kappa}{\gamma+T-1}\left(\frac{2(B+C)}{\mu}+\frac{\mu \gamma}{2} \mathbb{E}\left\|\mathbf{w}_{1}-\mathbf{w}^{*}\right\|^{2}\right) \tag{6}
\end{equation*}
$$

Alternatively, we can select $K$ indices from $[N]$ uniformly at random without replacement. As a consequence, we need a different aggregation strategy. Assumption 6 assumes the $K$ indices are selected uniformly without replacement and the aggregation step is the same as in Section 2. However, to guarantee convergence, we require an additional assumption of balanced data.

Assumption 6 (Scheme II). Assume $\mathcal{S}_{t}$ contains a subset of $K$ indices uniformly sampled from $[N]$ without replacement. Assume the data is balanced in the sense that $p_{1}=\cdots=p_{N}=\frac{1}{N}$. The aggregation step of FedAvg performs $\mathbf{w}_{t} \longleftarrow \frac{N}{K} \sum_{k \in \mathcal{S}_{t}} p_{k} \mathbf{w}_{t}^{k}$.

Theorem 3. Replace Assumption 5 by Assumption 6 and $C$ by $C=\frac{N-K}{N-1} \frac{4}{K} E^{2} G^{2}$. Then the same bound in Theorem 2 holds.

Scheme II requires $p_{1}=\cdots=p_{N}=\frac{1}{N}$ which obviously violates the unbalance nature of FL. Fortunately, this can be addressed by the following transformation. Let $\widetilde{F}_{k}(\mathbf{w})=p_{k} N F_{k}(\mathbf{w})$ be a scaled local objective $F_{k}$. Then the global objective becomes a simple average of all scaled local objectives:

$$
F(\mathbf{w})=\sum_{k=1}^{N} p_{k} F_{k}(\mathbf{w})=\frac{1}{N} \sum_{k=1}^{N} \widetilde{F}_{k}(\mathbf{w})
$$

Theorem 3 still holds if $L, \mu, \sigma_{k}, G$ are replaced by $\widetilde{L} \triangleq \nu L, \widetilde{\mu} \triangleq \varsigma \mu, \widetilde{\sigma}_{k}=\sqrt{\nu} \sigma$, and $\widetilde{G}=\sqrt{\nu} G$, respectively. Here, $\nu=N \cdot \max _{k} p_{k}$ and $\varsigma=N \cdot \min _{k} p_{k}$.

### 3.4 DISCUSSIONS

Choice of $E$. Since $\left\|\mathbf{w}_{0}-\mathbf{w}^{*}\right\|^{2} \leq \frac{4}{\mu^{2}} G^{2}$ for $\mu$-strongly convex $F$, the dominating term in eqn. (6) is

$$
\begin{equation*}
\mathcal{O}\left(\frac{\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+L \Gamma+\left(1+\frac{1}{K}\right) E^{2} G^{2}+\gamma G^{2}}{\mu T}\right) \tag{7}
\end{equation*}
$$

Let $T_{\epsilon}$ denote the number of required steps for FedAvg to achieve an $\epsilon$ accuracy. It follows from eqn. (7) that the number of required communication rounds is roughly ${ }^{7}$

$$
\begin{equation*}
\frac{T_{\epsilon}}{E} \propto\left(1+\frac{1}{K}\right) E G^{2}+\frac{\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+L \Gamma+\kappa G^{2}}{E}+G^{2} \tag{8}
\end{equation*}
$$

Thus, $\frac{T_{E}}{E}$ is a function of $E$ that first decreases and then increases, which implies that over-small or over-large $E$ may lead to high communication cost and that the optimal $E$ exists.

Stich (2018) showed that if the data are iid, then $E$ can be set to $\mathcal{O}(\sqrt{T})$. However, this setting does not work if the data are non-iid. Theorem 1 implies that $E$ must not exceed $\Omega(\sqrt{T})$; otherwise, convergence is not guaranteed. Here we give an intuitive explanation. If $E$ is set big, then $\mathbf{w}_{t}^{k}$ can converge to the minimizer of $F_{k}$, and thus FedAvg becomes the one-shot average Zhang et al. (2013) of the local solutions. If the data are non-iid, the one-shot averaging does not work because weighted average of the minimizers of $F_{1}, \cdots, F_{N}$ can be very different from the minimizer of $F$.

Choice of $K$. Stich (2018) showed that if the data are iid, the convergence rate improves substantially as $K$ increases. However, under the non-iid setting, the convergence rate has a weak dependence on $K$, as we show in Theorems 2 and 3. This implies FedAvg is unable to achieve linear speedup. We have empirically observed this phenomenon (see Section 6). Thus, in practice, the participation ratio $\frac{K}{N}$ can be set small to alleviate the straggler's effect without affecting the convergence rate.

Choice of sampling schemes. We considered two sampling and averaging schemes in Theorems 2 and 3. Scheme I selects $K$ devices according to the probabilities $p_{1}, \cdots, p_{N}$ with replacement. The non-uniform sampling results in faster convergence than uniform sampling, especially when $p_{1}, \cdots, p_{N}$ are highly non-uniform. If the system can choose to activate any of the $N$ devices at any time, then Scheme I should be used.

However, oftentimes the system has no control over the sampling; instead, the server simply uses the first $K$ returned results for the update. In this case, we can assume the $K$ devices are uniformly sampled from all the $N$ devices and use Theorem 3 to guarantee the convergence. If $p_{1}, \cdots, p_{N}$ are highly non-uniform, then $\nu=N \cdot \max _{k} p_{k}$ is big and $\varsigma=N \cdot \min _{k} p_{k}$ is small, which makes the convergence of FedAvg slow. This point of view is empirically verified in our experiments.

## 4 NECESSITY OF LEARNING RATE DECAY

In this section, we point out that diminishing learning rates are crucial for the convergence of FedAvg in the non-iid setting. Specifically, we establish the following theorem by constructing a ridge regression model (which is strongly convex and smooth).[^3]

Theorem 4. We artificially construct a strongly convex and smooth distributed optimization problem. With full batch size, $E>1$, and any fixed step size, FedAvg will converge to sub-optimal points. Specifically, let $\tilde{\mathbf{w}}^{*}$ be the solution produced by FedAvg with a small enough and constant $\eta$, and $\mathrm{w}^{*}$ the optimal solution. Then we have

$$
\left\|\tilde{\mathbf{w}}^{*}-\mathbf{w}^{*}\right\|_{2}=\Omega((E-1) \eta) \cdot\left\|\mathbf{w}^{*}\right\|_{2}
$$

where we hide some problem dependent constants.

Theorem 4 and its proof provide several implications. First, the decay of learning rate is necessary of FedAvg. On the one hand, Theorem 1 shows with $E>1$ and a decaying learning rate, FedAvg converges to the optimum. On the other hand, Theorem 4 shows that with $E>1$ and any fixed learning rate, FedAvg does not converges to the optimum.

Second, FedAvg behaves very differently from gradient descent. Note that FedAvg with $E=1$ and full batch size is exactly the Full Gradient Descent; with a proper and fixed learning rate, its global convergence to the optimum is guaranteed Nesterov (2013). However, Theorem 4 shows that FedAvg with $E>1$ and full batch size cannot possibly converge to the optimum. This conclusion doesn't contradict with Theorem 1 in Khaled et al. (2019), which, when translated into our case, asserts that $\tilde{\mathbf{w}}^{*}$ will locate in the neighborhood of $\mathbf{w}^{*}$ with a constant learning rate.

Third, Theorem 4 shows the requirement of learning rate decay is not an artifact of our analysis; instead, it is inherently required by FedAvg. An explanation is that constant learning rates, combined with $E$ steps of possibly-biased local updates, form a sub-optimal update scheme, but a diminishing learning rate can gradually eliminate such bias.

The efficiency of FedAvg principally results from the fact that it performs several update steps on a local model before communicating with other workers, which saves communication. Diminishing step sizes often hinders fast convergence, which may counteract the benefit of performing multiple local updates. Theorem 4 motivates more efficient alternatives to FedAvg.

## 5 RELATED WORK

Federated learning (FL) was first proposed by McMahan et al. (2017) for collaboratively learning a model without collecting users' data. The research work on FL is focused on the communicationefficiency Konevcnỳ et al. (2016); McMahan et al. (2017); Sahu et al. (2018); Smith et al. (2017) and data privacy Bagdasaryan et al. (2018); Bonawitz et al. (2017); Geyer et al. (2017); Hitaj et al. (2017); Melis et al. (2019). This work is focused on the communication-efficiency issue.

FedAvg, a synchronous distributed optimization algorithm, was proposed by McMahan et al. (2017) as an effective heuristic. Sattler et al. (2019); Zhao et al. (2018) studied the non-iid setting, however, they do not have convergence rate. A contemporaneous and independent work Xie et al. (2019) analyzed asynchronous FedAvg; while they did not require iid data, their bound do not guarantee convergence to saddle point or local minimum. Sahu et al. (2018) proposed a federated optimization framework called FedProx to deal with statistical heterogeneity and provided the convergence guarantees in non-iid setting. FedProx adds a proximal term to each local objective. When these proximal terms vanish, FedP rox is reduced to FedAvg. However, their convergence theory requires the proximal terms always exist and hence fails to cover FedAvg.

When data are iid distributed and all devices are active, FedAvg is referred to as LocalSGD. Due to the two assumptions, theoretical analysis of LocalSGD is easier than FedAvg. Stich (2018) demonstrated LocalSGD provably achieves the same linear speedup with strictly less communication for strongly-convex stochastic optimization. Coppola (2015); Zhou and Cong (2017); Wang and Joshi (2018) studied LocalSGD in the non-convex setting and established convergence results. Yu et al. (2019); Wang et al. (2019) recently analyzed LocalSGD for non-convex functions in heterogeneous settings. In particular, Yu et al. (2019) demonstrated LocalSGD also achieves $\mathcal{O}(1 / \sqrt{N T})$ convergence (i.e., linear speedup) for non-convex optimization. Lin et al. (2018) empirically shows variants of LocalSGD increase training efficiency and improve the generalization performance of large batch sizes while reducing communication. For LocalGD on non-iid data (as opposed to LocalSGD), the best result is by the contemporaneous work (but slightly later than our first version) (Khaled et al., 2019). Khaled et al. (2019) used fixed learning rate $\eta$ and showed $\mathcal{O}\left(\frac{1}{T}\right)$
convergence to a point $\mathcal{O}\left(\eta^{2} E^{2}\right)$ away from the optimal. In fact, the suboptimality is due to their fixed learning rate. As we show in Theorem 4, using a fixed learning rate $\eta$ throughout, the solution by LocalGD is at least $\Omega((E-1) \eta)$ away from the optimal.

If the data are iid, distributed optimization can be efficiently solved by the second-order algorithms Mahajan et al. (2018); Reddi et al. (2016); Shamir et al. (2014); Shusen Wang et al. (2018); Zhang and Lin (2015) and the one-shot methods Lee et al. (2017); Lin et al. (2017); Wang (2019); Zhang et al. (2013; 2015b). The primal-dual algorithms Hong et al. (2018); Smith et al. (2016; 2017) are more generally applicable and more relevant to FL.

## 6 NUMERICAL EXPERIMENTS

Models and datasets We examine our theoretical results on a logistic regression with weight decay $\lambda=1 e-4$. This is a stochastic convex optimization problem. We distribute MNIST dataset (LeCun et al., 1998) among $N=100$ workers in a non-iid fashion such that each device contains samples of only two digits. We further obtain two datasets: mnist balanced and mnist unbalanced. The former is balanced such that the number of samples in each device is the same, while the latter is highly unbalanced with the number of samples among devices following a power law. To manipulate heterogeneity more precisly, we synthesize unbalanced datasets following the setup in Sahu et al. (2018) and denote it as synthet ic ( $\alpha, \beta$ ) where $\alpha$ controls how much local models differ from each other and $\beta$ controls how much the local data at each device differs from that of other devices. We obtain two datasets: synthetic $(0,0)$ and synthetic $(1,1)$. Details can be found in Appendix D.

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=344&width=1418&top_left_y=1292&top_left_x=337)

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=252&width=347&top_left_y=1305&top_left_x=347)

(a) The impact of $E$

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=258&width=349&top_left_y=1302&top_left_x=693)

(b) The impact of $K$

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=252&width=349&top_left_y=1305&top_left_x=1037)

(c) Different schemes

![](https://cdn.mathpix.com/cropped/2024_06_04_4d7cc05675f9bdd72206g-08.jpg?height=258&width=350&top_left_y=1302&top_left_x=1386)

(d) Different schemes

Figure 1: (a) To obtain an $\epsilon$ accuracy, the required rounds first decrease and then increase when we increase the local steps $E$. (b) In Synthetic $(0,0)$ dataset, decreasing the numbers of active devices each round has little effect on the convergence process. (c) In mnist balanced dataset, Scheme I slightly outperforms Scheme II. They both performs better than the original scheme. Here transformed Scheme II coincides with Scheme II due to the balanced data. (d) In mnist unbal anced dataset, Scheme I performs better than Scheme II and the original scheme. Scheme II suffers from instability while transformed Scheme II has a lower convergence rate.

Experiment settings For all experiments, we initialize all runnings with $\mathbf{w}_{0}=0$. In each round, all selected devices run $E$ steps of SGD in parallel. We decay the learning rate at the end of each round by the following scheme $\eta_{t}=\frac{\eta_{0}}{1+t}$, where $\eta_{0}$ is chosen from the set $\{1,0.1,0.01\}$. We evaluate the averaged model after each global synchronization on the corresponding global objective. For fair comparison, we control all randomness in experiments so that the set of activated devices is the same across all different algorithms on one configuration.

Impact of $E \quad$ We expect that $T_{\epsilon} / E$, the required communication round to achieve curtain accuracy, is a hyperbolic finction of $E$ as equ (8) indicates. Intuitively, a small $E$ means a heavy communication burden, while a large $E$ means a low convergence rate. One needs to trade off between communication efficiency and fast convergence. We empirically observe this phenomenon on unbalanced datasets in Figure 1a. The reason why the phenomenon does not appear in mnist balanced dataset requires future investigations.

Impact of $K$ Our theory suggests that a larger $K$ may slightly accelerate convergence since $T_{\epsilon} / E$ contains a term $\mathcal{O}\left(\frac{E G^{2}}{K}\right)$. Figure $1 \mathrm{~b}$ shows that $K$ has limited influence on the convergence of FedAvg in synthetic $(0,0)$ dataset. It reveals that the curve of a large enough $K$ is slightly better. We observe similar phenomenon among the other three datasets and attach additional results in Appendix D. This justifies that when the variance resulting sampling is not too large (i.e., $B \gg C$ ), one can use a small number of devices without severely harming the training process, which also removes the need to sample as many devices as possible in convex federated optimization.

Effect of sampling and averaging schemes. We compare four schemes among four federated datasets. Since the original scheme involves a history term and may be conservative, we carefully set the initial learning rate for it. Figure 1c indicates that when data are balanced, Schemes I and II achieve nearly the same performance, both better than the original scheme. Figure 1d shows that when the data are unbalanced, i.e., $p_{k}$ 's are uneven, Scheme I performs the best. Scheme II suffers from some instability in this case. This is not contradictory with our theory since we don't guarantee the convergence of Scheme II when data is unbalanced. As expected, transformed Scheme II performs stably at the price of a lower convergence rate. Compared to Scheme I, the original scheme converges at a slower speed even if its learning rate is fine tuned. All the results show the crucial position of appropriate sampling and averaging schemes for FedAvg.

## 7 CONCLUSION

Federated learning becomes increasingly popular in machine learning and optimization communities. In this paper we have studied the convergence of FedAvg, a heuristic algorithm suitable for federated setting. We have investigated the influence of sampling and averaging schemes. We have provided theoretical guarantees for two schemes and empirically demonstrated their performances. Our work sheds light on theoretical understanding of FedAvg and provides insights for algorithm design in realistic applications. Though our analyses are constrained in convex problems, we hope our insights and proof techniques can inspire future work.

## ACKNOWLEDGEMENTS

Li, Yang and Zhang have been supported by the National Natural Science Foundation of China (No. 11771002 and 61572017), Beijing Natural Science Foundation (Z190001), the Key Project of MOST of China (No. 2018AAA0101000), and Beijing Academy of Artificial Intelligence (BAAI).

## REFERENCES

Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly Shmatikov. How to backdoor federated learning. arXiv preprint arXiv:1807.00459, 2018. 7

Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacypreserving machine learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, 2017. 7

Gregory Francis Coppola. Iterative parameter mixing for distributed large-margin training of structured predictors for natural language processing. PhD thesis, 2015. 4, 7

Robin C Geyer, Tassilo Klein, Moin Nabi, and SAP SE. Differentially private federated learning: A client level perspective. arXiv preprint arXiv:1712.07557, 2017. 7

Briland Hitaj, Giuseppe Ateniese, and Fernando Pérez-Cruz. Deep models under the GAN: information leakage from collaborative deep learning. In ACM SIGSAC Conference on Computer and Communications Security, 2017. 7

Mingyi Hong, Meisam Razaviyayn, and Jason Lee. Gradient primal-dual algorithm converges to second-order stationary solution for nonconvex distributed optimization over networks. In International Conference on Machine Learning (ICML), 2018. 8

Dusan Jakovetic. Distributed optimization: algorithms and convergence rates. PhD, Carnegie Mellon University, Pittsburgh PA, USA, 2013. 1

Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. First analysis of local gd on heterogeneous data. arXiv preprint arXiv:1909.04715, 2019. 2, 7

Jakub Konevcnỳ. Stochastic, distributed and federated optimization for machine learning. arXiv preprint arXiv:1707.01155, 2017. 1

Jakub Konevcnỳ, Brendan McMahan, and Daniel Ramage. Federated optimization: distributed optimization beyond the datacenter. arXiv preprint arXiv:1511.03575, 2015. 1

Jakub Konevcnỳ, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016. 7

Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998. 8, 24

Jason D Lee, Qiang Liu, Yuekai Sun, and Jonathan E Taylor. Communication-efficient sparse regression. The Journal of Machine Learning Research, 18(1):115-144, 2017. 8

Mu Li, David G Andersen, Jun Woo Park, Alexander J Smola, Amr Ahmed, Vanja Josifovski, James Long, Eugene J Shekita, and Bor-Yiing Su. Scaling distributed machine learning with the parameter server. In 11th \{USENIX\} Symposium on Operating Systems Design and Implementation (\{OSDI\} 14), pages 583-598, 2014a. 1

Mu Li, David G Andersen, Alexander J Smola, and Kai Yu. Communication efficient distributed machine learning with the parameter server. In Advances in Neural Information Processing Systems (NIPS), 2014b. 1

Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. arXiv preprint arXiv:1908.07873, 2019. 1

Shao-Bo Lin, Xin Guo, and Ding-Xuan Zhou. Distributed learning with regularized least squares. Journal of Machine Learning Research, 18(1):3202-3232, 2017. 8

Tao Lin, Sebastian U Stich, and Martin Jaggi. Don't use large mini-batches, use local sgd. arXiv preprint arXiv:1808.07217, 2018. 7

Dhruv Mahajan, Nikunj Agrawal, S Sathiya Keerthi, Sundararajan Sellamanickam, and Léon Bottou. An efficient distributed learning algorithm based on effective local functional approximations. Journal of Machine Learning Research, 19(1):2942-2978, 2018. 8

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2017. 1, 2, 3, 7, 17, 25

Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov. Exploiting unintended feature leakage in collaborative learning. In IEEE Symposium on Security \& Privacy (S\&P). IEEE, 2019. 7

Xiangrui Meng, Joseph Bradley, Burak Yavuz, Evan Sparks, Shivaram Venkataraman, Davies Liu, Jeremy Freeman, DB Tsai, Manish Amde, and Sean Owen. MLlib: machine learning in Apache Spark. Journal of Machine Learning Research, 17(34):1-7, 2016. 1

Yurii Nesterov. Introductory lectures on convex optimization: a basic course, volume 87. Springer Science \& Business Media, 2013. 7

Sashank J Reddi, Jakub Konecnỳ, Peter Richtárik, Barnabás Póczós, and Alex Smola. AIDE: fast and communication efficient distributed optimization. arXiv preprint arXiv:1608.06879, 2016. 1, 8

Peter Richtárik and Martin Takác. Distributed coordinate descent method for learning with big data. Journal of Machine Learning Research, 17(1):2657-2681, 2016. 1

Anit Kumar Sahu, Tian Li, Maziar Sanjabi, Manzil Zaheer, Ameet Talwalkar, and Virginia Smith. Federated optimization for heterogeneous networks. arXiv preprint arXiv:1812.06127, 2018. 1, 2, $3,5,7,8,17,24,25$

Felix Sattler, Simon Wiedemann, Klaus-Robert Müller, and Wojciech Samek. Robust and communication-efficient federated learning from non-iid data. arXiv preprint arXiv:1903.02891, 2019. 2,7

Ohad Shamir, Nati Srebro, and Tong Zhang. Communication-efficient distributed optimization using an approximate Newton-type method. In International conference on machine learning (ICML), 2014. 1,8

Reza Shokri and Vitaly Shmatikov. Privacy-preserving deep learning. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, 2015. 1

Shusen Wang, Farbod Roosta Khorasani, Peng Xu, and Michael W. Mahoney. GIANT: Globally improved approximate newton method for distributed optimization. In Conference on Neural Information Processing Systems (NeurIPS), 2018. 1, 8

Virginia Smith, Simone Forte, Chenxin Ma, Martin Takac, Michael I Jordan, and Martin Jaggi. CoCoA: A general framework for communication-efficient distributed optimization. arXiv preprint arXiv:1611.02189, 2016. 1, 8

Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar. Federated multi-task learning. In Advances in Neural Information Processing Systems (NIPS), 2017. 2, 7, 8

Sebastian U Stich. Local SGD converges fast and communicates little. arXiv preprint arXiv:1805.09767, 2018. 2, 4, 5, 6, 7, 12

Sebastian U Stich, Jean-Baptiste Cordonnier, and Martin Jaggi. Sparsified SGD with memory. In Advances in Neural Information Processing Systems (NIPS), pages 4447-4458, 2018. 5

Jianyu Wang and Gauri Joshi. Cooperative SGD: A unified framework for the design and analysis of communication-efficient SGD algorithms. arXiv preprint arXiv:1808.07576, 2018. 2, 4, 7

Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin K Leung, Christian Makaya, Ting He, and Kevin Chan. Adaptive federated learning in resource constrained edge computing systems. IEEE Journal on Selected Areas in Communications, 2019. 2, 4, 7

Shusen Wang. A sharper generalization bound for divide-and-conquer ridge regression. In The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI), 2019. 8

Blake E Woodworth, Jialei Wang, Adam Smith, Brendan McMahan, and Nati Srebro. Graph oracle models, lower bounds, and gaps for parallel stochastic optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2018. 2, 4

Cong Xie, Sanmi Koyejo, and Indranil Gupta. Asynchronous federated optimization. arXiv preprint arXiv:1903.03934, 2019. 7

Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted sgd with faster convergence and less communication: Demystifying why model averaging works for deep learning. In AAAI Conference on Artificial Intelligence, 2019. 2, 4, 5, 7

Sixin Zhang, Anna E Choromanska, and Yann LeCun. Deep learning with elastic averaging SGD. In Advances in Neural Information Processing Systems (NIPS), 2015a. 4

Yuchen Zhang and Xiao Lin. DiSCO: distributed optimization for self-concordant empirical loss. In International Conference on Machine Learning (ICML), 2015. 1, 8

Yuchen Zhang, John C. Duchi, and Martin J. Wainwright. Communication-efficient algorithms for statistical optimization. Journal of Machine Learning Research, 14:3321-3363, 2013. 5, 6, 8

Yuchen Zhang, John Duchi, and Martin Wainwright. Divide and conquer kernel ridge regression: a distributed algorithm with minimax optimal rates. Journal of Machine Learning Research, 16: $3299-3340,2015$ b. 8

Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated learning with non-iid data. arXiv preprint arXiv:1806.00582, 2018. 7

Shun Zheng, Fen Xia, Wei Xu, and Tong Zhang. A general distributed dual coordinate optimization framework for regularized loss minimization. arXiv preprint arXiv:1604.03763, 2016. 1

Fan Zhou and Guojing Cong. On the convergence properties of a k-step averaging stochastic gradient descent algorithm for nonconvex optimization. arXiv preprint arXiv:1708.01012, 2017. 2, 4, 7

Hankz Hankui Zhuo, Wenfeng Feng, Qian Xu, Qiang Yang, and Yufeng Lin. Federated reinforcement learning. arXiv preprint arXiv:1901.08277, 2019. 1
</end of paper 2>


