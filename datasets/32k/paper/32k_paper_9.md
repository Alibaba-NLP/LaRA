<paper 0>
# Scaling Advantage in Approximate Optimization with Quantum Annealing 

Humberto Munoz Bauza ${ }^{1,2}$ and Daniel Lidar ${ }^{1,2,3,4}$<br>${ }^{1}$ Department of Physics and Astronomy, University of Southern California, Los Angeles, California 90089, USA<br>${ }^{2}$ Center for Quantum Information Science 8 Technology,<br>University of Southern California, Los Angeles, California 90089, USA<br>${ }^{3}$ Department of Electrical Engineering, University of Southern California, Los Angeles, California 90089, USA<br>${ }^{4}$ Department of Chemistry, University of Southern California, Los Angeles, California 90089, USA


#### Abstract

Quantum annealing is a heuristic optimization algorithm that exploits quantum evolution to approximately find lowest energy states [1,2]. Quantum annealers have scaled up in recent years to tackle increasingly larger and more highly connected discrete optimization and quantum simulation problems [3-7]. Nevertheless, despite numerous attempts, a computational quantum advantage in exact optimization using quantum annealing hardware has so far remained elusive [8-16]. Here, we present evidence for a quantum annealing scaling advantage in approximate optimization. The advantage is relative to the top classical heuristic algorithm: parallel tempering with isoenergetic cluster moves (PT-ICM) [17]. The setting is a family of 2D spin-glass problems with high-precision spin-spin interactions. To achieve this advantage, we implement quantum annealing correction (QAC) [18]: an embedding of a bit-flip error-correcting code with energy penalties that leverages the properties of the D-Wave Advantage quantum annealer to yield over 1, 300 error-suppressed logical qubits on a degree-5 interaction graph. We generate random spin-glass instances on this graph and benchmark their time-to-epsilon, a generalization of the time-to-solution metric [8] for low-energy states. We demonstrate that with QAC, quantum annealing exhibits a scaling advantage over PTICM at sampling low energy states with an optimality gap [19] of at least $1.0 \%$. This amounts to the first demonstration of an algorithmic quantum speedup in approximate optimization.


The pursuit of a quantum speedup using quantum processors is a major theme in modern physics and computer science. Two of the leading application areas are discrete optimization and quantum simulation. In the latter context, impressive recent quantum annealing (QA) advances have been made for fast, coherent anneals lasting on the order of the single superconducting flux qubit coherence time $[20,21]$. While this diabatic approach is considered promIsing [22], it cannot be expected to scale up without the introduction of error correction or suppression, as decoherence and control errors pose insurmountable challenges to Hamiltonian quantum computation models [23-28], just as they do for gate-model quantum computers. In the absence of a fault-tolerance threshold theorem [29] for QA, a variety of Hamiltonian error suppression techniques have been proposed and analyzed as ways to reduce the error rates of this computational model and the closely related model of adiabatic quantum computation [30-36], providing tools towards scalability.

However, despite these theoretical advances, there are currently practical limitations to the types and locality of programmable interactions in the Hamiltonians of quantum annealing hardware, even as new devices have started to emerge [16, 37-39]. To address these limitations, quantum annealing correction (QAC) [18] was developed as a realizable error suppression method for quantum annealing, targeting the available and restricted set of control operations in quantum annealers. QAC has been demonstrated to enhance the success probability of quantum annealing and mitigate the analog control problem $[18,40,41]$. The QAC method is based on a repetition-code encoding of the Hamiltonian and does not fully realize a Hamiltonian stabilizer code. Despite this, it has been shown theoretically to increase the energy gap of the encoded Hamiltonian and reduce tunneling barriers, thus softening the onset of the associated critical dynamics as well as lowering the effective temperature [42].

Here, departing from the traditional paradigm of using QA for exact optimization, we demonstrate-by incorporating QAC-the first genuine scaling advantage in approximate optimization (low-energy sampling) using a quantum annealer. Even approximate optimization can be computationally hard unless $\mathrm{P}=\mathrm{NP}[43,44]$, so we do not expect the advantage we exhibit to amount to more than a polynomial speedup. However, whereas the scaling advantages reported in previous work were relative to simulated annealing $[13,16]$, the advantage we find here is over the best currently available general heuristic classical optimization method: parallel tempering with isoenergetic cluster moves (PT-ICM) [17]. This result is enabled by implementing QAC on the D-Wave Advantage quantum annealer for the Sidon-set spin glass instance class [9], embedded on the logical graph formed after the encoding step. The advantage of quantum annealing over PT-ICM is diminished without QAC, thus highlighting the crucial role of quantum error suppression.

## Quantum annealing

The D-Wave quantum processing unit (QPU) uses superconducting flux qubits to implement the transverse field
(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_e9e9bc1d27b2833366ebg-02.jpg?height=517&width=545&top_left_y=262&top_left_x=259)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_e9e9bc1d27b2833366ebg-02.jpg?height=529&width=531&top_left_y=248&top_left_x=865)

(c)

![](https://cdn.mathpix.com/cropped/2024_06_04_e9e9bc1d27b2833366ebg-02.jpg?height=501&width=507&top_left_y=270&top_left_x=1427)

FIG. 1. Embedding of QAC on the Pegasus hardware graph. (a) Close-up of the embedding of three clusters of two coupled logical qubits each. Physical qubits are denoted by circles and couplings by lines. Penalty qubits are marked by a $\mathrm{P}$ and the three penalty couplings (thin lines) to their corresponding physical data qubits. Thick lines indicate the physical couplings between the corresponding physical qubits of different logical qubits. Only a subset of all possible couplings are shown. (b) A zoomed-out view from (a) showing the logical qubits (circles) and the logical couplings induced by the QAC embedding. Thick lines indicate the logical couplings shown in (a), whose type is colored by direction: horizontal/vertical/diagonal (long or short). (c) A zoomed-out view from (b) showing a $4 \times 4$ induced logical graph of QAC. The logical graph is equivalent to a honeycomb graph with additional non-planar bonds. The induced logical graph of the D-Wave Advantage 4.1 QPU has a maximum of 1322 logical qubits; the largest available side length is $L=15$.

Ising Hamiltonian

$$
\begin{equation*}
H(s)=-A(s) \sum_{i \in \mathcal{V}} \sigma_{i}^{x}+B(s) H_{z} \tag{1}
\end{equation*}
$$

where $\mathcal{V}$ is the vertex set of the hardware graph of the $\mathrm{QPU}, i$ is the qubit index, $\sigma_{i}^{x}$ are Pauli matrices, and $A(s)$ and $B(s)$ are the annealing schedules, respectively decreasing to 0 and increasing from 0 with $s: 0 \rightarrow 1$. $H_{z}$ is the Ising problem Hamiltonian:

$$
\begin{equation*}
H_{z}=\sum_{i \in \mathcal{V}} h_{i} \sigma_{i}^{z}+\sum_{\{i, j\} \in \mathcal{E}} J_{i j} \sigma_{i}^{z} \sigma_{j}^{z} \tag{2}
\end{equation*}
$$

where $h_{i}$ and $J_{i j}$ are programmable local fields and couplings, respectively, and $\mathcal{E}$ is the edge set of the hardware graph. Many NP-complete and NP-hard problems can be mapped to $H_{z}$ [45] by minor-embedding onto the hardware graph. We performed QA experiments on the D-Wave Advantage 4.1 QPU accessed through the D-Wave Leap cloud interface, featuring the Pegasus hardware graph [46].

## Quantum annealing correction

We implement the $[[3,1,3]]_{1}$ QAC encoding introduced in Ref. [18], which encodes a logical qubit into three physical "data qubits," each of which is coupled to the same additional "energy penalty qubit" with a fixed coupling strength $J_{p}$; the logical qubit is decoded via a majority vote on the data qubits. The logical subgraph induced by the QAC encoding on the Pegasus graph has a bulk degree of 5 and admits native loops of length 5. Fig. 1 illustrates the encoding and the induced logical graph. All previous QAC results were obtained using the "Chimera" hardware graph of the previous generation of D-Wave QPUs, which has degree 3 and no odd-length loops. The features of the induced Pegasus logical graph permit the benchmarking of significantly harder spin-glass instances than was possible on Chimera. The induced logical graphs we examine have side length $L \in[5,15]$, corresponding to a problem size range of $N \in[142,1322]$ logical, error-corrected qubits.

Problems on the logical QAC graph can also be encoded directly by setting $J_{p}=0$, resulting in three uncoupled and unprotected, parallel classical copies of the problem instance. We then extract the energies of all three copies as independent annealing samples and denote this "unprotected" QA method by U3. We use the U3 method below to test whether QAC has a genuine advantage over simple classical repetition coding.

## Spin-glass instances

We generate random native spin-glass instances on the induced logical graph. These types of instances have been widely used to benchmark the previous D-Wave QPUs (with the Chimera hardware graph) against classical algorithms $[4,8,9]$. We tested three types of spin-glass disorder: binomial, where $J_{i j}$ was randomly selected as $\pm 1$ with equal probability, Sidon- 28 (S28) [9], where $J_{i j}$ was randomly sampled from the set $\pm\{8 / 28,13 / 28,19 / 28,1\}$, and finally range 6 (R6)
disorder, where $J_{i j} \in \pm\{1 / 6, \ldots, 6 / 6\}$. In a Sidon set, the sum of any two set members gives a number that is not part of the set. Moreover, no five numbers from the S28 set sum to zero, which prevents the occurrence of "floppy" qubits [47] given the bulk degree- 5 of the Pegasus graph. Binomial disorder generally admits a degenerate ground state, simplifying the optimization problem. In contrast, the S28 disorder can yield instances with a unique ground state [26]. The ground states are robust to small errors in the implementation of the $J_{i j}$ values in the binomial disorder case, but this is not the case when high precision in implementing the $J_{i j}$ values is required (as for Sidon disorder). The latter case is expected to benefit more from QAC than the former [41]. From here on, we focus on the S28 case; see Methods.

## Time-to-epsilon metric

The standard performance metric for exact optimization using heuristic solvers is the time-to-solution (TTS), which quantifies the time to reach the ground state at least once with $99 \%$ probability, given that the success probability per run is $p$ : TTS $=t_{f} R$, where each annealing run lasts time $t_{f}$ and $R=\frac{\log (1-0.99)}{\log (1-p)}$ is the expected number of runs [8]. To address approximate optimization, we instead consider the time required to reach an energy within a fraction $\varepsilon$ of the ground state energy $E_{0}$, and define the time-to-epsilon for an instance as

$$
\begin{equation*}
\mathrm{TT} \varepsilon=t_{f} \frac{\log (1-0.99)}{\log \left(1-p_{E \leq E_{0}+\varepsilon\left|E_{0}\right|}\right)} \tag{3}
\end{equation*}
$$

where $p_{E \leq E_{0}+\varepsilon\left|E_{0}\right|}$ is the probability that the energy $E$ of a sample is no more than $\varepsilon\left|E_{0}\right|$ above $E_{0}$. The TTS is the special case $\varepsilon=0$. In a mixed integer programming optimization context, $\varepsilon$ is known as the optimality gap [19], which is how we refer to it here. The ground state energies are known for our instances (see Methods ), so $\varepsilon$ is exactly calculated for each sample rather than bounded. An alternative time metric is the residual energy density from the ground state [21]; we focus on the optimality gap due to its relevance to benchmarking approximate optimization algorithms.

We define $[\mathrm{TT} \varepsilon]_{q}$ of an instance class as the $q$-th quantile of $\mathrm{TT} \varepsilon$ over the entire instance class. Here, we focus only on the median quantile, $q=0.5$, denoted $[\mathrm{TT} \varepsilon]_{\mathrm{Med}}$. For a given disorder, instance size, and $\varepsilon$-target, we find the annealing time $t_{f}$ (and penalty strength for QAC) that minimizes $[\mathrm{TT} \varepsilon]_{\text {Med. }}$. We restrict the penalty coupling strengths to the set $J_{p} \in\{0.1,0.2,0.3\}$ to reduce resource requirements for parameter optimization, as $J_{p}=0.2$ is the penalty strength that most frequently optimizes the success probabilities of individual instances, and the dependence on $J_{p}$ above 0.2 becomes weak.

## Fitting the TT $\varepsilon$

Below, we fit the $\mathrm{TT} \varepsilon$ to a power law: $\mathrm{TT} \varepsilon(N)=c N^{\alpha}$, where $\alpha$ is the scaling exponent, the quantity we use to quantify the scaling of the different algorithms we compare. The choice of a power law is motivated by the existence of an $O(N)$ classical algorithm for the residual energy density; we describe such an algorithm in Methods (though this algorithm is utterly impractical due to its huge prefactor). Due to the power law fit, we should account for factors that can modify the scaling exponent. Indeed, we could use all $N_{\max }$ qubits of the QPU and embed $N_{\max } / N$ parallel copies of each problem of size $N$, then select the best of these copies. Since, in reality, we work with only one copy due to a small fraction of the qubits and couplers being absent, we multiply the $\mathrm{TT} \varepsilon$ by a factor of $N / N_{\max }$ [8]. The U3 TT $\varepsilon$ is similarly multiplied by a constant factor of $3 / 4$ since, due to a lack of needed couplings, each instance is repeated only over the data qubits, thus leaving $1 / 4$ of the available (penalty) qubits unused.

## Parallel tempering algorithm

Our baseline classical algorithm is parallel tempering with isoenergetic cluster moves (PT-ICM) [17]. The runtime of this algorithm has the best scaling with problem size known in the task of finding the ground state of various benchmark problems on D-Wave QPUs [12, 14], with the only known exception being certain XORSAT instances for which highly specialized solvers have been developed [15, 48]. Our optimization of the algorithmic parameters of PT-ICM is described in Methods.

## Results

It is well known that the TTS metric generates unreliable results unless the annealing time is optimized for each size $N[8,13]$. This is because an artificially high TTS at small $N$ results in an overly flat TTS scaling. The same considerations apply to the $\mathrm{TT} \varepsilon$, so here we find the annealing time $t_{f}$ that minimizes $[\mathrm{TT} \varepsilon]_{\text {Med }}$ for each $N$-denoted $t_{f}^{\text {opt }}(N)$-and report the resulting median $\mathrm{TT} \varepsilon$ and its scaling estimate for QAC and U3 in Fig. 2, along with the analogously optimized PT-ICM results.

The shortest available annealing time on the D-Wave Advantage QPU accessed via Leap is $t_{f}^{\min }=0.5 \mu \mathrm{s}$, and the bottom panels of Fig. 2 show that as the target residual energy density is increased, progressively larger problem sizes are needed to ensure that $t_{f}^{\mathrm{opt}}(N)>t_{f}^{\min }$. We cannot rule out that with access to lower annealing times, one would find $t_{f}^{\mathrm{opt}}(N)<t_{f}^{\min }$ for all $N$ values where we empirically find $t_{f}^{\text {opt }}(N)=t_{f}^{\mathrm{min}}$. We thus formulate a null hypothesis for each $N$ that $t_{f}^{\mathrm{opt}}(N) \leq t_{f}^{\min }$ and compute a $P$ value as the empirical number of bootstrap samples whose $t_{f}^{\mathrm{opt}}(N)=t_{f}^{\mathrm{min}}$, out of a total of 200 samples (see Methods for details of our statistical analysis). To compute the $\mathrm{TT} \varepsilon$ scaling, i.e., the slope $\alpha$ in a fit to $\mathrm{TT} \varepsilon=c N^{\alpha}$, for each $\varepsilon$ we use only those $t_{f}^{\mathrm{opt}}(N)$ values whose $P<0.05$ (filled circles in Fig. 2). We can thus be
![](https://cdn.mathpix.com/cropped/2024_06_04_e9e9bc1d27b2833366ebg-04.jpg?height=592&width=1768&top_left_y=172&top_left_x=186)

FIG. 2. Time-to-epsilon scaling for QAC, U3, and PT-ICM for Sidon-28 (S28) spin-glass disorder. The bottom panels show the optimal annealing times of U3 and QAC. The top panels show the TT $\varepsilon$ results for the corresponding optimal annealing times, along with optimized PT-ICM results. The straight lines are best fits assuming power law scaling TT $\varepsilon=c N^{\alpha}$ and the accompanying numbers are the corresponding slopes $\alpha$. As indicated in the legends, the target $\varepsilon$ increases from left to right, with a corresponding improvement in quantum annealing's performance: PT-ICM is better at low $\varepsilon$, but for $\varepsilon=1 \%$, while U3 is still worse than PT-ICM, QAC already outperforms PT-ICM. At $\varepsilon=1.25 \%$, both QAC and U3 have better scaling than PT-ICM. For higher residual energy targets, the scaling of QAC becomes unreliable since we can no longer guarantee that the optimal $t_{f}$ has been identified. We used $t_{f} \in[0.5,27] \mu \mathrm{s}$ and $N \in[142,1322]$. Error bars for $\mathrm{TT} \varepsilon$ data points are twice the standard error of the parameter estimate calculated using bootstrapping. Filled (open) circles correspond to a $P$ value of 0.05 (0.20) that $t_{f}^{\text {opt }}(N)>t_{f}^{\min }$ for the corresponding problem size $N$ (see Methods for details). We use only those $t_{f}$ values for which $P \leq 0.05$ to compute the slopes.

confident that the reported slopes reflect the true scaling of U3 and QAC.

Our first observation is that the $Q A C$ scaling is always better than the U3 scaling, which is consistent with previous studies concerning the effect of analog coupling errors (" $J$-chaos") on the TTS for spin-glass instances [41]. Such errors are expected for the S28 instances due to the relatively high precision their specification requires.

Second, we observe that U3 and QAC reduce the absolute algorithmic runtime by four orders of magnitude compared to PT-ICM. However, this is not a scaling advantage, and since our PT-ICM calculations could be sped up by employing faster classical processors, we do not consider this a robust finding. Similarly, we exclude the programming and readout time used by the D-Wave QPU. The primary source of overhead is the readout time per sample, which scales with problem size and can reach $200 \mu$ s per sample for this QPU. This timing varies by hardware generation, and its inclusion obfuscates the dominant scaling source.

Third, and most significantly, we observe that as the target optimality gap increases $Q A$ 's scaling overtakes PT-ICM. Notably, at $\varepsilon=1 \%$, QAC exhibits a scaling exponent of $1.69 \pm 0.12$, compared to PT-ICM's $1.93 \pm 0.03$, and at $\varepsilon=1.25 \%, \mathrm{QAC}$ and $\mathrm{U} 3$ exhibit scaling exponents of $1.15 \pm 0.22$ and $1.76 \pm 0.06$, respectively, compared to PT-ICM's $1.87 \pm 0.02$. The scaling of U3 at optimal annealing times continues to decrease as $\varepsilon$ increases; at $\varepsilon=1.5 \%$ (not shown), the scaling exponents for U3 and
PT-ICM become $1.60 \pm 0.07$ and $1.86 \pm 0.04$, respectively. This is robust evidence of a quantum annealing scaling advantage over the best available classical heuristic optimization algorithm.

We are unable to determine the scaling of QAC for $\varepsilon>$ $1.25 \%$, as we cannot confirm that $t_{f}^{\mathrm{opt}}(N)>t_{f}^{\min }$ for any $N$; as can be seen in Fig. 2, already for $\varepsilon=1.25 \%$ only the largest three $N$ values satisfy the $P<0.05$ criterion. However, given the consistently better scaling of QAC for lower values of $\varepsilon$, where $t_{f}^{\mathrm{opt}}(N)>t_{f}^{\min }$ for QAC over a range of problem sizes, it is reasonable to conclude that the QAC scaling would be a further improvement over $\mathrm{U} 3$ if its true $t_{f}^{\mathrm{opt}}(N)$ could be established for $\varepsilon>1.25 \%$; this would require access to shorter annealing times or larger system sizes.

We note that it is unsurprIsing that, given a sufficiently large target optimality gap, the D-Wave QPU returns sample energies within that gap. Similarly, we can expect that for large enough $\varepsilon$, even simulated annealing or greedy descent will be nearly guaranteed success in polynomial time. The significance of our result is that $Q A$ reaches near-linear scaling at a smaller optimality gap target than PT-ICM. Thus, we refer to this result as an approximate optimization advantage for quantum annealing. We also note that Ref. [21] similarly reported a QA optimization advantage for the residual energy density (for 3D spin glasses), but this was done at a fixed problem size (of $N=5374$ physical qubits), and was instead concerned with the convergence of the
residual energy density with the annealing time. We reemphasize that, in contrast, we are reporting a scaling advantage as a function of problem size, the proper context for quantum speedup claims. We explain in Methods that our results stand regardless of whether an optimality gap or residual energy density target is chosen for benchmarking.

## Dynamical critical scaling

As an additional perspective on the different quantum annealing dynamics resulting from error suppression, we study the difference in the dynamical critical scaling under the Kibble-Zurek (KZ) scaling ansatz [49, 50], where an annealing time of $t_{f} \sim L^{\mu}$ is required to suppress diabatic excitations with a correlation length of $L$, and where $\mu$ is the dynamical critical exponent or $\mathrm{KZ}$ exponent. We examined signatures of dynamical critical scaling by calculating the Binder cumulant $U=\frac{1}{2}\left(3-\left\langle q^{4}\right\rangle /\left\langle q^{2}\right\rangle^{2}\right)$, where $\langle\cdot\rangle$ denotes the sample average, either from U3 or after decoding with QAC, and $U$ is averaged over all instances for each system size $N$ and annealing time $t_{f}$. Here $q=\frac{1}{N} \sum_{i=1}^{N} \sigma_{i}^{z} \sigma_{i}^{z \prime}$ is the overlap between two replicas, i.e., independently annealed $N$-spin states $\left\{\sigma_{i}^{z}\right\}$ and $\left\{\sigma_{i}^{z \prime}\right\}$ of a given disorder realization (set of couplings $J_{i j}$ ).

We performed a finite-size scaling analysis and data collapse, and the collapsed Binder cumulant for the S28 instances is shown in Fig. 3. We find that $\mu_{\mathrm{QAC}}=4.81 \pm$ 0.22 (at a penalty strength of 0.1 ) compared to $\mu_{\mathrm{U} 3}=$ $7.53 \pm 0.47$. The reduction of $\mu$ is lost when $\lambda=0.2$ (not shown), suggesting that $\lambda=0.1$ is optimal in the sense of diabatic error suppression.

This effect indicates that QAC is much more effective at suppressing diabatic excitations. I.e., at equal annealing times, the dynamics are more adiabatic under QAC, in agreement with theory [42]. The significant reduction in $\mu$ suggests that in addition to $J$-chaos suppression, diabatic error suppression by QAC is responsible for the improved $\mathrm{TT} \varepsilon$ and shorter optimal annealing times. Additional context is provided in Methods.

## Discussion

Using the largest available quantum annealer to date we have demonstrated an approximate optimization timescaling advantage for quantum annealing on a family of spin-glass problems with low ground-state degeneracy and high-precision couplings. Our demonstration involves up to 1322 logical qubits, the largest number to date in an error-corrected setting. Our key result is the demonstration of an algorithmic quantum scaling advantage. The advantage is relative to PT-ICM, the best classical heuristic algorithm currently known for such spin glass problems, and appears at optimality gaps $\gtrsim 1 \%$.

There are a few limitations to the scope of our conclusion. First, our results do not imply that finding states

![](https://cdn.mathpix.com/cropped/2024_06_04_e9e9bc1d27b2833366ebg-05.jpg?height=794&width=805&top_left_y=178&top_left_x=1126)

FIG. 3. Collapsed Binder cumulant for samples collected for U3 and QAC with a penalty strength of $\lambda=0.1$. Here $t_{f} \in[0.5,5.2] \mu \mathrm{s}$ and $L \in[5,15]$.

within small, constant gaps (and indeed finding the ground state itself) is easy for quantum annealing, nor do they imply that all spin glass problems are amenable to an approximate optimization scaling advantage via quantum annealing. Second, being finite-range and two-dimensional limits the range of applications of the problem family we have studied here. To achieve an algorithmic quantum advantage in an application setting, the next challenge for quantum optimization is demonstrating a hardware-scalable advantage in densely connected problems at sufficiently small optimality gaps.

## Acknowledgements

We thank Dr. Evgeny Mozgunov for suggesting the use of the time-to-epsilon metric and Dr. Victor Kasatkin for discussions about the theoretical lower bound of finiterange approximate optimization. We also thank Dr. Mohammad Amin, Dr. Carleton Coffrin, Dr. Itay Hen, and Dr. Tameem Albash for various helpful discussions and suggestions. This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00112190071. This research was also supported by the ARO MURI grant W911NF-22-S-0007. The authors acknowledge the Center for Advanced Research Computing (CARC) at the University of Southern California for computing resources.

[1] T. Kadowaki and H. Nishimori, Quantum annealing in the transverse Ising model, Physical Review E 58, 5355 (1998).

[2] P. Hauke, H. G. Katzgraber, W. Lechner, H. Nishimori, and W. D. Oliver, Perspectives of quantum annealing: Methods and implementations, Reports on Progress in Physics (2020).

[3] M. W. Johnson, M. H. S. Amin, S. Gildert, T. Lanting, F. Hamze, N. Dickson, R. Harris, A. J. Berkley, J. Johansson, P. Bunyk, E. M. Chapple, C. Enderud, J. P. Hilton, K. Karimi, E. Ladizinsky, N. Ladizinsky, T. Oh, I. Perminov, C. Rich, M. C. Thom, E. Tolkacheva, C. J. S. Truncik, S. Uchaikin, J. Wang, B. Wilson, and G. Rose, Quantum annealing with manufactured spins, Nature 473, 194 (2011).

[4] S. Boixo, T. F. Rønnow, S. V. Isakov, Z. Wang, D. Wecker, D. A. Lidar, J. M. Martinis, and M. Troyer, Evidence for quantum annealing with more than one hundred qubits, Nature Physics 10, 218 (2014).

[5] A. D. King, J. Carrasquilla, J. Raymond, I. Ozfidan, E. Andriyash, A. Berkley, M. Reis, T. Lanting, R. Harris, F. Altomare, K. Boothby, P. I. Bunyk, C. Enderud, A. Fréchette, E. Hoskinson, N. Ladizinsky, T. Oh, G. Poulin-Lamarre, C. Rich, Y. Sato, A. Y. Smirnov, L. J. Swenson, M. H. Volkmann, J. Whittaker, J. Yao, E. Ladizinsky, M. W. Johnson, J. Hilton, and M. H. Amin, Observation of topological phenomena in a programmable lattice of 1,800 qubits, Nature 560, 456 (2018).

[6] R. Harris, Y. Sato, A. J. Berkley, M. Reis, F. Altomare, M. H. Amin, K. Boothby, P. Bunyk, C. Deng, C. Enderud, S. Huang, E. Hoskinson, M. W. Johnson, E. Ladizinsky, N. Ladizinsky, T. Lanting, R. Li, T. Medina, R. Molavi, R. Neufeld, T. Oh, I. Pavlov, I. Perminov, G. PoulinLamarre, C. Rich, A. Smirnov, L. Swenson, N. Tsai, M. Volkmann, J. Whittaker, and J. Yao, Phase transitions in a programmable quantum spin glass simulator, Science 361,162 (2018)

[7] P. Weinberg, M. Tylutki, J. M. Rönkkö, J. Westerholm, J. A. Åström, P. Manninen, P. Törmä, and A. W. Sandvik, Scaling and diabatic effects in quantum annealing with a d-wave device, Physical Review Letters 124, 090502 (2020).

[8] T. F. Ronnow, Z. Wang, J. Job, S. Boixo, S. V. Isakov, D. Wecker, J. M. Martinis, D. A. Lidar, and M. Troyer, Defining and detecting quantum speedup, Science 345, 420 (2014).

[9] H. G. Katzgraber, F. Hamze, Z. Zhu, A. J. Ochoa, and H. Munoz-Bauza, Seeking Quantum Speedup Through Spin Glasses: The Good, the Bad, and the Ugly, Physical Review X 5, 031026 (2015).

[10] D. Venturelli, S. Mandrà, S. Knysh, B. O'Gorman, R. Biswas, and V. Smelyanskiy, Quantum optimization of fully connected spin glasses, Phys. Rev. X 5, 031040 (2015).

[11] V. S. Denchev, S. Boixo, S. V. Isakov, N. Ding, R. Babbush, V. Smelyanskiy, J. Martinis, and H. Neven, What is the computational value of finite-range tunneling?, Phys. Rev. X 6, 031015 (2016).

[12] S. Mandrà, H. G. Katzgraber, and C. Thomas, The pitfalls of planar spin-glass benchmarks: RaIsing the bar for quantum annealers (again), Quantum Science and Tech- nology 2, 038501 (2017).

[13] T. Albash and D. A. Lidar, Demonstration of a Scaling Advantage for a Quantum Annealer over Simulated Annealing, Physical Review X 8, 031016 (2018).

[14] S. Mandrà and H. G. Katzgraber, A deceptive step towards quantum speedup detection, Quantum Science and Technology 3, 04LT01 (2018).

[15] M. Kowalsky, T. Albash, I. Hen, and D. A. Lidar, 3regular three-xorsat planted solutions benchmark of classical and quantum heuristic optimizers, Quantum Science and Technology 7, 025008 (2022).

[16] S. Ebadi, A. Keesling, M. Cain, T. T. Wang, H. Levine, D. Bluvstein, G. Semeghini, A. Omran, J. Liu, R. Samajdar, X.-Z. Luo, B. Nash, X. Gao, B. Barak, E. Farhi, S. Sachdev, N. Gemelke, L. Zhou, S. Choi, H. Pichler, S. Wang, M. Greiner, V. Vuletic, and M. D. Lukin, Quantum Optimization of Maximum Independent Set using Rydberg Atom Arrays, Science 376, 1209 (2022).

[17] Z. Zhu, A. J. Ochoa, and H. G. Katzgraber, Efficient Cluster Algorithm for Spin Glasses in Any Space Dimension, Physical Review Letters 115, 077201 (2015).

[18] K. L. Pudenz, T. Albash, and D. A. Lidar, Errorcorrected quantum annealing with hundreds of qubits, $\mathrm{Na}-$ ture Communications 5, 3243 (2014).

[19] Gurobi Optimization: MIPGap.

[20] A. D. King, S. Suzuki, J. Raymond, A. Zucca, T. Lanting, F. Altomare, A. J. Berkley, S. Ejtemaee, E. Hoskinson, S. Huang, E. Ladizinsky, A. J. R. MacDonald, G. Marsden, T. Oh, G. Poulin-Lamarre, M. Reis, C. Rich, Y. Sato, J. D. Whittaker, J. Yao, R. Harris, D. A. Lidar, H. Nishimori, and M. H. Amin, Coherent quantum annealing in a programmable 2,000 qubit Ising chain, Nature Physics 18,1324 (2022).

[21] A. D. King, J. Raymond, T. Lanting, R. Harris, A. Zucca, F. Altomare, A. J. Berkley, K. Boothby, S. Ejtemaee, C. Enderud, E. Hoskinson, S. Huang, E. Ladizinsky, A. J. R. MacDonald, G. Marsden, R. Molavi, T. Oh, G. Poulin-Lamarre, M. Reis, C. Rich, Y. Sato, N. Tsai, M. Volkmann, J. D. Whittaker, J. Yao, A. W. Sandvik, and M. H. Amin, Quantum critical dynamics in a 5,000qubit programmable spin glass, Nature 617, 61 (2023).

[22] E. J. Crosson and D. A. Lidar, Prospects for quantum enhancement with diabatic quantum annealing, Nature Reviews Physics 3, 466 (2021).

[23] A. M. Childs, E. Farhi, and J. Preskill, Robustness of adiabatic quantum computation, Phys. Rev. A 65, 012322 (2001).

[24] M. H. S. Amin, D. V. Averin, and J. A. Nesteroff, Decoherence in adiabatic quantum computation, Phys. Rev. A 79,022107 (2009).

[25] T. Albash and D. A. Lidar, Decoherence in adiabatic quantum computation, Physical Review A 91, 062320 (2015).

[26] Z. Zhu, A. J. Ochoa, S. Schnabel, F. Hamze, and H. G. Katzgraber, Best-case performance of quantum annealers on native spin-glass benchmarks: How chaos can affect success probabilities, Physical Review A 93, 012317 (2016).

[27] T. Albash, V. Martin-Mayor, and I. Hen, Temperature scaling law for quantum annealing optimizers, Physical Review Letters 119, 110502 (2017).

[28] T. Albash, V. Martin-Mayor, and I. Hen, Analog errors in Ising machines, Quantum Sci. Technol. 4, 02LT03 (2019).

[29] E. T. Campbell, B. M. Terhal, and C. Vuillot, Roads to-
wards fault-tolerant universal quantum computation, Nature 549, 172 EP (2017).

[30] S. P. Jordan, E. Farhi, and P. W. Shor, Error-correcting codes for adiabatic quantum computation, Physical Review A 74, 052322 (2006).

[31] D. A. Lidar, Towards fault tolerant adiabatic quantum computation, Phys. Rev. Lett. 100, 160506 (2008).

[32] K. C. Young, M. Sarovar, and R. Blume-Kohout, Error suppression and error correction in adiabatic quantum computation: Techniques and challenges, Phys. Rev. X 3, 041013 (2013).

[33] A. Ganti, U. Onunkwo, and K. Young, Family of [[6k,2k,2]] codes for practical, scalable adiabatic quantum computation, Phys. Rev. A 89, 042313 (2014).

[34] A. D. Bookatz, E. Farhi, and L. Zhou, Error suppression in Hamiltonian-based quantum computation using energy penalties, Physical Review A 92, 022317 (2015).

[35] Z. Jiang and E. G. Rieffel, Non-commuting two-local hamiltonians for quantum error suppression, Quant. Inf. Proc. 16, 89 (2017).

[36] M. Marvian and D. A. Lidar, Error Suppression for Hamiltonian-Based Quantum Computation Using Subsystem Codes, Physical Review Letters 118, 030504 (2017).

[37] A. W. Glaetzle, R. M. W. van Bijnen, P. Zoller, and W. Lechner, A coherent quantum annealer with Rydberg atoms, Nature Communications 8, 15813 EP (2017).

[38] R. Hamerly, T. Inagaki, P. L. McMahon, D. Venturelli, A. Marandi, T. Onodera, E. Ng, C. Langrock, K. Inaba, T. Honjo, K. Enbutsu, T. Umeki, R. Kasahara, S. Utsunomiya, S. Kako, K.-i. Kawarabayashi, R. L. Byer, M. M. Fejer, H. Mabuchi, D. Englund, E. Rieffel, H. Takesue, and Y. Yamamoto, Experimental investigation of performance differences between coherent Ising machines and a quantum annealer, Science Advances 5, eaau0823 (2019).

[39] D. M. Tennant, X. Dai, A. J. Martinez, R. Trappen, D. Melanson, M. A. Yurtalan, Y. Tang, S. Bedkihal, R. Yang, S. Novikov, J. A. Grover, S. M. Disseler, J. I. Basham, R. Das, D. K. Kim, A. J. Melville, B. M. Niedzielski, S. J. Weber, J. L. Yoder, A. J. Kerman, E. Mozgunov, D. A. Lidar, and A. Lupascu, Demonstration of long-range correlations via susceptibility measurements in a one-dimensional superconducting Josephson spin chain, npj Quantum Information 8, 85 (2022).

[40] W. Vinci, T. Albash, and D. A. Lidar, Nested quantum annealing correction, npj Quantum Information 2, 16017 (2016).

[41] A. Pearson, A. Mishra, I. Hen, and D. A. Lidar, Analog errors in quantum annealing: Doom and hope, npj Quantum Information 5, 1 (2019).

[42] S. Matsuura, H. Nishimori, W. Vinci, T. Albash, and D. A. Lidar, Quantum-annealing correction at finite temperature: Ferromagnetic $p$-spin models, Physical Review A 95, 022308 (2017).

[43] L. Trevisan, Inapproximability of Combinatorial Optimization Problems, arXiv e-prints (2004), arxiv:cs/0409043.

[44] S. Arora and B. Barak, Computational Complexity: A Modern Approach (Cambridge University Press, 2009).

[45] A. Lucas, Ising formulations of many NP problems, Front. Phys. 2, 5 (2014).

[46] K. Boothby, P. Bunyk, J. Raymond, and A. Roy, Nextgeneration topology of d-wave quantum processors (2020), arXiv:2003.00133 [quant-ph].

[47] S. Boixo, T. Albash, F. M. Spedalieri, N. Chancellor, and D. A. Lidar, Experimental signature of programmable quantum annealing, Nat. Commun. 4, 2067 (2013).

[48] M. Bernaschi, M. Bisson, M. Fatica, E. Marinari, V. Martin-Mayor, G. Parisi, and F. Ricci-Tersenghi, How we are leading a 3-XORSAT challenge: From the energy landscape to the algorithm and its efficient implementation on GPUs, Europhysics Letters 133, 60005 (2021).

[49] W. H. Zurek, U. Dorner, and P. Zoller, Dynamics of a quantum phase transition, Physical Review Letters 95, 105701 (2005).

[50] A. del Campo, Universal statistics of topological defects formed in a quantum phase transition, Physical Review Letters 121, 200601 (2018).

## Methods

## Ground state energies

We solved all S28 instances to optimality using Gurobi 10 within feasible runtime, except for 7 instances of size $L=15$. For these remaining instances, Gurobi proved an optimality gap of at most $1.2 \%$, meaning the lowest energy found is guaranteed to be no greater than $1.2 \%$ above the true ground state. Furthermore, the lowest energies found by PT-ICM were no higher than those found by Gurobi for all instances. Thus, we assigned the ground state energies of the remaining instances to the values found by PT-ICM with high confidence. As we use a median over instances as our summary statistic, our conclusions are unaffected by the possibility that the ground state energy was never reached for such few instances.

## PT-ICM parameter optimization

We first ran every replica once for $N_{\mathrm{sw}}^{\max }=500,000$ sweeps to ensure the ground state energy was reached. For the largest size $L=15$, we determined $N_{\mathrm{sw}}^{(90 \%)}$, the number of sweeps that were required for $90 \%$ of the instances to reach their lowest recorded energy, where we found $N_{\mathrm{sw}}^{(90 \%)} \approx 31,000$ for the S28 instances. As this was significantly less than $N_{\mathrm{sw}}^{\max }$, we considered the ground states for these instances as "validated" by PT-ICM to calculate median quantities over the instances. We finally ran PT-ICM 100 times for each instance, setting $N_{\mathrm{sw}}=N_{\mathrm{sw}}^{(90 \%)}$. This yields an empirical cumulative density function for $p_{E \leq E_{0}+\varepsilon}$ as a function of the runtime of PT-ICM. The TT $\varepsilon$ is then evaluated for each instance by optimizing over the runtime of the PT-ICM repetition (where $t_{f}$ is now the time needed to reach the target rather than the annealing time).

The scaling of PT-ICM is ideally evaluated using the parameters that best optimize the $\mathrm{TT} \varepsilon$ for each disorder realization, instance size, and target $\varepsilon$. However, a rigorous optimization of the number and choice of replica temperatures for all target $\varepsilon$ 's and system sizes is computationally infeasible. To ensure our results hold for any choice of reasonably optimized temperatures, we repeated the $\mathrm{TT} \varepsilon$ evaluation with four temperature sets summarized in Table I, which includes both logarithmically-spaced and feedback-optimized [1] temperatures. The TT $\varepsilon$ for a given disorder, instance size, and energy target was chosen from the best $\mathrm{TT} \varepsilon$ out of the four temperature sets, illustrated in Fig. 4. At the optimality gap targets of interest, most temperature sets' differences are not appreciable and are unlikely to affect our conclusions. A more comprehensive but computationally expensive optimization of PT-ICM would involve a grid search over a range of the parameters $\beta_{\min }, \beta_{\max }$, and $N_{\mathrm{icm}}$.

|  |  | $\mathbf{S 2 8}$ |  |  |
| :---: | :---: | :---: | :---: | :---: |
| Set | $N_{T}$ | $\beta_{\min }$ | $\beta_{\max }$ | $N_{\mathrm{icm}}$ |
| 1 | 32 | 0.1 | 5.0 | 8 |
| 2 | 24 | 0.2 | 10.0 | 6 |
| 3 | 32 | 0.2 | 10.0 | 8 |
| 4 | 32 | 0.1 | 20.0 | 8 |

TABLE I. Temperature sets used for PT-ICM for S28 instances, where $\beta_{\min }$ is the hottest temperature, $\beta_{\max }$ is the coldest temperature, and $N_{\mathrm{icm}}$ is the number of low temperature (largest $\beta$ ) subject to ICM moves. The temperatures are logarithmically spaced in sets 1 and 4. The temperatures in sets 2 and 3 were feedback-optimized with initially logarithmically spaced temperatures.
![](https://cdn.mathpix.com/cropped/2024_06_04_e9e9bc1d27b2833366ebg-08.jpg?height=704&width=834&top_left_y=764&top_left_x=1100)

FIG. 4. Time-to-epsilon of the different temperature sets used for PT-ICM and their individual scaling for $\varepsilon=0.75 \%$ (left) and $\varepsilon=1.1 \%$ (right). The four sets correspond, in order, to the sets shown in Table I. B5: Logarithmically-spaced with $\beta_{\max }=5$. F24: Feedback-optimized with 24 temperatures. F32: Feedback-optimized with 32 temperatures. B20: Logarithmically-spaced with $\beta_{\max }=20$.

## Software Implementation

Our PT-ICM implementation is available as the TAMC software package [2]. Our implementation of QAC for the D-Wave Advantage graph is available as part of the PegasusTools Python package [3].

We also implemented and considered the performance of simulated annealing but do not present its results here as this algorithm was not competitive at large problem sizes. Our PT-ICM implementation is a general-purpose solver written in the Rust programming language, targeting CPU execution, and accepts instances with any connectivity. This is in contrast to previous studies that have used simulated annealing or the Hamze-de Freitas-Selby algorithm solvers that are specialized for problems defined on the Chimera graph $[4,5]$.

## Statistical methods

For a given problem size and target optimality gap, we calculate the median $\mathrm{TT} \varepsilon$ from the quantum annealing data using a three-step Bayesian bootstrap procedure over the levels of readout samples, gauges (spinreversal transformations at the hardware level), and instances: (1) the success probability for each gauge is resampled from a beta distribution for $N_{\text {samp }}$ samples per gauge, (2) the statistical weight of each gauge is sampled from a Dirichlet distribution of length $N_{\text {gauge }}$ to take the weighted average success probability for each instance, and (3) the statistical weight of each instance is sampled from a Dirichlet distribution of length $N_{\text {inst }}$ to take the weighted median. We performed our quantum annealing experiments with $N_{\text {samp }}=1000(\mathrm{QAC}) / 3000(\mathrm{U} 3)$, $N_{\text {gauge }}=10$, and $N_{\text {inst }}=125$. We performed $N_{\text {boots }}=$ 200 bootstrap samples per size and energy density target pair and found the annealing time and penalty strength that resulted in an optimal $\mathrm{TT} \varepsilon$ for each bootstrap sample. The distribution of the optimal median TT $\varepsilon$ values and the distribution of optimal annealing parameters are the two final products of this sampling procedure shown in Fig. 2, with $2 \sigma$ error intervals for the optimal TT $\varepsilon$.

Next, we formulate a null hypothesis and compute $P$ values as follows. The null hypothesis is that the optimal annealing time $t_{f}$ is the minimum accessible $t_{f}^{\min }=0.5 \mu \mathrm{s}$, i.e., that the true optimal annealing time is not above $t_{f}^{\min }$. The $P$ value is the empirical number of bootstrap samples whose optimal $t_{f}$ was $0.5 \mu \mathrm{s}$, out of 200 samples. Filled circles in Fig. 2 mean $P<0.05$ for the probability that $t_{f}=0.5 \mu \mathrm{s}$ in the bootstrap sample, while open circles mean $P<0.20$. The filled circles show which points have the highest confidence that the optimal annealing time is not below $0.5 \mu \mathrm{s}$.

## Non-universality of the observed $\mathrm{KZ}$ critical exponent

The Binder cumulant $U$ is well-known to provide a statistical signature of phase transitions. Under the dynamic finite-size scaling ansatz, $U\left(L, t_{f}\right)$ is expected to collapse onto a common curve for all system sizes $N$ when $t_{f}$ is rescaled by $L^{-z-1 / \nu}$, where $\nu$ and $z$ are the correlation length and dynamic critical exponents, respectively. This reflects the $\mathrm{KZ}$ ansatz: the annealing time required for the system to remain adiabatic up to a correlation length of $L$ scales as

$$
\begin{equation*}
t_{f}(L) \sim L^{\mu}, \quad \mu=z+\frac{1}{\nu} \tag{4}
\end{equation*}
$$

While the collapse seen in Fig. 3 is visually convincing, the error estimates for $U$ are, unfortunately, too large to determine the $\mathrm{KZ}$ exponent or to extract $\nu$ and $z$. In addition, we do not observe that the extracted KZ exponent is universal: the estimate for $\mu$ that best collapses the Binder cumulant for binomial (as opposed to S28) spin glass instances is $\mu \approx 9 \pm 1$ for $\mathrm{U} 3$, and is reduced to $\mu \approx 7.5 \pm 0.6$ with $\mathrm{QAC}$ at $\lambda=0.1$. Thus, while the scaling ansatz yields a useful complementary way to quantify the advantage of QAC in terms of sampled quantities in addition to the $\mathrm{TT} \varepsilon$ metric, the lack of universality and the possibility that the spinglass transition temperature is zero [6] do not clearly support a universal critical description of the annealing dynamics, quite unlike the conclusions of Refs. [7, 8].

## Time-to-residual energy

Ref. [21] performed energy decay measurements of QA dynamics in 3D spin glass instances as captured through the residual energy, a dimensionless quantity,

$$
\begin{equation*}
\rho=\frac{\left\langle H_{z}\right\rangle-E_{0}}{J N} \tag{5}
\end{equation*}
$$

where $N$ is the number of spins and $J$ is the characteristic energy scale of the Ising Hamiltonian (which is simply 1 for all of our instance classes). This motivates an alternative measure for approximate optimization, which we call time-to-residual energy, or time-to-rho,

$$
\begin{equation*}
\operatorname{TTR}(N)=t_{f} \frac{\log (1-0.99)}{\log \left(1-p_{E \leq E_{0}+\rho J N}\right)} \tag{6}
\end{equation*}
$$

where now $\rho$ sets the target energy difference from the ground state in units of $J N$, rather than $E_{0}$ as in the case of TT $\varepsilon$. Nevertheless, both $\varepsilon$ and $\rho$ are targets that grow in proportion to the problem size or number of variables (in finite dimensions). The TTR analog of Fig. 2 at $\rho=1.1 \%$ is shown in Fig. 5. While the precise scaling exponents $\alpha$ of U3 and QAC vary slightly, the optimal annealing times are nearly identical. The trends of the scaling exponents as functions of either $\rho$ or $\varepsilon$ are also qualitatively similar. In finite-dimensional spin glasses, the variance of $E_{0}$ across instances appears to scale with $\sqrt{N}$ in the thermodynamic limit [9]. Hence, $E_{0} / N J$ converges to a constant, and there is effectively no distinction between $\varepsilon$ and $\rho$ in the thermodynamic limit.

## Classical complexity of approximate optimization

Under the TTR metric, it can be shown that the approximate optimization of finite-range spin glasses has, in theory, a linear scaling using a simple divide-and-conquer algorithm. Namely, for a given residual energy target $\rho$, an algorithm exists whose TTR scales linearly in the system size for a sufficiently large size, with a large prefactor depending on the residual energy. For 2D spin glasses on a square lattice, it can be summarized as follows:

1. Partition the size- $N=L^{2}$ instance into $K \times K$ subgraphs $G_{x, y}$, with $x, y \in\{1, \ldots, K\}$, each with constant side length of $L_{0}=L / K$ spins;
2. Find the local ground states for each subgraph using an exact or heuristic solver for each $L_{0} \times L_{0}$

![](https://cdn.mathpix.com/cropped/2024_06_04_e9e9bc1d27b2833366ebg-10.jpg?height=900&width=721&top_left_y=190&top_left_x=255)

FIG. 5. Time-to-rho performance and scaling of S28 instances, which are comparable to those using the time-toepsilon metric.

instance. This step has cost $C\left(L_{0}\right)$, potentially exponential in $L_{0}^{2}$ on a non-planar lattice;

3. Patch together all subgraph ground state configurations as the approximate ground state for the global Hamiltonian and return this state's energy as the approximate ground state energy $E^{*}(N)$. This last step requires $O(N)$ time due to the need to sum the energies of $K^{2}=O(N)$ patches.

The overall complexity is, therefore, $C\left(L_{0}\right) O(N)$.

This algorithm optimizes bulk energies throughout the volume of the spin-glass at the expense of large energy violations over regions scaling with the surface area of the spin-glass, i.e., at the patch boundaries. The boundary excitations become negligible for sufficiently large sizes compared to the bulk energies, with the latter eventually reaching the desired residual energy. More precisely, up to $4 L_{0}$ boundary couplings may be violated per volume of $L_{0}^{2}$, so the residual energy density for this algorithm is upper-bounded by $4 / L_{0}$. Thus, we estimate that the regime of system sizes where this algorithm applies for the $\rho$ targets we examine, e.g., $1 \%$, would require a reliable, efficient solver for instances with at least a sub-problem side length of $L_{0} \approx 400$, resulting in a prefactor of $2^{400} \sim 10^{120}$, which is entirely impractical. Nevertheless, this algorithm could be a starting point for parallel and quantum-classical hybrid algorithms for massive, finite-dimensional problems. Furthermore,
![](https://cdn.mathpix.com/cropped/2024_06_04_e9e9bc1d27b2833366ebg-10.jpg?height=754&width=894&top_left_y=173&top_left_x=1079)

FIG. 6. Time-to-epsilon scaling of spin glass instances with R6 disorder for (a) $\varepsilon=0.75 \%$ and (b) $\varepsilon=1.10 \%$. To expedite finding the optimal $\mathrm{TT} \varepsilon$, the annealing times were limited to the range $t_{f} \in[0.5,9] \mu \mathrm{s}$ and $\mathrm{QAC}$ was limited to $J_{p}=0.2$ only.

such an algorithm could reach a similar target $\varepsilon$ in linear time with increasing probability as the system size increases due to the decreasing variance of $E_{0} /(J N)$.

## Alternative spin-glass disorder cases

We performed a similar study of instances with binomial disorder $J= \pm 1$. UnsurprIsingly, there was no substantial difference in scaling between U3 and QAC, and additionally, we were unable to validate the optimal annealing time for QAC at $\varepsilon=1 \%$. Thus, we considered the binomial disorder instances unsuitably easy for our purposes.

In an attempt to estimate the influence of precision and ground state degeneracy in the S28 disorder, we also studied range 6 (R6) instances for which the couplings were randomly drawn from $J \in$ $\{ \pm 1 / 6, \pm 2 / 6, \pm 3 / 6, \pm 4 / 6, \pm 5 / 6, \pm 1\}$. The main contrast between R6 and S28 is that the minimum non-zero local field a qubit may experience is much smaller in S28 disorder. Furthermore, in the absence of the Sidon property, the R6 disorder is more susceptible to ground-state degeneracies due to floppy qubits. However, it is less likely to occur than in a smaller range disorder case such as range 3 .

The $\mathrm{TT} \varepsilon$ scaling for the R6 case is shown in Fig. 6. For a smaller $\varepsilon=0.75 \%$, the scaling of QAC is better with R6 disorder than S28 disorder (Fig. 2), going from 2.26 to 2.02 , then equalizes as $\varepsilon$ increases. Perhaps surprIsingly, the opposite is true of U3, which scales worse on R6 disorder at smaller epsilon (2.92 to 3.10 ) before scaling better than S28 disorder above $\varepsilon \approx 1.1 \%$ (2.02
to 1.82). That is, while S28 is overall more challenging than R6 under QAC, low energy sampling of R6 is more challenging for unprotected QA than S28. This is likely caused by the worse relative precision of small coupling strengths in unprotected QA, which would not affect the S28 disorder.

[1] H. G. Katzgraber, S. Trebst, D. A. Huse, and M. Troyer, Feedback-optimized parallel tempering Monte Carlo, Journal of Statistical Mechanics: Theory and Experiment 2006, P03018 (2006).

[2] H. Munoz Bauza, TAMC software package (2023).

[3] H. Munoz Bauza, PegasusTools Python package (2023).

[4] F. Hamze and N. de Freitas, From fields to trees, in Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence, UAI '04 (AUAI Press, Arlington, Virginia, USA, 2004) pp. 243-250.

[5] A. Selby, Efficient subgraph-based sampling of Ising-type models with frustration, arXiv:1409.3934 (2014).
[6] H. G. Katzgraber, F. Hamze, and R. S. Andrist, Glassy Chimeras could be blind to quantum speedup: Designing better benchmarks for quantum annealing machines, Physical Review X 4, 021008 (2014).

[7] A. D. King, S. Suzuki, J. Raymond, A. Zucca, T. Lanting, F. Altomare, A. J. Berkley, S. Ejtemaee, E. Hoskinson, S. Huang, E. Ladizinsky, A. J. R. MacDonald, G. Marsden, T. Oh, G. Poulin-Lamarre, M. Reis, C. Rich, Y. Sato, J. D. Whittaker, J. Yao, R. Harris, D. A. Lidar, H. Nishimori, and M. H. Amin, Coherent quantum annealing in a programmable 2,000 qubit Ising chain, Nature Physics 18, 1324 (2022).

[8] A. D. King, J. Raymond, T. Lanting, R. Harris, A. Zucca, F. Altomare, A. J. Berkley, K. Boothby, S. Ejtemaee, C. Enderud, E. Hoskinson, S. Huang, E. Ladizinsky, A. J. R. MacDonald, G. Marsden, R. Molavi, T. Oh, G. Poulin-Lamarre, M. Reis, C. Rich, Y. Sato, N. Tsai, M. Volkmann, J. D. Whittaker, J. Yao, A. W. Sandvik, and M. H. Amin, Quantum critical dynamics in a 5,000qubit programmable spin glass, Nature 617, 61 (2023).

[9] J.-P. Bouchaud, F. Krzakala, and O. C. Martin, Energy exponents and corrections to scaling in Ising spin glasses, Physical Review B 68, 224404 (2003).

</end of paper 0>


<paper 1>
# 4-clique Network Minor Embedding for Quantum Annealers 

Elijah Pelofske*1<br>${ }^{1}$ Los Alamos National Laboratory, CCS-3 Information Sciences


#### Abstract

Quantum annealing is a quantum algorithm for computing solutions to combinatorial optimization problems. This study proposes a method for minor embedding optimization problems onto sparse quantum annealing hardware graphs called 4-clique network minor embedding. This method is in contrast to the standard minor embedding technique of using a path of linearly connected qubits in order to represent a logical variable state. The 4-clique minor embedding is possible on Pegasus graph connectivity, which is the native hardware graph for some of the current D-Wave quantum annealers. The Pegasus hardware graph contains many cliques of size 4, making it possible to form a graph composed entirely of paths of connected 4-cliques on which a problem can be minor embedded. The 4-clique chains come at the cost of additional qubit usage on the hardware graph, but they allow for stronger coupling within each chain thereby increasing chain integrity, reducing chain breaks, and allow for greater usage of the available energy scale for programming logical problem coefficients on current quantum annealers. The 4-clique minor embedding technique is compared against the standard linear path minor embedding with experiments on two D-Wave quantum annealing processors with Pegasus hardware graphs. We show proof of concept experiments where the 4 -clique minor embeddings can use weak chain strengths while successfully carrying out the computation of minimizing random all-to-all spin glass problem instances.


## 1 Introduction

Quantum annealing is a type of analog quantum computation which is effectively a relaxed version of Adiabatic Quantum Computing (AQC). Quantum annealing is designed to solve combinatorial optimization problems by using quantum fluctuations in order to minimize an encoded problem Hamiltonian 1-8. This process of computation works by starting the system in the easy-to-prepare ground state of a Hamiltonian, and then slowly transitioning the system into a second Hamiltonian which we wish to find the ground state of (generally this second Hamiltonian corresponds to a problem which we do not know the ground state of because it is difficult to compute). In the transverse field Ising model version of quantum annealing, the system is put into an initial uniform superposition that is the ground state of the Hamiltonian:

$$
\begin{equation*}
H_{\text {initial }}=\sum_{i}^{n} \sigma_{i}^{x} \tag{1}
\end{equation*}
$$

Where $\sigma_{i}^{x}$ is the Pauli matrix for each qubit at index $i$. The user programmed problem Hamiltonian is then turned on over the course of the anneal:

$$
\begin{equation*}
H(t)=A(t) H_{\text {initial }}+B(t) H_{\text {ising }} \tag{2}
\end{equation*}
$$

Combined, $A(t)$ and $B(t)$ define the anneal schedules. Typically, at $t=0$ the $A(t)$ term is dominating as the system is prepared in the ground state of $H_{\text {initial }}$, and therefore the qubits are put into an initial superposition, and at the end of the anneal $B(t)$ (the problem Hamiltonian) is dominating. The annealing time over which these schedules are applied can be varied, and in the case of physically implemented quantum annealers the possible annealing time is constrained by the hardware. At the end of the anneal, the qubit states are read out as classical bits, which correspond to the variable states. Those samples are intended to be low energy solutions to the problem Ising $H_{\text {ising }}$ that the user has specified. For D-Wave quantum annealers, the user can program the anneal schedule - specifically the ratio between $A(s)$ and $B(s)$ (known as the anneal fraction) for each point in time during the anneal. The classical problem Hamiltonian, composed of single and two body terms, is defined as:[^0]

$$
\begin{equation*}
H_{\text {ising }}=\sum_{i}^{n} h_{i} \sigma_{i}^{z}+\sum_{i<j}^{n} J_{i j} \sigma_{i}^{z} \sigma_{j}^{z} \tag{3}
\end{equation*}
$$

The classical problem Hamiltonian is equivalent to large class of combinatorial optimization problems. Specifically for optimization problems where the decision variables are discrete; for Ising models the variable states can be either +1 or -1 . There are formulations known for converting many NP-Hard problems into discrete combinatorial optimization problems of this form, thereby allowing quantum annealers to sample low energy classical solutions of the programmed optimization problems. The company D-Wave has created a number of quantum annealing processors using superconducting flux qubits; these devices have been applied to a wide range of problems $4,9,10$. Quantum annealing has experimentally been shown to be competitive for good heuristic sampling of combinatorial optimization problems $11-14$ and simulation of frustrated magnetic systems 15-17. Although the size (e.g. the number of qubits) has been increasing as hardware development improves, the hardware of these quantum annealers is relatively sparsely connected, this limiting what problem connectivity graphs can be sampled on these devices.

Minor embedding is the only mechanism that allows logical problems with structure that does not directly match the underlying hardware graph to be programmed onto the hardware 18 25. In minor embedding for quantum annealing, each variable on the logical problem graph can be represented by a collection of physical qubits which are linked together ferromagnetically. The ferromagnetic coupling attempts to ensure that the variables representing each logical qubit are in agreement as to what the logical variable state is; there is an energy penalty for a qubit not having the same spin state as its neighbors. The standard minor embedding that is used creates a linear path of physically linked qubits, typically a linear nearest neighbors (LNN) graph, which form this ferromagnetically-bound logical variable. These linear path groups of qubits are typically referred to as chains.

The essential idea of computing graph minors, in particular for embedding a problem onto a fixed hardware graph, is that the minor does not need to be a linear path; it can be effectively any graph structure which is embeddable onto the target graph. The constraint is that the minor embeddings ideally should require as little additional hardware qubits (and couplers) as possible to fit more problems, or larger problems, onto the hardware chip. Therefore, the linear paths are used so as to reduce the overhead of using additional physical qubits, so that each logical variable can be routed so that the required logical quadratic variable interactions can occur.

In this article we propose a new method of minor embedding which we will refer to as 4 -clique network minor embedding. This method is based on a property of one of the D-Wave quantum annealing device connectivities, called Pegasus [24, 26 [28], which (while still quite sparse) contains a large number of 4 -cliques throughout its hardware lattice. The first D-Wave devices that were manufactured had a connectivity graph called Chimera 24 , 29, which is sparser than Pegasus. The newest generation of D-Wave quantum annealing hardware has a graph connectivity called Zephyr 28, 30. With the 4-cliques in the Pegasus hardware graph, it is possible to form connected paths of 4-cliques (see Figure 1) from 4-clique chains in order to create minor embeddings onto Pegasus chip hardware. While the 4 -clique minor embedding uses more qubits, it allows significantly more ferromagnetic couplers to be programmed in each chain, which reinforces the integrity of each chain more than a linear path connectivity. With linear path embeddings the measured qubits in a chain often disagree on their logical spin state, especially for very long chains. Chains that have measured spins which are not the same are referred to as a broken chains. Therefore, 4-clique embeddings allow more ferromagnetic chain break penalty weights to be used for each logical variable, thereby ideally reducing the number of chain breaks. The intuitive reasoning is that with a greater number of ferromagnetic couplers enforcing the state of each logical variable, there will be fewer chain breaks and therefore smaller chain strengths can be used.

One of the problems in general with minor embedding is that the strong ferromagnetic couplers can dominate the programmable energy range on the chip; current D-Wave devices have a set range of physical weights that the user can program to specify the problem Hamiltonian. There is limited precision when encoding these weights (approximately two decimal places), and therefore it is important to use as much of the physical programmable weight range as possible for the problem coefficients (instead of using those weights for encoding a minor embedding). Adding in the strong ferromagnetic chain couplers reduces the effective range that can be used to program the logical problem weights. The 4 -clique network minor embedding is therefore useful for a second reason; that is by increasing the number of ferromagnetic couplers per chain, the (relative) magnitude of the ferromagnetic couplers can be reduced compared to linear path embeddings which could allow for a larger effective programming weight range to be used for programming the weights of the logical problem.

We note that a related idea is Quantum Annealing Correction (QAC) 12, 31-37, where the states of problem variables are reinforced using ferromagnetic couplings to a common penalty qubit.

Section 2 describes the 4-clique graph construction in detail, along with creating some example minor embeddings. Section 3 describes the quantum annealing experimental results on the 4 -clique minor embeddings compared

![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-03.jpg?height=494&width=493&top_left_y=211&top_left_x=816)

Figure 1: A path of 4-cliques.

| D-Wave QPU chip id | Topology <br> name | Available <br> qubits | Available <br> couplers | Annealing time <br> (min, max $)$ |
| :--- | :--- | :--- | :--- | :--- |
| Advantage_system4.1 | Pegasus $P_{16}$ | 5627 | 40279 | $(0.5,2000)$ |
| Advantage_system6.1 | Pegasus $P_{16}$ | 5616 | 40135 | $(0.5,2000)$ |

Table 1: D-Wave quantum annealing processor hardware summary. Note that each of these devices have some hardware defects which cause the available hardware (qubits and couplers) to be smaller than the ideal Pegasus $P_{16}$ graph lattice structure.

Input: Hardware graph $G$

4-cliques $\leftarrow$ Compute all cliques of size 4 in $G$

for $K \in 4-c l i q u e s$ do

If any of the nodes in $K$ have already been contracted in a previous iteration, skip this iteration

Randomly choose two of the nodes $n_{1}, n_{2}$ from $K$

Contract edge $\left(n_{1}, n_{2}\right)$ to form a node called $n_{1} n_{2}$

Choose the remaining two nodes $n_{3}, n_{4}$ from $K$

Contract edge $\left(n_{3}, n_{4}\right)$ to form a node called $n_{3} n_{4}$

Remove any self edges that may have been generated from these two edge contractions

end for

Remove all nodes and edges in $G$ which were not formed by edge contraction. Specifically, remove all nodes that are not named with the form $n_{x} n_{y}$ and remove all edges that were not formed by edge contractions (e.g. remove all edges that do not have the form of $\left.\left(n_{x_{1}} n_{y_{1}}, n_{x_{2}} n_{y_{2}}\right)\right)$

12: Return: $G$

ALGORITHM 1: Contract hardware graph to a 4-clique network

to linear path minor embeddings when executed on D-Wave quantum annealers. Section 4 concludes with what the results show in regards the effectiveness of the 4 -clique minor embedding and future research questions. The figures in this article were generated using matplotlib 38, 39, networkx 40], and dwave-networkx 41] in Python 3. Data associated with this paper, including raw D-Wave measurements and minor embeddings, is available as a public dataset 42 .

## 2 Methods

Section 2.1 describes the 4-clique graph construction on a Pegasus graph, and how the minor embedding process works using this 4 -clique network. Section 2.2 describes the implementation of the 4-clique minor embeddings on quantum annealing hardware; specifically the problem instances which are used to compare the 4-clique and equivalent linear path minor embeddings are described, along with the D-Wave parameter settings used for the
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-04.jpg?height=1102&width=1684&top_left_y=198&top_left_x=210)

Figure 2: Contracted 4-clique graph of Advantage_system4.1 with kamada kawai layout (top left), spring layout (top middle), and spectral layout (top right). Contracted 4-clique graph of Advantage_system6. 1 with kamada kawai layout (bottom left), spring layout (bottom middle), and spectral layout (bottom right). The contracted 4-clique graph of Advantage_system4.1 has 2471 nodes and 6270 edges. The contracted 4-clique graph of Advantage_system6. 1 has 2463 nodes and 6245 edges. As defined by Algorithm 1 , an edge in a contracted 4-clique graph represents 4 edges in the underlying hardware graph, and a node represents 2 physical qubits in the hardware graph. These contracted clique graphs are quite sparse; the maximum clique of both of these graphs are 2. When the clique contraction is performed on the hardware graphs, there are many small unconnected components that are generated. These figures are showing only the largest connected component since it is the one which can be used for large minor embeddings.

experiments.

### 2.1 4-clique network minor embedding

Algorithm 1 constructs a network of 4-clique paths from a hardware graph connectivity. Algorithm 1 assumes that the hardware graph contains at least one clique of size 4 , otherwise it does not contract any edges in the graph all. For example, Algorithm 1 performs no contractions to a Chimera graph since the Maximum Clique of a Chimera lattice is 2. The largest connected component (there are other smaller components which are not connected to the main graph) of the contracted 4-clique graphs of Advantage_system6.1 and Advantage_system4.1 are shown using different layout algorithms in Figure 2. Note that the resulting contracted 4-clique graph from Algorithm 1 may not be connected, and may not have large connected components. The set of connected components, and thus the largest connected component, was computed using networkx 40,43. Once this contracted 4-clique graph of a target hardware graph has been created, standard minor embedding algorithms such as minorminer 44, 45, can be applied to create a minor embedding composed of chains of 4-cliques, such as in Figure 1. For the purpose of constructing the 4-clique graph from the contracted clique graph provided by Algorithm 1, one can take the subgraph of the device hardware graph induced by the separated nodes $n_{x}, n_{y}$, given by the name of the contracted nodes for all of the contracted nodes in the 4 -clique chain. Table 1 shows a hardware summary in terms of qubits and couplers for
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-05.jpg?height=826&width=1690&top_left_y=208&top_left_x=217)

Figure 3: Minor-embedding of a $K_{5}$ clique onto the target graph connectivity of Pegasus, using a 4-clique minor embedding (left) and an equivalent linear path minor embedding (right). Nodes in the graph are physical qubits, and edges are physical couplers. Grey edges represent the physical couplers onto which the problem specific coefficients would be encoded. Colored edges and nodes denote the minor embeddings; each chain (comprised of qubits and couplers) is uniquely colored. Notice that the 4 -clique minor embedding by default uses chains with length 2 as the smallest possible chains to encode a logical variable; these together form a single node pair $n_{x}, n_{y}$ that are two of the variables in a 4 -clique in the hardware. The linear path embedding by contrast does use chains of length 1 (meaning there is no chain). Qubits and couplers not used by the minor embedding are not shown. Because Pegasus natively has cliques of size 4 , it makes no sense to actually use a minor-embedding of size 4 . Therefore, this specific minor embedding diagram is purely for the purposes of describing the 4 -clique minor embedding algorithm.

the two Pegasus hardware graph D-Wave devices that are used for analyzing and implementing example spin glass problems, embedded using the 4 -clique network minor embeddings (and compared against the linear path minor embeddings). The largest connected component of the contracted 4-clique network for Advantage_system4.1 uses a proportion of 0.878 of the available hardware qubits, and a proportion of 0.623 of the available hardware couplers. The largest connected component of the contracted 4-clique network for Advantage_system6.1 uses a proportion of 0.877 of the available hardware qubits, and a proportion of 0.622 of the available hardware couplers.

For making two variable interactions possible in the 4 -clique minor embedding, since all chains are part of a 4-clique network already, every 4 -clique chain can be connected to another 4 -clique chain by 4 physical couplers for each single coupler that was computed in the minor embedding for the contracted clique graph. Note that although a linear path minor embedding is usually computed by forming a single path of connected qubits, it is possible for the minor embedding algorithm to use some branching if required; and therefore sometimes the standard minor embeddings are not strictly linear path (although almost always they do create linear path chains).

To provide a direct comparison between a 4-clique embedding and a linear path embedding, we can take any minor-embedding of a problem connectivity with the target of the contracted clique graphs (for example in Figure 22, and separate out a linear path path by taking one of the two variables in each node pair $n_{x}, n_{y}$. This means that we can compute a minor embedding of a problem on a contracted 4-clique graph, and then compute a linear path embedding of the same path length (but half the number of qubits in each chain) for the purpose of directly comparing the two embeddings since they have high hardware overlap.

In linear path minor embedding chains, the node degrees are usually either 2 , for nodes within the chain, or 1 for nodes at the end of the chain assuming the chain is strictly linear nearest neighbors (LNN). The minor embedding may utilize branching of a linear path, resulting in node degrees of 3 , for example. As shown by Figure 1 in 4 -clique chains the node degrees are 5 for nodes within the chain and degree 3 for nodes at the end of the chains (again, assuming the 4 -clique path is linear, and not branching).
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-06.jpg?height=816&width=1664&top_left_y=210&top_left_x=229)

Figure 4: Minor-embedding of a $K_{8}$ clique using a 4-clique minor embedding (left) and an equivalent linear path minor embedding (right) on a Pegasus graph. Grey edges represent the physical couplers onto which the problem specific coefficients would be encoded. Colored edges and nodes denote the minor embeddings; each chain (comprised of qubits and couplers) is uniquely colored. As in Figure 3, the parts of the Pegasus graph which are not used by the minor embeddings are not shown.

The largest all-to-all minor embeddings which could be constructed, using heuristic minor embedding algorithms, on the contracted 4-clique networks for both Advantage_system6. 1 and Advantage_system4.1 are 32 node cliques. Example 4-clique random minor embeddings for $K_{32}$ graphs are shown in Figure 5, overlayed onto the Pegasus hardware graph. Table 2 in Appendix B details all of the computed minor embedding chain length statistics for the 4-clique minor embeddings from $N=3, \ldots, 32$.

Currently, D-Wave quantum annealers have three distinct hardware connectivities, with the names of Chimera, Pegasus, and Zephyr. Chimera graphs are too sparse for the 4-clique minor embedding scheme (the maximum clique of a Chimera graph is 2). Pegasus works perfectly for this idea; in fact its highly connected 4-cliques motivated this idea. Zephyr also has cliques of size 4, however Zephyr can not form a large fully connected 4-clique network, instead the 4-clique contractions result in disconnected subgraphs. The contracted clique graph of a Zephyr $Z_{16}$ graph is shown in Appendix A. This means that large 4-clique minor embeddings could not be created using Zephyr hardware, whereas for Pegasus larger 4-clique minor embeddings can be created.

### 2.2 Implementation on Quantum Annealing Hardware

An important point when implementing 4-clique minor embeddings is that the number of qubits used is likely significantly more than an equivalent linear embedding. Therefore, like when minor embedding very large problems with linear path chains, it is important to consider whether the uniform problem coefficient spreading causes the programmed weights to fall below the machine precision. If this is the case, for very large minor embeddings, it may be necessary to create non-uniform problem weight encoding on the chains. The number of couplers used to actually encode quadratic terms could also be varied; in the 4 -clique embedding 4 couplers can be used, and those weights could be distributed in non-uniform ways (for example the weights could be placed entirely on one coupler). For the experimental results shown in Section 3, uniform weight distributions are used for both the linear and the quadratic terms.

Section 3 reports experimental energy results from using both the equivalent linear minor embedding and the 4-clique network minor embedding. The linear minor embedding is constructed from the 4 -clique minor embedding by taking only one linear path down the 4 -clique chain - and thereby using exactly one half of the qubits as the equivalent 4-clique minor embedding. This conversion from a 4-clique minor embedding to the linear path minor
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-07.jpg?height=852&width=1704&top_left_y=194&top_left_x=210)

Figure 5: $K_{32}$ 4-clique minor embedding on the $P_{16}$ Pegasus hardware graphs of Advantage_system4.1 (left) and Advantage_system6. 1 (right). Each of the 32 chains are uniquely colored. These are the largest all-to-all 4-clique minor embeddings that could be computed using minorminer in a reasonable amount of time.
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-07.jpg?height=684&width=1720&top_left_y=1238&top_left_x=210)

Figure 6: Sampling a single $N=8$ variable random spin glass instance using Advantage_system4.1 (left column), and Advantage_system6.1 (right column). Chain strength of 1.01 (top row) and chain strength of 1.1 (bottom row). Each plot is comprised of the spectrum of energy results from the 4-clique minor embedding in the left portion, and the corresponding equivalent linear path minor embedding energy spectrum plots in the right hand portion. Each set of data, for each combination of annealing time chain strength and minor embedding, is comprised of exactly 1000 samples. Annealing times (AT) of $0.5,1,10,100,1000$, and 2000 microseconds are used so as to evaluate how the two minor embeddings compare over different annealing times. Any outlier energy data points are represented as small blue dots, which may overlap on each other.

embedding is shown as side by side comparisons in Figures 3 and 4 . The chain strength is the primary parameter of interest when comparing these minor embeddings because that chain strength will be applied to both the linear path minor embeddings and the 4-clique minor embeddings. The relevant question is whether there is a difference between the two embeddings when the chain strength is the same. Importantly, the comparison against the linear
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-08.jpg?height=690&width=1736&top_left_y=180&top_left_x=194)

Figure 7: $N=10$ variable random spin glass sample energies from Advantage_system4.1 (left column), Advantage_system6.1 (right column). Chain strength of 1.01 (top row) and chain strength of 1.1 (bottom row). Each plot is comprised of the distribution of energy results from the 4-clique minor embedding in the left portion, and the corresponding equivalent linear path minor embedding energy spectrum plots in the right hand portion. Annealing times (AT) of $0.5,1,10,100,1000$, and 2000 microseconds are used so as to evaluate how the two minor embeddings compare over different annealing times.

path minor embeddings are a fair comparison of the impact of varying chain strength since the same region of the hardware graph is used. Note however that optimized linear path minor embeddings could have significantly shorter chain lengths and thus perform better for smaller problem instances where chain breaks and incoherent quantum annealing are dominant sources of error.

When executing the problem instances on the D-Wave quantum annealers, auto scaling is left on. Auto scaling is a backend parameter which, if left on, will scale the provided problem coefficients into the maximum energy scale that is possible on the quantum annealer, for both the linear and quadratic terms. Each parameter combination of the quantum annealer uses exactly 1000 anneals. The annealing time is varied for the purpose of observing what effect it has on the results of the computation. The parameter programming_thermalization was set to 0 microseconds, and all other programming parameters are set to their default values.

Another critical component of using minor embeddings is how to handle cases where the chains do not agree on the logical variable state (i.e. the chain is broken). There are simple chain break resolution methods such as majority vote which can classically repair the chain with post processing. With the aim of illustrating the chain break frequency, along with solution quality in the D-Wave results reported in Section 3 no classical post processing is used. In particular, any sample which contains a broken chain is set to have an energy value of 0 , meaning that the overall measurement statistics still have 1000 data points per setting.

The problem instances we will consider are random spin glasses, defined on an all-to-all connected graph $G=(V, E)$ of size $N$

$$
\begin{equation*}
C(z)=\sum_{v \in V} w_{v} z_{v}+\sum_{(i, j) \in E} w_{i j} z_{i} z_{j} \tag{4}
\end{equation*}
$$

Where $w_{i j}$ and $w_{v}$ denote random coefficients, chosen uniformly at random from $\{+1,-1\}$, making these problems effectively discrete-coefficient Sherrington-Kirkpatrick models 46 with local fields. The goal is to find the vector of variables $z=\left[z_{0}, z_{1}, \ldots, z_{N}\right]$ such that the cost function in eq. (4) is minimized, where the decision variables $z_{i}$ are spins $(+1$ or -1 ). This cost function evaluation, as it is an Ising model, is referred to as the energy for that given set of variable assignments. For each problem size $N$ that is tested, a new problem instance is generated with new random coefficients.
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-09.jpg?height=688&width=1736&top_left_y=184&top_left_x=194)

Figure 8: $N=15$ variable random spin glass energy distributions from Advantage_system4.1 (left column), Advantage_system6.1 (right column). Chain strength of 1.01 (top row) and chain strength of 1.1 (bottom row). Each plot is comprised of the spectrum of energy results from the 4-clique minor embedding in the left portion, and the corresponding equivalent linear path minor embedding energy spectrum plots in the right hand portion. Annealing times (AT) of $0.5,1,10,100,1000$, and 2000 microseconds are used so as to evaluate how the two minor embeddings compare for different annealing times.

## 3 Results

This section analyzes experimental results from executing random spin glasses on Advantage_system4.1 and Advantage_system6.1.

Figure 6 shows a side by side comparison of 4 -clique and linear path minor embeddings on $N=8$ random problem instances, using relatively small chain strengths of 1.1 and 1.01. Figure 6 shows that there is very little difference between the energy distributions for the two minor embeddings - both minor embeddings have very low chain break rates and both have converged to the same optimal solution. Note that this proof of concept with this 8 variable problem instance would perform better on the hardware if encoded using optimized small chain length linear path minor embeddings compared to the 4-clique network minor embedding given how small the problem instance is. We will probe larger problem instances next so as to see how more complicated 4-clique minor embeddings perform.

Figure 7 shows the same comparison as Figure 6, except the problem size was increased to $N=10$. In these plots, there is now a clear difference between the 4 -clique and the comparable linear path minor embeddings. At these very low chain strengths, the greater connectivity of the 4 -clique path minor embedding allowed the computation to remain stable and finds low energy solutions. By contrast, the linear path minor embedding has an extremely high chain break frequency and therefore the computations are not as robust at finding low energy solutions. This shows the 4-clique minor embedding is able to utilize a smaller proportion of the available programmable coefficient range on the hardware by being able to carry out the computation with only a chain strength of 1.01 (note that the hardware option of autoscaling is turned on), compared the the equivalent minor embedding technique. There are a few outlier instances where the linear path minor embedding is able to sample the optimal solution, but only when the chain strength was set to 1.1.

Figure 8 continues the trend observed in Figure 7 where the 4 -clique minor embedding is able to sample low energy solutions with minimal chain breaks compared to the equivalent linear path minor embedding for a $N=15$ problem instance. Notably, all of the samples for the linear path minor embedding had broken chains and therefore clearly performed worse than the 4 -clique minor embedding. This is a critical observation that the 4 -clique network minor embedding was able to obtain low energy samples of the Ising model using the comparatively extremely small chain strength of 1.01, leaving much of the available programmable coupler energy scale available for encoding of the coefficients of the original Ising model instead of dedicating that energy scale towards ferromagnetic chains.

Figure 9 shows the energy results for a $N=20$ problem instance. Because of the increased chain lengths (see Table 22, the small chain strengths of 1.01 and 1.1 do not work well for this problem size. Therefore, this plot also includes results for a chain strength of 1.4. Although the chain strength needed to be increased to see good low
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-10.jpg?height=1024&width=1740&top_left_y=184&top_left_x=190)

Figure 9: $N=20$ variable random spin glass sample energies from Advantage_system4.1 (left column), Advantage_system6.1 (right column). Chain strength of 1.01 (top row), chain strength of 1.1 (middle row), and chain strength of 1.4 (bottom row). Each plot is comprised of the distribution of energy results from the 4-clique minor embedding in the left portion of the plot, and the corresponding equivalent linear path minor embedding energy spectrum plots in the right hand portion of the plot. Annealing times (AT) of 0.5, 1, 10, 100, 1000, and 2000 microseconds are used so as to evaluate how the two minor embeddings compare over different annealing times.

energy state sampling with the 4-clique minor embedding, it is still the case that the 4-clique embedding performed better than the linear path minor embedding especially with respect to chain break frequency. At this very long chain size, the linear path minor embedding results always had samples with broken chains.

Figure 10 shows the energy results for a $N=32$ Ising model problem instance. The chain lengths for this problem size are dramatically larger than the all of the other tested minor embeddings (see Table 2 ), and consequently much larger chain strengths were required to obtain reasonable results. At a chain strength of 1.4, nearly all of the samples had broken chains or an energy of 0 ; only the 4 -clique minor embedding at 1000 and 2000 microsecond anneal times produced a few low energy samples. Once again, even at these larger chain strengths, the 4-clique minor embedding still had much fewer chain breaks compared to the linear path minor embedding and therefore better samples, at least at sufficiently long annealing times (e.g. 1000 or 2000 microseconds). However, here we begin to see where too large of chain strengths can be detrimental for the 4 -clique minor embedding. At a chain strength of 8 , the 4-clique energy spectrum begins to clearly get worse than 0 , whereas with a chain strength of 5 the energy results were actually better than with a chain strength of 8 . This shows that the chain strength used in the 4 -clique minor embedding needs to be carefully tuned such that results do not get worse because of the chain strength using too much of the available programmable coupler weight. At comparatively small chain strengths the 4-clique minor embedding performs very well.

Figures 6, 7, 8, 9, and 10 all show there is a clear trend for the 4-clique minor embedding across annealing times; the longer annealing times result in consistently lower energy solutions. These figures also all show that Advantage_system4.1 samples lower energy solutions more consistently compared to Advantage_system6.1.
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-11.jpg?height=1350&width=1740&top_left_y=184&top_left_x=190)

Figure 10: $N=32$ variable random spin glass energy results from Advantage_system4.1 (left column), Advantage_system6.1 (right column). Chain strength of 1.4 (top row), chain strength of 2 (top-middle row), chain strength of 5 (middle-bottom row), chain strength of 8 (bottom row). Each plot is comprised of the spectrum of energy results, represented by separate boxplots, from the 4 -clique minor embedding in the left portion, and the corresponding equivalent linear path minor embedding energy spectrum plots in the right hand portion. Annealing times (AT) of $0.5,1,10,100,1000$, and 2000 microseconds are used so as to evaluate how the two minor embeddings compare over different annealing times.

## 4 Discussion and Conclusion

In this article we have introduced a new method for minor embedding discrete combinatorial optimization problems onto D-Wave quantum annealers with hardware graphs that contain networks of 4-cliques. Importantly, we do not expect this method to likely be used for the current $P_{16}$ Pegasus devices. The reason for this is that the largest minor embeddings which can be formed from the contracted 4-clique networks of $P_{16}$ graphs are not especially large in comparison to equivalent linear path minor embeddings which can be computed, and in particular linear path minor embeddings have considerably shorter chain lengths for small problem sizes and will thus perform considerably better and use less of the hardware graph than the proposed 4-clique network minor embedding. The reason for this is because of the longer chain lengths required to create a minor embedding on the contracted 4-clique graphs; linear path minor embeddings can be constructed to use much shorter chain lengths than the side by side comparisons done in Section 3. For example, for problem instances with a small number of variables (such as on the order of 10 variables) and a standard coefficient precision range, the standard optimized linear path minor embeddings will perform much better than a 4-clique network minor embedding on a Pegasus graph due to decreased hardware usage and likely more coherent quantum annealing. The regime where the 4 -clique network minor embedding could
be potentially useful is for much larger quantum annealers (which can form a 4-clique network, such as devices with a Pegasus hardware graph) where minor embedding of significantly large problem sizes (e.g. hundreds or thousands of logical variables) will require exceedingly large ferromagnetic chain strengths and long chains using the equivalent linear path chain minor embedding (which, if the chains are sufficiently long, will be prone to breaking). There, the 4-clique chains could provide lower levels of chain breaks due to the increased inter-chain integrity and require a smaller proportion of the programmable energy scale of the quantum annealer compared to alternative minor embedding methods. For example, if for a fully connected problem instance embedded on a large sparse hardware graph the linear path chain lengths are already significantly long such that significant chain breaks occur (and the maximum chain strengths are limited), then the 4-clique network minor embedding could be applied in order to perform improved quantum annealing on that hardware platform. On sparse hardware graphs, fully connected minor embeddings quickly use the available hardware graph as the number of variables increases and thus use long chains. Therefore 4-clique network minor embeddings could be used most effectively for fully connected minor embeddings on future large quantum annealing hardware graphs where long chains are inevitable even for linear path minor embeddings. Because of the limited size of current quantum annealing hardware, this type of large problem instance minor embedding comparison can not be performed on current D-Wave quantum annealers but this comparison should be performed on future quantum annealing hardware.

Additionally, even for small problem sizes, if the energy scale range of the problem coefficients is incredibly important for a minor embedded problem, then the 4-clique network minor embedding could be employed to make more of the coefficient range available on the chip to be used for problem coefficients, instead of inter-chain couplers. This type of problem may occur for optimization problems where the high precision of the coefficients being programmed onto the hardware graph is important to represent the problem faithfully compared to what can be programmed using linear path minor embeddings, which can occur if the coefficient range of the original problem is large.

There are several future research questions that can be considered:

1. Make the weights on the chain non-uniform - perhaps make them proportional to their degree within the 4-clique chain.
2. Use flux bias offset and anneal offset calibration 47, 48 to balance the chain statistics in the minor embeddings, both for 4-clique and the equivalent linear minor embeddings, with the goal of reducing bias in the quantum annealing samples.
3. Create more structured minor embeddings of the contracted 4-clique graphs, specifically with the goal of making more uniform chain lengths in the minor embeddings.
4. Investigate encoding variable states across even larger subgraphs of the hardware topology. By encoding variable states into larger pieces of the physical hardware, it becomes harder for noise to induce errors in the physical group of qubits.
5. Investigate how the 4-clique chains break. Is there a pattern with respect to the qubit degree within the 4-clique chain, or with respect to the position of the qubit in the chain?
6. Numerical simulations of exact quantum state evolution when using the standard linear minor embedding compared to the proposed 4-clique network minor embedding. Currently, such classical simulators are severely limited in the total number of qubits that can be simulated, making comparisons between even smaller toy examples computationally difficult. Future improved quantum annealing simulation software could be used to perform such numerical simulations, giving better insights on how these two minor embedded methods work.
7. Use denser minor-embeddings, such as the 4-clique network minor embedding, selectively and not for every single logical variable. For example, nodes in the logical graph which are highly connected could be represented by a large dense chain (such as a 4-clique chain), while less dense variables are represented by smaller chains.

## 5 Acknowledgments

This work was supported by the U.S. Department of Energy through the Los Alamos National Laboratory. Los Alamos National Laboratory is operated by Triad National Security, LLC, for the National Nuclear Security Administration of U.S. Department of Energy (Contract No. 89233218CNA000001). The research presented in this article was supported by the Laboratory Directed Research and Development program of Los Alamos National Laboratory under project numbers 20220656ER and 20210114ER and by the NNSA's Advanced Simulation and Computing Beyond Moore's Law Program. This research used resources provided by the Los Alamos National Laboratory Institutional Computing Program, which is supported by the U.S. Department of Energy National

Nuclear Security Administration under Contract No. 89233218CNA000001. Thanks to Carleton Coffrin for helpful discussions. This work has been assigned the LANL technical report number LA-UR-23-20504.

## A Contracted 4-clique graphs on Zephyr

Figure 11 shows the contracted 4-clique graphs for a logical Zephyr $Z_{16}$ graph with no hardware defects.
![](https://cdn.mathpix.com/cropped/2024_06_04_d0a50e5d4ba09274174cg-13.jpg?height=832&width=1652&top_left_y=516&top_left_x=258)

Figure 11: Contracted 4-clique graph for a Zephyr $Z_{16}$ graph (left), which is composed of 31 unconnected graphs. Each of those 31 unconnected subgraphs are isomorphic to each other; their structure is shown in the right side plot. These graphs are drawn using the kamada kawai layout algorithm.

## B Contracted 4-clique Pegasus minor embedding chain lengths

Table 2 details chain length statistics of the computed contracted 4-clique random minor embeddings.

| $\mathrm{N}$ | Advantage_system4.1 <br> 4-clique minor embedding chain <br> lengths <br> (min, mean $\pm \sigma, \max )$ | Advantage_system6.1 <br> 4-clique minor embedding chain <br> lengths <br> (min, mean $\pm \sigma$, max $)$ |
| :---: | :--- | :--- |
| 3 | $(2,2.667 \pm 0.943,4)$ | $(2,2.667 \pm 0.943,4)$ |
| 4 | $(2,3.0 \pm 1.0,4)$ | $(2,3.0 \pm 1.0,4)$ |
| 5 | $(2,3.2 \pm 0.98,4)$ | $(2,3.2 \pm 0.98,4)$ |
| 6 | $(2,4.0 \pm 1.633,6)$ | $(2,4.0 \pm 1.633,6)$ |
| 7 | $(2,4.857 \pm 2.1,8)$ | $(2,4.857 \pm 2.1,8)$ |
| 8 | $(4,7.0 \pm 2.236,10)$ | $(4,7.0 \pm 2.646,10)$ |
| 9 | $(4,8.222 \pm 2.393,12)$ | $(4,8.222 \pm 2.393,12)$ |
| 10 | $(6,10.4 \pm 2.939,14)$ | $(6,10.4 \pm 2.939,14)$ |
| 11 | $(8,13.818 \pm 3.459,18)$ | $(6,12.182 \pm 3.242,18)$ |
| 12 | $(10,15.667 \pm 3.986,22)$ | $(8,16.167 \pm 6.189,26)$ |
| 13 | $(12,18.769 \pm 4.475,26)$ | $(12,19.231 \pm 4.933,28)$ |
| 14 | $(14,23.429 \pm 5.368,30)$ | $(12,22.429 \pm 5.716,28)$ |
| 15 | $(22,27.467 \pm 3.222,32)$ | $(16,25.867 \pm 4.646,34)$ |
| 16 | $(16,31.75 \pm 7.71,42)$ | $(18,30.75 \pm 7.137,42)$ |
| 17 | $(24,34.824 \pm 5.576,42)$ | $(16,35.647 \pm 10.295,46)$ |
| 18 | $(26,37.778 \pm 6.25,46)$ | $(24,37.222 \pm 7.634,46)$ |
| 19 | $(26,43.263 \pm 7.9,52)$ | $(32,41.579 \pm 7.802,56)$ |
| 20 | $(34,47.7 \pm 7.246,58)$ | $(28,47.5 \pm 11.897,64)$ |
| 21 | $(36,50.476 \pm 7.998,60)$ | $(32,50.095 \pm 7.47,60)$ |
| 22 | $(32,54.455 \pm 9.552,66)$ | $(32,51.909 \pm 11.305,64)$ |
| 23 | $(40,60.261 \pm 13.835,82)$ | $(38,56.087 \pm 7.235,64)$ |
| 24 | $(44,62.417 \pm 11.604,84)$ | $(34,61.667 \pm 13.972,80)$ |
| 25 | $(44,66.48 \pm 9.753,78)$ | $(44,63.92 \pm 10.438,82)$ |
| 26 | $(44,70.462 \pm 13.165,92)$ | $(52,69.538 \pm 9.548,86)$ |
| 27 | $(54,73.185 \pm 9.495,86)$ | $(56,73.778 \pm 8.35,84)$ |
| 28 | $(60,81.214 \pm 13.356,106)$ | $(56,79.5 \pm 9.571,92)$ |
| 29 | $(60,85.31 \pm 12.879,104)$ | $(54,77.655 \pm 12.606,96)$ |
| 30 | $(66,94.4 \pm 16.584,122)$ | $(62,92.867 \pm 14.603,112)$ |
| 31 | $(70,96.839 \pm 18.815,124)$ | $(68,94.645 \pm 14.063,116)$ |
| 32 | $(62,99.625 \pm 20.152,134)$ | $(70,97.688 \pm 20.689,132)$ |

Table 2: Summary statistics for random 4-clique all-to-all minor embedding chain lengths that were computed using several iterations of the minorminer embedding heuristic. The all-to-all minor embeddings were computed for sizes 3 through 32 , represented as each row in the table. Specifically, of the chain lengths in the minor embedding, the minimum, maximum, mean, and standard deviation of those lengths are reported for the minor embeddings computed on the contracted 4-clique graphs of Advantage_system4.1 and Advantage_system6.1. Here chain length is referring to the total number of physical qubits used in the minor embedding. All quantities are rounded to three decimal places.

## References

[1] Edward Farhi et al. Quantum Computation by Adiabatic Evolution. 2000. DOI: 10. 48550/ARXIV . QUANTPH/0001106, URL: https://arxiv.org/abs/quant-ph/0001106.

[2] Satoshi Morita and Hidetoshi Nishimori. "Mathematical foundation of quantum annealing". In: Journal of Mathematical Physics 49.12 (2008), p. 125210. DOI: 10.1063/1.2995837.

[3] Arnab Das and Bikas K Chakrabarti. "Colloquium: Quantum annealing and analog quantum computation". In: Reviews of Modern Physics 80.3 (2008), p. 1061. DOI: 10.1103/revmodphys.80.1061.

[4] Philipp Hauke et al. "Perspectives of quantum annealing: methods and implementations". In: Reports on Progress in Physics 83.5 (2020), p. 054401. DOI: 10.1088/1361-6633/ab85b8. URL: https://dx.doi.org/ $10.1088 / 1361-6633 / a b 85 b 8$.

[5] T. Lanting et al. "Entanglement in a Quantum Annealing Processor". In: Phys. Rev. X 4 (2 2014), p. 021041. DOI: 10.1103/PhysRevX.4.021041. URL: https://link.aps.org/doi/10.1103/PhysRevX.4.021041.

[6] Giuseppe E Santoro and Erio Tosatti. "Optimization using quantum mechanics: quantum annealing through adiabatic evolution". In: Journal of Physics A: Mathematical and General 39.36 (2006), R393. DOI: 10.1088/ 0305-4470/39/36/R01. URL: https://dx.doi.org/10.1088/0305-4470/39/36/R01.

[7] Mark W Johnson et al. "Quantum annealing with manufactured spins". In: Nature 473.7346 (2011), pp. 194198. DOI: $10.1038 /$ nature10012.

[8] Sergio Boixo et al. "Experimental signature of programmable quantum annealing". In: Nature communications 4.1 (2013), pp. 1-8. DOI: $10.1038 /$ ncomms3067.

[9] Davide Venturelli et al. "Quantum Optimization of Fully Connected Spin Glasses". In: Phys. Rev. X 5 (3 2015), p. 031040. DOI: 10.1103 /PhysRevX . 5 . 031040. URL: https : / / link . aps . org / doi / 10 . 1103/ PhysRevX.5.031040.

[10] Zsolt Tabi et al. "Quantum Optimization for the Graph Coloring Problem with Space-Efficient Embedding". In: 2020 IEEE International Conference on Quantum Computing and Engineering (QCE). 2020, pp. 56-62. DOI: 10.1109/QCE49297.2020.00018.

[11] Byron Tasseff et al. On the Emerging Potential of Quantum Annealing Hardware for Combinatorial Optimization. 2022. DOI: 10.48550/ARXIV.2210.04291. URL: https://arxiv.org/abs/2210.04291.

[12] Humberto Munoz Bauza and Daniel A. Lidar. Scaling Advantage in Approximate Optimization with Quantum Annealing. 2024. arXiv: 2401.07184 [quant-ph].

[13] Tameem Albash and Daniel A. Lidar. "Demonstration of a Scaling Advantage for a Quantum Annealer over Simulated Annealing". In: Physical Review X 8.3 (2018). DOI: 10.1103/physrevx . 8.031016. URL: https://doi.org/10.1103\%2Fphysrevx.8.031016.

[14] Elijah Pelofske, Georg Hahn, and Hristo N. Djidjev. "Parallel quantum annealing". In: Scientific Reports 12.1 (Mar. 2022). ISSN: 2045-2322. DOI: 10.1038/s41598-022-08394-8, URL: http://dx.doi.org/10.1038/ s41598-022-08394-8.

[15] Andrew D. King et al. "Scaling advantage over path-integral Monte Carlo in quantum simulation of geometrically frustrated magnets". In: Nature Communications 12.1 (2021). DOI: 10.1038/s41467-021-20901-5, URL: https://doi.org/10.1038\%2Fs41467-021-20901-5.

[16] Andrew D. King et al. "Qubit spin ice". In: Science 373.6554 (2021), pp. 576-580. DOI: 10.1126/science. abe2824. URL: https://doi.org/10.1126\%2Fscience.abe2824.

[17] Alejandro Lopez-Bezanilla et al. "Kagome qubit ice". In: Nature Communications 14.1 (Feb. 2023). ISSN: 2041-1723. DOI: 10.1038/s41467-023-36760-1. URL: http://dx.doi.org/10.1038/s41467-023-36760-1.

[18] Christine Klymko, Blair D Sullivan, and Travis S Humble. "Adiabatic quantum programming: minor embedding with hard faults". In: Quantum information processing 13.3 (2014), pp. 709-729. DOI: $10.1007 /$ s11128013-0683-9.

[19] David E Bernal et al. "Integer programming techniques for minor-embedding in quantum annealers". In: International Conference on Integration of Constraint Programming, Artificial Intelligence, and Operations Research. Springer. 2020, pp. 112-129. DOI: $10.1007 / 978-3-030-58942-4 \_8$.

[20] Vicky Choi. "Minor-embedding in adiabatic quantum computation: II. Minor-universal graph design". In: Quantum Information Processing 10.3 (2011), pp. 343-353. DOI: 10.1007/s11128-010-0200-3.

[21] Vicky Choi. "Minor-embedding in adiabatic quantum computation: I. The parameter setting problem". In: Quantum Information Processing 7.5 (2008), pp. 193-209. DOI: $10.1007 /$ s11128-008-0082-9.

[22] Tomas Boothby, Andrew D King, and Aidan Roy. "Fast clique minor generation in Chimera qubit connectivity graphs". In: Quantum Information Processing 15.1 (2016), pp. 495-508. DOI: 10.1007/s11128-015-1150-6.

[23] Mario S. Könz et al. "Embedding Overhead Scaling of Optimization Problems in Quantum Annealing". In: PRX Quantum 2 (4 2021), p. 040322. DOI: 10.1103/PRXQuantum.2.040322. URL: https://link.aps.org/ doi/10.1103/PRXQuantum.2.040322.

[24] Stefanie Zbinden et al. "Embedding algorithms for quantum annealers with chimera and pegasus connection topologies". In: International Conference on High Performance Computing. Springer. 2020, pp. 187-206.

[25] Andrew Lucas. "Hard combinatorial problems and minor embeddings on lattice graphs". In: Quantum Information Processing 18.7 (2019), pp. 1-38. DOI: $10.1007 /$ s11128-019-2323-5.

[26] Kelly Boothby et al. Next-Generation Topology of D-Wave Quantum Processors. 2020. DOI: 10.48550/ARXIV. 2003.00133. URL: https://arxiv.org/abs/2003.00133.

[27] Nike Dattani, Szilard Szalay, and Nick Chancellor. Pegasus: The second connectivity graph for large-scale quantum annealing hardware. 2019. DOI: 10.48550/ARXIV.1901.07636. URL: https://arxiv.org/abs/ 1901.07636 .

[28] Kelly Boothby et al. Architectural considerations in the design of a third-generation superconducting quantum annealing processor. 2021. arXiv: 2108.02322 [quant-ph].

[29] Elisabeth Lobe, Lukas Schürmann, and Tobias Stollenwerk. "Embedding of complete graphs in broken Chimera graphs". In: Quantum Information Processing 20.7 (2021). DOI: $10.1007 / \mathrm{s} 11128-021-03168-\mathrm{z}$. URL: https://doi.org/10.1007\%2Fs11128-021-03168-z.

[30] DWave Networkx Zephyr Graph. https://web.archive.org/web/20230608182151/https://docs.ocean. dwavesys.com/en/stable/docs_dnx/reference/generated/dwave_networkx.zephyr_graph.html.

[31] Kristen L Pudenz, Tameem Albash, and Daniel A Lidar. "Quantum annealing correction for random Ising problems". In: Physical Review A 91.4 (2015), p. 042302. DOI: 10.1103/PhysRevA.91.042302.

[32] Walter Vinci et al. "Quantum annealing correction with minor embedding". In: Physical Review A 92.4 (2015), p. 042310. DOI: $10.1103 /$ PhysRevA.92.042310.

[33] Walter Vinci, Tameem Albash, and Daniel A Lidar. "Nested quantum annealing correction". In: npj Quantum Information 2.1 (2016), pp. 1-6. DOI: 10.1038/npjqi.2016.17.

[34] Anurag Mishra, Tameem Albash, and Daniel A Lidar. "Performance of two different quantum annealing correction codes". In: Quantum Information Processing 15.2 (2016), pp. 609-636. DOI: $10.1007 / \mathrm{s} 11128-$ 015-1201-z.

[35] Kristen L Pudenz, Tameem Albash, and Daniel A Lidar. "Error-corrected quantum annealing with hundreds of qubits". In: Nature communications 5.1 (2014), pp. 1-10. DOI: 10.1038/ncomms4243.

[36] Shunji Matsuura et al. "Nested quantum annealing correction at finite temperature: p-spin models". In: Physical Review A 99.6 (June 2019). ISSN: 2469-9934. DOI: $10.1103 /$ physreva. 99.062307. URL: http: //dx.doi.org/10.1103/PhysRevA.99.062307.

[37] Shunji Matsuura et al. "Quantum-annealing correction at finite temperature: Ferromagnetic p-spin models". In: Physical Review A 95.2 (Feb. 2017). ISSN: 2469-9934. DOI: 10.1103/physreva. 95.022308, urL: http: //dx.doi.org/10.1103/PhysRevA.95.022308

[38] Thomas A Caswell et al. matplotlib/matplotlib. Version v3.4.3. DoI: 10.5281/zenodo.5194481.

[39] J. D. Hunter. "Matplotlib: A 2D graphics environment". In: Computing in Science $\mathcal{E}$ Engineering 9.3 (2007), pp. 90-95. DOI: 10.1109/MCSE. 2007.55.

[40] Aric Hagberg, Pieter Swart, and Daniel S Chult. Exploring network structure, dynamics, and function using NetworkX. Tech. rep. Los Alamos National Lab.(LANL), Los Alamos, NM (United States), 2008.

[41] D-Wave NetworkX.https://web.archive.org/web/20230401000000*/https://github.com/dwavesystems/ dwave-networkx.

[42] Elijah Pelofske. Dataset for 4-clique network minor embedding for quantum annealers. Jan. 2023. DOI: 10. 5281/zenodo.7552776. URL: https://doi.org/10.5281/zenodo.7552776.

[43] Networkx connected components. https://web.archive.org/web/20230907060839/https://networkx. org/documentation/stable/reference/algorithms/generated/networkx . algorithms . components . connected_components.html.

[44] Jun Cai, William G. Macready, and Aidan Roy. A practical heuristic for finding graph minors. 2014. DOI: 10.48550/ARXIV.1406.2741. URL: https://arxiv.org/abs/1406.2741.

[45] minorminer. https://web.archive.org/web/20230401000000*/https://github.com/dwavesystems/ minorminer.

[46] David Sherrington and Scott Kirkpatrick. "Solvable Model of a Spin-Glass". In: Phys. Rev. Lett. 35 (26 1975), pp. 1792-1796. DOI: 10.1103/PhysRevLett.35.1792. URL: https://link.aps.org/doi/10.1103/ PhysRevLett.35.1792.

[47] Kevin Chern et al. "Tutorial: Calibration refinement in quantum annealing". In: arXiv preprint (2023). arXiv: 2304.10352 .

[48] D-Wave Error-Correction Features. https://web.archive.org/web/20240000000000*/https://docs . dwavesys.com/docs/latest/c_qpu_error_correction.html.


[^0]:    *Email: epelofske@lanl.gov

</end of paper 1>


