<paper 0>
# VQE-generated Quantum Circuit Dataset for Machine Learning 

Akimoto Nakayama, ${ }^{1}$ Kosuke Mitarai, ${ }^{1,2, *}$ Leonardo Placidi, ${ }^{1,2}$ Takanori Sugimoto, ${ }^{2}$ and Keisuke Fujii ${ }^{1,2,3, \dagger}$<br>${ }^{1}$ Graduate School of Engineering Science, Osaka University,<br>1-3 Machikaneyama, Toyonaka, Osaka 560-8531, Japan<br>${ }^{2}$ Center for Quantum Information and Quantum Biology,<br>Osaka University, 1-2 Machikaneyama, Toyonaka 560-0043, Japan<br>${ }^{3}$ RIKEN Center for Quantum Computing (RQC), Hirosawa 2-1, Wako, Saitama 351-0198, Japan

(Dated: June 2, 2023)


#### Abstract

Quantum machine learning has the potential to computationally outperform classical machine learning, but it is not yet clear whether it will actually be valuable for practical problems. While some artificial scenarios have shown that certain quantum machine learning techniques may be advantageous compared to their classical counterpart, it is unlikely that quantum machine learning will outclass traditional methods on popular classical datasets such as MNIST. In contrast, dealing with quantum data, such as quantum states or circuits, may be the task where we can benefit from quantum methods. Therefore, it is important to develop practically meaningful quantum datasets for which we expect quantum methods to be superior. In this paper, we propose a machine learning task that is likely to soon arise in the real world: clustering and classification of quantum circuits. We provide a dataset of quantum circuits optimized by the variational quantum eigensolver. We utilized six common types of Hamiltonians in condensed matter physics, with a range of 4 to 20 qubits, and applied ten different ansätze with varying depths (ranging from 3 to 32) to generate a quantum circuit dataset of six distinct classes, each containing 300 samples. We show that this dataset can be easily learned using quantum methods. In particular, we demonstrate a successful classification of our dataset using real 4-qubit devices available through IBMQ. By providing a setting and an elementary dataset where quantum machine learning is expected to be beneficial, we hope to encourage and ease the advancement of the field.


## I. INTRODUCTION

Quantum machine learning has attracted much attention in recent years as a promising application of quantum computers $[1,2]$. Many techniques, such as quantum neural networks $[3-5]$, quantum generative models $[6,7]$, quantum kernel methods [8], and so on have been developed for achieving possible quantum speedups in machine learning tasks. They have also been realized experimentally [7-11]. While some artificial, carefully-designed scenarios have demonstrated that certain quantum machine learning techniques may be advantageous compared to classical methods [12-16], it is not yet clear whether quantum techniques would be beneficial for practical applications.

In traditional machine learning, standard datasets, such as MNIST handwritten digits [17], are used to evaluate the performance and thus the practicality of new models. However, it is rather unlikely that quantum machine learning methods can outperform the state-ofthe-art classical machine learning procedures on those datasets, looking at their recent great success. With a large-scale numerical experiment involving up to 30 qubits, Huang et al. [13] have shown that the FashionMNIST dataset [18] is better learned by classical models. In contrast, when working with "quantum data", such as quantum states or circuits, there is a good reason to be-[^0]

lieve that quantum computers may provide a significant advantage. In another work by Huang et al. [19], it has been rigorously shown that quantum machine learning is beneficial when learning unknown quantum states or processes provided from physical experiments. It is therefore important to develop a practical quantum dataset in which we can expect quantum methods to be superior.

Several works have made efforts in this direction. Schatski et al. [20] proposed a dataset consisting of parameterized quantum circuits with different structure whose parameters are optimized to give output states with certain values of entanglement. Also, Huang et al. [13] proposed to relabel a classical dataset by outputs of quantum circuits so that the relabeled one is difficult to be learned by classical computers. These examples, while giving datasets that are possibly hard to learn by classical computers, are not plausible "real-world" quantum data. Note that Perrier, Youssry and Ferrie [21] provides a quantum dataset called QDataSet, but its aim is to provide a benchmark for classical machine learning applied to quantum physics, and therefore it does not fit to the context of this paper.

In this work, we propose a more practical machine learning task that we expect to naturally arise in near future and in real-world: a clustering and classification of many quantum circuits, and provide an elementary dataset for this task. A successful model that can perform such a task could be beneficial to providers of cloud quantum computers; it would allow them to understand the preferences of their users by analyzing circuits submitted by them. While there are many possible ways to analyze the circuits, in this work, we focus on a setting

![](https://cdn.mathpix.com/cropped/2024_06_04_12e0b57a5e5701d5b765g-2.jpg?height=355&width=833&top_left_y=172&top_left_x=185)

FIG. 1. Scenario of the proposed machine learning task. Cloud quantum computer providers get descriptions of quantum circuits constantly from their users and return execution results to them. The providers wish to investigate the user activities, e.g., whether two users are interested in solving a similar computational task or not, from the circuit data.

where the providers want to cluster or classify circuits based on their similarity of output states. This task is easy when we have access to quantum computers because similarity, which can be measured in terms of overlaps between output states, can be readily estimated on them. Such estimation is likely to be hard for classical computers when a quantum circuit is large enough.

The quantum dataset provided in this work is a set of quantum circuits optimized by the variational quantum eigensolver (VQE) [22, 23], which is an algorithm to find circuits that output approximate ground states of quantum systems. More specifically, we use six model Hamiltonians that are famous in condensed matter physics, and optimize 300 different parametrized circuits with varying structure and depth to output each of their ground states. The dataset includes circuits with up to 20 qubits, but it can be easily extended to larger numbers of qubits with access to quantum hardware. To demonstrate the potential of the dataset, we perform a proof-of-principle experiment using quantum circuit simulators and show that the circuits can be accurately clustered. We also demonstrate successful classification of our 4-qubit dataset using real quantum hardware.

The dataset is freely accessible on the GitHub repository [24], therefore the reader will be able to freely use it for research or benchmarking. For each number of qubit, data are stored in QASM [25] format and are publicly accessible. By providing a natural setting and an elementary dataset where quantum machine learning is expected to be beneficial, we hope to support the thoughtful and grounded advancement of the field.

## II. DATASET CONSTRUCTION

## A. Idea overview

The machine learning task that we consider in this work is clustering and classification of quantum circuits based on the similarity of their output states. More specifically, the task is to classify $M$ quantum circuits $\left\{U_{m} \mid m=1,2, \cdots, M\right\}$ based on the fidelities of output states $\left|\left\langle 0\left|U_{m}^{\dagger} U_{m^{\prime}}\right| 0\right\rangle\right|^{2}$. We expect this task to naturally arise when quantum computer providers wish to analyze what their users do on their hardware. Also, we believe this task to be hard in general when we have access only to classical computers, since estimation of $|\langle 0|U| 0\rangle|^{2}$ to an accuracy of $\epsilon$ for a general polynomial-sized quantum circuit $U$ in polynomial time is clearly a BQP-complete task.

To construct an elementary dataset for this task, we use the VQE [22, 23]. It is a method to generate a quantum circuit which outputs an approximate ground state of a target Hamiltonian $H$. This is usually done by using a parameterized quantum circuit, also referred to as an ansatz, $U(\boldsymbol{\theta})$ whose parameter $\boldsymbol{\theta}$ is optimized to minimize the energy expectation value $\langle H(\boldsymbol{\theta})\rangle$ := $\left\langle 0\left|U^{\dagger}(\boldsymbol{\theta}) H U(\boldsymbol{\theta})\right| 0\right\rangle$. The dataset is constructed by optimizing various ansätze to generate ground states of different Hamiltonians $\left\{H_{l} \mid l=0,1, \cdots, L-1\right\}$ which have ground states $\left|g_{l}\right\rangle$ that are mutually almost orthogonal. Labeling each optimized ansatz $U_{m}$ based on the Hamiltonian to which it is optimized, we define a dataset $\left\{\left(U_{m}, l_{m}\right) \mid m=1,2, \cdots, M\right\}$ and $l_{m} \in\{0,1, \cdots, L-1\}$ as a set of pairs of a quantum circuit and its label.

We can expect this dataset to have a nice property that $\left|\left\langle 0\left|U_{m}^{\dagger} U_{m^{\prime}}\right| 0\right\rangle\right|^{2} \approx 1$ when $l_{m}=l_{m^{\prime}}$ and $\left|\left\langle 0\left|U_{m}^{\dagger} U_{m^{\prime}}\right| 0\right\rangle\right|^{2} \ll 1$ otherwise. Note that this property persists even if the optimization is imperfect. Suppose two quantum circuits $U_{1}\left(\boldsymbol{\theta}_{1}\right)$ and $U_{2}\left(\boldsymbol{\theta}_{2}\right)$ are respectively optimized to output non-degenerate ground states $\left|g_{l_{1}}\right\rangle$ and $\left|g_{l_{2}}\right\rangle$ of Hamiltonians $H_{l_{1}}$ and $H_{l_{2}}$. Moreover, let us assume the optimization is imperfect and $\left|\left\langle g_{l_{m}}\left|U_{m}\left(\boldsymbol{\theta}_{m}\right)\right| 0\right\rangle\right|^{2}=\frac{3}{4}$ for $m=1,2$. When $H_{l_{1}}=H_{l_{2}},\left|\left\langle 0\left|U_{1}^{\dagger}\left(\boldsymbol{\theta}_{1}\right) U_{2}\left(\boldsymbol{\theta}_{2}\right)\right| 0\right\rangle\right|^{2} \geq 1 / 4$. On the contrary, assuming $H_{l_{1}} \neq H_{l_{2}}$ and $\left\langle g_{l_{1}} \mid g_{l_{2}}\right\rangle=0$, $\left|\left\langle 0\left|U_{1}^{\dagger}\left(\boldsymbol{\theta}_{1}\right) U_{2}\left(\boldsymbol{\theta}_{2}\right)\right| 0\right\rangle\right|^{2} \leq 1 / 16$. We expect this property makes it easier to extend the dataset by actual experiments using quantum hardware.

## B. Dataset construction details

Table I shows the overview of the dataset that we provide in this work. To define the $L=6$ dataset, we use the Hamiltonians in Table II. $X_{n}, Y_{n}$ and $Z_{n}$ $(n=1,2, \cdots, N)$ are respectively Pauli $X, Y$ and $Z$ operators acting on the $n$th qubit. $a_{i, \sigma}^{\dagger}$ and $a_{i, \sigma}$ is fermion creation and annihilation operators acting on the $i$ th site with the spin $\sigma$. For the Hubbard models, we map the Hamiltonians to qubit ones by Jordan-Wigner transformation; specifically, defining $\tilde{a}_{i}(i=1,2, \ldots, N)$ as $\tilde{a}_{2 j-1}=a_{j, \uparrow}$ and $\tilde{a}_{2 j}=a_{j, \downarrow}$ for $j=1,2, \ldots, N / 2$, we map them as,

$$
\begin{equation*}
\tilde{a}_{i} \rightarrow \frac{1}{2}\left(X_{i}+i Y_{i}\right) Z_{i-1} \cdots Z_{1} \tag{1}
\end{equation*}
$$

TABLE I. Overview of the dataset.

| Numbers of qubits | $N$ | $4,8,12,16,20$ |
| :--- | :--- | :--- |
| Number of labels | $L$ | $6(5$ for $N=4)$ |
| Number of circuits for each | $M / L$ | 300 |
| label <br> Total number of circuits | $M$ | $1800(1500$ for $N=4)$ |

The sites of $2 \mathrm{D}$ Hubbard model are defined on $1 \times 2,2 \times 2$, $3 \times 2,4 \times 2$, and $5 \times 2$ square grids for $N=4,8,12,16$, and 20 respectively, and a $2 \mathrm{D}$ site index $\left(j_{x}, j_{y}\right)$ is mapped to the one-dimensional index $j$ as $j=2\left(j_{x}-1\right)+j_{y}$ for $j_{x}=1,2, \cdots, N / 4$ and $j_{y}=1,2$.

For ansätze, we use the ones listed in Table III. The one which we refer to as hardware-efficient ansatz has been a popular choice for the VQE ansatz to perform proof-of-principle demonstration of ideas [26-30]. The brick-block ansätze where we sequentially apply parameterized two-qubit gates are also a popular choice [31-34]. The last type of ansatz used in this work is Hamiltonian ansatz where we sequentially apply Pauli rotations $R_{\sigma}(\theta)=\exp (-i \theta \sigma / 2)$ for all Pauli operators $\sigma=X, Y, Z$ appearing in the problem Hamiltonian. This ansatz is physically motivated by the adiabatic evolution [35, 36]. For each ansatz, we vary the circuit depth $D$ from 3 to 32 so that we obtain 30 optimized circuits from each ansatz type. For each $D$, we sample initial parameters for optimization from the uniform distribution on $[-2 \pi, 2 \pi)$, except for Hamiltonian ansatz where we sample from $[0,0.1)$, for 10 times, and adopt the one which achieves the lowest energy expectation value after the optimization to the dataset. The optimization of parameters is performed by the BFGS method implemented on SciPy [37]. Using the exact expectation values without statistical noise computed by Qulacs [38], we optimize the parameters until the norm of gradient becomes less than $10^{-5}$ or the number of iterations exceeds 1000 .

## III. DATASET PROPERTIES

## A. Visualization of dataset

First, we visualize the constructed dataset using tstochastic neighbor embedding (t-SNE) [39] to understand the distribution of data intuitively. t-SNE is a visualization method which takes a distance matrix $d_{m, m^{\prime}}$ of a dataset $\left\{\boldsymbol{x}_{m}\right\}$ consisting of high-dimensional vectors as its input, and generate low-dimensional points which maintain the similarities among the data points. Here, we adopt $d_{m, m^{\prime}}=1-\left|\left\langle 0\left|U_{m}^{\dagger} U_{m^{\prime}}\right| 0\right\rangle\right|^{2}$ as the distance matrix of our dataset consisting of quantum circuits $\left\{U_{m}\right\}$. We use the exact values for inner product of $\left\langle 0\left|U_{m}^{\dagger} U_{m^{\prime}}\right| 0\right\rangle$ calculated by Qulacs [38].

The visualization result is shown in Fig. 2 (top panels). Each point corresponds to a circuit $U_{m}$, and it is colored depending on its label. We observe that the $4-, 8-, 12$ - and 16-qubit datasets are well clustered, while a portion of the data from the 20 -qubit dataset appear to be somewhat intermingled. We find it harder to solve Hamiltonians corresponding to 20 qubits than the others for some ansatz employed in this work. We present in the Appendix the fidelity between the output state and the true ground state of the Hamiltonian for each ansatz. Another interesting feature is the existence of multiple clusters in a label. We believe this is an artificial effect caused by the t-SNE visualization. This feature cannot be observed by another visualization technique called multidimensional scaling (MDS) [40], which is shown in the Appendix.

Then, we perform clustering based on the exact value of $d_{i j}$ to show that the clustering of the proposed dataset is indeed easy for ideal quantum computers. We employ the k-medoids algorithm implemented in PyClustering [41]. The performance of the clustering is evaluated by adjusted Rand index (ARI) [42], which takes a value between 0 (random clustering) and 1 (perfect clustering). The ARI is evaluated as the mean of 10 trials of clustering with different random seeds.

The result is visualized in Fig. 2. The ARI of the 4-, 8-, 12-, 16- and 20-qubit dataset is respectively $0.992,0.968$, $0.927,0.883$ and 0.692 . The relatively low ARI for the 20-qubit dataset is caused by the difficulty of producing quantum circuits to output the ground state of label 5 . This result indicates that it is possible to cluster this dataset, even in the unsupervised setting, with an ideal quantum computer.

## B. Clustering using real quantum hardware and noise model simulator

Here, we show our 4-qubit dataset can be reliably learned by using real quantum computers that are presently available. To this end, we perform the clustering by running quantum circuits $U_{m}^{\dagger} U_{m^{\prime}}$ for all pairs of $m$ and $m^{\prime}$ on the ibmq_manila device available at IBMQ to get fidelity between the two output states. The number of measurements is set to $2 \times 10^{4}$ for each $U_{m}^{\dagger} U_{m^{\prime}}$. Only HE and 1D-BB ansätze with $D=3$ to 12 are used in the experiments, and thus the number of data is 10 for each ansatz.

In Fig. 3, we visualize the dataset by t-SNE using the distance matrix obtained from the experiment. As we can observe, we are able to perfectly cluster the dataset. This is because fidelities between the quantum states belonging to the same labels are maintained to be much larger than those with different labels even in a noisy environment. This result shows the actual quantum computers are capable of learning the dataset we propose.

To investigate the possibility of learning our 20 -qubit dataset on actual devices, we also perform clustering by running circuits on FakeAuckland backend available at IBMQ which is a simulator mimicking ibm_auckland device. The result is visualized in Fig. 3 in the same manner as the above experiment. The ARI is 0.720 . This

TABLE II. Hamiltonians used to generate the dataset.

| Label | Name | Hamiltonian |
| :--- | :--- | :--- |
| 0 | 1D transverse-field Ising model | $\sum_{n=1}^{N-1} Z_{n} Z_{n+1}+2 \sum_{n=1}^{N} X_{n}$ |
| 1 | 1D Heisenberg model | $\sum_{n=1}^{N-1}\left(X_{n} X_{n+1}+Y_{n} Y_{n+1}+Z_{n} Z_{n+1}\right)+2 \sum_{n=1}^{N} Z_{n}$ |
| 2 | Su-Schrieffer-Heeger model | $\sum_{n=1}^{N-1}\left(1+\frac{3}{2}(-1)^{n}\right)\left(X_{n} X_{n+1}+Y_{n} Y_{n+1}+Z_{n} Z_{n+1}\right)$ |
| 3 | $J_{1}-J_{2}$ model | $\sum_{n=1}^{N-1}\left[\left(X_{n} X_{n+1}+Y_{n} Y_{n+1}+Z_{n} Z_{n+1}\right)+3\left(X_{n} X_{n+2}+Y_{n} Y_{n+2}+Z_{n} Z_{n+2}\right)\right]$ |
| 4 | 1D Hubbard model | $-\sum_{j=1}^{N / 2-1} \sum_{\sigma \in\{\uparrow, \downarrow\}}\left(a_{j, \sigma}^{\dagger} a_{j+1, \sigma}+\right.$ H.c. $)+\sum_{j=1}^{N / 2}\left(a_{j, \uparrow}^{\dagger} a_{j, \uparrow}-\frac{1}{2}\right)\left(a_{j, \downarrow}^{\dagger} a_{j, \downarrow}-\frac{1}{2}\right)$ |
| 5 | 2D Hubbard model | $-\sum_{\sigma \in\{\uparrow, \downarrow\}}\left(\sum_{j_{x}=1}^{N / 4-1} \sum_{j_{y}=1}^{2} a_{j_{x}, j_{y}, \sigma}^{\dagger} a_{j_{x}+1, j_{y}, \sigma}+\sum_{j_{x}=1}^{N / 4} a_{j_{x}, 1, \sigma}^{\dagger} a_{j_{x}, 2, \sigma}+\right.$ H.c. $)$ |
|  |  | $+\sum_{j_{x}=1}^{N / 4} \sum_{j_{y}=1}^{2}\left(a_{j_{x}, j_{y}, \uparrow}^{\dagger} a_{j_{x}, j_{y}, \uparrow}-\frac{1}{2}\right)\left(a_{j_{x}, j_{y}, \downarrow}^{\dagger} a_{j_{x}, j_{y}, \downarrow}-\frac{1}{2}\right)$ |

TABLE III. Ansätze used to generate the dataset. Angles in every rotation gates are treated as independent parameters of circuit. $R_{\sigma}$ is a rotation gate defined as $R_{\sigma}(\theta)=\exp (-i \theta \sigma / 2)$ for Pauli operators $\sigma \in\{I, X, Y, Z\}^{\otimes N} . V_{n, n^{\prime}}$ is a four-parameter unitary gate defined as $V_{n, n^{\prime}}=\mathrm{C}_{\mathrm{X}_{n, n^{\prime}}} R_{Y_{n}} R_{Y_{n^{\prime}}} \operatorname{CNOT}_{n, n^{\prime}} R_{Y_{n}} R_{Y_{n^{\prime}}}$, where $\operatorname{CNOT}_{n, n^{\prime}}$ is a NOT gate on the qubit $n^{\prime}$ controlled with the qubit $n$, which is taken from [31]. In addition, the following $\mathcal{P}_{\mathrm{S}}$ represent various ordered sets of pairs of sites for the 2-qubit gate. $\mathcal{P}_{\text {chain }}=\{(N-2 j-1, N-2 j) \mid j=1,2, \cdots, N / 2-1\} \cup\{(N-2 j, N-2 j+1) \mid j=1,2, \cdots, N / 2\}$, $\mathcal{P}_{\text {stair }}=\{(N-n, N-n+1) \mid n=1,2, \cdots, N-1\}, \mathcal{P}_{\text {complete }}=\left\{\left(N-n, N-n^{\prime}\right) \mid n^{\prime}=0,1, \cdots, n-1 ; n=1,2, \cdots, N-1\right\}$, $\mathcal{P}_{\text {ladder }}=\{(N-2 j-1, N-2 j+1) \mid j=1,2, \cdots, N / 2-1\} \cup\{(N-2 j, N-2 j+2) \mid j=1,2, \cdots, N / 2-1\} \cup\{(2 j-1,2 j) \mid$ $j=1,2, \cdots, N / 2\}, \mathcal{P}_{\text {cross-ladder }}=\mathcal{P}_{\text {ladder }} \cup\{(N-2 j, N-2 j+1),(N-2 j-1, N-2 j+2) \mid j=1,2, \cdots, N / 2-1\}$, where $N$ is the number of qubits.

Name

Hamiltonian ansatz

Hardware-efficient ansatz (HE)

Complete Hardware-efficient ansatz (Complete-HE)

Ladder Hardware-efficient ansatz (Ladder-HE)

Cross-Ladder Hardware-efficient ansatz (Cross-Ladder-HE)

1D brick-block ansatz (1D-BB)

Stair brick-block ansatz (Stair-BB)

Complete brick-block ansatz (Complete-BB)

Ladder brick-block ansatz (Ladder-BB)

Cross-Ladder brick-block ansatz (Cross-Ladder-BB)

$$
\begin{aligned}
& \text { Ansatz definition } \\
& \prod_{d=1}^{D}\left(\prod_{\sigma \in H} R_{\sigma} \prod_{n=1}^{N} R_{X_{n}} R_{Z_{n}}\right) \\
& \prod_{d=1}^{D}\left[\prod_{n=1}^{N} R_{Z_{n}} R_{Y_{n}}\left[\prod_{p \in \mathcal{P}_{\text {chain }}} \mathrm{CZ}_{p}\right]\right] \prod_{n=1}^{N} R_{Z_{n}} R_{Y_{n}} \\
& \prod_{d=1}^{D}\left[\prod_{n=1}^{N} R_{Z_{n}} R_{Y_{n}}\left[\prod_{p \in \mathcal{P}_{\text {complete }}} \mathrm{CZ}_{p}\right]\right] \prod_{n=1}^{N} R_{Z_{n}} R_{Y_{n}} \\
& \prod_{d=1}^{D}\left[\prod_{n=1}^{N} R_{Z_{n}} R_{Y_{n}}\left[\prod_{p \in \mathcal{P}_{\text {ladder }}} \mathrm{CZ}_{p}\right]\right] \prod_{n=1}^{N} R_{Z_{n}} R_{Y_{n}} \\
& \prod_{d=1}^{D}\left[\prod_{n=1}^{N} R_{Z_{n}} R_{Y_{n}}\left[\prod_{p \in \mathcal{P}_{\text {cross-ladder }}} \mathrm{CZ}_{p}\right]\right] \prod_{n=1}^{N} R_{Z_{n}} R_{Y_{n}} \\
& \prod_{n=1}^{N} R_{Y_{n}} \prod_{d=1}^{D}\left[\prod_{p \in \mathcal{P}_{\text {chain }}} \mathrm{V}_{p}\right] \\
& \prod_{n=1}^{N} R_{Y_{n}} \prod_{d=1}^{D}\left\lfloor\prod_{p \in \mathcal{P}_{\text {stair }}} \mathrm{V}_{p}\right] \\
& \prod_{n=1}^{N} R_{Y_{n}} \prod_{d=1}^{D}\left\lceil\prod_{p \in \mathcal{P}_{\text {complete }}} \mathrm{V}_{p}\right] \\
& \prod_{n=1}^{N} R_{Y_{n}} \prod_{d=1}^{D}\left[\prod_{p \in \mathcal{P}_{\text {ladder }}} \mathrm{V}_{p}\right]
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2024_06_04_12e0b57a5e5701d5b765g-4.jpg?height=74&width=526&top_left_y=1611&top_left_x=1084)

indicates the possibility of learning our dataset using real devices even at the size of 20 qubits.

## C. Classical machine learning applied to the dataset

Our dataset is expected to be hard to be learned solely by classical computers. The most challenging part in applying classical machine learning algorithms is to find a good feature map to transform the description of a quantum circuit to e.g., a real-valued vector, which can then be processed by various techniques such as neural networks. After some trials, we find the task to be very non-trivial. Straight-forward applications of graph neural networks for the classification task of the proposed dataset, in an analogous way to Ref. [43], give us labels that are basically equivalent to random guessing.
To make the problem easier, we consider the task of classifying the dataset consisting of the same type of ansätze in a supervised setting. More concretely, we investigate the feasibility of classifying quantum circuits using the parameter vector $\boldsymbol{\theta}$ of the circuits as their features.

For this task, we employ kernel support vector machine (SVM) model [44]. To remove the difference in the lengths of the parameter vectors $\boldsymbol{\theta}$ depending on the different depths of the circuits, we extend $\boldsymbol{\theta}$ to match the deepest circuits within each ansatz type and fill the extended elements with zero. We split $80 \%$ of all data into training data and the rest into test data to ensure that the divided data has roughly the same proportions of different labels. We train the model 10 times with different splitting and report the mean accuracy score. The regularization strength, types of kernel, and the parameters in kernel are treated as hyperparameters and
![](https://cdn.mathpix.com/cropped/2024_06_04_12e0b57a5e5701d5b765g-5.jpg?height=878&width=1800&top_left_y=176&top_left_x=173)

FIG. 2. Visualization and clustering result of the dataset. (Top panels) Visualization of the dataset using t-SNE. The points are colored depending on their true labels. (Bottom panels) Clustering result by k-medoids algorithm. The points are colored based on the clusters. Both plots are generated using exact values of distance matrix $d_{m, m^{\prime}}$. Different markers correspond to different ansatzes.

optimized through the grid search technique combined with Stratified three-fold cross validation. We note that other classification methods such as random forest have given us similar results.

Figure 4 shows the classification accuracy. For Hamiltonian ansatz, we reached a higher classification accuracy of up to about $80 \%$ than other ansätze for all $n$. Such a high accuracy may be caused by the difference of quantum circuit structures from one label to another, which only exist in the case of the Hamiltonian ansatz. Except for the Hamiltonian ansatz, we only reached a classification accuracy of up to about $20 \%$ accuracy. This means that labels are predicted almost randomly. Although there may be a possible improvement to be made to this result by using more sophisticated methods such as neural networks, we can at least say "standard" classification models such as SVM do not work well on the proposed dataset even for the very simplified task; quantum computers are capable of solving the same task in an unsupervised setting with different ansätze.

## IV. CONCLUSION

In this paper, we proposed a quantum circuits classification problem as a more practical machine learning task. We introduced the dataset of $N=4,8,12,16,20$ qubit quantum circuits optimized by VQE for different Hamiltonians using different types of ansätze. We verified that the unsupervised clustering of the dataset is easy for ideal quantum computers. In 4-, 8-, 12-, and 16-qubit cases, we achieved the ARI score of over 0.88 , and in 20 -qubit case, we achieved 0.69. In particular, we demonstrate a successful classification of our 4-qubit subdataset using the actual 4 -qubit device.

Possible future directions are the followings. First, the capability to create the dataset by VQE implies that actual quantum devices can produce a dataset with even larger numbers of qubits. It is also interesting to explore whether other variational quantum algorithms or FTQC algorithms can also construct a dataset similar to the one provided in this work. Our dataset, which provides a set of practical quantum circuits optimized to output ground states, can also be useful as a benchmark for quantum circuit compilers/transpilers. The devices may be able to conduct similar experiments on our dataset of more numbers of qubits. We finally note that it is worth investigating whether state-of-the-art classical machine learning algorithms can solve the dataset. We published the dataset on GitHub [24].

![](https://cdn.mathpix.com/cropped/2024_06_04_12e0b57a5e5701d5b765g-6.jpg?height=943&width=889&top_left_y=179&top_left_x=173)

Actual 4-qubit device Fake 27-qubit device
![](https://cdn.mathpix.com/cropped/2024_06_04_12e0b57a5e5701d5b765g-6.jpg?height=878&width=852&top_left_y=232&top_left_x=192)

FIG. 3. Results of clustering quantum circuit in the dataset using IBM's 4-qubit quantum computer and 27-qubit fake device. (left panel) The result of clustering on the 4-qubit dataset by IBM's 4-qubit quantum computer. (right panel) The result of clustering on the 20 -qubit dataset by noise model simulation using real noise data of IBM's 27-qubit quantum computer. (Top panel) Visualization of the dataset using t-SNE. The points are colored depending on their true labels. (Bottom panel) Clustering result by k-medoids algorithm.

![](https://cdn.mathpix.com/cropped/2024_06_04_12e0b57a5e5701d5b765g-6.jpg?height=612&width=840&top_left_y=190&top_left_x=1103)

FIG. 4. Classification result using support vector machine on the circuit parameters.

## ACKNOWLEDGMENTS

K.M. is supported by JST PRESTO Grant No. JPMJPR2019 and JSPS KAKENHI Grant No. 20K22330. K.F. is supported by JST ERATO Grant No. JPMJER1601 and JST CREST Grant No. JPMJCR1673. This work is supported by MEXT Quantum Leap Flagship Program (MEXTQLEAP) Grant No. JPMXS0118067394 and JPMXS0120319794. We also acknowledge support from JSTCOI-NEXT program.

## Appendix: supplementary data analysis

Here we present additional details for the main results in our manuscript. In Fig. 5, we present the visualizations of the dataset by MDS [40]. Figure 6 shows the fidelity between each data and ground state of corresponding Hamiltonian by violin plots to understand how close the output of each circuit is to the ground states.
[1] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, Quantum machine learning, Nature 549, 195 (2017).

[2] M. Cerezo, G. Verdon, H.-Y. Huang, L. Cincio, and P. J. Coles, Challenges and opportunities in quantum machine learning, Nature Computational Science 2, 567 (2022).

[3] E. Farhi and H. Neven, Classification with quantum neural networks on near term processors, arXiv:1802.06002v2 [quant-ph] (2018).

[4] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, Quantum circuit learning, Physical Review A 98, 032309 (2018).

[5] M. Schuld and N. Killoran, Quantum machine learning in feature hilbert spaces, Phys. Rev. Lett. 122, 040504 (2019).
[6] J.-G. Liu and L. Wang, Differentiable learning of quantum circuit born machines, Phys. Rev. A 98, 062324 (2018).

[7] M. Benedetti, D. Garcia-Pintos, O. Perdomo, V. LeytonOrtega, Y. Nam, and A. Perdomo-Ortiz, A generative modeling approach for benchmarking and training shallow quantum circuits, npj Quantum Information 5, 45 (2019).

[8] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, Supervised learning with quantum-enhanced feature spaces, Nature 567, 209 (2019).

[9] T. Kusumoto, K. Mitarai, K. Fujii, M. Kitagawa, and M. Negoro, Experimental quantum kernel trick with nuclear spins in a solid, npj Quantum Information 7, 94 (2021).
![](https://cdn.mathpix.com/cropped/2024_06_04_12e0b57a5e5701d5b765g-7.jpg?height=1244&width=1800&top_left_y=175&top_left_x=173)

FIG. 5. Visualization and clustering result of the dataset (Top panel) Visualization of the dataset using MDS. The points are colored depending on their true labels. (Middle panel) Clustering result by k-medoids algorithm. (Bottom panel) The same visualization but are colored depending on their fidelity with the ground state of each label's Hamiltonian.

[10] K. Bartkiewicz, C. Gneiting, A. Cernoch, K. Jiráková, K. Lemr, and F. Nori, Experimental kernel-based quantum machine learning in finite feature space, Scientific Reports 10, 1 (2020).

[11] M. S. Rudolph, N. B. Toussaint, A. Katabarwa, S. Johri, B. Peropadre, and A. Perdomo-Ortiz, Generation of highresolution handwritten digits with an ion-trap quantum computer, Phys. Rev. X 12, 031010 (2022).

[12] Y. Liu, S. Arunachalam, and K. Temme, A rigorous and robust quantum speed-up in supervised machine learning, Nature Physics 17, 1013 (2021).

[13] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, Power of data in quantum machine learning, Nature Communications 12, 2631 (2021).

[14] V. Dunjko, Y.-K. Liu, X. Wu, and J. M. Taylor, Exponential improvements for quantum-accessible reinforcement learning, arXiv:1710.11160v3 [quant-ph] (2017).

[15] S. Jerbi, C. Gyurik, S. Marshall, H. Briegel, and V. Dunjko, Parametrized quantum policies for reinforcement learning, in Advances in Neural Information Processing Systems, Vol. 34, edited by M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan (Curran As- sociates, Inc., 2021) pp. 28362-28375.

[16] N. Pirnay, R. Sweke, J. Eisert, and J.-P. Seifert, A super-polynomial quantum-classical separation for density modelling, arXiv:2210.14936v1 [quant-ph] (2022).

[17] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, Gradient-based learning applied to document recognition, Proceedings of the IEEE 86, 2278 (1998).

[18] H. Xiao, K. Rasul, and R. Vollgraf, Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, arXiv:1708.07747v2 [cs.LG] (2017).

[19] H.-Y. Huang, M. Broughton, J. Cotler, S. Chen, J. Li, M. Mohseni, H. Neven, R. Babbush, R. Kueng, J. Preskill, and J. R. McClean, Quantum advantage in learning from experiments, Science 376, 1182 (2022).

[20] L. Schatzki, A. Arrasmith, P. J. Coles, and M. Cerezo, Entangled datasets for quantum machine learning, arXiv:2109.03400v2 [quant-ph] (2021).

[21] E. Perrier, A. Youssry, and C. Ferrie, Qdataset, quantum datasets for machine learning, Scientific Data 9, 582 (2022).

[22] A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru-Guzik, and J. L. O'Brien, A variational eigenvalue solver on a photonic quantum
![](https://cdn.mathpix.com/cropped/2024_06_04_12e0b57a5e5701d5b765g-8.jpg?height=1432&width=1786&top_left_y=178&top_left_x=172)

FIG. 6. Fidelity between each data and the ground state of each label's Hamiltonian

processor, Nature Communications 5, 4213 (2014).

[23] J. Tilly, H. Chen, S. Cao, D. Picozzi, K. Setia, Y. Li, E. Grant, L. Wossnig, I. Rungger, G. H. Booth, and J. Tennyson, The variational quantum eigensolver: A review of methods and best practices, Physics Reports 986, 1 (2022), the Variational Quantum Eigensolver: a review of methods and best practices.

[24] N. Akimoto, M. Kosuke, P. Leonardo, S. Takanori, and F. Keisuke, VQE-generated quantum circuit dataset (2023).

[25] A. Cross, A. Javadi-Abhari, T. Alexander, N. De Beaudrap, L. S. Bishop, S. Heidel, C. A. Ryan, P. Sivarajah, J. Smolin, J. M. Gambetta, and B. R. Johnson, OpenQASM 3: A Broader and Deeper Quantum Assembly Language, ACM Transactions on Quantum Computing 3, 10.1145/3505636 (2022).

[26] A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow, and J. M. Gambetta, Hardwareefficient variational quantum eigensolver for small molecules and quantum magnets, Nature 549, 242
(2017).

[27] K. M. Nakanishi, K. Mitarai, and K. Fujii, Subspacesearch variational quantum eigensolver for excited states, Phys. Rev. Res. 1, 033062 (2019).

[28] J. M. Kübler, A. Arrasmith, L. Cincio, and P. J. Coles, An Adaptive Optimizer for Measurement-Frugal Variational Algorithms, Quantum 4, 263 (2020).

[29] K. M. Nakanishi, K. Fujii, and S. Todo, Sequential minimal optimization for quantum-classical hybrid algorithms, Phys. Rev. Res. 2, 043158 (2020).

[30] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, Barren plateaus in quantum neural network training landscapes, Nature communications $\mathbf{9}$, 4812 (2018).

[31] R. M. Parrish, E. G. Hohenstein, P. L. McMahon, and T. J. Martínez, Quantum computation of electronic transitions using a variational quantum eigensolver, Phys. Rev. Lett. 122, 230401 (2019).

[32] L. Slattery, B. Villalonga, and B. K. Clark, Unitary block optimization for variational quantum algorithms, Phys.

Rev. Res. 4, 023072 (2022).

[33] J. Dborin, F. Barratt, V. Wimalaweera, L. Wright, and A. G. Green, Matrix product state pre-training for quantum machine learning, Quantum Science and Technology 7,035014 (2022).

[34] M. S. Rudolph, J. Miller, J. Chen, A. Acharya, and A. Perdomo-Ortiz, Synergy between quantum circuits and tensor networks: Short-cutting the race to practical quantum advantage, arXiv:2208.13673v1 [quant-ph] (2022).

[35] D. Wecker, M. B. Hastings, and M. Troyer, Progress towards practical quantum variational algorithms, Phys. Rev. A 92, 042303 (2015).

[36] R. Wiersema, C. Zhou, Y. de Sereville, J. F. Carrasquilla, Y. B. Kim, and H. Yuen, Exploring entanglement and optimization within the hamiltonian variational ansatz, PRX Quantum 1, 020319 (2020).

[37] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, İ. Polat, Y. Feng, E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. van Mulbregt, and SciPy 1.0 Contributors,
SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python, Nature Methods 17, 261 (2020).

[38] Y. Suzuki, Y. Kawase, Y. Masumura, Y. Hiraga, M. Nakadai, J. Chen, K. M. Nakanishi, K. Mitarai, R. Imai, S. Tamiya, T. Yamamoto, T. Yan, T. Kawakubo, Y. O. Nakagawa, Y. Ibe, Y. Zhang, H. Yamashita, H. Yoshimura, A. Hayashi, and K. Fujii, Qulacs: a fast and versatile quantum circuit simulator for research purpose, Quantum 5, 559 (2021).

[39] L. van der Maaten and G. Hinton, Visualizing data using t-SNE, Journal of Machine Learning Research 9, 2579 (2008).

[40] J. Kruskal, Nonmetric multidimensional scaling: A numerical method, Psychometrika 29, 115-129 (1964).

[41] A. Novikov, PyClustering: Data Mining Library, Journal of Open Source Software 4, 1230 (2019).

[42] L. Hubert and P. Arabie, Comparing partitions, Journal of classification 2, 193 (1985).

[43] H. Wang, P. Liu, J. Cheng, Z. Liang, J. Gu, Z. Li, Y. Ding, W. Jiang, Y. Shi, X. Qian, D. Z. Pan, F. T. Chong, and S. Han, Quest: Graph transformer for quantum circuit reliability estimation, arXiv:2210.16724v1 [quant-ph] (2022).

[44] J. Platt, Probabilistic outputs for support vector machines and comparison to regularized likelihood methods, in Advances in Large Margin Classifiers (2000).


[^0]:    * mitarai.kosuke.es@osaka-u.ac.jp

    † fujii@qc.ee.es.osaka-u.ac.jp

</end of paper 0>


<paper 1>
# SantaQlaus: A resource-efficient method to leverage quantum shot-noise for optimization of variational quantum algorithms 

Kosuke Ito $^{1 *}$ and Keisuke Fujii ${ }^{1,2,3 \dagger}$<br>${ }^{1}$ Center for Quantum Information and Quantum Biology,<br>International Advanced Research Institute, Osaka University, Osaka 560-8531, Japan<br>${ }^{2}$ Graduate School of Engineering Science, Osaka University,<br>1-3 Machikaneyama, Toyonaka, Osaka 560-8531, Japan<br>${ }^{3}$ RIKEN Center for Quantum Computing (RQC), Hirosawa 2-1, Wako, Saitama 351-0198, Japan


#### Abstract

We introduce SantaQlaus, a resource-efficient optimization algorithm tailored for variational quantum algorithms (VQAs), including applications in the variational quantum eigensolver (VQE) and quantum machine learning (QML). Classical optimization strategies for VQAs are often hindered by the complex landscapes of local minima and saddle points. Although some existing quantum-aware optimizers adaptively adjust the number of measurement shots, their primary focus is on maximizing gain per iteration rather than strategically utilizing quantum shot-noise (QSN) to address these challenges. Inspired by the classical Stochastic AnNealing Thermostats with Adaptive momentum (Santa) algorithm, SantaQlaus explicitly leverages inherent QSN for optimization. The algorithm dynamically adjusts the number of quantum measurement shots in an annealing framework: fewer shots are allocated during the early, high-temperature stages for efficient resource utilization and landscape exploration, while more shots are employed later for enhanced precision. Numerical simulations on VQE and QML demonstrate that SantaQlaus outperforms existing optimizers, particularly in mitigating the risks of converging to poor local optima, all while maintaining shot efficiency. This paves the way for efficient and robust training of quantum variational models.


## I. INTRODUCTION

Variational quantum algorithms (VQAs) are promising methods as potential applications for noisy intermediatescale quantum (NISQ) devices [1, 2]. In VQAs, the loss function is calculated using quantum circuits, and variational parameters of the circuit are optimized via classical computing. Various VQA paradigms have been introduced [2], including the variational quantum eigensolver (VQE) for ground-state energy approximation [3-6] and others such as QAOA [7-9] and quantum machine learning (QML) [10-17]. In QML, quantum neural networks (QNNs) are expected to offer advantages by operating in a feature space that is classically infeasible. While variational (explicit) QNNs can be integrated into quantum kernel methods (implicit model) [18], their distinct importance is underscored by suggestions that they may excel in generalization performance over the kernel methods [19]. When making predictions, quantum kernel models require evaluation of the kernel between new input data and all training data. Variational QNNs simply need an evaluation of the input data. Notably, in certain QNN classes, this evaluation can be performed using a classical surrogate [20]. Consequently, the efficient training of such variational models gains significant importance.

Classical optimization in VQAs faces multiple challenges that influence the efficiency and reliability of the optimization process. Barren plateau phenomena, characterized by exponentially vanishing gradients, hinder[^0]

trainability [21-28]. Likewise, the landscape of local minima and saddle points introduces additional roadblocks for optimization $[29,30]$. Although methodologies to mitigate barren plateaus are a topic of ongoing investigation, specific choices of ansatz and cost functions can help in this regard $[31-34]$. A core aspect of our research focuses on tackling the issues associated with saddle points and suboptimal local minima as well as efficient resource usage.

Generic classical optimization algorithms, such as Adam [35], Nelder-Mead [36], Powell [37], and SPSA [38] have been widely used in VQAs $[6,13,39-41]$. In addition to these, quantum-aware optimization schemes have been introduced [40, 42-56]. Interestingly, earlier works has noted that low levels of various types of stochastic noise can positively affect the optimization process in VQAs [34, 47, 57-59]. Especially, in VQAs, the evaluation of expectation values inherently includes statistical noise from quantum measurements, which is called quantum shot-noise (QSN). Theoretical analysis supports the idea that QSN can assist in escaping the aforementioned optimization traps [60]. Nevertheless, current optimization algorithms tend to capitalize on positive effects of QSN only implicitly.

In addition, effective resource allocation during optimization is critical for the practical application of VQAs. The emphasis should be on effectively utilizing classical data derived from quantum measurements, instead of merely aiming for precise expectation values [40, 47, 52, 61]. Existing techniques adjust the number of measurement shots to balance resource use, but often without sufficient regard for the risk of encountering saddle points or converging to poor local optima $[40,42,50,52,62,63]$.

Motivated by these observations, we address the following question: Can inherent stochasticity of quantum measurements be strategically leveraged in the optimization process of VQAs in a resource efficient way? In this paper, to explore this avenue, we propose an optimization algorithm Stochastic AnNealing Thermostats with Adaptive momentum and Quantum-noise Leveraging by Adjusted Use of Shots (SantaQlaus). SantaQlaus is inspired by a classical optimizer called Stochastic AnNealing Thermostats with Adaptive momentum (Santa) [64]. Santa employs simulated Langevin diffusion, guided by an annealing thermostat utilizing injected thermal noise, to approach global optimality. A key advantage of Santa is its robustness against noise variations, which aligns well with our objectives.

Our main contribution is an extension of the classical Santa optimizer by integrating the leveraging of QSN, resulting in the SantaQlaus algorithm for the optimization of VQAs. Our proposal seeks to replace the thermal noise in Santa with inherent QSN. We design SantaQlaus to adaptively adjust the number of shots to ensure the variance of the QSN aligns with the thermal noise utilized in Santa, thus enhancing the efficiency of loss function optimization through annealed thermostats. Specifically, during the annealing process, fewer shots are required in the early, high-noise stages, while more shots are allocated to the later, low-noise stages, thus demanding more accurate evaluations. This strategy ensures efficient use of resources without sacrificing accuracy.

Our algorithm is applicable to a wide range of VQAs, especially when the gradient estimator for the loss function shows asymptotic normality. Our method encompasses a general QML framework, accommodating both linear and non-linear dependencies in loss functions, such as mean squared error (MSE), offering flexibility in selecting suitable loss functions for various QML tasks. Additionally, it is compatible with data-independent VQAs like VQE and QAOA. Indeed, we show the asymptotic normality and its explicit form of mini-batch gradient estimators for linear and quadratic loss functions, extendable to general polynomial loss functions. From this analysis, we can compute the appropriate number of shots used in our algorithm.

Through numerical simulations on VQE and QML tasks, we demonstrate the superiority of SantaQlaus over established optimizers like Adam and gCANS, showcasing its efficiency in reducing the number of shots required and improving accuracy.

The remainder of this paper is structured as follows. Sec. II presents a framework for general QML loss functions in VQAs and the evaluation of the associated gradient estimators. In Sec. III, we provide a comprehensive review of the classical Santa algorithm, which lays the groundwork for the method we introduce. Sec. IV introduces the SantaQlaus algorithm, the principal contribution of this study, beginning with the establishment of the concept of asymptotic normality for the gradient estimators. Sec. V details a series of targeted numerical simulations to assess the performance of the SantaQlaus algorithm, offering empirical evidence to validate its efficacy. The paper concludes with Section VI, summarizing key findings and suggesting avenues for future research.

## II. FRAMEWORK

## A. Loss functions for VQAs

Many VQAs are formulated to achieve a goal by minimizing a loss function which is computed by a quantum device with classical parameters to be optimized classically. In this section, we present a general framework for the loss functions in such VQAs mainly focusing on QML, in a similar manner to Refs. [10, 62]. We remark that a wide variety of VQAs can be treated in the framework of QML, including VQE and QAOA just by disregarding the data dependency. The loss functions we explore are formulated to embrace the wide-ranging nature of QML tasks, explicitly accommodating both linear and non-linear dependencies on the expectation values. As non-linear loss functions are prevalent in machine learning and the choice of loss functions critically influence task performance, it is imperative to ensure the generality of our framework to encompass such functions.

In QML, we have a set of quantum states $\mathcal{S}=$ $\left\{\rho\left(\boldsymbol{x}_{1}\right), \cdots, \rho\left(\boldsymbol{x}_{N}\right)\right\}$ used for training specified by a set of input data $\mathcal{D}=\left\{\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{N}\right\}$. Both QC and QQ categories of QML can be put into this formulation, where QC (QQ) aims to learn a classical (quantum) dataset utilizing a quantum algorithm [65-67]. For QC category, the target dataset is the classical data $\mathcal{D}$, and $\mathcal{S}$ is a set of the data-encoded quantum states. Hence, the choice of the data-encoding feature map $\boldsymbol{x} \mapsto \rho(\boldsymbol{x})$ from input data $\boldsymbol{x}$ into quantum feature $\rho(\boldsymbol{x})$ is important, though it is not discussed in detail in this paper. For QQ purpose, $\boldsymbol{x}_{i}$ is regarded as a specification of each target data quantum state, such as classical descriptions of quantum circuits which generate the states [68].

In variational $\mathrm{QML}$, a quantum model is given by a parameterized quantum channel $\mathcal{M}_{\boldsymbol{\theta}}$, and an observable $H$ to be measured. In general, we can consider data dependence of the channel as $\rho(\boldsymbol{x}, \boldsymbol{\theta})=\mathcal{M}_{\boldsymbol{\theta}}[\boldsymbol{x}](\rho(\boldsymbol{x}))$, as in data re-uploading models [69-71]. An observable is given by a hermitian matrix which can be decomposed into directly measured observables $h_{j}(\boldsymbol{x})$ as

$$
\begin{equation*}
H(\boldsymbol{x}, \boldsymbol{w})=\sum_{j=1}^{J} w_{j} c_{j}(\boldsymbol{x}) h_{j}(\boldsymbol{x}) \tag{1}
\end{equation*}
$$

where the observable can be dependent on data $\boldsymbol{x}$ and weight vector $\boldsymbol{w}$ which is also optimized in general. Then, the loss function to be minimized for a given task is given as a $p_{i}$ weighted average

$$
\begin{equation*}
L(\boldsymbol{\theta}, \boldsymbol{w})=\sum_{i=1}^{N} p_{i} \ell\left(\boldsymbol{x}_{i}, E\left(\boldsymbol{x}_{i}, \boldsymbol{\theta}, \boldsymbol{w}\right)\right) \tag{2}
\end{equation*}
$$

where $\ell$ is a generic function of the data and the expectation value

$$
\begin{align*}
E(\boldsymbol{x}, \boldsymbol{\theta}, \boldsymbol{w}) & :=\operatorname{Tr}\left[\mathcal{M}_{\boldsymbol{\theta}}[\boldsymbol{x}](\rho(\boldsymbol{x})) H(\boldsymbol{x}, \boldsymbol{w})\right] \\
& =\sum_{j=1}^{J} w_{j} c_{j}(\boldsymbol{x}) \operatorname{Tr}\left[\mathcal{M}_{\boldsymbol{\theta}}[\boldsymbol{x}](\rho(\boldsymbol{x})) h_{j}(\boldsymbol{x})\right] \\
& =: \sum_{j=1}^{J} w_{j} c_{j}(\boldsymbol{x})\left\langle h_{j}(\boldsymbol{x})\right\rangle_{\boldsymbol{x}, \boldsymbol{\theta}} \tag{3}
\end{align*}
$$

Additionally, some regularization term $\lambda f(\boldsymbol{\theta}, \boldsymbol{w})$ with a hyperparameter $\lambda$ may be added to the loss function to enhance the generalization performance. Here, the dependence of the observable on data $\boldsymbol{x}$ is considered for the sake of generality. For example, such a loss function appears in a task to make the output pure state $\rho(\boldsymbol{x}, \boldsymbol{\theta})=|\phi(\boldsymbol{x}, \boldsymbol{\theta})\rangle\langle\phi(\boldsymbol{x}, \boldsymbol{\theta})|$ of the model close to the correct output state $\rho_{x}^{\text {out }}$, where the loss is given by the average fidelity $\sum_{i=1}^{N}\left\langle\phi\left(\boldsymbol{x}_{i}, \boldsymbol{\theta}\right)\left|\rho_{\boldsymbol{x}_{i}}^{\text {out }}\right| \phi\left(\boldsymbol{x}_{i}, \boldsymbol{\theta}\right)\right\rangle / N[72,73]$. This is the case where $\ell(\boldsymbol{x}, E)=E, H(\boldsymbol{x})=\rho_{\boldsymbol{x}}^{\text {out }}$ and $p_{i}=1 / N$ without $\boldsymbol{w}$. Variational quantum error correction (VQEC) $[74,75]$ is another example, where the loss function is given by the fidelity between the error corrected (possibly mixed) output state and the ideal pure state.

This framework is applicable to both supervised and unsupervised learning. In the context of supervised machine learning, we associate each input data point $\boldsymbol{x}_{i}$ with a label $y_{i}=y\left(\boldsymbol{x}_{i}\right)$. It should be noted that the functions $H(\boldsymbol{x}, \boldsymbol{w})$ and $\ell(\boldsymbol{x}, E)$ implicitly depend on these labels, as they are functions of the data points $\boldsymbol{x}$ which include label information via $y(\boldsymbol{x})$. This label-dependency will be considered in the definitions and usage of these functions throughout our discussion. A wide range of the loss functions of VQAs are covered by this form. A few concrete examples are found in the tasks we employ for numerical simulations in Sec. V. A comprehensive review of various loss functions in QML can be found in Ref. [62]. For VQAs not involving input data, such as VQE and QAOA, the framework applies by regarding just a single input data to specify the input state being considered [10]. In VQE, $\boldsymbol{w}$ is fixed and not optimized.

The gradient of the loss function reads

$$
\begin{align*}
\frac{\partial L}{\partial \theta_{j}} & =\sum_{i=1}^{N} p_{i} \frac{\partial \ell}{\partial E} \frac{\partial E}{\partial \theta_{j}}  \tag{4}\\
\frac{\partial L}{\partial w_{j}} & =\sum_{i=1}^{N} p_{i} \frac{\partial \ell}{\partial E} \frac{\partial E}{\partial w_{j}} \tag{5}
\end{align*}
$$

by the chain rule. The derivative with respect to a weight parameter $\frac{\partial E}{\partial w_{j}}$ is computed as

$$
\begin{equation*}
\frac{\partial E}{\partial w_{j}}=c_{j}(\boldsymbol{x})\left\langle h_{j}(\boldsymbol{x})\right\rangle_{\boldsymbol{x}, \boldsymbol{\theta}} \tag{6}
\end{equation*}
$$

On the other hand, the derivative $\frac{\partial E}{\partial \theta_{j}}$ is nontrivial in general. Because the expectation value is estimated from finite samples of the measurement outcomes, the simple numerical differentiation becomes inaccurate due to the statistical errors [76]. Instead, in our model, we assume that this derivative can be computed by an analytic form via a parameter-shift rule $[10,77,78]$

$$
\begin{equation*}
\frac{\partial E}{\partial \theta_{j}}=\sum_{k=1}^{R_{j}} a_{k} E\left(\boldsymbol{x}, \boldsymbol{\theta}+\epsilon_{j, k} \boldsymbol{e}_{j}, \boldsymbol{w}\right) \tag{7}
\end{equation*}
$$

where $\boldsymbol{e}_{j}$ is the unit vector in the $j$-th component direction, $a_{k}, \epsilon_{j, k}$ and $R_{i}$ are constants determined by the model. In fact, a parameter-shift rule (PSR) holds for a wide range of the model $\mathcal{M}_{\boldsymbol{\theta}}$ given by parametric unitary gates. Especially, if the parameter is given by the gate $U_{j}\left(\theta_{j}\right)=\exp \left[-i \theta_{j} A_{j} / 2\right]$ with $A_{j}^{2}=I$, we have $[10,78]$

$$
\begin{equation*}
\frac{\partial E}{\partial \theta_{j}}=\frac{E\left(\boldsymbol{x}, \boldsymbol{\theta}+\frac{\pi}{2} e_{j}, \boldsymbol{w}\right)-E\left(\boldsymbol{x}, \boldsymbol{\theta}-\frac{\pi}{2} \boldsymbol{e}_{j}, \boldsymbol{w}\right)}{2} \tag{8}
\end{equation*}
$$

In this paper, we assume that two-point PSR (8) holds for all the partial derivatives with respect to $\theta_{j}$ in our model.

In the following, unless necessary, we omit $\boldsymbol{w}$ for brevity. We call the simplest kind of a loss function with $\ell(\boldsymbol{x}, E)=E$ a linear loss function. Linear loss functions are used in various QML tasks such as quantum auto encoder $[31,79]$ and VQEC $[74,75]$, as well as the energy expectation value used in VQE and QAOA. Non-linear loss functions are also common in QML, such as MSE and the cross entropy (CE) loss functions. Especially, polynomial loss functions given below are amenable in QML due to the tractability of constructing unbiased estimator:

$$
\begin{equation*}
\ell(\boldsymbol{x}, E)=\sum_{n=0}^{D} a_{n}(\boldsymbol{x}) E^{n} \tag{9}
\end{equation*}
$$

where $D$ is the degree of the polynomial. For MSE given the label $y(\boldsymbol{x})$ for the data $\boldsymbol{x}$, the function $\ell_{\text {MSE }}$ is given as $\ell_{\text {MSE }}(\boldsymbol{x}, E(\boldsymbol{x}, \boldsymbol{\theta}))=(y(\boldsymbol{x})-\tilde{y}(E(\boldsymbol{x}, \boldsymbol{\theta})))^{2}$, where $\tilde{y}(E(\boldsymbol{x}, \boldsymbol{\theta}))$ is a prediction by the model. Typically, the prediction is given by the expectation value of an observable itself $\tilde{y}(E(\boldsymbol{x}, \boldsymbol{\theta}))=E(\boldsymbol{x}, \boldsymbol{\theta})$. In this case, MSE is a kind of the polynomial loss function with $a_{0}(\boldsymbol{x})=y(\boldsymbol{x})^{2}$, $a_{1}(\boldsymbol{x})=-2 y(\boldsymbol{x}), a_{2}(\boldsymbol{x})=1$, and $D=2$. More general polynomial loss functions are actually used in classical machine learning [80]. Ref. [80] introduces Taylor-CE, a truncated Taylor series expansion of the CE loss, with the truncation degree serving as a hyperparameter. Notably, Taylor-CE has been demonstrated to outperform its counterparts in various multiclass classification tasks with label noise, provided that the truncation degree is selected appropriately.

The gradient of a polynomial loss function is given as

$$
\begin{aligned}
& \frac{\partial \ell}{\partial E}(\boldsymbol{x}, E(\boldsymbol{x}, \boldsymbol{\theta})) \\
= & \sum_{n=1}^{D} n a_{n}(\boldsymbol{x}) E(\boldsymbol{x}, \boldsymbol{\theta})^{n-1}
\end{aligned}
$$

$$
\begin{align*}
& =\sum_{n=1}^{D} n a_{n}(\boldsymbol{x})\left(\sum_{j} w_{j} c_{j}(\boldsymbol{x})\left\langle h_{j}(\boldsymbol{x})\right\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}\right)^{n-1} \\
& =\sum_{n=1}^{D} n a_{n}(\boldsymbol{x}) \\
& \sum_{b_{1}+b_{2}+\cdots+b_{J}=n-1}\binom{n-1}{\boldsymbol{b}} \prod_{j}\left(w_{j} c_{j}(\boldsymbol{x})\left\langle h_{j}(\boldsymbol{x})\right\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}\right)^{b_{j}} \tag{10}
\end{align*}
$$

where $\binom{n-1}{b}:=\frac{(n-1)!}{b_{1}!b_{2}!\cdots b_{J}!}$. Thus, computing the derivative with respect to $\theta_{j}\left(w_{j}\right)$ for $D \geq 3(D \geq 2)$ needs an estimate of $\left\langle h_{j}(\boldsymbol{x})\right\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}^{n}$ with $n \geq 2$. For MSE, the derivative with respect to the weight $w_{j}$ needs an estimate of the squared expectation value.

## B. Shot allocation

We must decide how to allocate the number of shots to use for the estimation of multiple terms $\left\langle h_{j}(\boldsymbol{x})\right\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}$, taking into account which terms can be measured simultaneously. Several strategies have been proposed for the efficient measurement of multiple observables [4, 6, 81-89]. One simple way is to allocate the shots proportionally to the weight of each simultaneously measurable group deterministically or randomly [42]. The weighted random sampling (WRS) achieves an unbiased estimates of the whole loss function even with few number of total shots [42], while weighted deterministic sampling (WDS) may be favorable if the observables are grouped into few groups with similar weights. For simplicity, let each $h_{j}(\boldsymbol{x})$ denote an observable composed of a single group with unit operator norm. Then, we suppose that the number of shots $s_{i, j}$ to measure $h_{j}\left(\boldsymbol{x}_{i}\right)$ is distributed following some probability distribution $P\left(s_{i, j}\right)$ in general. Care is needed to estimate a power of an expectation value $\left\langle h_{j}\left(\boldsymbol{x}_{i}\right)\right\rangle^{n}$, as the power of a sample average is not an unbiased estimator because powers of each single sample introduce the bias. In our framework, based on the U-statistic formalism [62,90], an unbiased estimator of $\left\langle h_{j}\left(\boldsymbol{x}_{i}\right)\right\rangle^{n}$ is obtained as

$$
\begin{equation*}
\hat{\mathcal{E}}_{i, j, n}=\frac{1}{\mathbb{E}\left[\binom{s_{i, j}}{n}\right]} \sum_{1 \leq k_{1}<k_{2}<\cdots<k_{n} \leq s_{i, j}} \prod_{l=1}^{n} r_{i, j, k_{l}} \tag{11}
\end{equation*}
$$

where $r_{i, j, k}$ denotes the outcome of $k$-th measurement of $h_{j}\left(\boldsymbol{x}_{i}\right)$, and $\mathbb{E}[X]$ denotes the expectation value of $X$.

## C. Mini-batch gradient

This subsection discusses the implementation and benefits of the mini-batch gradient approach in the context of quantum machine learning, contrasting it with the random shot allocation strategy. A recent optimizer, Refoqus, as mentioned in Ref. [62], incorporates a unique strategy wherein the number of shots is randomly allocated among data points. This allocation is based on the weight of each term and is an extension of Rosalin [42] designed for VQE. Consequently, the data points under evaluation are randomly chosen with replacement during the estimation of each gradient component. The number of data points evaluated is autonomously determined through this method.

Despite providing an unbiased gradient estimator that respects weights, this strategy has potential pitfalls for machine learning applications. Firstly, by independently selecting random data points for each gradient component, inter-component gradient correlations are overlooked. As a result, the estimated gradient could be noisier compared to when these correlations are considered. Secondly, choosing data points with replacement means that assessing the entire dataset requires more time than without replacement. Furthermore, in common scenarios with a uniform weight $p_{i}=1 / N$, uniformly distributing shots across data points in Refoqus often results in the evaluation of a maximal number of data points for the given shot count. Although such a distribution can minimize the variance of the estimator for a preset shot count, this does not necessarily lead to better optimization performance. This observation mirrors the fact that stochastic gradient descent often outperforms full-batch gradient descent, even though the latter uses the 'true' gradient. In general, the number of data points evaluated at each iteration can have intricate effects on generalization performance, with fewer data points typically yielding superior results.

Consistent with prevalent machine learning practices, we opt for a mini-batch gradient. For simplicity, we consider cases with a uniform weight $p_{i}=1 / N$. In a minibatch strategy, the mini-batch gradient $\nabla \tilde{L}$ of the loss function is evaluated as

$$
\begin{align*}
\frac{\partial \tilde{L}}{\partial \theta_{j}}(\boldsymbol{\theta}) & =\frac{1}{m} \sum_{l=1}^{m} \frac{\partial \ell}{\partial E}\left(\boldsymbol{x}_{i_{l}}, E\left(\boldsymbol{x}_{i_{l}}, \boldsymbol{\theta}\right)\right) \frac{\partial E}{\partial \theta_{j}}\left(\boldsymbol{x}_{i_{l}}, \boldsymbol{\theta}\right) \\
& =: \frac{1}{m} \sum_{l=1}^{m} \mathrm{f}_{j}\left(\boldsymbol{x}_{i_{l}}, \boldsymbol{\theta}\right) \tag{12}
\end{align*}
$$

Here, a mini-batch $\left\{\boldsymbol{x}_{i_{1}}, \cdots, \boldsymbol{x}_{i_{m}}\right\}$ of size $m$ is chosen randomly from the dataset $\mathcal{D}$ without replacement until the whole dataset is evaluated, at which point it is refreshed. We allocate an equal number of shots for the evaluation of each data point. In cases where the weight $p_{i}$ is not uniform, a more effective strategy would be to allocate the number of shots in accordance with this weight. While this strategy introduces the mini-batch size $m$ as an additional hyperparameter, it can lead to superior performance. Indeed, in our simulations in Sec. V, Refoqus underperforms relative to optimizers that utilize a minibatch gradient, including the one we propose.

## III. STOCHASTIC ANNEALING THERMOSTATS IN CLASSICAL MACHINE LEARNING

In this section, as a preliminary to presenting our algorithm, we discuss Santa [64], a classical optimizer that serves as the foundational basis for our approach. Santa was originally developed as an intermediary between stochastic-gradient Markov chain Monte Carlo (SG-MCMC) methods and stochastic optimization, effectively amalgamating the two paradigms. We start by reviewing the underlying principles of Bayesian sampling and SG-MCMC algorithms, emphasizing their relevance to stochastic optimization, as articulated in Ref. [64].

## A. Bayesian sampling and stochastic optimizations, SG-MCMC algorithms

Stochastic optimization methods aim to obtain optimal parameters for an objective function. Common stochastic optimization methods, such as stochastic gradient descent (SGD), can only find some local minima for non-convex objective functions. On the other hand, the Bayesian approach provides a probabilistic framework, offering not just point estimates but entire distributions over possible parameter values. Here, instead of directly finding the optimal parameters, we estimate the likelihood of the parameters given the data. This probabilistic viewpoint allows for a more exploratory and holistic understanding of the parameter space.

More precisely, the Bayesian approach to machine learning aims to infer the model parameters $\boldsymbol{\theta} \in \mathbb{R}^{d}$, from the posterior given by

$$
\begin{equation*}
p\left(\boldsymbol{\theta} \mid \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}\right)=\frac{p(\boldsymbol{\theta}) \prod_{i=1}^{N} p\left(\boldsymbol{x}_{i} \mid \boldsymbol{\theta}\right)}{\int p(\boldsymbol{\theta}) \prod_{j=1}^{N} p\left(\boldsymbol{x}_{j} \mid \boldsymbol{\theta}\right) d \boldsymbol{\theta}} \tag{13}
\end{equation*}
$$

upon observing data $\left\{\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}\right\}$. Here, $p(\boldsymbol{\theta})$ represents the prior, while $p\left(\boldsymbol{x}_{i} \mid \boldsymbol{\theta}\right)$ denotes the likelihood of data $\boldsymbol{x}_{i}$ given the model parameter $\boldsymbol{\theta}$. Sampling from the Bayesian posterior distribution offers unique advantages. It allows models to express uncertainty about parameter values, potentially leading to more robust and generalizable solutions.

This task can be equivalently framed as sampling from a probability distribution proportional to $p(\boldsymbol{\theta}) \prod_{i=1}^{N} p\left(\boldsymbol{x}_{i} \mid \boldsymbol{\theta}\right)=e^{-L(\boldsymbol{\theta})}$, where the negative logposterior $L(\boldsymbol{\theta})$, is defined as

$$
\begin{equation*}
L(\boldsymbol{\theta})=-\log p(\boldsymbol{\theta})-\sum_{i=1}^{N} \log p\left(\boldsymbol{x}_{i} \mid \boldsymbol{\theta}\right) \tag{14}
\end{equation*}
$$

If $N$ is large, we use a mini-batch stochastic loss function $\tilde{L}_{t}(\boldsymbol{\theta}):=-\log p(\boldsymbol{\theta})-\frac{N}{m} \sum_{j=1}^{m} \log p\left(\boldsymbol{x}_{j_{i}} \mid \boldsymbol{\theta}\right)$ in the same way as the stochastic optimization. The posterior is the same as the Gibbs distribution $\propto e^{-\beta L(\boldsymbol{\theta})}$, with the inverse temperature $\beta=1$ and the potential energy $L(\boldsymbol{\theta})$.
In fact, the tempered Bayesian posterior $\beta \neq 1$ is recognized as a generalized form of the Bayesian posterior [9195]. Lowering the value of $\beta$ to less than 1 has been shown to improve the robustness of convergence [96]. This effect has been further explored within the framework of safeBayesian methods [97-100]. Recent research [101-103] has highlighted the cold posterior effect, wherein sampling from a cold posterior with $\beta>1$ can offer even better generalization in some scenarios. This approach emphasizes regions of the parameter space that are more consistent with both the prior and the data, providing a nuanced balance between fitting the data and not overfitting. More generally, the posterior given as the Gibbs distribution $\propto e^{-\beta L(\boldsymbol{\theta})}$ for general loss function $L$ is called Gibbs posterior. This analogy offers a link between the Bayesian posterior sampling and the stochastic optimization approaches. Particularly, as $\beta$ tends toward infinity, the procedure converges to the maximum a posteriori (MAP) estimation, aiming for the global minima of the loss function $L$ [104]. Thus, sampling from a cold posterior $\propto e^{-\beta L(\boldsymbol{\theta})}$ with an elevated $\beta$ guides us closer to minimizing $L$ globally.

In that sense, the sampling approach can also be applied to optimization of general objective functions $L(\boldsymbol{\theta})$ that extend beyond the negative log-posterior, even when they do not have a clear interpretation related to a posterior. Additionally, it's worth noting that the term $-\log p(\boldsymbol{\theta})$ is translated to a regularization term. Indeed, the well-known $L_{2}$-regularization aligns with the Gaussian prior.

In practice, exact sampling from the posterior is too expensive and we need some approximated sampling method. SG-MCMC approaches offer efficient sampling methods. The basic one is the stochastic gradient Langevin dynamics (SGLD) [105] which updates the parameters as $\boldsymbol{\theta}_{t}=\boldsymbol{\theta}_{t-1}-\eta_{t} \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)+\sqrt{2 \eta_{t} \beta^{-1}} \boldsymbol{\zeta}_{t}$ with the learning rate $\eta_{t}$ and the inverse temperature $\beta$, where $\zeta_{t} \sim \mathcal{N}\left(0, I_{d}\right)$ is the additional noise term drawn from the standard normal distribution. SGLD has been also applied in VQAs [59]. SGLD approximates the (cold) posterior for $\beta=1(\beta>1)$ by simulating the overdamped Langevin dynamics. The stochastic gradient Hamiltonian Monte Carlo (SGHMC) [106] incorporates momentum, which corresponds to the underdamped Langevin diffusion. Given the recognized importance of momentum in deep model training within stochastic optimization [107], its incorporation is a logical progression. Indeed, SGHMC's connection to SGD with momentum was already highlighted in Ref. [106]. Introducing the friction term in SGHMC is crucial to stabilize the target stationary distribution, preventing stochastic noise from blowing the parameters far away. Yet, even with this friction term, SGHMC can deviate from the desired thermal equilibrium distribution, particularly when stochastic noise model is poorly estimated. As remedies, the stochastic gradient Nosé-Hoover thermostat (SGNHT) [108] and its multivariate counterpart (mSGNHT) [109] were proposed, adapting the friction term to emulate the energy
conservation of the Nosé-Hoover thermostat $[110,111]$.

Another important technique is the preconditioning. In the context of stochastic optimization and SG-MCMC, preconditioning is a technique that aims to improve convergence by appropriately transforming the underlying parameter space, thereby using a metric incorporating a geometric structure that fits the problem. For a preconditioned SGD [112-114], the update rule is modified by a preconditioner matrix $P_{t}$ as $\boldsymbol{\theta}_{t}=\boldsymbol{\theta}_{t-1}-\eta_{t} P_{t} \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)$. Adaptive stochastic optimizers such as AdaGrad [115], RMSprop [116], Adam [35], and their variants incorporate adaptive preconditioning by utilizing historical gradient information to adjust the learning rate for each parameter. Preconditioning strategies have also found their way into SG-MCMC algorithms [117-119]. Actually, the significance of judicious preconditioning is widely acknowledged in both realms [112, 117-119].

## B. The classical Santa optimizer

Extending the mSGNHT by incorporating adaptive preconditioning and a simulated annealing scheme on the system temperature, Stochastic AnNealing Thermostats with Adaptive momentum (Santa) was proposed as an optimizer for classical objective functions [64]. Santa algorithm is based on a simulated dynamics of the following stochastic differential equation (SDE) given an inverse temperature $\beta$ :

$$
\left\{\begin{align*}
d \boldsymbol{\theta}= & G_{1}(\boldsymbol{\theta}) \boldsymbol{p} d t \\
d \boldsymbol{p}= & \left(-G_{1}(\boldsymbol{\theta}) \nabla \tilde{L}_{t}(\boldsymbol{\theta})-\boldsymbol{\Xi} \boldsymbol{p}\right) d t \\
& +\left(\frac{1}{\beta} \nabla G_{1}(\boldsymbol{\theta})+G_{1}(\boldsymbol{\theta})\left(\boldsymbol{\Xi}-G_{2}(\boldsymbol{\theta})\right) \nabla G_{2}(\boldsymbol{\theta})\right) d t  \tag{15}\\
& +\sqrt{\frac{2}{\beta}} G_{2}(\boldsymbol{\theta}) d \boldsymbol{w} \\
d \boldsymbol{\Xi}= & \left(\operatorname{diag}\left(\boldsymbol{p}^{2}\right)-\frac{1}{\beta} I\right) d t
\end{align*}\right.
$$

where $\boldsymbol{w}$ is the standard Brownian motion, $G_{1}$ and $G_{2}$ respectively gives preconditioning for $L(\boldsymbol{\theta})$ and the Brownian motion, which encode the respective geometric information. Here, $\nabla G(\boldsymbol{\theta})$ for a matrix $G$ denotes a vector whose $i$-th element is $\sum_{j} \frac{\partial}{\partial \theta_{j}} G_{i, j}(\boldsymbol{\theta})$. Setting $G_{1}=I$ and $G_{2}$ constant reduces to the SDE for mSGNHT [109]. The terms with $\nabla G_{1}(\boldsymbol{\theta})$ and $\nabla G_{2}(\boldsymbol{\theta})$ reflect the spatial variation of the metrics so as to maintain the stationary distribution. Actually, SDE (15) has the target distribution $p_{\beta}(\boldsymbol{\theta})=e^{-\beta L(\boldsymbol{\theta})} / Z_{\beta}$ with the normalization factor $Z_{\beta}$ as its stationary distribution under reasonable assumptions on stochastic noise as we discuss later. A remarkable feature of SDE (15) is that the thermostats $\boldsymbol{\Xi}$ maintain the target stationary distribution irrespective of the detail of stochastic noise. Hence, assuming the ergodicity [120, 121], we obtain approximate samples from the target distribution $p_{\beta}$ after sufficiently long time. Then, in the Santa algorithm, the inverse temperature $\beta$ is slowly annealed towards sufficiently large values (hence low temperature) to explore the parameter space to reach near the global optima of the objective function. After this exploration stage, Santa enters the refinement stage by taking the zero-temperature limit $\beta \rightarrow \infty$ and $\boldsymbol{\Xi}$ is fixed to a "learned value" in the exploration stage. The refinement stage is a stochastic optimization with adaptive preconditioning using a learned friction parameter $\boldsymbol{\Xi}$. Indeed, under mild conditions, the asymptotic convergence of Santa algorithm towards the global optima has been shown [64]. In fact, it is reported that Santa outperforms conventional optimizers such as SGD, SGD with momentum, SGLD, RMSprop, and Adam in several benchmark tasks [64]. Ref. [64] tested Santa on training feedforward and convolutional neural networks on the MNIST dataset as well as training recurrent neural network for the task of sequence modeling on four different polyphonic music sequences of piano. In all cited tasks, Santa demonstrates the highest level of performance.

To numerically implement the parameter updates according to SDE (15), we need approximations. As a simplest Euler scheme introduces relatively high approximation error $[64,122]$, the symmetric splitting scheme (SSS) $[64,122,123]$ is recommended. In Santa, splitting SDE (15), SSS is implemented by solving each of the following sub-SDEs for finite time steps:

$A:\left\{\begin{array}{l}d \boldsymbol{\theta}=G_{1}(\boldsymbol{\theta}) \boldsymbol{p} d t \\ d \boldsymbol{p}=0 \\ d \boldsymbol{\Xi}=\left(\operatorname{diag}\left(\boldsymbol{p}^{2}\right)-\frac{1}{\beta_{t}} I\right) d t\end{array}\right.$,

$B:\left\{\begin{array}{l}d \boldsymbol{\theta}=0 \\ d \boldsymbol{p}=-\boldsymbol{\Xi} \boldsymbol{p} d t, \\ d \boldsymbol{\Xi}=0\end{array}\right.$

$O:\left\{\begin{aligned} d \boldsymbol{\theta} & =0 \\ d \boldsymbol{p} & =-G_{1}(\boldsymbol{\theta}) \nabla \tilde{L}_{t}(\boldsymbol{\theta}) d t \\ & +\left(\frac{1}{\beta_{t}} \nabla G_{1}(\boldsymbol{\theta})+G_{1}(\boldsymbol{\theta})\left(\boldsymbol{\Xi}-G_{2}(\boldsymbol{\theta})\right) \nabla G_{2}(\boldsymbol{\theta})\right) d t \\ & +\sqrt{\frac{2}{\beta_{t}} G_{2}(\boldsymbol{\theta})} d \boldsymbol{w} \\ d \boldsymbol{\Xi} & =0 .\end{aligned}\right.$

For a step size $h$, the solutions are given as

$$
\begin{aligned}
& A:\left\{\begin{array}{l}
\boldsymbol{\theta}_{t}=\boldsymbol{\theta}_{t-1}+G_{1}\left(\boldsymbol{\theta}_{t}\right) \boldsymbol{p}_{t-1} h \\
\boldsymbol{p}_{t}=\boldsymbol{p}_{t-1} \\
\boldsymbol{\Xi}_{t}=\boldsymbol{\Xi}_{t-1}+\left(\operatorname{diag}\left(\boldsymbol{p}_{t-1}^{2}\right)-\frac{1}{\beta_{t}} I\right) h
\end{array}\right. \\
& B:\left\{\begin{aligned}
\boldsymbol{\theta}_{t} & =\boldsymbol{\theta}_{t-1} \\
\boldsymbol{p}_{t} & =\exp \left(-\boldsymbol{\Xi}_{t-1} h\right) \boldsymbol{p}_{t-1} \\
\boldsymbol{\Xi}_{t} & =\boldsymbol{\Xi}_{t-1}
\end{aligned}\right.
\end{aligned}
$$

$$
O:\left\{\begin{align*}
\boldsymbol{\theta}_{t} & =\boldsymbol{\theta}_{t-1}  \tag{16}\\
\boldsymbol{p}_{t} & =\boldsymbol{p}_{t-1}-G_{1}\left(\boldsymbol{\theta}_{t}\right) \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right) h \\
& +\left(\frac{1}{\beta_{t}} \nabla G_{1}\left(\boldsymbol{\theta}_{t}\right)+G_{1}\left(\boldsymbol{\theta}_{t}\right)\left(\boldsymbol{\Xi}-G_{2}\left(\boldsymbol{\theta}_{t}\right)\right) \nabla G_{2}\left(\boldsymbol{\theta}_{t}\right)\right) h \\
& +\sqrt{\frac{2}{\beta_{t}} h G_{2}\left(\boldsymbol{\theta}_{t}\right) \boldsymbol{\zeta}_{t}} \\
\boldsymbol{\Xi}_{t} & =\boldsymbol{\Xi}_{t-1}
\end{align*}\right.
$$

where $\boldsymbol{\zeta}_{t}$ is a random vector drawn from $\mathcal{N}(\mathbf{0}, I)$. Here, we neglect the change in $G_{1}(\boldsymbol{\theta})$ and $G_{2}(\boldsymbol{\theta})$ within a single parameter update given that $h$ is small enough. Then, in Santa, parameters are updated in order $A-B-O-B-A$ with half steps $h / 2$ on $A$ and $B$ updates, and full steps $h$ on the $O$ updates.

SDE (15) implicitly includes stochastic noise in the mini-batch approximation of the gradient of the loss function $\nabla \tilde{L}_{t}(\boldsymbol{\theta})$. Based on the central limit theorem, this stochastic noise can be approximated as $G_{1}(\boldsymbol{\theta}) \nabla \tilde{L}_{t}(\boldsymbol{\theta}) d t \approx G_{1}(\boldsymbol{\theta}) \nabla L(\boldsymbol{\theta}) d t+\mathcal{N}(0,2 B(\boldsymbol{\theta}) d t)$ with the diffusion matrix $B(\boldsymbol{\theta})$ of the stochastic gradient noise [106]. Then, for general total diffusion matrix $D(\boldsymbol{\theta})=$ $\frac{1}{\beta} G_{2}(\boldsymbol{\theta})+B(\boldsymbol{\theta})$ including both the injected part $\frac{1}{\beta} G_{2}(\boldsymbol{\theta})$ and stochastic noise part $B(\boldsymbol{\theta})$, more precise SDE with the desired stationary distribution becomes [108]

$$
\left\{\begin{align*}
d \boldsymbol{\theta}= & G_{1}(\boldsymbol{\theta}) \boldsymbol{p} d t  \tag{17}\\
d \boldsymbol{p}= & \left(-G_{1}(\boldsymbol{\theta}) \nabla L(\boldsymbol{\theta})-\boldsymbol{\Xi} \boldsymbol{p}\right) d t \\
& +\left(\frac{1}{\beta} \nabla G_{1}(\boldsymbol{\theta})+G_{1}(\boldsymbol{\theta})(\boldsymbol{\Xi}-D(\boldsymbol{\theta})) \nabla D(\boldsymbol{\theta})\right) d t \\
& +\sqrt{2 D(\boldsymbol{\theta})} d \boldsymbol{w} \\
d \boldsymbol{\Xi}= & \left(\boldsymbol{p} \boldsymbol{p}^{T}-\frac{1}{\beta} I\right) d t
\end{align*}\right.
$$

In fact, it has been shown [108] that the stationary distribution $\pi_{\beta}(\boldsymbol{\theta}, \boldsymbol{p}, \boldsymbol{\Xi})$ of $\mathrm{SDE}$ (17) is given by

$$
\begin{equation*}
\pi_{\beta}(\boldsymbol{\theta}, \boldsymbol{p}, \boldsymbol{\Xi}) \propto e^{-\beta L(\boldsymbol{\theta})-\frac{\beta}{2} \boldsymbol{p}^{T} \boldsymbol{p}-\frac{\beta}{2}} \operatorname{Tr}\left[(\boldsymbol{\Xi}-D(\boldsymbol{\theta}))^{T}(\boldsymbol{\Xi}-D(\boldsymbol{\theta}))\right] \tag{18}
\end{equation*}
$$

Hence, the marginal stationary distribution of (18) is the desired one $p_{\beta}(\boldsymbol{\theta}) \propto e^{-\beta L(\boldsymbol{\theta})}$. SDE (15) used in Santa is assured to have the same stationary distribution by assuming the diagonal form of $D(\boldsymbol{\theta})$ and neglecting $B(\boldsymbol{\theta})$ in the term with $\nabla D(\boldsymbol{\theta})$. This assumption is reasonable when $G_{2}(\boldsymbol{\theta})$ is diagonal and the number of the evaluated data is large enough so that $B(\boldsymbol{\theta})$ and its derivatives are small compared to $G_{2}(\boldsymbol{\theta})$ [108]. Indeed, only diagonal $G_{2}(\boldsymbol{\theta})$ is used in Santa. Remarkably, the offdiagonal parts of the SDE for $\boldsymbol{\Xi}$ can be omitted in this case, which considerably reduces the computational cost.

The thermostat variable $\boldsymbol{\Xi}$ maintains the system temperature by controlling the friction to imposing that the component-wise kinetic energy $p_{i}^{2} / 2$ is close to $1 /(2 \beta)$. If the energy is bigger than this value, the momentum $\boldsymbol{p}$ experiences more friction, and the opposite for lower energy values. Moreover, the thermostat "absorbs" the effects of the stochastic term of $D(\boldsymbol{\theta})$ [108]. In fact, as seen from the stationary distribution (18), the distribution of $\Xi$ is changed to a matrix normal distribution with mean $D(\boldsymbol{\theta})$ reflecting stochastic noise $D(\boldsymbol{\theta})$, and the marginal stationary distribution of $\boldsymbol{\theta}$ is left invariant irrespective of $D(\boldsymbol{\theta})$. As a result, even with imprecise estimation of stochastic noise $B(\boldsymbol{\theta})$, we can stably obtain the stationary distribution by the dynamics (15). This feature is especially beneficial in its application to VQAs.

## IV. THE SANTAQLAUS ALGORITHM

The classical Santa algorithm obtains (near) global optimality by leveraging annealed thermostats with injected noise. In VQAs, evaluations of the objective function are inherently noisy due to quantum measurements. Inspired by this natural intersection of noise characteristics, we propose SantaQlaus, an optimization algorithm designed to leverage intrinsic QSN in VQAs in a resourceefficient manner, emulating thermal noise used in Santa. Incorporating the asymptotic normality of gradient estimators, our approach justifies the use of QSN as an analogue to thermal noise. Specifically, the asymptotic normality not only provides theoretical underpinning for its use but also guides the adjustment of shot numbers to align with the desired level of thermal noise. We start by discussing the asymptotic normality of gradient estimators of loss functions.

## A. Asymptotic normality of gradient estimators

This section delves into the asymptotic normality of gradient estimators for loss functions as a precursor to the deployment of the SantaQlaus algorithm.

Let $\hat{\mathbf{f}}(\boldsymbol{x}, \boldsymbol{\theta})_{\boldsymbol{s}}=\left(\hat{\mathrm{f}}_{1}(\boldsymbol{x}, \boldsymbol{\theta})_{\boldsymbol{s}}, \cdots, \hat{\mathrm{f}}_{d}(\boldsymbol{x}, \boldsymbol{\theta})_{\boldsymbol{s}}\right)^{T}$ denote an estimator of the stochastic gradient of loss function $L$ evaluated at a single data point $\boldsymbol{x}$ (see Eq. (12)) using $s=\left(s_{1}, \cdots, s_{d}\right)$ shots for each component, where $d$ is the number of the parameters including the weight parameters. We assume a central limit theorem-like asymptotic normality in the following form:

$$
\begin{align*}
\hat{\mathrm{f}}_{j}(\boldsymbol{x}, \boldsymbol{\theta})_{\boldsymbol{s}} & =\mathrm{f}_{j}(\boldsymbol{x}, \boldsymbol{\theta})+\mathcal{N}\left(0, \frac{S_{j}(\boldsymbol{x}, \boldsymbol{\theta})}{s_{j}}\right)+o\left(\frac{1}{\sqrt{s_{j}}}\right) \\
& \approx \mathrm{f}_{j}(\boldsymbol{x}, \boldsymbol{\theta})+\mathcal{N}\left(0, \frac{S_{j}(\boldsymbol{x}, \boldsymbol{\theta})}{s_{j}}\right) \tag{19}
\end{align*}
$$

with some function $S_{j}(\boldsymbol{x}, \boldsymbol{\theta})$, where $\approx$ denotes an approximation up to the leading order terms. Here, the notation $\mathcal{N}\left(\mu, \sigma^{2}\right)$ is abused to denote a random variable following the normal distribution with mean $\mu$ and variance $\sigma^{2}$. A wide range of loss functions actually satisfies this form
of asymptotic normality as seen below. It is worth noting that we do not enforce $s_{j}$ to be strictly equal to the total number of shots used for estimating each $\mathrm{f}_{j}(\boldsymbol{x}, \boldsymbol{\theta})$. Rather, $s_{j}$ is considered a parameter that characterizes the estimator in such a way that it appears in Eq. (19). A set of rules for determining the number of shots required to evaluate the expectation values needed for estimating $\mathrm{f}_{j}$ is defined in terms of $s_{j}$.

For instance, we consider a simple linear loss function with $\ell(E(\boldsymbol{x}, \boldsymbol{\theta}, w))=w\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}$ given by a single observable $h(\boldsymbol{x})$. We have $\mathrm{f}_{j}(\boldsymbol{x}, \boldsymbol{\theta}, w)=w\left(\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}+\frac{\pi}{2}} \boldsymbol{e}_{j}-\right.$ $\left.\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}-\frac{\pi}{2} e_{j}}\right) / 2(j \leq d-1)$ and $\mathrm{f}_{d}(\boldsymbol{x}, \boldsymbol{\theta}, w)=\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}$, where the weight $w$ is assigned as $d$-th parameter. Hence, an estimator $\hat{\mathrm{f}}_{j}(\boldsymbol{x}, \boldsymbol{\theta})_{\boldsymbol{s}}$ is simply given by sample means as

$$
\begin{align*}
& \hat{\mathrm{f}}_{j}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{s}=\frac{w}{2}\left(\sum_{k=1}^{s_{j}} \frac{r_{k}^{+}}{s_{j}}-\sum_{k^{\prime}=1}^{s_{j}} \frac{r_{k^{\prime}}^{-}}{s_{j}}\right) \quad(j \leq d-1)  \tag{20}\\
& \hat{\mathrm{f}}_{d}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{s}=\sum_{k=1}^{s_{d}} \frac{r_{k}}{s_{d}} \tag{21}
\end{align*}
$$

where $r_{k}$ and $r_{k}^{ \pm}$denote the outcome of $k$-th measurement of $h(\boldsymbol{x})$ with parameters $\boldsymbol{\theta}$ and $\boldsymbol{\theta} \pm \frac{\pi}{2} \boldsymbol{e}_{j}$ respectively. We note that $2 s_{j}$ shots are used to estimate $j$-th component because $s_{j}$ shots are used for each shifted parameter. In this case, the central limit theorem reads

$$
\begin{align*}
& \hat{\mathrm{f}}_{j}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{\boldsymbol{s}} \\
\approx & \mathrm{f}_{j}(\boldsymbol{x}, \boldsymbol{\theta}, w)+\mathcal{N}\left(0, w^{2} \frac{\sigma_{\boldsymbol{x}, \boldsymbol{\theta}+\frac{\pi}{2} e_{j}}^{2}+\sigma_{\boldsymbol{x}, \boldsymbol{\theta}-\frac{\pi}{2} e_{j}}^{2}}{4 s_{j}}\right) \quad(j \leq d-1) \\
& \hat{\mathrm{f}}_{d}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{\boldsymbol{s}} \\
\approx & \mathrm{f}_{d}(\boldsymbol{x}, \boldsymbol{\theta}, w)+\mathcal{N}\left(0, \frac{\sigma_{\boldsymbol{x}, \boldsymbol{\theta}}^{2}}{s_{d}}\right) \tag{22}
\end{align*}
$$

where $\sigma_{\boldsymbol{x}, \boldsymbol{\theta}}^{2}$ denotes the variance of $h(\boldsymbol{x})$ with respect to the state $\rho(\boldsymbol{x}, \boldsymbol{\theta})$. It is straightforward to generalize Eq. (22) to more general cases where we have multiple observables $h_{k}(\boldsymbol{x})$ to evaluate. We remark that the samples obtained with WRS are equivalent to $s_{j}$ i.i.d. samples where each single sample is drawn by measuring a randomly chosen term $h_{k}(\boldsymbol{x})$. Hence, we can directly apply the above arguments.

For polynomial loss functions, $\mathrm{f}_{j}$ includes a polynomial of the expectation values of observables in general. We can apply a central limit theorem for Ustatistic [90] in such cases. As a typical example, we consider a quadratic loss function $\ell(\boldsymbol{x}, E(\boldsymbol{x}, \boldsymbol{\theta}, w))=$ $\sum_{z=0}^{2} a_{z}(\boldsymbol{x})\left(w\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}\right)^{z}$, including the MSE loss for a prediction given by $w\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}$. In this case, the partial derivatives reads

$$
\begin{aligned}
& \mathrm{f}_{j}(\boldsymbol{x}, \boldsymbol{\theta}, w) \\
= & w\left(\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}+\frac{\pi}{2}} e_{j}-\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}-\frac{\pi}{2}} e_{j}\right)
\end{aligned}
$$

$$
\begin{align*}
& \times\left(\frac{a_{1}(\boldsymbol{x})}{2}+w a_{2}(\boldsymbol{x})\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}\right) \quad(j \leq d-1)  \tag{23}\\
& \mathrm{f}_{d}(\boldsymbol{x}, \boldsymbol{\theta}, w) \\
= & a_{1}(\boldsymbol{x})\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}+2 w a_{2}(\boldsymbol{x})\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}^{2} \tag{24}
\end{align*}
$$

Then, for given $s$, unbiased estimators of the derivatives are given by

$$
\begin{align*}
\hat{\mathrm{f}}_{j}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{s}= & w\left(\sum_{k=1}^{s_{j}} \frac{r_{k}^{+}}{s_{j}}-\sum_{k^{\prime}=1}^{s_{j}} \frac{r_{k^{\prime}}^{-}}{s_{j}}\right) \\
& \times\left(\frac{a_{1}(\boldsymbol{x})}{2}+w a_{2}(\boldsymbol{x}) \sum_{k=1}^{s_{d}} \frac{r_{k}}{s_{d}}\right) \quad(j \leq d-1)  \tag{25}\\
\hat{\mathrm{f}}_{d}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{s}= & a_{1}(\boldsymbol{x}) \sum_{k=1}^{s_{d}} \frac{r_{k}}{s_{d}}+2 w a_{2}(\boldsymbol{x}) \sum_{1 \leq k_{1} \neq k_{2} \leq s_{d}} \frac{r_{k_{1}} r_{k_{2}}}{s_{d}\left(s_{d}-1\right)} \tag{26}
\end{align*}
$$

with the same notations as above. In this estimator, we use $s_{j}$ shots to evaluate at each shifted parameter and $s_{d}$ shots for the unshifted parameter. Hence, $2 \sum_{j=1}^{d-1} s_{j}+s_{d}$ shots are used in total. For the circuit parameters $j \leq$ $d-1$, the estimator composed of a product of sample means of independent random variables. We note that sequences of two independent and identically distributed (i.i.d.) random variables $X_{1}, \cdots, X_{n_{1}}$ and $Y_{1}, \cdots, Y_{n_{2}}$ with respective means $\mu_{X}, \mu_{Y}$ satisfy

$$
\begin{align*}
& \sum_{k=1}^{n_{1}} \frac{X_{k}}{n_{1}} \sum_{k^{\prime}=1}^{n_{2}} \frac{Y_{k^{\prime}}}{n_{2}}-\mu_{X} \mu_{Y} \\
= & \mu_{X}\left(\sum_{k^{\prime}=1}^{n_{2}} \frac{Y_{k^{\prime}}}{n_{2}}-\mu_{Y}\right)+\left(\sum_{k=1}^{n_{1}} \frac{X_{k}}{n_{1}}-\mu_{X}\right) \mu_{Y} \\
& +\left(\sum_{k=1}^{n_{1}} \frac{X_{k}}{n_{1}}-\mu_{X}\right)\left(\sum_{k^{\prime}=1}^{n_{2}} \frac{Y_{k^{\prime}}}{n_{2}}-\mu_{Y}\right) \tag{27}
\end{align*}
$$

Then, we can apply the central limit theorem to $\sum_{k=1}^{n_{1}} \frac{X_{k}}{n_{1}}-\mu_{X}$ and $\sum_{k^{\prime}=1}^{n_{2}} \frac{Y_{k^{\prime}}}{n_{2}}-\mu_{Y}$. Because the product of the convergent sequences converges to the product of their limits, the third term of the right hand side (RHS) of Eq. (27) turns out to be of sub-leading order. Hence, we obtain

$$
\begin{equation*}
\hat{\mathrm{f}}_{j}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{s} \approx \mathrm{f}_{j}(\boldsymbol{x}, \boldsymbol{\theta}, w)+\mathcal{N}\left(0, \frac{S_{j}(\boldsymbol{x}, \boldsymbol{\theta})}{s_{j}}\right) \tag{28}
\end{equation*}
$$

with

$$
\begin{align*}
& S_{j}(\boldsymbol{x}, \boldsymbol{\theta}) \\
= & \mu_{2}^{2}\left(\sigma_{\boldsymbol{x}, \boldsymbol{\theta}+\frac{\pi}{2} \boldsymbol{e}_{j}}^{2}+\sigma_{\boldsymbol{x}, \boldsymbol{\theta}-\frac{\pi}{2} \boldsymbol{e}_{j}}^{2}\right)+\kappa_{j} \mu_{1}^{2} w^{2} a_{2}(\boldsymbol{x})^{2} \sigma_{\boldsymbol{x}, \boldsymbol{\theta}}^{2} \tag{29}
\end{align*}
$$

where $\left.\mu_{1}:=h(\boldsymbol{x})\right\rangle_{\boldsymbol{x}, \boldsymbol{\theta}+\frac{\pi}{2}} \boldsymbol{e}_{j}-\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}-\frac{\pi}{2}} e_{j}$ and $\mu_{2}:=$ $\frac{a_{1}(\boldsymbol{x})}{2}+w a_{2}(\boldsymbol{x})\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}$, and we assume that $s_{j} / s_{d} \rightarrow \kappa_{j}$.

For the weight parameter $j=d$, we apply the theory of U-statistic. For i.i.d. random variables $X_{1}, \cdots, X_{n}$, U-statistic with respect
to a symmetric function $\Phi\left(\xi_{1}, \cdots, \xi_{m}\right)$ is defined as $\sum_{1 \leq k_{1} \neq \cdots \neq k_{m} \leq n} \Phi\left(X_{k_{1}}, \cdots, X_{k_{m}}\right) /[n(n-1) \cdots(n-m+$ 1)]. Thus, Eq. (26) coincides with the U-statistic with respect to the function $\Phi\left(\xi_{1}, \xi_{2}\right)=\frac{a_{1}(\boldsymbol{x})}{2}\left(\xi_{1}+\xi_{2}\right)+$ $2 w a_{2}(\boldsymbol{x}) \xi_{1} \xi_{2}$. Therefore, applying a central limit theorem for U-statistics [90], we obtain

$$
\begin{equation*}
\hat{\mathrm{f}}_{d}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{\boldsymbol{s}} \approx \mathrm{f}_{d}(\boldsymbol{x}, \boldsymbol{\theta}, w)_{\boldsymbol{s}}+\mathcal{N}\left(0, \frac{S_{d}(\boldsymbol{x}, \boldsymbol{\theta})}{s_{d}}\right) \tag{30}
\end{equation*}
$$

where

$$
\begin{align*}
S_{d}(\boldsymbol{x}, \boldsymbol{\theta})= & \sigma_{\boldsymbol{x}, \boldsymbol{\theta}}^{2}\left[a_{1}(\boldsymbol{x})^{2}+8 w a_{1}(\boldsymbol{x}) a_{2}(\boldsymbol{x})\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}\right. \\
& \left.+16\left(w a_{2}(\boldsymbol{x})\langle h(\boldsymbol{x})\rangle_{\boldsymbol{x}, \boldsymbol{\theta}}\right)^{2}\right] . \tag{31}
\end{align*}
$$

This way, we obtain the asymptotic normality in a concrete form for quadratic loss functions. We can apply similar calculations to more general polynomial loss functions.

An unbiased estimator $\hat{\mathbf{f}}(\boldsymbol{\theta})_{\boldsymbol{s}}$ of the stochastic gradient $\nabla \tilde{L}(\boldsymbol{\theta})$ is given as

$$
\begin{equation*}
\hat{\mathrm{f}}_{j}(\boldsymbol{\theta})_{s}:=\frac{1}{m} \sum_{l=1}^{m} \hat{\mathrm{f}}_{j}\left(\boldsymbol{x}_{i_{l}}, \boldsymbol{\theta}\right)_{s} \tag{32}
\end{equation*}
$$

with respect to a mini-batch $\left\{\boldsymbol{x}_{i_{1}}, \cdots, \boldsymbol{x}_{i_{m}}\right\}$ of size $m$. Then, we further apply the central limit theorem with respect to the mini-batch average as follows. From the asymptotic normality Eq. (19), we have

$$
\begin{align*}
\hat{\mathrm{f}}_{j}(\boldsymbol{\theta})_{s} & \approx \frac{1}{m} \sum_{l=1}^{m}\left(\mathrm{f}_{j}\left(\boldsymbol{x}_{i_{l}}, \boldsymbol{\theta}\right)+\sqrt{\frac{S_{j}\left(\boldsymbol{x}_{i_{l}}, \boldsymbol{\theta}\right)}{s_{j}}} \zeta_{l}\right) \\
& =\frac{\partial \tilde{L}}{\partial \theta_{j}}(\boldsymbol{\theta})+\frac{1}{m} \sum_{l=1}^{m} \sqrt{\frac{S_{j}\left(\boldsymbol{x}_{i_{l}}, \boldsymbol{\theta}\right)}{s_{j}}} \zeta_{l, j} \tag{33}
\end{align*}
$$

where $\zeta_{l, j}(l=1, \cdots, m),(j=1, \cdots, d)$ are independent standard normally distributed random variables $\sim$ $\mathcal{N}(0,1)$. For simplicity, assuming that $N$ is large enough compared to $m$, we neglect the influence of the sampling without replacement, so that $x_{i_{l}}(l=1, \cdots, m)$ can be treated as approximately i.i.d. random variables, which are uniformly drawn from the dataset $\mathcal{D}$. Indeed, the following arguments based on this approximation can be justified by a central limit theorem for random partition $[124,125]$. Then, $\sqrt{\frac{S\left(\boldsymbol{x}_{i_{1}}, \boldsymbol{\theta}\right)}{\boldsymbol{s}}} \zeta_{1}, \cdots, \sqrt{\frac{S\left(\boldsymbol{x}_{i_{m}}, \boldsymbol{\theta}\right)}{s}} \zeta_{m}$ can be regarded as a sequence of i.i.d. random vectors, whose mean is $\mathbf{0}$, and the covariance reads

$$
\begin{align*}
\mathbb{E}\left[\sqrt{\frac{S_{j}(\boldsymbol{X}, \boldsymbol{\theta})}{s_{j}} \frac{S_{k}(\boldsymbol{X}, \boldsymbol{\theta})}{s_{k}}} \zeta_{j} \zeta_{k}\right] & =\frac{\mathbb{E}_{\boldsymbol{X}}\left[S_{j}(\boldsymbol{X}, \boldsymbol{\theta})\right]}{s_{j}} \delta_{j, k} \\
& =: \frac{S_{j}(\boldsymbol{\theta})}{s_{j}} \delta_{j, k} \tag{34}
\end{align*}
$$

where $\mathbb{E}_{\boldsymbol{X}}$ denotes the expectation with respect to the data sampling $\boldsymbol{X}$. Here a fraction of vectors mean component-wise. Product $\boldsymbol{v}_{1} \boldsymbol{v}_{2}$ of two vectors $\boldsymbol{v}_{1}$ and $\boldsymbol{v}_{2}$ also denote component-wise product in the following. Then, from Eq. (33), the central limit theorem with respect to the sum $\frac{1}{m} \sum_{l=1}^{m} \sqrt{\frac{S\left(\boldsymbol{x}_{\left.i_{l}, \boldsymbol{\theta}\right)}\right.}{s}} \zeta_{l}$ yields an approximation

$$
\begin{equation*}
\hat{\mathbf{f}}(\boldsymbol{\theta})_{\boldsymbol{s}} \approx \nabla \tilde{L}_{t}(\boldsymbol{\theta})+\mathcal{N}\left(\mathbf{0}, \operatorname{diag}\left(\frac{\boldsymbol{S}(\boldsymbol{\theta})}{m \boldsymbol{s}}\right)\right) \tag{35}
\end{equation*}
$$

where $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ denotes a random vector following the multivariate Gaussian distribution with mean $\boldsymbol{\mu}$ and covariance matrix $\Sigma$.

## B. Details of SantaQlaus

As detailed in the previous section, because samples obtained by quantum measurements are i.i.d. random variables, we can apply central limit theorems to typical loss functions, so that QSN in a quantum evaluation of each component of the stochastic gradient follows a Gaussian distribution. More precisely, with fairly broad applicability, we assume an asymptotic normality Eq. (19) of an estimator $\hat{\mathbf{f}}(\boldsymbol{x}, \boldsymbol{\theta})_{s}$ of the stochastic gradient of loss function $L$ evaluated at each data point $\boldsymbol{x}$. Based on this, we have derived an approximation of the stochastic gradient

$$
\begin{equation*}
\hat{\mathbf{f}}(\boldsymbol{\theta})_{\boldsymbol{s}} \approx \nabla \tilde{L}_{t}(\boldsymbol{\theta})+\mathcal{N}\left(\mathbf{0}, \operatorname{diag}\left(\frac{\boldsymbol{S}(\boldsymbol{\theta})}{m \boldsymbol{s}}\right)\right) \tag{36}
\end{equation*}
$$

which is restated for convenience. In Eq. (36), the term $\nabla \tilde{L}_{t}(\boldsymbol{\theta})$ only includes noise due to the mini-batch approximation, and QSN is isolated as Gaussian noise $\mathcal{N}\left(\mathbf{0}, \operatorname{diag}\left(\frac{\boldsymbol{S}(\boldsymbol{\theta})}{m \boldsymbol{s}}\right)\right)$. Hence, as QSN is approximated as additive Gaussian noise with diagonal covariance matrix, we can use it as thermal noise in Santa. We can achieve this by adjusting the number $\boldsymbol{n}=m \boldsymbol{s}$ to yield the variance corresponding to the desired thermal noise. In the parameter update rule (16) of Santa, with an estimate $\hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)_{s}$ of $G_{1}\left(\boldsymbol{\theta}_{t}\right)$, the stochastic gradient appears as $\hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)_{s} \hat{\mathbf{f}}\left(\boldsymbol{\theta}_{t}\right)_{s} h$, which reads

$$
\begin{align*}
& \hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)_{\boldsymbol{s}} \hat{\mathbf{f}}\left(\boldsymbol{\theta}_{t}\right)_{\boldsymbol{s}} h \\
\approx & \hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)_{s} \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right) h+h \hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)_{\boldsymbol{s}} \operatorname{diag}\left(\sqrt{\frac{\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{n}}}\right) \boldsymbol{\zeta}_{t} \tag{37}
\end{align*}
$$

Comparing with the thermal noise term $\mathcal{N}\left(\mathbf{0}, \frac{2}{\beta_{t}} h G_{2}\left(\boldsymbol{\theta}_{t}\right)\right)$, we obtain the following equation to be satisfied for emulating the thermal noise by QSN:

$$
\begin{align*}
& \hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)_{\boldsymbol{s}} \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)+\hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)_{\boldsymbol{s}} \operatorname{diag}\left(\sqrt{\frac{\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{n}}}\right) \boldsymbol{\zeta}_{t} \\
= & G_{1}\left(\boldsymbol{\theta}_{t}\right) \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)+\sqrt{\frac{2}{\beta_{t} h} G_{2}\left(\boldsymbol{\theta}_{t}\right)} \boldsymbol{\zeta}_{t} \tag{38}
\end{align*}
$$

We remark that $\hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)_{\boldsymbol{s}}$ may also includes the noise in general. Although this condition is not always possible to satisfy for arbitrary $G_{1}$ and $G_{2}$, using diagonal preconditioners $G_{1}$ and $G_{2}$ makes it possible to solve Eq. (38) easily. Indeed, we use the RMSprop preconditioner and set $G_{2}(\boldsymbol{\theta}) h=G_{1}(\boldsymbol{\theta})$ as in the original Santa [64] for computational feasibility (see Algorithm 1). RMSprop preconditioner corresponds to $G_{1}\left(\boldsymbol{\theta}_{t}\right)=\operatorname{diag}\left(1 / \boldsymbol{v}_{t}^{1 / 4}\right)$, where $\boldsymbol{v}_{t}=\sigma \boldsymbol{v}_{t-1}+(1-\sigma)\left(\nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)\right)^{2}$ is an exponential moving average of the squared gradient with a parameter $0<\sigma<1$. We leave as future works to incorporate more general preconditioners such as the quantum Fisher metric $[43,44]$. In addition, we assume that $\hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right)$ can be approximated as

$$
\begin{equation*}
\hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right) \approx G_{1}\left(\boldsymbol{\theta}_{t}\right)\left(I+\operatorname{diag}\left(\boldsymbol{g}\left(\boldsymbol{\theta}_{\boldsymbol{t}}\right) \sqrt{\frac{\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{n}}}\right) \boldsymbol{\zeta}_{t}\right) \tag{39}
\end{equation*}
$$

up to the leading order, where the random vector $\zeta_{t}=$ $\mathcal{N}(\mathbf{0}, I)$ is shared with the noise in $\hat{\mathbf{f}}\left(\boldsymbol{\theta}_{t}\right)_{\boldsymbol{s}} \approx \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)+$ $\sqrt{\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right) / \boldsymbol{n}} \boldsymbol{\zeta}_{t}$. In this case, Eq. (38) reads

$$
\begin{align*}
& G_{1}\left(\boldsymbol{\theta}_{t}\right) \operatorname{diag}\left(\boldsymbol{g}\left(\boldsymbol{\theta}_{t}\right) \sqrt{\frac{\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{n}}}\right) \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right) \boldsymbol{\zeta}_{t} \\
& \quad+G_{1}\left(\boldsymbol{\theta}_{t}\right) \operatorname{diag}\left(\sqrt{\frac{\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{n}}}\right) \boldsymbol{\zeta}_{t} \\
& =\sqrt{\frac{2}{\beta_{t} h} G_{2}\left(\boldsymbol{\theta}_{t}\right)} \boldsymbol{\zeta}_{t} \tag{40}
\end{align*}
$$

Thus, substituting $G_{2}(\boldsymbol{\theta})=G_{1}(\boldsymbol{\theta}) / h$, we obtain the appropriate number of shots as follows:

$$
\begin{align*}
& \boldsymbol{n} \\
= & \left\lceil\frac{\beta_{t} h^{2}}{2} G_{1}\left(\boldsymbol{\theta}_{t}\right)\left(1+\boldsymbol{g}\left(\boldsymbol{\theta}_{t}\right) \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)\right)^{2} \boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)\right] \text {. } \tag{41}
\end{align*}
$$

In particular, by neglecting the noise in $\boldsymbol{v}_{t-1}$ of the previous iteration, $G_{1}$ of the RMSprop preconditioner actually satisfies Eq. (39) since it is given by an estimate of the squared gradient. For RMSprop preconditioner, we have

$$
\begin{align*}
& \hat{G}_{1}\left(\boldsymbol{\theta}_{t}\right) \\
\approx & \operatorname{diag}\left(\left(\sigma \boldsymbol{v}_{t-1}+(1-\sigma) \hat{\mathbf{f}}\left(\boldsymbol{\theta}_{t}\right)_{\boldsymbol{s}}^{2}\right)^{-\frac{1}{4}}\right) \\
\approx & G_{1}\left(\boldsymbol{\theta}_{t}\right)\left(I+2(1-\sigma) \operatorname{diag}\left(\frac{\nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{v}_{t}} \sqrt{\frac{\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{n}}}\right) \boldsymbol{\zeta}_{t}\right)^{-\frac{1}{4}} \\
\approx & G_{1}\left(\boldsymbol{\theta}_{t}\right)\left(I-\frac{(1-\sigma)}{2} \operatorname{diag}\left(\frac{\nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{v}_{t}} \sqrt{\frac{\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)}{\boldsymbol{n}}}\right) \boldsymbol{\zeta}_{t}\right) \tag{42}
\end{align*}
$$

where we use the Taylor expansion and neglect the terms of sub-leading order. Hence, $\boldsymbol{g}\left(\boldsymbol{\theta}_{t}\right)=-0.5(1-$ $\sigma) \nabla \tilde{L}_{t}\left(\boldsymbol{\theta}_{t}\right) / \boldsymbol{v}_{t}$ holds. Then, estimating the gradient using the number of shots given by Eq. (41), we apply parameter update rule (16) of Santa. We call this optimization algorithm SantaQlaus, which stands for Stochastic
AnNealing Thermostats with Adaptive momentum and Quantum-noise Leveraging by Adjusted Use of Shots. A small number of shots is sufficient when the temperature is high, and as the optimization proceeds and the temperature is lowered, the number of shots required is increased. This allows us to efficiently leverage QSN to explore the parameter space in accordance with the Langevin diffusion while saving the number of shots. Though the exact evaluation of $\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)$ is infeasible, we can estimate it by computing the mini-batch average of an estimator of $\boldsymbol{S}\left(\boldsymbol{x}_{\boldsymbol{i}_{l}}, \boldsymbol{\theta}\right)$. For linear loss function, we can estimate $\boldsymbol{S}\left(\boldsymbol{x}_{\boldsymbol{i}_{l}}, \boldsymbol{\theta}\right)$ by the unbiased variance. For general polynomial loss functions, we can estimate it via corresponding U-statistics. In particular, it is straightforward to calculate the U-statistics to estimate Eq. (29) and (31) for quadratic loss functions including MSE. However, we should note that an unbiased estimator given by a Ustatistic is not guaranteed to be positive for general cases. If an estimate of $\boldsymbol{S}\left(\boldsymbol{x}_{\boldsymbol{i}_{l}}, \boldsymbol{\theta}\right)$ is negative, we cannot use it to calculate the number of shots. In such a case, when the obtained estimator takes a negative value, we switch to use a biased estimator guaranteed to be positive, which is obtained by simply substituting corresponding sample means (unbiased variance) to expectation values (variance). In addition, we use the exponential moving average as estimates of the quantities for the next iteration similarly to $[40,50]$. Of course this approach is subject to errors resulting from taking the ceil and estimation errors, but the thermostat absorbs their effects, as seen in the previous section. We remark that the overhead of estimating $\boldsymbol{S}\left(\boldsymbol{\theta}_{t}\right)$ is only classical calculation of the statistics using the same samples of quantum outputs as those used in estimating the gradient. Santa algorithm is especially a good fit for leveraging QSN this way. This strategy can be straightforwardly applied to other VQAs without input data such as VQE, not only limited to $\mathrm{QML}$.

We note that the variance of QSN cannot exceed that at the minimum number of shots. An option to address this constraint, as well as the variance estimation error, is to inject noise with the missing variance. Although this approach might potentially enhance the ergodicity of the dynamics, our numerical experiments did not validate the benefit of this option, as it primarily delayed the optimization. The pitfalls seem to eclipse the advantages, given there is no necessity for precise sampling at the outset and the appropriate variance of the noise to inject is challenging to estimate accurately.

The convergence theorem towards the global optima under mild conditions for Santa [64] remains valid for SantaQlaus. Even if the actual variance of QSN does not exactly yield thermal noise $\frac{2}{\beta_{t}} G_{2}\left(\boldsymbol{\theta}_{t}\right) \boldsymbol{\zeta}_{t}$ to be emulated, we can redefine $G_{2}(\boldsymbol{\theta})$ so that thermal noise with it coincides with actual QSN. In this way, we obtain the same SDE as the original Santa with this redefined $G_{2}(\boldsymbol{\theta})$. Thus, we can apply convergence theorem [64, Theorem 2] to SantaQlaus to guarantee its convergence. Moreover, even though the approximation by the cen-
tral limit theorem might be imprecise due to the limited shots in early iterations, regarding an iteration with a sufficient number of shots as the starting point ensures that the asymptotic convergence behavior remains unaffected. Practically speaking, the sampling accuracy in initial iterations does not significantly influence the optimization.

Another remark is drawn for the application of SantaQlaus to quadratic loss functions. As it is impractical to set the ratio $\kappa_{j}=s_{j} / s_{d}$ in Eq. (29) a priori, we instead enforce $\kappa_{j} \leq 1$ and use an upper bound $\bar{S}_{j}(\boldsymbol{x}, \boldsymbol{\theta}):=$ $\mu_{2}^{2}\left(\sigma_{\boldsymbol{x}, \boldsymbol{\theta}+\frac{\pi}{2} \boldsymbol{e}_{j}}^{2}+\sigma_{\boldsymbol{x}, \boldsymbol{\theta}-\frac{\pi}{2}}^{2} \boldsymbol{e}_{j}\right)+\mu_{1}^{2} w^{2} a_{2}(\boldsymbol{x})^{2} \sigma_{\boldsymbol{x}, \boldsymbol{\theta}}^{2} \geq S_{j}(\boldsymbol{x}, \boldsymbol{\theta})$ to determine $s_{j}$. We can do so by first computing $s_{j}$ using $\bar{S}_{j}(\boldsymbol{x}, \boldsymbol{\theta})(j \leq d-1)$, and $s_{d}$ via $S_{d}(\boldsymbol{x}, \boldsymbol{\theta})$. Then, we use $\max \left\{s_{1}, \cdots, s_{d}\right\}$ shots to measure $h(\boldsymbol{x})$ at the unshifted parameter $\boldsymbol{\theta}$. In this way, we can adjust the variance of QSN to be at least smaller than that of the desired thermal noise.

A practical implementation of our SantaQlaus algorithm is summarized in Algorithm 1. Here, we reparameterize as $\eta=h^{2}, \boldsymbol{u}=\sqrt{\eta} \boldsymbol{p}$, and $\operatorname{diag}(\boldsymbol{\alpha})=\sqrt{\eta} \boldsymbol{\Xi}$ as in Ref. $[64,108]$. We do not use the update rule of the refinement stage because we cannot make QSN exactly zero in contrast to the classical case. Instead, we can incorporate the refinement stage into the annealing schedule of $\beta_{t}$ in such a way that a large enough value of $\beta_{t}$ is used in the refinement. We detail an example of such a strategy later. One obstacle in an implementation of Santa's update rule is the terms with $\nabla G_{1}(\boldsymbol{\theta})$ and $\nabla G_{2}(\boldsymbol{\theta})$ in SDE (15). Exact calculation of $\nabla G_{1}(\boldsymbol{\theta})$ and $\nabla G_{2}(\boldsymbol{\theta})$ is infeasible in general. As shown in Ref. [64], one approach is to approximate them by applying Taylor expansion with respect to the parameter update. This approach actually yields a computationally efficient approximation. However, in practice, this approximation can be numerically unstable due to the time discretization and rapid changes in some components of the preconditioner. As a remedy, it was empirically found beneficial to simply drop out the terms with $\nabla G_{1}$ and $\nabla G_{2}$ [64]. Even if this is done, slightly biased samples do not affect the optimization so much as our purpose is not to obtain the accurate sampling from the target distribution. In fact, such a small bias is absorbed into the error in the estimated stochastic noise. Even for a sampling task, it has been shown that the bias caused by neglecting the derivatives of the preconditioner can be controlled to be small for the RMSprop preconditioner [119]. Then, we also neglect the terms with $\nabla G_{1}$ and $\nabla G_{2}$ in a practical implementation, just as done in the classical Santa [64].

Additionally, we introduce a warm-up iteration number, denoted as $t_{0}$, and a component-wise scale factor, $\boldsymbol{g}_{t}$, for the preconditioner to enhance flexibility. During the very early iterations, the estimation for quantities such as $\boldsymbol{S}$ and $\boldsymbol{G}$ can be unstable, with some components disproportionately large. Incorporating such unstable values into the moving average can be detrimental. Consequently, the number of shots $\boldsymbol{n}$ computed from these

```
Algorithm 1. SantaQlaus (a practical implementa-
```

tion). The function $i \operatorname{Evaluate}(\boldsymbol{\theta}, s, m)$ evaluates the
mini-batch gradient estimator $\mathbf{f}=\hat{\mathbf{f}}(\boldsymbol{\theta})_{s}$ for the ob-
jective loss function with a size $m$ mini-batch, using
the number of shots determined by $s$. This function
also returns an estimator $\boldsymbol{S}$ of $\boldsymbol{S}(\boldsymbol{\theta})$ obtained by tak-
ing mini-batch average of the corresponding U-statistic
computed from the measurement outcomes. The reg-
ularization term (or a prior) can also be specified in
iEvaluate. The function $s$ Count $(s, m)$ returns the to-
tal number of shots expended in iEvaluate $(\boldsymbol{\theta}, s, m)$.
For example, for a linear loss function whose gradi-
ent is computed by two-point parameter shift rule (8),
$s \operatorname{Count}(s, m)=2 m \sum_{i} s_{i}$. We can also straightfor-
wardly apply this algorithm to a VQA without data
dependence such as VQE just by neglecting the argu-
ment $m$ and setting $m=1$.
Input: Learning rate $\eta_{t}$, initial parameter $\boldsymbol{\theta}_{0}$, minimum
number of shots $s_{\min }$, total shot budget $s_{\text {max }}$ available
for the optimization, annealing schedule $\beta_{t}$, mini-batch
size $m_{t}$, scale factor of the preconditioner $\boldsymbol{g}_{t}$, warm-up it-
eration number $t_{0}$, hyperparameters $\sigma, C$, and $\lambda$, running
average constant $\mu$
: Initialize: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}_{0}, \quad t \leftarrow 1, s_{\text {tot }} \leftarrow 0, \boldsymbol{s} \leftarrow$
$\left(s_{\text {min }}, \cdots, s_{\text {min }}\right)^{\mathrm{T}}, \boldsymbol{\xi}^{\prime} \leftarrow(0, \cdots, 0)^{\mathrm{T}}, \boldsymbol{\chi}^{\prime} \leftarrow(0, \cdots, 0)^{\mathrm{T}}$,
$\boldsymbol{\Gamma}^{\prime} \leftarrow(0, \cdots, 0)^{\mathrm{T}}, \boldsymbol{v} \leftarrow(0, \cdots, 0)^{T}, \boldsymbol{u} \leftarrow \sqrt{\eta_{1}} \mathcal{N}(0, I)$,
$\alpha \leftarrow \sqrt{\eta_{1}} C$
while $s_{\text {tot }} \leq s_{\max }$ do
$\mathbf{f}, \boldsymbol{S} \leftarrow i \operatorname{Evaluate}\left(\boldsymbol{\theta}, s, m_{t}\right)$
$s_{\text {tot }} \leftarrow s_{\text {tot }}+s \operatorname{Count}\left(\boldsymbol{s}, m_{t}\right)$
$\boldsymbol{v} \leftarrow \sigma \boldsymbol{v}+(1-\sigma) \mathbf{f}^{2}$
$\boldsymbol{G} \leftarrow \boldsymbol{g}_{t} / \sqrt{\lambda+\sqrt{\boldsymbol{v}}}$
$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\boldsymbol{G} \boldsymbol{u} / 2$
$\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha}+\left(\boldsymbol{u}^{2}-\eta_{t} / \beta_{t}\right) / 2$
$\boldsymbol{u} \leftarrow \exp (-\boldsymbol{\alpha} / 2) \boldsymbol{u}$
$\boldsymbol{u} \leftarrow \boldsymbol{u}-\eta_{t} \boldsymbol{G} \mathbf{f}$
$\boldsymbol{u} \leftarrow \exp (-\boldsymbol{\alpha} / 2) \boldsymbol{u}$
$\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha}+\left(\boldsymbol{u}^{2}-\eta_{t} / \beta_{t}\right) / 2$
$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\boldsymbol{G} \boldsymbol{u} / 2$
$t \leftarrow t+1$
if $t>t_{0}$ then
$\boldsymbol{\xi}^{\prime} \leftarrow \mu \boldsymbol{\xi}^{\prime}+(1-\mu) \boldsymbol{S}$
$\boldsymbol{\xi} \leftarrow \boldsymbol{\xi}^{\prime} /\left(1-\mu^{t-t_{0}}\right)$
$\chi^{\prime} \leftarrow \mu \chi^{\prime}+(1-\mu) \mathbf{f}$
$\chi \leftarrow \chi^{\prime} /\left(1-\mu^{t-t_{0}}\right)$
$\boldsymbol{\Gamma}^{\prime} \leftarrow \mu \boldsymbol{\Gamma}^{\prime}+(1-\mu) \boldsymbol{G}$
$\boldsymbol{\Gamma} \leftarrow \boldsymbol{\Gamma}^{\prime} /\left(1-\mu^{t-t_{0}}\right)$
$\boldsymbol{v}^{\prime} \leftarrow \sigma \boldsymbol{v}+(1-\sigma) \boldsymbol{\chi}$
$\gamma \leftarrow\left(1-0.5(1-\sigma) \chi^{2} / \boldsymbol{v}^{\prime}\right)^{2}$
$\boldsymbol{n} \leftarrow\left\lceil\beta_{t} \eta_{t} \boldsymbol{\Gamma} \boldsymbol{\xi} / 2\right\rceil$
$s \leftarrow\left\lceil\boldsymbol{n} / m_{t}\right\rceil$
$s \leftarrow \operatorname{clip}\left(s, s_{\text {min }}\right.$, None $)$
else
$\boldsymbol{s} \leftarrow\left(s_{\text {min }}, \cdots, s_{\text {min }}\right)^{\mathrm{T}}$
end if
end while

early estimates may be erratic. To counteract this, it
is advantageous to disregard these values in a few early iterations. We might choose a small value for $t_{0}$, such as 5 . While the default for $\boldsymbol{g}_{t}$ can be set to 1 , when components have varying optimal scales, adjusting the learning rate individually for each component is beneficial. For instance, in regression tasks of QML with a scaled observable, as discussed in Sec. V B, allowing the scale parameter to adjust quickly is favorable as it significantly influences the success of the regression. However, introducing a component-wise different learning rate $\boldsymbol{\eta}_{t}$ is not justified because $\eta_{t}$ represents time. We empirically found that such modifications degrade the performance.

As the shot budget is important, schedules of hyperparameter settings of such as $\beta_{t}, \eta_{t}$, or $m_{t}$ based on the number of shots may be useful. For instance, we use the following function of the expended shots $s_{\text {tot }}$ to determine the value of the hyperparameters to be used:

$f_{s_{0}, s_{\text {end }}}^{y_{0}, y_{\text {end }}, a}\left(s_{\text {tot }}\right):=y_{0}\left[\frac{s_{\text {tot }}-s_{0}}{s_{\text {end }}-s_{0}}\left(\left(\frac{y_{\text {end }}}{y_{0}}\right)^{\frac{1}{a}}-1\right)+1\right]^{a}$.

This function takes the predetermined values $y_{0}$ and $y_{\text {end }}$ at the start $s_{0}$ and the end $s_{\text {end }}$ respectively, and the curve of the growth is controlled by $a \neq 0$.

## C. Incorporating quantum error mitigation

In the context of VQAs executed on NISQ devices, addressing hardware noise is important. Quantum error mitigation (QEM) techniques offer a pathway for estimating the output of an ideal, noiseless circuit from noisy ones [126-138], without large resource overhead required in quantum error correction. In principle, it is possible to consider the noisy ansatz instead as our model. However, to obtain an accurate result, we finally need to remove the effects of the hardware noise at least for the evaluation of the final outputs after the optimization, highlighting the need for QEM in real devices $[6,133,139]$. Thus, we should aim to optimize the parameters with respect to the noiseless model. While some VQA tasks exhibit resilience to hardware noise $[57,140,141]$, the presence of such noise can introduce computational bias in gradient estimation, thereby altering the loss landscape and complicating optimization in general [22, 141]. Prior research indicates that employing QEM during optimization can potentially enhance the performance [13, 142-144], although QEM is unlikely to resolve the noise-induced barren plateau [22, 142]. However, it is not always the case, as the application of QEM inherently introduces a trade-off: while it reduces bias, it increases the sampling variance $[142,145-148]$. For instance, probabilistic error cancellation (PEC) effectively applying the inverse map of the noise channel via sampling the circuits according to a quasi-probability decomposition of the inverse map [128, 129, 144]. As a result, it yields an unbiased estimator of the noiseless expectation value at the expense of increased variance. Zero noise extrapolation (ZNE) $[126,128,129]$ also yields a bias-reduced estimator with the increased variance.

The integration of QEM techniques into SantaQlaus may offer a resource-efficient use of these methods during optimization. For example, PEC can be seamlessly incorporated into the SantaQlaus framework. PEC provides an unbiased estimator for the gradient, enabling optimization of the noiseless model effectively [143]. The distinction lies in the increased number of circuit terms that must be sampled, as dictated by the quasi-probability decomposition of the inverse noise map. In SantaQlaus, an appropriate total number of measurement shots can be computed from the gradient estimator's variance $\boldsymbol{S}(\boldsymbol{\theta})$, obtained from the PEC samples. These shots are then allocated based on the quasi-probability distribution. This approach enables SantaQlaus to operate using an unbiased gradient in the presence of hardware noise via PEC, while also automatically adjusting resource usage for efficiency. In a sense, SantaQlaus may also leverage hardware noise that is "converted" to sampling noise with increased variance via $\mathrm{QEM}$.

Similarly, ZNE can be incorporated into SantaQlaus to provide a bias-reduced estimator of the gradient. Although the estimator is not perfectly unbiased in ZNE, such a bias-reduced estimator may still offer satisfactory performance in optimizing variational parameters, as observed in an experiment [13]. Other QEM methods such as Clifford data regression [130] can be incorporated in a similar manner. This way, SantaQlaus has a potential to utilize QEM in a resource-efficient manner.

## V. NUMERICAL SIMULATIONS

In this section, we demonstrate the performance of SantaQlaus optimizers against a range of existing optimizers through numerical simulations of a benchmark VQAs. For the optimization process, we simulate the sampling of the outcomes of quantum measurements of loss functions via Qulacs [149]. The resulting optimization curves are plotted based on the exact evaluations via state vector calculations provided by Qulacs. We do not include hardware noise.

We benchmark SantaQlaus against several other optimization algorithms. As a representative of generic optimizers, we employ Adam [35], renowned for its broadly effective performance. To test the advantages of adaptive shots-adjusting strategy of SantaQlaus over simpler methods, we use Adam with a predetermined increased number of shots, denoted as Adam with dynamic shots (Adam-DS). In addition, we compare SantaQlaus with existing shot-adaptive optimizers, such as gCANS [50] and its QML-specific variant, Refoqus [62].

As another optimizer that aims for global optimality with MCMC approach, we pick up MCMC-VQA [58]. It is based on Metropolis-Hastings steps using injected stochastic noise, which can be resource-intensive to
achieve sufficient mixing. The shot resource efficiency of MCMC-VQA has not been comprehensively studied. We also compare SantaQlaus with MCMC-VQA in a VQE task (Sec. V A).

We fix a part of the hyperparameters to recommended values throughout the benchmarks. For SantaQlaus, we use $\sigma=0.99, C=5, \lambda=10^{-8}, \mu=0.99, \eta_{1}=0.01$, $s_{\min }=4$, and $t_{0}=5$. For Adam, $\beta_{1}=0.9, \beta_{2}=0.99$, $\epsilon=10^{-8}$ are used. In MCMC-VQA, we use the inverse temperature $\beta=0.2$ and the noise parameter $\xi=0.5$ as recommended in Ref. [58]. We do not use artificial noise injection in SantaQlaus since we could not find any improvements. For each optimizer, we choose the best hyperparameters in a grid search. For all scheduled hyperparameters, we use the function $f_{s_{0}, y_{\text {end }}, a}^{y_{0}, y_{\text {end }}, a}\left(s_{\text {tot }}\right)$ given in Eq. (43) to assign the value according to the expended shots. For the performance with respect to the shots resource usage, we empirically found this strategy is better than the usual one based on a function of the number of the iteration.

We use a gradually decreasing learning rate $\eta_{t}=$ $f_{0, s_{\max }}^{\eta_{1}, \eta_{\text {end }}, a_{\text {LR }}}\left(s_{t}\right)$, where $s_{t}$ denotes $s_{\text {tot }}$ up to $t$-th iteration. For gCANS and Refoqus, the fixed learning rate $1 / L$ is used with the Lipschitz constant $L$ of the gradient. Throughout the benchmarks and the other optimizers, we use $\eta_{1}=0.01$ and $\eta_{\text {end }}=0.001$. The exponent $a_{\mathrm{LR}}$ is chosen for each benchmark and each optimizer.

As for the annealing schedule of SantaQlaus, we employ two-fold stages corresponding to burn-in and refinement, though the update rules remain unchanged. We set the number of burn-in shots $s_{\mathrm{b}}$ such that the stage switches when that number of shots is used. From the start until $s_{\mathrm{b}}$ shots used, $\beta_{t}$ is given by $\beta_{t}=f_{0, s_{\mathrm{b}}}^{\beta_{0}, \beta_{\mathrm{b}}, a_{1}}\left(s_{t}\right)$ with parameters $\beta_{0}, \beta_{\mathrm{b}}$, and $a_{1}$ which determine the schedule, where $s_{t}$ is the total number of shots used before $t$-th iteration. We use $\beta_{0}=10$ throughout the simulations. Then, we use another schedule $f_{s_{\mathrm{b}}, \beta_{\mathrm{max}}}^{\beta_{\mathrm{r}}, a_{\mathrm{r}}, a_{2}}\left(s_{t}\right)$ for the refinement stage. In addition, we scale the inverse temperature as $\beta_{t}=f_{s_{\mathrm{b}}, s_{\mathrm{m}}}^{\beta_{\mathrm{b}}, a_{2}}\left(s_{t}\right) /\left(r \eta_{t}\right)$ in the refinement stage, in order to avoid decrease in the number of shots when $\eta_{t}$ is decreased. We use $r=100$.

## A. VQE of 1-dimensional Transverse field Ising model

As a benchmark, we begin by a VQE task of the 1dimensional transverse field Ising spin chain with open boundary conditions for the system size $N=6$ and 12 , where the Hamiltonian is given as

$$
\begin{equation*}
H=-J \sum_{i=1}^{N-1} Z_{i} Z_{i+1}-g \sum_{i=1}^{N} X_{i} \tag{44}
\end{equation*}
$$

Our goal is to obtain the ground state energy by minimizing the expected energy as the loss function. We consider the case with $g / J=1.5$. It is obvious that the interaction term $\sum_{i=1}^{N-1} Z_{i} Z_{i+1}$ or the transverse field term $\sum_{i=1}^{N} X_{i}$

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-13.jpg?height=629&width=830&top_left_y=171&top_left_x=1103)

FIG. 1. The parameterized quantum circuit used in the numerical simulation of the VQE task of a 1dimensional transverse Ising spin chain.

can be measured at once. We employ WDS to allocate the number of shots for them, as they form only two groups with similar weights. If $s$ shots are used in total, we deterministically allocate $s J(N-1) / M$ to the former and $\operatorname{sg} N / M$ to the latter, where $M=J(N-1)+g N$. The results are shown in terms of the precision of the per-site energy $\Delta E /(J N)$, where $\Delta E$ denotes the difference of the exact energy expectation value evaluated at the parameters obtained by the optimization from the groundstate energy. For this problem, we used the ansatz shown in Fig. 1 with $D=3$ following [40].

The results depicted in Fig. 2 shows the performance of various optimizers in the context of VQE tasks for 1d transverse Ising spin chains with open boundary conditions. The median of the loss function and the interquartile range (IQR), i. e. the range between the first and third quartiles, are displayed. Across scenarios with $N=6$ and $N=12$, SantaQlaus consistently demonstrates superior performance relative to the other optimizers.

For SantaQlaus, we employ $s_{\mathrm{b}}=0.8 s_{\max }, \beta_{\mathrm{b}}=\beta_{\mathrm{r}}=$ $10^{4}, a_{1}=a_{2}=5$, and $a_{\mathrm{LR}}=0.5$ for both $N=6$ and $N=12$. The learning rate exponent $a_{\mathrm{LR}}=0.1$ is used for Adam. In Adam-DS, the number of shots is gradually increased from 4 to 100 (500) according to the function (43) with $a=10$ for $N=6(N=12)$. For MCMCVQA, MCMC stage is chosen as $0.4 s_{\max }$, which is the best among $0.4 s_{\max }, 0.6 s_{\max }$, and $0.8 s_{\max }$. The learning rate exponent $a_{\mathrm{LR}}=0.3$ is used.

For $N=6$, we include Adam with fixed number of shots, denoted by Adam10 and Adam100 with each used number of shots. As a result of the grid search, 10 is the best for the fixed number of shots among 10, 50, 100, 500, and 1000. Adam-DS with tuned hyperparameters performs better than them, which implies a merit of simple dynamic shot-number increasing. SantaQlaus is even faster than that and achieves better accuracy, indicating
![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-14.jpg?height=572&width=1826&top_left_y=164&top_left_x=168)

FIG. 2. Comparison of performance of the optimizers for the VQE tasks of $1 \mathrm{~d}$ transverse Ising spin chain with open boundary conditions. Every graph shows the median (solid curve) and the IQR (the highlighted region) of 20 runs of the optimizations with different random initial parameters. The exact expectation values are plotted to indicate the performance. (a) $N=4$. The per-site ground energy approximation precision $\Delta E /(J N)$ vs number of shots. (b) $N=12$. The value $\Delta E /(J N)$ vs number of shots.

benefits beyond simple dynamic shot strategies.

For $N=12$, SantaQlaus attains the best median precision again. While Adam-DS sometimes achieves comparable precision with SantaQlaus, it gets to poor local minima in some trials, as seen from its widespread of the IQR. In stark contrast, SantaQlaus reliably maintains high precision, underscored by its narrower interquartile spread.

Despite its rapid initial progress, gCANS gets stuck in local minima at an early stage of optimization. One contributing factor is that when the parameters enter a local mode or reach a stationary point, the shot-allocation rule of gCANS prescribes a large number of shots. This is because the number of shots in gCANS is inversely proportional to the norm of the gradient. MCMC-VQA does not seem to achieve sufficient mixing within this shot budget to exhibit the efficacy of MCMC sampling.

Overall, these results show that SantaQlaus is more efficient and effective in exploring the loss landscape compared to other optimizers, consistently finding better solutions.

## B. Benchmark regression task

We next test SantaQlaus on a QML regression task investigated in Ref. [19]. A QNN used here consists of feature encoding unitary $U(\boldsymbol{x})$ and the trainable unitary $V(\boldsymbol{\theta})$ which form the model state $|\psi(\boldsymbol{x} ; \boldsymbol{\theta})\rangle=$ $V(\boldsymbol{\theta}) U(\boldsymbol{x})|0\rangle$. Given some set $\mathcal{D}=\left\{\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{N}\right\}$ of input data vectors, the label $g(\boldsymbol{x})$ of each data $\boldsymbol{x}$ is generated by the model with a randomly chosen target parameter $\boldsymbol{\theta}^{*} \in[0,2 \pi)$ as

$$
\begin{equation*}
g(\boldsymbol{x})=w^{*}\left\langle\psi\left(\boldsymbol{x} ; \boldsymbol{\theta}^{*}\right)\left|Z_{1}\right| \psi\left(\boldsymbol{x} ; \boldsymbol{\theta}^{*}\right)\right\rangle \tag{45}
\end{equation*}
$$

where $w^{*}$ is a scale factor that sets the standard deviation of the labels to 1 over the dataset, and $Z_{1}$ denotes the
Pauli $Z$ on the first qubit. Our goal is to do a regression of these labels by $w\left\langle\psi\left(\boldsymbol{x} ; \boldsymbol{\theta}^{*}\right)\left|Z_{1}\right| \psi\left(\boldsymbol{x} ; \boldsymbol{\theta}^{*}\right)\right\rangle$ via tuning $w$ and $\boldsymbol{\theta}$ without knowing the target parameters. We use the MSE as the loss function. As the correct parameters exist, the representability of the ansatz does not matter in this task.

As the input data, we use a dimensionality-reduced feature vectors of fashion MNIST [150] via principal component analysis following Ref. [19]. We use $M=1100$ data points as the whole dataset. In each training, the size of the training dataset is 880 and the test data size is 220. We use the same feature encoding circuit $U(\boldsymbol{x})$ and trainable circuit $V(\boldsymbol{\theta})$ as Ref. [19]. The feature encoding is similar to the one proposed in Ref. [13], which is given as:

$$
\begin{equation*}
U(\boldsymbol{x})\left|0^{\otimes N}\right\rangle=U_{z}(\boldsymbol{x}) H^{\otimes N} U_{z}(\boldsymbol{x}) H^{\otimes N}\left|0^{\otimes N}\right\rangle \tag{46}
\end{equation*}
$$

with

$$
\begin{equation*}
U_{z}(\boldsymbol{x})=\exp \left(-i \pi\left[\sum_{i=1}^{N} x_{i} Z_{i}+\sum_{j=1, j>i}^{N} x_{i} x_{j} Z_{i} Z_{j}\right]\right) \tag{47}
\end{equation*}
$$

for $N$-qubit system, where $H$ denotes the Hadamard gate. As for the trainable circuit $V(\theta)$, these are composed of $D$ layers of single-qubit rotations $R\left(\theta_{i, j}\right)$ on each of the qubits, interlaid with $C Z=|1\rangle\langle 1| \otimes Z$ gates between nearest neighbours in the circuit, where $R\left(\theta_{i, j}\right)=$ $R_{X}\left(\theta_{i, j, 0}\right) R_{Y}\left(\theta_{i, j, 1}\right) R_{Z}\left(\theta_{i, j, 2}\right)$. We implemented simulations for the case of $N=4, D=10$ (133 parameters), and $N=10, D=2$ (91 parameters). We employ $s_{\mathrm{b}}=0.5 s_{\max }, \beta_{\mathrm{b}}=5000(500), \beta_{\mathrm{r}}=10^{4}\left(10^{3}\right)$, and $a_{1}=a_{2}=3$ for $N=4(N=10)$. The learning rate exponent $a_{\mathrm{LR}}=0.3$ is used for both SantaQlaus and Adam-DS. In Adam-DS, the number of shots is gradually increased from 4 to 100 according to the function (43) with $a=2$. For both optimizers, the batch size is

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-15.jpg?height=1124&width=1827&top_left_y=159&top_left_x=165)

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-15.jpg?height=545&width=889&top_left_y=172&top_left_x=171)

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-15.jpg?height=556&width=863&top_left_y=725&top_left_x=181)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-15.jpg?height=544&width=870&top_left_y=183&top_left_x=1113)

(d)

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-15.jpg?height=505&width=840&top_left_y=772&top_left_x=1117)

FIG. 3. Comparison of performance of the optimizers for the regression task. Every graph shows the median (solid curve) and the IQR (the highlighted region) of 20 trials of the regression with different training data and the initial parameters. The MSE for the prediction obtained by the exact expectation values is plotted to indicate the performance. (a) 4-qubit 10 layers. MSE for train data vs number of shots. (b) 4-qubit 10 layers. MSE for test data vs number of shots. (c) 10-qubit 2 layers. MSE for train data vs number of shots. (d) 10-qubit 2 layers. MSE for test data vs number of shots.

gradually increased from 2 to 16 according to the function (43) with $a=2$.

The performance of different optimizers for the regression task is shown in Fig. 3. Clearly, SantaQlaus demonstrates superior performance compared to other optimizers, achieving the lowest median MSE for both training and test data. Regarding the 4 -qubit case shown in (a) and (b) of Fig. 3, while Adam-DS displays some optimization progression, SantaQlaus consistently achieves lower MSE, indicating its enhanced accuracy and efficient use of shot resources. Turning to the 10 -qubit scenario in graphs (c) and (d) of Fig. 3, the optimization landscape appears notably challenging. Here, Refoqus is unable to escape from a plateau without showing significant improvement as the number of shots increases. Adam-DS, while presenting some improvement in the median MSE, has a large upper quartile, suggesting that it often becomes trapped in less optimal regions of the optimization landscape. In contrast, SantaQlaus consistently delivers a lower median MSE and exhibits a decreasing upper quartile, highlighting its robust capability in navigating and optimizing even in such challenging landscapes.

## C. Classification of Iris dataset with local depolarizing noise

Finally, we test SantaQlaus on a classification task using the Iris dataset [151]. The Iris dataset comprises three classes of Iris flowers, with 50 samples per class, and includes four features for each sample. We use the normalized input data $\boldsymbol{z}$ given by $z_{i, j}=\left(x_{i, j}-\right.$ $\left.\min _{k}\left(x_{k, j}\right)\right) /\left(\max _{k}\left(x_{k, j}\right)-\min _{k}\left(x_{k, j}\right)\right)$, where $x_{i, j}$ denotes the $j$-th feature of the $i$-th data point. We randomly select 120 data points used for the training. Then, the remaining 30 data points are used as the test data.

We employ the ansatz $|\psi(\boldsymbol{x} ; \boldsymbol{\theta})\rangle$ given by the same feature encoding and trainable circuits as those for the previous QML benchmark in Sec. VB with $N=4$ and $D=4$. We use $Z_{1} \otimes Z_{2}$ as the observable to be measured, which yields two bits $\left(b_{1}, b_{2}\right)$ as an outcome. Then, we assign a label of the class $y\left(b_{1}, b_{2}\right):=b_{1}+2 b_{2}(\bmod 3)$ to each outcome. The label prediction for each data point $x$ is determined by the most frequently occurring value of $y\left(b_{1}, b_{2}\right)$, based on repeated measurements of $|\psi(\boldsymbol{x} ; \boldsymbol{\theta})\rangle$. Hence, the label value with the highest proba-
(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-16.jpg?height=515&width=898&top_left_y=211&top_left_x=169)

(c)

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-16.jpg?height=510&width=872&top_left_y=772&top_left_x=171)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-16.jpg?height=510&width=878&top_left_y=214&top_left_x=1103)

(d)

![](https://cdn.mathpix.com/cropped/2024_06_04_80a0a9730f39f5f427afg-16.jpg?height=499&width=857&top_left_y=775&top_left_x=1103)

FIG. 4. Comparison of performance of the optimizers for the classification of the Iris dataset. Every graph shows the median (solid curve) and the IQR (the highlighted region) of 20 trials of the training with different training data and the initial parameters. The dashed curves depict the median of learning curves in the absence of the depolarizing noise (only shot noise). Plotted MSE and error rate are computed from the exact expectation values without hardware noise, while training are done with finite number of shots under the influence of the local depolarizing noise. (a) MSE for train data vs number of shots. (b) MSE for test data vs number of shots. (c) Error rate for train data vs number of shots. (d) Error rate for test data vs number of shots.

bility emerges as the predicted label with infinitely many measurements. As the loss function to be minimized, we use the MSE of the success probability given as

$$
\begin{equation*}
L(\boldsymbol{\theta})=\frac{1}{M} \sum_{i=1}^{M}\left(1-p\left(\boldsymbol{x}_{i}, \boldsymbol{\theta}\right)\right)^{2} \tag{48}
\end{equation*}
$$

where $p\left(\boldsymbol{x}_{i}, \boldsymbol{\theta}\right)$ denotes the probability of obtaining the correct label for the data point $\boldsymbol{x}_{i}$ with the model parameter $\boldsymbol{\theta}$. This is the same as the mean squared failure probability. Because this is MSE, we obtain unbiased estimators of the gradient and its variance as we have seen in Sec. II A and IV A. That is the reason of this choice of the loss function in this benchmark.

In this simulation, hardware noise is incorporated, specifically modeled as local depolarizing noise. To represent this, a single-qubit (two-qubit) depolarizing channel is introduced after every single-qubit (two-qubit) gate, with the exception of $Z$-rotation gates. The error probabilities assigned are $10^{-3}$ for single-qubit gates and $10^{-2}$ for two-qubit gates.
As for the hyperparameters specific to this benchmark, we employ $s_{\mathrm{b}}=0.5 s_{\max }, \beta_{\mathrm{b}}=10^{4}, \beta_{\mathrm{r}}=5 \times 10^{4}$, and $a_{1}=a_{2}=3$ for SantaQlaus. The learning rate exponent $a_{\mathrm{LR}}=0.3$ is used for both SantaQlaus and Adam-DS. In Adam-DS, the number of shots is gradually increased from 4 to 10 according to the function (43) with $a=3$. For both optimizers, the batch size is gradually increased from 2 to 16 according to the function (43) with $a=1$.

The performance of different optimizers for this classification task is shown in Fig. 4. Here, the error rate is defined as the proportion of incorrectly predicted labels to the total number of data points. Both MSE and the error rate are calculated by the exact expectation values in the absence of noise, to evaluate the achievable performance of the obtained model. To account for misclassification arising from statistical errors, we assume a worst-case scenario. In this scenario, a data point $\boldsymbol{x}_{i}$ is considered misclassified if the difference $\Delta p\left(\boldsymbol{x}_{i}\right)$ between the highest and the second-highest label probabilities is less than $2 \epsilon$, where $\epsilon$ is a positive constant. This approach is relevant because if the statistical error in probability
distribution estimation exceeds $\epsilon$ and $\Delta p\left(\boldsymbol{x}_{i}\right)$ is smaller than $2 \epsilon$, there is a risk of predicting an incorrect label, even when the highest probability corresponds to the correct label. To estimate the probability within the precision of $\epsilon$, we need $1 / \epsilon^{2}$ times sampling overhead for both the direct estimation using a quantum device, and the acquisition of a classical surrogate [20]. Here, we choose $\epsilon=10^{-2}$ corresponding to the sampling overhead $10^{4}$.

According to the resulting learning curves in Fig. 4, SantaQlaus demonstrates the highest performance even in the presence of the local depolarizing noise. For both the MSE and the error rate, we can clearly see that the ability of SantaQlaus to efficiently exploring better optimal parameters surpasses the others. Notably, our results imply that SantaQlaus is much more robust to the hardware noise than the other methods. Indeed, the presence of hardware noise significantly impairs the learning performance of the Adam-DS optimizer. In contrast, SantaQlaus maintains nearly consistent performance levels under the same noise. This robustness may be attributed to the efficient exploration strategy of SantaQlaus, which leverages QSN with its adaptively adjusted variance. It is suggested that this strategy is also effective in navigating challenging landscape which may be caused by hardware noise.

## VI. CONCLUSION

In this study, we introduced SantaQlaus, an optimizer designed to strategically leverage inherent QSN for efficient loss landscape exploration while minimizing the number of quantum measurement shots. The algorithm is applicable to a broad spectrum of loss functions encountered in VQAs and QML, including those that are non-linear. Incorporating principles from the classical Santa algorithm [64], SantaQlaus exploits annealed QSN to effectively evade saddle points and poor local minima. The algorithm adjusts the number of measurement shots to emulate appropriate thermal noise based on the asymptotic normality of QSN in the gradient estimator. This adjustment requires only a small classical computational overhead for variance estimation. Moreover, the update rule of our algorithm includes thermostats from Santa that provide robustness against estimation errors of the variance of QSN. SantaQlaus naturally attains resource efficiency by initiating the optimization process with a low shot count during the high-temperature early stages, and gradually increasing the shot count for more precise gradient estimation as the temperature decreases.

We have demonstrated the efficacy of SantaQlaus through numerical simulations on benchmark tasks in VQE, regression task, and a multiclass classification under the influence of the local depolarizing noise. Our optimizer consistently outperforms existing algorithms, which often get stuck in suboptimal local minima or flat regions of the loss landscape. Our results imply that compared to shot-adaptive strategies like gCANS, SantaQlaus excels in directly addressing the challenges in the loss landscape rather than merely maximizing iteration gains. SantaQlaus also exhibits advantages over basic shot-number annealing approaches like Adam-DS. Moreover, our simulation implies that SantaQlaus is robust against hardware noise, which may also highlight the efficiency of the exploration strategy of SantaQlaus leveraging QSN.

Looking ahead, additional research is needed to assess performance of SantaQlaus in experiments. The demonstrated robustness of SantaQlaus against hardware noise indicates its potential for promising results in practical experiments. This aspect warrants comprehensive investigation to fully understand and leverage its capabilities in real-world applications. Incorporating QEM techniques into SantaQlaus also offers a promising route for experimental deployments. For QNNs, combining SantaQlaus with a recent noise-aware training strategy [144] could potentially enhance robustness and efficiency under realistic conditions. Incorporating advanced preconditioning techniques, such as Fisher information, may provide further improvements. These avenues remain open for future exploration.

## ACKNOWLEDGMENTS

This work is supported by MEXT Quantum Leap Flagship Program (MEXT QLEAP) Grant Number JPMXS0120319794, and JST COI-NEXT Grant Number JPMJPF2014.
[1] J. Preskill, "Quantum Computing in the NISQ era and beyond," Quantum 2, 79 (2018).

[2] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, "Variational quantum algorithms," Nature Reviews Physics 3, 625 (2021).

[3] A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.Q. Zhou, P. J. Love, A. Aspuru-Guzik, and J. L. O'Brien, "A variational eigenvalue solver on a photonic quantum processor," Nature Communications 5, 4213
(2014).

[4] P. J. J. O'Malley, R. Babbush, I. D. Kivlichan, J. Romero, J. R. McClean, R. Barends, J. Kelly, P. Roushan, A. Tranter, N. Ding, B. Campbell, Y. Chen, Z. Chen, B. Chiaro, A. Dunsworth, A. G. Fowler, E. Jeffrey, E. Lucero, A. Megrant, J. Y. Mutus, M. Neeley, C. Neill, C. Quintana, D. Sank, A. Vainsencher, J. Wenner, T. C. White, P. V. Coveney, P. J. Love, H. Neven, A. Aspuru-Guzik, and J. M. Martinis, "Scalable quantum simulation of molecular energies," Phys. Rev. X 6,

031007 (2016).

[5] B. Bauer, D. Wecker, A. J. Millis, M. B. Hastings, and M. Troyer, "Hybrid quantum-classical approach to correlated materials," Phys. Rev. X 6, 031045 (2016).

[6] A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow, and J. M. Gambetta, "Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets," Nature 549, 242 (2017).

[7] E. Farhi, J. Goldstone, and S. Gutmann, "A Quantum Approximate Optimization Algorithm," arXiv:1411.4028 (2014).

[8] E. Farhi and A. W. Harrow, "Quantum Supremacy through the Quantum Approximate Optimization Algorithm," arXiv:1602.07674 (2016).

[9] J. S. Otterbach, R. Manenti, N. Alidoust, A. Bestwick, M. Block, B. Bloom, S. Caldwell, N. Didier, E. S. Fried, S. Hong, P. Karalekas, C. B. Osborn, A. Papageorge, E. C. Peterson, G. Prawiroatmodjo, N. Rubin, C. A. Ryan, D. Scarabelli, M. Scheer, E. A. Sete, P. Sivarajah, R. S. Smith, A. Staley, N. Tezak, W. J. Zeng, A. Hudson, B. R. Johnson, M. Reagor, M. P. da Silva, and C. Rigetti, "Unsupervised Machine Learning on a Hybrid Quantum Computer," arXiv:1712.05771 (2017).

[10] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, "Quantum circuit learning," Phys. Rev. A 98, 032309 (2018).

[11] E. Farhi and H. Neven, "Classification with Quantum Neural Networks on Near Term Processors," arXiv:1802.06002 (2018).

[12] M. Benedetti, E. Lloyd, S. Sack, and M. Fiorentini, "Parameterized quantum circuits as machine learning models," Quantum Science and Technology 4, 043001 (2019).

[13] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," Nature 567, 209 (2019).

[14] M. Schuld and N. Killoran, "Quantum machine learning in feature hilbert spaces," Phys. Rev. Lett. 122, 040504 (2019).

[15] M. Schuld and F. Petruccione, Supervised Learning with Quantum Computers, 1st ed. (Springer Publishing Company, Incorporated, 2018).

[16] M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, "Circuit-centric quantum classifiers," Phys. Rev. A 101, 032308 (2020).

[17] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, "Power of data in quantum machine learning," Nature Communications 12, 2631 (2021).

[18] M. Schuld and F. Petruccione, "Quantum models as kernel methods," in Machine Learning with Quantum Computers (Springer International Publishing, Cham, 2021) pp. 217-245.

[19] S. Jerbi, L. J. Fiderer, H. Poulsen Nautrup, J. M. Kübler, H. J. Briegel, and V. Dunjko, "Quantum machine learning beyond kernel methods," Nature Communications 14, 517 (2023).

[20] F. J. Schreiber, J. Eisert, and J. J. Meyer, "Classical surrogates for quantum learning models," Phys. Rev. Lett. 131, 100803 (2023).

[21] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neu- ral network training landscapes," Nature Communications 9, 4812 (2018).

[22] S. Wang, E. Fontana, M. Cerezo, K. Sharma, A. Sone, L. Cincio, and P. J. Coles, "Noise-induced barren plateaus in variational quantum algorithms," Nature Communications 12, 6961 (2021).

[23] A. Arrasmith, M. Cerezo, P. Czarnik, L. Cincio, and P. J. Coles, "Effect of barren plateaus on gradient-free optimization," Quantum 5, 558 (2021).

[24] A. V. Uvarov and J. D. Biamonte, "On barren plateaus and cost function locality in variational quantum algorithms," Journal of Physics A: Mathematical and Theoretical 54, 245301 (2021).

[25] K. Sharma, M. Cerezo, L. Cincio, and P. J. Coles, "Trainability of dissipative perceptron-based quantum neural networks," Phys. Rev. Lett. 128, 180505 (2022).

[26] C. Ortiz Marrero, M. Kieferová, and N. Wiebe, "Entanglement-induced barren plateaus," PRX Quantum 2, 040316 (2021).

[27] Z. Holmes, K. Sharma, M. Cerezo, and P. J. Coles, "Connecting ansatz expressibility to gradient magnitudes and barren plateaus," PRX Quantum 3, 010313 (2022).

[28] Z. Holmes, A. Arrasmith, B. Yan, P. J. Coles, A. Albrecht, and A. T. Sornborger, "Barren plateaus preclude learning scramblers," Phys. Rev. Lett. 126, 190501 (2021).

[29] L. Bittel and M. Kliesch, "Training variational quantum algorithms is np-hard," Phys. Rev. Lett. 127, 120502 (2021).

[30] E. R. Anschuetz and B. T. Kiani, "Quantum variational algorithms are swamped with traps," Nature Communications 13, 7760 (2022).

[31] M. Cerezo, A. Sone, T. Volkoff, L. Cincio, and P. J. Coles, "Cost function dependent barren plateaus in shallow parametrized quantum circuits," Nature Communications 12, 1791 (2021).

[32] A. Pesah, M. Cerezo, S. Wang, T. Volkoff, A. T. Sornborger, and P. J. Coles, "Absence of barren plateaus in quantum convolutional neural networks," Phys. Rev. X 11, 041011 (2021).

[33] K. Zhang, M.-H. Hsieh, L. Liu, and D. Tao, "Toward Trainability of Deep Quantum Neural Networks," arXiv:2112.15002 (2021).

[34] T. L. Patti, K. Najafi, X. Gao, and S. F. Yelin, "Entanglement devised barren plateau mitigation," Phys. Rev. Res. 3, 033090 (2021).

[35] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," Proceedings of the 3rd International Conference on Learning Representations (2014).

[36] J. A. Nelder and R. Mead, "A simplex method for function minimization," Computer Journal 7, 308 (1965).

[37] M. J. D. Powell, "An efficient method for finding the minimum of a function of several variables without calculating derivatives," The Computer Journal 7, 155 (1964), https://academic.oup.com/comjnl/article-pdf/7/2/155/959784/070

[38] J. C. Spall, "A stochastic approximation technique for generating maximum likelihood parameter estimates," in Proceedings of the American Control Conference (Minneapolis, MN, 1987) pp. 1161-1167.

[39] W. Lavrijsen, A. Tudor, J. Müller, C. Iancu, and W. de Jong, "Classical optimizers for noisy intermediate-scale quantum devices," in 2020 IEEE In-
ternational Conference on Quantum Computing and Engineering (QCE) (2020) pp. 267-277.

[40] J. M. Kübler, A. Arrasmith, L. Cincio, and P. J. Coles, "An Adaptive Optimizer for Measurement-Frugal Variational Algorithms," Quantum 4, 263 (2020).

[41] R. LaRose, A. Tikku, É. O'Neel-Judy, L. Cincio, and P. J. Coles, "Variational quantum state diagonalization," npj Quantum Information 5, 57 (2019).

[42] A. Arrasmith, L. Cincio, R. D. Somma, and P. J. Coles, "Operator Sampling for Shot-frugal Optimization in Variational Algorithms," arXiv:2004.06252 (2020).

[43] J. Stokes, J. Izaac, N. Killoran, and G. Carleo, "Quantum Natural Gradient," Quantum 4, 269 (2020).

[44] B. Koczor and S. C. Benjamin, "Quantum natural gradient generalized to noisy and nonunitary circuits," Phys. Rev. A 106, 062416 (2022).

[45] K. M. Nakanishi, K. Fujii, and S. Todo, "Sequential minimal optimization for quantum-classical hybrid algorithms," Physical Review Research 2, 043158 (2020).

[46] R. M. Parrish, J. T. Iosue, A. Ozaeta, and P. L. McMahon, "A Jacobi Diagonalization and Anderson Acceleration Algorithm For Variational Quantum Algorithm Parameter Optimization," arXiv:1904.03206 (2019).

[47] R. Sweke, F. Wilde, J. Meyer, M. Schuld, P. K. Faehrmann, B. Meynard-Piganeau, and J. Eisert, "Stochastic gradient descent for hybrid quantumclassical optimization," Quantum 4, 314 (2020).

[48] B. Koczor and S. C. Benjamin, "Quantum analytic descent," Phys. Rev. Research 4, 023017 (2022).

[49] B. van Straaten and B. Koczor, "Measurement cost of metric-aware variational quantum algorithms," PRX Quantum 2, 030324 (2021).

[50] A. Gu, A. Lowe, P. A. Dub, P. J. Coles, and A. Arrasmith, "Adaptive shot allocation for fast convergence in variational quantum algorithms," arXiv:2108.10434 (2021).

[51] M. Menickelly, Y. Ha, and M. Otten, "Latency considerations for stochastic optimizers in variational quantum algorithms," arXiv:2201.13438 (2022).

[52] S. Tamiya and H. Yamasaki, "Stochastic gradient line bayesian optimization for efficient noise-robust optimization of parameterized quantum circuits," npj Quantum Information 8, 90 (2022).

[53] L. Bittel, J. Watty, and M. Kliesch, "Fast gradient estimation for variational quantum algorithms," arXiv:2210.06484 (2022).

[54] C. Moussa, M. H. Gordon, M. Baczyk, M. Cerezo, L. Cincio, and P. J. Coles, "Resource frugal optimizer for quantum machine learning," arXiv:2211.04965 (2022).

[55] G. Boyd and B. Koczor, "Training variational quantum circuits with covar: Covariance root finding with classical shadows," Phys. Rev. X 12, 041022 (2022).

[56] E. Fontana, M. Cerezo, A. Arrasmith, I. Rungger, and P. J. Coles, "Non-trivial symmetries in quantum landscapes and their resilience to quantum noise," Quantum 6,804 (2022).

[57] L. Gentini, A. Cuccoli, S. Pirandola, P. Verrucchi, and L. Banchi, "Noise-resilient variational hybrid quantumclassical optimization," Phys. Rev. A 102, 052414 (2020).

[58] T. L. Patti, O. Shehab, K. Najafi, and S. F. Yelin, "Markov chain monte carlo enhanced variational quan- tum algorithms," Quantum Science and Technology 8, 015019 (2022).

[59] S. Duffield, M. Benedetti, and M. Rosenkranz, "Bayesian learning of parameterised quantum circuits," Machine Learning: Science and Technology 4, 025007 (2023).

[60] J. Liu, F. Wilde, A. A. Mele, L. Jiang, and J. Eisert, "Stochastic noise can be helpful for variational quantum algorithms," arXiv:2210.06723 (2022).

[61] D. Wecker, M. B. Hastings, and M. Troyer, "Progress towards practical quantum variational algorithms," Phys. Rev. A 92, 042303 (2015).

[62] C. Moussa, M. H. Gordon, M. Baczyk, M. Cerezo, L. Cincio, and P. J. Coles, "Resource frugal optimizer for quantum machine learning," Quantum Science and Technology 8, 045019 (2023).

[63] K. Ito, "Latency-aware adaptive shot allocation for run-time efficient variational quantum algorithms," arXiv:2302.04422 (2023).

[64] C. Chen, D. Carlson, Z. Gan, C. Li, and L. Carin, "Bridging the gap between stochastic gradient mcmc and stochastic optimization," in Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, Vol. 51, edited by A. Gretton and C. C. Robert (PMLR, Cadiz, Spain, 2016) pp. 1051-1060.

[65] E. Aïmeur, G. Brassard, and S. Gambs, "Machine learning in a quantum world," in Advances in Artificial Intelligence, edited by L. Lamontagne and M. Marchand (Springer Berlin Heidelberg, Berlin, Heidelberg, 2006) pp. 431-442.

[66] V. Dunjko, J. M. Taylor, and H. J. Briegel, "Quantumenhanced machine learning," Phys. Rev. Lett. 117, 130501 (2016).

[67] L. Buffoni and F. Caruso, "New trends in quantum machine learning(a)," Europhysics Letters 132, 60004 (2021).

[68] A. Nakayama, K. Mitarai, L. Placidi, T. Sugimoto, and K. Fujii, "VQE-generated Quantum Circuit Dataset for Machine Learning," arXiv:2302.09751 (2023).

[69] A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre, "Data re-uploading for a universal quantum classifier," Quantum 4, 226 (2020).

[70] M. Schuld, R. Sweke, and J. J. Meyer, "Effect of data encoding on the expressive power of variational quantum-machine-learning models," Phys. Rev. A 103, 032430 (2021).

[71] M. C. Caro, E. Gil-Fuster, J. J. Meyer, J. Eisert, and R. Sweke, "Encoding-dependent generalization bounds for parametrized quantum circuits," Quantum 5, 582 (2021).

[72] K. Sharma, M. Cerezo, L. Cincio, and P. J. Coles, "Trainability of dissipative perceptron-based quantum neural networks," Phys. Rev. Lett. 128, 180505 (2022).

[73] K. Beer, D. Bondarenko, T. Farrelly, T. J. Osborne, R. Salzmann, D. Scheiermann, and R. Wolf, "Training deep quantum neural networks," Nature Communications 11, 808 (2020).

[74] I. Cong, S. Choi, and M. D. Lukin, "Quantum convolutional neural networks," Nature Physics 15, 1273 (2019).

[75] P. D. Johnson, J. Romero, J. Olson, Y. Cao, and A. Aspuru-Guzik, "QVECTOR: an algorithm for device-tailored quantum error correction,"

arXiv:1711.02249 (2017).

[76] A. W. Harrow and J. C. Napp, "Low-depth gradient measurements can improve convergence in variational hybrid quantum-classical algorithms," Phys. Rev. Lett. 126, 140502 (2021).

[77] D. Wierichs, J. Izaac, C. Wang, and C. Y.-Y. Lin, "General parameter-shift rules for quantum gradients," Quantum 6, 677 (2022).

[78] M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, "Evaluating analytic gradients on quantum hardware," Phys. Rev. A 99, 032331 (2019).

[79] J. Romero, J. P. Olson, and A. Aspuru-Guzik, "Quantum autoencoders for efficient compression of quantum data," Quantum Science and Technology 2, 045001 (2017).

[80] L. Feng, S. Shu, Z. Lin, F. Lv, L. Li, and B. An, "Can cross entropy loss be robust to label noise?" in Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI'20 (2021).

[81] H.-Y. Huang, R. Kueng, and J. Preskill, "Predicting many properties of a quantum system from very few measurements," Nature Physics 16, 1050 (2020).

[82] I. Hamamura and T. Imamichi, "Efficient evaluation of quantum observables using entangled measurements," npj Quantum Information 6, 56 (2020).

[83] O. Crawford, B. v. Straaten, D. Wang, T. Parks, E. Campbell, and S. Brierley, "Efficient quantum measurement of Pauli operators in the presence of finite sampling error," Quantum 5, 385 (2021).

[84] C. Hempel, C. Maier, J. Romero, J. McClean, T. Monz, H. Shen, P. Jurcevic, B. P. Lanyon, P. Love, R. Babbush, A. Aspuru-Guzik, R. Blatt, and C. F. Roos, "Quantum chemistry calculations on a trapped-ion quantum simulator," Phys. Rev. X 8, 031022 (2018).

[85] A. F. Izmaylov, T.-C. Yen, R. A. Lang, and V. Verteletskyi, "Unitary partitioning approach to the measurement problem in the variational quantum eigensolver method," Journal of Chemical Theory and Computation 16, 190 (2020).

[86] H. J. Vallury, M. A. Jones, C. D. Hill, and L. C. L. Hollenberg, "Quantum computed moments correction to variational estimates," Quantum 4, 373 (2020).

[87] V. Verteletskyi, T.-C. Yen, and A. F. Izmaylov, "Measurement optimization in the variational quantum eigensolver using a minimum clique cover," The Journal of Chemical Physics 152, 124114 (2020), https://doi.org/10.1063/1.5141458.

[88] A. Zhao, A. Tranter, W. M. Kirby, S. F. Ung, A. Miyake, and P. J. Love, "Measurement reduction in variational quantum algorithms," Phys. Rev. A 101, 062322 (2020).

[89] B. Wu, J. Sun, Q. Huang, and X. Yuan, "Overlapped grouping measurement: A unified framework for measuring quantum states," Quantum 7, 896 (2023).

[90] W. Hoeffding, "A Class of Statistics with Asymptotically Normal Distribution," The Annals of Mathematical Statistics 19, 293 (1948)

[91] .

[92] D. A. McAllester, "Pac-bayesian stochastic model selection," Machine Learning 51, 5 (2003).

[93] A. Barron and T. Cover, "Minimum complexity density estimation," IEEE Transactions on Information Theory 37, 1034 (1991).

[94] S. Walker and N. L. Hjort, "On bayesian consistency," Journal of the Royal Statistical Society. Series B (Sta- tistical Methodology) 63, 811 (2001).

[95] T. Zhang, "From $\epsilon$-entropy to kl-entropy: Analysis of minimum information complexity density estimation," The Annals of Statistics 34, 2180 (2006).

[96] T. Zhang, "Learning bounds for a generalized family of bayesian posterior distributions," in Advances in Neural Information Processing Systems, Vol. 16, edited by S. Thrun, L. Saul, and B. Schölkopf (MIT Press, 2003).

[97] P. Grünwald, "Safe learning: bridging the gap between bayes, mdl and statistical learning theory via empirical convexity," in Proceedings of the 24th Annual Conference on Learning Theory, Proceedings of Machine Learning Research, Vol. 19, edited by S. M. Kakade and U. von Luxburg (PMLR, Budapest, Hungary, 2011) pp. $397-420$.

[98] P. Grünwald, "The safe bayesian," in Algorithmic Learning Theory, edited by N. H. Bshouty, G. Stoltz, N. Vayatis, and T. Zeugmann (Springer Berlin Heidelberg, Berlin, Heidelberg, 2012) pp. 169-183.

[99] P. Grünwald, "Safe probability," Journal of Statistical Planning and Inference 195, 47 (2018), confidence distributions.

[100] P. Grünwald and T. van Ommen, "Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It," Bayesian Analysis 12, 1069 (2017).

[101] F. Wenzel, K. Roth, B. Veeling, J. Swiatkowski, L. Tran, S. Mandt, J. Snoek, T. Salimans, R. Jenatton, and S. Nowozin, "How good is the Bayes posterior in deep neural networks really?" in Proceedings of the 37th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 119, edited by H. D. III and A. Singh (PMLR, 2020) pp. 10248-10259.

[102] V. Fortuin, A. Garriga-Alonso, F. Wenzel, G. Ratsch, R. E. Turner, M. van der Wilk, and L. Aitchison, "Bayesian neural network priors revisited," in Third Symposium on Advances in Approximate Bayesian Inference (2021).

[103] K. Pitas and J. Arbel, "Cold Posteriors through PAC-Bayes," arXiv:2206.11173 (2022).

[104] S. Geman and D. Geman, "Stochastic relaxation, gibbs distributions, and the bayesian restoration of images," IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-6, 721 (1984).

[105] M. Welling and Y. W. Teh, "Bayesian learning via stochastic gradient langevin dynamics." in ICML, edited by L. Getoor and T. Scheffer (Omnipress, 2011) pp. 681-688.

[106] T. Chen, E. Fox, and C. Guestrin, "Stochastic gradient hamiltonian monte carlo," in Proceedings of the 31st International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 32, edited by E. P. Xing and T. Jebara (PMLR, Bejing, China, 2014) pp. 1683-1691.

[107] I. Sutskever, J. Martens, G. Dahl, and G. Hinton, "On the importance of initialization and momentum in deep learning," in Proceedings of the 30th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 28, edited by S. Dasgupta and D. McAllester (PMLR, Atlanta, Georgia, USA, 2013) pp. 1139-1147.

[108] N. Ding, Y. Fang, R. Babbush, C. Chen, R. D. Skeel, and H. Neven, "Bayesian sampling using stochastic gradient thermostats," in Advances in Neural Information

Processing Systems, Vol. 27, edited by Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger (Curran Associates, Inc., 2014).

[109] Z. Gan, C. Chen, R. Henao, D. Carlson, and L. Carin, "Scalable deep poisson factor analysis for topic modeling," in Proceedings of the 32nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 37, edited by F. Bach and D. Blei (PMLR, Lille, France, 2015) pp. 1823-1832.

[110] S. Nosé, "A unified formulation of the constant temperature molecular dynamics methods," The Journal of Chemical Physics 81, 511 (1984), https://pubs.aip.org/aip/jcp/article-pdf/81/1/511/9722

[111] W. G. Hoover, "Canonical dynamics: Equilibrium phase-space distributions," Phys. Rev. A 31, 1695 (1985).

[112] Y. Dauphin, H. de Vries, and Y. Bengio, "Equilibrated adaptive learning rates for non-convex optimization," in Advances in Neural Information Processing Systems, Vol. 28, edited by C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett (Curran Associates, Inc., 2015).

[113] X.-L. Li, "Preconditioned stochastic gradient descent," IEEE Transactions on Neural Networks and Learning Systems 29, 1454 (2018).

[114] X.-L. Li, "Online Second Order Methods for NonConvex Stochastic Optimizations," arXiv:1803.09383 (2018).

[115] J. Duchi, E. Hazan, and Y. Singer, "Adaptive subgradient methods for online learning and stochastic optimization," Journal of Machine Learning Research 12, 2121 (2011).

[116] T. Tieleman and G. Hinton, "Lecture 6.5 - rmsprop: Divide the gradient by a running average of its recent magnitude," Technical report (2012).

[117] M. Girolami and B. Calderhead, "Riemann manifold langevin and hamiltonian monte carlo methods," Journal of the Royal Statistical Society: Series B (Statistical Methodology) 73, 123 (2011), https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j. 14

[118] S. Patterson and Y. W. Teh, "Stochastic gradient riemannian langevin dynamics on the probability simplex," in Advances in Neural Information Processing Systems, Vol. 26, edited by C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Weinberger (Curran Associates, Inc., 2013).

[119] C. Li, C. Chen, D. Carlson, and L. Carin, "Preconditioned stochastic gradient langevin dynamics for deep neural networks," in Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, AAAI'16 (AAAI Press, 2016) pp. 1788-1794.

[120] R. Khasminskii and G. Milstein, Stochastic Stability of Differential Equations, Stochastic Modelling and Applied Probability (Springer Berlin Heidelberg, 2011).

[121] S. J. Vollmer, K. C. Zygalakis, and Y. W. Teh, "Exploration of the (non-)asymptotic bias and variance of stochastic gradient langevin dynamics," Journal of Machine Learning Research 17, 1 (2016).

[122] C. Chen, N. Ding, and L. Carin, "On the convergence of stochastic gradient mcmc algorithms with high-order integrators," in Proceedings of the 28th International Conference on Neural Information Processing Systems Volume 2, NIPS'15 (MIT Press, Cambridge, MA, USA, 2015) pp. 2278-2286.
[123] C. Li, C. Chen, K. Fan, and L. Carin, "High-order stochastic gradient thermostats for bayesian learning of deep models," in Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, AAAI'16 (AAAI Press, 2016) pp. 1795-1801.

[124] X. Li and P. Ding, "General forms of finite population central limit theorems with applications to causal inference," Journal of the American Statistical Association 112, 1759 (2017), https://doi.org/10.1080/01621459.2017.1295865.

[125] E. L. Lehmann, Nonparametrics: Statistical Methods Based on Ranks (Holden-Day, Inc., San Francisco, 6/511_11.

[126] Y. Li and S. C. Benjamin, "Efficient variational quantum simulator incorporating active error minimization," Phys. Rev. X 7, 021050 (2017).

[127] S. Endo, Z. Cai, S. C. Benjamin, and X. Yuan, "Hybrid quantum-classical algorithms and quantum error mitigation," Journal of the Physical Society of Japan 90, 032001 (2021), https://doi.org/10.7566/JPSJ.90.032001.

[128] K. Temme, S. Bravyi, and J. M. Gambetta, "Error mitigation for short-depth quantum circuits," Phys. Rev. Lett. 119, 180509 (2017).

[129] S. Endo, S. C. Benjamin, and Y. Li, "Practical quantum error mitigation for near-future applications," Phys. Rev. X 8, 031027 (2018).

[130] P. Czarnik, A. Arrasmith, P. J. Coles, and L. Cincio, "Error mitigation with Clifford quantum-circuit data," Quantum 5, 592 (2021).

[131] A. Strikis, D. Qin, Y. Chen, S. C. Benjamin, and Y. Li, "Learning-based quantum error mitigation," PRX Quantum 2, 040330 (2021).

[132] E. van den Berg, Z. K. Minev, A. Kandala, and K. Temme, "Probabilistic error cancellation with sparse pauli-lindblad models on noisy quantum processors," Nature Physics 19, 1116 (2023).

[133] A. Kandala, K. Temme, A. D. Córcoles, A. Mezzacapo, J. M. Chow, and J. M. Gambetta, "Error mitigation 67-9868.2extends6bue. computational reach of a noisy quantum processor," Nature 567, 491 (2019).

[134] Z. Cai, "Resource-efficient Purification-based Quantum Error Mitigation," arXiv:2107.07279 (2021).

[135] A. Mari, N. Shammah, and W. J. Zeng, "Extending quantum probabilistic error cancellation by noise scaling," Phys. Rev. A 104, 052607 (2021).

[136] P. D. Nation, H. Kang, N. Sundaresan, and J. M. Gambetta, "Scalable mitigation of measurement errors on quantum computers," PRX Quantum 2, 040326 (2021).

[137] B. Yang, R. Raymond, and S. Uno, "Efficient quantum readout-error mitigation for sparse measurement outcomes of near-term quantum devices," Phys. Rev. A 106, 012423 (2022).

[138] Z. Cai, R. Babbush, S. C. Benjamin, S. Endo, W. J. Huggins, Y. Li, J. R. McClean, and T. E. O'Brien, "Quantum Error Mitigation," arXiv:2210.00921 (2022).

[139] Y. Kim, A. Eddins, S. Anand, K. X. Wei, E. van den Berg, S. Rosenblatt, H. Nayfeh, Y. Wu, M. Zaletel, K. Temme, and A. Kandala, "Evidence for the utility of quantum computing before fault tolerance," Nature 618,500 (2023).

[140] K. Sharma, S. Khatri, M. Cerezo, and P. J. Coles, "Noise resilience of variational quantum compiling," New Journal of Physics 22, 043006 (2020).

[141] E. Fontana, N. Fitzpatrick, D. M. n. Ramo, R. Duncan, and I. Rungger, "Evaluating the noise resilience of variational quantum algorithms," Phys. Rev. A 104, 022403 (2021).

[142] S. Wang, P. Czarnik, A. Arrasmith, M. Cerezo, L. Cincio, and P. J. Coles, "Can Error Mitigation Improve Trainability of Noisy Variational Quantum Algorithms?" arXiv:2109.01051 (2021).

[143] S. T. Jose and O. Simeone, "Error-mitigation-aided optimization of parameterized quantum circuits: Convergence analysis," IEEE Transactions on Quantum Engineering 3, 1 (2022).

[144] H. Wang, J. Gu, Y. Ding, Z. Li, F. T. Chong, D. Z. Pan, and S. Han, "Quantumnat: Quantum noise-aware training with noise injection, quantization and normalization," in Proceedings of the 59th ACM/IEEE Design Automation Conference, DAC '22 (Association for Computing Machinery, New York, NY, USA, 2022) pp. 1-6.

[145] Y. Quek, D. S. França, S. Khatri, J. J. Meyer, and J. Eisert, "Exponentially tighter bounds on limitations of quantum error mitigation," arXiv:2210.11505 (2022).

[146] K. Tsubouchi, T. Sagawa, and N. Yoshioka, "Univer- sal cost bound of quantum error mitigation based on quantum estimation theory," arXiv:2208.09385 (2022).

[147] R. Takagi, H. Tajima, and M. Gu, "Universal sampling lower bounds for quantum error mitigation," arXiv:2208.09178 (2022).

[148] R. Takagi, S. Endo, S. Minagawa, and M. Gu, "Fundamental limits of quantum error mitigation," npj Quantum Information 8, 114 (2022).

[149] Y. Suzuki, Y. Kawase, Y. Masumura, Y. Hiraga, M. Nakadai, J. Chen, K. M. Nakanishi, K. Mitarai, R. Imai, S. Tamiya, T. Yamamoto, T. Yan, T. Kawakubo, Y. O. Nakagawa, Y. Ibe, Y. Zhang, H. Yamashita, H. Yoshimura, A. Hayashi, and K. Fujii, "Qulacs: a fast and versatile quantum circuit simulator for research purpose," Quantum 5, 559 (2021).

[150] H. Xiao, K. Rasul, and R. Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms," arXiv:1708.07747 (2017).

[151] E. Anderson, "The species problem in iris," Annals of the Missouri Botanical Garden 23, 457 (1936).


[^0]:    * kosuke.ito.qiqb@osaka-u.ac.jp

    $\dagger$ fujii@qc.ee.es.osaka-u.ac.jp

</end of paper 1>


<paper 2>
# Better than classical? The subtle art of benchmarking quantum machine learning models 

Joseph Bowles, ${ }^{1, *}$ Shahnawaz Ahmed, ${ }^{1,2, \dagger}$ and Maria Schuld ${ }^{1, \ddagger}$<br>${ }^{1}$ Xanadu, Toronto, ON, M5G 2C8, Canada<br>${ }^{2}$ Chalmers University of Technology

(Dated: March 15, 2024)


#### Abstract

Benchmarking models via classical simulations is one of the main ways to judge ideas in quantum machine learning before noise-free hardware is available. However, the huge impact of the experimental design on the results, the small scales within reach today, as well as narratives influenced by the commercialisation of quantum technologies make it difficult to gain robust insights. To facilitate better decision-making we develop an open-source package based on the PennyLane software framework and use it to conduct a large-scale study that systematically tests 12 popular quantum machine learning models on 6 binary classification tasks used to create 160 individual datasets. We find that overall, out-of-the-box classical machine learning models outperform the quantum classifiers. Moreover, removing entanglement from a quantum model often results in as good or better performance, suggesting that "quantumness" may not be the crucial ingredient for the small learning tasks considered here. Our benchmarks also unlock investigations beyond simplistic leaderboard comparisons, and we identify five important questions for quantum model design that follow from our results.


Much has been written about the "potential" of quantum machine learning, a discipline that asks how quantum computers fundamentally change what we can learn from data $[1,2]$. While we have no means of running quantum algorithms on noise-free hardware yet, there are only a limited number of tools available to assess this potential. Besides proving advantages for artificial problem settings on paper, certain ideas - most prominently, variational models designed for near-term quantum technologies - can be tested in classical simulations using small datasets. Such benchmarks have in fact become a standard practice in the quantum machine learning literature and are found in almost every paper.

A taste for the results derived from small-scale benchmarks can be obtained through a simple literature review exercise. Out of 55 relevant papers published on the preprint server arXiv ${ }^{1}$ until December 2023 that contain the terms "quantum machine learning" and "outperform" in title or abstract, one finds that about $40 \%$ claim that a quantum model outperforms a classical model, while about $50 \%$ claim that some improvement to a quantum machine learning method outperforms the original one (such as optimisers $[3,4]$, pre-training strategies [5] or symmetry-aware ansatze $[6,7]$ ). Only 3 papers or $4 \%$ find that a quantum model does not outperform a classi-[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-01.jpg?height=537&width=702&top_left_y=1022&top_left_x=1167)

FIG. 1. The scope of the benchmark study at a glance.

cal one $[8-10]$; two of these are quick to mention that this is "due to the noise level in the available quantum hardware" [8] or that "the proposed methods are ready for [...] quantum computing devices which have the potential to outperform classical systems in the near future" [10]. Only one paper [9] draws critical conclusions from their empirical results. If we assume that this literature review is representative ${ }^{2}$, then the overwhelming signal is that quantum machine learning algorithm design is progressing rapidly on all fronts, with ample evidence from small-scale experiments that it is already beating classical machine learning in generic domains. But can we trust this picture when judging the potential of current ideas?[^1]

In search for an evidence-based evaluation of proposals in near-term quantum machine learning algorithm design, this paper conducts a large-scale benchmark study that systematically tests popular quantum machine learning models on classification tasks. The code, built on the PennyLane software framework [11], is made available under https://github.com/XanaduAI/ qml-benchmarks, and the datasets can also be found under https://pennylane.ai/datasets/. We designed the study with a strong emphasis on scientific best practices that aim at reducing a positivity (or negativity) bias. To achieve this, we selected 12 prototypical, influential quantum machine learning models ranging from socalled quantum neural networks $[12,13]$ to convolutional quantum neural networks [14, 15] and quantum kernel methods $[16,17]$ through a randomised procedure and implemented them as faithfully as possible.

The 6 data generation procedures for the classification tasks are chosen based on principles such as structural diversity, comparability, control of important variables, and theoretical relevance. They generate 160 individual datasets grouped into 10 benchmarks, where each benchmark varies either the dimension or a parameter controlling the difficulty of the learning problem that the models have to solve. Note that not all models are tested on all datasets and benchmarks; some, like translationinvariant tasks, were designed for specific models, while others required too many computational resources for the extensive hyperparameter optimisation performed in this paper. ${ }^{3}$

We compare the quantum models to likewise prototypical and influential classical machine learning models like Support Vector Machines and simple Neural Networks (i.e., not "state-of-the-art", but rather "out-of-the-box" models for small datasets). The goal is to find signals across the experiments that are consistent and can give us clues about which research questions are worth investigating further.

Our results show that - contrary to the picture emerging from the literature sample above - the prototypical classical baselines perform systematically better than the prototypical quantum models on the small-scale datasets chosen here. Furthermore, for most models there are no significant drops in performance if we use comparable quantum models that do not use entanglement and are classically simulable at scale, suggesting that "quantumness" may not be a defining factor.

With small variations, the overall rankings between models are surprisingly consistent throughout the different benchmarks and allow some more nuanced observations. For example, hybrid quantum-classical models -[^2]

such as models that use quantum circuits inside neural networks or a support vector machine - perform similarly to their purely classical "hosts", suggesting that the quantum components play a similar role to the classical ones they replace. While a layer-wise, trainable encoding with trainable input feature scaling ("data-reuploading" [18]) shows some promising behaviour, models based on so-called "amplitude encoding" [13, 19, 20] struggle with the classification tasks, even if given copies of inputs. Interestingly, almost all quantum models perform particularly badly on the benchmarks we deemed simplest, a linearly separable dataset.

Although the quantum models we tested failed to provide compelling results, we are not necessarily advocating for a research program that attempts to optimise them for the datasets in this work. Rather, the poor performance relative to baseline classical models across a range of tasks should bolster a hard-to-swallow fact about current quantum machine learning research: namely, that the inductive bias of near-term quantum models, the added benefit of "quantumness" as well as the problems for which both are useful, are still poorly understood $[9,21-23]$.

We finally note that while independent benchmark "meta-studies" like ours are still rare, an increasing number of papers aim at systematically studying aspects of quantum model design. For example, Kashif et al. [5] look at the role of parameter initialisation for trainability, and [24] provide a software framework to test generative models. Moussa et al. [25] investigate the role of hyperparameters on the generalisation performance, and some of their findings are confirmed by our results.

The remainder of the paper will discuss common benchmarking pitfalls (Section I), the model (Section II) and data (Section III) selection method used in this study, and insights from hyperparameter optimisation (Section IV). Our main results are presented in Section V and Section VI discusses questions that follow for the design of quantum models. The conclusion (Section VII) will reflect on important lessons learnt in this study, which can hopefully contribute to more robust and critical benchmarking practices for quantum machine learning.

## I. WHAT CAN WE LEARN FROM BENCHMARKS?

Eagerly following its parent discipline of classical machine learning, benchmarks in the quantum machine learning literature are commonly motivated by statements like "to demonstrate the performance of our quantum model we test it on the XYZ dataset". But how much can benchmarks give us an insight into the quality of a model or idea? Before entering into the technical details we want to discuss this question critically as a cautionary tale that informs our own benchmark study.

## A. The need for scientific rigour

In classical machine learning, increasing doubts about the singular reliance on performance metrics have emerged in recent years $[26-28]^{4}$. Concerns are supported by a range of introspective studies that show how benchmarking results are subject to high variance when seemingly small experimental design decisions are changed.

Firstly, the dataset selection has a significant impact on the comparative performance of one model over others. This is epitomised by no-free-lunch theorems [29] suggesting that for a large enough set of problems, the average performance of any model is the same, and we can only hope for a good performance on a relevant subset (automatically paying the price of a bad performance on another subset). ${ }^{5}$ An illustrative, even if highly simplified, example is shown in Figure 2, where a minor rearrangement of the data in a classification task causes an admittedly adversarially hand-crafted - quantum model to switch from being nearly unable to learn, to gaining perfect accuracy.

In more serious examples, studies like Dehghani et al. [26] in their aptly named paper "The benchmark lottery", show for a range of hotly contested benchmarks that significantly different leaderboard rankings are obtained when excluding a few datasets from benchmarking suites or making changes to the way that scores are aggregated. Northcutt et al. [30] find that correcting labeling errors in the ubiquitous ImageNet validation set likewise changes the ranking of models, and Recht et al. [31, 32] demonstrate that using a different test-train split in popular image recognition datasets decreases the accuracies of models by as much as $14 \%^{6}$. While "every benchmark makes a statement about what it perceives to be important" [26], it can be shown that which benchmarks are deemed relevant is largely influenced by trends in research [26,33-35], and therefore by the social makeup of a scientific community rather than methodological considerations.

Secondly, even if the data is selected carefully, variations in the remainder of the study design can hugely influence the results of benchmarking, leading to a host of contradicting results when judging the performance of a new method. For example, large-scale comparative studies suggest that small adjustments to transformer architectures advertised in the literature have no measurable effect on their performance [36], and that contrary to a common belief, standard convolutional neural networks[^3]

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-03.jpg?height=261&width=377&top_left_y=192&top_left_x=1123)

test accuracy $=0.53$

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-03.jpg?height=282&width=393&top_left_y=190&top_left_x=1535)

FIG. 2. Illustrative example showing the effect of slight variations in a dataset on model performance. The same quantum model is trained on two different datasets to predict two classes, red and blue. The decision regions are displayed as the shaded areas. Depending on a small variation of the classification task the same model can perform poorly (left) or have a perfect test score (right). We used a "vanilla" quantum neural network model with two layers of angle embedding interspersed with CNOT entanglers, and three layers of a trainable variational circuit, followed by a $Z$-measurement on the first qubit. The classifier is trained on the points with round markers and tested on the points marked as triangles.

still beat transformers when trained on a comparable amount of data [37]. Similarly, extensive hyperparameter optimisation can give baseline models of generative adversarial networks [38], deep reinforcement learning [39] and Bayesian deep neural nets [40] the performance of more advanced proposals. These findings question the large effort invested into optimising model design. And not only are there variations in the model and data, but the choice of scoring metric like accuracy, F-score and Area-Under-Curve make different assumptions on what is deemed important and may correlate poorly with each other [41]. Lastly, Dodge et al. [42] demonstrate how integrating computational costs into performance considerations for Natural Language Processing models highlight regimes where a method as basic as logistic regression is superior to deep learning.

A third danger of benchmarks that judge the "power" of a model over another is a systematic positivity bias, since the goal of designing models that are better than existing ones creates an incentive to publish positive results. This can be illustrated in a simple thought experiment (see Figure 3): Consider 100 researchers, each investigating 20 different types of quantum models before settling on a promising design and publishing benchmarks of the best one against a classical model of their choice. Let us assume there is some kind of ground truth that the performance of quantum and classical machine learning models is normally distributed, and on average classical models perform better and more consistently (say, with a mean of 0.65 and standard deviation of 0.07 ) than quantum ones (mean 0.55, standard deviation 0.1). But the bias created from discarding 19 models means that the researchers will overall find that quantum models are better than classical models; when running a simple simulation we find that the published mean performance of the quantum models is 0.74 vs. 0.65 of classical ones: the scales are flipped! Since leaderboard-driven research ac-

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-04.jpg?height=290&width=854&top_left_y=191&top_left_x=169)

FIG. 3. Numerical illustration of the thought experiment on a positivity bias. Assume that the "true" performance of classical and quantum models is distributed normally (blue and red curves) with the mean for classical model performance higher than in the quantum case. The dashed lines report numerical calculations of the mean of model performance if 100 researchers report only on the top-performing candidate out of 20 quantum models, but do not select the best classical model in a similar manner. The bias from discarding the 19 worst-performing quantum models reverses the observed average performance with respect to the true one.

tively searches for good models, a positivity bias of this nature is not a question of ethical misconduct, but built into the research methodology.

Together, these arguments make a convincing case that "perfomance" or "power" of a model cannot be viewed as a property that can be "demonstrated" by an empirical study, but as a highly context-dependent signal that has to be artfully coaxed out of a system with many interacting parts - not unlike the challenge of an experimental physicist trying to make sense of a complex physical system.

Luckily, we have a tried-and-tested weapon for this challenging task: scientific rigour. For example, benchmarks can become a powerful tool when asking questions of a clearly defined scope. "Does my quantum model beat classical machine learning?" is impossible to answer by a numerical experiment, while "Does the performance of a model on a specific task increase as measured by some metric, if we change the data, or training conditions?" might be. A carefully selected task for such a question, for example an artificial dataset allowing systematic control of the properties of interest, allow much clearer interpretations of the results. Testing when and how a model can be made to perform poorly will give other researchers clues about mechanisms at play, rather than advertising one method over another. And lastly, searching for patterns that can be reproduced across widely different settings will constitute a signal that allows the field to take objective decisions on which ideas show promise. In this study we aim to be explicitly mindful of these pitfalls in the process of model and data selection, experimental design and interpretation of the results.

## B. Should we use MNIST for quantum machine learning?

In order to illustrate some specific problems in quantum machine learning research with its peculiar challenges, let us get back to the question of data selection once more. Unless a quantum researcher focuses on their favourite domain, such as physics itself, the choice of dataset tends to be directly adopted from the classical machine learning literature. But without error-free hardware to run algorithms on, quantum machine learning research finds itself in a very different situation: data is either borrowed from machine learning some decades in the past, or needs to be downscaled and simplified in order to suit the scale of simulation capabilities or noisy small-scale hardware. We want to demonstrate some issues arising from this predicament with a discussion of the (in)famous MNIST handwritten digits datasets which is used widely in quantum machine learning studies. Note that somewhat contradictory, we will still add MNIST as a task here in order to make our results easier to compare with so many other studies.

The MNIST dataset consists of $60.00028 \times 28$ images of handwritten digits [43] and played a crucial historical role in the benchmarking of classical machine learning. While largely superseded by data like ImageNet [44] and CIFAIR [45], it is still occasionally used as a sanity check of new models, as this quote by Geoffrey Hinton, one of the pioneers of deep learning, demonstrates:

MNIST has been well studied and the performance of simple neural networks trained with backpropagation is well known. This makes MNIST very convenient for testing out new learning algorithms to see if they actually work. [46]

Hinton goes on to summarise the baselines for a reasonable success on MNIST:

Sensibly-engineered convolutional neural nets with a few hidden layers typically get about $0.6 \%$ test error. In the "permutationinvariant" version of the task, the neural net is not given any information about the spatial layout of the pixels [and] feed-forward neural networks with a few fully connected hidden layers of Rectified Linear Units (ReLUs) typically get about $1.4 \%$ test error. This can be reduced to around $1.1 \%$ test error using a variety of regularizers [...] [46]

While this may suggests MNIST as a useful benchmark for innovation in quantum machine learning, there is a serious caveat to consider: The typical input size that can be handled with simulations (not to mention hardware) is of the order of tens of features, but downscaling the 784 original pixels by that much voids our extensive knowledge about the baseline to beat.

For example, a representative analysis of 15 randomly selected papers in quantum machine learning using MNIST benchmarks ${ }^{7}$ shows that more than half use a pre-processing strategy like resolution reduction or PCA to lower the number of input features, while most others investigate hybrid models where the dimensionality reduction is implicitly achieved by a classical layer such as a convolutional or tensor network. ${ }^{8}$ Preprocessing the data changes the hardness and nature of the learning problem and the meaningfulness of using MNIST has to be reassessed. For example, the 2-dimensional PCA-reduced MNIST 3-5 classification problem consists of about 10,000 datapoints that lie in overlapping blobs (see Figure 5 further below), a task that is hardly meaningful. Furthermore, as Bausch [47] remarks, reducing images to a $4 \times 4$ resolution leads to $70 \%$ of test images also being present during training. And of course, using powerful classical layers to read in the data makes it hard to judge the impact of the quantum model.

There are other issues besides pre-processing. For example, only four of the papers in our representative sample of 15 use the original MNIST multi-class problem, while all others distinguish between two of the 10 digits - which tends to make the problem easier. Distinguishing between $0-1$, for instance, can be achieved by a linear classifier to almost $100 \%$ accuracy. Possibly blinded by the size of the original dataset, not all authors seem to be aware that their quantum model scores highly on an almost trivial task.

It is no surprise that the overall results - which in the sample of 15 papers range from accuracies of $70-99.6 \%$ are mixed and hard to interpret with respect to the known baseline. Collectively, they certainly do not give convincing evidence for quantum models to systematically reach the test accuracies of $98.6 \%$ or even $99.4 \%$ put forward by Hinton with respect to the original MNIST task - at least not if unaided by powerful classical models.

Overall, the arguments summarised here put into question the practice of adopting datasets from classical machine learning blindly when their benefits are lost. Alternatives, such as 1-d MNIST [48] designed to scale down deep learning experiments, may be a lot more suitable.

With the methodological concerns sufficiently emphasised, we now proceed to introducing the technical details and decisions taken in this work.[^4]

## II. MODELS TO BE BENCHMARKED

## A. Selection methodology and bias

The goal of our paper selection procedure was to identify a diverse set of influential ideas in quantum machine learning that are implementable on current-day standard simulators. Importantly, we are not claiming to capture the "best" or most advanced algorithmic designs, but rather those that are the foundation of models proposed today.

After attempting other strategies that failed to produce a representative sample we settled on the following step-by-step selection method:

1. Thematic match: We first collected all papers accessible via the arXiv API in the period 1 January 2018 to 1 June 2023 in the "quant-ph" category with keywords "classif*", "learn*", "supervised", "MNIST". By this we intended to find the overwhelming share of papers that are part of the current quantum machine learning discourse. [3500 papers]
2. Influential: From the previous set we sub-selected papers with 30 or more Google Scholar citations on the cutoff date 2 June 2023. This was intended to bias the selection towards ideas that stood the test of time and are used widely in quantum machine learning research. It inevitably also biases towards older papers (see Figure 4). [561 papers]
3. Suitable for benchmarks: From reading the abstract, we sub-selected papers that propose a new quantum model for classification on conventional classical data which can be numerically tested by a standard qubit-based simulator. This ensured that the models can be meaningfully compared via a focused set of benchmarks, and that we limit ourselves to original proposals. [29 papers]
4. Implementable: To achieve a reasonable workload, we sampled a random subset of 15 papers from the 29 candidates. After reading the full body of these papers we had to exclude another four that we could not reasonably implement, either because the data encoding was too expensive [12, 49], or because the paper did not specify important parts of our classification pipeline like training [50] or an embedding strategy for classical datasets [14]. [11 papers]

From this final selection of 11 papers $[13,15,18-20,51-$ 56] we implemented 12 quantum machine learning models. Havlíček et al. [52] compared two distinct architectures, both of which we implemented.

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-06.jpg?height=406&width=700&top_left_y=182&top_left_x=257)

FIG. 4. Publishing date versus citations of the eleven selected papers (red crosses) from an initial set of 3500 papers drawn from the ArXiv API (gray dots). Outliers with over 1500 citations are not shown. Selecting papers with 30 or more citations introduces a bias towards less recent work.

## B. Summary of the models

The 12 models we implemented can be split into three families: quantum neural networks, quantum kernel methods and quantum convolutional neural networks. Here we explain the principle of each family and give a brief overview of the central idea of each model. Technical terms commonly known within the quantum machine learning community, but not necessarily clear to every reader, are highlighted in orange italic font and explained in a glossary at the end of this paper (see Appendix B). More detailed descriptions of the models can be found in Appendix C, and Table I provides the basic facts at a glance. We use camel case to refer to models, and the names correspond to the classes used in the software package.

## 1. Quantum neural network models

So-called "quantum neural networks" are - somewhat misleadingly named - simply variational quantum circuits that encode data and are trainable by gradientdescent methods. Variational quantum circuits have been extensively studied in the quantum machine learning literature and take the form

$$
\begin{equation*}
f(\boldsymbol{\theta}, \boldsymbol{x})=\operatorname{tr}[O(\boldsymbol{x}, \boldsymbol{\theta}) \rho(\boldsymbol{x}, \boldsymbol{\theta})] \tag{1}
\end{equation*}
$$

where $\rho$ is a density matrix, $O$ is an observable, and both may depend on an input data point $\boldsymbol{x}$ and trainable parameters $\boldsymbol{\theta}$. For our purposes, a quantum neural network model is one that combines one or more such variational quantum circuits $\left\{f_{1}, \ldots, f_{L}\right\}$ to classify data into one of two labels $y= \pm 1$ through a class prediction function $f_{\text {pred }}:$

$$
\begin{equation*}
y=f_{\text {pred }}\left(f_{1}\left(\boldsymbol{\theta}_{1}, \boldsymbol{x}\right), \cdots, f_{L}\left(\boldsymbol{\theta}_{L}, \boldsymbol{x}\right)\right) \tag{2}
\end{equation*}
$$

In the simplest case, where the model uses a single quantum function $(L=1)$ and an observable $O(\boldsymbol{x}, \boldsymbol{\theta})$ with eigenvalues $\pm 1, f_{\text {pred }}$ is typically the sign function, so that the sign of the measured observable indicates the predicted class.

To train these models, one defines a differentiable cost function

$$
\begin{equation*}
\mathcal{L}(\boldsymbol{\theta}, \mathbf{X}, \boldsymbol{y})=\frac{1}{N} \sum_{i} \ell\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}, y_{i}\right) \tag{3}
\end{equation*}
$$

where the loss $\ell\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}, y_{i}\right)$ measures the model performance for a specific training data point $\boldsymbol{x}_{i}$ whose true label $y_{i}$ we know, and $\mathbf{X}, \boldsymbol{y}$ summarise all training inputs and labels into a matrix/vector. Like neural networks, the loss can be numerically minimised in an endto-end fashion via stochastic gradient descent by application of the chain rule, where so-called parameter-shift rules $[57,58]$ allow us to evaluate gradients of quantum circuit parameters on hardware.

We study six quantum neural network classifiers:

- CircuitCentricClassifier [13]: A generic quantum neural network model in which copies of amplitude embedded data are followed by a trainable unitary and measurement of a single-qubit observable.
- DataReuploadingClassifier [18]: A model in which the data is scaled by trainable classical parameters and fed to a quantum circuit via layers of trainable angle embedding. One single-qubit rotation thereby takes three input features at once. The aim of training is to maximise the fidelities of the output qubits to the desired class state (either $|0\rangle$ or $|1\rangle)$.
- DressedQuantumCircuitClassifier [51]: A model that preprocesses the data using a classical neural network, which is fed into a generic quantum circuit via angle embedding. The expectation values at the output are then post-processed by another classical neural network for prediction. Both the classical neural networks and the quantum circuit are trainable.
- IQPVariationalClassifier [52]: A model that encodes the input features via an angle embedding using a circuit structure inspired from Instantaneous Quantum Polynomial (IQP) circuits, which are known to be hard to simulate classically. [59].
- QuantumBoltzmannMachine [53]: A model inspired from classical Boltzmann machines that trains the Hamiltonian of a multi-qubit Gibbs state and measures an observable on a subset of its qubits for prediction.
- QuantumMetricLearner [56]: A model that optimises a trainable embedding of the data to increase the distance of states with different labels. Prediction relies on evaluating state overlaps of a new embedded input with the embedded training data.
- TreeTensorClassifier [19]: A model that uses amplitude embedding followed by a trainable unitary with a tree-like structure that is designed to avoid vanishing gradients.

The choice of loss function for training varies among models ${ }^{9}$. Two models use a cross entropy loss, three use a square loss, one uses a linear loss, and one uses a loss based on distances between embedded quantum states (see Table I).

## 2. Quantum kernel methods

Kernel methods [61, 62] form a well known family of machine learning model that take the form

$$
\begin{equation*}
f(\boldsymbol{x})=\sum_{i} \alpha_{i} k\left(\boldsymbol{x}_{i}, \boldsymbol{x}\right) \tag{4}
\end{equation*}
$$

where $\alpha_{i}$ are real trainable parameters and $k$ is a kernel function: a positive definite function that measures the similarity between data points. Since the values $\alpha_{i}$ typically take the same sign as $y_{i}$, these models have the flavour of a weighted nearest neighbour classifier in which the distance to neighbours is mediated by the kernel function. A fundamental result in the theory of kernel methods states that any such model is equivalent to a linear classifier in a potentially infinite dimensional complex feature space $|\phi(\boldsymbol{x})\rangle$ defined via the inner product

$$
\begin{equation*}
k\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\left\langle\phi(\boldsymbol{x}) \mid \phi\left(\boldsymbol{x}^{\prime}\right)\right\rangle \tag{5}
\end{equation*}
$$

and a rich mathematical theory of these methods has been developed as a result. To train such models, one typically seeks a maximum margin classifier of the data, which can be shown to be equivalent to solving a simple convex optimization problem in the parameters $\alpha_{i}$ [61].

A quantum kernel method is one in which the kernel function is evaluated with the aid of a quantum computer. A common strategy is to define an embedding $\rho(\boldsymbol{x})$ of the classical data into quantum states, and use the function

$$
\begin{equation*}
k\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\operatorname{tr}\left[\rho\left(\boldsymbol{x}_{i}\right) \rho\left(\boldsymbol{x}_{j}\right)\right] \tag{6}
\end{equation*}
$$

which is a kernel by virtue of being an inner product, and can be evaluated by methods that evaluate state overlaps. In principle, the kernel function and any of the quantum circuits needed to evaluate it may also depend on trainable parameters, and in some models these are optimized[^5]

as part of a wider training pipeline. For a deeper discussion on the connection between kernel methods and quantum models, we point the reader to [63].

We implemented three quantum kernel methods:

- IQPKernelClassifier [52]: A model that uses a quantum kernel of the form of Eq. (6), where the embedding is the same IQP-inspired angle embedding used in IQPVariationalClassifier.
- ProjectedQuantumKernelClassifier [54]: A model that attempts to avoid problems related to the exponential size of Hilbert space by projecting the embedded state to a smaller space defined via its reduced density matrices, and using a Gaussian kernel in that space. The initial quantum embedding corresponds to a trotterized evolution given by a 1D Heisenberg Hamiltonian.
- QuantumKitchenSinks [55]: A model in which input data is first transformed by random linear maps and then by a number of fixed quantum circuits via angle embedding. Output bit-string samples of the quantum circuits are concatenated to form a feature vector that is fed to a linear classifier for training and prediction.

We loosely include QuantumKitchenSinks in the above list since, as in quantum kernel methods, the linear classifier finds an optimal hyperplane in a feature mapped space given by the quantum model. However, note that the implementation does use a SVM as in the other two models.

## 3. Quantum convolutional neural network models

Our third family consists of models can be seen as analogues of convolutional neural networks [64, 65]: a class of classical neural network model designed for computer vision related tasks which exploit a form of translation symmetry between layers called "equivariance" [66]. The literature features a large number of quantum convolutional models, undoubtedly due to the enormous success of classical convolutional models and the general tendency to 'quantumify' any model that gains sufficient fame in the classical literature. We do not attempt to capture these models in a strict mathematical definition, but rather identify them by the fact that they are examples of quantum neural network models that-like classical convolutional models - also exploit a form of translation symmetry [67], and are therefore designed for data that respects such symmetries.

We study two such models:

- QuanvolutionalNeuralNetwork [15]: This model is equivalent to a classical convolutional neural network in which the convolutional filter that defines the first layer is evaluated by a random quantum circuit that encodes the input data via angle encoding.
- WeiNet [20]: Uses a quantum version of a convolutional layer based on amplitude embedding and linear combination of unitaries, with the goal of having fewer trainable parameters than a classical convolutional neural network.


## 4. Classical models

In addition to the above quantum models, we use a set of standard classical models, which we define as algorithms that are classically simulable at scale (even if they might be quantum-inspired).

Typical strategies for selecting a baseline to compare quantum models with try to match architectural components, like the number of parameters or layers in the model. However, the role that these components play and the effect they have on the computational resources differ vastly between architectures, and we do not believe that this comparison is meaningful in our context - much like one does not enforce the same number of parameters when comparing kernel methods with neural networks.

Instead, we employ two selection criteria for the classical competitors. On the one hand we use a standard feed-forward neural network, a support vector classifier with Gaussian kernel, and a convolutional neural network model as natural equivalents to the three quantum model families defined above. The first two of these were implemented using scikit-learn's MLPClassifier and SVC classes, and the third that we call ConvolutionalNeuralNetwork was implemented using Flax [68]. Similarly to the quantum model selection, these are out-of-the-box versions of models that represent popular ideas in machine learning research, and that are widely used by practitioners for small-scale datasets. They are not intended to be state-of-the-art models.

We also conduct experiments with models that are classically simulable but inspired by quantum models. For example, the SeparableVariationalClassifier represents a standard quantum neural network with layer-wise, trainable angle embedding but uses no entangling gates or non-product measurements. The SeparableKernelClassifier is a support vector machine with a quantum kernel that embeds data using (non-trainable) layers of non-entangling angle embedding. The quantum circuits of these models can be simulated by circuits consisting of a single qubit only.

## C. Implementation

We develop a software pipeline built from PennyLane [11], JAX [69], optax [70] and scikit-learn [71] as our software framework. While PennyLane's differentiable state-vector simulators in combination with JAX's justin-time compilation tools allow us to run trainable quantum circuits, scikit-learn provides a simple API for model training and evaluation, as well as a wealth of machine learning functionalities like data pre-processing, crossvalidation and performance metrics. It also offers a broad range of standard classical machine learning models to compare with. The code can be found in the repository https://github.com/XanaduAI/qml-benchmarks.

When implementing the models from the selected papers we followed these principles:

1. Faithful implementation: We carefully deduced the model design and training procedure from the paper. If specific details needed for implementation were missing in the paper, they were defined based on what appeared natural given our own judgement (see also Appendix C).
2. Convergence criteria: All quantum neural network models and quantum convolutional models used the same convergence criterion, which was based on tracking the variance of the loss values over time and stopping when the mean loss value did not change significantly over 200 parameter updates ${ }^{10}$ (see Appendix D for details). Loss histories were routinely inspected by eye to ensure the criterion was working as desired. Quantum kernel methods followed the default convergence criterion of the specific scikit-learn model class (SVC, Logistic Regression) that they employ. Pure scikit-learn baseline models (MLPClassifier, SVC) used their default convergence criterion except for the max_iter parameter of MLPClassifier which was increased to 3000 . If a training run did not converge during grid search, which happened very rarely, that particular result was ignored in the hyperparameter optimisation procedure.
3. Batches in SGD: Since the batch size for models using gradient descent training plays an important role in the runtime and memory resources, we did not optimize this hyperparameter with respect to the performance of the model, but with respect to simulation resources. Note that in an unrelated study, Moussa et al. [25] found that the batch size did not have a significant impact on model performance. All models use a batch size of 32 , except for the computationally expensive QuantumMetricLearner that uses a batch size of 16 .
4. Data preprocessing: Not all models define a data preprocessing strategy, even though most data em-[^6]

| Model | Embedding | Measurement | Hyperparameters | Classical processing | Loss |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Quantum Neural Networks |  |  |  |  |  |
| Circuit Centric <br> Classifier | copies of ampli- <br> tude embedding | single-qubit Pauli <br> $Z$ | - learning_rate <br> - n_input_copies <br> - n_layers | trainable bias added to <br> output of circuit | square |
| Data Reupload- <br> ing Classifier | layers of train- <br> able angle <br> embedding | multi-qubit Pauli <br> $\mathrm{Z}$ | - learning_rate <br> - n_layers <br> - observable_type | input features and <br> output fidelities mul- <br> tiplied by trainable <br> weights | square |
| Dressed Quan- <br> tum Circuit Clas- <br> sifier | layers of angle <br> embedding | multi-qubit Pauli <br> $\mathrm{Z}$ | - learning_rate <br> - n_layers | input and output fea- <br> tures processed by <br> trainable classical neu- <br> ral network | cross en- <br> tropy <br> (softmax) |
| IQP Variational <br> Classifier | layers of IQP- <br> inspired angle <br> embedding | two-qubit $\mathrm{ZZ}$ | - learning_rate <br> - n_layers <br> - repeats | input extended by <br> product of features | linear |
| Quantum Boltz- <br> mann Machine | angle embedding | multi-qubit Pauli <br> $\mathrm{Z}$ | - learning_rate <br> - temperature <br> - visible_qubits | input features mul- <br> tiplied by trainable <br> weights | cross <br> entropy |
| Quantum Metric <br> Learner | layers of QAOA- <br> inspired angle <br> embedding | pairwise state <br> overlaps | - learning_rate <br> - n_layers | None | distance <br> between <br> embedded <br> classes |
| Tree Tensor <br> Classifier | amplitude <br> embedding | single-qubit Pauli <br> $Z$ | - learning_rate | trainable bias added to <br> output of circuit | square |
| Quantum Kernel Methods |  |  |  |  |  |
| IQP Kernel <br> Classifier | layers of angle <br> embedding | pairwise state <br> overlaps | - repeats <br> - $C$ (SVM regularisation) | quantum kernel used <br> in SVM | hinge |
| Projected Quan- <br> tum Kernel | layers of <br> Hamiltonian- <br> inspired angle <br> embedding | $\mathrm{X}, \mathrm{Y}, \mathrm{Z}$ on all <br> qubits | - trotter_steps <br> - $C$ (SVM regularisation) <br> - $t$ (evolution time) <br> - gamma_factor (RBF <br> bandwidth) | quantum kernel used <br> in SVM | hinge |
| Quantum <br> Kitchen Sinks | angle embedding | computational <br> basis samples | - n_episodes <br> - n_qfearures | quantum features used <br> in logistic regression | cross <br> entropy |
| Quantum Convolutional Neural Networks |  |  |  |  |  |
| Quanvolutional <br> Neural Network | angle embedding | computational <br> basis samples | - learning_rate <br> - n_qchannels <br> - qkernel_shape <br> - kernel_shape | classical convolutional <br> neural network | cross en- <br> tropy <br> (sigmoid) |
| Wei Net | amplitude <br> embedding | single- and <br> double-qubit Z | - learning_rate <br> - filter_type | single layer neural net- <br> work applied to the <br> circuit output values | cross en- <br> tropy <br> (sigmoid) |

TABLE I. Overview of models used in the benchmarks. For definitions of the terms, consult the glossary in Appendix B. More details on the models are found in Appendix C.
beddings are sensitive to the range on which the input data is confined. If no preprocessing was specified, we pre-scaled to natural intervals; for example, if angle embedding is used all features were scaled to lie in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ (also consistent with findings in [25]). For models that use amplitude embedding, we set the first $d$ values of the state vector equal to $\boldsymbol{x}$, pad the remaining values with a constant $1 / 2^{n}$ (with $n$ the number of qubits), then normalize all values as required.

## D. Difficulties of scaling to higher qubit numbers

As quantum software gets better, simulations in the two-digit qubit numbers have become a standard for benchmark studies. However, quantum machine learning benchmarks pose particular challenges to numerical computations, as a circuit is not run once, but hundreds or thousands of times to compute one test accuracy during hyperparameter optimisation using the complex workflows quantum models are constructed of. Although hyperparameter search was parallelized such that each 5fold cross validation run (i.e. training a model 5 times for fixed a hyperparameter setting) was sent to a cluster of 10 CPUs with a maximum compute time of 24 hours, we nevertheless ran into runtime limits for even modest qubit numbers.

There were different causes for models to be computationally demanding. For example, ProjectedQuantumKernel has a large hyperparameter grid which consists of 108 distinct hyperparameter settings. Since we perform 5 -fold cross validation, we therefore need to train 540 models for every dataset we consider, and training each model involves evaluating hundreds of quantum circuits. For other models the number of circuits required poses the biggest issue. For example, to train on $N=250$ datapoints we need to evaluate a quadratic number of circuits - around 30,000 - for IQPKernelClassifier. QuantumMetricLearner, instead, requires multiple circuits for prediction, as it computes the distance of a new data sample to training samples. The QuanvolutionalNeuralNetwork model involves performing a convolution over the input image, and needs to evaluate a number of circuits that scales with the number of pixels, which in case of a $16 \times 16$ pixel grid amounts to many millions of circuit evaluations. Another costly example is the QuantumBoltzmannMachine which is based on simulating a density matrix that requires quadratically more memory than state vector simulation. For large qubit numbers, some quantum neural network models were also very slow to reach convergence. This was particularly the case for DressedQuantumCircuit, which for a 15 qubit circuit failed to converge after 24 hours of training on some of the datasets. As a result of these challenges, computing a quantum model's test accuracy on a single dataset can already take over a day on a cluster for single-digit qubit numbers, and reaching of the order of 20 qubits becomes unfeasible for the scope of our study.

There are many mitigation strategies that can speed up simulations, such as a more judicious choice of parameter initialization, more resources on the cluster, better runtime-versus-memory trade-off choices, snapshotting jobs or by turning to GPU simulation such as available in the PennyLane Lightning suite. The variety of computational bottlenecks however requires different solutions to be found for different models, and we therefore have to defer more extensive code optimisation to future work on the benchmark software package.

## III. DATASETS

Choosing meaningful datasets for general benchmarking studies is difficult, and, as discussed in Section I, can have a huge impact on the findings. For example, should we use datasets that suit the inductive bias of the quantum models, since these would be likely future applications? Shall we use small datasets that were relevant for machine learning in the 1990s? Shall we use popular current-day benchmarking tasks and reduce them to manageable scales? Should we focus on data in the form of quantum states [72, 73]? While we do not claim to provide satisfying answers to these questions - an endeavour that is worth a multi-year research programme and will unlikely find a single answer - we want to make transparent the rationale behind choosing the 6 different flavours of data that we employ in this study, and what we expect them to measure.

We followed three overarching principles: Firstly, we predominantly use artificially generated data which allows us to understand and vary the properties and size of the datasets. This may limit conclusions with respect to "real-world" data, but is an essential ability in the early research stage that quantum machine learning is in. Secondly, we aim at maximising the diversity of the datasets by using fundamentally different functional relationships and procedures as generators - in the hope to increase the chance that consistent trends found in this study may be found in other data as well. Thirdly, in the last three out of six data generation procedures introduced below we follow the "manifold hypothesis" [74-76], which states that typical data in modern machine learning effectively lives on low-dimensional manifolds.

With this in mind we define 6 data generation procedures, with which we generate data for 10 benchmarks (in the following named in capital letters). Each benchmark consists in turn of several datasets that differ by varying parameters in the data generation procedure (in most cases the input dimension). Overall, the benchmarks consist of 160 individual datasets. While the 6 data generation procedures and their associated benchmarks are summarised in the following list and illustrated in Figure 5, the precise generation settings can be found
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-11.jpg?height=924&width=1750&top_left_y=188&top_left_x=184)

FIG. 5. Illustrative examples of datasets created by the different data generation procedures. For the scatter plots, the two classes are shown in blue and orange, and training points are shown in round vs. test points in an ' $x$ ' shape. The linearly separable pannel shows data for the Linearly separable benchmark in 2 and 3 dimensions. The left two plots for the MNIST data correspond to $2 \mathrm{~d}$ and 3d MNIST PCA data, and the rightmost image shows examples from the MNIST-CG dataset for $32 \mathrm{x}$ $32,16 \times 16,8 \times 8$ and $4 \times 4$ pixel grids. The hidden manifold examples correspond to a $1 \mathrm{~d}$ (left) and $2 \mathrm{~d}$ (center) and $3 \mathrm{~d}$ (right) manifold embedded into 3 dimensions. The bars and stripes panel shows examples from the BARS \& STRIPES dataset for a $16 \mathrm{x}$ 16 pixel grid. The examples from the TWO CURVES DIFF benchmark show a degree of $2,10,20$ for the Fourier series, embedding the curves into 10 dimensions (of which three are plotted). The hyperplanes pannel shows data from the HYPERPLANES DIFF benchmark, where there are two (left) and five (right) hyperplanes used to decide the class labels.

in Appendix E.

1. Linearly separable. This data generation procedure consists of linearly separable data and serves as the "fruit-fly" example of learning: it is easy to understand and has well-defined performance baselines, since it is known to be solvable by a perceptron model - a neural network with one layer and output "neuron" - since the early days of artificial intelligence research $[77,78]^{11}$. The datasets are generated by sampling inputs uniformly from a $d$-dimensional hypercube and dividing them into two classes by the hyperplane orthogonal to the $(1, \ldots, 1)^{T}$ vector (including a small data-free margin). The benchmark that we will refer to as LINEARLY SEPARABLE consists of 19 datasets that vary in the dimension $d=2, \ldots, 20$ of the input space.
2. Bars and stripes. As a second "fruit-fly" task, but this time tailor-made for the convolutional models,[^7]

we create images of noisy bars and stripes on 2dimensional pixel grids. These datasets are among the simplest examples of translation invariant data and can thus be used as a sanity check of convolutional models. The data generation procedure involves sampling a number of images with values $\pm 1$, corresponding to either bars or stripes, and adding independent Gaussian noise to each pixel value with standard deviation 0.5. The BARS \& STRIPES benchmark consists of four datasets where we vary the image size between $4 \times 4,8 \times 8,16 \times 16$ and $32 \times 32$.

3. Downscaled MNIST. While we cautioned against the use of downsized MNIST datasets in Section IB, we want to report on this ubiquitous dataset here for the sake of comparability with other studies. We define three benchmarks: For the quantum neural network models we use Principal Component Analysis (PCA) to reduce the dimensions to $d=2, \ldots, 20$ (MNIST PCA). For quantum kernel methods, which need to simulate up to $N(N-1) / 2$ quantum circuits during training if $N$ is the number of training samples, 250 training and
test points are subsampled from the MNIST PCA datasets (MNIST PCA ${ }^{-}$). For the CNN architectures we reduce the resolution of the images by "coarsegraining" or extending the images to size $4 \times 4$, $8 \times 8,16 \times 16$, and $32 \times 32$ in order to keep the spatial pattern of the data intact (MNIST-CG). The three benchmarks consist of 42 datasets in total.
4. Hidden manifold model. Goldt et al. [79] introduced this data generation procedure as a means to probe the effect of structure in the data, such as the size of a hidden manifold conjectured to control the difficulty of the problem, on learning. In particular, it allows the analytical computation of average generalisation errors, a property that could be of interest in quantum machine learning beyond this study. We generate inputs on a low-dimensional manifold and label them by a simple neural network initialised at random. The inputs are then projected to the final $d$-dimensional space. We generate two benchmarks this way: HIDDEN MANIFOLD varies only the dimension $d=2, \ldots, 20$ and keeps the dimension of the manifold at $m=6$, while HIDDEN MANIFOLD DIFF keeps the dimension constant at $d=10$ and varies the dimensionality $m$ of the manifold between $m=2, \ldots 20$; in total we produce 38 datasets.
5. Two curves. This data generation procedure is inspired by a theoretical study [80] that proves how the performance of neural networks depends on the curvature and distance of two 1-dimensional curves embedded into a higher-dimensional space. Here we implement their proposal by using lowdegree Fourier series to embed two sets of data sampled from a 1-d interval - one for each class - as curves into $d$ dimensions while adding some Gaussian noise. The curves are embedded using identical functional relationships, except from an offset applied to one of them, which controls their distance. We generate two benchmarks with in total 38 datasets. TWO CURVES fixes the degree to $D=5$, offset to $\Delta=0.1$ and varies the dimension $d=2, . ., 20$. TWO CURVES DIFF fixes the dimension $d=10$ and varies the degree $D$ of the polynomial $D=2, \ldots, 20$ while adapting the offset $\Delta=\frac{1}{2 D}$ between the two curves.
6. Hyperplanes and parity. Finally, we devise a data generation procedure that fixes several hyperplanes in a $k$-dimensional space and labels randomly sampled points consulting the parity of perceptron classifiers that have these hyperplanes as decision boundaries. In other words, a label tells us whether a point lies on the "positive" side of an even number of hyperplanes. The motivation for this data generation procedure is to add a labeling strategy that requires information about the "global" structure of the problem, i.e. the position of all hyperplanes.

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-12.jpg?height=328&width=680&top_left_y=188&top_left_x=1183)

FIG. 6. Average difficulty score with respect to the 22 difficult measures computed by the ECol package ${ }^{a}$ introduced in Lorena et al. [81] for some of the datasets. The datasets ending in "DIFF" (blue, red and green curves) depend on a variable between 2 and 20 that we claim controls their difficulty, which is supported by the quantifier shown here. The other datasets vary the dimension, and - with the curious exception from LINEARLY SEPARABLE - decrease in difficulty when the input space gets larger. Note that the measures exhibit a huge variance, and the results from this or other data complexity measures should be interpreted with care.

${ }^{a}$ https://github.com/lpfgarcia/ECoL

A single benchmark, HYPERPLANES DIFF, fixes the dimension of the data to $d=10$ and varies the number $k$ of hyperplanes $k=2, \ldots, 20$ defined on a 3-dimensional submanifold.

As seen, some of the 10 benchmarks consist of datasets that vary the input dimension where others vary parameters that supposedly control the complexity of the data. While the controversial debate about the best way of quantifying the complexity of data (for example, [82-84]) lies outside of the scope of this paper, we give some support to the claim of an increasing complexity by reporting the average difficulty score of the measures proposed in Lorena et al. [81] extending a seminal paper from 2002 [85] in Figure 6.

## IV. HYPERPARAMETER TUNING

Hyperparameter optimisation is one of the most important steps in classical machine learning to allow models to reveal their true potential. Likewise, quantum machine learning models tend to show a wide variety in performance depending on hyperparameters such as the number of layers in the ansatz. Even seemingly inconspicious hyperparameters such as the learning rate of an optimiser can influence generalisation errors significantly [25]. As mentioned in Section I A, including a wide enough hyperparameter search grid can make baseline models match the performance of state-of-the-art methods.

Hyperparameter optimisation is also one of the most cumbersome aspects of training (quantum) machine learning algorithms since it involves searching a large configuration space that is not amenable to gradient-
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-13.jpg?height=1042&width=828&top_left_y=172&top_left_x=190)

FIG. 7. Ranges of test accuracies of models during grid search to find the best hyperparameters. There is a large variation in the results which becomes very important to consider as poor hyperparameter choices could lead to misleading conclusions about the power of (quantum) machine learning models.

based optimization ${ }^{12}$, while each point or hyperparameter configuration involves several training runs during procedures like cross-validation.

We conduct an extensive hyperparameter search for all models and datasets in all our experiments, using a full grid search algorithm implemented by the scikit-learn package [71] with the default five-fold cross-validation, using the accuracy score ${ }^{13}$ to pick the best model. While there are more sophisticated techniques [89, 90], a full grid search has the advantage of simplicity, and allows us to extract and study unbiased information about the hyperparameter landscapes. We remark that the number of hyperparameters varies significantly from model to model since our aim was to follow the proposals in the original papers. In some cases, we were forced to select

12 Although recently, to address such issues, techniques such as implicit differentiation have been adapted to perform gradientbased optimization of continuous-valued hyperparameters [86, 87], but exploring these techniques exceed the scope of this study.

13 The accuracy can be an overly simplistic measure for some classification tasks and especially with unbalanced data [88]. However, we utilise it here for its clarity in interpretation, and since our datasets are balanced.

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-13.jpg?height=515&width=702&top_left_y=173&top_left_x=1167)

FIG. 8. Correlation between values chosen for a particular hyperparameter and test accuracy during cross-validation. We show aggregated information across all classifiers, datasets and runs during grid search. Note that some hyperparameters only appear in a single model, while others - such as the number of variational layers (n_layers) - are averaged over several models.

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-13.jpg?height=439&width=615&top_left_y=1022&top_left_x=1210)

FIG. 9. Correlation of the number of quantum layers and test set classification accuracy during cross-validation; averaged across classifiers that use this hyperparameter and all datasets in the three benchmarks reported. While intuitively, more layers should lead to better performance when the dimension of the inputs (MNIST PCA/LINEARLY SEPARABLE) or manifold (HIDDEN MANIFOLD DIFF) grows, we see that the results can vary depending on the benchmark.

a subset due to the exponential increase in grid size with the number of hyperparameters. In these cases, the most relevant hyperparameters were selected by analysing preliminary results first on smaller-scale data. We describe the hyperparameter ranges for each model in Appendix C and summarise a few findings of potential interest here.

First, the performance of both classical and quantum models varies significantly across hyperparameter choices: Figure 7 shows the ranges in the test accuracy during grid search for a select few classifiers and benchmarks, some of which lie between almost random guessing and near-perfect classification. This makes hyperparameter tuning crucial: a highly optimised architecture of one model type can easily outperform another model that has been given less resources to select the right configuration.

We also show the correlations (using the Pearson correlation coefficient) between the accuracy and hyperparameter values in Figure 8 to indicate the relative influence each hyperparameter has on the model performance. We find that aggregated over all datasets and models, some hyperparameters have a high correlation with the accuracy, e.g., increasing the number of episodes in the QuantumKitchenSink model seems to improve its performance, while decreasing the simulation time $t$ improves the ProjectedQuantumKernel model. However, the best hyperparameter choice can vary significantly with the dataset: Figure 9 shows three different benchmarks where the correlation between the number of layers of a quantum model and the test accuracy shows very different trends. In case of the MNIST PCA benchmark, increasing the number of layers leads to higher accuracies, while for the LINEARLY SEPARABLE benchmark we observe the opposite effect. Both trends get stronger for higher input dimensions. At the same time, for the HIDDEN MANIFOLD DIFF benchmark the correlation between accuracy and the number of layers is not significant.

These simple insights from the hyperparameter study show that hyperparameter choice can be very nonintuitive, especially as models increase in size. The hyperparameter choices for a small datasets cannot be expected to be optimal for more complicated scenarios. In case of quantum models, hyperparameter exploration becomes computationally expensive even for moderatesized models.

## V. RESULTS

We finally report our findings from the benchmark study. As a reminder, our goal is twofold: We were motivated to independently test the overwhelming evidence that quantum models perform better than classical ones emerging from benchmarks conducted in the literature. However, this only helps to judge where we currently are, not necessarily where the future potential of quantum models lies. A much more important question we are interested in is which ideas in current near-term quantum machine learning model design hold promise and which ones do not - in other words, what research is worthwhile pursuing in order to use quantum computers effectively for learning. As we will see, the benchmark results give us a number of interesting clues that we will discuss in the next Section VI.

## A. Out-of-the-box classical models outperform quantum models

A very clear finding across our experiments was that the out-of-the-box classical models systematically outperform the quantum models. Figure 10 shows the number of different rankings (first, second, and so forth) across all benchmark experiments we ran for the different

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-14.jpg?height=754&width=873&top_left_y=176&top_left_x=1103)

FIG. 10. Number of rankings (blue/first to red/last) across all datasets that a model was tested on, for the three model families. The models were sorted from top to bottom according to the average normalised rank. The three classical out-ofthe-box models perform best. Note that the total number of benchmarks a model competed in varies due to runtime limitations, and since the convolutional architectures were only tested on MNIST CG and BARS \& STRIPES.

models. The models within a family are sorted according to the expected normalised ${ }^{14}$ rank: If a model came first in a benchmark with 10 competitors (normalised ranks $0.1,0.2, \ldots, 1$ ), and fourth in a benchmark with five competitors (normalised ranks $0.2,0.4,0.6,0.8,1$ ), its expected normalised rank is $(0.1+0.8) / 2=0.45$. Note that this is one choice of many, and employing other reasonable ordering and aggregation mechanisms that we tested change the picture slightly, but not significantly: In all three model families, the prototypical classical models came first.

Breaking down the fully aggregated results, one finds that the rankings of models between benchmarks were surprisingly consistent (see Appendix F for a full report). For example, in four out of seven benchmarks used for QNN models, the MLPClassifier ranked first with the DressedQuantumCircuitClassifier coming second, whereas in the remaining three benchmarks the roles between these (conceptually very similar) models were reversed. The DataReuploadingClassifier and QuantumBoltzmannMachine (which was not run on the 10-d DIFF benchmarks) usually share places three and four. In five out of the seven benchmarks, the bottom ranks are taken by the CircuitCentricClassifier and[^8]

TreeTensorClassifier as the worst-performing models - interestingly the only two models based on amplitude encoding.

For the kernel methods, the support vector machine (SVC) does best on 4 out of 7 benchmarks, showing a very similar behaviour to the other two SVM-based classifiers. Consistently worst-performing is QuantumKitchenSinks, a model that uses a quantum circuit mapping to computational basis samples as a random, non-trainable feature map. $^{15}$

The two quantum convolutional neural networks were also outperformed by the vanilla ConvolutionalNeuralNetwork model. Surprisingly, WeiNet failed entirely to learn the BARS \& STRIPES data; a task which we considered easy for a model of its kind.

## B. No systematic overfitting or scaling effects

There are three interesting general observations of effects one might expect but which we did not observe. These are exemplified with the selected detailed benchmark results shown in Figure 11. Firstly, although not all quantum models - in particular the QNN methods - use explicit regularisers, they do not show systematic overfitting of the training data. (Some exceptions can be seen in the HIDDEN MANIFOLD, HIDDEN MANIFOLD DIFF and TWO CURVES DIFF benchmarks shown in Appendix F, for which also the classical models struggle with overfitting.)

Secondly, we do not observe any improvement in performance of the quantum models relative to the classical baselines for increasing problem difficulties. For the difficulty-controlled benchmarks, HYPERPLANES DIFF (Figure 11) and HIDDEN MANIFOLD DIFF (Appendix F), the trends of the quantum models' test accuracies generally follow the trend of the classical model. For the difficulty-controlled benchmark TWO CURVES DIFF (Appendix F), the quantum models perform worse than the corresponding classical method as the difficulty is increased. Interestingly, for the hardest datasets from this benchmark, the classical baseline models achieve high $(>90 \%)$ test accuracy whereas all quantum models appear to struggle. This is somewhat surprising, since the embedded curve that defines the structure of the data is a Fourier series, and one may expect quantum models to have a natural bias for this kind of data [91].

Thirdly, except from the LINEARLY SEPARABLE benchmark we do not observe a significant scaling effect with the size of the dimension; also here quantum models do not get significantly better or worse in performance compared to the classical baseline.

15 It is important to mention how Figure 8 revealed that the number of episodes in the model, which controls the number of feature vectors created from quantum circuit samples, correlated positively with the test accuracy, and allowing for significantly more than 2000 episodes may have boosted this model's performance.

## C. Quantum circuits without entanglement do well

An important question for quantum machine learning benchmarks is how the performance of a model depends on properties that we consider to be "quantum". There are many different definitions of this notion (such as "experiments that produce non-classical statistics"), and without being explicitly stated very often, the definition of "not classically simulatable" dominates the thinking in the quantum machine learning community. An experimental design to probe the question is therefore to replace the quantum model architecture with a circuit that is classically tractable (i.e., it can be simulated at scale) and measure if the performance deteriorates. If not, we have evidence that other properties than "quantumness" are responsible for the performance we see - at least in the small-scale datasets chosen here.

To put "quantumness" to the test in our benchmarks we add the SeparableVariationalClassifier and SeparableKernelClassifier described in Section II to our quiver of non-convolutional models. These are fully disentangled $n$-qubit models that can be divided into $n$ separate single-qubit circuits, and hence easily classically simulated at scale. We do not add a separable convolutional model since the ConvolutionalNeuralNetwork itself can be seen as a special case of one, since it is equivalent to replacing the quantum layer of the QuanvolutionalNeuralNetwork model by a circuit that implements the identity transformation. We already know that this model does not perform worse than the entanglement-using QuantumConvolutionalNeuralNetwork and WeiNet.

Replotting the test accuracies from Figure 11 with the new models added we see in Figure 12 that the non-entangled models do surprisingly well. This can be confirmed by including the separable models in the ranking results from Figure 10 (see Appendix F). One finds that compared to the quantum kernel methods, the SeparableKernelClassifier takes second rank after the SVC. Among the QNNs, the SeparableVariationalClassifier is only consistently beaten by the MLPClassifier, DressedQuantumCircuitClassifier and QuantumBoltzmannMachine, as well as occasionally by DataReuploadingClassifier.

Are these three QNNs better than our disentangled QNN because of their entanglement, or is this due to other design choices? Figure 13 compares the original implementations of these models with variations that remove any entangling gates or measurements from the quantum circuits they use. ${ }^{16}$ The results suggest that the entangling gates do not play a role in the[^9]

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-16.jpg?height=2092&width=1767&top_left_y=201&top_left_x=190)

LINEARLY SEPARABLE
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-16.jpg?height=810&width=844&top_left_y=236&top_left_x=206)

MNIST PCA
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-16.jpg?height=362&width=842&top_left_y=1128&top_left_x=207)

MNIST CG

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-16.jpg?height=355&width=448&top_left_y=1576&top_left_x=207)

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-16.jpg?height=357&width=383&top_left_y=1578&top_left_x=665)
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-16.jpg?height=1722&width=854&top_left_y=213&top_left_x=1080)

- MLPClassifier

$\rightarrow$ CircuitCentricClassifier

-- DataReuploadingClassifier

$\_$DressedQuantumCircuitClassifier

$\rightarrow$ IQPVariationalClassifier

$\rightarrow-$ SVC

- QuantumMetricLearner
- QuantumBoltzmannMachine

--.- ConvolutionalNeuralNetwork

- ProjectedQuantumKerne

$\longrightarrow$ TreeTensorClassifier

—- QuanvolutionalNeuralNetwork

FIG. 11. Selected detailed train and test accuracies for some of the benchmarks. As reflected in the aggregated results in Figure 10, the three classical baseline models usually outperform the quantum models across the quantity that is varied. However, there are nuances: while quantum models perform particularly poorly on the LINEARLY SEPARABLE benchmark, most of them follow the classical model performance closely on the HYPERPLANES DIFF benchmark. The accuracies of the QNN family on MNIST PCA mimic the trend of the classical neural network (MLPClassifier), but with an offset towards lower scores, while most quantum kernel methods perform as well as the SVM (SVC).

LINEARLY SEPARABLE
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-17.jpg?height=782&width=400&top_left_y=233&top_left_x=206)

MNIST PCA

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-17.jpg?height=345&width=401&top_left_y=1096&top_left_x=209)

-     - MLPClassifier

$\sim$ CircuitCentricClassifier

-- DataReuploadingClassifier

$\rightarrow$ DressedQuantumCircuitClassifier $\rightarrow$ IQPVariationalClassifier

$\simeq$ QuantumMetricLearner

-     - QuantumBoltzmannMachine
- TreeTensorClassifier

$\because$ SeparableVariationalClassifier
LINEARLY SEPARABLE

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-17.jpg?height=383&width=402&top_left_y=215&top_left_x=623)

HYPERPLANES DIFF

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-17.jpg?height=357&width=398&top_left_y=659&top_left_x=625)

MNIST PCA-

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-17.jpg?height=350&width=398&top_left_y=1088&top_left_x=625)

FIG. 12. Replotting the test accuracies from Figure 11 while adding SeparableVariationalClassifier and SeparableKernelClassifier. These fully classically simulatable models perform similarly or better than most other quantum models.

top-performing DressedQuantumCircuitClassifier. However, removing entanglement does decrease the test accuracy of QuantumBoltzmannMachine and DataReuploadingClassifier. Whether the "quantumness" of the entangling gates is the deciding factor, or whether the removal of certain gates could be mitigated by a better non-entangling design that enriches the expressivity of the models is an important subject for further studies.
MNIST PCA test

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-17.jpg?height=264&width=744&top_left_y=196&top_left_x=1154)

$23456789 \quad 23456789 \quad 23456789$

number of features

$\rightarrow$ DressedQuantumCircuitClassifier

-- DressedQuantumCircuitClassifierSeparable

$\rightarrow$ DataReuploadingClassifier

-     - DataReuploadingClassifierSeparable

$\rightarrow$ - QuantumBoltzmannMachine

-- QuantumBoltzmannMachineSeparable

FIG. 13. Comparison of the test accuracy of the three QNN models that performed better than the SeparableVariationalClassifier, shown on MNIST PCA up to 9 dimensions, with variations of the models that remove any entangling operations from the circuits. The DressedQuantumCircuitClassifier shows no drop in performance, while DataReuploadingClassifier, and to some extent QuantumBoltzmannMachine, do worse without entanglement.

## VI. QUESTIONS RAISED ABOUT QUANTUM MODEL DESIGN

Benchmarks cannot only give us evidence on which model is better than another, but open up paths to more qualitative questions, for example by systematically removing parts of a model, or by visualising simple cases. We want to give a few examples here.

## A. Do quantum components in hybrid models help?

By far the best QNN model is the DressedQuantumCircuitClassifier, which replaces a layer of a neural network with a standard variational quantum model. The central question for such a hybrid model is whether or not the "quantum layer" plays a qualitatively different role to a possible classical layer. Figure 14a shows the input transformations of the two neural network layers and the quantum layer for a very simple $2 \mathrm{~d}$ dataset, and compares it with the same model in which we exchanged the quantum layer by a neural network of the same architecture as the first layer. In this small experiment, the qualitative effect of both kinds of models is similar, namely to reshape the data to a one-dimensional manifold that facilitates the subsequent linear classification. This is consistent with the fact that in most experiments, the DressedQuantumCircuitClassifier's performance followed the classical neural network closely.

The QuanvolutionalNeuralNetwork is a hybrid model of similar flavour since it adds a quantum circuit
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-18.jpg?height=372&width=748&top_left_y=184&top_left_x=248)

(a) Transformation of a 2-dimensional moons dataset throughout the trained layers of the DressedQuantumCircuitClassifier (top row), compared to a model where we replaced the quantum circuit by another classical layer (bottom row).
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-18.jpg?height=366&width=768&top_left_y=778&top_left_x=229)

(b) The left-most images are two examples of the MNIST-CG data for a $16 \times 16$ pixel grid. The three images to the right show three channels of the data after being feature mapped by the initial quantum layer of QuanvolutionalNeuralNetwork. The quantum feature map appears to produce different, noisier versions of the input image.

FIG. 14. The effect of quantum layers in hybrid classicalquantum neural network models.

as the first layer to a convolutional neural network. In Figure 14b we show the result of this layer for a model with $n_{-}$qchannels $=3$ for two input examples of $16 \times 16$ pixels. It is unclear if the first quantum layer is generally useful for learning from image data, since in most cases the map seems to simply create a noisy version of the original image. Given that the model performs worse than the ConvolutionalNeuralNetwork (at least on the small datasets we were able to get results for), it seems that this feature map actually degrades the data so that it is subsequently more difficult for the classical convolution to learn from.

For the wide range of studies into hybrid quantumclassical neural network architectures, an important question is therefore whether the quantum layer introduces a genuinely different kind of transformation that helps the layer-wise feature extraction, or if it is simply 'not doing much harm'.

## B. What makes data reuploading work?

Besides the hybrid neural network architectures (and the computationally costly QuantumBoltzmannMachine for which we unfortunately only have limited data available) the DataReuploadingClassifier performs relatively well compared to other quantum neural networks of a similar design. What features of the model explain these results?

While the term "data reuploading" is often used rather generally to describe the layer-wise repetition of an encoding, there are a few other distinctive features in the original model we implemented here. For example, the inputs are individually rescaled by classical trainable parameters before feeding them into quantum gates, which in the picture of quantum models as a Fourier series [91] re-scales the frequency spectrum that the Fourier series is built from. Furthermore, there is no separation between the embedding and variational part of the circuit; instead the embedding layer is trainable (see Eq. C1 in App. C), leading to a sort of "trainable data encoding" (a feature that the QuantumMetricLearner also exhibits). Furthermore, the cost function differs from standard approaches as it measures the fidelity to certain states associated with a class label, and contains more trainable parameters. Which of these features is important for the success of the model - or is it a combination of them?

As a first probe into this question, Figure 15 shows the test accuracy of the DataReuploadingClassifier when we remove the three properties - the trainable rescaling, the mixing of variational and embedding circuits, and the peculiar cost function - individually from the model in a small ablation study. We use the MNIST PCA benchmark up to 9 dimensions once more.

The results suggest that both the trainable re-scaling and embedding are crucial for performance, while the special cost function is not. This is particularly interesting, since follow-up papers often only consider the trainable embedding as a design feature - however, the interplay of these two features may be important.

## C. What distance measures do quantum kernels define?

We observe that the quantum kernel methods (except QuantumKitchenSinks, which makes very special design decisions compared to the other two) have a surprisingly similar performance to the support vector machine with a Gaussian kernel. A kernel defines a distance measure on the input space. The distance measure is used to weigh the influence of a data point on the class label of another. What distance measure do the quantum kernels define, and are they similar to the Gaussian kernel?

Figure 16 shows the shape of the kernels used by models trained on 2-dimensional versions of our benchmarks. We include the SeparableKernelClassifier for interest, and define the kernel of QuantumKitchenSinks as

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-19.jpg?height=610&width=740&top_left_y=172&top_left_x=237)

MNIST PCA

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-19.jpg?height=366&width=414&top_left_y=240&top_left_x=411)

-- DataReuploadingClassifier

$\rightarrow$ DataReuploadingClassifierNoScaling

$\rightarrow$ DataReuploadingClassifierNoCost

$\rightarrow$ DataReuploadingClassifierNoTrainableEmbedding

FIG. 15. Comparison of the test accuracy of the Data Reuploading Classifier on MNIST PCA up to 9 dimensions with three modifications of the original implementation: The NoCost variation replaces the special cost function with a standard cross entropy cost function, the NoScaling version removes the trainable parameters multiplied to the inputs, and the NoTrainableEmbedding variation first applies all data encoding gates and then all variational gates. While the former has only a small influence on the performance, the latter two seem to both change the accuracy scores of higher dimensions significantly.

the inner product of the feature vectors created by the quantum circuit. With a few occasional exceptions - notably on TWO CURVES, a dataset that seems to require very narrow bandwidths in kernels and encourages quantum kernels to extend into their periodic regions - the kernel shapes do indeed resemble the SVC's Gaussian kernel.

Does that mean that quantum kernels are just approximations to a method that has been around for decades? While 2-dimensional visualisations of the kernel function can help us gain some understanding, we need to look into higher dimensions. Here geometric visualisations become tricky to interpret, and it can be useful to compare the actual Gram matrices $G$ with entries $G_{i j}=\kappa\left(\mathbf{x}_{i}, \mathbf{x}_{i}\right)$ for pairs of training points $\mathbf{x}_{i}, \mathbf{x}_{j}$, which are used when fitting the models ${ }^{17}$. A popular measure is the kernel alignment [92] that computes the product of corresponding entries of two matrices. However, this measure makes it hard to distinguish regions in which one Gram matrix has finite values and another near-zero ones from regions where both have near-zero values. To achieve a more insightful comparison we rescale the Gram matrices to have entries in $[0,1]$ and use the distance

$$
d\left(G \mid G^{\prime}\right)=\frac{\sum_{i j}\left(G_{i j}-G_{i j}^{\prime}\right)^{2}}{|G|}
$$[^10]

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-19.jpg?height=1106&width=895&top_left_y=176&top_left_x=1100)

FIG. 16. Kernels used by models for different 2-dimensional datasets, selecting the best hyperparameters found during grid search. The plots are generated by fixing one datapoint at $(\pi / 2, \pi / 2)$ in the $x$-y-plane and varying the other one, while plotting the kernel value for the two datapoints on the $\mathrm{z}$-axis. The kernel value shows how much weight the classifier gives a potential datapoint in the $\mathrm{x}$-y-plane when predicting the class of the fixed point. While there are some variations, most quantum kernel methods have a similar radial shape to the Gaussian kernel.

where $|G|$ refers to the number of entries in $G$ (which is the same as $\left.\left|G^{\prime}\right|\right)$. This can be seen as a "difference measure" where 0 signals identical Gram matrices, and 1 maximally different ones.

The results, of which some representative examples in $2 \mathrm{~d}$ versus $10 \mathrm{~d}$ are shown in Figure 17, give a slightly different picture that can only faintly be seen in the 2d kernel plots. In higher dimensions, only the ProjectedQuantumKernel resembles the SVC model, while the other three quantum kernels resemble each other. ${ }^{18}$ Does this mean that the projected quantum circuit is not so much responsible for learning, but rather the subsequent Gaussian kernel applied to the features computed by the quantum circuit? The other three quan-[^11]

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-20.jpg?height=1036&width=878&top_left_y=154&top_left_x=168)

FIG. 17. Average squared difference $d\left(G \mid G^{\prime}\right)$ between the entries of the Gram matrices of different kernels. The Gram matrices are constructed from the training data using the best hyperparameters found during grid search for the particular dataset. Here we show the four benchmarks that contained datasets in both 2 and 10 dimensions. In higher dimensions, the Gram matrices of quantum models tend to look similarly, with the exception of the ProjectedQuantumKernel model, that in turn tends to resemble the SVC.

tum kernels, in turn, produce very similar Gram matrices in high dimensions. Do most "non-projected" quantum kernel designs share this behaviour?

Overall, attempting to understand the distance measure induced by a quantum kernel in high dimensions, rather than only focusing on its classical intractability, is an important open challenge in which benchmarks can help.

## D. Why are polynomial features not working?

Another consistent finding on the lower end of the spectrum is that the two QNN models that encode data via amplitude encoding, the CircuitCentricClassifier and TreeTensorClassifers, perform poorly. A ready explanation for the TreeTensorClassifier is that neither the amplitude encoding nor the variational unitary evolution of the model can change the distance between (pre-processed) input data points. Moreover, due to the goal of avoiding vanishing gradients, the model uses a very limited number of parameters that scales only log-
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-20.jpg?height=554&width=718&top_left_y=156&top_left_x=1170)

-r- CircuitCentricClassifier

-.-. TreeTensorClassifier

FIG. 18. The average test accuracy for selected datasets of input dimension six when multiplying the input features by different scaling factors. The models were trained 5 times on each dataset, and the best hyperparameters were taken from our regular hyperparameter search. The default scaling we used in the previous results is shown by the dashed line, and is not always the optimal value, which could sometimes explain the poor performance of models based on amplitude encoding.

arithmically with the number of qubits; for example, for a problem with 16 features, one has only 7 variational parameters to train. This severely limits the expressivity of the model.

However, these arguments do not hold for the CircuitCentricClassifier, which uses an expressive variational circuit and several copies of the amplitude encoded state. This creates an overall state that is a tensor product $\mathbf{x} \times \mathbf{x} \times \ldots$ of the (pre-processed) input. Since we allow for up to three copies in hyperparameter search, this model has the ability to build up to 3rd order polynomial features like $\mathbf{x}_{1}^{3}$ or $\mathbf{x}_{1} \mathbf{x}_{6}^{2}$ through the embedding. It is therefore interesting to ask whether the lack in performance of the CircuitCentricClassifier is due to polynomial features not being useful in the datasets we consider here, or if a degree of 3 is too low. Looking at the hyperparameter optimisation (compare also the low correlation of n_input_copies with the test accuracy in Figure 8), we find that the optimal number of copies is often 1 or 2 instead of the maximum possible 3, which means that the model does not systematically prefer more complex features.

Note that another explanation is that we missed an important hyperparameter during grid search. Figure 18 plots the average accuracy of relevant models over all datasets that have an input dimension of 6 , when scaling the input data by different amounts. Scaling the data to smaller values than we used as default can have a beneficial effect, even when selecting the best hyperparameters found using this default scaling.

Overall, an important question is whether there are datasets and hyperparameter configurations for which the ability to construct high-order polynomial features using quantum circuits would be interesting - or is am-
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-21.jpg?height=664&width=722&top_left_y=178&top_left_x=254)

FIG. 19. Decision boundary of selected models trained on the 2d linearly separable dataset. Training points are shown in round and test points in triangular shape. All four models use angle embedding, therefore forming decision boundaries based on periodic functions. This may introduce an inductive bias that is not suited for data requiring linear decision boundaries in higher dimensions.

plitude embedding just not a good design choice?

## E. Why do quantum models struggle with the linearly separable benchmark?

As discussed above, the behaviour of both QNNs and quantum kernel methods on the LINEARLY SEPARABLE benchmark is a notable outlier, as the performance of nearly all quantum models is poor and gets worse with the dimensionality of the data while the classical models achieve near-perfect classification. ${ }^{19}$ But is this really a property of linearly separable datasets? It is more likely that many of the quantum models in our selection, in particular those using angle embedding, have an inductive bias against linear decision boundaries like the one implanted into the LINEARLY SEPARABLE data generation procedure. (Instead, they may do very well on linearly separable data of Gaussian blobs.) For example, Figure 19 shows the decision boundaries for selected models on the 2-dimensional dataset of the LINEARLY SEPARABLE benchmark using the best hyperparameters found during grid search - all models try to fit the data with periodic regions. While this may work for low dimensions, the constraints could be incompatible with linear decision boundaries in higher dimensions. An interesting theoretical study would be to analyse which kinds of[^12]

data this behaviour is or is not suited for, and what the resource requirements, such as more embedding layers, are necessary to overcome the problem.

In contrast, most quantum models performed almost similar to the classical baselines on the HYPERPLANES DIFF benchmark, suggesting that perhaps none of the models contained a bias that was particularly aligned with this data. As a reminder, the data generation procedure was intended to test the ability of models to detect a "global" labeling rule, for which the positioning of several hyperplanes in a low-dimensional manifold is relevant.

## VII. CONCLUSION

One of the most important lessons one learns when undertaking a study of this kind is that benchmarking is a subtle art that is filled with difficult decisions and potential pitfalls. As such, the benchmarking of new quantum machine learning proposals should be considered an extreme challenge, rather than as a task that can be safely given to lesser experienced researchers or relegated to the afterthought of a study. It is hard to coax robust and meaningful results from systems as complex as machine learning models trained on data, and non-robust claims can have a profound impact on where the community searches for good ideas. The single most effective remedy is scientific rigour in the methodological design of studies, including extensive reporting on the choices made and their potential bias on the results.

Perhaps the most important question regarding the future of benchmarking of quantum models is what kind of data to choose. More studies that focus on questions of structure in data are crucial for the design of meaningful benchmarks: What mathematical properties do real-world applications of relevance have? How can we isolate, downscale and artificially reproduce them? How can we connect them to the mathematical properties of quantum models? This is a task shared with classical machine learning research, but further exacerbated by the fact that the areas in which quantum computers can unlock new capabilities of learning are not yet identified. Using the right data and finding quantum models with an advantage hence becomes a "chicken and egg problem" that is best tackled from two sides; however in the current literature, the focus on model design dominates by far.

Aside from these conceptual challenges, benchmarking quantum machine learning models also poses a formidable challenge to current quantum software. On the one hand, this is due to the immense resource requirements of hyperparameter optimisation. On the other hand, quantum machine learning models are usually elaborate pipelines of hybrid quantum-classical systems, each of which requires different logics for performance tools like compilation, parallelisation, caching or GPU use. Furthermore, results on small datasets cannot be used to reason about larger datasets, as we know from deep
learning that big data leads to surprisingly different behaviour. There is hence a need to study how results scale to larger datasets, which typically pushes the number of qubits to the limits of what is possible today.

Finally, instead of considering rankings only, benchmarks can help us to gain qualitative insights into which parts of a model design are crucial and which ones replaceable. Since the question of quantum advantage is undercurrent to almost all studies in quantum machine learning, a particularly important experiment that should become a standard in benchmarking is to remove "quantumness" from a model in a non-invasive manner and test if the results hold. Of course, there are other ways than removing entanglement to make models classically tractable or "non-quantum", such as limiting gates to the Clifford family, replacing unitary transformations by stochastic ones (see Appendix in [93]) or using Matrix Product State simulators with low bond dimension. Comparing to such circuit designs will provide invaluable information into the promise of ideas around variational quantum circuits.

## ACKNOWLEDGMENTS

Our computations were performed on the Cedar supercomputer at the SciNet HPC Consortium. SciNet is funded by Innovation, Science and Economic Development Canada; the Digital Research Alliance of Canada; the Ontario Research Fund: Research Excellence; and the University of Toronto.
[1] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, Quantum machine learning, Nature 549, 195 (2017).

[2] M. Schuld and F. Petruccione, Machine learning with quantum computers (Springer, 2021).

[3] K. Ito and K. Fujii, Santaqlaus: A resource-efficient method to leverage quantum shot-noise for optimization of variational quantum algorithms, arXiv preprint arXiv:2312.15791 (2023).

[4] M. Wiedmann, M. Hölle, M. Periyasamy, N. Meyer, C. Ufrecht, D. D. Scherer, A. Plinge, and C. Mutschler, An empirical comparison of optimizers for quantum machine learning with spsa-based gradients, arXiv preprint arXiv:2305.00224 (2023).

[5] M. Kashif, M. Rashid, S. Al-Kuwari, and M. Shafique, Alleviating barren plateaus in parameterized quantum machine learning circuits: Investigating advanced parameter initialization strategies, arXiv preprint arXiv:2311.13218 (2023).

[6] I. N. M. Le, O. Kiss, J. Schuhmacher, I. Tavernelli, and F. Tacchino, Symmetry-invariant quantum machine learning force fields, arXiv preprint arXiv:2311.11362 (2023).

[7] M. T. West, M. Sevior, and M. Usman, Reflection equivariant quantum neural networks for enhanced image classification, Machine Learning: Science and Technology 4, 035027 (2023).

[8] S. Bordoni, D. Stanev, T. Santantonio, and S. Giagu, Long-lived particles anomaly detection with parametrized quantum circuits, Particles 6, 297 (2023).

[9] F. J. Schreiber, J. Eisert, and J. J. Meyer, Classical surrogates for quantum learning models, Physical Review Letters 131, 100803 (2023).

[10] N. Piatkowski, T. Gerlach, R. Hugues, R. Sifa, C. Bauckhage, and F. Barbaresco, Towards bundle adjustment for satellite imaging via quantum machine learning, in 2022 25th International Conference on Information Fusion (FUSION) (IEEE, 2022) pp. 1-8.

[11] V. Bergholm, J. Izaac, M. Schuld, C. Gogolin, S. Ahmed, V. Ajith, M. S. Alam, G. Alonso-Linaje, B. AkashNarayanan, A. Asadi, et al., Pennylane: Automatic differentiation of hybrid quantum-classical computations,
arXiv preprint arXiv:1811.04968 (2018).

[12] E. Farhi and H. Neven, Classification with quantum neural networks on near term processors, arXiv preprint arXiv:1802.06002 (2018).

[13] M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, Circuit-centric quantum classifiers, Physical Review A 101, 032308 (2020).

[14] I. Cong, S. Choi, and M. D. Lukin, Quantum convolutional neural networks, Nature Physics 15, 1273 (2019).

[15] M. Henderson, S. Shakya, S. Pradhan, and T. Cook, Quanvolutional neural networks: powering image recognition with quantum circuits, Quantum Machine Intelligence 2, 2 (2020).

[16] M. Schuld and N. Killoran, Quantum machine learning in feature hilbert spaces, Physical review letters 122, 040504 (2019).

[17] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, Supervised learning with quantum-enhanced feature spaces, Nature 567, 209 (2019).

[18] A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre, Data re-uploading for a universal quantum classifier, Quantum 4, 226 (2020).

[19] K. Zhang, M.-H. Hsieh, L. Liu, and D. Tao, Toward trainability of quantum neural networks, arXiv preprint arXiv:2011.06258 (2020).

[20] S. Wei, Y. Chen, Z. Zhou, and G. Long, A quantum convolutional neural network on nisq devices, AAPPS Bulletin 32, 1 (2022).

[21] J. Kübler, S. Buchholz, and B. Schölkopf, The inductive bias of quantum kernels, Advances in Neural Information Processing Systems 34, 12661 (2021).

[22] M. Larocca, F. Sauvage, F. M. Sbahi, G. Verdon, P. J. Coles, and M. Cerezo, Group-invariant quantum machine learning, PRX Quantum 3, 030341 (2022).

[23] J. Bowles, V. J. Wright, M. Farkas, N. Killoran, and M. Schuld, Contextuality and inductive bias in quantum machine learning, arXiv preprint arXiv:2302.01365 (2023).

[24] F. J. Kiwit, M. Marso, P. Ross, C. A. Riofrío, J. Klepsch, and A. Luckow, Application-oriented benchmarking of quantum generative learning using quark, in 2023 IEEE

International Conference on Quantum Computing and Engineering (QCE), Vol. 1 (IEEE, 2023) pp. 475-484.

[25] C. Moussa, Y. J. Patel, V. Dunjko, T. Bäck, and J. N. van Rijn, Hyperparameter importance and optimization of quantum neural networks across small datasets, Machine Learning 10.1007/s10994-023-06389-8 (2023).

[26] M. Dehghani, Y. Tay, A. A. Gritsenko, Z. Zhao, N. Houlsby, F. Diaz, D. Metzler, and O. Vinyals, The benchmark lottery, arXiv preprint arXiv:2107.07002 (2021).

[27] D. Sculley, J. Snoek, A. Wiltschko, and A. Rahimi, Winner's curse? on pace, progress, and empirical rigor, ICLR Workshop track (2018).

[28] K. Ethayarajh and D. Jurafsky, Utility is in the eye of the user: A critique of nlp leaderboards, arXiv preprint arXiv:2009.13888 (2020).

[29] D. H. Wolpert, The lack of a priori distinctions between learning algorithms, Neural computation 8, 1341 (1996).

[30] C. G. Northcutt, A. Athalye, and J. Mueller, Pervasive label errors in test sets destabilize machine learning benchmarks, arXiv preprint arXiv:2103.14749 (2021).

[31] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, Do cifar-10 classifiers generalize to cifar-10?, arXiv preprint arXiv:1806.00451 (2018).

[32] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, Do imagenet classifiers generalize to imagenet?, in International conference on machine learning (PMLR, 2019) pp. $5389-5400$.

[33] B. Koch, E. Denton, A. Hanna, and J. G. Foster, Reduced, reused and recycled: The life of a dataset in machine learning research, arXiv preprint arXiv:2112.01716 (2021).

[34] A. Paullada, I. D. Raji, E. M. Bender, E. Denton, and A. Hanna, Data and its (dis) contents: A survey of dataset development and use in machine learning research, Patterns 2 (2021).

[35] R. Dotan and S. Milli, Value-laden disciplinary shifts in machine learning, arXiv preprint arXiv:1912.01172 (2019).

[36] S. Narang, H. W. Chung, Y. Tay, W. Fedus, T. Fevry, M. Matena, K. Malkan, N. Fiedel, N. Shazeer, Z. Lan, et al., Do transformer modifications transfer across implementations and applications?, arXiv preprint arXiv:2102.11972 (2021).

[37] S. L. Smith, A. Brock, L. Berrada, and S. De, Convnets match vision transformers at scale, arXiv preprint arXiv:2310.16764 (2023).

[38] M. Lucic, K. Kurach, M. Michalski, S. Gelly, and O. Bousquet, Are gans created equal? a large-scale study, Advances in neural information processing systems 31 (2018).

[39] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, Deep reinforcement learning that matters, in Proceedings of the AAAI conference on artificial intelligence, Vol. 32 (2018).

[40] C. Riquelme, G. Tucker, and J. Snoek, Deep bayesian bandits showdown: An empirical comparison of bayesian deep networks for thompson sampling, arXiv preprint arXiv:1802.09127 (2018).

[41] P. Flach, Performance evaluation in machine learning: the good, the bad, the ugly, and the way forward, in Proceedings of the AAAI conference on artificial intelligence, Vol. 33 (2019) pp. 9808-9814.

[42] J. Dodge, S. Gururangan, D. Card, R. Schwartz, and
N. A. Smith, Show your work: Improved reporting of experimental results, arXiv preprint arXiv:1909.03004 (2019).

[43] Y. LeCun, The mnist database of handwritten digits, http://yann.lecun.com/exdb/mnist/ (1998).

[44] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. FeiFei, Imagenet: A large-scale hierarchical image database, in 2009 IEEE conference on computer vision and pattern recognition (Ieee, 2009) pp. 248-255.

[45] A. Krizhevsky, G. Hinton, et al., Learning multiple layers of features from tiny images, (2009).

[46] G. Hinton, The forward-forward algorithm: Some preliminary investigations, arXiv preprint arXiv:2212.13345 (2022).

[47] J. Bausch, Recurrent quantum neural networks, Advances in neural information processing systems $\mathbf{3 3}, 1368$ (2020).

[48] S. Greydanus, Scaling down deep learning, arXiv preprint arXiv:2011.14439 (2020).

[49] J. Zhao, Y.-H. Zhang, C.-P. Shao, Y.-C. Wu, G.-C. Guo, and G.-P. Guo, Building quantum neural networks based on a swap test, Physical Review A 100, 012334 (2019).

[50] F. Tacchino, C. Macchiavello, D. Gerace, and D. Bajoni, An artificial neuron implemented on an actual quantum processor, npj Quantum Information 5, 26 (2019).

[51] A. Mari, T. R. Bromley, J. Izaac, M. Schuld, and N. Killoran, Transfer learning in hybrid classical-quantum neural networks, Quantum 4, 340 (2020).

[52] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, Supervised learning with quantum-enhanced feature spaces, Nature 567, 209 (2019).

[53] C. Zoufal, A. Lucchi, and S. Woerner, Variational quantum boltzmann machines, Quantum Machine Intelligence 3, 1 (2021).

[54] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, Power of data in quantum machine learning, Nature Communications 12, 2631 (2021).

[55] C. Wilson, J. Otterbach, N. Tezak, R. Smith, A. Polloreno, P. J. Karalekas, S. Heidel, M. S. Alam, G. Crooks, and M. da Silva, Quantum kitchen sinks: An algorithm for machine learning on near-term quantum computers, arXiv preprint arXiv:1806.08321 (2018).

[56] S. Lloyd, M. Schuld, A. Ijaz, J. Izaac, and N. Killoran, Quantum embeddings for machine learning, arXiv preprint arXiv:2001.03622 (2020).

[57] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, Quantum circuit learning, Physical Review A 98, 032309 (2018).

[58] M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, Evaluating analytic gradients on quantum hardware, Physical Review A 99, 032331 (2019).

[59] M. J. Bremner, R. Jozsa, and D. J. Shepherd, Classical simulation of commuting quantum computations implies collapse of the polynomial hierarchy, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 467, 459 (2011).

[60] T. Hastie, R. Tibshirani, J. H. Friedman, and J. H. Friedman, The elements of statistical learning: data mining, inference, and prediction, Vol. 2 (Springer, 2009).

[61] I. Steinwart and A. Christmann, Support vector machines (Springer Science \& Business Media, 2008).

[62] T. Hofmann, B. Schölkopf, and A. J. Smola, Kernel meth-
ods in machine learning (2008).

[63] M. Schuld, Supervised quantum machine learning models are kernel methods, arXiv preprint arXiv:2101.11020 (2021).

[64] K. Fukushima, Neocognitron: A hierarchical neural network capable of visual pattern recognition, Neural networks 1, 119 (1988).

[65] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, Gradient-based learning applied to document recognition, Proceedings of the IEEE 86, 2278 (1998).

[66] M. Weiler, P. Forré, E. Verlinde, and M. Welling, Equivariant and Coordinate Independent Convolutional Networks (2023).

[67] T. S. Cohen, M. Geiger, and M. Weiler, A general theory of equivariant cnns on homogeneous spaces, Advances in neural information processing systems 32 (2019).

[68] Flax software package.

[69] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang, JAX: composable transformations of Python+NumPy programs (2018).

[70] DeepMind, I. Babuschkin, K. Baumli, A. Bell, S. Bhupatiraju, J. Bruce, P. Buchlovsky, D. Budden, T. Cai, A. Clark, I. Danihelka, A. Dedieu, C. Fantacci, J. Godwin, C. Jones, R. Hemsley, T. Hennigan, M. Hessel, S. Hou, S. Kapturowski, T. Keck, I. Kemaev, M. King, M. Kunesch, L. Martens, H. Merzic, V. Mikulik, T. Norman, G. Papamakarios, J. Quan, R. Ring, F. Ruiz, A. Sanchez, L. Sartran, R. Schneider, E. Sezener, S. Spencer, S. Srinivasan, M. Stanojević, W. Stokowiec, L. Wang, G. Zhou, and F. Viola, The DeepMind JAX Ecosystem (2020).

[71] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, Scikit-learn: Machine learning in Python, Journal of Machine Learning Research 12, 2825 (2011).

[72] L. Schatzki, A. Arrasmith, P. J. Coles, and M. Cerezo, Entangled datasets for quantum machine learning, arXiv preprint arXiv:2109.03400 (2021).

[73] E. Perrier, A. Youssry, and C. Ferrie, Qdataset, quantum datasets for machine learning, Scientific data 9, 582 (2022).

[74] Y. Bengio, A. Courville, and P. Vincent, Representation learning: A review and new perspectives, IEEE transactions on pattern analysis and machine intelligence 35, 1798 (2013).

[75] H. Narayanan and S. Mitter, Sample complexity of testing the manifold hypothesis, Advances in neural information processing systems 23 (2010).

[76] P. Pope, C. Zhu, A. Abdelkader, M. Goldblum, and T. Goldstein, The intrinsic dimension of images and its impact on learning, arXiv preprint arXiv:2104.08894 (2021).

[77] F. Rosenblatt, The perceptron: a probabilistic model for information storage and organization in the brain., Psychological review 65, 386 (1958).

[78] M. Minsky and S. A. Papert, Perceptrons, reissue of the 1988 expanded edition with a new foreword by Léon Bottou: an introduction to computational geometry (MIT press, 2017).

[79] S. Goldt, M. Mézard, F. Krzakala, and L. Zdeborová,
Modeling the influence of data structure on learning in neural networks: The hidden manifold model, Physical Review X 10, 041044 (2020).

[80] S. Buchanan, D. Gilboa, and J. Wright, Deep networks and the multiple manifold problem, in International Conference on Learning Representations (2021).

[81] A. C. Lorena, A. I. Maciel, P. B. de Miranda, I. G. Costa, and R. B. Prudêncio, Data complexity meta-features for regression problems, Machine Learning 107, 209 (2018).

[82] S. Guan and M. Loew, A novel intrinsic measure of data separability, Applied Intelligence 52, 17734 (2022).

[83] M. R. Smith, T. Martinez, and C. Giraud-Carrier, An instance level analysis of data complexity, Machine learning 95, 225 (2014).

[84] J. M. Sotoca, J. S. Sánchez, and R. A. Mollineda, A review of data complexity measures and their applicability to pattern classification problems, Actas del III Taller Nacional de Mineria de Datos y Aprendizaje. TAMIDA , 77 (2005).

[85] T. K. Ho and M. Basu, Complexity measures of supervised classification problems, IEEE transactions on pattern analysis and machine intelligence 24, 289 (2002).

[86] J. Lorraine, P. Vicol, and D. Duvenaud, Optimizing millions of hyperparameters by implicit differentiation, in Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, Vol. 108, edited by S. Chiappa and R. Calandra (PMLR, 2020) pp. 1540-1552.

[87] S. Ahmed, N. Killoran, and J. F. C. Álvarez, Implicit differentiation of variational quantum algorithms (2022), arXiv:2211.13765 [quant-ph].

[88] F. J. Provost, T. Fawcett, R. Kohavi, et al., The case against accuracy estimation for comparing induction algorithms., in ICML, Vol. 98 (1998) pp. 445-453.

[89] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, Optuna: A next-generation hyperparameter optimization framework, in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2019).

[90] D. Golovin, B. Solnik, S. Moitra, G. Kochanski, J. Karro, and D. Sculley, Google vizier: A service for black-box optimization, in Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '17 (Association for Computing Machinery, New York, NY, USA, 2017) p. 1487-1495.

[91] M. Schuld, R. Sweke, and J. J. Meyer, Effect of data encoding on the expressive power of variational quantummachine-learning models, Physical Review A 103, 032430 (2021).

[92] N. Cristianini, J. Shawe-Taylor, A. Elisseeff, and J. Kandola, On kernel-target alignment, Advances in neural information processing systems 14 (2001).

[93] A. Abbas, R. King, H.-Y. Huang, W. J. Huggins, R. Movassagh, D. Gilboa, and J. R. McClean, On quantum backpropagation, information reuse, and cheating measurement collapse, arXiv preprint arXiv:2305.13362 (2023).

[94] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, Barren plateaus in quantum neural network training landscapes, Nature communications 9 , 4812 (2018).

[95] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, arXiv preprint arXiv:1412.6980 (2014).

## Appendix A: $\mathrm{CO}_{2}$ Emission Table

The following calculations are made using energy consumption data for the Cedar supercomputing cluster at the SciNet HPC Consortium, on which the vast majority of our simulations were run.

| Numerical simulations |  |
| :--- | :---: |
| Total CPU usage [core years] | $\approx 27$ |
| Cluster energy consumption (per core) $[\mathrm{W}]$ | $\approx 11$ |
| Total Energy Consumption Simulations $[\mathrm{kWh}]$ | $\approx 2600$ |
| Average Emissions Of $\mathrm{CO}_{2}$ In Canada $[\mathrm{kg} / \mathrm{kWh}]$ | $\approx 0.13$ |
| Total $\mathrm{CO}_{2}$ Emissions For numerical simulations $[\mathrm{kg}]$ | $\approx 340$ |
| Transport |  |
| Total $\mathrm{CO}_{2}$ Emission For Transport $[\mathrm{kg}]$ | 0 |
| Total $\mathrm{CO}_{2}$ Emission $[\mathrm{kg}]$ | $\approx 340$ |
| Were The Emissions Offset? | No |

## Appendix B: Glossary of quantum machine learning

 conceptsAmplitude embedding - An input data vector $\boldsymbol{x}$ is said to be amplitude embedded into a pure quantum state $|\psi(\boldsymbol{x})\rangle$ if the quantum state takes the form

$$
\begin{equation*}
|\psi(\boldsymbol{x})\rangle=\boldsymbol{x} \oplus \boldsymbol{c} / \mathcal{N} \tag{B1}
\end{equation*}
$$

where $\boldsymbol{c}$ is a vector with constant entries and $\mathcal{N}=\sqrt{\boldsymbol{x}^{\dagger} \cdot \boldsymbol{x}+\boldsymbol{c}^{\dagger} \cdot \boldsymbol{c}}$ is the state normalization.

Angle embedding-An input data vector $\boldsymbol{x}=\left(x_{j}\right)$ is said to be angle embedded into a pure quantum state $|\psi(\boldsymbol{x})\rangle$ if the quantum state takes the form

$$
\begin{equation*}
|\psi(\boldsymbol{x})\rangle=\prod_{j} \exp \left(-i G_{j} x_{j}\right)\left|\psi_{0}\right\rangle \tag{B2}
\end{equation*}
$$

where $\left|\psi_{0}\right\rangle$ is some initial quantum state and $G_{j}$ are Hermitian operators. If the operators $G_{j}$ act non-trivially on single qubits only, we call it a product angle embedding. This process of angle embedding is sometimes repeated a number of times, which is often called data reuploading.

Binary cross entropy loss-Given class probabilities $P\left( \pm 1 \mid \boldsymbol{\theta}, \boldsymbol{x}_{i}\right)$ and a label $y_{i}= \pm 1$, the cross entropy loss is

$$
\begin{equation*}
\ell\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}, \boldsymbol{y}_{\boldsymbol{i}}\right)=-\log P\left(y_{i} \mid \boldsymbol{\theta}, \boldsymbol{x}_{i}\right) \tag{B3}
\end{equation*}
$$

Minimising the cross entropy loss on a dataset is equivalent to maximising the log likelihood of the data, and is the preferred loss in binary classification tasks.

Gibbs state-A Gibbs state is a quantum density matrix that is diagonal in the energy eigenbasis given by a Hamiltonian $H$, and whose probability distribution over energy eigenstates forms a Gibbs distribution. Mathematically we have

$$
\begin{equation*}
\rho=\exp \left(-\frac{H}{k_{b} T}\right) / Z \tag{B4}
\end{equation*}
$$

where $T>0$ is a temperature, $k_{b}$ Boltzmann's constant and $Z=\operatorname{tr} \exp \left(-\frac{H}{k_{b} T}\right)$.

Instantaneous Quantum Polynomial circuit-A circuit that consists of input state preparation $|0\rangle$, quantum gates of the form $\exp \left(-i G_{X} \theta\right)$ where $G_{X}$ is a product of $\sigma_{X}$ operators on a subset of qubits, and measurement of a diagonal observable.

Linear loss-Given class probabilities $P\left( \pm 1 \mid \boldsymbol{\theta}, \boldsymbol{x}_{i}\right)$ and a label $y_{i}= \pm 1$, the linear loss is

$$
\begin{equation*}
\ell\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}, \boldsymbol{y}_{\boldsymbol{i}}\right)=-P\left(y_{i} \mid \boldsymbol{\theta}, \boldsymbol{x}_{i}\right) \tag{B5}
\end{equation*}
$$

Minimising the linear loss over a dataset is therefore equivalent to maximising the sum of the probabilities to classify each input correctly.

Maximum margin (linear) classifier-A linear classifier that separates the two classes such that the minimum distance of any training point to the hyperplane that defines the decision boundary is maxised.

Gaussian Kernel-A kernel of the form

$$
\begin{equation*}
k\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\exp \left(-\gamma\left\|\boldsymbol{x}-\boldsymbol{x}^{\prime}\right\|\right) \tag{B6}
\end{equation*}
$$

where $\gamma$ is a free hyperparameter.

Square loss-Given a model whose output is $f(\boldsymbol{\theta}, \boldsymbol{x})$, the square loss is

$$
\begin{equation*}
\ell\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}, \boldsymbol{y}_{\boldsymbol{i}}\right)=\left(f\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right)-y_{i}\right)^{2} \tag{B7}
\end{equation*}
$$

Vanishing gradients-A phenomenon that commonly occurs for deep or highly expressive variational quantum circuits whereby the expected magnitude of a typical gradient component decreases exponentially with the number of qubits when initialising all parameters uniformly at random $[94]$.

## Appendix C: Detailed description of the models

In this appendix we describe each model used in the study in further detail. Unless otherwise specified all variational models are trained via gradient-descent using the Adam optimizer [95] implemented in Optax [70] with default parameters except for the learning rate which we vary during hyperparameter search.

## 1. CircuitCentricClassifier [13]

An input vector $\mathbf{x}$ of dimension $d$ gets amplitude embedded into a suitable number of qubits (including padding and normalisation, see Glossary above). Note that this preprocessing strategy induces a qualitatively different behaviour when $d$ is a power of 2 , since without padding, normalisation looses all information about the length of the inputs.

A hyperparameter allows for the pre-processed input to be encoded into multiple registers, which creates copies of amplitude encoding states. This effectively creates tensor products of the pre-processed inputs. A variational circuit that uses arbitrary single qubit rotations followed by cascades of controlled arbitrary single qubit rotations is followed by a $\mathrm{Z}$ measurement of the first qubit.

The following is an example of the quantum circuit used in the CircuitCentricClassifier for two copies of an input $\mathbf{x} \in \mathbb{R}^{4}$ embedded into state $\left|\psi_{\mathbf{x}}\right\rangle$ and PennyLane's StronglyEntanglingLayers template implementing the variational ansatz:

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-26.jpg?height=355&width=377&top_left_y=1096&top_left_x=408)

The expected value of the measurement is added to a trainable classical bias. The parameters of the variational circuit, as well as the bias, are optimised using the square loss.

| hyperparameter | values |
| :--- | :--- |
| learning_rate | $[0.001,0.01,0.1]$ |
| n_layers (variational layers) | $[1,5,10]$ |
| n_input_copies | $[1,2,3]$ |

## 2. DataReuploadingClassifier [18]

This model uses successive, trainable angle embeddings of data. Each qubit embedding gate takes a vector $\boldsymbol{x}$ of three features, two trainable three-dimensional real vectors $\boldsymbol{w}$ and $\boldsymbol{\theta}$, and encodes them as

$$
\begin{equation*}
U(\boldsymbol{x} \circ \boldsymbol{\omega}+\boldsymbol{\theta}) \tag{C1}
\end{equation*}
$$

where

$$
\begin{equation*}
U(\phi)=e^{i Z \phi_{1} / 2} e^{i Y \phi_{2} / 2} e^{i Z \phi_{3} / 2} \tag{C2}
\end{equation*}
$$

parameterizes a general $\mathrm{SU}(2)$ rotation on a single qubit, and $\circ$ denotes element-wise multiplication of vectors.
To encode data input $\boldsymbol{x} \in \mathbb{R}^{d}$, we therefore split $\boldsymbol{x}$ into $\left\lceil\frac{d}{3}\right\rceil$ vectors of size 3 , and feed each vector into a distinct qubit embedding gate (padding input vectors with zero if necessary). The number of qubits is therefore set by the dimension of the input data features. This is followed by a sequence of CZ gates in a ladder structure (See Figure 5 of [18]), and the process is repeated for a number of layers, which increases the expressivity of the model.

The following is an example of the quantum circuit used in the DataReuploadingClassifier for the input $\mathbf{x}=(0.1,0.2,0.3,0.4)^{T}$ embedded in 3 trainable layers, and with the scaling parameters $\boldsymbol{\omega}$ all set to 1 .

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-26.jpg?height=361&width=742&top_left_y=714&top_left_x=1147)

Training is based on the fidelities of the output qubits to one of two class states: here either $|0\rangle$ or $|1\rangle$. Defining $F_{j}^{0}(\boldsymbol{x}), F_{j}^{1}(\boldsymbol{x})$ as the fidelity of the $j^{\text {th }}$ output qubit to the state $|0\rangle,|1\rangle$, the loss $\ell$ for a single data point is given by

$$
\begin{equation*}
\ell\left(\boldsymbol{\theta}, \boldsymbol{\omega}, \boldsymbol{\alpha}, \boldsymbol{x}_{i}\right)=\sum_{j=1}^{n_{\max }}\left(\alpha_{j}^{0} F_{j}^{0}-\left(1-y_{i}\right)\right)^{2}+\left(\alpha_{j}^{1} F_{j}^{1}-y_{i}\right)^{2} \tag{C3}
\end{equation*}
$$

where the $\alpha_{j}^{0}, \alpha_{j}^{1}$ are trainable parameters, and $n_{\max }$ determines the number of qubits to use for training and prediction.

For prediction, we use the average fidelity to either $|0\rangle$ or $|1\rangle$ over the same qubits:

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{argmax}\left(\left\langle F_{j}^{0}\right\rangle,\left\langle F_{j}^{1}\right\rangle\right) \tag{C4}
\end{equation*}
$$

where $\left\langle F_{j}^{0}\right\rangle=\frac{1}{n_{\max }} \sum_{j=1}^{n_{\max }} F_{j}^{0}$. This is not specified in [18], but is a natural generalisation of the $n_{\max }=1$ case they focus on. The choice of the hyperparameter observable_type determines the number $n_{\max }$ of qubits used to evaluate the weighted cost function.

| hyperparameter | values |
| :--- | :--- |
| learning_rate | $[0.001,0.01,0.1]$ |
| n_layers (reuploading layers) | $[1,5,10,15]$ |
| observable_type | [single, half, full] |

## 3. DressedQuantumCircuitClassifier [51]

This model maps an input data point $\boldsymbol{x}$ two a 2 dimensional vector via

$$
f\left(\boldsymbol{\theta}, \boldsymbol{W}_{\text {in }}, \boldsymbol{W}_{\text {out }}, \boldsymbol{x}\right)=f_{\text {out }}\left(\boldsymbol{W}_{\text {out }}, f_{Q}\left(\boldsymbol{\theta}, f_{\text {in }}\left(\boldsymbol{W}_{\text {in }}, \boldsymbol{x}\right)\right)\right)
$$

The functions $f_{\text {in }}\left(\boldsymbol{W}_{\text {in }}, \cdot\right), f_{\text {out }}\left(\boldsymbol{W}_{\text {out }}, \cdot\right)$ are single layer fully connected feed-forward neural networks with weights $\boldsymbol{W}_{\text {in }} \in \mathbb{R}^{d \times n}, \boldsymbol{W}_{\text {out }} \in \mathbb{R}^{n \times 2}$, where $f_{\text {in }}$ has a tanh activation scaled by $\pi / 2$, and $f_{\text {out }}$ has no activation. The function $f_{Q}$ corresponds to a parameterised quantum circuit where input features are angle-encoded into individual qubits, followed by layers of single-qubit $Y$ rotations and CNOT gates applied in a ring pattern. The output of the circuit is an $n$-dimensional vector whose elements are the expectation values of single-qubit $Z$ measurements on each qubit.

The following is an example of the quantum circuit used in the DressedQuantumCircuitClassifier for the input $\mathbf{x}=(0.1,0.2,0.3,0.4)^{T}$ and 3 variational layers:

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-27.jpg?height=523&width=832&top_left_y=823&top_left_x=191)

The output vector is sent through a softmax layer that is used for prediction and gives the class probabilities, and the cross entropy loss is used to train $\boldsymbol{W}_{\text {in }}, \boldsymbol{W}_{\text {out }}$ and $\boldsymbol{\theta}$ simultaneously.

```
hyperparameter values
learning_rate $\quad[0.001,0.01,0.1]$
n_layers $\quad[1,5,10,15]$
```


## 4. IQPVariationalClassifier [52]

This model uses angle encoding $V(\boldsymbol{x})$ inspired from IQP circuits, which is implemented by PennyLane's IQPEmbedding class. This is followed by a trainable parameterised circuit $U(\boldsymbol{\theta})$, implemented by PennyLane's StronglyEntanglingLayers class.

Prediction is given by measurement of $Z_{1} Z_{2}$ on the first two qubits:

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{sign} f(\boldsymbol{\theta}, \boldsymbol{x}) \tag{C5}
\end{equation*}
$$

where

$$
\begin{equation*}
f(\boldsymbol{\theta}, \boldsymbol{x})=\left\langle 0\left|V^{\dagger}(\boldsymbol{x}) U^{\dagger}(\boldsymbol{\theta}) Z_{1} Z_{2} U(\boldsymbol{\theta}) V(\boldsymbol{x})\right| 0\right\rangle \tag{C6}
\end{equation*}
$$

The loss is equal to the linear loss:

$$
\begin{equation*}
\ell(\boldsymbol{\theta}, \boldsymbol{x})=(1-y \cdot f(\boldsymbol{\theta}, \boldsymbol{x})) / 2 \tag{C7}
\end{equation*}
$$

The following is an example of the quantum circuit used in the IQPVariationalClassifier for the input $\mathbf{x}=(0.1,0.2,0.3,0.4)^{T}:$

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-27.jpg?height=366&width=393&top_left_y=454&top_left_x=1319)

| hyperparameter | values |
| :--- | :--- |
| learning_rate | $[0.001,0.01,0.1]$ |
| n_layers (variational layers) | $[1,5,10,15]$ |
| repeats(embedding layers) | $[1,5,10]$ |

## 5. QuantumBoltzmannMachine [53]

This model encodes data into a Gibbs state of a $n$ qubit Hamiltonian. We use a Hamiltonian that is a natural generalization of the one studied in the two-qubit example in [53]:

$$
\begin{equation*}
H(\boldsymbol{\theta}, \boldsymbol{x})=\sum_{j} Z_{j} \boldsymbol{\theta}_{j}^{Z} \cdot \boldsymbol{x}+\sum_{j} X_{j} \boldsymbol{\theta}_{i}^{X} \cdot \boldsymbol{x}+\sum_{j>k} Z_{j} Z_{k} \boldsymbol{\theta}_{j k} \cdot \boldsymbol{x} \tag{C8}
\end{equation*}
$$

where $\boldsymbol{\theta}_{j}^{Z}, \boldsymbol{\theta}_{j}^{X}, \boldsymbol{\theta}_{j k}$ are vectors of trainable parameters that we collect into $\boldsymbol{\theta}$. We take $n=d$ so that the number of qubits scales with the number of features.

Since Gibbs state preparation is hard, the paper gives a recipe to parameterize a trial state for the Gibbs state and perform variational imaginary time evolution to approximate the desired state. Since this is quite computationally involved, we assume (as they do in [53]) that we have access to the perfect Gibbs state. It is therefore unclear whether the full algorithm can be expected to perform as well as our implementation.

For prediction, a diagonal $\pm 1$ valued observable $O$ is measured on a subset of qubits of size $n_{\text {vis }}$ (called the visible qubits, controlled by hyperparameter visible_qubits). The sign of the expectation value determines the label:

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{sign}(\operatorname{tr}[\rho(\boldsymbol{\theta}, \boldsymbol{x}) O]) \tag{C9}
\end{equation*}
$$

where

$$
\begin{equation*}
\rho(\boldsymbol{\theta})=\frac{\exp \left(-\frac{H(\boldsymbol{\theta}, \boldsymbol{x})}{T}\right)}{\operatorname{tr}\left[\exp \left(-\frac{H(\boldsymbol{\theta}, \boldsymbol{x})}{T}\right)\right]} \tag{C10}
\end{equation*}
$$

is a Gibbs state, and we choose $O$ to be

$$
\begin{equation*}
O=\frac{1}{n_{\text {vis }}} \sum_{j=1}^{n_{\text {vis }}} Z_{j} \tag{C11}
\end{equation*}
$$

(note [53] does not recommend a general form for the observable).

Training is done with a binary cross entropy loss. To define the probability of $y=1$, we use:

$$
\begin{equation*}
P(y=1 \mid \boldsymbol{\theta}, \boldsymbol{x})=\frac{1+\langle O\rangle}{2} \tag{C12}
\end{equation*}
$$

which is in $[0,1]$ and agrees with the example in [53] for $n_{\text {vis }}=1$.

We note that since we are forced to work with mixed states, this implies a larger memory cost of simulation. As a result we were not able to test this model for as large qubit number as others. It is also the only model we implemented without the use of a PennyLane circuit, but rather by constructing the density matrix directly.

| hyperparameter values |
| :--- | :--- |
| learning_rate $\quad[0.001,0.01,0.1]$ |
| temperature $(\mathrm{T})[1,10,100]$ |
| visible_qubits $\quad[$ single, half, all $]$ |

## 6. QuantumBoltzmannMachineSeparable

This model is a version of QuantumBoltzmannMachine that does not use entanglement. Once again we take $n=$ $d$ so that the number of qubits scales with the number of features. The Hamiltonian is

$$
\begin{equation*}
H(\boldsymbol{\theta}, \boldsymbol{x})=\sum_{j} Z_{j} \boldsymbol{\theta}_{j}^{Z} \cdot \boldsymbol{x}+\sum_{j} X_{j} \boldsymbol{\theta}_{i}^{X} \cdot \boldsymbol{x} \tag{C13}
\end{equation*}
$$

whose corresponding Gibbs state is a product mixed state by virtue of the Hamiltonian being product. The model is equivalent to QuantumBoltzmannMachine otherwise and uses the same hyperparameter grid.

## 7. QuantumMetricLearner [56]

The QuantumMetricLearner works quite differently from other quantum neural networks. It uses a trainable, layer-wise embedding inspired by the QAOA algorithm implemented by PennyLane's QAOAEmbedding template, which employs one more qubit than there are features. The ansatz for one layer encodes input features into $\mathrm{X}$ rotations, followed by parametrised $\mathrm{ZZ}$ and $\mathrm{Y}$ rotations. The additional qubit with a constant angle is used as a "latent feature".

Training of the embedding is performed by measuring the overlap between a pair of embedded data points from the same [different] classes and minimising [maximising] their fidelity.

More precisely, if $\left|\phi_{\theta}(\mathbf{x})\right\rangle$ is the quantum state embedding an input vector $\mathbf{x}$, and $A A, B B[A B]$ are sets of randomly sampled training input pairs $\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$ from training data in the same class $A$ or $B$ [different classes], the cost function is defined as

$$
\begin{align*}
c(A, B)= & 1-\sum_{(\mathbf{a}, \mathbf{b}) \in A B}\left|\left\langle\phi_{\boldsymbol{\theta}}(\mathbf{a}) \mid \phi_{\boldsymbol{\theta}}(\mathbf{b})\right\rangle\right|^{2} \\
& +0.5 \sum_{\left(\mathbf{a}, \mathbf{a}^{\prime}\right) \in A A}\left|\left\langle\phi_{\boldsymbol{\theta}}(\mathbf{a}) \mid \phi_{\boldsymbol{\theta}}\left(\mathbf{a}^{\prime}\right)\right\rangle\right|^{2} \\
& +0.5 \sum_{\left(\mathbf{b}, \mathbf{b}^{\prime}\right) \in B B}\left|\left\langle\phi_{\boldsymbol{\theta}}(\mathbf{b}) \mid \phi_{\boldsymbol{\theta}}\left(\mathbf{b}^{\prime}\right)\right\rangle\right|^{2} \tag{C14}
\end{align*}
$$

Note that in the original paper, all possible pairs of datapoints within a random batch are compared, however to have a better control of how many circuits are run we sample a batch of random pairs instead.

Once trained, the embedding is directly used for prediction: A new input is embedded and the resulting state compared to a random batch of embedded training data points $A^{\prime}$ and $B^{\prime}$ from each class. The class that it is closest to on average is assigned. This rule corresponds to the "fidelity classifier" from the paper:

$$
\begin{equation*}
f(\boldsymbol{\theta}, \mathbf{x})=\sum_{\mathbf{a} \in A^{\prime}}\left|\left\langle\phi_{\boldsymbol{\theta}}(\mathbf{a}) \mid \phi_{\boldsymbol{\theta}}(\mathbf{x})\right\rangle\right|^{2}-\sum_{\mathbf{b} \in B^{\prime}}\left|\left\langle\phi_{\boldsymbol{\theta}}(\mathbf{b}) \mid \phi_{\boldsymbol{\theta}}(\mathbf{x})\right\rangle\right|^{2} \tag{C15}
\end{equation*}
$$

The final prediction is made by taking the sign,

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{sign} f(\boldsymbol{\theta}, \mathbf{x}) \tag{C16}
\end{equation*}
$$

The following is the quantum circuit used to evaluate overlaps in the QuantumMetricLearner for two inputs $\mathbf{x}, \mathbf{x}^{\prime} \in \mathbb{R}^{4}$ :

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-28.jpg?height=380&width=317&top_left_y=1729&top_left_x=1359)

The number of circuits run for each training step and prediction scales linearly with the size of the batch of example pairs used - in this paper we fixed 32 for both. Larger batch sizes allow a more reliable estimate of the cost and predicted label.

| hyperparameter | values |
| :--- | :--- |
| learning_rate | $[0.001,0.01,0.1]$ |
| n_layers (embedding layers) | $[1,3,4]$ |

## 8. TreeTensorClassifier [19]

This model was designed to avoid the phenomenon of barren plateaus. We implement the 'tree tensor' structure shown in Figure 1 of [19]. The variational circuit in this model has a tree-like structure and therefore requires a number of qubits that is a power of 2. In [19] one first optimizes a variational circuit that finds a state that approximates an amplitude-encoded data state. The reason for this is to make the algorithm more efficient; here, to avoid an additional variational optimisation, we assume direct access to the exact amplitude encoded state $V(\boldsymbol{x})|0\rangle$ (padding with constant values $1 / 2^{n}$ with $n$ the number of qubits when necessary).

This state is then fed into the variational circuit $U(\boldsymbol{\theta})$ consisting of trainable single-qubit $Y$ rotations and CNOTs. The tree structure means that there are few parameters; for a circuit with $n$ qubits one has only $2 n-1$ parameters.

Prediction is given by measurement of $Z$ on the first qubit,

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{sign} f(\boldsymbol{\theta}, \boldsymbol{x}) \tag{C17}
\end{equation*}
$$

where

$$
\begin{equation*}
f(\boldsymbol{\theta}, \boldsymbol{x})=\left\langle 0\left|V^{\dagger}(\boldsymbol{x}) U^{\dagger}(\boldsymbol{\theta}) Z_{1} U(\boldsymbol{\theta}) V(\boldsymbol{x})\right| 0\right\rangle \tag{C18}
\end{equation*}
$$

and training is via the square loss:

$$
\begin{equation*}
\ell(\boldsymbol{\theta}, \boldsymbol{x}, y)=(f(\boldsymbol{\theta}, \boldsymbol{x})-y)^{2} \tag{C19}
\end{equation*}
$$

The following is an example of the quantum circuit used in the TreeTensorClassifier for a 4-dimensional input encoded into state $\left|\psi_{\mathbf{x}}\right\rangle$ :

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-29.jpg?height=162&width=483&top_left_y=1651&top_left_x=363)

This model has few suggested hyperparameters, so we vary only the learning rate.

hyperparameter values

learning_rate $\quad[0.001,0.01,0.1]$

## 9. IQPKernelClassifier [52]

This model is a kernel equivalent of the IQP variational model. The embedding $V(\boldsymbol{x})|0\rangle$ is the same IQPinspired embedding given by PennyLane's IQPEmbedding template. This defines a kernel

$$
\begin{equation*}
k\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\operatorname{tr}\left[\rho(\boldsymbol{x}) \rho\left(\boldsymbol{x}^{\prime}\right)\right]=\left|\left\langle 0\left|V^{\dagger}(\boldsymbol{x}) V\left(\boldsymbol{x}^{\prime}\right)\right| 0\right\rangle\right|^{2} \tag{C20}
\end{equation*}
$$

which we evaluate by applying the unitary $V^{\dagger}(\boldsymbol{x}) V\left(\boldsymbol{x}^{\prime}\right)$ to an input state $|0\rangle$ and calculating the probability to measure $|0\rangle$.

The following is an example of the quantum circuit used in the IQPKernelClassifier for two inputs $\mathbf{x}, \mathbf{x} \in$ $\mathbb{R}^{4}$ :

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-29.jpg?height=372&width=393&top_left_y=489&top_left_x=1321)

The kernel matrix $K$ is fed to scikit-learn's SVC class, which trains a support vector machine classifier. Prediction is given by

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{sign} \sum_{i} \alpha_{i} k\left(\boldsymbol{x}_{i}, \boldsymbol{x}\right) \tag{C21}
\end{equation*}
$$

with $\alpha_{i}$ the weights of the support vector machine.

The hyperparameter search values are as follows:

| hyperparameter | values |
| :--- | :--- |
| repeats (embedding layers) | $[1,5,10]$ |
| $C$ (SVC regularization) | $[0.1,1,10,100]$ |

## 10. ProjectedQuantumKernel [54]

This model is a kernel method that uses a Hamiltonianinspired data embedding that resembles a Trotter evolution of a 1D-Heisenberg model with random couplings. This consists of applying random single-qubit rotations to an input state $|0\rangle$ of $n=d+1$ qubits, followed by $L$ layers of two-qubit rotations with generators $X X, Y Y$, $Z Z$ to pairs of adjacent qubits, with angles given by the elements of $\boldsymbol{x}$ :

$$
\begin{equation*}
\prod_{j=1}^{n} \exp \left(-\mathrm{i} \frac{t}{L} x_{j}\left(X_{j} X_{j+1}+Y_{j} Y_{j+1}+Z_{j} Z_{j+1}\right)\right) \tag{C22}
\end{equation*}
$$

where $t$ is a hyperparameter of the model. Writing the embedded states as $\rho\left(\boldsymbol{x}_{i}\right)$, the kernel function $k\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)$ is

$$
\begin{equation*}
\exp \left(-\gamma \sum_{k=1}^{n} \sum_{P \in\{X, Y, Z\}}\left(\operatorname{tr}\left[P \rho_{k}\left(\boldsymbol{x}_{i}\right)\right]-\operatorname{tr}\left[P \rho_{k}\left(\boldsymbol{x}_{j}\right)\right]\right)^{2}\right) \tag{C23}
\end{equation*}
$$

where $\rho_{k}$ is the reduced state of $k^{\text {th }}$ qubit of $\rho$. This is simply the RBF kernel with bandwidth $\gamma$ applied to feature mapped vectors $\boldsymbol{\phi}(\boldsymbol{x})$ with elements $\left(\operatorname{tr}\left[P \rho_{k}(\boldsymbol{x})\right]\right)$.

The following is an example of the quantum circuit used to compute feature vectors from measurements in ProjectedQuantumKernel for the input $\mathbf{x}=$ $(0.1,0.2,0.3,0.4)^{T}$ :
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-30.jpg?height=778&width=814&top_left_y=558&top_left_x=206)

According to [54], the default $\gamma$ value is set as

$$
\begin{equation*}
\gamma_{0}=\frac{1}{\operatorname{Var}(\boldsymbol{\phi}) d} \tag{C24}
\end{equation*}
$$

where $\operatorname{Var}(\boldsymbol{\phi})$ is the variance of the elements of all the vectors $\boldsymbol{\phi}\left(\boldsymbol{x}_{i}\right)$. We include another hyperparameter 'gamma_factor' that scales this default value.

A support vector machine is then trained using scikitlearn's SVC class, and prediction is given by

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{sign} \sum_{i} \alpha_{i} k\left(\boldsymbol{x}_{i}, \boldsymbol{x}\right) \tag{C25}
\end{equation*}
$$

| hyperparameter | values |
| :--- | :--- |
| trotter_steps(embedding layers) | $L[1,3,5]$ |
| $C$ (SVC regularization) | $[0.1,1,10,100]$ |
| $t$ (time) | $[0.01,0.1,1.0]$ |
| gamma_factor | $[0.1,1,10]$ |

## 11. QuantumKitchenSinks [55]

This model uses a quantum circuit to define a feature map given by the concatenation of its output bit-strings. These features are then used to train a linear classifier. The feature map procedure works as follows:

- Linearly transform the input feature vector $\boldsymbol{x}$ as $\boldsymbol{x}_{k}^{\prime}=W_{k} \boldsymbol{x}+\boldsymbol{b}_{k}$ using randomly sampled $W_{k}, \boldsymbol{b}_{k}$ for $k=1, \cdots, k_{\max }$. Here $W_{k}, \boldsymbol{b}_{k}$ are such that $\boldsymbol{x}^{\prime}$ has dimension $n$ which may be different from $d$.
- Feed each $\boldsymbol{x}_{k}^{\prime}$ into a circuit (described below) that returns a single measurement sample $\boldsymbol{z}_{k} \in\{0,1\}^{n}$. The concatenated vector $\boldsymbol{z}_{1} \oplus \cdots \oplus \boldsymbol{z}_{k_{\max }}$ is the feature mapped vector of size $n \cdot k_{\max }$.

The circuit used in the second step above consists of angle encoding the feature mapped vectors with $X$ rotations on individual qubits, and applying two layers of CNOT gates: the first between adjacent qubits $(j, j+1)$; the second between qubits $(j, j+2)$ a distance 2 apart. This choice is a natural generalisation of the example present in [55], which does not present a specific circuit structure for circuits beyond 4 qubits.

The following is an example of the quantum circuit used to compute feature vectors QuantumKitchenSinks for the input $\mathbf{x}=(0.1,0.2,0.3,0.4)^{T}$ (here without first applying the classical linear transformation):

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-30.jpg?height=306&width=480&top_left_y=1105&top_left_x=1275)

The feature mapped vectors are then passed to a linear classifier, which we implement using scikit-learn's LogisticRegression class (note logistic regression can be used for linear classification via the cross entropy loss).

For prediction, a new feature-mapped vector $\boldsymbol{z}$ is created via the same process, and

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{sign} \boldsymbol{w} \cdot \boldsymbol{z} \tag{C26}
\end{equation*}
$$

where $\boldsymbol{w}$ is the linear classifier found during training.

The hyperparameter search values are as follows:

| hyperparameter | values |
| :--- | :--- |
| $n_{-}$qfeatures (circuit size) | $[d,\lfloor d / 2\rfloor]$ |
| $n_{-}$episodes (number of circuits $\left.k_{\max }\right)$ | $[10,100,500,2000]$ |

## 12. QuanvolutionalNeuralNetwork [15]

This model consists of a fixed quantum feature map followed by a trainable convolutional neural network. The data is first scaled to lie in the range $[-1,1]$ and a binary threshold function (with threshold zero) is applied and the data scaled by $\pi$ so that it takes values in $\{0, \pi\}$. One then applies a convolutional layer to the data, where
the convolutional filters are given by a $n_{q}^{2}$ qubit quantum circuits that take as input $n_{q} \times n_{q}$ sized windows of the input data, in an analogous manner to a classical convolutional filter. The quantum circuits consists of random gates that we implement via PennyLane's RandomLayers class, and the output of the circuits is the number of ones in the bitstring that has the highest probability to be sampled from the circuit. As with convolutional neural networks, we allow for more than one channel in this layer, controlled by the hyperparameter n_qchannels.

The following is an example of the quantum circuit used in the QuanvolutionalNeuralNetwork for a $2 \times 2$ dimensional input window, where the thresholded and rescaled pixel values form a pre-processed input vector $(0,0, \pi, \pi)^{T}$ :

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-31.jpg?height=371&width=380&top_left_y=861&top_left_x=409)

The resulting data is then fed into a convolutional neural network with the same specifications as for the ConvolutionalNeuralNetwork.

| hyperparameter | values |
| :--- | :--- |
| learning_rate | $[0.0001,0.001,0.01]$ |
| n_qchannels | $[1,5,10]$ |
| qkernel_shape $\left(n_{q}\right)$ | $[2,3]$ |
| kernel_shape (CNN filter size) | $[2,3,5]$ |

## 13. WeiNet $[20]$

This model (that we call WeiNet following the first author of the paper) implements a convolutional layer as a unitary operation that acts on input data that is amplitude encoded into a quantum state. The model has two registers: the ancilliary register and the work register. The ancilliary register is used to parameterise a 4 qubit state which in turn controls a number of unitaries that act on the work register, where the data is encoded via amplitude encoding. Note that in figure 2 of [20], the Hadamard gates on the ancilla register have no effect since we trace this register out. The effect of this register is then to simply perform a classical mixture of the unitaries $Q_{i}$ defined therein on the work register. For simplicity (and to save qubit numbers), we parameterise this distribution via 16 real trainable parameters.

Two of the qubits are then traced out, which is equivalent to a type of pooling. All single and double correlators $\langle Z\rangle$ and $\langle Z Z\rangle$ are measured, and a linear model on these values is used for classification.

The following is an example of one of the quantum circuits used in the WeiNet model for a $4 \times 4$-dimensional input window encoded into state $\left|\psi_{\mathbf{x}}\right\rangle$ acted on by "filter unitary" $U$ :

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-31.jpg?height=363&width=393&top_left_y=515&top_left_x=1321)

The paper does not specify the training loss; like the other convolutional models we use the binary cross entropy with a sigmoid (logistic) activation. The corresponding probabilities from the sigmoid activation are used for prediction.

| hyperparameter values |  |
| :--- | :--- |
| learning_rate | $[0.0001,0.001,0.01]$ |
| filter_type | [edge_detect, smooth, sharpen $]$ |

## 14. MLPClassifier

This is a multiplayer perception model implemented via scikit-learn's MLPClassifier class. An input feature vector $\boldsymbol{x}$ is transformed using a sequence of linear transformations $W_{l}$ and element wise-activation functions $a$ as

$$
\begin{equation*}
f(\boldsymbol{x})=a\left(W_{L} \cdots a\left(W_{2}\left(a\left(W_{1} \boldsymbol{x}\right)\right)\right) \cdots\right) \tag{C27}
\end{equation*}
$$

'We use the rectified linear unit activation $a(x)=\max (0, x)$. The trainable parameters of the model are the weights of the matrices $W_{k}$. We vary the number of layers $L$ and the shapes of the weight matrices $W_{k}$ via the model parameter hidden_layer_size, which we set to be one of $[(100),,(10,10,10,10),(50,10,5)]$. Here, each element of the tuple corresponds to a different layer and the values give the output dimensions of the corresponding matrices. The last layer is not included here since it is always of dimension 1. Training is done via gradient descent with the binary cross entropy loss using the adam update and the default class fit method. We vary the initial learning rate and regularisation strength alpha. All other parameters are set to the class defaults, except the maximum number of iterations, max_iter, that we set to 3000 .

| hyperparameter | values |
| :--- | :--- |
| learning_rate | $[0.001,0.01,0.1]$ |
| hidden_layer_sizes | $[(100),,(10,10,10,10),(50,10,5)]$ |
| alpha (regularisation) | $[0.01,0.001,0.0001]$ |

15. SVC

This model is a support vector machine classifier implemented via scikit-learn's SVC class. We use the radial basis function kernel:

$$
\begin{equation*}
k\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\exp \left(-g a m m a\left\|x-x^{\prime}\right\|^{2}\right) \tag{C28}
\end{equation*}
$$

During hyperparameter search, we vary the bandwidth parameter gamma and the regularisation strength $C$.

| hyperparameter | values |
| :--- | :--- |
| $C($ SVC regularization $)$ | $[0.1,1,10,100]$ |
| gamma | $[0.001,0.01,0.1,1]$ |

## 16. ConvolutionalNeuralNetwork

This model is a vanilla implementation of a convolutional neural network (CNN), written in flax. The structure of the network is as follows

- a 2D convolutional layer with 32 output channels
- a max pool layer
- a 2D convolutional layer with 64 output channels
- a max pool layer
- a two layer fully connected feedforward neural network with 128 hidden neurons and one output neuron

The probability of class 1 is given by

$$
\begin{equation*}
P(+1 \mid \boldsymbol{w}, \boldsymbol{x})=\sigma(f(\boldsymbol{w}), \boldsymbol{x}) \tag{C29}
\end{equation*}
$$

where $\boldsymbol{w}$ are the weights of the model, $f(\boldsymbol{w})$ is the value of the final neuron, and $\sigma$ is the logistic function. These probabilities are fed to binary cross entropy loss for training.

| hyperparameter values |  |
| :--- | :--- |
| learning_rate | $[0.0001,0.001,0.01]$ |
| kernel_shape | $[2,3,5]$ |

## 17. SeparableVariationalClassifier

This is a simple quantum neural network model that does not use entanglement. The data encoding $V$ ( consists of $L$ layers, where in each layer arbitrary trainable single-qubit rotations are performed followed by a product angle embedding of the data via Pauli Y rotation gates. The encoding is proceeded by another layer of trainable single-qubit rotations, and prediction is given by measurement of $O=\frac{1}{n}\left(Z_{1}+\cdots+Z_{n}\right)$, i.e.

$$
\begin{equation*}
y_{\text {pred }}=\operatorname{sign} f(\boldsymbol{\theta}, \boldsymbol{x}) \tag{C30}
\end{equation*}
$$

where

$$
\begin{equation*}
f(\boldsymbol{\theta}, \boldsymbol{x})=\left\langle\frac{1}{n}\left(Z_{1}+\cdots+Z_{n}\right)\right\rangle \tag{C31}
\end{equation*}
$$

where $\boldsymbol{\theta}$ represents all trainable parameters in the $L$ layers.

The following is an example of the single-qubit quantum circuit used in the SeparableVariationalClassifier model to process the first value of the input vector $\mathbf{x}=(0.1,0.2,0.3,0.4)^{T}$ (whereas the remaining features are processed by similar circuits):

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-32.jpg?height=95&width=463&top_left_y=939&top_left_x=1281)

To train the model, we define class probabilities via the logistic function

$$
\begin{equation*}
P(+1 \mid \boldsymbol{\theta}, \boldsymbol{x})=\sigma(6\langle O\rangle) \tag{C32}
\end{equation*}
$$

where we multiply the observable value by 6 since the sigmoid function varies significantly over the range $[-6,6]$. These probabilities are then used in a binary cross entropy loss function.

| hyperparameter | values |
| :--- | :--- |
| learning_rate | $[0.001,0.01,0.1]$ |
| encoding_layers $(L)$ | $[1,3,5,10]$ |

## 18. SeparableKernelClassifier

This model is the kernel equivalent of the above. The data encoding consists $L$ layers, where in each layer an $X$ rotation with angle $\pi / 4$ is applied to each qubit followed by a product of $Y$ rotations that encode each element of $\boldsymbol{x}$ into individual qubits (so $n=d$ ).

The kernel is given by (6):

$$
\begin{equation*}
k\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\operatorname{tr}\left[\rho(\boldsymbol{x}) \rho\left(\boldsymbol{x}^{\prime}\right)\right] \tag{C33}
\end{equation*}
$$

and the model is trained using scikit-learn's SVC class.

The following is an example of the single-qubit quantum circuit used in the SeparableKernelClassifier model to process the first value of the inputs $\mathbf{x}, \mathbf{x}^{\prime}=$ $(0.1,0.2,0.3,0.4)^{T}$, using 2 layers:

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-32.jpg?height=79&width=802&top_left_y=2389&top_left_x=1125)
hyperparameter values

encoding layers $(L)[1,3,5,10]$

SVM regularization $[0.1,1,10,100]$

## Appendix D: Convergence criteria of variational models

Here we describe the criterion used to decide convergence during training of variational models. This was used for all quantum neural network and quantum convolutional neural network models, as well as the vanilla convolutional neural network model. Convergence is decided as follows:

- During training, the previous 400 values of the loss are stored, and at each step after step 400, the sample mean $\mu_{1}$ of the loss values for the first 200 and the sample mean $\mu_{2}$ of the loss values of last 200 steps is calculated, as well as the standard deviation $\sigma_{2}$ of the last 200 loss values.
- If the model has converged, the statistics of the loss values in the two intervals should be approximately the same, so that the statistics of the sample means $\mu_{1}$ and $\mu_{2}$ are the same. Under an assumption of normality, the standard deviation of $\mu_{2}$ is $\sigma_{2} / \sqrt{200}$, and so for a converged model one expects

$$
\begin{equation*}
\left|\mu_{1}-\mu_{2}\right|<\frac{\sigma_{2}}{\sqrt{200}} \tag{D1}
\end{equation*}
$$

with reasonable probability, whereas a model that is still training and whose loss decreases significantly in 200 steps is unlikely to satisfy the criterion.

- We adopted a slightly smaller interval than this to decrease the likelyhood of accidental early stopping. Namely, we used the criterion

$$
\begin{equation*}
\left|\mu_{1}-\mu_{2}\right|<\frac{\sigma_{2}}{2 \sqrt{200}} \tag{D2}
\end{equation*}
$$

to decide convergence. From visual inspection of loss plots, we found that this always gave a reasonable stopping criterion that corresponded to an approximately flat loss curve over the last 200 steps.

## Appendix E: Details on the datasets used in this study

We used 6 types of data generation procedures, five of which are purely synthetic, while one (MNIST) uses postprocessing to reduce the dimension of the feature vectors. The data generation procedures provide different variables that can be tuned to create individual datasets. We refer to datasets for which one such variable is tuned as "benchmarks" and denote them by capital spelling.
This section describes the procedures in more detail, and gives the settings for generating the benchmarks. The code can be found in the github repository.

Note that unless otherwise stated, data was split into training and test sets using a ratio of test/train $=0.2$.

## 1. Linearly separable

The linearly separable data generation procedure creates data by a perceptron with fixed weights that labels input vectors sampled from a hypercube. The data is hence guaranteed to be linearly separable, and the dataset can be considered as the easiest or "fruit-fly" task in classification.

## Procedure:

1. Sample input vectors $\mathbf{x} \in \mathbb{R}^{d}$ uniformly from a $d$-dimensional hypercube spanning the interval $[-1,1]$ in each dimension.
2. Pick a weight vector for some $\mathbf{w} \in \mathbb{R}^{d}$ to define the linear decision boundary at $\mathbf{w} \mathbf{x}=0$. Only retain the first $N$ input vectors that do not lie within a margin of size $0.02 d$ around the decision boundary, or $\|\mathbf{w x}\|>0.02 d$.
3. Generate continuous-valued labels $y^{\prime}=\mathbf{w x}$.
4. Binarise the labels via

$$
y=\left\{\begin{aligned}
1 & \text { if } y^{\prime}-y_{\mathrm{med}}>0 \\
-1 & \text { else }
\end{aligned}\right.
$$

where $y_{\text {med }}$ is the median of all continuous labels. This standarisation procedure ensures that the classes are balanced.

## Settings for the LINEARLY SEPARABLE benchmark:

- Number of features $d \in\{2, \ldots, 20\}$
- Number of samples $N=300$
- Perceptron weights $\mathbf{w}=(1, \ldots, 1)^{T}$


## 2. Bars and stripes

The bars and stripes generation procedure is intended to be a simple task for the three convolutional models. It creates gray-scale images of either vertical bars or horizontal stripe on a 2D pixel grid.

Procedure:

1. Sample $N$ labels $y_{i}=-1,1$ uniformly at random.
2. For each $y_{i}$ create a pixel grid of shape $d \times d$ that will store the data $\boldsymbol{x}_{i}$. If $y_{i}=-1$, for each column, sample a random variable taking values $\pm 1$ with equal probability, and fill the column of $\boldsymbol{x}_{i}$ with this value. If $y_{i}=1$, do the same for the rows.
3. For each image $\boldsymbol{x}_{i}$ add independent Gaussian noise with standard deviation $\sigma$ and mean 0 .

Settings for the BARS \& STRIPES benchmark:

- Image width $d=4,8,16,32$.
- noise standard deviation $\sigma=0.5$
- Number of data samples $N=1000$


## 3. Downscaled MNIST

The MNIST datasets are based on the famous handwritten digits data [43] using digits 3 and 5 which are amongst the hardest to distinguish. The ratio between test and training set for this procedure is test/train $=0.17$. The original data was processed by different methods for dimensionality reduction.

The MNIST PCA, MNIST PCA- benchmarks use principal component analysis (PCA) to reduce dimensions.

## Procedure:

1. Flatten and standarise the inputs images, which is important for PCA to work well. The standarisation parameters are derived from the training set, and then used to standarise the test set. ${ }^{20}$
2. Compute the $d$ largest principal components of the pre-processed training set inputs via Principal Component Analysis.
3. Project the training and test set inputs onto those components to gain new input vectors of dimension d. This is the MNIST PCA dataset.
4. For the MNIST $\mathrm{PCA}^{-}$dataset, we sampled a subset of 250 data points for each of the training and test set from MNIST PCA.

## Settings for the MNIST PCA benchmark:

- Number of features $d \in\{2, \ldots, 20\}$

Settings for the MNIST PCA- benchmark:
- Number of features $d \in\{2, \ldots, 20\}$
- Number of samples $N=250$ for training and test set each.[^13]

The MNIST CG benchmark coarse-grains the pixels of the original images to try and preserve the correlation structure used by convolutional neural networks. As before, we use the digits 3 and 5 only.

## Procedure:

1. Resize the original 28 x28 pixel data to a pixel grid of size $H \times H$, using bilinear interpolation.
2. Flatten and standardize the images.

Settings for the MNIST CG benchmark:

- Pixel grid height/width $H \in\{4,8,16,32\}$.


## 4. Hidden manifold model

This data generation procedure is based on Goldt et al. [79], who classify data sampled from a $m$-dimensional manifold by a neural network, and then embed the data into a $d$-dimensional space. The properties of this data generation process allow the authors to compute analytical generalisation error dynamics using tools from statistical physics. The structure intends to mimic datasets used in image recognition (such as MNIST), which have been shown to effectively use low-dimensional manifolds.

Procedure:

1. Randomly sample $N$ feature vectors $\mathbf{c}^{m} \in \mathbb{R}^{m}$ with entries from a standard normal distribution. These vectors lie on the "hidden manifold".
2. Create an embedding matrix $F \in \mathbf{R}^{d \times m}$.
3. Embed the feature vectors via

$$
\begin{array}{r}
\mathbf{x}=\phi(\mathbf{F} \mathbf{c} / \sqrt{m}) \\
\text { where } \phi_{i}(\mathbf{x})=\tanh \left(x_{i}-b_{i}\right)
\end{array}
$$

4. Generate continuous-valued labels using a neural network applied to the vectors on the manifold

$$
y^{\prime}=\mathbf{v}^{T} \varphi(\mathbf{W} \mathbf{c} / \sqrt{m})
$$

using component-wise tanh functions as the activation $\varphi$, and the entries in $W \in \mathbb{R}^{m, m}, \mathbf{v} \in \mathbb{R}^{m}$ sampled from a standard distribution.

5. In order to get balanced classes, we rescale the data by subtracting the median $y_{\text {med }}$ of all labels and then apply a thresholding function

$$
y=\left\{\begin{array}{r}
1 \text { if } y^{\prime}-y_{\mathrm{med}}>0 \\
-1 \text { else }
\end{array}\right.
$$

Settings for the HIDDEN MANIFOLD benchmark:

- Number of features $d \in\{2, \ldots, 20\}$
- Number of samples $N=300$
- Manifold dimension $m=6$
- Entries of feature matrix $F$ sampled from a standard distribution

Settings for the HIDDEN MANIFOLD DIFF benchmark:

- Number of features $d=10$
- Number of samples $N=300$
- Manifold dimension $m \in\{2, \ldots, 20\}$
- Entries of feature matrix $F$ sampled from a standard distribution


## 5. Two curves

This data generation procedure is inspired by Buchanan et al. [80], who consider data sampled from two curves - one for each class - embedded into a $d$ dimensional space to prove that the maximum curvature and minimum distance of these curves determine the resources required by a neural network to generalise well. We can hence understand curvature and distance as two variables that influence the difficulty of the data.

To control the curvature we use a one-dimensional Fourier series of a maximum degree $D$ in each dimension. To control the average distance of the curves we use the same embedding, but shift one curve by some constant.

## Procedure:

1. Sample $N$ values $t \in \mathbb{R}$ uniformly at random from the interval $[0,1]$. (This value defines the position of a data point on the curve we embed.)
2. To create the inputs for class 1 , embed half of the $t$-values into a $d$-dimensional space via a Fourier series defined in every dimension,

$$
x_{i}=\sum_{n=0}^{D} \alpha_{n}^{i} \cos (n t)+\beta_{n}^{i} \sin (n t)+\epsilon
$$

where $D$ is the maximum degree of the Fourier series and $\left\{\alpha_{n}^{i}\right\},\left\{\beta_{n}^{i}\right\}$ are real-valued Fourier coefficients that we sample uniformly from the interval $[0,1]$. The noise factor $\epsilon$ determines the variance of a random "spread" added to the curves around their trajectory in the high-dimensional space.

3. To create the inputs for class -1 , embed the other half of the $t$-values using the same procedure and Fourier coefficients, but adding an offset of $\Delta$ to each dimension.
Settings for the TWO CURVES benchmark:

- Number of features $d \in\{2, \ldots, 20\}$
- Number of samples $N=300$
- Noise factor $\epsilon=0.01$
- Maximum degree $D=5$
- Curve offset $\Delta=0.1$

Settings for the TWO CURVES DIFF benchmark:

- Number of features $d=10$
- Number of samples $N=300$
- Noise factor $\epsilon=0.01$
- Maximum degree $D \in\{2, \ldots, 20\}$
- Curve offset $\Delta=\frac{1}{2 D}$


## 6. Hyperplanes and parity

We created an artificial dataset that classifies lowdimensional feature vectors by whether they lie on the "positive" side of an even or odd number of a set of $k$ hyperplanes. The feature vectors are then embedded into a higher-dimensional space via a linear transform. The result is a division of the space into regions of different classes that are delineated by hyperplane intersections. The parity operation makes sure that a model implicitly has to learn all hyperplane positions to guess the right label. The difficulty of the classification problem is expected to increase with the number of hyperplanes.

## Procedure:

1. Sample $N$ feature vectors $\mathbf{c} \in \mathbb{R}^{m}$ from a standard normal distribution.
2. Embed each feature vector into $\mathbb{R}^{d}$ by multiplying the feature vectors with a matrix $\mathbf{M} \in \mathbb{R}^{d \times m}$,

$$
\mathbf{x}=\mathbf{M c}
$$

3. Compute $k$ predictions for the $m$-dimensional feature vectors via

$$
p^{(j)}=\left\{\begin{array}{l}
1 \text { if } \mathbf{w}^{(j)} \mathbf{c}+b^{(j)}>0 \\
-1 \text { else }
\end{array} \quad j=1, \ldots, k\right.
$$

using uniformly sampled weight vectors $\left\{\mathbf{w}^{(j)} \in\right.$ $\left.\mathbb{R}^{m}\right\}$ and biases $\left\{b^{(j)}\right\}$.

4. The final label is defined as the parity of these predictions, or whether the number of 1-predictions is even:

$$
y=\left\{\begin{aligned}
& 1 \text { if } \sum_{j=1}^{k} \frac{p^{(j)}+1}{2} \text { even } \\
&-1 \text { else }
\end{aligned}\right.
$$

5. To ensure balanced classes we initially sample a larger number of datapoints from which we subsample the desired number for each class.
6. Standarise the inputs.

Settings for the HYPERPLANES DIFF benchmark:

- Dimension $d=10$
- Number of hyperplanes $k \in\{2, \ldots, 20\}$
- Number of data samples $N=1000$
- Dimension of hyperplane and initial feature vectors $m=3$
- Entries of $\mathbf{M}$ are uniformly sampled from $[0,1]$


## Appendix F: Collection of detailed results

We add the ranking and accuracy plots of all benchmarks here for readers who are interested in the detailed results.

## LINEARLY SEPARABLE

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-37.jpg?height=1450&width=1412&top_left_y=251&top_left_x=365)

FIG. 20. Ranking plots like shown in Figure 10 for selected benchmarks.

## HIDDEN MANIFOLD

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-38.jpg?height=618&width=680&top_left_y=249&top_left_x=365)

TWO CURVES

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-38.jpg?height=615&width=683&top_left_y=950&top_left_x=366)

HIDDEN MANIFOLD DIFF

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-38.jpg?height=612&width=680&top_left_y=255&top_left_x=1080)

TWO CURVES DIFF

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-38.jpg?height=615&width=680&top_left_y=950&top_left_x=1080)

HYPERPLANES DIFF

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-38.jpg?height=615&width=680&top_left_y=1655&top_left_x=728)

FIG. 21. Ranking plots like shown in Figure 10 for selected benchmarks (continued).
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-39.jpg?height=2054&width=1724&top_left_y=198&top_left_x=211)

$\rightarrow$ CircuitCentricClassifier

-- DataReuploadingClassifier

$\simeq$ DressedQuantumCircuitClassifier

$\rightarrow$ IQPVariationalClassifier

$\rightarrow$ QuantumMetricLearner

$\rightarrow$ - QuantumBoltzmannMachine

- SVC

$\longrightarrow$ TreeTensorClassifier

$\rightarrow$ IQPKernelClassifier

$\rightarrow$ ProjectedQuantumKernel

$\rightarrow$ QuantumKitchenSinks

FIG. 22. Detailed training and test accuracies for the benchmarks not shown in Figure 11.

HIDDEN MANIFOLD
![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-40.jpg?height=426&width=848&top_left_y=226&top_left_x=194)

HIDDEN MANIFOLD DIFF

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-40.jpg?height=352&width=414&top_left_y=667&top_left_x=194)

TWO CURVES

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-40.jpg?height=355&width=418&top_left_y=1086&top_left_x=192)

TWO CURVES DIFF

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-40.jpg?height=355&width=414&top_left_y=1511&top_left_x=194)

- MLPClassifier

$\sim$ CircuitCentricClassifier

-- DataReuploadingClassifier

- DressedQuantumCircuitClassifier

$\rightarrow$ IQPVariationalClassifier

-- QuantumBoltzmannMachine

TreeTensorClassifier

$\rightarrow$ QuantumKitchenSinks

$\longrightarrow$ SeparableVariationalClassifier $\_$SeparableKernelClassifier

FIG. 23. Results for separable models for the benchmarks not shown in Figure 12.

![](https://cdn.mathpix.com/cropped/2024_06_04_e53d13f86f8cf396c755g-41.jpg?height=734&width=854&top_left_y=178&top_left_x=191)

FIG. 24. Total ranking results as reported in Figure 10, but with the two separable models included.


[^0]:    * joseph@xanadu.ai

    $\dagger$ shahnawaz.ahmed95@gmail.com

    $\ddagger$ maria@xanadu.ai

    ${ }^{1}$ It is standard practice in the field of quantum computing to publish articles on the arXiv server, and we therefore expect it to provide representative samples of the literature. The 55 papers are a subset of 73 papers returned by the keyword search, and were selected by manually reading the abstracts and discarding papers that related to quantum-inspired methods, other fields than quantum machine learning, or did not explicitly mention a method outperforming another.

[^1]:    2 Other search terms than "outperform*", such as "better than" or "benchmark*", were tried to detect a possible selection bias caused by the keyword search, but showed similar patterns.

[^2]:    ${ }^{3}$ We ran our simulation on the Digital Research Alliance of Canada's Cedar supercomputer and limited runtimes to at most 24 hours for one 5 -fold cross-validation (i.e., training the model five times) using a 10-core cluster with 40GB RAM. The largest size of a circuit simulated in the study has 18 qubits, a number that we are determined to push further going forward.

[^3]:    ${ }^{4}$ Calls for more methodological rigour also led to the inception of datasets \& benchmarking tracks at leading conferences like NeurIPS.

    5 This raises once more the question whether existence proofs for quantum advantages in learning brings us any closer to useful quantum machine learning.

    6 Although, reassuringly, the ranking between models is largely unaffected by this intervention.

[^4]:    7 These were sampled out of 46 papers that we identified to use MNIST for supervised learning benchmarks listed in the arxiv 'quant-ph' category between January 2018 and June 2023. The papers were in turn selected from a total of over 100 papers in the 'quant-ph' category on the arXiv in the same period that mention MNIST in title or abstract.

    8 A notable exception are papers based on so-called "quanvolutional" architectures, where the first layer uses a small "filter" or "kernel" composed of a quantum circuit [15].

[^5]:    9 The choice of square and linear loss in some models is curious, since it can be argued that a more natural choice for a classification loss is the cross entropy loss, as it corresponds to the maximum likelihood estimator of the data [60]. A square loss, instead, is known to be more naturally suited to regression problems with continuous labels.

[^6]:    10 As with all variational models, it can be difficult to know whether a model has converged or if the model is stuck on a particularly flat part of the optimisation landscape, and test accuracies can both improve or worsen with more training. The choice to decide on convergence over 200 updates is therefore to some extent arbitrary.

[^7]:    11 Note that a subtlety here is the size of the margin between the classes in increasing dimensions, which has an influence on how easy it is to generalise from the training data.

[^8]:    14 We normalise the rank since not all models competed in all benchmarks, and it is easier to have a better rank when competing in experiments with fewer overall competitors.

[^9]:    16 We chose to run this experiment on MNIST PCA dimensions as the quantum models showed a prototypical performance with respect to their overall rankings, and the variations between individual datasets of different input dimensions were small.

[^10]:    17 We found that the test set Gram matrices do not lead to different results.

[^11]:    18 An exception is the TWO CURVES benchmark, where all Gram matrices are similar.

[^12]:    19 It is interesting to note that in a similar manner, WeiNet - the only non-hybrid quantum convolutional neural network we tested - performs badly on BARS \& STRIPES which we considered to be a very simple task.

[^13]:    20 This best practice takes into account that in applications one does not necessarily have access to the test set at training time.

</end of paper 2>


<paper 3>
# Hype or Heuristic? Quantum Reinforcement Learning for Join Order Optimisation 

Maja Franz<br>Technical University of<br>Applied Sciences Regensburg<br>Regensburg, Germany<br>maja.franz@othr.de

Tobias Winker<br>University of Lübeck<br>Lübeck, Germany<br>t.winker@uni-luebeck.de

Sven Groppe<br>University of Lübeck<br>Lübeck, Germany<br>sven.groppe@uni-luebeck.de

Wolfgang Mauerer<br>Technical University of<br>Applied Sciences Regensburg<br>Siemens AG, Technology<br>Regensburg/Munich, Germany<br>wolfgang.mauerer@othr.de


#### Abstract

Identifying optimal join orders (JOs) stands out as a key challenge in database research and engineering. Owing to the large search space, established classical methods rely on approximations and heuristics. Recent efforts have successfully explored reinforcement learning (RL) for JO. Likewise, quantum versions of RL have received considerable scientific attention. Yet, it is an open question if they can achieve sustainable, overall practical advantages with improved quantum processors.

In this paper, we present a novel approach that uses quantum reinforcement learning (QRL) for JO based on a hybrid variational quantum ansatz. It is able to handle general bushy join trees instead of resorting to simpler left-deep variants as compared to approaches based on quantum(-inspired) optimisation, yet requires multiple orders of magnitudes fewer qubits, which is a scarce resource even for post-NISQ systems.

Despite moderate circuit depth, the ansatz exceeds current NISQ capabilities, which requires an evaluation by numerical simulations. While QRL may not significantly outperform classical approaches in solving the $\mathrm{JO}$ problem with respect to result quality (albeit we see parity), we find a drastic reduction in required trainable parameters. This benefits practically relevant aspects ranging from shorter training times compared to classical RL, less involved classical optimisation passes, or better use of available training data, and fits data-stream and low-latency processing scenarios. Our comprehensive evaluation and careful discussion delivers a balanced perspective on possible practical quantum advantage, provides insights for future systemic approaches, and allows for quantitatively assessing trade-offs of quantum approaches for one of the most crucial problems of database management systems.


Index Terms-Quantum Machine Learning, Reinforcement Learning, Query Optimisation, Database Management Systems

## I. INTRODUCTION

In database research and industrial practice, finding good orders in which joins between table columns are executed in a query-the so-called join order (JO) problem-counts amongs the most fundamental issues of database management systems (DMBS) [1]-[9]. The chosen order substantially impacts query execution time. While the problem does only need little amounts of input information (the query to be executed, and characteristics of the payload data obtained from statistical samples), the problem is know to be NP-hard in general, and also for common restricted scenarios [10]. An optimal JO cannot be efficiently found deterministically. The last few decades have seen various classical heuristics that can find suboptimal JOs in polynomial time [11]-[13].
Recent classical work [14]-[21] explores the application of reinforcement learning (RL) to tackle the JO problem. RL is considered to be beneficial in scenarios where the solution to a problem can be determined by a series of subsequent decision steps, and where finding one such good sequence for a problem generalises well to others, or when highly dynamic problems are considered. By learning from experience of past query evaluation, RL can find good decision sequences in vast search spaces, and only requires information about the current state of the system. This is particularly advantageous for the JO problem, as very typical scenarios in database systems need to process information at a high temporal frequency.

In this paper, we approach RL for JO from the perspective of quantum machine learning (QML), an emerging technique that leverages the principles of quantum mechanics for potential computational speed-ups. It has been shown that certain problems [22], [23] can be solved more efficiently using quantum algorithms over classical approaches. However, the practical utility of these algorithms is limited on the current generation of quantum computers, so-called noisy intermediate-scale quantum (NISQ) systems [24], as they only offer a limited amount of qubits and are prone to noise and imperfections [25] that strongly limit possible circuits depth and thus the length of quantum computations. To address these limitations, hybrid quantum-classical algorithms are proposed, where only a limited number of steps is performed on a quantum computer and the remaining steps on classical machines. As Pirnay et al. [26] show, fault-tolerant quantum computers can provably provide super-polynomial advantage for optimisation problems over classical algorithms. Hybrid variational algorithms [27]-[29] are considered key candidates for exploiting advantages of near-term quantum devices, but could also be beneficial in post-NISQ systems because of their resource efficiency.

Within the class of hybrid variational algorithms, quantum machine learning (QML) has shown promise by moving certain parts of classical machine learning to quantum computers. QCs will, despite common misperceptions, likely be inapt for handling large amounts of data [30]. This makes quantum reinforcement learning (QRL) [31]-[33], which requires little training data, a promising approach. As established JO approaches mostly rely on statistical estimates of properties of the database, JO seems a good match for QRL.

QML in general has been shown to outperform classical machine learning for certain tasks [34]-[40]. Specifically, for QRL it is hypothesised that fewer parameters are required than for classical neural networks (NNs) to address RL tasks [31], [41]. Several studies also suggest that QRL can solve tasks that are intractable to classical machine learning [42], or that it may have an advantage over classical NNs in terms of sampling complexity, that is, fewer interactions with the environment are required to achieve optimality for certain problems [32], [33]. For these reasons, the application of QML to database problems is also considered promising [43].

However, as detailed in Ref. [44], many approaches for QML that claim quantum advantage rest on artificially constructed scenarios (e.g., [37], [38], [40]). Consequently, a practical definition of QML goals is required, which should not imply an exponential speed-up compared to classical approaches, but rather is a matter of details.

We have chosen to use a recent classical RL-based approach to join ordering by Marcus and Papaemmanouil [14] as baseline that is well aligned with intensively studied quantum variants of reinforcement learning [45]. It is known that a careful consideration of various factors is necessary to gauge potential improvements. This includes a sound classical baseline, data representation, quantum circuit structure, and hyperparameters. Further, we provide a high-level evaluation of hardware requirements. Our detailed contributions are:

- We systematically replicate ${ }^{1}$ the classical baseline [14] and generalise it to the quantum case. As the baseline does not provide source code or hyperparameters, this is an important prerequisite to ascertain a fair comparison, and allows us to consider all aspects of the DBMS.
- We comprehensively simulate the performance of our approach on the join order benchmark (JOB), which is a universally accepted touchstone in the database community, and compare it against the classical baseline and a single-step QML technique [46] that was shown to outperform established classical approaches. Multi-step QRL can achieve up to $17 \%$ lower median costs than single-step QML on the selected dataset and cost model.
- We identify potentials for improvement in view of future hardware development, and carefully address the issue of judging realistic potentials for practical improvements over classical heuristics.
- We provide an open-source reproduction package [47] that makes our code transparent to the community, and can serve as basis to build further experiments upon, and benchmark alternative approaches against.

We aim to provide a comprehensive perspective on the quantum advantage landscape in RL for the JO problem. By combining optimistic hypotheses with an acknowledgement of established challenges and limitations, we strive to present a balanced view. This balance is important to guiding future[^0]

research directions and manage expectations regarding the (near- and far-term) practical benefits of quantum algorithms in the field of database management systems.

The paper is structured as follows: Sec. II reviews existing literature on classical approaches for the JO problem and QC for databases. Sec. III describes the theoretical background for the application of the JO problem and the method of classical and quantum RL, followed by an overview of our methodology in Sec. IV. Sec. V outlines our experiments, which are discussed in Sec. VI. We conclude in Sec. VII.

## II. RELATED WORK

The problem of query optimisation, which is formally defined in Sec. III-A, has been studied for over 40 years [13], and new results appear frequently [3], [4], [8], [9]. Since the search space for the JO problem scales factorial [10], an exhaustive search for the optimal $\mathrm{JO}$ is only feasible for a small number of relations, even when relying on dynamic programming (DP) approaches [13], [48]-[51], necessitating heuristic methods [1], [11], [52]-[55] for large queries.

Heuristics require to calculate costs; for instance, execution time or number of intermediate results. These, in turn, depend on estimates of the cardinalities of subqueries. Ref. [56] reviews cardinality estimation techniques and their impact on JO optimisation. Some approaches apply machine learning for cardinality or cost estimation [57]-[59], to improve the DP optimiser, or to directly determine the JO [14], [16]-[19].

Using quantum approaches to address database problems is a relatively new field of research, even with early work by Trummer and Koch on solving multi-query optimisation with quantum annealers only going back to 2016 [60]. A recent review [61] summarises existing work and classifies potential use-cases. For instance, transaction scheduling [43], [62]-[64] schema matching [65] or tuning index configurations [66] have been addressed using quantum methods.

The join order problem has been cast as an optimisation problem in quadratic unconstrained binary optimisation (QUBO) form by Schönberger et al. based on known transformations to mixed-integer linear programming [67], and using a direct encoding that has also been evaluated on quantum-inspired hardware [68]. These two solutions for the JO problem are restricted to left-deep join trees; alternative formulations that allow for handling general bushy join trees were given by Nayak et al. [69] and Schönberger et al. [70] (we discuss differences in their scalability in Sec. VI). Finally, Ref. [46] introduces an RL inspired approach for the JO problem using VQCs. It uses rewards to measure the quality of different join orders, but creates a join order in a single step and not over multiple interactions with an environment.

## III. PRELIMINARIES

This section introduces the three main concepts relevant to this work, namely the JO problem (Sec. III-A), and classical (Sec. III-B) and quantum (Sec. III-C) RL.

## A. Background on the Join Order Problem

The JO problem constitutes of three basic elements:

1) Query: A query formulated in the structured query language (SQL) (see left of Fig. 2 for an example), can be represented as an expression of relational algebra to be optimised before execution [71]. In this work, we focus on the important problem of $\mathrm{JO}$ optimisation with consideration of selection (i.e., filter) operations while the query, in general, may also consist of other operations. Here, a query $Q$ can be characterised by a join graph and predicates, which can be further decomposed into join predicates and selection predicates. A join graph for a query is defined by relations that represent the vertices of the graph and filter on which two relations can be joined. These are called join predicates (e.g., a1.a=D . a in Fig. 2) and correspond to the edges of the join graph. The join graph is given by a symmetric adjacency matrix $G \in \mathbb{F}_{2}^{r \times r}$, where $r$ is the number of relations. If there is a join predicate in $Q$ connecting the relations $r_{i}$ and $r_{j}$, the entry $g_{i, j}$ in $G$ is 1 . Selection predicates are additional filters that act on one relation (e.g., D.c $>5$ in Fig. 2), and can be formalised as described in Par. IV-C1b or Par. IV-C2a.
2) Join Tree: In contrast to a query graph, which serves as the input for the JO problem, a join tree embodies a solution. Its leaf nodes represent the base relations to be joined, while its intermediate nodes denote join operations. Each join node, requiring two operands, has two predecessors: either a) a base relation or $\mathrm{b}$ ) another join tree node, which itself will be further joined. The result of a join serves as an operand for another join, indicated by an outgoing edge connecting to its successor. The only exception is the final join, which does not serve as an operand for any subsequent join. In this study, we refer to intermediate join trees as "sub-trees". The top of Fig. 3 illustrates the sequence for constructing a complete join tree.

While these requirements apply universally to join trees, certain JO methods impose additional constraints on their structure to enhance efficiency by reducing the search space. Particularly, some methods exclusively consider left-deep join trees, necessitating at least one base relation as an operand for each join. Consequently, directly joining two pairs of relations is precluded, as it necessitates a join operation on the results of two preceding joins. Valid left-deep join orders must therefore represent a permutation of relations. This restriction to leftdeep trees was employed in two quantum approaches for JO [67], [68]. Nonetheless, the detrimental impact of this constraint on solution quality can be significant, as demonstrated, for instance, by the empirical analysis conducted by Neumann and Radke [3]. Hence, our QML approaches consider general or bushy join trees, devoid of further structural constraints. The divergence in scalability between existing quantum-based left-deep and bushy variants is described in Sec. VI-B.

3) Cost Functions: Finally, a cost function evaluates the join tree, by assigning it a cost value. The literature proposes various definitions of cost functions [72]; some are straightforward yet less precise, while others are more intricate, taking into account multiple factors and closely reflecting real costs (i.e., query execution time including I/O costs). To evaluate the selected join order, we use the established cost function $C_{\text {out }}$ [10], which considers the cardinalities (i.e., the number of tuples in a query result set) as an approximation of query complexity:

$$
\begin{equation*}
C_{\text {out }}(T)=|T|+C_{\text {out }}\left(T_{1}\right)+C_{\text {out }}\left(T_{2}\right) \tag{1}
\end{equation*}
$$

where $n$ is the maximum number of joins in the query, a join tree is defined as $T=T_{1} \bowtie T_{2}$, and $|T|$ represents the true cardinality of $T\left(C_{\text {out }}(T)=0\right.$ if $T \in\left\{r_{1}, r_{2}, \ldots\right\}$ is a leaf).

## B. Background on Reinforcement Learning

The setup in RL is typically described by the notion of a Markov decision process (MDP) [73], where an agent interacts with an environment at discrete time steps $t$. In each time step, the current configuration of the agent in the environment is summarised by the state $S_{t} \in \mathcal{S}$, where $\mathcal{S}$ is the set of all possible states. Based on this information, the agent selects an action $A_{t}$ from a set of possible actions $\mathcal{A}$ according to a policy $\pi(s, a)=\mathbb{P}\left[A_{t}=a \mid S_{t}=s\right]$, which gives the probability $\mathbb{P}$ of taking action $a$ in state $s$. Executing the selected action causes the environment to transition to a next state $S_{t+1} \in \mathcal{S}$. Simultaneously, the agent receives a scalar reward $R_{t+1} \in \mathcal{R}$ that quantifies the contribution of the selected action towards solving the task, with $\mathcal{R} \subset \mathbb{R}$ being the set of all rewards. $S_{t+1}$ and $R_{t+1}$ are determined by the environment's dynamics $p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A}$, which characterises the probability distribution of a transition $\left(S_{t}, A_{t}, R_{t+1}, S_{t+1}\right)$.

The agent's goal is to maximise the return [73] $G_{t}=$ $\sum_{t^{\prime}=t}^{T} \gamma^{t^{\prime}-t} R_{t^{\prime}+1}$, that is, the discounted sum of rewards, until a terminal timestep $T$ is reached, where the discount factor $\gamma \in(0,1]$ controls how much the agent favours immediate over future rewards. The period between the initial time step and $T$ is often referred to as an episode.

To find a good policy that maximises the return, various RL methods exist [73]. As our baseline [14], in this work we focus on Proximal Policy Optimization (PPO) from the class of policy gradient methods [74]. The goal of policy gradient methods is to directly learn the parameterised policy $\pi_{\boldsymbol{\theta}}: \mathcal{S} \times \mathcal{A} \rightarrow[0,1]$, where $\boldsymbol{\theta}$ denote trainable parameters of a function approximator, such as a neural network (NN), or a variational quantum circuit (VQC). In PPO, the parameters $\theta$ can be optimised using a gradient ascent method, maximising the following objective, consisting of three parts:

$$
\begin{equation*}
L_{t}^{\text {clip }+\mathrm{VF}+\mathrm{S}}(\boldsymbol{\theta})=\mathbb{E}_{t}\left[L_{t}^{\text {clip }}(\boldsymbol{\theta})-c_{1} L_{t}^{\mathrm{VF}}(\boldsymbol{\theta})+c_{2} S\left(\pi_{\boldsymbol{\theta}}\right)\right] \tag{2}
\end{equation*}
$$

The PPO algorithm alternates between sampling and optimisation stages. Therefore, $\mathbb{E}_{t}$ indicates the average over a finite batch of samples, which is gathered prior to each optimisation stage. $c_{1}$ and $c_{2} \in \mathbb{R}^{+}$are hyperparameters. The clip-objective $L^{\text {clip }}(\boldsymbol{\theta})$, is defined as $r_{t}(\boldsymbol{\theta}) \mathbb{A}_{t}$, where the ratio $r_{t}(\boldsymbol{\theta})=\frac{\pi_{\boldsymbol{\theta}}\left(a_{t}, s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t}, s_{t}\right)}$ is clipped in $1 \pm \epsilon$ with $\boldsymbol{\theta}_{\text {old }}$ being the parameters before the update, $\epsilon \in \mathbb{R}$ a hyperparameter and $\mathbb{A}_{t}$ an advantage estimation of the current policy. The advantage estimation $\mathbb{A}_{t}$ itself can be learned by a function approximator based on the value function in an MDP $V(s)=\mathbb{E}_{t}\left[G_{t} \mid S_{t}=s\right]$, that is an estimation of the return, and optimised through the objective $L_{t}^{\mathrm{VF}}$, which is a squarederror loss function of estimated values from the function
approximator and target values, collected in the sampling stage. The third part in Eq. $2 S\left[\pi_{\theta}\right]$ denotes the entropy of $\pi_{\theta}$, which is added to ensure sufficient exploration. For a detailed discussion on PPO, we point readers to Ref. [74].

We refer to the policy function approximator that mainly contributes to $L^{\text {clip }}(\boldsymbol{\theta})$ as the actor, as it represents the policy that "acts" in the environment and to the advantage estimator, which is optimised through $L^{\mathrm{VF}}(\boldsymbol{\theta})$ as the critic, which evaluates a current policy. We investigates classical and quantum versions of actor/critic in Sec. V-B.

## C. Background on Quantum Machine Learning

As a variational quantum circuit (VQC) is proven to be a universal function approximator [75], similar to a classical NN [76], it can be employed as a set-in for NNs in a variety of settings (e.g., [27], [77]), including PPO. A VQC's structure often follows the data processing flow of a classical $\mathrm{NN}$ and comprises three fundamental components: In the first part, a quantum state is prepared to represent the classical input data $\boldsymbol{x}$ through applying a unitary gate $\hat{U}_{\text {enc }}(\boldsymbol{x})$ to the initial quantum state, which by convention is $\otimes_{n}|0\rangle$ for a configuration with $n$ qubits [78]. In the second so-called variational part, the quantum state is then transformed by applying a parameterised unitary $\hat{U}_{\text {var }}(\boldsymbol{\theta})$. An exemplary gate sequence for the encoding and the variational part is depicted in Fig. 4. Finally, classical information $\langle\hat{O}\rangle$ is obtained from the quantum circuit by measuring the state. The notation $\langle\hat{O}\rangle$ refers to the expectation value of an observable $\hat{O}$.

The parameters of the VQC are optimised using classical approaches such as gradient ascent to maximise an objective function, where the gradient of a parameter with respect to the measurement can be calculated using the parametershift rule [77], [79]. As algorithms involving VQCs perform calculations on both, the quantum processing unit (QPU) and CPU, they are called hybrid approaches.

1) Data Encoding: The encoding unitary $\hat{U}_{\text {enc }}(\boldsymbol{x})$ depends on the encoding strategy; Weigold et al. [80], [81] survey common strategies. Among these, we focus on angle encoding, which uses a Pauli-rotation gate to encode one real value into one qubit. The corresponding unitary can comprise one (e.g., [32]) or multiple (e.g., [31]) parameterised rotation gates per qubit. Given that the gates are periodic, each input element must be scaled to an interval smaller than $2 \pi$.

Even if payload data are not required to encode JO problems, a simple angle encoding scheme for JO exceeds the capability of NISQ devices for even small instances. We therefore employ incremental data uploading [82] to spread the encoding gates for the input elements throughout the quantum circuit with parameterised unitaries in between them, which increases circuit depth (i.e., the longest gate sequence), but decreases qubit count. As there is no limit on the maximum number of repetitions of input elements, encoding unitaries can be re-introduced multiple times into the VQC. This approach, known as data re-uploading (DRU) [75], is suggested to increase the expressivity of a VQC [83], which in turn determines the class of functions a VQC can approximate. In
Sec. V-B we empirically evaluate and compare the combination of incremental data uploading and DRU.

2) Data Decoding: Several techniques are known to map "outputs" of a VQC (i.e., the expectation value of multiple measurements) to a set of output values that is smaller than or equal to the number of qubits [42], [84]. Few existing approaches [85] decode quantum states to larger output spaces. As described in Sec. IV-C1, action and output space are typically larger than the number of qubits for JO. We therefore determine the expectation value for each qubit individually using $\hat{Z}$ observables and feed the outcomes into one classical NN layer with the correct output size for the actor. For the critic model, which only requires one output component to estimate the advantage, circuit outcome is determined by observable $\otimes_{n} \hat{Z}$. Since the expectation value of $\hat{Z}$ lies in $[-1,1]$ the critic model outcome is scaled using an additional trainable classical parameter and bias.

## IV. MethodOLOGY

To understand how RL can be utilised for the JO problem on QCs, we commence with discussing the differences between building the join order step-wise or returning the full join order within one single step using a machine learning (ML) model. We also introduce a single-step approach based on QML. Subsequently, we outline our classical baseline ReJoin and the adjustments required for quantum RL.

## A. Single-Step versus Multi-Step Join Ordering

Join (action)

Multi-Step $A \bowtie(B \bowtie C), D \bowtie E, F$ Single-Step $(A \bowtie(B \bowtie C)) \bowtie((D \bowtie E) \bowtie F)$

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-04.jpg?height=469&width=886&top_left_y=1495&top_left_x=1075)

Fig. 1. Single-step versus multi-step approach presented in an RL fashion. Here, $A$ to $F$ are the relations to join. We neglect selection predicates.

A join tree can be created by an ML model in multiple steps or in a single step. Fig. 1 summarises the differences: The state of the environment contains (1) already determined subjoins, (2) relations to be joined, and (3) selection predicates. An ML model acts as an agent in the RL context. It predicts and emits the next best subjoin (i.e., the action) connecting two of the subjoins and relations of the previous state. Thereby, the environment of already determined subjoins and to be joined relations is updated. By determining the quality (i.e., the reward) of the intermediate join order, the model can be
trained to better predict the next best subjoin. In the multistep approach, a single join is added to the join tree in each step until all input relations are joined (i.e., a terminal state is reached). In the single-step approach [46], the model directly generates a complete join order without intermediate steps (i.e., a terminal state is reached after one step).

## B. Single-Step $Q M L$

In the single-step QML approach [46], all join orders are enumerated, and each join tree is associated with a quantum state. The join order associated with the most commonly measured quantum state is taken as join tree. The quality of join orders can vary greatly and multiple join orders can have equal or nearly equal good quality. For instance, the secondbest join order might only be slightly worse, while another might have a large difference in quality. Thus, it is not a good approach to make a binary choice between right or wrong for a join order. Instead, each join order is assigned a reward depending on its quality. We use the VQC to predict these rewards and choose the join order with the highest reward.

## C. Multi-Step QRL

Fig. 2 visualises our QRL multi-step approach. By using a state representation based on Ref. [14], a VQC can choose the next join in an iterative process until a complete join order is built. The classical baseline as well as the modifications required for the application of QRL are described below.

1) Classical Baseline-ReJoin: For our multi-step approach, we utilised the method described in Ref. [14]. Although the literature proposes various RL methods for the JO problem (cf. Sec. II), we opted for ReJoin as a foundation because of its compact input feature space. Other approaches, such as RTOS [19] or JOGGER [18], utilise sophisticated classical machine learning techniques to represent states of queries and databases, which lack a direct equivalent in the domain of quantum computing. Investigating novel methods that apply these advanced classical machine learning techniques to a quantum domain is beyond the scope of this study. Instead, our QRL approach should evaluate the capabilities of existing QML methods on small input spaces of the JO problem to establish a lower bound for the potential of using QRL, or QC in general. Additionally, due to the limited number of qubits on current NISQ devices and each quantum circuit gate being a potential source for noise and imperfections, it is beneficial to reduce the classical data encoded into the quantum gates to a minimum. As outlined below, ReJoin employs a total of $a+2 r^{2}$ input features, where $r$ denotes the number of tables and $a$ represents the total number of attributes in the database with $a>r$. As we show in Par. IV-C2a we are able to reduce the input space even further. In contrast, for example $D Q$ [16] necessitates roughly $r \times(a+1)$ features, resulting in a larger input space considering that the number of attributes typically outweighs the number of relations in the database.

a) MDP: The MDP's state for the JO problem is represented by a query $Q$ and a set of relations or (sub-)jointrees $\mathcal{F}$. The PPO agent sequentially combines two sub-trees $T_{k}, T_{l} \in \mathcal{F}$, which corresponds to an action, until a complete join order is build. Building the join order for one query, represents an episode. The agent aims for a join order that achieves minimum costs respectively a maximum reward.

b) State Representation: Formally, one part of the state representation is the join graph $G$, defined in Sec. III-A. Additionally, selection predicates in the query $Q$ are represented by a vector of length $a$, which is the number of attributes in the database. Selection predicates are one-hot encoded: If a predicate is present in $Q$, the corresponding value in the predicate vector $P$ is one; otherwise zero. Furthermore, each intermediate sub-tree $T_{k} \in \mathcal{F}$, that is the tree structure, is encoded as a row vector $\tau_{k}$ of size $r$. If a relation $r_{i}$ is equal to $T_{k}$ ( $r_{i}$ is a leaf) or is present in $T_{k}$, then the corresponding value in the row vector $\tau_{k, i}$ is $\frac{1}{h(i, k)}$, where $h(i, k)$ is the height of $r_{i}$ in $T_{k}$. To ensure an evenly sized input space throughout the training process, for each subtree $T_{k}$ that is successfully joined to another subtree $T_{l}, \tau_{k}$ is set to $\overrightarrow{0}$. There exist $r$ sub-tree row vectors $T$ in total, since at the beginning of each join-process each relation correspond to one sub-tree. An exemplary sequence of row vectors that is encountered until a full join order is built is depicted in Fig. 3, which uses the reduced encoding introduced in Par. IV-C2a. The complete state for the baseline can be expressed through concatenation, $S_{t}=G^{f} \oplus P \oplus(\oplus_{\tau_{k} \in \overbrace{T}} \tau_{k})$, where $G^{f}$ denotes the flattened join graph as a vector and $\oplus$ concatenation with $\left|S_{t}\right|=a+2 r^{2}$.

c) Action Representation: The PPO actor returns a probability distribution over all actions $A_{t} \in \mathcal{A}$. The set of actions $\mathcal{A}$ comprises all combinations of two sub-trees $\left(T_{k}, T_{l}\right) \forall T_{k}, T_{l} \in \mathcal{F}, k \neq l$, resulting in an action space of size $r \times(r-1)$. It encompasses actions with relations that are not present in the query, or lead to a cross join (i.e. a join between relations that are not connected by a join predicate). As these typically involve high costs, we apply a mask to the policy by multiplying each value that represents an invalid action with zero to prevent them from being sampled.

d) Reward Signal: In previous studies on RL for JO (e.g., Refs. [14], [16], [19]) the reward, as function of cost, is only assigned at the end of each episode when the full join order is built by the RL policy. Intermediate steps receive a zero reward. This seems counter-productive, given that one property of RL is to determine an action based on a current state and reward signal ${ }^{2}$. Therefore, we propose a multi-step reward signal: Assuming the cost difference $C_{t}$ between timesteps $t$ and $t-1$ with costs $c_{k}$ for subtrees $T_{k} \in \mathcal{F}_{t}$ in a state $S_{t}$ is

$$
C_{t}= \begin{cases}\sum_{T_{k} \in \mathcal{F}_{t}} c_{k}-\sum_{T_{l} \in \mathcal{F}_{t-1}} c_{l} & \text { if } t>0  \tag{3}\\ 0 & \text { if } t=0\end{cases}
$$

and the cost assigned to the best join order of the full query determined by a DP exhaustive search is $C_{\mathrm{DP}}$, we propose the clipped reward at $t$ as

$$
\begin{equation*}
R_{t}=\frac{1}{n-1}\left[-\min \left(\frac{C_{t}}{C_{\mathrm{DP}}}, n-1\right)+2\right] \tag{4}
\end{equation*}
$$[^1]

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-06.jpg?height=1168&width=1808&top_left_y=161&top_left_x=148)

Fig. 2. Interplay between data encoding (top) and variational quantum circuit (bottom) processing in our approach. Starting from the query and the baseline encoding of Ref. [14], we prune unnecessary features and flatten the core input data into a vector that is statically fed into the encoding quantum gates $\hat{U}_{\text {enc }}$. The variational quantum circuit (using a configurable number of qubits) is initialised with qubits in state $|0\rangle$, and iteratively executes block of intermingled encoding and variational ( $\hat{U}_{\text {var }}$ ) gates; following a measurement, a classical optimisation procedure delivers new parameter estimates for the variational gates, and the updated circuit is iteratively re-executed. Following established conventions, solid lines indicate quantum information, double lines concern classical information (measurement results that may change in each run of the quantum circuit), and dashed lines represent parameters that are statically fed into the quantum circuit (remaining constant across circuit runs). Grey, thick lines symbolise logical flow.

This requires $n-1$ joins (and actions) to build the join tree for a query with $n$ relations. Clipping, shifting and normalising the ratio reduces the chances of steeper gradients during training, which is a known cause of suboptimal training [86].

2) Quantum ReJoin: For ReJoin, a VQC can be employed as the actor-, as well as critic-part of PPO, or both. In both cases, the VQC encodes the state $S_{t}$. Policy or advantage estimations are obtained using the approach of Sec. III-C2.

As the number of inputs that a QPU can process is restricted by the hardware capabilities of QPUs, it is advantageous to minimise this number. As described in Sec. IV-C1, the state representation of the classical baseline suggests a state space with $a+2 r^{2}$ features for a database with $r$ relations ${ }^{3}$ and $a$ attributes. For the JOB, which encompasses 208 attributes across 39 different aliases throughout the JOB query set, there are 3250 input elements for one state.[^2]

a) Reducing the Input Size: To reduce the observation space, we specify a maximum number of relations $n$ that can be joined. As for the baseline, we employ a join graph and a tree structure representation, which are defined analogous to the baseline over the $n$ relations present in a given query. This leads to $n^{2}$ elements in both, the join graph and the sub-tree structure representation. To specify, which tables are referenced in a query, the tables in the database are enumerated and assigned with an index $I: \mathcal{T} \rightarrow[0, r-1]$, where $\mathcal{T}$ is the set of all tables and $r$ is the number of tables in the database. The indices $i \in \bigcup_{\mathcal{T}_{q} \in Q} I\left(\mathcal{T}_{q}\right)$ for a query $Q$ are added to the input components. To represent the information, which is given through the selection predicates, we obtain the selectivity (i.e., the fraction of tuples present in a result when filtering for the corresponding selection predicates of a specific table) for every table in a query and add these to the input components.

The reduced state representation leads to $2\left(n^{2}+n\right)$ input elements. For $n=17$ as maximum size in the JOB, this results in 612 elements, over $80 \%$ less than in the baseline.

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-07.jpg?height=595&width=1779&top_left_y=169&top_left_x=151)

Fig. 3. Processing sequence to iteratively determine join orders. Once the query has been parsed and encoded, subsequent invocations of the variational quantum circuit as illustrated in Fig. 2, determine more and more joins, until a complete order has been found.

b) Circuit dimensions: One advantage of quantum algorithms involving VQCs is that they allow for a certain degree of controllability of the circuit depth and number of qubits, which is especially desirable for NISQ devices [24]. Utilising the incremental data-uploading [82] and the DRU [75] approaches, we can choose the structure of the quantum circuit. We opted to divide the $2\left(n^{2}+n\right)$ input features in $n$ equally sized parts $p_{l}$. Each feature $f_{i} \in p_{l}$ is then scaled to a range of $[0, \pi]$ and used as $\mathrm{s}$ rotation angle for a $\hat{R}_{x}$ gate acting on qubit $i$ in the layer $l$. The input parts are interleaved with parameterised gates $\hat{R}_{y}$ and $\hat{R}_{z}$ that act on each qubit and introduce trainable parameters, and a circular sequence of $\mathrm{C}-\hat{Z}$ gates between two adjacent qubits, which create entanglement. Fig. 4 visualises this gate sequence for one encoding and one variational layer. This layer structure is chosen as it is seen as highly expressive throughout the literature [32], [87]. We considered two types of circuits: In the first, we apply DRU and repeat the encoding pattern several times, which can increase quantum expressivity [75]. In the second, we omit the input encoding part after each input feature is present in the circuit once, that is, we do not apply DRU, resulting in a flatter circuit. Both variants are evaluated empirically in Sec. V-B.

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-07.jpg?height=298&width=753&top_left_y=1862&top_left_x=228)

Fig. 4. Details of quantum state manipulation: Parametrised rotations around the $x$ axis $\left(\hat{R}_{x}(\theta)\right)$ encode information. The variational part comprises parametrised rotations around the $y$ and $z$ axes, implemented by $\hat{R}_{y}(\gamma)$ and $\hat{R}_{z}(\delta)$, followed by a cyclic sequence of $\mathrm{C}-\hat{Z}$ gates that create entanglement.

## V. EXPERIMENTS

We commence with the experimental setup (fully reproducible with our reproduction package), followed by the train- ing results for quantum-based versions of ReJoin in Sec. V-B.

## A. Experimental Setup

1) Training and Test Data: Following the approaches presented in Sec. II that evaluate their methods using various industrial benchmark datasets [88]-[90], for classical ReJoin, we used the 113 queries from the join order benchmark (JOB) [88]. As we lack access to a sufficiently large quantum machine to process data for all queries in the JOB, we concentrate on training with four relations per query. Since the JOB only provides three queries with four relations, we generate new queries based on subplans to enlarge the dataset, following Krishnan et al. [16]. However, instead of obtaining subplans from the traditional optimiser, we rely on a single ReJoin training run, generating over 12000 subqueries, from which we randomly select 497 that join four relations, and combine them with three JOB queries. To the resulting dataset of size 500, we apply a ten-fold cross-validation scheme [91], whereby the dataset is split into ten distinct parts. Each part is excluded from the training set once to be utilised for testing, leading to ten different train-test-splits.
2) Python Libraries: Since the original source code for ReJoin is not available, and other implementations for solving the JO problem by the means of RL [92]-[94] utilise different RL methods [95], and a different encoding for states and actions [93], [94] we modified and fine-tuned a third-party replication [96] based on the descriptions in Ref. [14] in collaboration with one of the original authors using the Python machine learning library Tensorflow [97] for the machine learning specific parts. For the quantum specific parts of our experiments, we additionally utilised the quantum frameworks Tensorflow Quantum [98] to simulate ideal quantum systems and Qiskit [99] to simulate noisy systems. Given the lack of capable quantum machines, we rely on simulations. The implementation can be found in our reproduction package.
3) Classical Baseline Replication: We were able to successfully replicate ReJoin, despite some minor deviations from the findings in Ref. [14], which could possibly attained to
differing hyperparameters or settings that were not specified in the original study. To further enhance the outcomes of our replication, and to allow for the reduced encoding described in Par. IV-C2a, we combined methods from other RL approaches for JO [16], [19] to improve the learning convergence in cost training. For more information on the classical replication and the baseline modification, the reader is referred to the supplementary material in the reproduction package.
4) PPO Models: We consider the following configurations:

a) Classical Model: As baseline, we use a classical NN with two hidden layers (128 units each) for actor and critic.

b) Quantum Model-Single-Step [46]: This model uses one qubit per relation in the query, resulting in 4 qubits for our dataset. For each relation in the query, the ID of the relation is encoded with an $\hat{R}_{x}$ gate and the combined selectivity of all filters on the relation is encoded with a $\hat{R}_{y}$ gate. As there are at most 15 possible join orders for 4 relations, $2^{4}$ quantum states are enough to have a state for each join order.

c) Quantum Models-Multi-Step: We consider three configurations: (a) $Q$-Critic, where a VQC is employed as the critic part of PPO, and a classical NN with the same dimensions as for the classical model serves as the actor; (b) $Q$ Actor with a VQC as actor in PPO, and classical critic; (c) Fully Quantum with VQCs for actor and critic. All quantum models use classical post-processing layer (see Sec. III-C2).

5) Data Re-Uploading (DRU) Setup: For each quantum model, we evaluate setups with and without DRU. The version utilising DRU employs $2-5$ repetitions of the gates necessary to encode all input features once. With four relations this results in $8,12,16$ and 20 variational layers for the multi-step QRL approach. To ensure a fair comparison with the singlestep QML approach, we repeat the input features, consisting of indices and selectivities, every four variational layers for the configurations with single-step QML and DRU. This results in the same number of input repetitions and variational layers as for the multi-step QRL approach. For the second configurations without DRU, we use the same number of variational layers and introduce an additional experiment with four layers to encode every input feature once without extra variational layers for multi-step QRL. Analogously, the configurations for single-step QML and without DRU encode the input features once followed by the respective number of variational layers.
6) Training and Evaluation: While incorporating noise during training, whether through direct execution on real QPUs or via noisy simulations utilising snapshots of actual devices, provides the most accurate assessment of our approach's performance on present or near-term quantum hardware, the computational demands of noisy simulation, particularly for large input sizes during optimisation, are substantial. Given these constraints, a complete training iteration exceeds the scope of this study. Nonetheless, to quantify the adverse effects of noise, we assess models trained in an ideal simulation in a noisy environment using the same test sets from the tenfold cross-validation. Specifically, we introduce depolarising errors [78], a prevalent error type in noisy simulations, with a predetermined probability applied to each gate within the models utilising a quantum actor (i.e., $Q$-Actor and Fully Quantum). For this probability, we select values ranging from $1 \%$ to $5 \%$, representing upper bounds of gate errors, to which current QPUs are prone [25]. The findings from our noisy evaluation are detailed in V-B2.

## B. Experimental Results

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-08.jpg?height=414&width=881&top_left_y=514&top_left_x=1077)

Fig. 5. Relative cost median during training. For the methods involving a quantum part the models with DRU and 20 variational layers are depicted.

1) Training Results from Ideal Simulations: As shown in Fig. 5, the classical model can achieve the median for optimal join orders after sampling roughly 8000 queries (episodes), while Q-Critic delivers comparable results. Since the singlestep approach surpasses conventional JO heuristics [43], [46] when trained on query execution times (i.e., true cost), it can be regarded as quantum baseline. We either outperform or match it in all three QRL variants. Specifically, the Q-Critic configuration can achieve up to $17 \%$ lower median costs than single-step QML. This implies that although the configurations that employ a VQC as an actor, as well as the single-step QML method, do not achieve an optimal cost median during training, the QRL approaches are competitive with established classical heuristics, assuming that careful hyperparameter tuning and incorporating true costs leads to better join orders. Since our focus is on the specific implications for quantum computing, and as training on a cost model may not necessarily translate to actual query execution times, we consider costs as performance indicator, following Refs [14], [16], [19].

Our results suggest that as the classical component of computation increases, the quality of results improves. This finding appears to contradict claims for quantum advantage in QML literature [37], [38], [40]. However, it aligns with a recent observation by Bowles et al. [100] who conducted benchmarks across various QML configurations and noted that models with a substantial portion of classical parameters often outperform those with a higher quantum component. Understanding the dynamic between classical and quantum methods remains an important future challenge.

As illustrated in Fig. 6, the quantity of variational layers impacts configurations with a higher proportion of VQC parameters (Q-Actor, Single-Step QML and Fully Quantum QRL), especially with DRU. We observe, consistent with findings in the literature [32], [34], [42], [46] that more layers lead to lower costs. In all other instances, optimal training

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-09.jpg?height=561&width=897&top_left_y=188&top_left_x=167)

Fig. 6. Relative cost median during training.

convergence is attainable with fewer variational layers, which translates to fewer parameters and shallower circuits.

2) Evaluation Results from Noisy Simulations: The configurations utilising a quantum actor (i.e., $Q$-Actor and Fully Quantum) demonstrate the capability to achieve nearly optimal results conducted on ideally simulated QPUs. However, when incorporating gate errors, the performance of quantum models tends to deteriorate. Fig. 7 shows that the median relative cost increases almost linearly across all configurations, with steeper increases observed for deeper circuits-those with more layers and DRU-, which inherently present more opportunities for errors. While this observation is sobering, it aligns with the expectation that models trained in a ideal environment may struggle when confronted with noise. Other studies [32], [101] suggest that incorporating noise during training, coupled with hyperparameter tuning tailored to such noise models, can yield successful outcomes even in the presence of noise. The exploration of noise's impact during training on the JO problem could be deferred to future investigations. However, as shown in Fig. 8, a comprehensive examination of results reveals that significant outliers persist, even in ideal and classical scenarios, indicating that while median performance appears reasonable, pronounced instabilities persist within the (Q)RL approach to the JO problem, necessitating further theoretical and empirical investigation of the methods itself.

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-09.jpg?height=458&width=894&top_left_y=1945&top_left_x=152)

Fig. 7. Relative cost median after training with different noise probabilities.

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-09.jpg?height=607&width=892&top_left_y=190&top_left_x=1077)

Fig. 8. Relative costs after training with different noise probabilities.

## VI. EVALUATION

While the quantum models may not outperform classical models in terms of cost efficiency post-training, other factors are pertinent to assess the effectiveness of QML methods for JO. This section discusses the influence of the circuit dimensionality on overall trainability, examining the number of parameters and scalability of our approaches compared to alternative quantum approaches for the JO problem.

## A. Parameter Efficiency

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-09.jpg?height=468&width=873&top_left_y=1376&top_left_x=1081)

Fig. 9. Number of parameters for classical and quantum methods. The pure quantum case requires substantially less parameters than the classical baseline, which reduces optimisation complexity. Partial quantum variants (Q-actor, Qcritic) are less parsimonious, yet retain advantages against the baseline. The inset shows the optimisation time of the Adam [102] optimiser, to apply gradients to the parameters. Lines show the median of 1000 measurements for each configuration; shaded areas depict first and third quartiles. Note that gradient calculation is excluded from our measurements.

Fig. 9 shows the total number of parameters (variational and classical) dependent on the number of relations in a query for the different methods. We considered the VQC structure with the dimensions described in Par. IV-C2b, including the classical post-processing layer and a fully-connected NN with two hidden layers and a constant hidden dimension of $128^{4}$.[^3]

While the classical baseline achieves lower and more stable costs, the QRL variants require fewer parameters. Considering the Q-Critic configuration, which achieves costs comparable to the baseline, we found that $47 \%$ less parameters suffice for four relations using 20 variational layers, that is, five DRU repetitions, and about $38 \%$ less parameters for 30 relations.

The corresponding run-times for the Adam optimiser [102] are shown in the inset of Fig. 9. We did not consider the time taken to calculate the gradients in our optimisation time measurements, as (1) gradient calculation or estimation methods for VQCs are still an ongoing area of research [79], [104], [105] and (2) our experiments are conducted on simulators instead of real QPUs, so the execution times may differ significantly. The parameter-shift rule [79], commonly used with VQCs, is computationally and necessitates two circuit executions per shot and parameter. Optimised techniques for gradient calculations have appeared [106]-[109], similar to classical ML over the past decades [110]. Yet, a comprehensive evaluation is beyond the scope of this paper. Based on our measurements, it is possible to achieve up to $12 \%$ improvement in median optimisation time for the Q-Critic configuration with one DRU repetition, in comparison to the classical model per optimisation step for 30 relations. As ML methods update parameters over multiple thousand iterations, this could significantly impact overall training time.

## B. Scalability of Quantum Approaches for Join Ordering

![](https://cdn.mathpix.com/cropped/2024_06_04_a8850389033ed495fb70g-10.jpg?height=466&width=892&top_left_y=1366&top_left_x=172)

Fig. 10. Number of qubits and circuit depth required to encode the JO problem for different quantum optimisation strategies.

As outlined in Sec. II, other quantum-based techniques address the JO problem. The number of qubits necessary to encode up to 30 relations for each of these strategies is depicted in Fig. 10. Refs. [67]-[70] aim to solve a specific class of problems, namely quadratic unconstrained binary optimisation (QUBO) problems, where the number of qubits required depends on the QUBO formulation. In contrast, QML approaches provide greater flexibility in the utilisation of qubits and circuit depth. As shown in the figure, both proposed QML approaches, single-step QML and multi-step QRL, are more efficiently in terms of qubit numbers, compared to the QUBO approaches. Furthermore, circuit depth is a widely accepted quantum runtime proxy, for which we provide bounds ${ }^{5}$ in Fig. 10. QRL generally requires only low circuit depth, comparable to the QUBO approach for bushy joins presented in Ref. [70]. However, not unlike with classical machine learning [111], while substantial progress with understanding capabilities of VQCs has been made [112], the learning dynamics based on the circuit dimension and theoretical underpinnings are not yet fully understood [44] and require further empirical and theoretical evaluation.

## VII. DISCUSSION AND OUTLOOK

We introduced a quantum reinforcement learning based approach to solve the long-standing, seminal join order problem, and replicated a classical reinforcement learning approach as suitable baseline to compare against the state of the art. In a systematic and comprehensive evaluation based on numerical simulation of quantum systems, we found that our approach at least matches classical performance in terms of result quality, which is not universally observed throughout the literature [45] for quantum algorithms. Apart from significantly reducing the input feature space of the classical baseline, we could show that substantially fewer trainable parameters are required, which is likely rooted in enhanced quantum expressivity. We believe the resulting reduction in classical optimisation efforts particularly benefits two scenarios: (a) Frequently changing data characteristics that necessitate continuous re-computation of join orders, and (b) low response latency requirements. Both appear in important commercial settings like stream data processing and high-frequency operation [113].

We also showed that our approach improves upon the scalability of existing quantum-RL solutions by nearly ten orders of magnitude in terms of qubit count. Given that this is the most scarce resource in current and future QPUs, we believe this is an important step towards practical utility.

Current NISQ capabilities prevents us from enjoying practical advantages right now. The limitations might, however, be circumvented even prior to the arrival of fully error-corrected hardware that is capable of delivering the behaviour predicted in our simulations by using custom-designed hardware. Additionally, it has recently been observed that the JO problem on quantum-inspired hardware can outperform established approaches [68]. Similar observations could generalise to other types of hardware, potentially applicable to the domain of variational algorithms or machine learning that our approach is based on. Finally, progress in the foundational understanding of QML could improve performance using more sophisticated quantum baseline methods, or data encoding strategies.

Acknowledgements MF, TW, SG and WM were supported by the German Federal Ministry of Education and Research (BMBF), funding program "Quantum Technologies-from Basic Research to Market", grants \#13N15647 and \#13NI6092 (MF and WM), and \#13N16090 (TW and SG). WM acknowledges support by the HighTech Agenda Bavaria.[^4]

## REFERENCES

[1] M. Steinbrunn, G. Moerkotte, and A. Kemper, "Heuristic and randomized optimization for the join ordering problem," The VLDB Journal The Int. Journal on Very Large Data Bases, vol. 6, no. 3, 1997.

[2] T. Neumann, "Query simplification: Graceful degradation for joinorder optimization," in Proc. of the 2009 ACM SIGMOD Int. Conf. on Management of data, 2009

[3] T. Neumann and B. Radke, "Adaptive optimization of very large join queries," in Proc. of the 2018 Int. Conf. on Management of Data, 2018 .

[4] I. Trummer and C. Koch, "Solving the join ordering problem via mixed integer linear programming," in Proc. of the 2017 ACM Int. Conf. on Management of Data, 2017.

[5] W.-S. Han and J. Lee, "Dependency-aware reordering for parallelizing query optimization in multi-core cpus," in Proc. of the 2009 ACM SIGMOD Int. Conf. on Management of data, 2009

[6] I. Kolchinsky and A. Schuster, Join query optimization techniques for complex event processing applications, 2018.

[7] F. A. Gonçalves, F. G. Guimarães, and M. J. Souza, "Query join ordering optimization with evolutionary multi-agent systems," Expert Systems with Applications, vol. 41, no. 15, 2014.

[8] V. Leis et al., "Query optimization through the looking glass, and what we found running the join order benchmark," The VLDB Journal, vol. 27, 2018.

[9] G. Moerkotte. "Building query compilers." (2023), [Online]. Available: pi3.informatik.uni-mannheim.de/ moer/querycompiler.pdf.

[10] S. Cluet and G. Moerkotte, "On the complexity of generating optimal left-deep processing trees with cross products," in Proc. of the 5th Int. Conf. on Database Theory, 1995.

[11] A. Swami, "Optimization of large join queries: Combining heuristics and combinatorial techniques," in Proc. ACM SIGMOD, 1989.

[12] R. Krishnamurthy, H. Boral, and C. Zaniolo, "Optimization of nonrecursive queries," in Proc. of the 12th Int. Conf. on Very Large Data Bases, 1986.

[13] P. G. Selinger et al., "Access path selection in a relational database management system," in Proc. of the 1979 ACM SIGMOD Int. Conf. on Management of Data, 1979.

[14] R. Marcus and O. Papaemmanouil, "Deep reinforcement learning for join order enumeration," in Proc. of the 1st Int. Workshop on Exploiting Artificial Intelligence Techniques for Data Management, 2018

[15] R. Marcus et al., "Neo: A learned query optimizer," Proc. VLDB Endow., vol. 12, no. 11, 2019.

[16] S. Krishnan et al., "Learning to optimize join queries with deep reinforcement learning," 2018.

[17] I. Trummer et al., "Skinnerdb: Regret-bounded query evaluation via reinforcement learning," ACM Trans. Database Syst., vol. 46, no. 3, 2021.

[18] J. Chen et al., "Efficient join order selection learning with graphbased representation," in Proc. of the 28th ACM SIGKDD Conf. on Knowledge Discovery and Data Mining, 2022.

[19] X. Yu et al., "Reinforcement learning with tree-lstm for join order selection," in 2020 IEEE 36th Int. Conf. on Data Engineering (ICDE), 2020

[20] J. Wang et al., Adopt: Adaptively optimizing attribute orders for worst-case optimal join algorithms via reinforcement learning, 2023.

[21] L. Ji et al., "Query join order optimization method based on dynamic double deep q-network," Electronics, vol. 12, no. 6, 2023.

[22] P. W. Shor, "Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer," SIAM Review, vol. 41, no. 2, 1999 .

[23] L. K. Grover, "A fast quantum mechanical algorithm for database search," in Proc. of the twenty-eighth annual ACM symposium on Theory of computing, 1996

[24] J. Preskill, "Quantum Computing in the NISQ era and beyond," Quantum, vol. 2, 2018

[25] F. Greiwe, T. Krüger, and W. Mauerer, "Effects of imperfections on quantum algorithms: A software engineering perspective," en, in 2023 IEEE Int. Conf. on Quantum Software (QSW), 2023.

[26] N. Pirnay et al., An in-principle super-polynomial quantum advantage for approximating combinatorial optimization problems, 2023
[27] E. Farhi, J. Goldstone, and S. Gutmann, A quantum approximate optimization algorithm, 2014.

[28] J. R. McClean et al., "The theory of variational hybrid quantumclassical algorithms," New Journal of Physics, vol. 18, no. 2, 2016.

[29] M. Cerezo et al., "Variational quantum algorithms," Nature Reviews Physics, vol. 3, no. 9, 2021.

[30] T. Hoefler, T. Häner, and M. Troyer, "Disentangling hype from practicality: On realistically achieving quantum advantage," Commun. ACM, vol. 66, no. 5, 2023.

[31] S. Y.-C. Chen et al., "Variational quantum circuits for deep reinforcement learning," IEEE Access, vol. 8, 2020.

[32] A. Skolik, S. Jerbi, and V. Dunjko, "Quantum agents in the Gym: A variational quantum algorithm for deep Q-learning," Quantum, vol. 6, 2022 .

[33] M. Franz et al., "Uncovering instabilities in variational-quantum deep q-networks," Journal of the Franklin Institute, 2022.

[34] R. Dilip et al., "Data compression for quantum machine learning," Phys. Rev. Res., vol. 4, 42022.

[35] H.-Y. Huang et al., "Quantum advantage in learning from experiments," Science, vol. 376, no. 6598, 2022.

[36] Y. Du et al., "Expressive power of parametrized quantum circuits," Phys. Rev. Res., vol. 2, 32020.

[37] H.-Y. Huang et al., "Power of data in quantum machine learning," Nature Communications, vol. 12, no. 1, 2021.

[38] Y. Liu, S. Arunachalam, and K. Temme, "A rigorous and robust quantum speed-up in supervised machine learning," Nature Physics, vol. 17, no. 9, 2021.

[39] R. Sweke et al., "On the Quantum versus Classical Learnability of Discrete Distributions," Quantum, vol. 5, 2021.

[40] V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," Nature, vol. 567, no. 7747, 2019.

[41] O. Lockwood and M. Si, "Reinforcement learning with quantum variational circuit," Proc. of the AAAI Conf. on Artificial Intelligence and Interactive Digital Entertainment, vol. 16, no. 1, 2020.

[42] S. Jerbi et al., Parametrized quantum policies for reinforcement learning, 2021

[43] U. Çalikyilmaz et al., "Opportunities for quantum acceleration of databases: Optimization of queries and transaction schedules," Proc. VLDB Endow., vol. 16, no. 9, 2023.

[44] M. Schuld and N. Killoran, "Is quantum advantage the right goal for quantum machine learning?" PRX Quantum, 2022.

[45] N. Meyer et al., A survey on quantum reinforcement learning, 2022.

[46] T. Winker et al., "Quantum machine learning for join order optimization using variational quantum circuits," in Proc. of the Int. Workshop on Big Data in Emergent Distributed Environments, 2023.

[47] W. Mauerer and S. Scherzinger, "1-2-3 reproducibility for quantum software experiments," in IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER), 2022.

[48] A. Meister and G. Saake, "GPU-accelerated dynamic programming for join-order optimization," 2020.

[49] G. Moerkotte and T. Neumann, "Analysis of two existing and one new dynamic programming algorithm for the generation of optimal bushy join trees without cross products," in Proc. VLDB Endow., ser. VLDB ’06, 2006.

[50] B. Vance and D. Maier, "Rapid bushy join-order optimization with cartesian products," in Proc. of the 1996 ACM SIGMOD Int. Conf. on Management of Data, ser. SIGMOD '96, 1996.

[51] G. Moerkotte and T. Neumann, "Dynamic programming strikes back," in Proc. of the 2008 ACM SIGMOD Int. Conf. on Management of Data, ser. SIGMOD '08, Association for Computing Machinery, 2008.

[52] J.-T. Horng, C.-Y. Kao, and B.-J. Liu, "A genetic algorithm for database query optimization," in Proceedings of the 1st IEEE Conf. on Evolutionary Computation. IEEE World Congress on Computational Intelligence, 1994.

[53] N. Bruno, C. Galindo-Legaria, and M. Joshi, "Polynomial heuristics for query optimization," in 2010 IEEE 26th Int. Conf. on Data Engineering (ICDE 2010), 2010.

[54] Y. E. Ioannidis and Y. Kang, "Randomized algorithms for optimizing large join queries," SIGMOD Rec., vol. 19, no. 2, 1990.

[55] I. Trummer and C. Koch, "Parallelizing query optimization on shared-nothing architectures," Proc. VLDB Endow., vol. 9, no. 9, 2016.

[56] Y. Han et al., Cardinality estimation in DBMS: A comprehensive benchmark evaluation, 2021.

[57] K. Kim et al., "Learned cardinality estimation: An in-depth study," in Proc. of the 2022 Int. Conf. on Management of Data, 2022.

[58] R. Hasan and F. Gandon, "A machine learning approach to sparql query performance prediction," in IEEE/WIC/ACM Int. Joint Conf.s on Web Intelligence (WI) and Intelligent Agent Technologies (IAT), IEEE, vol. $1,2014$.

[59] M. Akdere et al., "Learning-based query performance modeling and prediction," in 2012 IEEE 28th Int. Conf. on Data Engineering, IEEE, 2012

[60] I. Trummer and C. Koch, "Multiple query optimization on the dwave 2x adiabatic quantum computer," Proc. VLDB Endow., vol. 9, no. $9,2016$.

[61] T. Winker et al., "Quantum machine learning: Foundation, new techniques, and opportunities for database research," in Companion of the 2023 Int. Conf. on Management of Data, 2023.

[62] S. Groppe and J. Groppe, "Optimizing transaction schedules on universal quantum computers via code generation for grover's search algorithm," in 25th Int. Database Engineering \& Applications Symposium, 2021.

[63] T. Bittner and S. Groppe, "Hardware accelerating the optimization of transaction schedules via quantum annealing by avoiding blocking," Open Journal of Cloud Computing (OJCC), vol. 7, no. 1, 2020.

[64] T. Bittner and S. Groppe, "Avoiding blocking by scheduling transactions using quantum annealing," in Proc. of the 24th Symposium on Int. Database Engineering \& Applications, 2020.

[65] K. Fritsch and S. Scherzinger, "Solving hard variants of database schema matching on quantum computers," Proc. VLDB Endow., vol. 16, no. 12, 2023.

[66] L. Gruenwald et al., "Index tuning with machine learning on quantum computers for large-scale database applications," in Proc. of QDSM@VLDB23, 2023.

[67] M. Schönberger, S. Scherzinger, and W. Mauerer, "Ready to leap (by co-design)? Join order optimisation on quantum hardware," 1 , vol. 1, 2023 .

[68] M. Schönberger, I. Trummer, and W. Mauerer, "Quantum-inspired digital annealing for join ordering," 3 , vol. 17, 2023.

[69] N. Nayak et al., "Constructing optimal bushy join trees by solving qubo problems on quantum hardware and simulators," in Proc. of BiDEDE@SIGMOD23, 2023.

[70] M. Schönberger, I. Trummer, and W. Mauerer, "Quantum optimisation of general join trees," in Proc. of QDSM@VLDB23, 2023.

[71] J. M. Smith and P. Y.-T. Chang, "Optimizing the performance of a relational algebra database interface," Communications of the ACM, vol. 18, no. 10, 1975 .

[72] R. K. Kurella, "Systematic literature review: Cost estimation in relational databases," M.S. thesis, University of Magdeburg, 2018.

[73] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. 2018.

[74] J. Schulman et al., Proximal policy optimization algorithms, 2017.

[75] A. Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier," Quantum, vol. 4, 2020.

[76] K. Hornik, M. Stinchcombe, and H. White, "Multilayer feedforward networks are universal approximators," Neural Networks, vol. 2, no. 5,1989 .

[77] K. Mitarai et al., "Quantum circuit learning," Phys. Rev. A, vol. 98, 32018 .

[78] M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information - 10th Anniversary Edition. 2010.

[79] M. Schuld et al., "Evaluating analytic gradients on quantum hardware," Phys. Rev. A, vol. 99, 32019.

[80] M. Weigold et al., "Data encoding patterns for quantum computing," in Proc. of the 27th Conf. on Pattern Languages of Programs, 2022.

[81] M. Weigold et al., "Encoding patterns for quantum algorithms," IET Quantum Communication, vol. 2, no. 4, 2021.

[82] M. Periyasamy et al., "Incremental data-uploading for full-quantum classification," in 2022 IEEE Int. Conf. on Quantum Computing and Engineering (QCE), 2022.

[83] M. Schuld, R. Sweke, and J. J. Meyer, "Effect of data encoding on the expressive power of variational quantum-machine-learning models," Phys. Rev. A, vol. 103, 32021.
[84] N. Meyer et al., "Quantum policy gradient algorithm with optimized action decoding," in Proc. of the 40th Int. Conf. on Machine Learning, A. Krause et al., Eds., vol. 202, 2023.

[85] O. Lockwood and M. Si, "Playing atari with hybrid quantumclassical reinforcement learning," in NeurIPS 2020 Workshop on Pre-registration in Machine Learning, L. Bertinetto et al., Eds., vol. 148, 2021.

[86] A. Laud and G. DeJong, "The influence of reward on the speed of reinforcement learning: An analysis of shaping," in Proc. of the 20th Int. Conf. on Machine Learning (ICML-03), 2003.

[87] A. Kandala et al., "Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets," Nature, vol. 549, no. 7671, 2017.

[88] V. Leis et al., "How good are query optimizers, really?" Proc. VLDB Endow., vol. 9, no. 3, 2015.

[89] M. Poess et al., "TPC-DS, taking decision support benchmarking to the next level," in Proc. of the 2002 ACM SIGMOD Int. Conf. on Management of Data, 2002.

[90] M. Poess and R. Nambiar, "Tpc benchmark h standard specification," 2010 .

[91] T. Fushiki, "Estimation of prediction error by using k-fold crossvalidation," Statistics and Computing, vol. 21, 2011.

[92] X. Yu et al. "Github repository: AI4DBCode." Commit: a8989bfa. (2022), [Online]. Available: https : / / github .com / TsinghuaDatabaseGroup/AI4DBCode.

[93] Z. Yang et al., "Balsa: Learning a query optimizer without expert demonstrations," in Proc. of the 2022 Int. Conf. on Management of Data, 2022.

[94] R. Marcus et al., "Bao: Making learned query optimization practical," in Proc. of the 2021 Int. Conf. on Management of Data, 2021.

[95] V. Mnih et al., Playing atari with deep reinforcement learning, 2013.

[96] G. Xintong and A. Mandamadiotis. "Github repository: Rejoin." Commit: 02365ab0. (2021), [Online]. Available: https://github.com/ GUOXINTONG/rejoin.

[97] Martín Abadi, Ashish Agarwal, Paul Barham, et al., TensorFlow: Large-scale machine learning on heterogeneous systems.

[98] M. Broughton et al., Tensorflow quantum: A software framework for quantum machine learning, 2021.

[99] Qiskit contributors, Qiskit: An open-source framework for quantum computing, 2023.

[100] J. Bowles, S. Ahmed, and M. Schuld, Better than classical? the subtle art of benchmarking quantum machine learning models, 2024.

[101] K. Borras et al., "Impact of quantum noise on the training of quantum generative adversarial networks," Journal of Physics: Conf. Series, vol. 2438, no. 1, 2023.

[102] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, 2017.

[103] K. G. Sheela and S. N. Deepa, "Review on methods to fix number of hidden neurons in neural networks," Mathematical Problems in Engineering, vol. 2013, 2013.

[104] D. Wierichs et al., "General parameter-shift rules for quantum gradients," Quantum, vol. 6, 2022.

[105] A. Gilyén, S. Arunachalam, and N. Wiebe, "Optimizing quantum optimization algorithms via faster quantum gradient computation," in Proc. of the 30th Annual ACM-SIAM Symposium on Discrete Algorithms, SIAM, 2019.

[106] M. Periyasamy et al., Guided-spsa: Simultaneous perturbation stochastic approximation assisted by the parameter shift rule, 2024.

[107] J. Stokes et al., "Quantum natural gradient," Quantum, vol. 4, 2020.

[108] L. Bittel, J. Watty, and M. Kliesch, Fast gradient estimation for variational quantum algorithms, 2022.

[109] J. Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation," IEEE Transactions on Automatic Control, vol. 37, no. 3, 1992.

[110] I. H. Sarker, "Deep learning: A comprehensive overview on techniques, taxonomy, applications and research directions," SN Computer Science, vol. 2, no. 6, 2021.

[111] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning (Adaptive computation and machine learning). 2016.

[112] J. Landman et al., Classically approximating variational quantum machine learning with random fourier features, 2022.

[113] J. Dean and L. A. Barroso, "The tail at scale," Commun. ACM, vol. 56, no. 2, 2013.


[^0]:    ${ }^{1}$ We follow ACM terminology on Artifact Review and Badging: A replication describes measurements obtained by a different team using a different experimental setup. The term re-implementation is also common in the literature, with identical meaning.

[^1]:    ${ }^{2}$ We provide a comparison to a method, which awards zero to intermediate steps in the supplementary material in our reproduction package

[^2]:    ${ }^{3}$ We assume $r$ is the number of different aliases occurring in the dataset and $a$ is the number of attributes corresponding to these aliases. One author of Ref. [14] confirmed that multi-aliases were handled as an additional tables.

[^3]:    ${ }^{4}$ Typically, the number of hidden units grows with input space [103], so the number of parameters for the classical model gives a lower bound.

[^4]:    ${ }^{5}$ Bounds are based on circuit depth for one data uploading block (QML approaches) and lower bounds on the circuit depth for the respective QAOA [27] circuit with $p=1$ (QUBO-based approaches). We consider the maximum number of entangling gates/quadratic terms that act on two qubits, and gates required for initialisation and mixer Hamiltonian.

</end of paper 3>


