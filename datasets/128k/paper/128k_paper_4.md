<paper 0>
# TabFairGAN: Fair Tabular Data Generation with Generative Adversarial Networks 

$1^{\text {st }}$ Amirarsalan Rajabi<br>Department of Computer Science<br>University of Central Florida<br>Orlando, FL, US<br>amirarsalan@ knights.ucf.edu

$2^{\text {nd }}$ Ozlem Ozmen Garibay<br>Department of Industrial Engineering and Management Systems<br>University of Central Florida<br>Orlando, FL, US<br>ozlem@ucf.edu


#### Abstract

With the increasing reliance on automated decision making, the issue of algorithmic fairness has gained increasing importance. In this paper, we propose a Generative Adversarial Network for tabular data generation. The model includes two phases of training. In the first phase, the model is trained to accurately generate synthetic data similar to the reference dataset. In the second phase we modify the value function to add fairness constraint, and continue training the network to generate data that is both accurate and fair. We test our results in both cases of unconstrained, and constrained fair data generation. In the unconstrained case, i.e. when the model is only trained in the first phase and is only meant to generate accurate data following the same joint probability distribution of the real data, the results show that the model beats state-of-the-art GANs proposed in the literature to produce synthetic tabular data. Also, in the constrained case in which the first phase of training is followed by the second phase, we train the network and test it on four datasets studied in the fairness literature and compare our results with another state-of-the-art pre-processing method, and present the promising results that it achieves. Comparing to other studies utilizing GANs for fair data generation, our model is comparably more stable by using only one critic, and also by avoiding major problems of original GAN model, such as mode-dropping and non-convergence, by implementing a Wasserstein GAN.


Index Terms-Fairness in Artificial Intelligence, Generative Adversarial Networks, WGAN

## I. INTRODUCTION

Artificial intelligence has gained paramount importance in the contemporary human life. With an ever-growing body of research and increasing processing capacity of computers, machine learning systems are being adopted by many firms and institutions for decision-making. Various industries such as insurance companies, financial institutions, and healthcare providers rely on automated decision making by machine learning models, making fairness-aware learning crucial since many of these automated decisions could have major impacts on the lives of individuals.

There are numerous evidence suggesting that bias exists in AI systems. One well known example is Correctional Offender Management Profiling for Alternative Sanctions (COMPAS), which is a decision making system deployed by the US criminal justice system to assess the likelihood of a criminal defendant's recidivism (re-offending). It is shown that COMPAS is biased against African American defendants [1]. Another example is a Google's targeted advertising that was found to have shown the high paid jobs significantly more to males than females [2].

The existence of such bias and unfair classifications in AI systems has led the research community to pay attention to the problem of bias in AI. There are different approaches to improve fairness existing in the AI fairness literature. Let $D=\{X, S, Y\}$ be a labelled dataset, where $X \in \mathbb{R}^{n}$ are the unprotected attributes, $S$ is the protected attribute, and $Y$ is the decision. From a legal perspective, protected attribute is the attribute identified by law, based on which it is illegal to discriminate [3], e.g. gender or race. The proposed fairness enforcement methods in the literature could be categorised into three main classes of pre-process methods, in-process methods, and post-process methods.

Pre-process methods include modifying the training data before feeding the data into machine learning algorithm. For instance, in one study [4], four methods are presented to remove bias including suppression which is to remove attributes highly correlated with protected attributes $S$, massaging the dataset which is to change labels $(Y)$ of some objects in the dataset, and reweighing that involves assigning weights to different instances in the dataset. These are preliminary and simpler methods that results in more fair predictions, however entail higher fairness-utility cost. In other words fairness is achieved at the expense of accuracy. Another preprocessing method proposed in the literature is the work of Feldman et al. [5] in which a repairment mechanism is proposed to modify the unprotected attributes $(X)$ and achieve fairness with higher accuracy comparing to the aforementioned methods. This method will be discussed in more detail in Section V-B as the baseline method. In-process approaches involve modifying the learning algorithm to achieve fairness during training [3]. These methods mostly include modifying the objective functions or adding regularization terms to the cost function. For example, [6] proposes adding a regularization term to the objective function which penalize mutual information between the protected attributes and the classifier predictions. Finally, post-process mechanisms include modifying the final decisions of the classifiers. For instance, Hardt et al. [7] propose a method to modify the final classification scores in order to enhance equalized odds.

The emergence of unfairness in AI systems is mostly
attributed to: 1) direct bias existing in the historical datasets being used to train the algorithms, 2) bias caused by missing data, 3) bias caused by proxy attributes, where bias against the minority population is present in non-protected attributes, and 4) bias resulting from algorithm objective functions, where the aggregate accuracy of the whole population is sought and therefore the algorithm might disregard the minority group for the sake of majority [3]. Since historical datasets are a major source of discrimination in $\mathrm{AI}$, we focus on generating unbiased datasets to achieve fairness.

There is a rich and growing literature on generative models. The main idea behind a generative model is to capture the probabilistic distribution that could generate data similar to a reference dataset [8]. Broadly speaking, generative models could be divided into two main classes of models [8]: Energybased models such as Boltzmann Machines [9], and cost function-based models such as autoencoders and generative adversarial networks (GANs) [10]. GANs address some deficiencies in traditional generative models, and are shown to excel in various tasks comparing to other generative models such as in image generation [11] and video generation [12].

The original GAN consists of two networks, generator and discriminator [10]. The two networks play a minimax game. The generator takes a latent random variable $Z$ as input and generates a sample $G(Z)$, that is similar to the real data. The discriminator, on the other hand, is fed with both real and generated samples, and its task is to correctly classify the input sample as real or generated. Over time if the networks have enough capacity, they are trained together and ideally optimized to reach an equilibrium state in which the generator produces data from the exact targeted distribution and the discriminator gives the real and generated samples an equal probability of 0.5 . The work in [10] shows that training the discriminator to optimality is equal to minimizing JensenShannon divergence [13]. The work of Arjovsky et al. develops Wasserstein GANs, where a critic replaces the discriminator, and minimizing Earth-mover's distance is used instead of minimizing Jensen-Shannon divergence [14]. They show that WGAN could address some common training problems attributed to GANs, usch as requirement to maintain a careful balance during training as well as mode dropping [15].

In recent studies adversarial training has been used to remove discrimination. One such study, for example, by formulating the model as a minimax problem, proposes an adversarial learning framework that could learn representations of data that are discrimination-free and do not contain explicit information about the protected attribute [16]. Other adversarial objectives are proposed by the works of [17], [18] to achieve group fairness measures such as demographic parity and equality of odds. The application of generative adversarial networks for fairness in tabular datasets is not discussed enough in the literature, but has recently attracted attention of the research community. For instance, the work of Sattigeri et al. [19] proposes an approach to generate image datasets such that demographic fairness in the generated dataset is imposed. In their work $\mathrm{Xu}$ et al. [20] design a GAN that produces discrimination free tabular datasets. Their network includes one generator and two discriminators. The generator is adopted from [21] and produces fake pairs of data $(\hat{X}, \hat{Y})$ following the conditional distribution $P_{G}(X, Y \mid S)$. One discriminator's task is to ensure generator produces data with good accuracy, and the second discriminator ensures the generator produces fair data.

In this paper, we propose a Wasserstein GAN, TabFairGAN, that can produce high quality tabular data with the same joint distribution as the original tabular dataset. In Section II we discuss the fairness measure: demographic parity and discrimination score. In Section III, we introduce the model architecture, data transformation, value functions, and the training process of the model. In section IV, we compare the results of TabFairGAN with two other state-of-the-art GANs for tabular data generation, namely TGAN [22] and CTGAN [23]. In SectionV, we show how the model could be used for fair data generation and test the model on four real dataset. We compare the results of our model with the method developed by [5], which is another pre-process methods to enforce fairness. Finally in Section V-D, we explore the fairnessaccuracy trade-off. This work has two main contributions. We show that in the case of no constraints present (no fairness), the model is able to produce high quality synthetic data, competing with the state-of-the-art GANs designed for tabular data generation. Second contribution is producing high quality fair synthetic data, by adding a fairness constraint in the loss function of the generator. Comparing our model to previous application of GANs for fair tabular data generation, the model is more stable based on two merits: 1) the proposed model is a Wasserstein GAN which is shown to improve original GAN model in terms of some common GAN pitfalls, such as modedropping phenomena [15], and 2) the model only uses one critic instead of two [20] or three [24] discriminators.

## II. DISCRIMINATION SCORE

Among the most frequently practiced fairness metrics specified in legal notions and the literature is demographic parity or statistical parity/fairness. The goal of demographic fairness is to ensure that the overall proportion of members with respect to the protected group receiving a positive decision is identical. In a binary case, let $D=\{X, S, Y\}$ be a labelled dataset, where $X \in \mathbb{R}^{n}$ is the unprotected attributes, $S$ is the protected attribute, and $Y$ is the decision. In this paper, we consider the binary case, and for notational convenience we assume that the protected attribute $S$ takes two values, where $S=0$ represents the underprivileged minority class, and $S=1$ represents the privileged majority class. For instance, in a binary racial discrimination study the value 0 will be assigned to "AfricanAmerican", whereas 1 is assigned to "White". We also assign 1 to $Y$ for a successful decision (for instance an admission to a higher education institution), and assign 0 to $Y$ for an unsuccessful decision (rejection). Demographic fairness for the labeled dataset is defined as follows [7]:

$$
\begin{equation*}
P(y=1 \mid s=1)=P(y=1 \mid s=0) \tag{1}
\end{equation*}
$$

In this context, demographic parity is defined by the difference between the conditional probability and its marginal. We define the discrimination with respect to the protected attribute $S$ by discrimination score (DS) and calculate it by: $D S=P(y=1 \mid s=1)-P(y=1 \mid s=0)$. A similar measure could be obtained for a labeled dataset $D$ and a classifier $f:(X, S) \rightarrow Y$ where the discrimination score for the classifier $f$ with respect to protected attribute $S$ can be obtained by:

$$
\begin{equation*}
P(\hat{y}=1 \mid x, s=1)-P(\hat{y}=1 \mid x, s=0) \tag{2}
\end{equation*}
$$

## III. MODEL DESCRIPTION

## A. Tabular Dataset Representation and Transformation

A tabular dataset contains $N_{C}$ numerical columns $\left\{c_{1}, \ldots, c_{N_{C}}\right\}$ and $N_{D}$ categorical columns $\left\{d_{1}, \ldots, d_{N_{D}}\right\}$. In this model, categorical columns are transformed and represented by one-hot vectors. Representing numerical columns on the other hand is non-trivial due to certain properties of numerical columns. One such property is that numerical columns are often sampled from multi-modal distributions. Some models such as [21] use min-max normalization to normalize and transform numerical columns. The work of $\mathrm{Xu}$ et al. [23] proposes a more complex process, namely a mode-specific normalization using variational Gaussian mixture model (VGM) to estimate the number of modes and fit a Gaussian mixture model to each numerical column. In our model, each numerical column is transformed using a quantile transformation [25]:

$$
\begin{equation*}
c_{i}^{\prime}=\Phi^{-1}\left(F\left(c_{i}\right)\right) \tag{3}
\end{equation*}
$$

Where $c_{i}$ is the $i$ th numerical feature, $F$ is the CDF (cumulative distbituion function) of the feature $c_{i}$, and $\Phi$ is the CDF of a uniform distribution. After transforming numerical and discrete columns, the representation of each transformed row of the data is as follows:

$$
\begin{align*}
& \mathbf{r}=c_{1}^{\prime} \oplus \ldots \oplus c_{N_{C}}^{\prime} \oplus d_{1}^{\prime} \oplus \ldots \oplus d_{N_{D}}^{\prime}  \tag{4}\\
& l_{i}=\operatorname{dim}\left(d_{i}^{\prime}\right)  \tag{5}\\
& l_{w}=\operatorname{dim}(r) \tag{6}
\end{align*}
$$

where $c^{\prime}{ }_{i}$ represents the $i$ th numerical column, $d^{\prime}{ }_{i}$ denotes the one-hot encoded vector of the $i$ th categorical columns, and $\oplus$ is the symbol denoting concatenation of vectors. Also, $l_{i}$ shows the dimension of the $i$ th discrete column's one-hot encoding vector and $l_{w}$ shows the the dimension of $r$.

## B. Network Structure

While traditional GANs suffer from problems such as nonconvergence and mode-collapse, the work of [15] developed Wasserstein GANs which improve training of GANs to some extent, and replace the discriminator with a critic. The network designed in this model is a WGAN with gradient penalty [26]. The WGAN value function using the Kantorovich-Rubinstein duality [27] is as follows [26]:

$$
\begin{equation*}
\min _{G} \max _{C \in \mathcal{C}} \underset{\mathbf{x} \sim P_{\text {data }}(\mathbf{x})}{\mathbb{E}}[C(\mathbf{x})]-\underset{\mathbf{z} \sim P_{z}(\mathbf{z})}{\mathbb{E}}[C(G(z))] \tag{7}
\end{equation*}
$$

Where $\mathcal{C}$ is the set of 1-Lipschitz functions. The generator receives a latent variable $Z$ from a standard multivariate normal distribution and produces a sample data point which is then forwarded to the critic. Once the critic and the generator are trained together, eventually the generator would become like a deterministic transformation that produces data similar to the real data.

The generator consists of a fully-connected first layer with ReLu activation function. The second hidden layer of the generator network is then formed by concatenation of multiple vectors that could form data similar to transformed original data. For the numerical variables, a fully connected layer of $\mathrm{FC}_{l_{w} \rightarrow N_{C}}$, with a $\mathrm{ReLu}$ activation is implemented. For nodes that are supposed to produce discrete columns, multiple fully connected layer of $\mathrm{FC}_{l_{w} \rightarrow l_{i}}$, with Gumble softmax [28] activation is used in order to produce one-hot vectors $\left(d_{i}^{\prime}\right)$. The resulting nodes are then concatenated to produce data similar to the transformed original data (with the same dimension of $l_{w}$ ), which is then fed to the critic network. The structure of the critic network is simple and includes 2 fully connected layers with Leaky ReLu activation functions.

The generator network's architecture is formally described as:

$\left\{\begin{array}{l}h_{0}=Z \text { (latent vector) } \\ h_{1}=\operatorname{ReLu}\left(\mathrm{FC}_{l_{w} \rightarrow l_{w}}\left(h_{0}\right)\right) \\ h_{2}=\operatorname{ReLu}\left(\mathrm{FC}_{l_{w} \rightarrow N_{C}}\left(h_{1}\right)\right) \oplus \text { gumble }_{0.2}\left(\mathrm{FC}_{l_{w} \rightarrow l_{1}}\left(h_{1}\right)\right) \oplus \\ \text { gumble }_{0.2}\left(\mathrm{FC}_{l_{w} \rightarrow l_{2}}\left(h_{1}\right)\right) \oplus \ldots \oplus \text { gumble }_{0.2}\left(\mathrm{FC}_{l_{w} \rightarrow l_{N_{D}}}\left(h_{1}\right)\right)\end{array}\right.$

Where $F C_{a \rightarrow b}$ denotes a fully connected layer with input size $a$ and output size $b, \operatorname{ReLu}(x)$ shows applying a ReLu activation on $x$, and gumble ${ }_{\tau}(x)$ denotes applying Gumble softmax with parameter $\tau$ on a vector $x$, and $\oplus$ denotes concatenation of vectors.

The critic network's architecture is formally described as:

$\left\{\begin{array}{l}h_{0}=X \text { (output of the generator or transformed real data) } \\ h_{1}=\mathrm{LeakyReLu}_{0.01}\left(\mathrm{FC}_{l_{w} \rightarrow l_{w}}\left(h_{0}\right)\right) \\ h_{2}=\operatorname{LeakyReLu}_{0.01}\left(\mathrm{FC}_{l_{w} \rightarrow l_{w}}\left(h_{1}\right)\right)\end{array}\right.$

Where LeakyReLu ${ }_{\tau}(x)$ denotes applying Leaky ReLu activation function [29] with slope $\tau$ on $x$. Fig. 1] shows the architecture of the model.

## C. Training

In this section we introduce the loss functions for the critic network and generator network of the developed WGAN. The overall process of training the model includes two phases. Phase I of training only focuses on training the model such that the generator could generate data with a joint probability distribution similar to that of the real data. Phase II of training

![](https://cdn.mathpix.com/cropped/2024_06_04_a31a2c0cdbfc87b415ddg-4.jpg?height=433&width=889&top_left_y=171&top_left_x=152)

Fig. 1. Model architecture. The generator consists of an initial fully connected layer with ReLu activation function, and a second layer which uses ReLu for numerical attributes generation and gumble-softmax to form one-hot representations of categorical attributes. The final data is then produced by concatenating all attributes in the last layer of the generator. The critic consists of fully-connected layers with LeakyReLu activation function.

further trains the generator to produce samples which have a joint probability distribution similar to that of real data and is also fair, with respect to discrimination score (DS) defined in Section $\square$.

1) Phase I: Training for Accuracy: In the first phase, generator and critic are trained with respect to their value functions. Critic's loss function with gradient penalty is [26]:

$V_{C}=\underset{\hat{\mathbf{x}} \sim P_{g}}{\mathbb{E}}[C(\hat{\mathbf{x}})]-\underset{\mathbf{x} \sim P_{r}}{\mathbb{E}}[C(\mathbf{x})]+\lambda \underset{\overline{\mathbf{x}} \sim P_{\mathbf{x}}}{\mathbb{E}}\left[\left(\left\|\nabla_{\overline{\mathbf{x}}} C(\overline{\mathbf{x}})\right\|_{2}-1\right)^{2}\right]$

Where $P_{\mathrm{r}}$ and $P_{\mathrm{g}}$ are real data distribution and generated data distribution, respectively. Note that the third term is the gradient penalty to enforce the Lipschitz constraint, and $P_{\overline{\mathbf{x}}}$ is implicitly defined sampling uniformly along straight lines between pairs of points sampled from the data distribution $P_{\mathbf{r}}$ and the generator distribution $P_{\mathrm{g}}$ [26].

The loss function for the generator network in Phase I of training is also as follows:

$$
\begin{equation*}
V_{G}=-\underset{\hat{\mathbf{x}} \sim P_{g}}{\mathbb{E}}[C(\hat{\mathbf{x}})] \tag{11}
\end{equation*}
$$

2) Phase II: Training for Fairness and Accuracy: In the second phase of training, fairness constraint is enforced on generator to produce fair data. Similar to the definitions in Section II, let $\hat{D}=\{\hat{X}, \hat{Y}, \hat{S}\}$ be a batch of generated data, where $X$ is the unprotected attribute of the generated data, $\hat{Y}$ is the decision with $\hat{Y}=1$ being the successful and favorable value for the decision (e.g. having an income of $>50 K$ for an adult in the adult income dataset), and $\hat{S}$ being the protected attribute with $\hat{S}=0$ showing the unprivileged minority group (for example having a gender of "female" in the adult income data set). The new loss function for the generator in Phase II of training is as follows:

$$
\begin{align*}
V_{G}=-\underset{(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{s}}) \sim P_{g}}{\mathbb{E}}[C(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{s}})]-\lambda_{f}\left(\underset{(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{s}}) \sim P_{g}}{\mathbb{E}}[\hat{\mathbf{y}} \mid \hat{\mathbf{s}}=0]-\right. \\
\left.\underset{(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{s}}) \sim P_{g}}{\mathbb{E}}[\hat{\mathbf{y}} \mid \hat{\mathbf{s}}=1]\right) \tag{12}
\end{align*}
$$

With the above loss function for the generator, the model aims to generate a fair dataset $\{\hat{X}, \hat{Y}, \hat{S}\} \sim P_{g}$ which achieves the demographic fairness with respect to the protected attribute $\hat{S}$ in the generated samples, by minimizing discrimination score in the generated data $P(\hat{Y} \mid \hat{S}=1)-P(\hat{Y} \mid \hat{S}=0)$. The goal in this phase of training is to train the generator to generate synthetic data which is both similar to the real data $\hat{D} \sim D$, and the generated data is fair based on demographic fairness measure. In the ideal case, the generator would produce synthetic data such that $\hat{Y} \perp \hat{S}$. After training is done, the samples are generated and inverse transformed to the original data format. The formal procedure of training the model is shown in Algorithm 1

```
Algorithm 1 training algorithm for the proposed WGAN. We
use $n_{\text {crit }}=4$, batch size of $256, \lambda_{p}=10$, Adam optimizer
with $\alpha=0.0002, \beta_{1}=0.5$, and $\beta_{2}=0.999$
    for $T_{1}$ do
        for $t=1, \ldots, n_{\text {crit }}$ do
            Sample batch $m D(x, y, s) \sim P_{r}$ and $z \sim P(z)$ and $\epsilon \sim U[0,1]$
            $\hat{D}=(\hat{x}, \hat{s}, \hat{y}) \leftarrow G_{\theta}(z)$
            $\bar{D} \leftarrow \epsilon(D)+(1-\epsilon)(\hat{D})$
            Update the critic by descending the gradient:
            $\nabla_{w} \frac{1}{m} \sum_{i=1}^{m} C_{w}(\hat{D})-C_{w}(D)+\lambda_{p}\left(\left\|\nabla_{\bar{D}} C_{w}(\bar{D})\right\|_{2}-1\right)^{2}$
        end for
        Sample a batch $m z \sim P(z)$
        Update the generator by descending the gradient:
        $\nabla_{\theta} \frac{1}{m} \sum_{i=1}^{m}-\left(C_{w}\left(G_{\theta}(z)\right)\right)$
    end for
    for $T_{2}$ do
        for $t=1, \ldots, n_{\text {crit }}$ do
            Sample batch $m D(x, y, s) \sim P_{r}$ and $z \sim P(z)$ and $\epsilon \sim U[0,1]$
            $\hat{D}=(\hat{x}, \hat{s}, \hat{y}) \leftarrow G_{\theta}(z)$
            $\bar{D} \leftarrow \epsilon(D)+(1-\epsilon)(\hat{D})$
            Update the critic by descending the gradient:
            $\nabla_{w} \frac{1}{m} \sum_{i=1}^{m} C_{w}(\hat{D})-C_{w}(D)+\lambda_{p}\left(\left\|\nabla_{\bar{D}} C_{w}(\bar{D})\right\|_{2}-1\right)^{2}$
        end for
        sample a batch $m \hat{D}=\hat{x}, \hat{s}, \hat{y} \sim P\left(G_{\theta}(z)\right)$
        Update the generator by descending the gradient:
            $\nabla_{\theta} \frac{1}{m} \sum_{i=1}^{m}-C_{w}(\hat{D})-\lambda_{f}\left(\frac{\left|D_{s=0, y=1}\right|}{\left|D_{s=0}\right|}-\frac{\left|D_{s=1, y=1}\right|}{\left|D_{s=1}\right|}\right)$
    end for
```


## IV. EXPERIMENT: ONLY PHASE I (NO FAIRNESS)

In this section, we evaluate the effectiveness of the model in producing synthetic data simialr to data coming from a known probability distribution. We show that the model is able to generate synthetic data similar to the reference dataset and compare our results with two state-of-the-art GAN models for generation of tabular datasets, namely TGAN [22] and CTGAN [23]. TGAN is a GAN-based model that generates relational tables by clustering numerical variables to deal with multi-modal distributions and adding noise and KL divergence into loss function to generate discrete features. In CTGAN, mode-specific normalization is applied to numerical values and
the generator works conditionally in order to overcome the imbalance in training data. We evaluate the model on UCI Adult Income Datase ${ }^{1}[30]$. The task we are trying to achieve is as follows: given a dataset $D=\{X, S, Y\} \sim P_{\text {data }}$, generate a dataset $\hat{D}_{\text {syn }}=\{\hat{X}, \hat{S}, \hat{Y}\} \sim P_{s y n}$ s.t. $P_{\text {syn }} \sim P_{\text {data }}$. We are not seeking to achieve fairness in this section, and we solely seek to generate data following the same distribution as real data to achieve data utility.

To compare data utility among generated datasets among different models, we evaluate the performance of using synthetic data as training data for machine learning. At first, the real dataset is divided into two parts: $\mathbf{D}_{\text {train }}$ and $\mathbf{D}_{\text {test }}$. Adult dataset contains a total of 48,842 rows. $90 \%$ of the data were assigned to $\mathbf{D}_{\text {train }}$ and the rest $10 \%$ were assigned to $\mathbf{D}_{\text {test }}$. Next, each model is trained on the training set $\mathbf{D}_{\text {train }}$ for 300 epochs three times. With each training, the trained models are used to generate their corresponding synthetic data $\mathbf{D}_{\text {syn }}$. Three machine learning classifiers are then chosen and trained on each generated $\mathbf{D}_{\text {syn }}$, tested on $\mathbf{D}_{\text {test }}$, and eventually the accuracy and F1 score of classification is recorded. The classifiers used are a Decision Tree Classifier, Logistic Regression, and a Multi Layer Perceptron. Table I reports the results of classification, and compares the results with the case that a classifier is trained on the original $\mathbf{D}_{\text {train }}$, and tested on $\mathbf{D}_{\text {test }}$ (reporting the means and standard deviations of evaluation metrics). The results shows that TabFairGAN and CTGAN outperform TGAN in all cases. TabFairGAN outperforms CTGAN with a DT Classifier. With a LR classifier, the performance of TabFairGAN and CTGAN is identical with respect to accuracy, and TabFairGAN performs slightly better than CTGAN with respect to F1 score. With a MLP classifier, CTGAN performs slightly better than TabFairGAN with respect to accuracy, while TabFairGAN outperforms CTGAN with respect to F1 score. These results display the effetiveness of TabFariGAN with respect to generating data identical to real tabular data.

## V. EXPERIMENTS: FAIR DATA GENERATION AND DATA UTILITY (TRAINING WITH BOTH PHASE I AND PHASE II)

In the second set of experiments, the effectiveness of the model in generating data which is both similar to the reference dataset and also fair is evaluated, and the tradeoff between machine learning efficacy and fairness is investigated. We will experiment with four datasets to test the fairness/utility tradeoff of the model. The four datasets and their attributes are first introduced. All four datasets used in experiments are studied in the literature of algorithmic fairness [3]. Next, we introduce the baseline method with which the results of TabFairGAN are compared. The results are presented and compared in Table II

## A. Datasets

The first dataset is UCI Adult Dataset [30]. This dataset is based on 1994 US census data and contains 48,842 rows with[^0]

attributes such as age, sex, occupation, and education level. for each person, and the target variable indicates whether that individual has an income that exceeds $\$ 50 \mathrm{~K}$ per year or not. In our experiments, we consider the protected attribute to be $\operatorname{sex}(S=$ "Sex", $Y=$ "Income").

The second dataset used in the experiments is the Bank Marketing Data Set [31]. This dataset contains information about a direct marketing campaign of a Portuguese banking institution. Each row of the dataset contains attributes about an individual such as age, job, marital status, housing, duration of that call, and the target variable determines whether that individual subscribed a term deposit or not. The dataset contains 45,211 records. Similar to [32], we have considered age to be the protected attribute (a young individual has a higher chance of being labeled as "yes" to subscribe a term deposit). In order to have a binary protected attribute, we set a cut-off value of 25 and an age of more than 25 is considered "older", while an age of less than or equal to 25 is considered "younger" ( $S=$ "Age", $Y=$ "Subscribed").

The third dataset used in this section is the ProPublica dataset from COMPAS risk assessment system [33]. This dataset contains information about defendants from Broward County, and contains attributes about defendants such as their ethnicity, language, marital status, sex, etc. ,and for each individual a score showing the likelihood of recidivism (reoffending). In this experiments we used a modified version of the dataset. First, attributes such as FirstName, LastName, MiddleName, CASE_ID, and DateOfBirth are removed. Studies have shown that this dataset is biased against African Americans [1]. Therefore, ethnicity is chosen to be the protected attribute for this study. Only African American and Caucasian individuals are kept and the rest are dropped. The target variable in this dataset is a risk decile score provided by COMPAS system, showing the likelihood of that individual to re-offend, which ranges from 1 to 10 . The final modified dataset contains 16,267 records with 16 features. To make the target variable binary, a cut-off value of 5 is considered and individuals with a declile score of less than 5 are considered "Low_Chance", while the rest are considered "High_Chance". ( $S=$ "Ethnicity", $Y=$ "Recidivism_Chance").

The last dataset used in experiments is the Law School Admission Council which is made by conducting a survey across 162 law schools in the United States [34]. This dataset contains information on 21,790 law students such as their GPA (grade-point average), LSAT score, race, and the target variable is whether the student had a high FYA (first year average grade). Similar to other studies (such as [35]), we have considered race to be the protected attribute. We only considered individuals with "Black" or "White" race. The modified data contains 19,567 records. ( $S=$ "Race", $Y=$ "FYA"). There discrimination score (DS) of all datasets are reported in Table II

TABLE I

COMPARING THE RESULTS TABFAIRGAN FOR ACCURATE DATA GENERATION WITH TGAN AND CTGAN MODELS

| Classifier | DTC |  | LR |  | MLP |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Accuracy | F1 | Accuracy | F1 | Accuracy | F1 |
| Original Data | $0.811 \pm 0.001$ | $0.606 \pm 0.002$ | $0.798 \pm 0.000$ | $0.378 \pm 0.000$ | $0.780 \pm 0.051$ | $0.488 \pm 0.075$ |
| TabFairGan | $\mathbf{0 . 7 8 3} \pm \mathbf{0 . 0 0 1}$ | $\mathbf{0 . 5 4 4} \pm \mathbf{0 . 0 0 2}$ | $\mathbf{0 . 7 9 4} \pm \mathbf{0 . 0 2 0}$ | $\mathbf{0 . 2 3 9} \pm \mathbf{0 . 0 1 2}$ | $0.778 \pm 0.045$ | $\mathbf{0 . 4 0 5} \pm \mathbf{0 . 1 7 4}$ |
| TGAN | $0.661 \pm 0.013$ | $0.503 \pm 0.012$ | $0.765 \pm 0.010$ | $0.170 \pm 0.008$ | $0.623 \pm 0.197$ | $0.376 \pm 0.159$ |
| CTGAN | $0.777 \pm 0.003$ | $0.482 \pm 0.004$ | $\mathbf{0 . 7 9 4} \pm \mathbf{0 . 0 2 3}$ | $0.232 \pm 0.012$ | $\mathbf{0 . 7 8 4} \pm \mathbf{0 . 0 0 7}$ | $0.305 \pm 0.104$ |

## B. Baseline Model: Certifying and Removing Disparate Impact

In their work Feldman et al. [5] proposed a method to modify a dataset to remove bias and preserve relevant information in the data. In dataset $D=\{X, S, Y\}$, given the protected attribute $S$ and a single numerical attribute $X$, let $X_{s}=\operatorname{Pr}(X \mid S=s)$ denote the marginal distribution on $X$ conditioned on $S=s$. Considering $F_{s}: X_{s} \rightarrow[0,1]$ the cumulative distribution function for values $x \in X_{s}$, they define a "median" distribution $A$ in terms of its quantile function $F_{A}^{-1}: F_{A}^{-1}(u)=$ median $_{s \in S} F_{s}^{-1}(u)$. They then propose a repair algorithm which creates $\bar{X}$, such that for all $x \in X_{s}$ the corresponding $\bar{x}=F_{A}^{-1}\left(F_{s}(x)\right)$. To control the tradeoff between fairness and accuracy, they define and calculate $\lambda$ - partial repair by:

$$
\begin{equation*}
\bar{F}_{s}^{-1}=(1-\lambda) F_{s}^{-1}+\lambda\left(F_{A}\right)^{-1} \tag{13}
\end{equation*}
$$

The result of such partial repair procedure is a dataset $\bar{D}=\{\bar{X}, S, Y\}$ which is more fair and preserves relevant information for classification task. We call this method CRDI henceforth.

## C. Results

The goal in this section is to train the proposed network on datasets and produce similar data that is also fair with respect to protected attributes defined for each dataset. The process is as follows: The models are first trained on each dataset. As mentioned in Section III-C, training of the network includes two phases: in the first phase, the network is only trained for accuracy for a certain number of epochs, and then in the second phase, the loss function of generator is modified and the network gets trained for accuracy and fairness. Once the training is finished, the generator of the network is used to produce synthetic data $\mathbf{D}_{\text {syn }}$. We also generated repaired datasets using CRDI method described in Section V-B to compare our results with. For each model, we train five times and report the means and standard deviations of evaluation results in Table II

The generated data $\mathbf{D}_{\text {syn }}$ is then evaluated from two perspective: fairness and utility. To evaluate the fairness of $\mathbf{D}_{\text {syn }}$, we adopt discrimination score (DS): $D S=P(y=1 \mid s=$ 1) $-P(y=1 \mid s=0)$. Looking into Table II the results show that comparing with CRDI, TabFairGAN could more effectively produce datasets s.t. demographic parity in the generated data is almost removed. The demographic parity of the produced datasets by TabFairGAN, beat the repaired datasets produced by CRDI.

To evaluate data utility, we adopt a decision tree classifier with the default parameter setting [36]. For TabFairGAN data, We train the decision tree classifier on $\mathbf{D}_{\text {syn }}$ and test it on $\mathbf{D}_{\text {test }}$, and report the accuracy and F1-score of the classifier. We also train decision tree classifiers on repaired data $\bar{D}$ produced by CRDI, and test on $\mathbf{D}_{\text {test }}$ and report accuracy and f1-score. Table II shows that repaired data $\bar{D}$ produced by CRDI has better data utility for adult dataset, COMPAS dataset, and Law School dataset by less than $5 \%$ in all cases, while the accuracy of $\mathbf{D}_{\text {syn }}$ produced by TabFairGAN is almost $8 \%$ higher than that of $\bar{D}$ produced by CRDI.

The last evaluation we perform on the produced datasets is to examine discrimination score (DS) of the classifier. we adopt discrimination score (DS) for classifier: $D S=P(\hat{y}=$ $1 \mid s=1)-P(\hat{y}=1 \mid s=0)$. The results in Table II show that discrimination score of the decision tree classifier trained on $\mathbf{D}_{\text {syn }}$ for Adult dataset and Law School is lower by almost $4 \%$ and $13 \%$, respectively, while the discrimination score of the decision tree classifier trained on $\overline{\mathbf{D}}$ for Bank dataset and COMPAS dataset is lower by $1 \%$ and $0.003 \%$, respectively.

The parameter settings of the models on each datasets is reported in the Appendix. The results show, while CRDI narrowly beats TabFairGAN in terms of data utility, TabFairGAN beats CRDI in terms of discrimination score in all cases for generated data and in 2 out of 4 cases in the generated classifiers. This is attributed to fairness utility tradeoff of TabFairGAN governed by $\lambda_{f}$. The case of COMPAS dataset is interesting since none of the models could decrease discrimination score in the classifier much, comparing to the discrimination score in the original dataset. Looking into the data and performing a correlation analysis, risk decile score (target variable) has a high Pearson correlation of 0.757 with one of columns names RecSupervisionLevel which denotes the supervisory status of each individual. This reveals that although the generated dataset $\mathbf{D}_{\text {syn }}$ has a lower discrimination score of 0.009 , disparate impact exists in the dataset, indicating that the discriminatory outcomes are not explicitly caused by the protected attribute, but are also from the proxy unprotected attributes [20].

TABLE II

COMPARING THE RESULTS OF TABFAIRGAN FOR FAIR DATA GENERATION WITH CRDI

| Dataset | Original Data |  |  | TabFairGAN |  |  |  | CRDI |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Orig. Acc. | F1 Orig. | DS in Orig. Data | DS Gen. Data | Acc. Gen. Data | F1 Gen. Data | DS in Classifier | DS Rep. Data. | Acc. Rep. Data | F1 Rep. Data | DS in Classifier |
| Adult | $0.816 \pm 0.005$ | $0.619 \pm 0.013$ | 0.195 | $0.009 \pm \mathbf{0 . 0 2 7}$ | $0.773 \pm 0.013$ | $0.536 \pm 0.022$ | $0.082 \pm 0.038$ | $0.165 \pm 0.048$ | $0.793 \pm 0.011$ | $0.558 \pm 0.029$ | $0.121 \pm 0.024$ |
| Bank | $0.879 \pm 0.004$ | $0.491 \pm 0.020$ | 0.126 | $0.001 \pm \mathbf{0 . 0 1 1}$ | $0.854 \pm 0.004$ | $0.373 \pm 0.024$ | $0.060 \pm 0.056$ | $0.122 \pm 0.004$ | $0.776 \pm 0.004$ | $0.384 \pm 0.011$ | $0.050 \pm 0.017$ |
| COMPAS | $0.903 \pm 0.007$ | $0.914 \pm 0.007$ | 0.258 | $0.009 \pm \mathbf{0 . 1 0 2}$ | $0.860 \pm 0.040$ | $0.876 \pm 0.033$ | $0.208 \pm 0.072$ | $0.119 \pm 0.128$ | $0.893 \pm 0.021$ | $0.906 \pm 0.020$ | $0.205 \pm 0.055$ |
| Law School | $0.854 \pm 0.008$ | $0.918 \pm 0.005$ | 0.302 | $0.024 \pm 0.036$ | $0.847 \pm 0.020$ | $0.916 \pm 0.012$ | $0.153 \pm 0.072$ | $0.233 \pm 0.103$ | $0.892 \pm 0.004$ | $0.941 \pm 0.002$ | $0.289 \pm 0.057$ |

![](https://cdn.mathpix.com/cropped/2024_06_04_a31a2c0cdbfc87b415ddg-7.jpg?height=521&width=789&top_left_y=607&top_left_x=169)

Fig. 2. Exploring the trade-off between accuracy and fairness by incremental increasing of parameter $\lambda_{f}$

## D. Utility and Fairness Trade-off

To explore the trade-off between utility and fairness of the generated data, we perform the following experiment: $\lambda_{f}$ was increased between $[0.05,0.7]$ in steps of 0.05 , and for each value of $\lambda_{f}$ the model was trained 170 epochs in phase I and 30 times in the phase II. For each $\lambda_{f}$ value, five training was performed and the average of Discrimination Score was recorded for each $\lambda_{f}$. Figure 2 shows the results, plotted along with standard deviation as confidence intervals. We can observe that discrimination score of the generated synthetic datasets ( $D_{\text {syn }}$ ) is decreasing significantly as $\lambda_{f}$ decreases. Meanwhile, classifier accuracy layoff, i.e. the reduction in decision tree classifier's accuracy comparing to the case in which the classifier is trained on the real original training dataset ( $D_{\text {train }}$ ), is increasing slightly as $\lambda_{f}$ increases.

## VI. CONCLUSION

In this paper, we proposed a Wasserstein Generative Adversarial Network that could generate synthetic data similar to a reference data. We showed that in the case of unconditional tabular data generation, i.e. with no fairness constrains, the model is able to produce data with high quality comparing to other GANs developed for the same purpose. We also showed that by adding a fairness constraint to the generator, the model is able to achieve data generation which improves the demographic parity of the generated data. We tested the model on four datasets studies in the fairness literature and compared our results with that of [5]. As a generative model, GANs have a great potential to be utilized for fair data generation, specially in the case that the real dataset is limited. There are other field in which GANs could be utilized for tabular data generation, such as the research involved with data privacy [37]. In the future work, we will explore other more sophisticated data generation constraints, e.g. considering enforcing other fairness metrics such as equality of odds and equality of opportunity. We also consider exploring utilizing GANs for fairness in other data types, such as text and image data.

## REFERENCES

[1] A. Chouldechova, "Fair prediction with disparate impact: A study of bias in recidivism prediction instruments," Big data, vol. 5, no. 2, pp. $153-163,2017$.

[2] A. Lambrecht and C. Tucker, "Algorithmic bias? an empirical study of apparent gender-based discrimination in the display of stem career ads," Management Science, vol. 65, no. 7, pp. 2966-2981, 2019.

[3] D. Pessach and E. Shmueli, "Algorithmic fairness," arXiv preprint arXiv:2001.09784, 2020.

[4] F. Kamiran and T. Calders, "Data preprocessing techniques for classification without discrimination," Knowledge and Information Systems, vol. 33, no. 1, pp. 1-33, 2012.

[5] M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and S. Venkatasubramanian, "Certifying and removing disparate impact," in proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, 2015, pp. 259-268.

[6] T. Kamishima, S. Akaho, H. Asoh, and J. Sakuma, "Fairness-aware classifier with prejudice remover regularizer," in Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 2012, pp. 35-50.

[7] M. Hardt, E. Price, and N. Srebro, "Equality of opportunity in supervised learning," Advances in neural information processing systems, vol. 29, pp. 3315-3323, 2016.

[8] A. Oussidi and A. Elhassouny, "Deep generative models: Survey," in 2018 International Conference on Intelligent Systems and Computer Vision (ISCV). IEEE, 2018, pp. 1-8.

[9] S. E. Fahlman, G. E. Hinton, and T. J. Sejnowski, "Massively parallel architectures for al: Netl, thistle, and boltzmann machines," in National Conference on Artificial Intelligence, AAAI, 1983

[10] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Advances in neural information processing systems, vol. 27, 2014.

[11] A. Brock, J. Donahue, and K. Simonyan, "Large scale gan training for high fidelity natural image synthesis," arXiv preprint arXiv:1809.11096, 2018 .

[12] C. Vondrick, H. Pirsiavash, and A. Torralba, "Generating videos with scene dynamics," Advances in neural information processing systems, vol. 29, pp. 613-621, 2016.

[13] M. Menéndez, J. Pardo, L. Pardo, and M. Pardo, "The jensen-shannon divergence," Journal of the Franklin Institute, vol. 334, no. 2, pp. 307$318,1997$.

[14] Y. Rubner, C. Tomasi, and L. J. Guibas, "The earth mover's distance as a metric for image retrieval," International journal of computer vision, vol. 40, no. 2, pp. 99-121, 2000.

[15] M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein generative adversarial networks," in International conference on machine learning. PMLR, 2017, pp. 214-223.

[16] H. Edwards and A. Storkey, "Censoring representations with an adversary," arXiv preprint arXiv:1511.05897, 2015.

[17] D. Madras, E. Creager, T. Pitassi, and R. Zemel, "Learning adversarially fair and transferable representations," in International Conference on Machine Learning. PMLR, 2018, pp. 3384-3393.

[18] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating unwanted biases with adversarial learning," in Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, 2018, pp. 335-340.

[19] P. Sattigeri, S. C. Hoffman, V. Chenthamarakshan, and K. R. Varshney, "Fairness gan: Generating datasets with fairness properties using a generative adversarial network," IBM Journal of Research and Development, vol. 63, no. 4/5, pp. 3-1, 2019.

[20] D. Xu, S. Yuan, L. Zhang, and X. Wu, "Fairgan: Fairness-aware generative adversarial networks," in 2018 IEEE International Conference on Big Data (Big Data). IEEE, 2018, pp. 570-575.

[21] E. Choi, S. Biswal, B. Malin, J. Duke, W. F. Stewart, and J. Sun, "Generating multi-label discrete patient records using generative adversarial networks," in Machine learning for healthcare conference. PMLR, 2017, pp. 286-305.

[22] L. Xu and K. Veeramachaneni, "Synthesizing tabular data using generative adversarial networks," arXiv preprint arXiv:1811.11264, 2018.

[23] L. Xu, M. Skoularidou, A. Cuesta-Infante, and K. Veeramachaneni, "Modeling tabular data using conditional gan," in Advances in Neural Information Processing Systems, 2019.

[24] D. Xu, S. Yuan, L. Zhang, and X. Wu, "Fairgan+: Achieving fair data generation and classification through generative adversarial nets," in 2019 IEEE International Conference on Big Data (Big Data). IEEE, 2019, pp. 1401-1406.

[25] T. M. Beasley, S. Erickson, and D. B. Allison, "Rank-based inverse normal transformations are increasingly used, but are they merited?" Behavior genetics, vol. 39, no. 5, pp. 580-595, 2009.

[26] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, "Improved training of wasserstein gans," in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/ file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf

[27] C. Villani, Optimal transport: old and new. Springer, 2009, vol. 338.

[28] E. Jang, S. Gu, and B. Poole, "Categorical reparameterization with gumbel-softmax," arXiv preprint arXiv:1611.01144, 2016.

[29] B. Xu, N. Wang, T. Chen, and M. Li, "Empirical evaluation of rectified activations in convolutional network," arXiv preprint arXiv:1505.00853, 2015.

[30] D. Dua and C. Graff, "UCI machine learning repository," 2017. [Online]. Available: http://archive.ics.uci.edu/ml

[31] S. Moro, P. Cortez, and P. Rita, "A data-driven approach to predict the success of bank telemarketing," Decision Support Systems, vol. 62, pp. 22-31, 2014.

[32] M. B. Zafar, I. Valera, M. G. Rogriguez, and K. P. Gummadi, "Fairness constraints: Mechanisms for fair classification," in Artificial Intelligence and Statistics. PMLR, 2017, pp. 962-970.

[33] J. Angwin, J. Larson, S. Mattu, and L. Kirchner. (2016) Machine bias propublica. [Online]. Available: https://www.propublica.org/article/ machine-bias-risk-assessments-in-criminal-sentencing

[34] L. F. Wightman, "Lsac national longitudinal bar passage study. 1sac research report series." 1998.

[35] Y. Bechavod and K. Ligett, "Penalizing unfairness in binary classification," arXiv preprint arXiv:1707.00044, 2017.

[36] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg et al., "Scikit-learn: Machine learning in python," the Journal of machine Learning research, vol. 12, pp. 2825-2830, 2011.

[37] N. Park, M. Mohammadi, K. Gorde, S. Jajodia, H. Park, and Y. Kim, "Data synthesis based on generative adversarial networks," arXiv preprint arXiv:1806.03384, 2018
</end of paper 0>


<paper 1>

#### Abstract

ה̈hrough a fair loo in image datasets

\section*{e}

AAmirarsalan Rajabi ${ }^{1}$

amirarsalan@knights.ucf.edu

Mehdi Yazdani-Jahromi ${ }^{2}$

yazdani@knights.uct.edu

ozlem Ozmen Garibay ${ }^{1,2}$

ozlem@ucf.edu

Gita Sukthankar ${ }^{1}$

cejtars@eecs.ucf.edu

${ }^{1}$ Department of Computer Science<br>University of Central Florida<br>Orlando, Florida, USA<br>${ }^{2}$ Department of Industrial Engineering<br>and Management Systems<br>University of Central Florida<br>Orlando, Florida, USA

Abstract

With the recent growth in computer vision applications, the question of how fair and unbiased they are has yet to be explored. There is abundant evidence that the bias present in training data is reflected in the models, or even amplified. Many previous methods for image dataset de-biasing, including models based on augmenting datasets, are computationally expensive to implement. In this study, we present a fast and effective model to de-bias an image dataset through reconstruction and minimizing the statistical dependence between intended variables. Our architecture includes a U-net to reconstruct images, combined with a pre-trained classifier which penalizes the statistical dependence between target attribute and the protected attribute. We evaluate our proposed model on CelebA dataset, compare the results with a state-of-the-art de-biasing method, and show that the model achieves a promising fairness-accuracy combination.


## 1 Introduction

Due to their increased usage within myriad software applications, artificial intelligence algorithms now influence many aspects of people's lives, particularly when they are embedded into decision-support tools used by educators, government agencies, and various industry sectors. Thus, it is crucial to make sure that these algorithms are scrutinized to ensure fairness and remove unjust biases. Bias has been shown to exist in several deployed AI systems, including the well known Correlational Offender Management Profiling for Alternative Sanctions (COMPAS). COMPAS is an automated decision making system used by the US criminal justice system for assessing a criminal defendant's likelihood of re-offending. By exploring the risk scores assigned to individuals, this system has been shown to be biased against African Americans [⿴]]. Other examples include a version of Google's targeted advertising system in which highly paid jobs were advertised more frequently to men vs. women [匚][^0]

Bias in computer vision is a major problem, often stemming from the training datasets used for computer vision models [匹]. There is evidence suggesting the existence of multiple types of bias, including capture and selection bias, in popular image datasets [⿴]. The problems arising from bias in computer vision can manifest in different ways. For instance, it is observed that in activity recognition models, when the datasets contain gender bias, the bias is further amplified by the models trained on those datasets [ㅈ]. Face recognition models may exhibit lower accuracy for some classes of race or gender $[\square]$.

Works such as [1, [7] suggest methods to mitigate bias in visual datasets. Several studies have deployed GANs for bias mitigation in image datasets. For example, [冖]] modified the value function of GAN to generate fair image datasets. FairFaceGAN [⿴囗⿰丨丨⿰讠己]) implements a facial image-to-image translation, preventing unwanted translation in protected attributes. Ramaswamy et al. propose a model to produce training data that is balanced for each protected attribute, by perturbing the latent vector of a GAN [四]. Other studies employing GANs for fair data generation include [ $\square, \square]$.

A variety of techniques beyond GANs have been applied to the problems of fairness in AI. A deep information maximization adaptation network was used to reduce racial bias in face image datasets [⿴], and reinforcement learning was used to learn a race-balanced network in [匚]]. Wang et al. propose a generative few-shot cross-domain adaptation algorithm to perform fair cross-domain adaption and improve performance on minority category [囬]. The work in [5] proposes adding a penalty term into the softmax loss function to mitigate bias and improve fairness performance in face recognition. Quadriento et al. [ㅁ] propose a method to discover fair representations of data with the same semantic meaning of the input data. Adversarial learning has also successfully been deployed for this task [⿴囗 , 120].

This paper addresses the issue of a decision-making process being dependent on protected attributes, where this dependence should ideally be avoided. From a legal perspective, a protected attribute is an attribute upon which discrimination is illegal [ㅁ]], e.g. gender or race. Let $D=(\mathcal{X}, \mathcal{S}, \mathcal{Y})$ be a dataset, where $\mathcal{X}$ represents unprotected attributes, $\mathcal{S}$ is the protected attribute, and $\mathcal{Y}$ be the target attribute. If in the dataset $D$, the target attribute is not independent of the protected attribute $(\mathcal{Y} \not \perp \mathcal{S})$, then it is very likely that the decisions $\hat{\mathcal{Y}}$ made by a decision-making system which is trained on $D$, is also not independent of the protected attribute $(\hat{\mathcal{Y}} \not \subset \mathcal{S})$.

We propose a model to reconstruct an image dataset to reduce statistical dependency between a protected attribute and target attribute. We modify a U-net [ $\square$ ] to reconstruct the image dataset and apply the Hilbert-Schmidt norm of the cross-covariance operator [] between reproducing kernel Hilbert spaces of the target attribute and the protected attribute, as a measure of statistical dependence. Unlike many previous algorithms, our proposed method doesn't require training new classifiers on the unbiased data, but instead reconstructing images in a way that reduces the bias entailed by using the same classifiers.

In Section 2 we present the problem, the notion of independence, and our proposed methodology. In Section 3 we describe the CelebA dataset and the choice of feature categorization, introduce the baseline model with which we compare our results [罒], our model's implementation details, and finally present the experiments and results.

Bias mitigation methods can be divided into three general categories of pre-process, inprocess, and post-process. Pre-process methods include modifying the training dataset before feeding it to the machine learning model. In-process methods include adding regularizing terms to penalize some representation of bias during the training process. Finally, post-process methods include modifying the final decisions of the classifiers [ $\square$ ]. Kamiran and Calders [匚] propose methods such as suppression which includes removing attributes highly correlated
with the protected attribute, reweighing, i.e. assigning weights to different instances in the data, and massaging the data to change labels of some objects. Bias mitigation methods often come at the expense of losing some accuracy, and these preliminary methods usually entail higher fairness-utility cost. More sophisticated methods with better results include using generative models to augment the biased training dataset with unbiased data [미], or training the models on entirely synthetic unbiased data [ㅈ]. Wang et al.[ㅈ] provide a set of analyses and a benchmark to evaluate and compare bias mitigation techniques in visual recognition models.

## 2 Methodology

Consider a dataset $D=(\mathcal{X}, \mathcal{S}, \mathcal{Y})$, where $\mathcal{X}$ is the set of images, $\mathcal{Y}=\{+1,-1\}$ is the target attribute such as attractiveness, and $\mathcal{S}=\{A, B, C, \ldots\}$ is the protected attribute such as gender. Assume there exists a classifier $f:(\mathcal{X}) \rightarrow \mathcal{Y}$, such that the classifier's prediction for target attribute is not independent from the protected attribute, i.e. $f(\mathcal{X}) \not \Perp \mathcal{S}$. Our objective is to design a transformation $g: \mathcal{X} \rightarrow \widetilde{\mathcal{X}}$, such that 1) $f(\widetilde{\mathcal{X}}) \perp \mathcal{S}$, i.e. the classifier's predictions for target attribute is independent of the protected attribute, and 2) $f(\widetilde{\mathcal{X}}) \approx f(\mathcal{X})$, i.e. the classifier still achieves high accuracy.

In other words we want to train a network to transform our original images, such that if the classifiers that are trained on the original and unmodified images, are used to predict the target attribute (attractiveness in our example) from the transformed version of an image, they still achieve high accuracy, while the predictions of those classifiers are independent of the protected attribute (gender in our example). It should be noted that we are not seeking to train new classifiers, but rather only aim to modify the input images. This is a main distinction between our methodology and most of other techniques (e.g. [⿴] and [四]), in which the process includes training new classifiers on modified new image datasets and achieving fair classifiers.

Our proposed model consists of a U-net [⿴囗]] as the neural network that transforms the original images. This type of network was originally proposed for medical image segmentation, and has been widely used since its introduction. The encoder-decoder network consists of two paths, a contracting path consisting of convolution and max pooling layers, and a consecutive expansive path consisting of upsampling of the feature map and convolutions. Contrary to [⿴] where each image is provided with a segmented image label, we provide our U-net with the exact same image as the label, and alter the loss function from cross-entropy to mean squared error, so that the network gets trained to produce an image as close to the original image as possible, in a pixel-wise manner.

While some previous fairness studies consider decorrelating the target attribute from the protected attributes, what must be ultimately sought however, is independence between the protected attribute and the target attribute. Dealing with two random variables which are uncorrelated is easier than independence, as two random variables might have a zero correlation, and still be dependent (e.g. two random variables $A$ and $B$ with recordings $A=[-2,-1,0,1,2]$ and $B=[4,1,0,1,4]$ have zero covariance, but are apparently not independent). Given a Borel probability distribution $\mathbf{P}_{a b}$ defined on a domain $\mathcal{A} \times \mathcal{B}$, and respective marginal distributions $\mathbf{P}_{a}$ and $\mathbf{P}_{b}$ on $\mathcal{A}$ and $\mathcal{B}$, independence of $a$ and $b(a \Perp b)$ is equal to $\mathbf{P}_{x y}$ factorizing as $\mathbf{P}_{x}$ and $\mathbf{P}_{y}$. Furthermore, two random variables $a$ and $b$ are independent, if and only if any bounded continuous function of the two random variables are uncorrelated [ [] .

Let $\mathcal{F}$ and $\mathcal{G}$ denote all real-value functions defined on domains $\mathcal{A}$ and $\mathcal{B}$ respectively.

![](https://cdn.mathpix.com/cropped/2024_06_04_0e50f4443c4eaaea4d3bg-04.jpg?height=649&width=928&top_left_y=84&top_left_x=215)

Figure 1: Our model consists of an encoder-decoder (U-net) and a double-output pre-trained ResNet classifier. First, the output batch of the U-net (reconstructed images) is compared with the original batch of images by calculating MSE loss. Then, the output batch of the U-net passes through the ResNet and statistical dependency of the two vectors is calculated by HSIC. Detailed architecture of the U-net is described in the supplementary material.

In their paper Gretton et al. $[⿴]$ define the Hilbert-Schmidt norm of the cross-covariance operator:

$$
\begin{equation*}
H S I C\left(\mathbf{P}_{a b}, \mathcal{F}, \mathcal{G}\right):=\left\|C_{a b}\right\|_{H S}^{2} \tag{1}
\end{equation*}
$$

where $C_{a b}$ is the cross-covariance operator. They show that if $\left\|C_{a b}\right\|_{H S}^{2}$ is zero, then $\operatorname{cov}(f, g)$ will be zero for any $f \in \mathcal{F}$ and $g \in \mathcal{G}$, and therefore the random variables $a$ and $b$ will be independent. Furthermore, they show if $\mathcal{Z}:=\left(a_{1}, b_{1}\right), \ldots,\left(a_{n}, b_{n}\right) \in \mathcal{A} \times \mathcal{B}$ are a series of $\mathrm{n}$ independent observations drawn from $\mathbf{P}_{a b}$, then a (biased) estimator of HSIC is [⿴]:

$$
\begin{equation*}
H S I C(\mathcal{Z}, \mathcal{F}, \mathcal{G}):=(n-1)^{-2} \operatorname{tr}(K H L H) \tag{2}
\end{equation*}
$$

where $H, K, L \in \mathbb{R}^{n \times n}, K$ and $L$ are Gram matrices [ $\left.\square\right], K_{i j}:=k\left(a_{i}, a_{j}\right), L_{i j}:=l\left(b_{i}, b_{j}\right), k$ and $l$ are universal kernels, and $H_{i j}:=\delta_{i j}-n^{-1}$ centers the observations in feature space. We use Hilbert-Schmidt independence criteria to penalize the model for dependence between the target attribute and the protected attribute.

### 2.1 Training Loss Function

We seek to modify a set of images, such that 1) the produced images are close to the original images, and 2) the predicted target attribute is independent from the predicted protected attribute. In the optimization problem, image quality (1) is measured by pixel-wise MSE loss. For independence (2), consider our U-net network as a mapping from original image to the transformed image, i.e. $U_{w}(\mathbf{x})=\widetilde{\mathbf{x}}$. Consider also a function $h: \mathcal{X} \rightarrow[0,1] \times[0,1]$, where $h\left(\mathbf{x}_{i}\right)=\left(h_{1}\left(\mathbf{x}_{i}\right), h_{2}\left(\mathbf{x}_{i}\right)\right)=\left(\mathrm{P}\left(y_{i}=1 \mid \mathbf{x}_{i}\right), \mathrm{P}\left(s_{i}=1 \mid \mathbf{x}_{i}\right)\right)$. Our objective is to train the parameters of $U_{w}$ such that $h_{1}\left(U_{w}(\mathbf{x})\right) \Perp h_{2}\left(U_{w}(\mathbf{x})\right)$, i.e. $h_{1}\left(U_{w}(\mathbf{x})\right)$ is independent of $h_{2}\left(U_{w}(\mathbf{x})\right)$.

Given $X$ representing a batch of $\mathrm{N}$ training images and $\widetilde{X}$ representing the transformed
![](https://cdn.mathpix.com/cropped/2024_06_04_0e50f4443c4eaaea4d3bg-05.jpg?height=344&width=996&top_left_y=111&top_left_x=215)

Figure 2: Examples of CelebA dataset original images. Images in the first row are labeled not Male and images in the second row are labeled Male. In each row, the first three images are labeled Attractive and the last three images are labeled not Attractive.

batch, our formal optimization problem is as follows:

$$
\begin{align*}
\underset{U_{w}}{\operatorname{minimize}} & \underbrace{\frac{1}{N C W H} \sum_{n=1}^{N} \sum_{i, j, k}\left(\mathbf{x}_{i j k}^{n}-\widetilde{\mathbf{x}}_{i j k}^{n}\right)^{2}}_{\text {image accuracy }}  \tag{3}\\
& +\lambda \times \underbrace{H S I C\left(h_{1}(\widetilde{X}), h_{2}(\widetilde{X})\right)}_{\text {independence }}
\end{align*}
$$

where $N$ is the number of samples, $C$ is the number of channels of an image, $W$ is the width of an image, $H$ is the height of an image, and $\lambda$ is the parameter that controls the trade-off between accuracy of the transformed images and independence (fairness). In practice, the mapping function $U_{w}$ that we use is a U-net, the function $h(\cdot)$ is a pre-trained classifier with two outputs $h_{1}$ and $h_{2}$, each being the output of a Sigmoid function within the range of $[0,1]$, where $h_{1}=\mathrm{P}(Y=1 \mid X)$ (a vector of size $N$ ), and $h_{2}=\mathrm{P}(S=1 \mid X)$ (also a vector of size $N$ ), and $\operatorname{HSIC}(\cdot, \cdot)$ denotes Hilbert-Schmidt Independence Criteria.

Figure 1 shows the network architecture and a schematic of the training procedure. Consider a batch of original images $X$ entering the U-net. The U-net then produces the reconstructed images $U_{w}(X)=\widetilde{X}$. To calculate the image accuracy part of the loss function, the original image batch $X$ is provided as label and the Mean Squared Error is calculated to measure the accuracy of the reconstructed images. The ResNet component in Figure 1 is our $h(\cdot)$ function as described before, which is a pre-trained ResNet classifier that takes as input a batch of images and returns two probability vectors. The second part of the loss function, independence, is calculated by entering the reconstructed images $\widetilde{X}$ into this ResNet classifier, and calculating the HSIC between the two vectors.

As noted before, the image dataset is reconstructed in a way that using them on the original biased classifiers, will result in an improvement in classifications. This is dissimilar to some previous works such as [四] and [匚], in which the model training process includes augmenting the original dataset with generated images and training new fair classifiers [四], or discovering fair representations of images and subsequently training new classifiers [ $[\square]$ ].

## 3 Experiments

In this section, we test the methodology described in Section 2 on CelebA dataset []․ We first introduce the CelebA dataset and the attribute categories in CelebA. We then describe the implementation details of our model. Subsequently, the method described in Ramaswamy et al. $[\mathbb{\square}]$ and the two versions of it that we use as baseline models to compare our results with are introduced. Finally, we introduce evaluation metrics and present the results.

### 3.1 CelebA dataset

CelebA is a popular dataset that is widely used for training and testing models for face detection, particularly recognising facial attributes. It consists of 202,599 face images of celebrities, with 10,177 identities. Each image is annotated with 40 different binary attributes describing the image, including attributes such as Black_Hair, Pale_Skin, Wavy_Hair, Oval_Face, Pointy_Nose, and other attributes such as Male, Attractive, Smiling, etc. The CelebA dataset is reported to be biased [B]]. In this experiment, we consider Male attribute as the protected attribute (with Male $=0$ showing the image does not belong to a man and Male $=1$ showing the image belongs to a man), and Attractive to be the target attribute. We divide the dataset into train and test sets, with train set containing 182,599 and test set containing 20,000 images. In the training set, $67.91 \%$ of images with Male $=0$ are annotated to be attractive (Attractive $=1$ ), while only $27.93 \%$ of images with Male $=1$ are annotated as being attractive (Attractive $=1$ ). This shows bias exists against images with Male $=1$.

In order to compare our results with [四], we follow their categorization of CelebA attributes. Leaving out gender (Male) as the protected attribute, among the rest 39 attributes in CelebA dataset, [匹] eliminates some attributes such as Blurry and Bald as they contain less than $5 \%$ positive images. The remaining 26 attributes is subsequently categorized into three groups. inconsistently-labeled attributes are the ones that by visually examining sets of examples, the authors often disagree with the labeling and could not distinguish between positive and negative examples [四]. This group includes attributes such as Straight_Hair, and Big_Hair. The second group of attributes are the ones that are called gender-dependent and the images are labeled to have (or not have) attributes based on the perceived gender [四]. These include attributes such as Young, Arched_Eyebrows and Receding_Hairline. Finally, the last group of attributes are called gender-independent. These attributes are fairly consistently labeled and are not much dependent on gender expression. This group includes attributes such as Black_Hair, Bangs, and Wearing_Hat. The list of all attributes is provided in supplementary material.

In order to compare our results with [四], we follow their categorization of CelebA attributes. Leaving out gender (Male) as the protected attribute, among the rest 39 attributes in CelebA dataset, [四] eliminates some attributes such as Blurry and Bald as they contain less than $5 \%$ positive images. The remaining 26 attributes is subsequently categorized into three groups. inconsistently-labeled attributes are the ones that by visually examining sets of examples, the authors often disagree with the labeling and could not distinguish

![](https://cdn.mathpix.com/cropped/2024_06_04_0e50f4443c4eaaea4d3bg-06.jpg?height=40&width=1286&top_left_y=1890&top_left_x=50)
Big_Lips, Big_Nose, Oval_Face, Pale_Skin, and Wavy_Hair. The second group of attributes are the ones that are called gender-dependent and the images are labeled to have (or not have) attributes based on the perceived gender [⿴囗⿰丨㇄]]. These include Young, Arched_Eyebrows, Attractive, Bushy_Eyebrows, Pointy_Nose, and Recedi
![](https://cdn.mathpix.com/cropped/2024_06_04_0e50f4443c4eaaea4d3bg-07.jpg?height=340&width=980&top_left_y=112&top_left_x=228)

Figure 3: Examples of CelebA dataset images and how the model reconstructs them. The first row shows a set of images from the original testing set, and the second row shows the reconstructed images.

Finally, the last group of attributes are called gender-independent. These attributes are fairly consistently labeled and are not much dependent on gender expression. This group of attributes include Black_Hair, Bangs, Blond_Hair, Brown_Hair, Chubby, Wearing_Earrings, Bags_Under_Eyes, Eyeglasses, Gray_Hair, High_C Mouth_Slightly_Open, Narrow_Eyes, Smiling, and Wearing_Hat.

### 3.2 Attribute classifiers

For attribute classifiers, we use ResNet-18 pre-trained on ImageNet, in which the last layer is replaced with a layer of size one, along with a Sigmoid activation for binary classification. We train all models for 5 epochs with batch sizes of 128. We use the Stochastic Gradient Descent optimizer with a learning rate of $1 \mathrm{e}-3$ and momentum of 0.9 . We use a step learning rate decay with step size of 7 and factor of 0.1 . After training, we will have 26 classifiers that receive an image and perform a binary classification on their respective attribute.

### 3.3 Implementation details

As shown in Figure 1, a ResNet-18 network is used to accompany the U-net to produce predictions for Male and Attractive. Prior to training the U-net, the ResNet-18 [⿴囗⿰丨丨] ] which is pre-trained on ImageNet, is modified by replacing its output layer with a layer of size two, outputing the probability of attractiveness and gender. The ResNet-18 is then trained for 5 epochs on the train set, with a batch size of 128 . We use the Stochastic Gradient Descent optimizer with a learning rate of $1 \mathrm{e}-3$ and momentum of 0.9 . We use a step learning rate decay with step size of 7 and factor of 0.1 . After the ResNet is trained and prepared, we train the U-net as described in Section 2 on the train set. The detailed architecture of the U-net is described in Supplementary Material. In our implementation of biased estimator of HSIC estimator in Equation 2, we use Gaussian RBF kernel function for $k(\cdot, \cdot)$ and $l(\cdot, \cdot)$. The training was conducted on a machine with two NVIDIA GeForce RTX 3090, and each training of the U-Net took 1 hour. When the training is complete, the U-net is ready to reconstruct images. Figure 3 shows six examples of how the U-net modifies the original images. We train our model for 5 epochs with an $\lambda=0.07$.

### 3.4 Comparison with baseline models

We compare our results with Ramaswamy et al.'s method, described in their paper 'Fair Attribute Classification through Latent Space De-biasing' [匹⿴囗 . Building on work by [⿴] which demonstrates a method to learn interpretable image modification directions, they develop an improved method by perturbing latent vector of a GAN, to produce training data that is balanced for each protected attribute. By augmenting the original dataset with the generated data, they train target classifiers on the augmented dataset, and show that these classifiers will be fair, with high accuracy. The second model that we compare our results with is explicit removal of biases from neural network embeddings, presented in [ $\square$ ]. The authors provide an algorithm to remove multiple sources of variation from the feature representation of a network. This is achieved by including secondary branches in a neural network with the aim to minimize a confusion loss, which in turn seeks to change the feature representation of data such that it becomes invariant to the spurious variations that are desired to be removed.

We implement Ramaswamy et al.'s method as follows: As mentioned in their paper, we used progressive GAN with 512-D latent space trained on the CelebA training set from the PyTorch GAN Zoo. We use 10,000 synthetic images and label the synthetic images with a ResNet-18 (modified by adding a fully connected layer with 1,000 neurons). Then we trained a linear SVM to learn the hyper-planes in the latent space as proposed in the original paper. We generate $\mathcal{X}_{\text {syn }}(160,000$ images) to generate a synthetic dataset which aims to de-bias Male from all 26 attributes one by one. Next, we train ResNet-18 classifiers on the new datasets consisting of augmenting $\mathcal{X}$ and $\mathcal{X}_{\text {syn }}$. We call this model as GANDeb. We use the implementation of [⿴囗⿰丨丨] with the uniform confusion loss $-(1 /|D|) \sum_{d} \log q_{d}$ provided in [지].

### 3.5 Evaluation metrics

In evaluating the results of our model with the baseline models, three metrics are used. To capture the accuracy of the classifiers, we measure the average precision. This metric combines precision and recall at every position and computes the average. A higher average precision (AP) is desired. To measure fairness, there are multiple metrics proposed in the literature [⿴囗⿰丨㇄] . Among the most commonly used metrics is demographic parity (DP). This metric captures the disparity of receiving a positive decision among different protected groups $(|P(\hat{Y}=1 \mid S=0)-P(\hat{Y}=1 \mid S=1)|)$. A smaller DP shows a fairer classification and is desired. Finally for our last fairness measure, we follow [四] and [ㅁ] ] and use difference in equality of opportunity (DEO), i.e. the absolute difference between the true positive rates for both gender expressions $(|T P R(S=0)-T P R(S=1)|)$. A smaller DEO is desired.

### 3.6 Results

All the values reported in this section, are evaluated on the same test set. Prior to comparing the results of our method with the comparison models, to assess the original training data, the performance of baseline classifiers being trained on the original train set, and tested on the test set is presented. The AP, DP, and DEO values of classifiers trained on the original training set is shown in Table 1 under Baseline. Looking into Baseline values, the AP of classifiers for gender-independent category of attributes is higher than gender-dependent category, and the AP of inconsistent category is less than the other two categories. As expected, DP and DEO for gender-dependent category of attributes is higher than the other two categories.
![](https://cdn.mathpix.com/cropped/2024_06_04_0e50f4443c4eaaea4d3bg-09.jpg?height=266&width=1276&top_left_y=90&top_left_x=82)

Figure 4: Exploring the trade-off between accuracy and fairness by incremental increasing of parameter $\lambda$. Each data point is the average over three trainings, with standard deviation of the three trainings shown as confidence intervals.

In Table 1, we compare our model with GAN Debiasing (GanDeb) [⿴囗⿰丨㇄], Adversarial debiasing (AdvDb) presented in [⿴]⿰丨丨], and the Baseline on the original data. Looking into the average precision scores, the results show that GanDeb is slightly performing better than Ours. This is anticipated, since half of the training data for GanDeb consists of the original images, and therefore a higher average precision is expected. AdvDb on the other hand is performing poorly in terms of average precision, with average precision scores far away from other models.

Looking into demographic parity scores, the results show that GanDeb falls behind the other two models in two out of three attribute categories. While Ours is performing better for gender dependent and gender independent attribute categories. Looking into the third fairness measure, difference in equality of opportunity, AdvDb and ours are performing better than GanDeb in all three categories of attributes. Ours beats AdvDb for inconsistent attributes category, AdvDb beats Ours in gender dependent category, and AdvDb slightly beats Ours for gender independent category of attributes. In summary, Ours is close to GanDeb in terms of maintaining high average precision scores, which means higher accuracy of prediction, while beating GanDeb in terms of fairness metrics. Also, while AdvDb performance in terms of fairness enforcement is better than ours in 3 out of 6 cases, it falls behind significantly in terms of average precision.

To explore the trade-off between fairness and precision, we perform the following experiment: $\lambda$ was increased between $[0.01,0.15]$ in steps of 0.01 , and for each value of $\lambda$, the model was trained three times, each time for 1 epoch. Figure 4 shows how AP, DEO, and DP change. The results show that by increasing $\lambda$, precision decreases while fairness measures improve.

|  | AP $\uparrow$ |  |  | DP $\downarrow$ |  |  | DEO $\downarrow$ |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Incons. | G-dep | G-indep | Incons. | G-dep | G-indep | Incons. | G-dep | G-indep |
| Baseline | 0.667 | 0.79 | 0.843 | 0.147 | 0.255 | 0.137 | 0.186 | 0.243 | 0.163 |
| GanDeb | 0.641 | 0.763 | 0.831 | 0.106 | 0.233 | 0.119 | 0.158 | 0.24 | 0.142 |
| AdvDb | 0.243 | 0.333 | 0.218 | 0.091 | 0.169 | 0.121 | 0.136 | 0.149 | 0.098 |
| Ours | 0.618 | 0.732 | 0.839 | 0.097 | 0.146 | 0.118 | 0.124 | 0.172 | 0.114 |

Table 1: Comparing the results of our model with Baseline, GAN debiasing (GanDeb), and Adversarial debiasing (AdvDb). Showing AP (Average Precision, higher the better), DP (Demographic Parity, lower the better), and DEO (Difference in Equality of Opportunity, lower the better) values for each attribute category. Each number is the average over all attributes within that specific attribute category.

![](https://cdn.mathpix.com/cropped/2024_06_04_0e50f4443c4eaaea4d3bg-10.jpg?height=524&width=782&top_left_y=88&top_left_x=296)

Figure 5: Displaying the relationship between an attribute's statistical dependence on Attractive attribute, and the extent to which the model modifies that attribute. Blue bars show the HSIC between each attribute with Attractive attribute in the original data. Red bars show the absolute difference in demographic parity of each attribute's classifier, acting on original images and transformed images, respectively.

### 3.7 Interpretation and the effect on other attributes

In this section, we aim to display the correspondence between an attribute's relationship with Attractive attribute, and the extent to which the model modifies that attribute. To do so, for each attribute, we record two values, namely HSIC value between that attribute and the Attractive attribute, and the change in demographic parity. To calculate the change in demographic parity, we first calculate the demographic parity of the classifier for that specific attribute, when the classifier classifies the original testing set images (similar to Baseline in previous tables, but for each attribute separately). We then calculate the demographic parity of the classifier for that specific attribute, when the classifier receives the modified training images Ours(5,0.07). We then subtract the two values, to get the change in demographic parity for that specific attribute. Figure 5 presents the results, with the red bars showing the change in demographic parity for each attribute, and the blue bars showing the statistical dependence measured by HSIC, between each attribute with Attractive attribute, in the original training data. The results show that the absolute change in demographic parity is positively correlated with that attribute's statistical dependence with the attribute Attractive, with a Pearson correlation coefficient of 0.757 . For instance, we observe large changes in demographic parity for attributes such as Young, Big_Nose, Pointy_Nose, Oval_Face, and Arched_Eyebrows, as they are typically associated with being attractive, and therefore reflected in the CelebA dataset labels.

## 4 Conclusions

We proposed an image reconstruction process to mitigate bias against a protected attribute. The model's performance was evaluated on CelebA dataset and compared with an augmentation

![](https://cdn.mathpix.com/cropped/2024_06_04_0e50f4443c4eaaea4d3bg-10.jpg?height=48&width=1289&top_left_y=1928&top_left_x=51)
bias while maintaining high precision for classifiers. An interesting aspect of the results is that although we only explicitly train the U-net to remove dependence between the target attribute (Attractive) and the protected attribute (Male), classifiers related to many other
attributes, most of which have a statistical dependency with the target attribute, become 'fairer'. An advantage of the proposed model is that it does not rely on modifying downstream classifiers, and rather includes only modifying the input data, hence making it suitable to be deployed in an automated machine learning pipeline more easily and with lower cost. As a potential future direction, we intend to consider the problem in a situation where multiple protected attributes are present, and attributes are non-binary. We also intend to apply similar methodology on other data types such as tabular data.

## References

[1] Mohsan Alvi, Andrew Zisserman, and Christoffer Nellåker. Turning a blind eye: Explicit removal of biases and variation from deep neural network embeddings. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pages 0-0, 2018.

[2] Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Conference on fairness, accountability and transparency, pages 77-91. PMLR, 2018.

[3] Kristy Choi, Aditya Grover, Trisha Singh, Rui Shu, and Stefano Ermon. Fair generative modeling via weak supervision. In International Conference on Machine Learning, pages 1887-1898. PMLR, 2020.

[4] Alexandra Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big data, 5(2):153-163, 2017.

[5] Emily Denton, Ben Hutchinson, Margaret Mitchell, Timnit Gebru, and Andrew Zaldivar. Image counterfactual sensitivity analysis for detecting unintended bias. arXiv preprint arXiv:1906.06439, 2019.

[6] Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. Measuring statistical dependence with hilbert-schmidt norms. In International conference on algorithmic learning theory, pages 63-77. Springer, 2005.

[7] Arthur Gretton, Ralf Herbrich, Alexander Smola, Olivier Bousquet, Bernhard Schölkopf, et al. Kernel methods for measuring independence. 2005.

[8] Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems, 29:3315-3323, 2016.

[9] Roger A Horn and Charles R Johnson. Matrix analysis. Cambridge university press, 2012.

[10] Sunhee Hwang, Sungho Park, Dohyung Kim, Mirae Do, and Hyeran Byun. Fairfacegan: Fairness-aware facial image-to-image translation. arXiv preprint arXiv:2012.00282, 2020 .

[11] Faisal Kamiran and Toon Calders. Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems, 33(1):1-33, 2012.

[12] Anja Lambrecht and Catherine Tucker. Algorithmic bias? an empirical study of apparent gender-based discrimination in the display of stem career ads. Management science, 65 (7):2966-2981, 2019.

[13] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.

[14] Vishnu Suresh Lokhande, Aditya Kumar Akash, Sathya N Ravi, and Vikas Singh. Fairalm: Augmented lagrangian method for training fair models with little regret. In European Conference on Computer Vision, pages 365-381. Springer, 2020.

[15] Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6):1-35, 2021.

[16] Dana Pessach and Erez Shmueli. Algorithmic fairness. arXiv preprint arXiv:2001.09784, 2020.

[17] Novi Quadrianto, Viktoriia Sharmanska, and Oliver Thomas. Discovering fair representations in the data domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8227-8236, 2019.

[18] Amirarsalan Rajabi and Ozlem Ozmen Garibay. Tabfairgan: Fair tabular data generation with generative adversarial networks. arXiv preprint arXiv:2109.00666, 2021.

[19] Vikram V Ramaswamy, Sunnie SY Kim, and Olga Russakovsky. Fair attribute classification through latent space de-biasing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9301-9310, 2021.

[20] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234-241. Springer, 2015.

[21] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115 (3): $211-252,2015$.

[22] Prasanna Sattigeri, Samuel C Hoffman, Vijil Chenthamarakshan, and Kush R Varshney. Fairness gan: Generating datasets with fairness properties using a generative adversarial network. IBM Journal of Research and Development, 63(4/5):3-1, 2019.

[23] Viktoriia Sharmanska, Lisa Anne Hendricks, Trevor Darrell, and Novi Quadrianto. Contrastive examples for addressing the tyranny of the majority. arXiv preprint $\operatorname{arXiv:2004.06524,2020.}$

[24] Tatiana Tommasi, Novi Patricia, Barbara Caputo, and Tinne Tuytelaars. A deeper look at dataset bias. In Domain adaptation in computer vision applications, pages 37-55. Springer, 2017.

[25] Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In CVPR 2011, pages 1521-1528. IEEE, 2011.

[26] Angelina Wang, Arvind Narayanan, and Olga Russakovsky. Revise: A tool for measuring and mitigating bias in visual datasets. In European Conference on Computer Vision, pages 733-751. Springer, 2020.

[27] Mei Wang and Weihong Deng. Mitigate bias in face recognition using skewness-aware reinforcement learning. arXiv preprint arXiv:1911.10692, 2019.

[28] Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, and Yaohai Huang. Racial faces in the wild: Reducing racial bias by information maximization adaptation network. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages $692-702,2019$.

[29] Tianlu Wang, Jieyu Zhao, Mark Yatskar, Kai-Wei Chang, and Vicente Ordonez. Balanced datasets are not enough: Estimating and mitigating gender bias in deep image representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5310-5319, 2019.

[30] Tongxin Wang, Zhengming Ding, Wei Shao, Haixu Tang, and Kun Huang. Towards fair cross-domain adaptation via generative learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 454-463, 2021.

[31] Zeyu Wang, Klint Qinami, Ioannis Christos Karakozis, Kyle Genova, Prem Nair, Kenji Hata, and Olga Russakovsky. Towards fairness in visual recognition: Effective strategies for bias mitigation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8919-8928, 2020.

[32] Xingkun Xu, Yuge Huang, Pengcheng Shen, Shaoxin Li, Jilin Li, Feiyue Huang, Yong Li, and Zhen Cui. Consistent instance false positive improves fairness in face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 578-586, 2021.

[33] Kaiyu Yang, Klint Qinami, Li Fei-Fei, Jia Deng, and Olga Russakovsky. Towards fairer datasets: Filtering and balancing the distribution of the people subtree in the imagenet hierarchy. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, pages 547-558, 2020.

[34] Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adversarial learning. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pages 335-340, 2018.

[35] Quanshi Zhang, Wenguan Wang, and Song-Chun Zhu. Examining cnn representations with respect to dataset bias. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

[36] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Men also like shopping: Reducing gender bias amplification using corpus-level constraints. arXiv preprint arXiv:1707.09457, 2017.


[^0]:    © 2021. The copyright of this document resides with its authors.

    It may be distributed unchanged freely in print or electronic forms.

</end of paper 1>


<paper 2>
# Debiasing Methods for Fairer Neural Models in Vision and Language Research: A Survey 


#### Abstract

OTAVIO PARRAGA*, MARTIN D. MORE*, CHRISTIAN M. OLIVEIRA*, NATHAN S. GAVENSKI*, LUCAS S. KUPSSINSKÜ, ADILSON MEDRONHA, LUIS V. MOURA, GABRIEL S. SIMÕES, and RODRIGO C. BARROS, Machine Learning Theory and Applications (MALTA) Lab, PUCRS, Brazil

Despite being responsible for state-of-the-art results in several computer vision and natural language processing tasks, neural networks have faced harsh criticism due to some of their current shortcomings. One of them is that neural networks are correlation machines prone to model biases within the data instead of focusing on actual useful causal relationships. This problem is particularly serious in application domains affected by aspects such as race, gender, and age. To prevent models from incurring on unfair decision-making, the AI community has concentrated efforts in correcting algorithmic biases, giving rise to the research area now widely known as fairness in $A I$. In this survey paper, we provide an in-depth overview of the main debiasing methods for fairness-aware neural networks in the context of vision and language research. We propose a novel taxonomy to better organize the literature on debiasing methods for fairness, and we discuss the current challenges, trends, and important future work directions for the interested researcher and practitioner.


CCS Concepts: $\cdot$ Computing methodologies $\rightarrow$ Natural language processing; Computer vision; Neural networks.

Additional Key Words and Phrases: fairness, neural networks, bias mitigation, computer vision, natural language processing

ACM Reference Format:

Parraga et al.. 2022. Debiasing Methods for Fairer Neural Models in Vision and Language Research: A Survey. ACM Comput. Surv. 00, 0, Article 000 ( 2022), 35 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 INTRODUCTION

Deep Learning is a subfield of Machine Learning (ML) that leverages the capabilities of artificial neural networks to automatically learn from data. These networks are fully-differentiable computational graphs optimized via gradient descent to learn representations from raw data [12], currently being the most efficient and effective data-oriented strategy to perform several Computer Vision (CV) and Natural Language Processing (NLP) tasks.

Despite producing exciting results, neural models have faced harsh criticism due to some of their current shortcomings. One of the main criticisms is that, since neural networks are correlation-based approaches, models often learn the influence of confounding factors that are present in the data instead of causal relationships [2]. This problem is exacerbated in application domains affected by sensitive (or protected) features, such as demographic information. For instance, race, gender, and age may confound the training process because the distribution of labels is often skewed, even if the intrinsic properties of interest are not related to them. Thus, the presence of[^0]confounding factors may result in models that are biased towards specific distributions, potentially exhibiting a severe drop in performance in unseen data or prioritizing certain subgroups within a distribution.

Given the widespread usage of neural models, one must consider the impact of their respective automated decisions. Applications such as product recommendations and automatic game-playing are considered to be low-stakes since biased behaviors do not significantly impact underrepresented groups in society. However, automated decisions in dating, hiring [20, 37], and loan management [185] software present considerably higher stakes, given that they may influence and perpetuate unfair economic and social disparities between groups.

To address and regulate algorithmic usage in high-stakes decision making, several governmental entities have proposed the creation of stronger laws requiring more transparency in automated decision-making. For example, the European Union GDPR [38] states that people should have the right to obtain an explanation of the decision reached by automated systems. Indeed, explainability seems to play a crucial role towards achieving trustworthy AI, i.e., systems that are lawful, technically robust, bias-resilient, and ethically adherent. Since neural networks are black-box models that require external tools to extract explanations [74], researchers and practitioners often ignore such tools, allowing the design and optimization of unfair models.

To prevent learning models from perpetuating the biases present in the data and producing unfair decisions in automated decision-making, the Artificial Intelligence (AI) community has concentrated efforts in correcting algorithmic biases, giving rise to the research area now widely known as fairness in AI. When considering the process of automated decision-making, the term fairness refers to the "absence of any prejudice or favoritism toward an individual or a group based on their inherent or acquired characteristics" [135]. Fairness in AI is a relatively new research area. Initially, fairness papers were mostly submitted to workshops focused on data privacy, with occasional papers appearing in main proceedings. Starting in 2014, we see the appearance of several workshops and conferences specialized in fairness, accountability, and transparency in machine learning. Notable examples include FAT/ML (2014-2018), AIES (2018-present), FAccT (formerly FAT*, 2018-present), and FATES (2019-present). As the research field matured, we see a noticeable increase in the number of fairness papers that appeared in the main proceedings of renowned AI and Machine Learning conferences, such as AAAI, NeurIPS, ICML, ICLR, and many others. These conferences now also occasionally host tutorials and workshops dedicated mainly for fairness-related approaches in data-driven learning, and some conferences, such as NeurIPS, are experimenting with implementing fairness awareness protocols, such as including a "Broader Impact" section, covering the ethical aspects of the algorithms being proposed, and proposing a list of best practices for responsible machine learning research $[145,146]$. This historical summary shows that the research community in general is starting to focus not only on improving the performance of algorithms, but also in creating fairer ones.

Given the relevancy and recent proliferation of fairness concerns in automated decision-making, our goal in this survey is to provide an overview of the progress and current state of the art in neural approaches for fairness and bias mitigation in AI. We focus on vision and language research since these data modalities and their intersection encompass the majority of neural networks research.

Several surveys of fairness in $\mathrm{AI}$ already exist, each covering different aspects of the research area. For instance, the work of Mehrabi et al. [135] focuses on fairness in ML while extensively detailing fairness definitions and types of biases, but offers a quick analysis of debiasing methods in ML in general. A similar content and structure can be found in $[29,53]$, where a broad analysis is performed for different ML tasks and algorithms. The work of Le Quy et al. [113] goes in a different direction by focusing exclusively on analyzing tabular datasets and their usages for bias mitigation. Tian et al. [195], in turn, cover fairness exclusively for image data. Finally, there are also several survey papers that extensively cover bias and fairness solely within the scope of NLP [7, 18, 44, 64, 134], be it only regarding pre-trained language models (LMs) [44, 134], or exclusively for deep learning [64].

In contrast with the aforementioned studies, this work focuses on an in-depth analysis of neural-based methods for debiasing in the context of the main unstructured data types, namely visual and textual data and their intersection (i.e., multimodal tasks). By focusing on debiasing approaches, we offer a new taxonomy for properly
categorizing those methods while following the detailed definitions of fairness and bias proposed by Mehrabi et al. [135], which are now well-accepted and widely used by the research community.

This work surveys 95 debiasing methods exclusively in the context of neural networks for vision and language. The only work to survey a similar amount of methods is the one by Caton and Haas [29], which reviews a total of 86 debiasing methods, though across all ML research. Still, there is only one paper in the intersection of this work and [29], which is the work from Edwards and Storkey [55]. The remaining survey papers on fairness review a much smaller set of methods for debiasing. Dunkelau and Leuschel [53] survey 24 debiasing methods, and once again its intersection with our work is only a single paper, [55]. Finally, the outstanding survey by Mehrabi et al. [135] only presents 13 debiasing methods, none of them falling under our criterion for acceptance, namely being a method for making neural models fairer in the context of vision and language research.

## 2 SCOPE AND ORGANIZATION

In this section, we detail the scope and organization of this paper. First, we contextualize fairness and its relationship with bias within neural network research in Section 3. Bias is an overused word in ML research and has several different meanings according to the context where it is used. Learning requires methods to incorporate inductive biases, which are preferences towards choosing/modeling certain solutions instead of others. In the context of neural networks, biases are also attribute-free parameters that shift the model according to prior information. The correlation-based approach implemented in most ML methods, neural networks being no exception, may capture relationships that lead to biased solutions, i.e., a model whose behavior may be undesired because of spurious correlations that were captured during training. In this paper, when we talk about biases we are not talking about the attribute-free parameters in neural networks. We are also rarely talking about inductive bias, since algorithmic biases are seldom the cause of fairness problems, though that may also happen.

Hence, most of the time we mention the word bias we mean unintended behavior resulting from correlationbased processing that ignores further context not explicit in the data. In practice, we assume that exploiting correlations is the own nature of ML algorithms, and it is the combination of biased samples and correlation identification that generates the problem of models that are biased towards undesired behavior.

Fairness is also a term with many proposed definitions. We follow the specialized literature and situate (un)fairness as a direct consequence of capturing spurious correlations during training, as long as those correlations result in the "favoritism toward an individual or a group based on their inherent or acquired characteristics" [135]. By situating problems with fairness as a direct consequence of existing correlation-based biases, we show that most bias-mitigating strategies can be used to improve the fairness of automatic systems, and therefore we establish the link between the areas of bias mitigation and fairness awareness in Section 3.

Next, we comment on the main metrics and evaluation measures for assessing the level of fairness of a neural model in the context of those applications in Section 4. We review both application-specific and general-purpose measures, pointing to their proper use, applicability, and known limitations. The core of this survey, however, is the critical analysis and discussion of several debiasing methods for neural models in the context of image and language processing, which we present in Section 5. Computer vision and natural language processing are arguably the two most important research areas in AI nowadays, given the amount of content produced that falls into these categories and their intersection. For instance, in 2021 approximately 1.4 trillion digital photographs were taken worldwide ${ }^{1}$ while Twitter users posted approximately 200 billion tweets ${ }^{2}$.

Finally, we list the current fairness challenges in neural models, highlighting trends and important future research directions in Section 6. We also specifically envision the challenges of fairness awareness in the so-called Foundation Models [23], which are extremely-large neural networks trained over massive amounts of data. The[^1]challenge of addressing fairness unawareness in models that were trained self-supervisedly is probably one of the main factors that prevent such models from being fully open-sourced, given that this category of neural models has the potential of being applied and adapted to several tasks and biases may be perpetuated or even potentialized if not considered and properly addressed, resulting in unfair decision-making.

## 3 BIAS AND FAIRNESS

The word bias is used in ML in many distinct contexts. Mitchell [139] defines bias as "any basis for choosing one generalization over another, other than strict consistency with the instances". This definition encompasses biases that are inherent to the learning task and are unavoidable, the so-called inductive biases. In convolutional neural networks, for instance, the implemented inductive bias explores the fact that the input data often has spatial coherence and that higher level features are translation and rotation invariant. Another common use of the word bias is to refer to the parameters of the neural network that are free w.r.t. the input features. In this survey, we will focus on biases that are not mandatory in the learning process and that can often skew the results of the model in undesired ways, with the possibility of causing social harm.

Since artificial neural networks are essentially data-driven methods, data is the main source of unwanted and potentially avoidable bias. In an ideal scenario, data should be complete, correct, and representative of a population of interest. However, it is often the case that the sampling scheme, the measurement, or the representation of the data is biased towards a group of subjects. Olteanu et al. [151] call this scenario data bias.

It is also common to use datasets that follow a given distribution and expect them to generalize to another (perhaps slightly distinct) one. A facial attribute recognition system could be trained using the CelebA dataset [123], but when the model is deployed to recognize facial features in the wild we can expect a performance drop given that the target population is not a group of celebrities. This type of bias arises with the distinction of distributions from training and production settings. Even assuming that the dataset is representative of the population that we are interested in, we are still prone to create biased models. Take the scenario of NLP systems that are trained in very large corpora of text. The word embeddings created by these systems incorporate stereotypes that are found in the text, such as associating the word engineering more frequently with men than women [190].

Many studies searched for ways to categorize bias following different taxonomies, each based on a different system and set of assumptions. The work of Mehrabi et al. [135] divides biases based on the relationship among three key entities that constantly interact with each other: user, algorithm, and data. Each entity will aggregate a group of possible biases that may occur in interactions within the ML life cycle, which may affect another entity in the process. The relationship between data and learning algorithm clearly allows for the identification of outcome disparities and biases. Historical, social, cultural, and economic factors, as well as cognitive human biases, may affect all the process of collecting data, biasing samples and entire datasets without being easily detected. The relationship between algorithm and user can also be a source for potential biases to arise, specially when the algorithm is responsible for guiding the human behaviour to specific patterns [49].

In the study of Suresh and Guttag [190], the types of biases are defined according to the ML life cycle. It is established that unintended biases can be inserted within the data (be it a problem of data generation, representation, or measurement) or by model building and implementation (due to the learning process, evaluation procedure, aggregation of models, or the deployment in an environment where the concepts that were modelled do not apply). Although this definition is more straightforward than the one in [135], it lacks the capability of differentiating that data biases such as content production and aggregation are created by user-data interaction and data-algorithm interaction, respectively.

We can also view the bias phenomena as an origin and consequence framework [179]. More focused on NLP applications, Shah et al. [179] define four origins for biases in source data: over-amplification, semantic, selection, and label; and two consequences in the outcomes: outcome disparity and error disparity. By using a standard

![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-05.jpg?height=805&width=1043&top_left_y=351&top_left_x=541)

Fig. 1. Network of terms found in the abstracts of the surveyed papers. The term bias and its relationship with other terms are highlighted. Created with VOSViewer [157].

supervised pipeline of NLP, the authors attribute each bias origin to a different step, from the embedding phase where the semantics are condensed into a dense vectorized representation to the possibility of over-amplification due to the combination of the learning algorithm and the data itself or the under-representative selection of instances. The consequences are defined by two disparities obtained when analyzing the output distributions.

Mitigating bias in neural models is not a trivial task. Despite the discussion regarding adopted taxonomies and a strict definition of bias [151, 183], there is a consensus that biased models can cause societal harm and that researchers and practitioners must be vigilant and employ tools to detect and mitigate this problem. A point should be made that, although humans have an unmatched capacity to learn from data and to generalize to different contexts, we too are prone to biases and to being unfair. Nonetheless, the problem is exacerbated by neural models since they can work on a larger scale and reinforce negative feedback loops in society [109].

When biases in the learned models are detrimental to specific groups, often defined by sensitive attributes, we have models that are unfair. The question of fairness is specially relevant when automated decisions exacerbate prejudicial behavior against socially-vulnerable groups or when it promotes favoritism towards a specific demography, gender, ethnicity, or race [49]. Model fairness leverages moral questions about how we collect, validate, and use data, especially now with models getting larger and datasets reaching massive sizes [25]. We cannot disregard group representation nor oversee undesired outcomes in trained models anymore.

When researchers and practitioners focus solely on traditional performance metrics such as accuracy, a trained model can propagate stereotypes and biases present in the training data. While general fairness is a well-defined concept, the definition of what is the best measure to capture fairness is still subject of debate [29]. One should be careful since distinct definitions of fairness derived from the use of different metrics have mutual interactions [104], e.g., when optimizing for group fairness it is possible to make the situation worse for individual fairness. These scenarios are presented with greater detail in Section 4.

We illustrate in Figure 1 the landscape of fairness research in neural networks for vision and language with a network of terms found in the abstracts of all surveyed papers. In such visualization, the node area is proportional to the number of occurrences of the term within the abstracts. Note that bias and data are the most used terms, considering that the main source of bias in ML applications is the data and their interaction with users and algorithms. The modalities that we are focusing on, nodes $c v$ and $n l p$, as well as their intersection are also commonly present in the metadata or through proxy terms such as language, image and embedding. We can also see that the term bias appears in association with attributes such as gender, and race, which are indeed often associated with subgroups or individuals that are harmed by unfair decisions. Other terms that relate to sensitive attributes are demographic, ethnicity, and social bias.

Fairness is not achieved solely by guaranteeing the same outcome for distinct groups. Mehrabi et al. [135] present the case where a seemingly unfair difference in annual income between men and women in the UCI Adult dataset can be explained in terms of working hours. In that dataset, men on average work more hours per week than women. Working hours is thus a legitimate attribute that explains the difference in annual income and should not be considered an issue in the context of fairness.

To promote fairness, a special subset of available features are regarded as sensitive or protected. They help define subgroups that are considered underprivileged in society. An example of protected attribute is gender, and there are several examples of ML systems presenting error rates skewed towards a particular gender category [26, 75, 101]. The definition of the the subset of attributes that are sensitive is not an entirely technical issue. Although there is certain agreement that ethnicity should be protected in some applications, it could be the case that other attributes such as zip code serve as a proxy for ethnicity and should also be protected. Furthermore, despite being a protected attribute, ethnicity matters in certain applications, e.g., medical diagnosis where genetic predisposition is a determining factor. In practice, any attribute that society perceives as a potential defining factor for aggregating underprivileged people is a potential sensitive attribute.

Fairness can also be stratified regarding groups, subgroups, or individuals. Fairness in groups can be defined through the Demographic Parity concept, which states that the likelihood of an outcome should be the same for a person in or out of the protected group. On the other hand, there is also the Counterfactual Fairness definition, which states that inferences over a single individual should be kept unaltered in a hypothetical counterfactual world where the same individual belongs to a distinct group. Note that satisfying every definition of fairness at the same time is not feasible in real-world applications [110], and we exemplify that in Section 4.

Issues of model fairness are often intertwined with biases in the ML pipeline. Datasets such as IJB-A [103] and Adience [114] do not have proper representation of gender and skin color [26]. Models trained in biased datasets will often produce worse results for individuals in underrepresented groups. They may, for instance, misclassify Black women with a higher probability than lighter-skinned men. In that situation, the unfairness of the model can be traced back to its origin in a biased dataset.

When the underprivileged group is defined by a combination of sensitive attributes, and the dynamics of individuals in this group is considerably distinct than when considering one sensitive attribute at a time, we have intersectional biases [26]. Unfairness caused by intersectional biases are harder to detect and to prevent because of intricate relationships among the attributes [75, 101].

Some applications of ML models are considered to be of low stakes, having a limited impact in society well-fare and errors are considered cheap. On the other hand, ML models are also being used in significant higher-stake applications, where a single error in inference is expensive and can negatively impact individuals and groups. Examples of these applications are facial recognition systems [26], criminal assessment [87], and occupation classification [43]. Fairness is of the upmost importance in high-stake applications. ML models should be subject to scrutiny as they are part of an entire ecosystem that influences different social groups, and can perpetuate harmful concepts and stereotypes when left unchecked.

When optimizing for model fairness, it is necessary to keep in mind that the level of data aggregation can also be a source of confusion. This type of problem is defined as Simpson's Paradox [19], which is the phenomenon in which the same dataset can lead to opposite trends by changing how the data is aggregated, always happening when the aggregated data hide conditional variables. Regarding neural networks, most features are learned by searching for correlations in data when trying to optimize an objective function, so the undesired lack of fairness may be a product of seeking good performance, creating a kind of performance-accuracy trade-off.

Most research in fairness cover the ML field without focusing on deep learning [29, 135, 190]. As a consequence, lack of fairness in multimodal tasks (e.g., text-to-image systems) are less understood than in more traditional classification tasks. Another specificity in neural-network research is the recent rise of Foundational Models (FMs) [23]. Biases in those models are challenging to identify, and the own nature of FMs as unfinished models that need to be further adapted makes them a perfect fit for the problem of bias propagation and exacerbation within many downstream tasks. We further discuss this challenge in Section 6.

As neural models are being increasingly deployed to real-world applications, the interest on fairness grows both in academia and industry. While there are countries that are starting to regulate some aspects of fairness in ML models and automated decision-making [16], there is much to be done to achieve a unified framework that the research community can agree on and that practitioners can apply when developing fairer applications. Among the requirements for the fairness research to provide practical impact in society, there should be effective methods that allow biases to be mitigated and underprivileged groups to be protected without a significant compromise in terms of efficiency and effectiveness. We categorize and review those methods in Section 5, while also discussing their advantages and shortcomings.

## 4 METRICS

This section describes the most relevant metrics to measure bias and fairness of neural models in CV and NLP, though all metrics presented in Section 4.1 apply to any type of classification problem.

### 4.1 Metrics for Classification

Classification is a traditional problem in machine learning and one of the most common tasks in computer vision. During the training process, given a dataset $D=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{N}$ the algorithm will learn how to correlate a set of inputs $X$ to a set of corresponding target variables $Y$. The most common metric to evaluate classification models is accuracy, the rate of correctly-classified objects. Consider the confusion matrix of a binary classification problem, comprised of True Positives (Negatives) as the amount of objects correctly classified from the positive (negative) class, and False Positives (Negatives) as the amount of misclassified objects from the positive (negative) class. With these same terms, we can derive three other widely-used metrics: precision, recall, and F1-Score.

Even though those metrics are well-establish for evaluating the general performance of a learning model, they are not well-suited for fairness evaluation. For example, consider a model trained on a dataset containing two groups, $A$ and $B$, where $A$ is privileged in society while $B$ is marginalized. Societal biases have affected model performance, leading it to perform better on $A$ than on $B$. Increasing accuracy using optimization techniques does not necessarily improve fairness. It is possible the model gets even better at classifying $A$ and slightly worse on $B$, leading to an even unfairer model. Similarly, precision, recall, and F1-score alone cannot guarantee model fairness, nor properly evaluate its nuances. With this in mind, fairness-specific metrics have been proposed for classification models. These metrics have been mainly divided into two categories: group fairness, which requires the average output of different demographic groups to be similar [54]; and individual fairness, which requires that individuals who are similar in respect to a specific task have similar probability distributions on classification outcomes, regardless of the demographic group they are part of [54, 61].

4.1.1 Group Fairness. In group fairness, examples are grouped according to a particular protected attribute, and statistics about model predictions are calculated for each group and compared across them [50]. Next, we describe the most used group fairness metrics.

Demographic/Statistical Parity states that the average algorithmic decision should be similar across different groups: $p\left(\hat{y}=c_{k} \mid z=p\right) \sim p\left(\hat{y}=c_{k} \mid z=u\right)$, where $\hat{y}=c_{k}$ is a predicted class and $z$ refers to a protected attribute such as race, gender, and age, in which $p$ and $u$ indicate privileged and underprivileged groups, respectively. This metric does not depend on ground-truth labels, so it is especially useful when this information is unavailable. When we are aware that historical biases may have skewed the data, demographic parity may be used in conjunction with a bias mitigation strategy to verify whether the model has learned those biases.

Equality of Opportunity states that the true positive rate for individuals who qualify for the desirable/positive outcome $(y=1)$ should be similar across different groups: $p(\hat{y}=1 \mid z=p, y=1) \sim p(\hat{y}=1 \mid z=u, y=1)$ where $y$ is the ground truth label. This metric considers that different groups may have different distributions, which leads to one of its main criticisms: it does not consider the effect of discrimination given the protected attributes.

Equality of Odds states that both the true positive rate for individuals who qualify for the desirable/positive outcome and the false positive rate for individuals who do not qualify for the desirable/positive outcome should be similar across different groups. The true positive rate is computed as the equality of opportunity, while the false positive rate is computed as $p(\hat{y}=1 \mid z=p, y=0) \sim p(\hat{y}=1 \mid z=u, y=0)$. Equality of odds is a more restrictive variation of equality of opportunity, but the same criticisms of the former still hold for this metric.

Conditional Statistical Parity states that different groups should have similar probability of being assigned to a positive outcome given that the individuals satisfy a set of legitimate factors $L$ : $p(\hat{y}=1 \mid L=1, z=p) \sim$ $p(\hat{y}=1 \mid L=1, z=u)$. The set of factors $L$ is task-dependent and should be modeled accordingly.

Treatment Equality states that the ratio of false negatives and false positives should be similar across different groups: $(F N / F P \mid y=1) \sim(F N / F P \mid y=0)$. This metric may be used as a policy lever while optimizing for other metrics. If a specific group has a higher ratio of false negatives in order to satisfy another metric, then it is actually treating these groups differently.

Overall Accuracy Equality states that accuracy should be similar for different groups: $((T P+T N) /(T P+$ $F P+T N+F N) \mid z=p) \sim((T P+T N) /(T P+F P+T N+F N) \mid z=u)$. This metric thus implies that true negatives are as desirable as true positives, being quite uncommonly-used for that reason.

Predictive Parity is similar to Overall Accuracy Equality, except that it considers that different groups should have similar precision instead of accuracy: $((T P) /(T P+F P) \mid z=p) \sim((T P) /(T P+F P) \mid z=u)$. Mathematically, a model with equal precision for different groups will also have equal false discovery rate: $((F P) /(T P+F P) \mid z=p) \sim((F P) /(T P+F P) \mid z=u)$.

Right for the Right Reasons: Many classification errors occur due to the model looking at the wrong evidence. It can happen based on any contextual evidence, e.g., a multimodal model ignoring the input image when asked "What color is the banana?" since the most likely answer is "yellow". To compute this measurement quantitatively for image-based models, Hendricks et al. [79] rely on two visual explanation techniques: GradCAM [177] and saliency maps generated by occluding image regions in a sliding window fashion. In their work, they use the MS-COCO ground-truth person segmentation to evaluate whether the Grad-CAM from the model overlaps with the correct areas of the image when classifying a given class. As this metric evaluates how the model gets its prediction, it is aligned with the current demand for more explainable AI.

4.1.2 Individual Fairness. Group fairness mainly ignores all information of the objects except for protected attribute $z$. That strategy might hide unfairness, and we exemplify a scenario in which that happens as follows. Suppose a model assigns a positive score to the same fraction of male and female applicants. However, assume male applicants were chosen randomly whereas female applicants were chosen based on reference attribute values.

Demographic parity will state that the model is actually fair, despite the discrepancy on how the applications were processed based on gender. The notion of individual fairness arises to deal with such a discrepancy.

Fairness through Awareness states that similar individuals should be treated similarly, where subject similarity is a task-specific metric. Consider two similar individuals, where the only major difference between them is that one is male and the other female. Since they are similar, individual fairness states that they must follow the same distribution, generating the same output. This strategy has already proven to be more efficient than Fairness through Unawareness, which relied on the assumption that an algorithm would be fair if none of the sensitive attributes were explicitly used in the predictive process. Unfortunately, building a similarity score considering the different types of attributes is the biggest obstacle for putting this metric into practice.

Counterfactual fairness states that an individual and its counterfactual copy whose only difference is the protected attribute value should have the same outcome. While fairness through awareness finds similar individuals through task-specific measures, this metric generates synthetic copies of a counterpart instance (e.g., female if the original instance is male in a gender-fair evaluation). It also takes into account that several features might be dependant on the protected attribute and, therefore, should also be altered accordingly.

4.1.3 Critical Analysis of the Classification Metrics. Suppose we want to train a model to admit students to a given University with 30 openings, and we have two groups of people: $A$, which comprises 70 students, and $B$, with 30 . In addition, assume $A$ is privileged in society whereas $B$ is part of a marginalized group. Historical biases led students of group $B$ to have, on average, lower grades than group $A$, even though both groups contain students with variable performance in terms of grades. Finally, assume 40 students from group $A$ and 12 from group $B$ reach the desirable grade level to be admitted to this University. We discuss two different learning models and how the proposed metrics measure what is happening. This discussion applies not only to neural networks but to any learning model and classification task in machine learning.

Approach \#1. Assume a model was trained to achieve similar acceptance rates for both groups. It selects 21 students from group $A$ and 9 from group B. This model satisfies the Demographic Parity criterion, since the percentage of accepted students was the same for both groups: $30 \%$. However, if the selection were made randomly in either $A$ or $B$ instead of considering the grades, this metric would still evaluate this model as fair. On the other hand, this model does not satisfy the Equality of Opportunity criterion: even if the selection process were made entirely based on each student's grades, the rate of acceptance considering only the students with desirable grades was higher for group B (75\%) than for group A (53\%).

Approach \#2. Assume a second model was trained to achieve a similar acceptance rate in both groups, but this time only considering the students with desirable grades. As a result, 23 and 7 students were accepted from groups $A$ and $B$, respectively. In this scenario, Equality of Opportunity is satisfied ( $58 \%$ for both groups), while Demographic Parity is not ( $33 \%$ for group $A$ and $23 \%$ for group B). The problem here is that the historical biases that led to the disproportion regarding the grades for each group are not taken into account. Therefore, this strategy alone will perpetuate historical biases that are currently present in society.

It is easy to see that one cannot satisfy all fairness metrics without a perfectly-balanced model that is fully aware of all possible sources of bias and prejudices. It is reasonable to assume such a thing will not be achievable any time soon, mainly because, by definition, machine learning deals with ill-posed problems and incomplete data. Counterfactual Fairness carries excellent insight on how we can evaluate fairness while also considering historical biases. However, this metric requires a deeper analysis of the data in order to find the dependencies of features regarding both the sensitive attribute(s) and the outcomes.

### 4.2 Metrics for Image Generation

Image generation is the task of creating images given an input, be it a user-guided signal (including another image) or simply random noise. Recent advances in the field have made it possible to create an image of almost
anything given a simple description [166]. Generative models are trained on colossal amounts of unbalanced data, which carry historical and societal biases. It is possible to measure whether a generative model is creating a similar number of samples for different groups by using a classifier on a large set of generated samples, verifying which group they belong to and then comparing these proportions. Choi et al. [35] used this strategy with a binary gender classifier to evaluate their face generation model trained on the CelebA dataset, which has a higher proportion of females. Cho et al. [34] made a more comprehensive evaluation concerning gender and race using CLIP [162] and human annotation. That analysis used images generated through professional and political prompts to check whether the model correlated the protected attributes with specific roles. Overall, this approach requires a classifier with good performance in the respective groups so manual labeling is not necessary.

### 4.3 Metrics for Language Modeling

Language modeling (LM) is a base task for most NLP models. By predicting the next token in a sentence, neural networks can learn to manipulate language, discovering multiple meanings of words that vary contextually. A common approach is to train an LM and later adapt it to a downstream task, following the idea of first teaching the model about the nature of the language and then fine-tuning its knowledge to a specific task. Instead of training a new model, it is possible to use only the generated embeddings (word or context-level) that encapsulate the semantics in a dense vector and then employ it in a different processing pipeline.

Many metrics use the LM output to estimate biases via the probabilities. Another alternative is to use the dense vectorized representation to measure bias, which can be used both with contextual and word embeddings. Besides those alternatives, NLP tasks can be used to measure bias when using specific evaluation datasets. Metrics tailor-made for LMs are described next.

Direct Bias (DB) [21] is a specific gender-bias metric which defines bias as a projection onto a gender subspace. To measure DB, we first need to identify $N$ words that are gender-neutral. Given those words and the gender direction learned $g$, we define the $\mathrm{DB}$ of a model as:

$$
\begin{equation*}
D B=\frac{1}{|N|} \sum_{w \in N}|\cos (\vec{w}, g)|^{c} \tag{1}
\end{equation*}
$$

where $\vec{w}$ is the embedding vector of $w$ and $c$ is a user-defined parameter that determines how strict DB will be. With $c=0, D B=0$ only when there is no overlap of $\vec{w}$ and $g$. With $c=1$ we have a more gradual bias measurement albeit with a small error margin.

Word Embedding Association Test (WEAT) [27] measures bias through the permutation of two sets of target words $X$ and $Y$ (e.g., male-dominated professions like engineers and female-dominated professions like nurses) and two sets of attribute words $A$ and $B$ (e.g., \{man, male,...\} and \{woman, female,...\}). In a scenario without biases, there would be no difference between the relative similarity of the two sets of target words and the two sets of attribute words. WEAT measures this difference in similarity to determine whether a given word is biased.

Co-occurrence Metrics [24] uses word co-occurrence for measuring bias in generated text. In a modelgenerated corpora, we can analyze the number of times that each word appear next to specific terms, i.e., how often a language model will connect professions, sentiments and areas of knowledge with protected groups.

Sentence Embedding Association Test (SEAT) [131] comes as the natural adaptation of WEAT [27] but for contextualized word embeddings. Since WEAT only tested associations among word embeddings, it lost utility for recent models that are contextual-based (e.g., based on Transformers). The authors adapt it to make use of sentence templates, e.g., "[He/She] is a [MASK]". The models then generate contextual embeddings with these templates and the cosine similarity is computed between two sets of attributes. Several other WEAT and SEAT variations have been proposed since, though with a similar usage principle.

Discovery of correlations (DisCo) [207] uses a template with two slots, e.g., "X likes to [MASK]", where $X$ is a word based on a specific set of words planned to trigger possible biases, and the [MASK] token is replaced with
the model prediction. It compares the predictions for the words used in the template to compute biases. The final result is the average of different predictions for the sets of words.

Log probability bias score [108] uses the prior probability of the model when predicting a specific [MASK] token to normalize the resulting probability of a word that appears in a given context. By doing so one can surpass limitations from metrics based on pre-trained models that do not consider the prior probability of a given word when generating the recommendation. A limitation of this method is that it only works for models trained with masked language modeling, where the priors can be extracted by masking more than one token in the sentence, e.g., "The [MASK] is a [MASK]", and analyzing them individually.

Context association test (CAT) [141] is a metric proposed in conjunction with the StereoSet dataset. That dataset comprises sentences to be completed (model has to fill a blank token or select a continuation for the sentence). The completion option, for all cases, contains stereotype, anti-stereotype, and meaningless options. The objective of the evaluation is to measure how many times a model would choose a meaningful sentence over a meaningless one and how many times it would choose a stereotyped option instead of an anti-stereotype.

CrowS-Pairs [143] is a pseudo-log-likelihood metric based on the perplexity measure of all tokens conditioned on the stereotypical tokens. Similar to the previous one, this metric also comes with a dataset. The templates for its usage follow a similar approach to StereoSet, with stereotyped and anti-stereotyped versions.

All Unmasked Likelihood (AUL) [92] is an extension of CrowS-Pairs. It leverages not only the masked tokens in the sentence to measure biases but also all unmasked tokens and multiple correct predictions. Kaneko and Bollegala [92] also proposes AULA, a variation that takes into account the attention weights.

4.3.1 Critical Analysis of the Language Modeling Metrics. LMs are assets of high relevance in NLP, especially after the adoption of pre-trained LMs as the basis for adaptation in downstream tasks. This reuse approach highlights the need of a consistent and robust fairness evaluation due to its high-spread potential. Metrics that rely on model probabilities are easier to interpret, especially when compared with the embedding-based metrics. The latter are divided into two categories: word embeddings and contextual embeddings, which share quite a few similarities. However, the distinctions become stronger as we look into metrics such as WEAT and SEAT, where we need to create specific scenarios to measure biases, generating a considerable level of human interference when evaluating the final embeddings. The metrics presented here use either probabilities or embeddings and can be called intrinsic metrics. However, they are not the only way to evaluate LMs. We can evaluate fairness through downstream tasks using task-specific metrics. Neural machine translation, coreference resolution, and language generation are examples where we can use specific datasets and their respective evaluation protocols to analyze fairness levels. Given the number of toxic terms in a text generated by a model, the quality of translations or the correlation of terms with protected attributes in complex and sensitive contexts may give a clearer picture of real-world problems that may occur.

### 4.4 Task-Agnostic Metrics

The following metrics do not rely on any specific task, i.e., they are of general purpose and can be easily computed when evaluating trained neural networks.

Bias Amplification [220] evaluates how much bias the model amplifies in a given dataset. For that, it measures the bias score $b\left(\hat{y}=c_{k}, z\right)$ as:

$$
\begin{equation*}
b\left(\hat{y}=c_{k}, z\right)=\frac{c\left(\hat{y}=c_{k}, z\right)}{\sum_{z^{\prime} \in Z} c\left(\hat{y}=c_{k}, z^{\prime}\right)} \tag{2}
\end{equation*}
$$

where $c\left(\hat{y}=c_{k}, z\right)$ is the amount of times the model outputs $\hat{y}=c_{k}$ taking into account protected attribute $z$, and $Z$ is the set of protected attributes.

The premise is that the evaluation set is identically distributed to the training set, and therefore if $\hat{y}=c_{k}$ positively correlates with $z$, and if $\tilde{b}\left(\hat{y}=c_{k}, z\right)$ (evaluation set) is larger than $b^{*}\left(\hat{y}=c_{k}, z\right)$ (training set), one can assume that the model has amplified that specific bias.

Assume we are measuring biases in a VQA application, and that the bias scores measured in a specific model are $b^{*}(\hat{y}=$ cooking, $z=$ woman $)=.66$ and $\tilde{b}(\hat{y}=$ cooking, $z=$ woman $)=.84$, respectively, when asked What is the person doing? In that case, one can assume that the model amplified the bias of cooking toward woman. The authors do this for each class in order to obtain the mean bias amplification, defined as:

$$
\begin{equation*}
\frac{1}{|K|} \sum_{z} \sum_{k \in\left\{k \in K \mid b^{*}\left(\hat{y}=c_{k}, z\right)>1 /\|Z\|\right\}} \tilde{b}\left(\hat{y}=c_{k}, z\right)-b^{*}\left(\hat{y}=c_{k}, z\right) \tag{3}
\end{equation*}
$$

However, the premise that the bias distribution will be the same from training to evaluation/test set might not hold and could misrepresent the bias-resilient capability of the model. We observed only one other work [83] that makes use of this metric outside the scope of the VQA(-CP) datasets.

KL-Divergence The Kullback-Leibler divergence score quantifies how much a given probability distribution differs from another. The KL-divergence between distributions $P$ and $Q, K L(P \| Q)$ is calculated as:

$$
\begin{equation*}
K L(P \| Q)=\sum_{i=1}^{N} P\left(x_{i}\right) \cdot \log \frac{P\left(x_{i}\right)}{Q\left(x_{i}\right)} \tag{4}
\end{equation*}
$$

The divergence is an optimal metric to measure how different protected attributes diverge in a task where the feature should not correlate to the problem, e.g., a curriculum vitae system should not take gender into consideration [155]. Hence, if the distribution between genders significantly diverges, we can conclude that the model is unfair towards a gender. Nevertheless, KL-Divergence is not a distance metric between two distributions since it is not symmetric. Therefore, $K L(P \| Q) \neq K L(Q \| P)$, requiring coupling KL-Divergence with another metric for bias mitigation measurement.

## 5 DEBIASING METHODS FOR FAIRNESS-AWARE NEURAL NETWORKS

This section organizes and discusses the existing literature for debiasing neural-network models in vision and language research. While the existing literature reviews divide the debiasing approaches into three categories, namely pre-processing, in-processing, and post-processing [29, 135], we understand that these categories are insufficient to organize all existing methods reviewed in this survey in a precise fashion. Therefore, we propose a new taxonomy that properly categorizes all debiasing methods, and we present it in Figure 2.

We call distributional all strategies that modify the dataset distribution prior to training. That includes sampling strategies that increase the amount of data examples artificially. In addition, we divide methods that focus on optimization via training into two categories:

(i) One-Step-Training, which includes fair models that are generated for a particular task via a single optimization procedure;

(ii) Two-Step-Training, where a new training phase has to be performed to fix an existing biased model, i.e., making it fairer.

Finally, we call inferential those strategies that address the problem of fairness based on the model outputs, i.e., that discover and remove social biases without requiring further weight optimization or dataset manipulation.

Our taxonomy is more precise than previously-proposed ones in that it better differentiates methods that remove biases during the training process of the downstream task (one-step-training) from those that optimize a pre-trained model. Note that when we say pre-trained, we mean trained in the final (goal) downstream task, not pre-trained in large general datasets (say Imagenet [172]). Hence, the difference between one-step-training and

![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-13.jpg?height=569&width=1277&top_left_y=347&top_left_x=424)

Fig. 2. Taxonomy for organizing the literature on debiasing methods for neural networks in vision and language research.

two-step-training is whether there is a previous biased model that works for the task of interest or not. Methods that fall in the one-step-training category may very well have been pre-trained on general-purpose datasets. This updated taxonomy reflects the fast advances we are witnessing in both industry and research communities with the adoption of the so-called Foundation Models [23].

Finally, we should mention that the categories of the taxonomy are not mutually exclusive. Let us assume a given method uses an adversarial strategy coupled with a regularization term for debiasing a model pre-trained on Imagenet. We categorize it as belonging to both adversarial and optimization categories within one-step-training.

Table 1 organizes all debiasing methods surveyed in this paper according to the proposed taxonomy and also the respective domain (vision, language, or multimodal).

### 5.1 Distributional

Bias-mitigation methods of the category distributional target at changing the data distribution to better represent socially-vulnerable groups or to change data objects of the dataset to remove unwanted biases. They rely

Table 1. Debiasing methods organized according to the proposed taxonomy.

| Category | Sub-category | Vision | Language | Multimodal |
| :---: | :---: | :---: | :---: | :---: |
| Distributional | Heuristic <br> Generative <br> Resampling | $[45]$ <br> $[32,62,147,165,217]$ <br> $[28,117,192]$ | $[126,186]$ <br> $[160]$ <br> - | - <br> - <br> $[214]$ |
| One-Step-Training | Adversarial <br> Causal Approaches <br> Disentanglement | $[55,115,205]$ <br> $[42,93,98]$ <br> $[41,98,153,193,213]$ | $[58,60,67,129,150,167,209]$ <br> $[76]$ <br> $[51]$ | $[14,214]$ <br> $[215]$ <br> - |
|  | Optimization | $[4,5,73,79,130,192,204]$ | $[24,31,51,67,90,118,182,207]$ <br> $[96,97,115,127,133,154,161,178]$ | $[99,203,220]$ |
| Two-Step-Training | Distillation <br> Fair-Modules <br> Fine-Tuning | $[88,116,132]$ <br> $[94,116]$ <br> - | $[76]$ <br> $[33,57,112,163,210]$ <br> $[56,68,122,211]$ | - <br> $[152,191,214]$ <br> $[14]$ |
| Inferential | Prompting | - | $[66,176,181,184,202]$ | $[137]$ |
|  | Vector-Space Manipulation | $[174]$ | $[3,21,22,47,52,89,119,120,197]$ <br> $[48,77,91,106,107,111,189,216]$ | $[203]$ |

on creating a modified dataset to improve fairness or applying systems and rules to remove or compensate underrepresented groups within the data. The rationale is that the neural network will be trained using a more representative data distribution, thus leading to a fairer model.

Distributional mitigation methods are stratified into 3 groups according to the changes made to the data:

(i) Heuristic methods modify objects of the dataset according to predefined algorithmic rules;

(ii) Generative methods aim to create or modify data objects using generative models;

(iii) Resampling methods are based on under or over-sampling the dataset to mitigate the under-representation of individuals in protected groups.

5.1.1 Heuristic. Unbalanced training sets may lead to models that reproduce certain disparities present within the data. For that, one might change the dataset by modifying, adding, or removing objects, somehow making it represent all protected groups/individuals equally. When making these modifications, one should consider that not all applications follow the same distributions, and hence are prone to the same set of rules. For instance, new sentences must follow pre-existing grammatical, lexical, and syntactic rules when one is modifying NLP datasets.

Debiasing through heuristic manipulation is a viable strategy for NLP tasks. Examples include using dictionaries or semantic and syntactic rules to replace words or add informational labels/tokens [126, 186]. More than just replacing words, those strategies seek to create new information, not only by changing one or two words but by adapting the entire surrounding information, avoiding the creation of nonsensical sentences. For computer vision, a possible heuristic approach is to apply pre-determined transformations in images to augment the dataset. Some methods can leverage this process to augment only instances underrepresented in the dataset [45].

While heuristic methods to change data distribution may be an interesting alternative for bias mitigation, these changes are only practical in scenarios where the application domain comprises explicit and well-known rules. One example is language modeling, where templates of sentences can be used to expand the original data. In other domains, however, heuristics do not scale and fail when the rules are not easy to define manually.

5.1.2 Generative. In domains like computer vision, where no explicit set of constraints is specified to delimit data distribution, more complex approaches may be necessary to adjust the level of fairness in a dataset. Finding a simple strategy or rule that can be applied to all instances is hardly possible, requiring the creation of deep networks capable of doing such modifications [165].

Generative Adversarial Networks (GANs) can be an option when looking for ways to increase a dataset with synthetic data $[62,147,165,217]$, since they can create high-quality new images when properly trained, balancing the dataset with regard to its potential misrepresentation and allowing the training of a new model over both original and synthetic data. Figure 3 gives an overview of this strategy of augmenting (or balancing) a dataset with a generative neural network.
![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-14.jpg?height=318&width=334&top_left_y=1822&top_left_x=411)

Unbalanced Dataset
![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-14.jpg?height=316&width=568&top_left_y=1820&top_left_x=1142)

Balanced Dataset

Fig. 3. General overview of generative methods for debiasing.

In the language context, Qian et al. [160] propose a sequence-to-sequence model that generates perturbations regarding the protected variables in dataset instances.

Systematic biases exist in many datasets, which is also the case with face datasets. As annotators usually consider female faces as being happy more often than men's, Chen and Joo [32] propose using Action Units (AU) that aim to measure facial expressions to address this problem objectively. The proposed framework does not need to modify labels, like other methodologies. Using AUs, the model classifies two similar AU samples as similar people besides their gender. To penalize for unfairness, they utilize the triplet loss function as a regularizer. Note that this method can be applied to other datasets (not only facial expressions), and one just needs to have other objective measures, like body key points. That work was the first to show the systematic effect of annotations in datasets for computer vision, and it is even more visible in in-the-wild datasets.

5.1.3 Resampling. Resampling debiasing methods only modify the data distribution by rearranging the existing dataset objects. While possibly lowering the number of objects available for training, no artificial data is generated. The most common resampling approaches balance the distribution of the training dataset according to a specific attribute such as gender or race [214] while others focus on discovering how to resample the original dataset to create a fairer distribution that is better suited to represent all protected groups in the data [117, 192].

A more sophisticated strategy is to use more than one dataset to build a more representative one. This process can use different sampling approaches to select the best configuration of available objects to better represent the diversity of protected attributes, thus providing a fairer dataset for training [28].

### 5.2 One-Step-Training

Although manipulating the data is a straightforward strategy to have more diverse or balanced data, it is often not enough to produce fair neural models. Some challenges that remain unsolved by distributional approaches are: i) (deep) neural networks are data hungry, which means undersampling strategies could reduce the data up to the point training becomes unfeasible; ii) even with data that perfectly represents the population distribution, undesirable characteristics such as stereotypes and prejudice that are present in society may arise [206]. For solving those problems, one may need to resort to additional strategies that happen either during training or during inference, which are the focus of this section.

We call one-step learning debiasing methods that act during the main training procedure. These methods are further divided into four distinct groups according to the debiasing strategy that is used:

(i) Adversarial methods make use of the adversarial framework or of adversarial examples to teach the model not to resort to undesired biases;

(ii) Causal methods use causal graphs and counterfactual examples to teach the model which relationships are relevant within the data;

(iii) Disentanglement methods separate the features in the latent space to manipulate them independently;

(iv) Optimization methods include loss function adaptions, addition of regularization terms, and other modifications for improving weight optimization.

5.2.1 Adversarial. It is a known fact that adversarial examples can deceive deep learning models. Neural networks may be fooled by intended perturbed images that do not contain human-perceivable changes [71]. Notwithstanding, adversarial examples can be included in the training dataset to create more robust models. That setup uses two models: one is trained to solve the task objective, whereas the other focuses on creating adversarial examples that try to confound the first model [70]. The objective is to teach the main network not to use the protected attribute to do the task $[60,115,150,209,212,218]$. To force the model not to rely on protected attributes, it is possible to erase or mask them from the data source [205], create new data, or even attack the deep representation generated by the model $[55,58,129]$, which is less interpretable. Figure 4 depicts this general idea.

![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-16.jpg?height=485&width=1570&top_left_y=362&top_left_x=272)

Fig. 4. General overview of adversarial methods for debiasing.

For the information retrieval domain, adversarial training is being used to create embeddings that are bad predictors for the protected attribute, but that are capable of accurately predicting the target variable [67, 167, 214].

Note that all adversarial methods rely on annotations of protected variables or groups in the dataset. They allow the models to focus on legitimate attributes while ignoring the protected ones. However, the need for annotation is a downside of these approaches. Fortunately, the annotations are not needed for inference time.

5.2.2 Causal. We can leverage any knowledge on the causal-effect relationship between protected attributes and outcomes to try and fix model unfairness. For identifying these relationships one can make use of causal graphs or counterfactual samples during training. Causal approaches are a popular choice to mitigate general biases, such as language priors in VQA [1, 30, 105, 149], and we explore how they can be adapted to solve fairness issues.

By using causal graphs we can force the model to learn valuable relationships between features, intermediate representations, and predictions. Yang et al. [215] propose a specific attention module in the network that is responsible for discovering such associations, and then use them for bias mitigation. Another benefit of building a causal graph is that it can later be used to explain predictions, an important artefact for explainable AI.

Creating counterfactual samples, on the other hand, allows the model to train over examples that are harder to predict, since they may not occur in the training distribution, improving the robustness of the model towards specific protected groups. We can employ linguistic rules to create new examples during training, forcing the model to learn with one group and its opposite (e.g., male and female genders) [76]. Other strategy is to disentangle visual representations in the latent space and use isolated features to create the counterfactual samples [98]. Additional work uses an auxiliary model to discover the causal graph and use discovered relations to create counterfactual samples using the protected attributes [42, 93].

Generating counterfactual samples is a good option to ensure that the model will see a larger number of distinct examples. However, such approaches may lead to creating extreme counterfactuals that do not represent valid instances from the dataset or the distribution they represent [100].

5.2.3 Disentanglement. During training, neural models create latent representations that represent automaticallylearned features extracted from the data objects. The difficulty of learning a task and the robustness and generalizability of a model are directly correlated with the quality of the respective learned latent representations. One way to increase such quality is via disentanglement. A disentangled representation is a representation where each learned factor corresponds to a single factor of variation in the data and is invariant to other factors of variation [12]. Disentangled representations offer many advantages, such as boosting predictive performance [124], increasing interpretability [81], and improving fairness [125].

![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-17.jpg?height=815&width=1477&top_left_y=346&top_left_x=259)

Fig. 5. General overview of the disentanglement approach for debiasing.

Learning disentangled representations means to break down features into new dimensions where information is split into independent spaces. This approach can be used to separate the underlying structure of some objects in their various parts in a more representative and interpretable way [80]. For instance, the shape of an object and its position in the image can be independently broken down. That isolation of features allows us to operate in specific details of the input data rather than modifying it entirely. StyleGAN [95] is an example of that, where it is possible to modify only the hair attributes of someone's face without further changes to the image.

That idea of splitting different data dimensions can also be explored for debiasing during training, inducing a model to learn only features that are invariant to the protected attributes. Figure 5 illustrates such an approach.

By decomposing the representation into different spaces and splitting the target attribute dimension from the protected attribute dimension, a model may learn not to correlate sensitive information to the real objective task, prioritizing only relevant and essential information.

This approach can be implemented as a new model or module responsible for the disentanglement [41] or as a regularization term that uses the labels to guide the division of features [193].

Xu et al. [213] show that disentangled latent representations may achieve superior performance in terms of demographic bias than simply adding the protected attributes in the classification layer in a typical fairnessthrough-awareness approach. In the study of Park et al. [153], a variational autoencoder (VAE) is employed to create a representation that disentangles the latent space into three distinct subspaces:

(i) The first one has information of the target variable and no protected attribute information;

(ii) The second contains both target and protected attribute information;

(iii) The third contains just protected attribute information.

Images often carry both protected and legitimate attributes that are hard to separate. Disentanglement learning is an alternative to achieving fairness through unawareness in computer vision Du et al. [51] propose to disentangle the latent space and to use only the subspace that spans the legitimate factor to perform the classification task.

Kim et al. [98] implement the concept of counterfactual fairness while requiring the counterfactual samples to generate disentangled representations of the attributes.

Disentanglement approaches are a good option for separating protected and non-protected features, making it possible for the model to consider only one group of attributes when doing a specific task. Unfortunately, to perform a task such as classification, the training phase must be divided into at least two steps: the first one to learn the disentangled representations and the second to learn the target task effectively.

5.2.4 Optimization. Optimizing a loss function is the training strategy of every neural network. Together with an optimization rule, they dictate how the weights must be learned for a specific task. By adding or modifying terms in a loss function, one can drastically change the solutions generated by a neural network. We categorize a method as being an optimization approach if it proposes changes in the optimization procedure during training towards increasing fairness, forcing it to follow a desired output distribution without modifying the input data [86].

Several studies propose new loss functions for penalizing or ensuring specific behaviors. Often these new loss terms are summed or used together with the objective loss (e.g., Cross-Entropy), ensuring that the model will learn the main task while also being subject to fairness constraints. One can use this strategy to penalize models that wrongfully predict the protected attribute [79]. It is also possible to re-calibrate the loss function according to the value of the protected attribute [5, 192] or to optimize for fairer data representation [24, 67, 73, 99, 154, 178].

With textual data, changes in loss function usually aim to remove bias from embedding representations. It is possible to manipulate the multi-dimensional space in which the embeddings are located to decouple them from non-desired sub-spaces [31, 90, 133, 161, 207].

Aside from proposing novel loss functions, it is possible to employ an algorithm to inject constraints during model training to ensure a fair distribution of predictions for the training data [115, 220]. Also, regularization terms can minimize the mutual information between feature embedding and bias [97]. Other possibility is to apply a norm clip based on data groups, which improves diversity in data generated by GANs [96].

Contrastive learning is also an optimization procedure capable of model debiasing. During training, the model faces examples from different classes, and it should classify them correctly while keeping examples from different classes apart in the embedding space. We can use this technique to increase fairness during training by picking both positive and negative examples conditioned on the protected attribute [127, 203] or by enforcing that distinct protected groups be distant in the embedding space [182].

Optimizing weight distributions for particular examples in order to penalize easier ones is also a common strategy for debiasing $[117,118,204]$. It is possible to reduce unfairness by maximizing ratios between losses in the reweighted dataset and the uncertainty in gold-standard labels. This strategy can come with a coefficient of variation as a data repair algorithm to curate fair data for a specific protected class and improve the actual positive rate for that class [4]. It has the advantage of working in both supervised and unsupervised learning.

Some studies aim to formulate the question of fairness as a multi-objective optimization problem. The work of Martinez et al. [130] defines each optimization objective as the protected group conditional risk. In such formulation, the goal is to find solutions at the Pareto frontier.

With fairness criteria tied to the training objective of the network, two challenges arise:

(i) There may be a demand for extra annotations in the data regarding the protected attributes to allow for computing novel losses;

(ii) The optimization modifications can incur in a trade-off between fairness and accuracy, which we must be aware of. This is further discussed in Section 6.

### 5.3 Two-Step-Training

It is customary (and even considered a best practice) to not always train neural networks from scratch. Instead, whenever possible, the weights of a model are set to the state of a previous training procedure, which may even

Who is playing tennis?

![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-19.jpg?height=336&width=919&top_left_y=404&top_left_x=576)

Fig. 6. General overview of the distillation approach for debiasing.

have happened using a different dataset. This typically accelerates training and often improves the performance and generalization capabilities of a model. Since this practice is frequently adopted due to its benefits, applying debiasing methods during the initial stages of model training may be unfeasible or undesirable, especially since this may imply "wasting" computational resources that were used in the pre-training phase. Thus, debiasing strategies that focus on adapting existing models become attractive to improve fairness. We have separated two-step training methods into three distinct groups, according to the debiasing strategy used:

(i) Distillation includes methods that train a new model using the teacher-student approach;

(ii) Fair Modules aggregates methods that add new modules to existing models to remove unfairness;

(iii) Fine-Tuning lists approaches to retrain a model without architectural changes.

5.3.1 Distillation. The core concept of distillation is to transfer knowledge between two models: the teacher and the student. The student network is optimized to match the teacher model's predictions in addition to a task cost function. The student and teacher models have similar architectures, but the student network has less capacity, i.e., less optimizable parameters. In this scenario, the distillation process creates a model with similar predictive capabilities while simultaneously reducing the amount of resources needed [175]. One of the benefits of this type of approach is increasing the generalization capabilities of models without requiring more annotated data [82].

In the scope of fair deep learning, one can use model distillation to produce fairer student models based on an unfair teacher. To that end, most studies add a specific fair-related loss [76, 116, 132] or regularization term [88] to the original distillation framework. Intuitively, the student model learns to emulate the teacher model's knowledge while being simultaneously instructed to not rely on protected attributes present in the dataset. Figure 6 illustrates the core ideas of this type of approach.

Despite its benefits, the distillation strategy also presents some drawbacks. For one, it is typically much more expensive than other available two-step training approaches. In scenarios where the intended teacher model's weights or (log-)likelihood outputs are not available, e.g., models behind APIs such as GPT-3 [25], it becomes impossible to train a student network. Finally, the debiasing loss may conflict with the distillation objective, which may result in generalization problems [187].

5.3.2 Fair Modules. Another useful group of debiasing techniques for pre-trained models involves adding new components, or modules, to an existing architecture, as depicted in Figure 7. By combining the original architecture with a new group of layers, we can transfer the responsibility of learning fairness objectives to the new modules, and potentially leave the original weights untouched.

One may use specialized modules to create additional representations that can improve the quality of embeddings, thus preventing the model from learning "shortcuts" that lead to biases [210]. Another potential approach is to include modules that detect or predict the presence of protected attributes, which is then used to modulate
![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-20.jpg?height=434&width=684&top_left_y=352&top_left_x=684)

Fig. 7. General overview of the fair-module strategy for debiasing.

the overall answer [94, 116, 152, 191]. Alternatively, some approaches add modules that are responsible for learning how to mitigate biases based on the inference of the original model [33, 57, 112]. Finally, Rajabi et al. [163] proposes to train an encoder-decoder module that processes inputs before the original classifier network to remove unwanted biases without changing the original model. Since the new module removes potentially protected attributes, they tend to not affect the final model's prediction, effectively removing their influence.

One of the main advantages of this strategy is that one can freeze the original model weights and prevent model degradation and catastrophic forgetting, a phenomenon where previously-learned concepts and tasks are replaced by new information [57]. However, depending on the original model capacity, these architectural modifications may not work as well as techniques that adapt the original model weights.

5.3.3 Fine-Tuning. Fine-tuning refers to adjusting some or all of the weights of a model, that has already been trained, for a new combination of task, dataset, and domain. Figure 8 shows the general framework of the fine-tuning strategy for debiasing neural models. In the context of fairness, we can view fine-tuning as a specialization procedure. That is, pre-training is responsible for teaching general concepts pertaining to a data modality (e.g., language modeling for NLP), while fine-tuning forces the model to shift focus to fair concepts.

Gira et al. [68] has shown that fine-tuning a GPT-2 model changing as little as $1 \%$ of the model's parameters using a curated dataset is enough to increase model fairness while retaining the knowledge acquired in the pre-training stage. Reinforcement learning algorithms may also be used to guide weight optimization and provide a better alternative than a simple loss or dataset modifications [56,122]. Wu et al. [211] hypothesizes that some network parameters may be correlated with biased predictions, and thus consider model pruning to be a potential solution. Paired with a specific dataset used as a guide, the authors prune a pre-trained model, removing weights responsible for such biased predictions while attempting to maintain a similar performance.

Differently from adding a new module to be responsible for debiasing, in fine-tuning approaches the driving factor is the dataset used to continue model training. In situations where a sufficiently large and complex dataset is not available, fixing the problems of a pre-trained model using fine-tuning approaches may not be possible. Additionally, debiasing via fine-tuning requires that the practitioner be aware of fairness problems that are already present in the pre-trained model (or if the model is affected by a specific bias of interest), which may imply a more thorough investigation via empirical analyses.

### 5.4 Inferential

Debiasing strategies that are applied during training or by changing the data distribution may be unfeasible in situations where the model is not available or when fine-tuning is a resource-intensive task. Large models such as
![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-21.jpg?height=404&width=1206&top_left_y=347&top_left_x=431)

Fig. 8. General overview of the fine-tuning strategy for debiasing.

GPT-3 [25] are not easily used by most practitioners due to hardware constraints. For instance, to instantiate an OPT [219] model, that has around 130 billion parameters, more than 300GB of VRAM are required, which makes it impractical to fine-tune for most use cases. Additionally, some large neural models are not open sourced and can only be accessed via commercial APIs, which makes unfeasible the application of the debiasing approaches presented so far. Despite their generalization capabilities, such large models are also prone to accruing societal biases [11]. Inferential methods are alternatives in such scenarios, because they intervene during inference time to make models fairer, leaving weights untouched. There are two possible alternatives to this approach:

(i) Prompting prepends or alters the model input with specific triggers that stimulate a bias-free result;

(ii) Vector-Space Manipulation manipulates the embedding space to remove undesired biases.

5.4.1 Prompting. When researching Foundation Models in NLP, Brown et al. [25] discovered that this category of large-scale models can perform few-shot learning when receiving specific instructions in their prompts [17, 25, 219]. In the context of FMs, prompts refer to the model input, and typically consist of tokens that are later converted to continuous representations (e.g., an embedding layer). This discovery started a separate field of study known as prompt engineering [168].

Besides the simple approach of limiting and filtering a model's prompt (e.g., by restricting the use of certain words) [137], recent studies have shown that language models are vulnerable when attacked with specific tokens in their prompts, a procedure known as triggers [202]. Triggers consist of tokens appended to the original prompt that lead to unexpected (or, in the case of fairness, desired) behaviors. One can use such induced behaviors to mitigate (or provoke) certain biases, which make triggering methods attractive to promote fairness in large models [184]. Recent studies [14, 66, 181] seek to discover specific tokens (discrete or continuous) to help debiasing models. Schick et al. [176] propose an unusual form of debiasing through prompting by using the model itself to identify and mitigate its own biases. In that study, the authors prepended templates such as "the following text discriminates against people because of their gender:" to the prompt and identified that this addition manipulates the distribution of words to lower the probability of discriminatory outputs.

This is a versatile and attractive approach since, with the correct triggers, it is possible to mitigate or even induce specific biases, and potentially deal with multiple protected groups. Figure 9 shows the general idea of this category of methods. A downside of prompting approaches is clearly exemplified in the context of NLP: triggers often add undesired contexts to sentence generation, since the model is conditioning its outputs on a token that may go against the initial context.

5.4.2 Vector-Space Manipulation. A common approach in NLP to learn the semantic meanings of words is to use word-embedding representations, which represent tokens as $d$-dimensional vectors [136, 156]. Compared to traditional NLP word representations (e.g., term frequency-inverse document frequency), word embeddings are a

![](https://cdn.mathpix.com/cropped/2024_06_04_4d36fbc85bbe35b153a3g-22.jpg?height=290&width=1528&top_left_y=348&top_left_x=274)

Fig. 9. General overview of the prompting strategy for debiasing.

far better alternative in the context of neural models, since it gives the optimization procedure the chance to learn better representations. The seminal work of Bolukbasi et al. [21] of fairness in word embeddings describes how such vectorized word representations encapsulate societal biases. Aimed with this knowledge, it became possible to use word-embedding manipulation techniques for bias mitigation in fairness-sensitive applications [21, 24]. Most studies of word embedding debiasing follow a similar strategy to the Hard-Debiasing method [21], where a "protected" subspace is identified and then words have their projection removed from that space. Other word embedding debiasing approaches use a wide variety of helper tools, such as clustering [52], a helper dictionary [91], causal inference [216], among others [3, 22, 47, 48, 60, 77, 89, 91, 106, 107, 111, 197].

With the rise of Transformer-based architectures [199], which use (self-)attention as their primary inductive biases, methods that manipulate embedding spaces had to be adapted since attention uses original word-embedding representations to create contextualized word representations. That is, the semantic meaning of a token depends on the entire sequence of tokens. A typical adaptation to work with contextual vector spaces is adding or complementing inputs with pre-defined sentence templates [108]. Liang et al. [119] use such templates to adapt the Hard-Debiasing approach to a contextual scenario. Other studies rely on strategies like iterative generation to mitigate biases [120, 189]. This type of method can be adapted to work on a wide range of pre-trained Transformer-based architectures, since they all rely on the attention mechanism.

Although vector-space manipulation is explored mainly in language applications, it is also possible to adapt the core ideas to CV. Salvador et al. [174] rely on clustering and embeddings to fix unfairness issues in the visual domain. The authors propose a clustering-based calibration method to ensure that similar images have similar scores according to which cluster they belong to. Also working with images, Wang et al. [203] remove the projection of CLIP embeddings from protected subspaces.

Much like other methods, vector-space manipulation approaches are not without drawbacks. For instance, in the context of Transformer models, biases have different effects in each layer, adding considerable complexity to mitigation strategies [46, 201]. Another difficulty is the necessity of a reference to guide the model to identify biased sub-spaces, which leads to the creation of repositories of biased and unbiased examples [21].

## 6 CHALLENGES AND FUTURE DIRECTIONS

Throughout this survey, we presented several challenges regarding how to define the concepts of bias and fairness (Section 3), how to evaluate ML models using individual or grouped notions of fairness (Section 4), and how to debias models to make them fairer (Section 5). After analyzing the aforementioned issues, we now focus on general challenges for fairness in neural models while also highlighting potential future directions of research. Specifically, we analyze (i) how to assess and present the fairness risks in neural models; (ii) how to identify who should be responsible for owning and dealing with fairness issues in neural models; (iii) how to deal with the fairness-accuracy trade-off; and (iv) current challenges and research directions for fairer Foundation Models.

### 6.1 Risk Awareness

Every machine learning model is a direct consequence of several design decisions, such as dataset collection and filtering, model architecture, optimization procedure, and hyperparameter selection. Since these design decisions have a direct influence not only on the performance of a model but also on its perceived fairness, it is essential to clearly communicate these when internally or publicly releasing a model to highlight its limitations and intended usage. That way, all relevant stakeholders can make informed decisions concerning the usage of specific technologies in the context where they operate. However, deciding how and when to communicate such design decisions is not a trivial matter. Thus, we now highlight recent advances and best practices to help ML researchers and practitioners create proper documentation to improve risk awareness.

6.1.1 Artifact Documentation. Several studies $[10,13,65,84,138]$ have attempted to standardize the communication of dataset and model design decisions in hopes of stimulating best practices in the creation of such artifacts, especially regarding sensitive topics such as fairness, privacy, and security. Datasheets for datasets [65] raises several questions (grouped into sections that roughly match the key stages of a dataset life cycle) to help dataset creators and consumers think about a dataset holistically, asking details on the motivation for creating the dataset, how the data was collected and aggregated, the recommended uses, and so on. Datasheets may help users to select more appropriate datasets for a specific task, increase transparency, mitigate unwanted societal biases, and even increase reproducibility. There are several alternative approaches for documenting datasets; two examples include the Dataset Nutrition Label [84] and Data Statements for NLP [10]. Critically, all of the aforementioned approaches highlight the importance of documenting datasets to inform end-users of fairness-related limitations since most biases are created or inherited in the data collection stage.

Regarding model documentation, Model Cards [138] are an excellent protocol to increase transparency concerning training and evaluation protocols across different bias and fairness metrics (especially among different values of protected attributes, both individually and combined), and clarifying the intended use cases of models to minimize their usage in inappropriate contexts. Model cards are especially important in research and open-source initiatives, since it leads to a more responsible and accountable model democratization process, which allows stakeholders to compare candidate models across not only traditional performance metrics (e.g., accuracy for classifiers) but also ethical and fairness considerations.

Often, artifact creators must also legally protect themselves from misuse. The Montreal Data License (MDL) [13] is a project that attempts to improve the taxonomy of data licensing to better suit machine learning and artificial intelligence. In the context of fairness, MDL provides optional restrictions regarding ethical considerations when granting rights to use and/or distribute a dataset, e.g., restricting usage in high-risk scenarios such as health-related fields or military applications.

6.1.2 Research Documentation. Several mechanisms exist (e.g., institutional review boards, conference program committees, and funding bodies) that provide the academic community with means to prevent unethical research. However, there are few implemented protocols concerning fair research in mainstream deep learning conferences, and deep learning papers often do not discuss such limitations. Since conference papers tend to be prioritized over journal publications in computer science (and deep learning especially), conference organizers should strive to improve fairness awareness in conference publications.

Some deep learning conferences have recently included protocols to help researchers communicate fairness constraints. In 2020, NeurIPS required that all papers include a "broader impact statement" covering the ethical and fairness aspects of research in their camera-ready versions [144] Additionally, the conference incorporated a new ethics review process where technical reviewers could flag papers for ethical concerns, which a pool of ethics reviewers would later analyze. These initiatives had a mixed reception from the academic community due to their perceived political nature. Several studies have analyzed the results of this experiment $[6,142,159]$
and arrived at the following conclusions: (i) although $10 \%$ of all papers opted out of writing the broader impact statement, this occurred mainly in theoretical subareas. Most authors took the opportunity to reflect on their work rather than stating that the statement was not applicable. (ii) the average statement length was 168 words and the distribution presents a long tail -the longest statement had 4000 words. The areas of CV and NLP differ in average statement length (166 and 223, respectively). (iii) authors focused more on positive societal impacts rather than negative ones, which intuitively goes against the main purpose of the experiment.

The lack of standardization and clear guidance regarding the expected contents of a broader impact statement and misaligned incentives for researchers may explain the results of this experiment. However, since this was the first time that broader impact statements were mandatory, it is hard to judge the idea's true potential and how researchers would behave in the following years. Following the feedback of the broader impact statement implementation, the NeurIPS conference opted to replace the statement with a "Paper Checklist" that provides researchers with a list of best practices for responsible machine learning research [145, 146]. This type of mechanism is an attempt to force researchers to think about a study's negative societal impacts before it is released, which may also increase risk awareness for ML practitioners who may use open-source code or attempt to reimplement algorithms.

6.1.3 Fairness Frameworks. One of the major goals of fairness research is the inclusion of fairness metrics and techniques into traditional ML pipelines to help mitigate biases. Considerable efforts have been made to create open-source fairness software toolkits that help ML practitioners audit datasets and models $[9,15,72$, $173,194,198,208]$. These toolkits implement visualization and interpretability techniques, fairness metrics, and state-of-the-art debiasing techniques, translating research into actionable procedures. However, there is a lack of application of such solutions in practice, especially in industry. According to Richardson and Gilbert [170], several factors explain this: (i) there are too many fairness metrics, and the differences between them are not clear to practitioners. Furthermore, major trade-offs exist between fairness metrics, and often it is mathematically impossible to optimize multiple fairness metrics. Choosing the "right" metric is delegated to practitioners, who are unfamiliar with the technical aspects of fairness research. (ii) there exists a disconnect between fairness research in academia and industry, which translates to a lack of applicability of procedures and metrics in industry settings. Additionally, fairness concepts are highly domain-dependant, differing substantially between domain applications, and yet most domains lack thorough and specific algorithmic bias guidance. (iii) frameworks often do not provide help with communicating fairness concerns and trade-offs to stakeholders, which inevitably reduces the adoption of frameworks due to organizational frictions.

Several suggestions have been proposed to mitigate these issues, and many involve improvements in communication between academics and practitioners. Friedler et al. [59] suggest that fairness experts clearly state the priorities of each fairness metric, while Verma and Rubin [200] suggest that researchers clarify which definitions are appropriate for which situations. Friedler et al. [59] argue that new fairness metrics should only be introduced if they behave fundamentally differently from existing metrics. Several practitioners have also requested domain-specific procedures and metrics, and that fairness experts create knowledge bases for each domain [85] Based on user feedback, fairness researchers should also consider creating taxonomies of potential harms and biases [39, 128], easy-to-digest summaries explaining biases and their potential sources [39, 63], guidelines for best practices throughout the ML pipeline [85], and tutorials exemplifying how to incorporate fairness [85, 128, 169]. We believe that organizations in industry settings also have a big part to play in ensuring the implementation of fairness protocols. Organizations should promote a culture of fairness awareness and incorporate fairness as a global objective (like security, privacy, and accessibility) [63, 128, 188]. Additionally, organizations should strive to provide practitioners with resources and fairness support teams that provide knowledge and actionable steps in fairness issues during all steps of the ML pipeline [128, 140].

6.1.4 Final Considerations. We believe it is the responsibility of data and model creators to communicate fairness risks. Artifact documentation may also be helpful for policy makers, investigative journalists, and individuals whose data are included in the datasets or who would be impacted by the deployment of such models. We recommend that the language used in such documents range from accessible to technical, allowing both laypersons and domain experts to understand the critical decisions made to create a specific artifact. However, none of the approaches presented in this section provide a complete solution for mitigating unwanted societal biases or potential risks accrued by the usage of any algorithmic approach. Society constantly changes, thus dataset and model creators will never anticipate every possible use of a particular artifact, and neither should they attempt to do so. Instead, they should focus on objective facts and actions taken during the artifact creation pipeline, which will help consumers, companies, and governmental entities decide whether the technology is appropriate for their context and society as a whole.

### 6.2 Risk Ownership

An important but often overlooked discussion is deciding who is responsible for preventing algorithmic misuse. Different scenarios require different levels of attention: systems that do not involve humans-in-the-loop (e.g., autonomous driving or automated medical diagnosis) significantly increase the impact of algorithmic biases. Sometimes, adjustments should be performed by the owners of the technology (not only developers but also organizations). In other cases, however, the user of an AI-based system must be responsible for correctly using a tool or adjusting outputs to fit the notions of bias and fairness within the context in which they operate. Ultimately, we believe that fairness risk ownership is tied to two concepts: how much control a user has over the model's outputs and whether the user is directly affected by model decisions. To illustrate this challenge and provide some guidance, we contrast two exemplar situations of model deployment, namely facial recognition and image generation, where we believe that the responsibility of dealing with unfairness falls to artifact owners and users, respectively.

6.2.1 Algorithm Owner. A facial recognition system extracts facial features to verify a person's identity and has a wide range of applications. Since facial recognition is one of the most popular biometric modalities due to its simple data collection process, several companies have invested in software-as-a-service facial recognition products and incorporated them into mainstream technological products, such as smartphones. However, some studies [26, $102,164]$ have analyzed the potential fairness pitfalls of neural facial recognition systems and concluded that they might discriminate based on protected attributes, such as race and gender, by performing significantly worse on specific demographics. This discrepancy in performance is worrisome, especially considering that one of the clients of such facial recognition systems includes governments and law enforcement. For instance, despite not directly determining the fate of an individual, such technology can be used to identify suspects in video surveillance footage, and erroneous misidentifications can have serious detrimental effects. In this scenario, the "victims" of facial recognition systems have absolutely no control of the system's output and can be directly impacted by its decisions. Thus, the ownership of fairness risks in this context must fall into the artifact's owners.

6.2.2 Algorithm User. Image generation is a research topic that can potentially disrupt several markets, such as art, design, fashion, retail, and many others. One example of an image generator is DALL-E [166], a text-to-image model created by OpenAI. Since DALL-E's training dataset was collected from the Internet, it was expected that the model would inherit societal biases. Regarding fairness, OpenAI implemented several measures to prevent algorithmic misuse [137] but, despite these efforts, the company was still heavily criticized, and DALL-E was described as a "harmful tool" for reproducing societal biases, such as the (lack of) association of certain ethnicities with certain roles in society (e.g., Black CEOs or female garbage collectors) while facilitating the creation of "deep fakes". OpenAI effectively positioned itself as the owner of the risk of someone misusing their product, and all
decisions of what constitutes an unfair generation became centralized at the company level. However, OpenAI did not have to own this risk since users can control the model's output via input prompts and are not directly harmed by generated images. Additionally, the model is indeed capable of generating diverse outputs, given the right prompts. A more straightforward solution would be to provide disclaimers regarding fairness limitations and instructions on how to construct prompts that generate "fairer" and more diverse images and let users decide how they want to steer image generation since different solutions result in different trade-offs.

### 6.3 Fairness-accuracy Trade-off

As explored in Section 4, fairness researchers have already proposed several quantitative fairness metrics, giving rise to optimized objectives alongside task-specific metrics. However, metrics usually either emphasize individual or group notions of fairness, and it is often mathematically impossible to optimize multiple fairness metrics $[8,36,158]$, which forces practitioners to select the most appropriate metric to optimize (which is context-dependant). Additionally, it has been empirically observed that a trade-off exists between fairness and task performance and that increasing fairness often results in lower overall performance. This gives rise to the challenge of analyzing the fairness-accuracy trade-off for a given scenario. Practitioners must be careful when determining how to measure model performance, since this can be done in several ways and the choice of performance measure(s) can disguise or create new ethical concerns. A reduction in accuracy may be the best outcome, especially if the difference in performance can be explained by algorithmic unfairness.

A potential solution to understand the trade-off characteristics between task and fairness metrics is to rely on advances in multi-task optimization literature to help neural models during optimization [40, 69, 196]. One example of such a technique is generating a Pareto frontier to determine the set of Pareto-efficient solutions for a specific combination of fairness and task-specific metrics [78, 121, 130, 148, 171, 180]. Pareto efficiency corresponds to a situation where the performance of a model regarding a specific criterion cannot be made better without reducing the performance on at least one individual metric. Another potential solution is to use techniques that increase fairness through proxies, such as disentanglement and causal inference, since they usually generalize better (although often at the cost of task performance).

### 6.4 Fairness in Foundation Models

A Foundation Model corresponds to "any model that is trained on broad data at scale and can be adapted to a wide range of downstream tasks" [23]. FMs were initially introduced in the context of NLP research but are rapidly causing a revolution in all areas of deep learning research and industry applications. The term "foundation" specifies the role of this category of models: a FM is incomplete by itself, but serves as the common basis from which many task-specific models are built via adaptation. Since such models have the potential of being adapted to several tasks, biases may be perpetuated or amplified if not adequately addressed, making fairness a fundamental research direction for FMs [23, 44, 134, 176].

Mitigating biases in FMs is not a trivial matter, especially due to their dataset and training regimes. The datasets that support the training of FMs contain hundreds of millions or even billions of instances. Thus applying pre-processing debiasing techniques is not an attractive option due to its potential impact on monetary cost and generalization capabilities. One-step-training debiasing techniques should also be avoided since it is impossible to optimize for all fairness concepts simultaneously, and FMs are used in several scenarios and contexts. For these reasons, the most promising research directions regarding bias mitigation in FMs are fine-tuning and prompting approaches $[14,66,68,176,181,184]$ (Sections 5.3.3 and 5.4.1). We also reiterate the importance of clearly communicating the risks of such technologies to prevent potential algorithmic misuse and highlight the importance of open-sourcing such models to accelerate the discovery of potential fairness issues.

## 7 CONCLUSION

In this survey paper, we have investigated debiasing methods targeting fairness-aware neural networks for language and vision research. We have contextualized fairness and its relationship with biases and their possible origins. We have presented the main metrics and evaluation measures for assessing the level of fairness provided by models for computer vision and natural language processing tasks, reviewing both application-specific and general-purpose measures, their proper use, applicability, and known limitations. Then, we have discussed, in depth, several debiasing methods for neural models under the perspective of a new taxonomy for the area, which is yet another contribution of this paper. We concluded with our thoughts on the most pressing fairness challenges in neural networks, calling attention for potential trends and future research directions.

We should point readers to Table 1, which facilitates the identification of research gaps and potential saturation for specific categories of methods. Following our proposed taxonomy, certain categories of methods have no proposed approaches for specific modality types, which could point to low-hanging fruits. Considering the prevalence of Foundation Models and the fairness problems they currently present, we also urge readers to dedicate significantly more effort into debiasing large-scale models.

We hope this survey enables researchers to quickly understand the issues we are facing, and we stress that it is not enough simply providing new neural models without paying attention to the potential harms and consequences that such models may have on underprivileged groups or individuals.

## ACKNOWLEDGMENTS

This work was funded by Motorola Mobility Brazil.

## REFERENCES

[1] Ehsan Abbasnejad, Damien Teney, Amin Parvaneh, Javen Shi, and Anton van den Hengel. 2020. Counterfactual vision and language learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10044-10054.

[2] Ehsan Adeli, Qingyu Zhao, Adolf Pfefferbaum, Edith V. Sullivan, Li Fei-Fei, Juan Carlos Niebles, and Kilian M. Pohl. 2019. Bias-Resilient Neural Network. ArXiv abs/1910.03676 (2019).

[3] Haswanth Aekula, Sugam Garg, and Animesh Gupta. 2021. [RE] Double-Hard Debias: Tailoring Word Embeddings for Gender Bias Mitigation. CoRR abs/2104.06973 (2021). arXiv:2104.06973 https://arxiv.org/abs/2104.06973

[4] Sharat Agarwal, Sumanyu Muku, Saket Anand, and Chetan Arora. 2022. Does Data Repair Lead to Fair Models? Curating Contextually Fair Data To Reduce Model Bias. In 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE.

[5] Alexander Amini, Ava P. Soleimany, Wilko Schwarting, Sangeeta N. Bhatia, and Daniela Rus. 2019. Uncovering and mitigating algorithmic bias through learned latent structure. Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (2019), 289 - 295.

[6] Carolyn Ashurst, Emmie Hine, Paul Sedille, and Alexis Carlier. 2022. AI Ethics Statements: Analysis and Lessons Learnt from NeurIPS Broader Impact Statements. In 2022 ACM Conference on Fairness, Accountability, and Transparency. 2047-2056.

[7] Rajas Bansal. 2022. A Survey on Bias and Fairness in Natural Language Processing. arXiv preprint arXiv:2204.09591 (2022).

[8] Solon Barocas, Moritz Hardt, and Arvind Narayanan. 2019. Fairness and Machine Learning. fairmlbook.org. http://www.fairmlbook.org.

[9] Rachel KE Bellamy et al. 2019. AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias. IBM fournal of Research and Development 63, 4/5 (2019), 4-1.

[10] Emily M Bender and Batya Friedman. 2018. Data statements for natural language processing: Toward mitigating system bias and enabling better science. Transactions of the Association for Computational Linguistics 6 (2018), 587-604.

[11] Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency. 610-623.

[12] Yoshua Bengio, Aaron Courville, and Pascal Vincent. 2013. Representation Learning: A Review and New Perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence 35, 8 (Aug 2013), 1798-1828.

[13] Misha Benjamin, Paul Gagnon, Negar Rostamzadeh, Chris Pal, Yoshua Bengio, and Alex Shee. 2019. Towards standardization of data licenses: The montreal data license. arXiv preprint arXiv:1903.12262 (2019).

[14] Hugo Berg, Siobhan Mackenzie Hall, Yash Bhalgat, Wonsuk Yang, Hannah Rose Kirk, Aleksandar Shtedritski, and Max Bain. 2022. A prompt array keeps the bias away: Debiasing vision-language models with adversarial learning. arXiv preprint arXiv:2203.11933 (2022).

[15] Sarah Bird, Miro Dudík, Richard Edgar, Brandon Horn, Roman Lutz, Vanessa Milan, Mehrnoosh Sameki, Hanna Wallach, and Kathleen Walker. 2020. Fairlearn: A toolkit for assessing and improving fairness in AI. Technical Report. Microsoft.

[16] Sarah Bird, Krishnaram Kenthapadi, Emre Kiciman, and Margaret Mitchell. 2019. Fairness-Aware Machine Learning: Practical Challenges and Lessons Learned (WSDM '19). Association for Computing Machinery, New York, NY, USA, 834-835.

[17] Sid Black et al. 2022. GPT-NeoX-20B: An Open-Source Autoregressive Language Model. CoRR abs/2204.06745 (2022).

[18] Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. 2020. Language (technology) is power: A critical survey of" bias" in nlp. arXiv preprint arXiv:2005.14050 (2020).

[19] Colin R. Blyth. 1972. On Simpson's Paradox and the Sure-Thing Principle. 7. Amer. Statist. Assoc. 67, 338 (1972), 364-366.

[20] Miranda Bogen and Aaron Rieke. 2018. Help wanted: An examination of hiring algorithms, equity, and bias. Upturn, December 7 (2018).

[21] Tolga Bolukbasi et al. 2016. Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. In Advances in Neural Information Processing Systems, Daniel D. Lee et al. (Eds.). 4349-4357.

[22] Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam Tauman Kalai. 2016. Quantifying and Reducing Stereotypes in Word Embeddings. CoRR abs/1606.06121 (2016). arXiv:1606.06121 http://arxiv.org/abs/1606.06121

[23] Rishi Bommasani et al. 2021. On the Opportunities and Risks of Foundation Models. ArXiv e-prints 2108.07258 (Aug 2021), 212.

[24] Shikha Bordia and Samuel R. Bowman. 2019. Identifying and reducing gender bias in word-level language models. North American Chapter of the Association for Computational Linguistics (2019), 7 - 15.

[25] Tom B. Brown et al. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems, Hugo Larochelle et al. (Eds.).

[26] Joy Buolamwini and Timnit Gebru. 2018. Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification. In Conference on Fairness, Accountability and Transparency, Sorelle A. Friedler and Christo Wilson (Eds.). PMLR, 77-91.

[27] Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. 2017. Semantics derived automatically from language corpora contain human-like biases. Science 356, 6334 (2017), 183-186.

[28] Yushi Cao et al. 2022. Fair and accurate age prediction using distribution aware data curation and augmentation. In IEEE/CVF Winter Conference on Applications of Computer Vision, WACV. IEEE, 2867-2877.

[29] Simon Caton and Christian Haas. 2020. Fairness in machine learning: A survey. arXiv preprint arXiv:2010.04053 (2020).

[30] Long Chen, Xin Yan, Jun Xiao, Hanwang Zhang, Shiliang Pu, and Yueting Zhuang. 2020. Counterfactual samples synthesizing for robust visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10800-10809.

[31] Xiuying Chen, Mingzhe Li, Rui Yan, Xin Gao, and Xiangliang Zhang. 2022. Unsupervised Mitigation of Gender Bias by Character Components: A Case Study of Chinese Word Embedding. Workshop on Gender Bias in Natural Language Processing (2022), 121 - 128.

[32] Yunliang Chen and Jungseock Joo. 2021. Understanding and Mitigating Annotation Bias in Facial Expression Recognition. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE.

[33] Pengyu Cheng, Weituo Hao, Siyang Yuan, Shijing Si, and Lawrence Carin. 2021. Fairfil: Contrastive neural debiasing method for pretrained text encoders. arXiv preprint arXiv:2103.06413 (2021).

[34] Jaemin Cho, Abhay Zala, and Mohit Bansal. 2022. DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers. https://arxiv.org/abs/2202.04053

[35] Kristy Choi, Aditya Grover, Trisha Singh, Rui Shu, and Stefano Ermon. 2020. Fair Generative Modeling via Weak Supervision. In ICML. $1887-1898$.

[36] Alexandra Chouldechova. 2017. Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments. Big Data 5,2 (2017), 153-163.

[37] Lee Cohen, Zachary C. Lipton, and Yishay Mansour. 2019. Efficient candidate screening under multiple tests and implications for fairness. arXiv:1905.11361

[38] Council of European Union. 2016. European Union General Data Protection Regulation, Article 22: "Automated individual decisionmaking, including profiling". https://www.privacy-regulation.eu/en/article-22-automated-individual-decision-making-includingprofiling-GDPR.htm.

[39] Henriette Cramer, Jean Garcia-Gathright, Aaron Springer, and Sravana Reddy. 2018. Assessing and addressing algorithmic bias in practice. Interactions 25, 6 (2018), 58-63.

[40] Michael Crawshaw. 2020. Multi-task learning with deep neural networks: A survey. arXiv preprint arXiv:2009.09796 (2020).

[41] Elliot Creager et al. 2019. Flexibly Fair Representation Learning by Disentanglement. In International Conference on Machine Learning, Kamalika Chaudhuri and Ruslan Salakhutdinov (Eds.). PMLR, 1436-1445.

[42] Saloni Dash, Vineeth N. Balasubramanian, and Amit Sharma. 2022. Evaluating and Mitigating Bias in Image Classifiers: A Causal Perspective Using Counterfactuals. IEEE/CVF Winter Conference on Applications of Computer Vision, WACV (2022), 3879 - 3888.

[43] Maria De-Arteaga et al. 2019. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In Conference on Fairness, Accountability, and Transparency. 120-128.

[44] Pieter Delobelle, Ewoenam Kwaku Tokpo, Toon Calders, and Bettina Berendt. 2021. Measuring fairness with biased rulers: A survey on quantifying biases in pretrained language models. arXiv preprint arXiv:2112.07447 (2021).

[45] Ekberjan Derman. 2021. Dataset Bias Mitigation Through Analysis of CNN Training Scores. CoRR abs/2106.14829 (2021). arXiv:2106.14829 https://arxiv.org/abs/2106.14829

[46] Sunipa Dev, Tao Li, Jeff M Phillips, and Vivek Srikumar. 2020. On measuring and mitigating biased inferences of word embeddings. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34. 7659-7666.

[47] Sunipa Dev, Tao Li, Jeff M. Phillips, and Vivek Srikumar. 2021. OSCaR: Orthogonal Subspace Correction and Rectification of Biases in Word Embeddings. In Conference on Empirical Methods in Natural Language Processing, Marie-Francine Moens et al. (Eds.). 5034-5050.

[48] Sunipa Dev and Jeff M. Phillips. 2019. Attenuating Bias in Word vectors. In International Conference on Artificial Intelligence and Statistics, Kamalika Chaudhuri and Masashi Sugiyama (Eds.). PMLR, 879-887.

[49] Mengnan Du, Fan Yang, Na Zou, and Xia Hu. 2021. Fairness in Deep Learning: A Computational Perspective. IEEE Intelligent Systems 36 (2021), 25-34.

[50] Mengnan Du, Fan Yang, Na Zou, and Xia Hu. 2021. Fairness in Deep Learning: A Computational Perspective. IEEE Intelligent Systems 36, 4 (July 2021), 25-34.

[51] Siyi Du, Ben Hers, Nourhan Bayasi, Ghassan Hamarneh, and Rafeef Garbi. 2022. FairDisCo: Fairer AI in Dermatology via Disentanglement Contrastive Learning. arXiv:2208.10013 [cs.CV]

[52] Yuhao Du and Kenneth Joseph. 2020. MDR Cluster-Debias: A Nonlinear WordEmbedding Debiasing Pipeline. CoRR abs/2006.11642 (2020). arXiv:2006.11642 https://arxiv.org/abs/2006.11642

[53] Jannik Dunkelau and Michael Leuschel. 2019. Fairness-Aware Machine Learning: An Extensive Overview. https://www3.hhu.de/ stups/downloads/pdf/fairness-survey.pdf

[54] Cynthia Dwork and Christina Ilvento. 2018. Fairness Under Composition. (2018).

[55] Harrison Edwards and Amos J. Storkey. 2016. Censoring Representations with an Adversary. In 4th International Conference on Learning Representations, ICLR 2016, San 7uan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, Yoshua Bengio and Yann LeCun (Eds.).

[56] Farshid Faal, Ketra A. Schmitt, and Jia Yuan Yu. 2022. Reward Modeling for Mitigating Toxicity in Transformer-based Language Models CoRR abs/2202.09662 (2022). arXiv:2202.09662 https://arxiv.org/abs/2202.09662

[57] Zahra Fatemi, Chen Xing, Wenhao Liu, and Caiming Xiong. 2021. Improving gender fairness of pre-trained language models without catastrophic forgetting. arXiv preprint arXiv:2110.05367 (2021).

[58] Eve Fleisig and Christiane Fellbaum. 2022. Mitigating Gender Bias in Machine Translation through Adversarial Learning. https: //arxiv.org/abs/2203.10675

[59] Sorelle A Friedler, Carlos Scheidegger, and Suresh Venkatasubramanian. 2016. On the (im) possibility of fairness. arXiv preprint arXiv:1609.07236 (2016).

[60] Yacine Gaci, Boualem Benatallah, Fabio Casati, and Khalid Benabdeslem. 2022. Iterative adversarial removal of gender bias in pretrained word embeddings. Proceedings of the ACM Symposium on Applied Computing (2022), 829 - 836.

[61] Pratik Gajane and Mykola Pechenizkiy. 2017. On Formalizing Fairness in Prediction with Machine Learning. https://arxiv.org/abs/ 1710.03184

[62] Sébastien Gambs and Rosin Claude Ngueveu. 2022. Fair mapping. https://arxiv.org/abs/2209.00617

[63] Jean Garcia-Gathright, Aaron Springer, and Henriette Cramer. 2018. Assessing and Addressing Algorithmic Bias - But Before We Get There. https://doi.org/10.48550/ARXIV.1809.03332

[64] Ismael Garrido-Muñoz, Arturo Montejo-Ráez, Fernando Martínez-Santiago, and L Alfonso Ureña-López. 2021. A survey on bias in deep NLP. Applied Sciences 11, 7 (2021), 3184.

[65] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. 2021. Datasheets for datasets. Commun. ACM 64, 12 (2021), 86-92.

[66] Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. 2020. RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models. In Findings of the Association for Computational Linguistics. 3356-3369.

[67] Daniel Cohen George Zerveas, Navid Rekabsaz and Carsten Eickhoff. 2022. Mitigating Bias in Search Results Through Contextual Document Reranking and Neutrality Regularization. In International ACM SIGIR Conference on Research and Development in Information Retrieval. 2532-2538.

[68] Michael Gira, Ruisu Zhang, and Kangwook Lee. 2022. Debiasing Pre-Trained Language Models via Efficient Fine-Tuning. In Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion. 59-69.

[69] Ting Gong et al. 2019. A comparison of loss weighting strategies for multi task learning in deep neural networks. IEEE Access 7 (2019), $141627-141632$

[70] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems. 2672-2680.

[71] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. 2015. Explaining and Harnessing Adversarial Examples. In International Conference on Learning Representations.

[72] Google. 2020. ML-fairness-gym: A Tool for Exploring Long-Term Impacts of Machine Learning Systems. https://ai.googleblog.com/ 2020/02/ml-fairness-gym-tool-for-exploring-long.html

[73] Adam Gronowski, William Paul, Fady Alajaji, Bahman Gharesifard, and Philippe Burlina. 2022. Achieving Utility, Fairness, and Compactness via Tunable Information Bottleneck Measures. https://arxiv.org/abs/2206.10043

[74] Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Franco Turini, Fosca Giannotti, and Dino Pedreschi. 2018. A survey of methods for explaining black box models. ACM computing surveys (CSUR) 51, 5 (2018), 1-42.

[75] Wei Guo and Aylin Caliskan. 2021. Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain a Distribution of Human-like Biases. In AAAI/ACM Conference on AI, Ethics, and Society. 122-133.

[76] Umang Gupta et al. 2022. Mitigating Gender Bias in Distilled Language Models via Counterfactual Role Reversal. https://arxiv.org/ $\mathrm{abs} / 2203.12574$

[77] Enoch Opanin Gyamfi, Yunbo Rao, Miao Gou, and Yanhua Shao. 2020. Deb2viz: Debiasing gender in word embedding data using subspace visualization. Proceedings of SPIE - The International Society for Optical Engineering 11373 (2020).

[78] Christian Haas. 2019. The price of fairness-A framework to explore trade-offs in algorithmic fairness. In 40th International Conference on Information Systems, ICIS 2019. Association for Information Systems.

[79] Lisa Anne Hendricks, Kaylee Burns, Kate Saenko, Trevor Darrell, and Anna Rohrbach. 2018. Women also snowboard: Overcoming bias in captioning models. In Proceedings of the European Conference on Computer Vision (ECCV). 771-787.

[80] Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. 2018. Towards a Definition of Disentangled Representations. https://arxiv.org/abs/1812.02230

[81] Irina Higgins, Loïc Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, Matthew M. Botvinick, Shakir Mohamed, and Alexander Lerchner. 2017. beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In ICLR.

[82] Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the Knowledge in a Neural Network. CoRR abs/1503.02531 (2015). arXiv:1503.02531 http://arxiv.org/abs/1503.02531

[83] Yusuke Hirota, Yuta Nakashima, and Noa Garcia. 2022. Quantifying Societal Bias Amplification in Image Captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13450-13459.

[84] Sarah Holland, Ahmed Hosny, Sarah Newman, Joshua Joseph, and Kasia Chmielinski. 2020. The dataset nutrition label. Data Protection and Privacy, Volume 12: Data Protection and Democracy 12 (2020), 1

[85] Kenneth Holstein, Jennifer Wortman Vaughan, Hal Daumé III, Miro Dudik, and Hanna Wallach. 2019. Improving fairness in machine learning systems: What do industry practitioners need?. In Proceedings of the 2019 CHI conference on human factors in computing systems. $1-16$.

[86] Bhanu Jain, Manfred Huber, and Ramez Elmasri. 2021. Increasing Fairness in Predictions Using Bias Parity Score Based Loss Function Regularization. https://arxiv.org/abs/2111.03638

[87] Surya Mattu Julia Angwin, Jeff Larson and Lauren Kirchner. 2016. Machine Bias. https://www.propublica.org/article/machine-biasrisk-assessments-in-criminal-sentencing. Accessed: 2022-08-23.

[88] Sangwon Jung, Donggyu Lee, Taeeon Park, and Taesup Moon. 2021. Fair Feature Distillation for Visual Recognition. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

[89] Masahiro Kaneko and Danushka Bollegala. 2019. Gender-preserving Debiasing for Pre-trained Word Embeddings. In Conference of the Association for Computational Linguistics. 1641-1650.

[90] Masahiro Kaneko and Danushka Bollegala. 2021. Debiasing Pre-trained Contextualised Embeddings. In Conference of the European Chapter of the Association for Computational Linguistics. 1256-1266.

[91] Masahiro Kaneko and Danushka Bollegala. 2021. Dictionary-based Debiasing of Pre-trained Word Embeddings. In Conference of the European Chapter of the Association for Computational Linguistics. 212-223.

[92] Masahiro Kaneko and Danushka Bollegala. 2021. Unmasking the Mask-Evaluating Social Biases in Masked Language Models. arXiv preprint arXiv:2104.07496 (2021).

[93] Sunghun Kang, Gwangsu Kim, and Chang D Yoo. 2022. Fair Facial Attribute Classification via Causal Graph-Based Attribute Translation. Sensors 22, 14 (2022), 5271.

[94] Cemre Karakas, Alara Dirik, Eylul Yalcinkaya, and Pinar Yanardag. 2022. FairStyle: Debiasing StyleGAN2 with Style Channel Manipulations. CoRR abs/2202.06240 (2022). arXiv:2202.06240 https://arxiv.org/abs/2202.06240

[95] Tero Karras, Samuli Laine, and Timo Aila. 2019. A Style-Based Generator Architecture for Generative Adversarial Networks. In IEEE Conference on Computer Vision and Pattern Recognition. 4401-4410.

[96] Patrik Joslin Kenfack, Kamil Sabbagh, Adín Ramírez Rivera, and Adil Khan. 2022. RepFair-GAN: Mitigating Representation Bias in GANs Using Gradient Clipping. https://arxiv.org/abs/2207.10653

[97] Byungju Kim, Hyunwoo Kim, Kyungsu Kim, Sungjin Kim, and Junmo Kim. 2019. Learning Not to Learn: Training Deep Neural Networks With Biased Data. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

[98] Hyemi Kim, Seungjae Shin, JoonHo Jang, Kyungwoo Song, Weonyoung Joo, Wanmo Kang, and Il-Chul Moon. 2021. Counterfactual Fairness with Disentangled Causal Effect Variational Autoencoder. In AAAI Conference on Artificial Intelligence. 8128-8136.

[99] Jin-Young Kim and Sung-Bae Cho. 2022. An information theoretic approach to reducing algorithmic bias for machine learning. Neurocomputing 500 (2022), 26 - 38 .

[100] Gary King and Langche Zeng. 2006. The dangers of extreme counterfactuals. (2006), 131-159.

[101] Hannah Rose Kirk et al. 2021. Bias Out-of-the-Box: An Empirical Analysis of Intersectional Occupational Biases in Popular Generative Language Models. In Advances in Neural Information Processing Systems. 2611-2624.

[102] Brendan F Klare, Mark J Burge, Joshua C Klontz, Richard W Vorder Bruegge, and Anil K Jain. 2012. Face recognition performance: Role of demographic information. IEEE Transactions on Information Forensics and Security 7, 6 (2012), 1789-1801.

[103] Brendan F. Klare, Mark J. Burge, Joshua C. Klontz, Richard W. Vorder Bruegge, and Anil K. Jain. 2012. Face Recognition Performance: Role of Demographic Information. IEEE Transactions on Information Forensics and Security 7, 6 (2012), 1789-1801.

[104] Jon Kleinberg, Jens Ludwig, Sendhil Mullainathan, and Ashesh Rambachan. 2018. Algorithmic fairness. In Aea papers and proceedings, Vol. 108. 22-27.

[105] Camila Kolling, Martin More, Nathan Gavenski, Eduardo Pooch, Otávio Parraga, and Rodrigo C Barros. 2022. courfactual Debiasing for Visual Question Answering. In IEEE/CVF Winter Conference on Applications of Computer Vision. 3001-3010.

[106] Vaibhav Kumar, Tenzin Singhay Bhotia, and Tanmoy Chakraborty. 2020. Nurse is Closer to Woman than Surgeon? Mitigating Gender-Biased Proximities in Word Embeddings. Trans. Assoc. Comput. Linguistics 8 (2020), 486-503.

[107] Vaibhav Kumar, Tenzin Singhay Bhotia, Vaibhav Kumar, and Tanmoy Chakraborty. 2021. Identifying and Mitigating Gender Bias in Hyperbolic Word Embeddings. https://arxiv.org/abs/2109.13767

[108] Keita Kurita, Nidhi Vyas, Ayush Pareek, Alan W Black, and Yulia Tsvetkov. 2019. Measuring bias in contextualized word representations. arXiv preprint arXiv:1906.07337 (2019).

[109] Matt J Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. 2017. Counterfactual Fairness. In Advances in Neural Information Processing Systems.

[110] Anja Lambrecht and Catherine Tucker. 2019. Algorithmic bias? An empirical study of apparent gender-based discrimination in the display of STEM career ads. Management science 65, 7 (2019), 2966-2981.

[111] Anne Lauscher, Goran Glavas, Simone Paolo Ponzetto, and Ivan Vulic. 2020. A General Framework for Implicit and Explicit Debiasing of Distributional Word Vector Spaces. In AAAI Conference on Artificial Intelligence. 8131-8138.

[112] Anne Lauscher, Tobias Lueken, and Goran Glavaš. 2021. Sustainable Modular Debiasing of Language Models. In Findings of the Association for Computational Linguistics. ACL, 4782-4797.

[113] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang, and Eirini Ntoutsi. 2022. A survey on datasets for fairness-aware machine learning. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery (2022), e1452.

[114] Gil Levi and Tal Hassncer. 2015. Age and gender classification using convolutional neural networks. In 2015 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 34-42.

[115] Peizhao Li, Han Zhao, and Hongfu Liu. 2020. Deep Fair Clustering for Visual Learning. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020. Computer Vision Foundation / IEEE, 9067-9076.

[116] Yong Li, Yufei Sun, Zhen Cui, Shiguang Shan, and Jian Yang. 2021. Learning Fair Face Representation With Progressive Cross Transformer. arXiv:2108.04983

[117] Yi Li and Nuno Vasconcelos. 2019. REPAIR: Removing Representation Bias by Dataset Resampling. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

[118] Yuantong Li, Xiaokai Wei, Zijian Wang, Shen Wang, Parminder Bhatia, Xiaofei Ma, and Andrew Arnold. 2022. Debiasing Neural Retrieval via In-batch Balancing Regularization. Workshop on Gender Bias in Natural Language Processing (2022), 58 - 66.

[119] Paul Pu Liang, Irene Mengze Li, Emily Zheng, Yao Chong Lim, Ruslan Salakhutdinov, and Louis-Philippe Morency. 2020. Towards Debiasing Sentence Representations. In Annual Meeting of the Association for Computational Linguistics. ACL, 5502-5515.

[120] Paul Pu Liang, Chiyu Wu, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2021. Towards Understanding and Mitigating Social Biases in Language Models. https://arxiv.org/abs/2106.13219

[121] Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang, and Sam Kwong. 2019. Pareto multi-task learning. Advances in neural information processing systems 32 (2019).

[122] Ruibo Liu, Chenyan Jia, Jason Wei, Guangxuan Xu, Lili Wang, and Soroush Vosoughi. 2021. Mitigating Political Bias in Language Models Through Reinforced Calibration. 35th AAAI Conference on Artificial Intelligence, AAAI 2021 17A (2021), 14857 - 14866.

[123] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. 2015. Deep Learning Face Attributes in the Wild. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015. IEEE Computer Society, Santiago, Chile, 3730-3738.

[124] Francesco Locatello et al. 2019. Challenging common assumptions in the unsupervised learning of disentangled representations. In International Conference on Machine Learning. 4114-4124.

[125] Francesco Locatello, Gabriele Abbati, Thomas Rainforth, Stefan Bauer, Bernhard Schölkopf, and Olivier Bachem. 2019. On the fairness of disentangled representations. Advances in Neural Information Processing Systems 32 (2019).

[126] Kaiji Lu, Piotr Mardziel, Fangjing Wu, Preetam Amancharla, and Anupam Datta. 2020. Gender bias in neural natural language processing. In Logic, Language, and Security. Springer, 189-202.

[127] Martin Q Ma, Yao-Hung Hubert Tsai, Paul Pu Liang, Han Zhao, Kun Zhang, Ruslan Salakhutdinov, and Louis-Philippe Morency. 2021. Conditional Contrastive Learning for Improving Fairness in Self-Supervised Learning. arXiv e-prints (2021), arXiv-2106.

[128] Michael A Madaio, Luke Stark, Jennifer Wortman Vaughan, and Hanna Wallach. 2020. Co-designing checklists to understand organizational challenges and opportunities around fairness in AI. In Proceedings of the $2020 \mathrm{CHI}$ Conference on Human Factors in Computing Systems. 1-14.

[129] Gaurav Maheshwari, Pascal Denis, Mikaela Keller, and Aurélien Bellet. 2022. Fair NLP Models with Differentially Private Text Encoders. https://arxiv.org/abs/2205.06135

[130] Natalia Martinez, Martin Bertran, and Guillermo Sapiro. 2020. Minimax pareto fairness: A multi objective perspective. In International Conference on Machine Learning. PMLR, 6755-6764.

[131] Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman, and Rachel Rudinger. 2019. On Measuring Social Biases in Sentence Encoders. In Conference of the North American Chapter of the Association for Computational Linguistics. 622-628.

[132] Pratik Mazumder, Pravendra Singh, and Vinay P. Namboodiri. 2022. Fair Visual Recognition in Limited Data Regime using SelfSupervision and Self-Distillation. 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2022), 3889-3897.

[133] Hope McGovern. 2021. A Source-Criticism Debiasing Method for GloVe Embeddings. CoRR abs/2106.13382 (2021). arXiv:2106.13382 https://arxiv.org/abs/2106.13382

[134] Nicholas Meade, Elinor Poole-Dayan, and Siva Reddy. 2021. An empirical survey of the effectiveness of debiasing techniques for pre-trained language models. arXiv preprint arXiv:2110.08527 (2021).

[135] Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. 2021. A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR) 54, 6 (2021), 1-35.

[136] Tomás Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient Estimation of Word Representations in Vector Space. In International Conference on Learning Representations.

[137] Pamela Mishkin, Lama Ahmad, Miles Brundage, Gretchen Krueger, and Girish Sastry. 2022. DALL$\cdot$E 2 Preview - Risks and Limitations. (2022).

[138] Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model cards for model reporting. In Conference on Fairness, Accountability, and Transparency. 220-229.

[139] Tom M Mitchell. 1980. The need for biases in learning generalizations. Department of Computer Science, Laboratory for Computer Science Research.

[140] Brent Mittelstadt. 2019. Principles alone cannot guarantee ethical AI. Nature Machine Intelligence 1, 11 (2019), 501-507.

[141] Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. StereoSet: Measuring stereotypical bias in pretrained language models. In Annual Meeting of the Association for Computational Linguistics. ACL, 5356-5371.

[142] Priyanka Nanayakkara, Jessica Hullman, and Nicholas Diakopoulos. 2021. Unpacking the expressed consequences of AI research in broader impact statements. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society. 795-806.

[143] Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman. 2020. CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. In Conference on Empirical Methods in Natural Language Processing. ACL, 1953-1967.

[144] NeurIPS. 2020. Call for Papers. https://neurips.cc/Conferences/2020/CallForPapers

[145] NeurIPS. 2021. Introducing the NeurIPS 2021 Paper Checklist. https://neuripsconf.medium.com/introducing-the-neurips-2021-paperchecklist-3220d6df500b

[146] NeurIPS. 2021. NeurIPS 2021 Paper Checklist Guidelines. https://neurips.cc/Conferences/2021/PaperInformation/PaperChecklist

[147] Mkhuseli Ngxande, Jules-Raymond Tapamo, and Michael Burke. 2020. Bias Remediation in Driver Drowsiness Detection Systems Using Generative Adversarial Networks. IEEE Access 8 (2020), 55592-55601.

[148] Vahid Partovi Nia, Alireza Ghaffari, Mahdi Zolnouri, and Yvon Savaria. 2022. Rethinking Pareto Frontier for Performance Evaluation of Deep Neural Networks. arXiv preprint arXiv:2202.09275 (2022).

[149] Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and Ji-Rong Wen. 2021. Counterfactual vqa: A cause-effect look at language bias. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12700-12710.

[150] Odbal, Guanhong Zhang, and Sophia Ananiadou. 2022. Examining and mitigating gender bias in text emotion detection task. Neurocomputing 493 (2022), 422 - 434 .

[151] Alexandra Olteanu, Carlos Castillo, Fernando Diaz, and Emre Kıcıman. 2019. Social data: Biases, methodological pitfalls, and ethical boundaries. Frontiers in Big Data 2 (2019), 13.

[152] Sungho Park, Sunhee Hwang, Jongkwang Hong, and Hyeran Byun. 2020. Fair-VQA: Fairness-Aware Visual Question Answering Through Sensitive Attribute Prediction. IEEE Access 8 (2020), 215091-215099.

[153] Sungho Park, Sunhee Hwang, Dohyung Kim, and Hyeran Byun. 2021. Learning Disentangled Representation for Fair Facial Attribute Classification via Fairness-aware Information Alignment. AAAI Conference on Artificial Intelligence 35, 3 (2021), 2403-2411.

[154] Pranita Patil and Kevin Purcell. 2022. Decorrelation-Based Deep Learning for Bias Mitigation. Future Internet 14, 4 (2022), 110.

[155] Alejandro Pena, Ignacio Serna, Aythami Morales, and Julian Fierrez. 2020. Bias in multimodal AI: Testbed for fair automatic recruitment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 28-29.

[156] Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 1532-1543.

[157] Antonio Perianes-Rodriguez, Ludo Waltman, and Nees Jan van Eck. 2016. Constructing bibliometric networks: A comparison between full and fractional counting. Journal of Informetrics 10, 4 (2016), 1178-1195.

[158] Dana Pessach and Erez Shmueli. 2020. Algorithmic fairness. arXiv preprint arXiv:2001.09784 (2020).

[159] Carina EA Prunkl, Carolyn Ashurst, Markus Anderljung, Helena Webb, Jan Leike, and Allan Dafoe. 2021. Institutionalizing ethics in AI through broader impact requirements. Nature Machine Intelligence 3, 2 (2021), 104-110.

[160] Rebecca Qian, Candace Ross, Jude Fernandes, Eric Smith, Douwe Kiela, and Adina Williams. 2022. Perturbation Augmentation for Fairer NLP. https://arxiv.org/abs/2205.12586

[161] Yusu Qian, Urwa Muaz, Ben Zhang, and Jae Won Hyun. 2019. Reducing Gender Bias in Word-Level Language Models with a Gender-Equalizing Loss Function. Annual Meeting of the Association for Computational Linguistics (2019), 223 - 228.

[162] Alec Radford et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In International Conference on Machine Learning, Vol. 139. 8748-8763.

[163] Amirarsalan Rajabi, Mehdi Yazdani-Jahromi, Ozlem Ozmen Garibay, and Gita Sukthankar. 2022. Through a fair looking-glass: mitigating bias in image datasets. https://arxiv.org/abs/2209.08648

[164] Inioluwa Deborah Raji, Timnit Gebru, Margaret Mitchell, Joy Buolamwini, Joonseok Lee, and Emily Denton. 2020. Saving face: Investigating the ethical concerns of facial recognition auditing. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society. $145-151$.

[165] Vikram V. Ramaswamy, Sunnie S. Y. Kim, and Olga Russakovsky. 2021. Fair Attribute Classification through Latent Space De-biasing. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

[166] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical Text-Conditional Image Generation with CLIP Latents. CoRR abs/2204.06125 (2022). arXiv:2204.06125

[167] Navid Rekabsaz, Simone Kopeinik, and Markus Schedl. 2021. Societal biases in retrieved contents: Measurement framework and adversarial mitigation of bert rankers. In International ACM SIGIR Conference on Research and Development in Information Retrieval. 306-316.

[168] Laria Reynolds and Kyle McDonell. 2021. Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm. In CHI Conference on Human Factors in Computing Systems. 314:1-314:7.

[169] Brianna Richardson, Jean Garcia-Gathright, Samuel F Way, Jennifer Thom, and Henriette Cramer. 2021. Towards Fairness in Practice: A Practitioner-Oriented Rubric for Evaluating Fair ML Toolkits. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 1-13.

[170] Brianna Richardson and Juan E Gilbert. 2021. A Framework for Fairness: A Systematic Review of Existing Fair AI Solutions. arXiv preprint arXiv:2112.05700 (2021).

[171] Michael Ruchte and Josif Grabocka. 2021. Scalable Pareto Front Approximation for Deep Multi-Objective Learning. In 2021 IEEE International Conference on Data Mining (ICDM). IEEE, 1306-1311.

[172] Olga Russakovsky et al. 2015. ImageNet Large Scale Visual Recognition Challenge. International fournal of Computer Vision 115, 3 (2015), 211-252.

[173] Pedro Saleiro, Benedict Kuester, Loren Hinkson, Jesse London, Abby Stevens, Ari Anisfeld, Kit T Rodolfa, and Rayid Ghani. 2018. Aequitas: A bias and fairness audit toolkit. arXiv preprint arXiv:1811.05577 (2018).

[174] Tiago Salvador, Stephanie Cairns, Vikram Voleti, Noah Marshall, and Adam M. Oberman. 2022. FairCal: Fairness Calibration for Face Verification. In International Conference on Learning Representations.

[175] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. CoRR abs/1910.01108 (2019). arXiv:1910.01108 http://arxiv.org/abs/1910.01108

[176] Timo Schick, Sahana Udupa, and Hinrich Schütze. 2021. Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP. Transactions of the Association for Computational Linguistics 9 (2021), 1408-1424.

[177] Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, and Dhruv Batra. 2016. Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization. arXiv:1610.02391

[178] Ignacio Serna, Aythami Morales, Julian Fierrez, and Nick Obradovich. 2022. Sensitive loss: Improving accuracy and fairness of face representations with discrimination-aware deep learning. Artificial Intelligence 305 (2022).

[179] Deven Shah, H. Andrew Schwartz, and Dirk Hovy. 2020. Predictive Biases in Natural Language Processing Models: A Conceptual Framework and Overview. In Annual Meeting of the Association for Computational Linguistics. 5248-5264.

[180] Kulin Shah, Pooja Gupta, Amit Deshpande, and Chiranjib Bhattacharyya. 2021. Rawlsian fair adaptation of deep learning classifiers. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society. 936-945.

[181] Shanya Sharma, Manan Dey, and Koustuv Sinha. 2022. How sensitive are translation systems to extra contexts? Mitigating gender bias in Neural Machine Translation models through relevant contexts. https://arxiv.org/abs/2205.10762

[182] Aili Shen, Xudong Han, Trevor Cohn, Timothy Baldwin, and Lea Frermann. 2021. Contrastive Learning for Fair Representations. https://arxiv.org/abs/2109.10645

[183] Tianshu Shen, Jiaru Li, Mohamed Reda Bouadjenek, Zheda Mai, and Scott Sanner. 2022. Unintended Bias in Language ModeldrivenConversational Recommendation. arXiv preprint arXiv:2201.06224 (2022).

[184] Emily Sheng, Kai-Wei Chang, Prem Natarajan, and Nanyun Peng. 2020. Towards Controllable Biases in Language Generation. In Findings of the Association for Computational Linguistics: EMNLP 2020. Association for Computational Linguistics, Online, 3239-3254.

[185] Ramya Srinivasan, Ajay Chander, and Pouya Pezeshkpour. 2019. Generating User-friendly Explanations for Loan Denials using GANs.

[186] Artūrs Stafanovičs, Toms Bergmanis, and Mārcis Pinnis. 2020. Mitigating Gender Bias in Machine Translation with Target Gender Annotations. In Conference on Machine Translation. ACL, 629-638.

[187] Samuel Don Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A Alemi, and Andrew Gordon Wilson. 2021. Does Knowledge Distillation Really Work?. In Advances in Neural Information Processing Systems.

[188] Luke Stark and Anna Lauren Hoffmann. 2019. Data is the new what? Popular metaphors \& professional ethics in emerging data culture. (2019).

[189] Shivashankar Subramanian, Xudong Han, Timothy Baldwin, Trevor Cohn, and Lea Frermann. 2021. Evaluating Debiasing Techniques for Intersectional Biases. In Conference on Empirical Methods in Natural Language Processing. 2492-2498.

[190] Harini Suresh and John V. Guttag. 2019. A Framework for Understanding Unintended Consequences of Machine Learning. CoRR abs/1901.10002 (2019). arXiv:1901.10002 http://arxiv.org/abs/1901.10002

[191] Ruixiang Tang, Mengnan Du, Yuening Li, Zirui Liu, Na Zou, and Xia Hu. 2021. Mitigating gender bias in captioning systems. In Proceedings of the Web Conference 2021. 633-645.

[192] Md Mehrab Tanjim et al. 2022. Generating and Controlling Diversity in Image Search. IEEE/CVF Winter Conference on Applications of Computer Vision (2022), 3908 - 3916.

[193] Enzo Tartaglione, Carlo Alberto Barbano, and Marco Grangetto. 2021. EnD: Entangling and Disentangling deep representations for bias correction. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

[194] TensorFlow. 2020. Fairness Indicators. https://www.tensorflow.org/responsible_ai/fairness_indicators/guide

[195] Huan Tian, Tianqing Zhu, Wei Liu, and Wanlei Zhou. 2022. Image fairness in deep learning: problems, models, and challenges. Neural Computing and Applications (March 2022).

[196] Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, and Luc Van Gool. 2021. Multi-task learning for dense prediction tasks: A survey. IEEE transactions on pattern analysis and machine intelligence (2021).

[197] Francisco Vargas and Ryan Cotterell. 2020. Exploring the Linear Subspace Hypothesis in Gender Bias Mitigation. In Conference on Empirical Methods in Natural Language Processing. 2902-2913.

[198] Sriram Vasudevan and Krishnaram Kenthapadi. 2020. Lift: A scalable framework for measuring fairness in ml applications. In Proceedings of the 29th ACM International Conference on Information \& Knowledge Management. 2773-2780.

[199] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems, NeurIPS. 5998-6008.

[200] Sahil Verma and Julia Rubin. 2018. Fairness definitions explained. In 2018 ieee/acm international workshop on software fairness (fairware). IEEE, $1-7$.

[201] Jesse Vig et al. 2020. Investigating gender bias in language models using causal mediation analysis. Advances in Neural Information Processing Systems (2020), 12388-12401.

[202] Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. 2019. Universal Adversarial Triggers for Attacking and Analyzing NLP. In Conference on Empirical Methods in Natural Language Processing. 2153-2162.

[203] Jialu Wang, Yang Liu, and Xin Eric Wang. 2021. Are gender-neutral queries really gender-neutral? mitigating gender bias in image search. arXiv preprint arXiv:2109.05433 (2021)

[204] Mei Wang and Weihong Deng. 2020. Mitigating Bias in Face Recognition Using Skewness-Aware Reinforcement Learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9319-9328.

[205] Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, and Yaohai Huang. 2018. Racial Faces in-the-Wild: Reducing Racial Bias by Information Maximization Adaptation Network. (2018).

[206] Tianlu Wang, Jieyu Zhao, Mark Yatskar, Kai-Wei Chang, and Vicente Ordonez. 2019. Balanced Datasets Are Not Enough: Estimating and Mitigating Gender Bias in Deep Image Representations. In IEEE/CVF International Conference on Computer Vision.

[207] Kellie Webster, Xuezhi Wang, Ian Tenney, Alex Beutel, Emily Pitler, Ellie Pavlick, Jilin Chen, Ed Chi, and Slav Petrov. 2020. Measuring and reducing gendered correlations in pre-trained models. arXiv preprint arXiv:2010.06032 (2020).

[208] James Wexler, Mahima Pushkarna, Tolga Bolukbasi, Martin Wattenberg, Fernanda Viégas, and Jimbo Wilson. 2019. The what-if tool: Interactive probing of machine learning models. IEEE transactions on visualization and computer graphics 26, 1 (2019), 56-65.

[209] Chuhan Wu, Fangzhao Wu, Xiting Wang, Yongfeng Huang, and Xing Xie. 2021. Fairness-aware news recommendation with decomposed adversarial learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 4462-4469.

[210] Fangsheng Wu et al. 2021. Understanding Social Biases Behind Location Names in Contextual Word Embedding Models. IEEE Transactions on Computational Social Systems 9, 2 (2021), 458-468.

[211] Yawen Wu, Dewen Zeng, Xiaowei Xu, Yiyu Shi, and Jingtong Hu. 2022. FairPrune: Achieving Fairness Through Pruning for Dermatological Disease Diagnosis. https://arxiv.org/abs/2203.02110

[212] Depeng Xu, Yongkai Wu, Shuhan Yuan, Lu Zhang, and Xintao Wu. 2019. Achieving Causal Fairness through Generative Adversarial Networks. In International foint Conference on Artificial Intelligence. 1452-1458.

[213] Tian Xu, Jennifer White, Sinan Kalkan, and Hatice Gunes. 2020. Investigating Bias and Fairness in Facial Expression Recognition. In Computer Vision - ECCV 2020 Workshops. Springer International Publishing, 506-523.

[214] Shen Yan, Di Huang, and Mohammad Soleymani. 2020. Mitigating biases in multimodal personality assessment. In Proceedings of the 2020 International Conference on Multimodal Interaction. 361-369.

[215] Xu Yang, Hanwang Zhang, Guojun Qi, and Jianfei Cai. 2021. Causal attention for vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9847-9857.

[216] Zekun Yang and Juan Feng. 2020. A Causal Inference Method for Reducing Gender Bias in Word Embedding Relations. In AAAI Conference on Artificial Intelligence. $9434-9441$.

[217] Seyma Yucer, Samet Akçay, Noura Al Moubayed, and Toby P. Breckon. 2020. Exploring Racial Bias within Face Recognition via per-subject Adversarially-Enabled Data Augmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition. 83-92.

[218] B. Zhang, Blake Lemoine, and Margaret Mitchell. 2018. Mitigating Unwanted Biases with Adversarial Learning. Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society (2018).

[219] Susan Zhang et al. 2022. OPT: Open Pre-trained Transformer Language Models. CoRR abs/2205.01068 (2022).

[220] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2017. Men also like shopping: Reducing gender bias amplification using corpus-level constraints. arXiv preprint arXiv:1707.09457 (2017).


[^0]:    *The authors contributed equally to this research.

    Authors' address: Otavio Parraga; Martin D. More; Christian M. Oliveira; Nathan S. Gavenski; Lucas S. Kupssinskü; Adilson Medronha; Luis V. Moura; Gabriel S. Simões; Rodrigo C. Barros, Machine Learning Theory and Applications (MALTA) Lab, PUCRS, Brazil.

    Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

    (c) 2022 Association for Computing Machinery.

    0360-0300/2022/0-ART000 $\$ 15.00$

    https://doi.org/XXXXXXX.XXXXXXX

[^1]:    ${ }^{1}$ https://blog.mylio.com/how-many-photos-taken-in-2022/

    ${ }^{2}$ https://www.internetlivestats.com/twitter-statistics/

</end of paper 2>


<paper 3>
# Fast Model Debias with Machine Unlearning 

Ruizhe Chen ${ }^{1}$, Jianfei Yang ${ }^{2}$, Huimin Xiong ${ }^{1}$, Jianhong Bai ${ }^{1}$, Tianxiang $\mathbf{H u}^{1}$, Jin Hao ${ }^{3}$,<br>Yang Feng ${ }^{4}$, Joey Tianyi Zhou ${ }^{5}$, Jian Wu ${ }^{1}$, Zuozhu Liu ${ }^{1 *}$<br>${ }^{1}$ Zhejiang University ${ }^{2}$ Nanyang Technological University<br>${ }^{3}$ Stanford University ${ }^{4}$ Angelalign Technology Inc. ${ }^{5}$ Centre for Frontier AI Research<br>ruizhec.21@intl.zju.edu.cn


#### Abstract

Recent discoveries have revealed that deep neural networks might behave in a biased manner in many real-world scenarios. For instance, deep networks trained on a large-scale face recognition dataset CelebA tend to predict blonde hair for females and black hair for males. Such biases not only jeopardize the robustness of models but also perpetuate and amplify social biases, which is especially concerning for automated decision-making processes in healthcare, recruitment, etc., as they could exacerbate unfair economic and social inequalities among different groups. Existing debiasing methods suffer from high costs in bias labeling or model re-training, while also exhibiting a deficiency in terms of elucidating the origins of biases within the model. To this respect, we propose a fast model debiasing framework (FMD) which offers an efficient approach to identify, evaluate and remove biases inherent in trained models. The FMD identifies biased attributes through an explicit counterfactual concept and quantifies the influence of data samples with influence functions. Moreover, we design a machine unlearning-based strategy to efficiently and effectively remove the bias in a trained model with a small counterfactual dataset. Experiments on the Colored MNIST, CelebA, and Adult Income datasets along with experiments with large language models demonstrate that our method achieves superior or competing accuracies compared with state-of-the-art methods while attaining significantly fewer biases and requiring much less debiasing cost. Notably, our method requires only a small external dataset and updating a minimal amount of model parameters, without the requirement of access to training data that may be too large or unavailable in practice.


## 1 Introduction

Biased predictions are not uncommon in well-trained deep neural networks [1-3]. Recent findings indicate that many deep neural networks exhibit biased behaviors and fail to generalize to unseen data [4, 5], e.g., convolutional neural networks (CNNs) might favor texture over shape for object classification [6]. For instance, well-trained networks on a large-scale dataset (e.g. CelebA) tend to predict a female person to be with blonde hair, and a male to be with black hair [7, 8]. This is because the number of <blonder hair, female> and <black hair, male> image pairs is significantly higher than others, although there is no causal relationship between hair color and gender [9]. In this case, the model does not learn the correct classification strategy based on human appearance, but rather shows a preference for specific individuals or groups based on irrelevant attributes (error correlations) [2]. Such error correlations not only affect the model's ability to make robust predictions but also perpetuate and exacerbate social bias, resulting in potential risks in many real-world scenarios, such as racism, underestimating minorities, or social disparities among groups in crime prediction [10], loan assessment [11], and recruitment [12] etc.[^0]

Efforts have been made to remove bias in models based on innate or acquired characteristics of individuals or groups. Existing debiasing mechanisms could be categorized into three types depending on when debiasing is conducted: pre-processing, in-processing, and post-processing [2, 13, 14]. Pre-processing debiasing methods usually modify the dataset for fair learning, which often involve reweighing samples [15, 16], modifying feature representations [17, 18], changing labels [19] etc. Another line of research accounts for fairness during training, i.e., in-processing [20--24], including feature-level data augmentation or adversarial training [25, 26] etc. However, the aforementioned methods require expensive costs for human labeling of misleading biases or computationally-intensive debiased model retraining, resulting in unsatisfactory scalability over modern large-scale datasets or models. Few research explore post-processing strategies to achieve fairness with minimal cost [27.29]. They ensure group fairness by alternating predictions of some selected samples, causing degraded accuracy or unfairness on individuals. Moreover, most methods assume that the biased attributes were known, while a generalized debiasing framework should be able to verify whether an attribute (e.g. shape, texture, and color in an image classification task) is biased or not as well [30].

![](https://cdn.mathpix.com/cropped/2024_06_04_3c39cd930b6e5467e029g-02.jpg?height=448&width=1198&top_left_y=844&top_left_x=472)

Figure 1: Pipeline of our proposed FMD.

In this paper, we propose FMD, an all-inclusive framework for fast model debiasing. As illustrated in Fig. 1, the FMD comprises three distinct steps: bias identification, biased-effect evaluation, and bias removal. In contrast to pre- or in-processing debiasing methods, our approach eliminates the need for supervised retraining of the entire model or additional labeling of bias attributes. Notably, FMD leverages only a small external dataset, thereby obviating the requirement for access to extensive or unavailable training data in practical scenarios. Furthermore, achieving fair outputs through FMD necessitates updating only a minimal number of parameters, such as the top MLP layers of pre-trained deep networks. Compared to post-processing debiasing methods, FMD yields superior debiasing performance and consistently enhances fairness across diverse bias metrics with little costs.

The FMD operates through the following procedure. Given an attribute and a well-trained model, our first step is to ascertain whether and to what extent the model exhibits bias towards the attribute. To achieve this, we construct a dataset comprising factual samples along with their corresponding counterfactual samples [31], wherein the attribute in question can be varied. By observing how the model's predictions change with the attribute variations, we can effectively identify any bias present. In the biased-effect evaluation phase, we quantitatively assess the extent to which a biased training sample contributes to the model's biased predictions. This evaluation entails measuring how the biased training sample misleads the model and influences its predictions. To this end, we extend the theory of influence functions [32], employing it to estimate the impact of perturbing a biased attribute within the training data on the model's prediction bias measurement. Finally, we introduce an unlearning mechanism that involves performing a Newton step [33] on the learned model parameters to remove the learned biased correlation. We further design an alternative strategy to unlearn biases with the counterfactual external dataset, avoiding hard requirements on access to the training data which might be unavailable in practice. Our unlearning strategy effectively eliminates the estimated influence of the biased attribute, leading to a more fair and unbiased model. Experiments on multiple datasets show that our method can achieve accuracies on par with bias-tailored training methods with a much smaller counterfactually constructed dataset. The corresponding biases and computational costs are significantly reduced as well. Our main contributions are summarized as:

- We propose a counterfactual inference-based framework that can quantitatively measure the biased degree of a trained (black-box) deep network with respect to different data attributes with a novel influence function.
- We propose an unlearning-based debiasing method that effectively and efficiently removes model biases with a small counterfactual dataset, getting rid of expensive network re-training or bias labeling. Our approach inherently applies to in-processing debiasing.
- Extensive experiments and detailed analysis on multiple datasets demonstrate that our framework can obtain competing accuracies with significantly smaller biases and much fewer data and computational costs.


## 2 Related Works

### 2.1 Group, Individual and Counterfactual Fairness

The pursuit of fairness in machine learning has led to the proposal of fairness-specific metrics. These metrics have been mainly categorized into two types: metrics for group fairness that require similar average outputs of different demographic groups [34-38]; and metrics for individual fairness that necessitate similarity in the probability distributions of individuals that are similar in respect to a specific task, regardless of their demographic group [39-42]. Generally, statistical parity among protected groups in each class (group fairness) could be intuitively unfair at the individual level [43]. Moreover, existing fairness metrics put a heavy emphasis on model predictions, while underestimating the significance of sensitive attributes for decision-making and are insufficient to explain the cause of unfairness in the task [31, 44]. Recently, [31] introduces counterfactual fairness, a causal approach to address individual fairness, which enforces that the distribution of potential predictions for an individual should remain consistent when the individual's protected attributes had been different in a causal sense. In contrast to existing individual bias metrics, counterfactual fairness can explicitly model the causality between biased attributes and unfair predictions, which provides explainability for different biases that may arise towards individuals based on sensitive attributes [45--47].

### 2.2 Bias Mitigation

Proposed debiasing mechanisms are typically categorized into three types 2, 13, 14]: pre-processing, in-processing, and post-processing. Pre- and in-processing algorithms account for fairness before and during the training process, where typical techniques entail dataset modification [15-19] and feature manipulation [20-24, 26, 25]. Post-processing algorithms are performed after training, intending to achieve fairness without the need of modifying data or re-training the model. Current post-processing algorithms make more fair decisions by tweaking the output scores [48-50]. For instance, Hardt [27] achieves equal odds or equal opportunity by flipping certain decisions of the classifier according to their sub-groups. [29, 28] select different thresholds for each group, in a manner that maximizes accuracy and minimizes demographic parity. However, achieving group fairness by simply changing the predictions of several individuals is questionable, e.g., the process might be unfair to the selected individuals, leading to an unsatisfactory trade-off between accuracy and fairness.

### 2.3 Machine Unlearning

Machine unlearning [51-53] is a new paradigm to forget a specific data sample and remove its corresponding influence from a trained model, without the requirement to re-train the model from scratch. It fulfills a user's right to unlearn her private information, i.e., the right to be forgotten, in accordance with requests from the General Data Protection Regulation (GDPR) [54]. Existing unlearning approaches can be roughly categorized into two types: exact unlearning [55, 56] and approximate unlearning [57,-60]. Data influence-based unlearning is a representative branch of approximate unlearning that utilizes influence functions [32] to approximate and remove the effect of a training sample on the model's parameters [61-63]. In this paper, we are inspired by the paradigm of machine unlearning and extend it to remove the model's bias from a deep network without retraining it from scratch.

## 3 Method

### 3.1 Overview and Preliminaries

Problem Formulation. Consider a supervised prediction task with fairness considerations that maps input attributes $\mathcal{A}$ (biased attribute) and $\mathcal{X}$ (other attributes except $\mathcal{A}$ ) to certain outputs $\mathcal{Y}$ (labels). The training dataset $D_{t r}$ can be represented as $\left\{z_{1}, z_{2}, \ldots, z_{n}\right\}$ where each training point $z_{i}=\left\{\left(a_{i}, x_{i}\right), y_{i}\right\} \in \mathcal{A} \times \mathcal{X} \times \mathcal{Y}$. Let $f_{\hat{\theta}}$ denote the trained model (predictor) with parameter $\hat{\theta}$. Let $L\left(z_{i}, \theta\right)$ denote the loss on the training sample $z_{i}$ w.r.t. parameter $\theta$. It is deemed biased if a biased attribute $a$ is highly correlated but wrongly correlated to the prediction $\hat{y}=f_{\hat{\theta}}(x, a)$, e.g., a CNN is biased if it predicts hair color (black/blonde) with the biased attribute genders (male/female).

Motivation. In large part, existing works focused on measuring fairness with implicit quantitative values (e.g. accuracy). However, they do not provide explicit illustrations on whether the decisionmaking is based on sensitive/protected attributes. Furthermore, based on the bias identified, research on how such bias is learned from training samples is limited. Our proposed method bridges this gap with two components: identifying bias from different predictions with counterfactual samples and evaluating the biased-effect from training samples with a modified influence function. Furthermore, we propose a novel machine unlearning-based method to efficiently and effectively remove the biases.

Counterfactual Fairness. We identify the biases of trained models with the concept of counterfactual fairness [31, 46, 45] which better models the causality between biased attributes and unfair predictions. We detail the definition following [31]:

Definition 1 (Counterfactual fairness). A trained model $f_{\hat{\theta}}$ is counterfactual fair on $\mathcal{A}$ if for any $a, \bar{a} \in \mathcal{A}$,

$$
\begin{equation*}
P\left(\hat{Y}_{A \leftarrow a}=y \mid X=x, A=a\right)=P\left(\hat{Y}_{A \leftarrow \bar{a}}=y \mid X=x, A=a\right) \tag{1}
\end{equation*}
$$

for all $x \in \mathcal{X}$ attainable by $X$.

Note that $y=f_{\bar{\theta}}(X, A)$, which implies the process of attribute changing. The definition suggests that, for any individual, changing $a$, i.e., from $a$ to $\bar{a}$, while holding other attributes $x$ unchanged should not change the distribution of $\hat{Y}$ if $a$ is a biased attribute.

Influence function. Influence functions, a standard technique from robust statistics, are recently extended to characterize the contribution of a given training sample to predictions in deep networks [32, 64, 65], e.g., identify whether a sample is helpful or harmful for model predictions. A popular implementation of influence functions is to approximate the effects by applying the perturbation $z=(x, y) \mapsto z_{\delta}=(x+\delta, y)$ [32] that define the parameters resulting from moving $\epsilon$ mass from $z$ onto $z_{\delta}: \hat{\theta}_{\epsilon, z_{\delta},-z}=\operatorname{argmin}_{\theta \in \Theta} \frac{1}{n} \sum_{i=1}^{n} L\left(z_{i}, \theta\right)+\epsilon L\left(z_{\delta}, \theta\right)-\epsilon L(z, \theta)$. An approximated computation of the influence as in [32] can be defined as:

$$
\begin{equation*}
\left.\frac{d \hat{\theta}_{\epsilon, z_{\delta},-z}}{d \epsilon}\right|_{\epsilon=0}=-H_{\hat{\theta}}^{-1}\left(\nabla_{\theta} L\left(z_{\delta}, \hat{\theta}\right)-\nabla_{\theta} L(z, \hat{\theta})\right) \tag{2}
\end{equation*}
$$

### 3.2 Bias Identification and Biased-Effect Evaluation

Counterfactual bias identification. We first identify the biases in a trained model with counterfactual concepts. Given a trained model $f_{\hat{\theta}}$ and an attribute of interest $\mathcal{A}$, a primary question is whether $f_{\hat{\theta}}$ is fair on $\mathcal{A}$. We employ an external dataset $D_{e x}$ (can be constructed from the test set) to identify biases. To measure how prediction changes in accordance with the attribute, for each sample $c_{i}=\left(x_{i}, a_{i}\right) \in D_{e x}$, where $a_{i} \in \mathcal{A}$, we alter $a_{i}$ while keeping $x_{i}$ unchanged based on the requirements of counterfactual fairness. The generated counterfactual sample is denoted as $\bar{c}_{i}=\left(x_{i}, \bar{a}_{i}\right), \bar{a}_{i} \in \mathcal{A}$. We further define the counterfactual bias of the model $f_{\hat{\theta}}$ on sample $c_{i}$ as the difference in predictions:

$$
\begin{equation*}
\left.\left.B\left(c_{i}, \mathcal{A}, \hat{\theta}\right)=\left|P\left(\hat{Y}=f_{\hat{\theta}}(X, A)\right)\right| X=x_{i}, A=a_{i}\right)\right)-P\left(\hat{Y}=f_{\hat{\theta}}(X, A) \mid X=x_{i}, A=\overline{a_{i}}\right) \mid \tag{3}
\end{equation*}
$$

The counterfactual bias on the whole dataset $D_{e x}$ can be represented as the average of individual counterfactual biases:

$$
\begin{equation*}
B\left(D_{e x}, \mathcal{A}, \hat{\theta}\right)=\frac{1}{\left|D_{e x}\right|} \sum_{i} B\left(c_{i}, \mathcal{A}, \hat{\theta}\right) \tag{4}
\end{equation*}
$$

The measured bias is a scalar normalized from 0 to 1 . We set a bias threshold $\delta$ that if the measured $B\left(D_{e x}, \mathcal{A}, f_{\hat{\theta}}\right)$ is larger than $\delta$, we regard $f_{\hat{\theta}}$ to be biased on $\mathcal{A}$. Note that our method could also generalize to other individual bias metrics besides Eq. 3 .

Biased-Effect Evaluation. Based on the identified counterfactual bias, we then investigate how the bias on $\mathcal{A}$ is learned by the model from training samples. Considering $B(\hat{\theta})$ measured on any $\mathcal{A}$ with any $D_{e x}$, our goal is to quantify how each training point $z_{k}$ in the training set $D_{t r}$ contributes to $B(\hat{\theta})$. Let's denote the empirical risk minimizer as $\hat{\theta}=\arg \min _{\theta} \frac{1}{n} \sum_{i=1}^{n} L\left(z_{i}, \theta\right)$, and assume that the empirical risk is twice-differentiable and strictly convex in $\theta$. The influence function [64] provides an approximation on the updates to parameters if $z_{k}$ were removed from $D_{t r}$ with a small coefficient $\epsilon$. The new parameters can be obtained as $\hat{\theta}_{\epsilon, z_{k}}=\arg \min _{\theta} \frac{1}{n} \sum_{i=1, i \neq k}^{n} L\left(z_{i}, \theta\right)+\epsilon L\left(z_{k}, \theta\right)$. By doing so, the influence of removing $z_{k}$ on the bias $B(\hat{\theta})$ can be defined as:

$$
\begin{equation*}
I_{u p, \text { bias }}\left(z_{k}, B(\hat{\theta})\right)=\left.\frac{d B\left(\hat{\theta}_{\epsilon, z_{k}}\right)}{d \epsilon}\right|_{\epsilon=0}=\left.\frac{d B\left(\hat{\theta}_{\epsilon, z_{k}}\right)}{d \hat{\theta}_{\epsilon, z_{k}}} \frac{d \hat{\theta}_{\epsilon, z_{k}}}{d \epsilon}\right|_{\epsilon=0}=-\nabla_{\hat{\theta}} B(\hat{\theta}) H_{\hat{\theta}}^{-1} \nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right) \tag{5}
\end{equation*}
$$

where $H_{\hat{\theta}} \stackrel{\text { def }}{=} \frac{1}{n} \sum_{i=1}^{n} \nabla_{\theta}^{2} L\left(z_{k}, \hat{\theta}\right)$ is the positive definite (PD) Hessian, and the closed form expression of $\left.\frac{d \hat{\theta}_{\epsilon, z_{k}}}{d \epsilon}\right|_{\epsilon=0}$, explaining the influence of $z_{k}$ to model parameters, is provided by the influence function [32]. Note that "up" denotes "upweight". Refer to Appendix A for the derivation. Intuitively, this equation can be understood in two parts: the latter part calculates the impact of removing on the parameters. The former part corresponds to the derivative of bias with respect to parameters, assessing how changes in parameters affect the bias. Hence, this equation quantifies the influence of removing on the bias. Note that $B(\hat{\theta})$ can be any bias measurement of interest. Taking $B\left(D_{\text {ex }}, \mathcal{A}, \hat{\theta}\right)$ defined in Eq. 4 as an example, the influence on counterfactual bias can be boiled down as:

$$
\begin{equation*}
I_{u p, b i a s}\left(z_{k}, B\left(D_{e x}, \mathcal{A}, \hat{\theta}\right)\right)=\frac{1}{\left|D_{e x}\right|} \sum_{c_{i} \in D_{e x}}\left(\nabla_{\hat{\theta}} f_{\hat{\theta}}\left(c_{i}\right)-\nabla_{\hat{\theta}} f_{\hat{\theta}}\left(\bar{c}_{i}\right)\right) H_{\hat{\theta}}^{-1} \nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right) \tag{6}
\end{equation*}
$$

where $I_{\text {up,bias }}\left(z_{k}, B\right)$ is a scalar that measures how each training sample contributes to $B$. If removing the point $z_{k}$ increases the bias, we regard $z_{k}$ as a helpful sample, or harmful otherwise. We provide an illustration of the helpful and harmful samples with a toy example in Section 4.2

### 3.3 Bias Removal via Machine Unlearning

After quantifying how biases are learned by the model from harmful samples, the next question is how to remove such biases. Here we propose a machine unlearning-based strategy to remove the biases caused by harmful samples. In particular, we exploit the powerful capability of machine unlearning paradigms for forgetting certain training samples [66, 62, 63, 61]. Specifically, for a bias measurement $B(\hat{\theta})$,we first rank the influence $I_{u p, \text { bias }}\left(z_{k}, B(\hat{\theta})\right)$ of every training sample $z_{k}$ in $D_{t r}$, and then select the top- $K$ harmful samples. Afterward, we unlearn, i.e., let the model forget, these samples by updating the model parameters $\theta$ with a Newton update step as in [63]:

$$
\begin{equation*}
\theta_{\text {new }}=\hat{\theta}+\sum_{k=1}^{K} H_{\hat{\theta}}^{-1} \nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right) \tag{7}
\end{equation*}
$$

where $H_{\hat{\theta}}^{-1} \nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right)=I_{u p, \text { params }}\left(z_{k}\right)$ is explained as the influence of $z_{k}$ on model parameter [32]. Note that $I_{\text {up,params }}\left(z_{k}\right)$ share similar computation as in Eq. 6 , while $I_{u p, \text { params }}\left(z_{k}\right)$ estimates the influence on model parameter and $I_{u p, \text { bias }}\left(z_{k}, B\right)$ focuses on influence on biases.

Our unlearning strategy is further refined following the observations from experiments in Section 4.2 In particular, by ranking and visualizing the harmful and helpful samples on the biases (as shown in Fig. 33, we have observed that the harmful samples heavily lead to biased/error correlations (i.e., bias-aligned) while the helpful samples behave oppositely (i.e., bias-conflicting). Hence, we propose a straightforward solution that further mitigates the influence of a harmful sample with a
bias-conflicting sample. Consequently, we update the parameters to unlearn the harmful samples by:

$$
\begin{equation*}
\theta_{n e w}=\hat{\theta}+\sum_{k=1}^{K} H_{\hat{\theta}}^{-1}\left(\nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right)-\nabla_{\hat{\theta}} L\left(\bar{z}_{k}, \hat{\theta}\right)\right) \tag{8}
\end{equation*}
$$

where $\bar{z}_{k}$ denotes the bias-conflicting sample of $z_{k}$. Following the explanation in influence theory [32], our unlearn mechanism removes the effect of perturbing a training point $(\bar{a}, x, y)$ to $(a, x, y)$. In other words, we not only remove the influence caused by harmful sample $z_{k}$, but further ensure fairness with the corresponding counterfactual sample $\bar{z}_{k}$, see more details in Section 4.2, 4.4 and Appendix.

Alternative Efficient Unlearn with Cheap External Datasets. In the above sections, the unlearning process is based on the assumption that we could access the original training sample $z_{k}$ to identify and evaluate biases and then forget them. However, in practice, the training set might be too large or even unavailable in the unlearning phase. In response, we further propose to approximate the unlearning mechanism with a small external dataset. As the influence to be removed can be obtained from the change of the protected attribute, we can construct the same modification to the protected attribute on external samples. In particular, we employ the $D_{e x}$ as in Section 3.2 to construct counterfactual pairs for unlearning, which redefines Eq. 8 as:

$$
\begin{equation*}
\theta_{\text {new }}=\hat{\theta}+\sum_{i} H_{\hat{\theta}}^{-1}\left(\nabla_{\hat{\theta}} L\left(c_{i}, \hat{\theta}\right)-\nabla_{\hat{\theta}} L\left(\bar{c}_{i}, \hat{\theta}\right)\right) \tag{9}
\end{equation*}
$$

As $D_{\text {ex }}$ can be easily obtained from an external dataset rather than the training set, the practical applicability of our method could be greatly enhanced, as demonstrated in the experiments.

### 3.4 Model Generalization

Extension to Different Biases. To fulfill different fairness demands, we further discuss the generalization of the bias function $B(\hat{\theta})$ in Eq. 6 to other bias measurements. We provide the extension to the most frequently used group fairness measurement demographic parity [34] which requires equal positive prediction assignment across subgroups (e.g. male and female). Eq. 6 can be rewritten as:

$$
\begin{equation*}
I_{u p, \text { bias }}\left(z_{k}\right)=-\left(\nabla_{\hat{\theta}} \frac{1}{\left|\mathcal{G}_{A=1}\right|} \sum_{c_{i} \in \mathcal{G}_{A=1}} f_{\hat{\theta}}\left(c_{i}\right)-\nabla_{\hat{\theta}} \frac{1}{\left|\mathcal{G}_{A=0}\right|} \sum_{c_{j} \in \mathcal{G}_{A=0}} f_{\hat{\theta}}\left(c_{j}\right)\right) H_{\hat{\theta}}^{-1} \nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right) \tag{10}
\end{equation*}
$$

where $\mathcal{G}_{A=1}$ and $\mathcal{G}_{A=0}$ represents the subgroup with protected attribute $A=1$ and 0 . The extension to equal opportunity [35], which requires the positive predictions to be equally assigned across positive classes, can be rewritten as:

$$
\begin{equation*}
I_{u p, \text { bias }}\left(z_{k}\right)=-\left(\nabla_{\hat{\theta}} \frac{1}{\left|\mathcal{G}_{1,1}\right|} \sum_{c_{i} \in \mathcal{G}_{1,1}} f_{\hat{\theta}}\left(c_{i}\right)-\nabla_{\hat{\theta}} \frac{1}{\left|\mathcal{G}_{0,1}\right|} \sum_{c_{j} \in \mathcal{G}_{0,1}} f_{\hat{\theta}}\left(c_{j}\right)\right) H_{\hat{\theta}}^{-1} \nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right) \tag{11}
\end{equation*}
$$

where $\mathcal{G}_{1,1}$ represents the sub-group where $A=1$ and $Y=1$.

Extension to Deep Models. In the previous sections, it's assumed that $\hat{\theta}$ could be the global minimum. However, if $\hat{\theta}$ is obtained in deep networks trained with SGD in a non-convex setting, it might be a local optimum and the exact influence can hardly be computed. We follow the strategy in [32] to approximate the influence in deep networks, and empirically demonstrate the effectiveness of FMD in deep models. Moreover, for deep networks where a linear classifier is stacked on a backbone feature extractor, we apply our unlearning mechanism to the linear classifier or several top MLP layers.

Efficient Influence Computation. A critical challenge to compute the influence in Eq. 6 is to explicitly calculate the inverse Hessian. Here we employ the implicit Hessianvector products (HVPs) [32, 67] to efficiently approximate $\nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right)$. Meanwhile, $\nabla_{\hat{\theta}} L\left(z_{k}, \hat{\theta}\right)$ in Eq. 6 can be precalculated and applied to different $\nabla_{\hat{\theta}} B(\hat{\theta})$. To avoid the $O\left(d^{3}\right)$ computational cost to calculate the inverse Hessian in every step, we pre-calculate it before the removal and keep it constant during unlearning phase [63]. The alternative strategy which continuously updates the inversion Hessian is also analyzed in the Appendix.

```
Algorithm 1: The FMD framework.
    Input: dataset $D_{e x}$, loss $\mathcal{L}$, attribute of
            interest $\mathcal{A}$, Hessian matrix $H$,
            bias threshold $\delta$, parameter $\theta$,
            $n=\left\|D_{e x}\right\|$.
    $B \leftarrow B\left(D_{e x}, \mathcal{A}, \hat{\theta}\right)$
    $H^{-1} \leftarrow$ Inverse $(H)$
    if $B>\delta$ then
        for $i=1,2,3, \ldots, n$ do
            $\triangle \leftarrow \nabla_{\hat{\theta}} L\left(c_{i}, \hat{\theta}\right)-\nabla_{\hat{\theta}} L\left(\bar{c}_{i}, \hat{\theta}\right)$
            $\theta \leftarrow \theta+H^{-1} \triangle$
        end
    end
    Output: $\theta$
```


## 4 Experiment

### 4.1 Experiment details

Dataset. Our experiments are conducted on three datasets. Colored MNIST is constructed by adding color bias to the MNIST dataset [68]. Bias-aligned samples are constructed by adding a particular color to a particular digit like \{Digit1_Color1\} while other colors are for bias-conflicting samples. Following [3, 69, 70], we build 3 different training sets by setting different biased ratios $\{0.995,0.99$, $0.95\}$ for biased-aligned training samples, where a high ratio indicates a high degree of bias. CelebA [71] is a face recognition dataset with 40 types of attributes like gender, age (young or not), and lots of facial characteristics (such as hair color, smile, beard). We choose Gender as the bias attribute, and Blonde hair and Attractive as the outputs following [7, 8]. Adult Income Dataset is a publicly available dataset in the UCI repository [72] based on the 1994 U.S. census data. The dataset records an individual's income (more or less than $\$ 50,000$ per year) along with features such as occupation, marital status, and education. In our experiment, we choose gender and race as biased attributes following [73, 74]. We follow the pre-processing procedures in [75]. As for the experiment on the language model, we use StereoSet [76] as our test set. StereoSet is a large-scale natural dataset to measure stereotypical biases in gender, profession, race, and religion.

Baselines. For the sanity check experiment on a toy Colored MNIST dataset, we use a vanilla logistic regression model as the baseline. For experiments with deep networks, we compare our method with one pre-processing baseline Reweigh [77], 6 in-processing debiasing baselines (LDR [25], LfF [78], Rebias [79], DRO [7], SenSEI [80], and SenSR [81]) and 4 post-processing baselines (EqOdd [35], CEqOdd [35], Reject [82] and PP-IF [83]). We compare our method on language model with five debiasing baselines: Counterfactual Data Augmentation (CDA) [84], Dropout [85], Iterative Null-space Projection (INLP) [86], Self-debias [87], and SentenceDebias [88]. Details can be referred to Appendix C.

Construction of Counterfactual Dataset $D_{e x}$. We separately construct counterfactual sets for the three datasets, while bias-aligned samples in the small dataset $D_{e x}$ are all split from the test set. In the Colored MNIST dataset, we randomly add another color on the same digit image as the counterfactual sample. As for the Adult dataset, we flip the protected attribute to the opposite while keeping other attributes and target labels exactly the same. For the CelebA dataset, we select images with the same target labels but the opposite protected attribute. To fulfill the request for counterfactual, we rank the similarity between images by comparing the overlap of other attributes and choose the most similar pair to form the factual and counterfactual samples. Part of the generated sample pairs is visualized in Fig. 2. Note that for the CelebA dataset, the counterfactual data are not that strict as the gender attribute is not independent of other features in the natural human facial images. We use Crows-Pairs [89] as our external dataset for the language model. Each sample in Crows-Pairs consists of two sentences: one that is more stereotyping and another that is less stereotyping, which can be utilized as counterfactual pairs.

![](https://cdn.mathpix.com/cropped/2024_06_04_3c39cd930b6e5467e029g-07.jpg?height=314&width=1247&top_left_y=1800&top_left_x=431)

Figure 2: Visualization of factual and counterfactual pairs for three datasets.

Implementation details. We use multi-layer perceptron (MLP) with three hidden layers for Colored MNIST and Adult, and ResNet-18 [90] for CelebA following the setting in [8]. During training, we set the batch size of 256 for Colored MNIST and Adult, respectively, and 64 for CelebA following [25, 78, 7]. We use pre-trained BERT [91] and GPT-2 [92], provided by Huggingface. During unlearning, we freeze the parameters of all other layers except the last classifier layer. The running time of all baselines is evaluated on a single RTX3090 GPU for a fair comparison. In our experiment, we select the number of samples $\mathrm{k}=5000$ for Colored MNIST, and $\mathrm{k}=200$ for both Adult and CelebA. The bias threshold is set to 0 .

### 4.2 Sanity Check on Logistic Regression with a Toy Dataset

We conduct an experiment on a logistic regression task to illustrate our method. We simplify the Colored MNIST classification task to a binary classification problem of distinguishing between only digits 3 and 8 , on a training set with a bias ratio of 0.95 . and a balanced test set. We trained a regularized logistic regressor: $\operatorname{argmin}_{w \in R^{d}} \sum_{i=1}^{n} l\left(w^{T} x_{i}, y_{i}\right)+\lambda\|w\|_{2}^{2}$. Fig. 3 (a) illustrates the classification results of the vanilla regressor on part of test samples. We denote Digit by shape (triangle and rectangle) and Color by color (red and blue). The solid line represents the learned classification boundary and the dotted line represents the expected classification boundary. The test accuracy is 0.6517 and it can be observed that most bias-conflict samples tend to be misclassified according to their colors. Moreover, we select and visualize the most helpful and harmful samples in Fig. 3.c) based on Eq. 6. We found that the most helpful samples are in the $5 \%$ bias-conflict samples while harmful samples are bias-aligned samples. The unlearning curve is provided in Fig. 3.b). With only 50 samples, the accuracy is improved amazingly by $25.71 \%$ and the counterfactual bias decreases by 0.2755 , demonstrating the effectiveness of our method.

![](https://cdn.mathpix.com/cropped/2024_06_04_3c39cd930b6e5467e029g-08.jpg?height=244&width=350&top_left_y=886&top_left_x=454)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_3c39cd930b6e5467e029g-08.jpg?height=247&width=361&top_left_y=890&top_left_x=817)

(b)
![](https://cdn.mathpix.com/cropped/2024_06_04_3c39cd930b6e5467e029g-08.jpg?height=224&width=208&top_left_y=912&top_left_x=1168)

![](https://cdn.mathpix.com/cropped/2024_06_04_3c39cd930b6e5467e029g-08.jpg?height=225&width=344&top_left_y=907&top_left_x=1324)

(c)

Figure 3: (a) Illustration of the learned pattern on our toy dataset.(b) Accuracy and bias curves during unlearning. (b) Visualization of helpful samples (top row) and harmful samples (bottom row).

### 4.3 Experiment on Deep Models

Results on Colored MNIST. Tab. 1 shows the comparisons on the Colored MNIST dataset. We reported test accuracy, counterfactual bias, debiasing time, and the number of samples used for all methods. Our approach demonstrates competing performance on accuracy and superior performance on bias compared with retraining baselines. Meanwhile, we only make use of one-tenth of unlearning samples and reduce the debiasing time by 1-2 magnitudes.

| Bias Ratio | Method | Acc.(\%) $\uparrow$ | Bias $\downarrow$ | Time(s) | \# Samp. |
| :---: | :---: | :---: | :---: | ---: | ---: |
| 0.995 | Vanilla | 38.59 | 0.5863 | - | - |
|  | LDR | 66.76 | 0.4144 | 1,261 | $50 \mathrm{k}$ |
|  | LfF | 56.45 | 0.3675 | 661 | $50 \mathrm{k}$ |
|  | Rebias | $\underline{71.24}$ | $\underline{0.3428}$ | 1,799 | $50 \mathrm{k}$ |
|  | Ours | $\mathbf{7 1 . 7 0}$ | $\mathbf{0 . 3 0 2 7}$ | $\mathbf{5 9}$ | $5 \mathrm{k}$ |
| 0.99 | Vanilla | 51.34 | 0.4931 | - | - |
|  | LDR | 76.48 | 0.2511 | 1,330 | $50 \mathrm{k}$ |
|  | LfF | 64.71 | 0.2366 | 726 | $50 \mathrm{k}$ |
|  | Rebias | $\mathbf{8 0 . 4 1}$ | $\underline{0.2302}$ | 1,658 | $50 \mathrm{k}$ |
|  | Ours | $\underline{80.04}$ | $\mathbf{0 . 2 0 4 2}$ | $\mathbf{4 8}$ | $5 \mathrm{k}$ |
| 0.95 | Vanilla | 77.63 | 0.2589 | - | - |
|  | LDR | $\mathbf{9 0 . 4 2}$ | 0.2334 | 1,180 | $50 \mathrm{k}$ |
|  | LfF | 85.55 | 0.1264 | 724 | $50 \mathrm{k}$ |
|  | Rebias | $\underline{89.63}$ | $\underline{0.1205}$ | 1,714 | $50 \mathrm{k}$ |
|  | Ours | $\underline{89.26}$ | $\mathbf{0 . 1 1 8 9}$ | $\mathbf{5 6}$ | $5 \mathrm{k}$ |

Table 1: Results on Colored MNIST. (bold: best performance, underline: second best performance.)

| Attr. | Method | Acc.(\%) $\uparrow$ | Bias $\downarrow$ | Time(s) | \# Samp. |
| :---: | :---: | :---: | :---: | ---: | ---: |
| Gender | Vanilla | $\mathbf{8 5 . 4 0}$ | 0.0195 | - | - |
|  | LDR | 77.69 | 0.0055 | 927 | 26,904 |
|  | LfF | 73.08 | 0.0036 | 525 | 26,904 |
|  | Rebias | 76.57 | 0.0041 | 1292 | 26,904 |
|  | Reweigh | 82.60 | 0.0051 | 36 | 26,904 |
|  | SenSR | 84.09 | 0.0049 | 571 | 26,904 |
|  | SenSeI | 83.91 | 0.0016 | 692 | 26,904 |
|  | PP-IF | 81.96 | 0.0027 | 13 | 26,904 |
|  | Ours | 81.89 | $\mathbf{0 . 0 0 0 5}$ | $\mathbf{2 . 4 9}$ | 500 |
| Race | Vanilla | $\mathbf{8 4 . 5 7}$ | 0.0089 | - |  |
|  | LDR | 78.32 | 0.0046 | 961 | 26,904 |
|  | LfF | 75.16 | 0.0024 | 501 | 26,904 |
|  | Rebias | 77.89 | 0.0038 | 1304 | 26,904 |
|  | Reweigh | 82.97 | 0.0015 | 36 | 26,904 |
|  | SenSR | 84.09 | 0.0036 | 571 | 26,904 |
|  | SenSeI | 83.91 | 0.0015 | 692 | 26,904 |
|  | PP-IF | 82.37 | 0.0015 | 13 | 26,904 |
|  | Ours | 83.80 | $\mathbf{0 . 0 0 1 3}$ | $\mathbf{2 . 5 4}$ | 500 |

Table 2: Results on Adult.

Results on Adult. The results are in Table 2 It can be observed that the vanilla method performed the best in accuracy on both tasks, since in the real-world dataset, race and gender are biased w.r.t income in both training and test set and the well-trained model fits this correlation. However, to achieve fair prediction, we would not expect biased attributes to dominate predictions. Compared with other debiasing methods, our method achieved the best results in both accuracy and bias, with much less debiasing time on a smaller dataset.

Results on CelebA. We compare on average accuracy (Avg.), Unbiased accuracy [8] (Unb.) tested on the balanced test set, and Worst-group accuracy [7] (Wor.) tested on the unprivileged group to illustrate the performance, as reported in Tab. 3 It can be observed that the vanilla model performs well on the whole dataset (Avg.) but scores a really low accuracy (Wor.) on the worst group, which means the learned model heavily relies on the bias attribute to achieve high accuracy. Our method obviously bridges this gap and outperforms all other debiasing baselines on Wor. and Unb. in the two experiments. The experiments also demonstrate that our method is consistently feasible even in the absence of perfect standardized counterfactual samples in real-world datasets, by selecting a certain amount of approximate counterfactual data.

Results on Large Language Models (LLM). We further extended our method to the LLM debiasing scenario. Results are presented in Tab. 4. We report two metrics: Language Modeling Score (LMS) measures the percentage of instances in which a language model prefers the meaningful over meaningless association. The LMS of an ideal language model will be

| Attr. | Method | Unb.(\%) $\uparrow$ | Wor.(\%) $\uparrow$ | Avg.(\%) $\uparrow$ | Bias $\downarrow$ | Time(s) |
| :---: | :--- | :---: | :---: | :---: | :---: | ---: |
| Blonde | Vanilla | 66.27 | 47.36 | $\mathbf{9 4 . 9 0}$ | 0.4211 | - |
|  | LfF | 84.33 | 81.24 | 93.52 | 0.2557 | 67,620 |
|  | LDR | 85.01 | 82.32 | 86.67 | 0.3126 | 24,180 |
|  | DRO | $\underline{85.66}$ | $\underline{84.36}$ | 92.90 | 0.3206 | 28,860 |
|  | Ours | $\mathbf{8 9 . 7 3}$ | $\mathbf{8 7 . 1 5}$ | $\underline{93.41}$ | $\mathbf{0 . 0 7 1 7}$ | $\mathbf{1 9 1}$ |
| Attractive | Vanilla | 63.17 | 40.59 | 77.42 | 0.3695 | - |
|  | LfF | 67.44 | 52.25 | 77.24 | 0.2815 | 67,560 |
|  | LDR | $\underline{68.14}$ | 54.47 | $\mathbf{8 1 . 7 0}$ | 0.2986 | 24,420 |
|  | DRO | $\mathbf{6 6 . 1 4}$ | $\underline{62.33}$ | 78.35 | 0.3004 | 30,540 |
|  | Ours | $\mathbf{7 2 . 1 8}$ | $\mathbf{6 8 . 1 6}$ | $\underline{80.99}$ | $\mathbf{0 . 1 2 7 3}$ | $\mathbf{1 8 7}$ |

100 (the higher the better). Stereotype

Table 3: Results on CelebA. Score (SS) measures the percentage of examples in which a model prefers a stereotypical association over an anti-stereotypical association. The SS of an ideal language model will be 50 (the closer to 50 the better). It shows that our method can outperform or achieve comparable performance with baseline methods. As for BERT, our method reaches the best (denoted by bold) or second best (denoted by underline) performance in 5 of 6 metrics.

| Backbone | Attribute | Method | $\mathrm{SS}$ | LMS | Attribute | Method | $\mathrm{SS}$ | LMS | Attribute | Method | $\mathrm{SS}$ | LMS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BERT | gender | Vanilla | 60.28 | 84.17 | race | Vanilla | 57.03 | 84.17 | religion | Vanilla | 59.7 | 84.17 |
|  |  | CDA | 59.61 | 83.08 |  | CDA | 56.73 | 83.41 |  | CDA | 58.37 | 83.24 |
|  |  | Dropout | 60.66 | 83.04 |  | Dropout | 57.07 | 83.04 |  | Dropout | 59.13 | 83.04 |
|  |  | INLP | $\mathbf{5 7 . 2 5}$ | 80.63 |  | INLP | 57.29 | 83.12 |  | INLP | 60.31 | 83.36 |
|  |  | Self-debias | 59.34 | 84.09 |  | Self-debias | 54.30 | 84.24 |  | Self-debias | 57.26 | 84.23 |
|  |  | SentDebias | 59.37 | 84.20 |  | SentDebias | 57.78 | 83.95 |  | SentDebias | 58.73 | 84.26 |
|  |  | Ours | 57.77 | $\overline{85.45}$ |  | Ours | 57.24 | $\overline{84.19}$ |  | Ours | 57.85 | 84.90 |
| GPT-2 | gender | Vanilla | 62.65 | 91.01 | race | Vanilla | 58.9 | 91.01 | religion | Vanilla | 63.26 | $\underline{91.01}$ |
|  |  | CDA | 64.02 | 90.36 |  | CDA | 57.31 | 90.36 |  | CDA | 63.55 | $\overline{90.36}$ |
|  |  | Dropout | 63.35 | 90.40 |  | Dropout | $\overline{57.50}$ | 90.40 |  | Dropout | 64.17 | 90.40 |
|  |  | INLP | 60.17 | 91.62 |  | INLP | 58.96 | 91.06 |  | INLP | 63.95 | 91.17 |
|  |  | Self-debias | $\overline{60.84}$ | 89.07 |  | Self-debias | 57.33 | $\overline{89.53}$ |  | Self-debias | 60.45 | 89.36 |
|  |  | SentDebias | 56.05 | 87.43 |  | SentDebias | 56.43 | 91.38 |  | SentDebias | 59.62 | 90.53 |
|  |  | Ours | 60.42 | $\underline{91.01}$ |  | Ours | 60.42 | 91.01 |  | Ours | $\overline{58.43}$ | 86.13 |

Table 4: Results with Large Language Models (BERT and GPT-2).

### 4.4 Analysis

Effectiveness on Different Bias Metrics. We validate the generalization ability of our unlearning method based on different fairness metrics on the Colored MNIST with bias severity 0.99 . In Tab. 5. we compare the performance of unlearning harmful samples based on three different biases: Counterfactual bias (Co.), Demographic parity bias (De.) [34], and Equal opportunity bias (Eo.) [35]. For each experiment, we report the changes in three biases. We can note that our method is consistently effective on all three bias metrics. Meanwhile, our counterfactual-based unlearning can significantly outperform the other two in terms of accuracy, Co., and De., and is comparable with them on Eo..

|  | Acc. $(\%) \uparrow$ | Co. $\downarrow$ | De. $\downarrow$ | Eo. $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| Vanilla | 65.17 | 0.3735 | 0.5895 | 0.2235 |
| Unlearn by De. | 71.52 | 0.1796 | 0.4026 | 0.0116 |
| Unlearn by Eo. | 71.12 | 0.1826 | 0.4217 | $\mathbf{0 . 0 1 0 3}$ |
| Unlearn by Co. (Ours) | $\mathbf{8 7 . 9 0}$ | $\mathbf{0 . 1 0 5 1}$ | $\mathbf{0 . 1 4 9 8}$ | 0.0108 |

Table 5: Ablation on Different Biases.

|  | Acc. $\uparrow$ | Bias $\downarrow$ | Time(s) $\downarrow$ |
| :--- | ---: | ---: | ---: |
| Vanilla | 65.17 | 0.3735 | - |
| Unlearn by Eq. 7 | 90.68 | 0.1182 | 36.87 |
| Unlearn by Eq. 8 | $\mathbf{9 1 . 1 8}$ | $\mathbf{0 . 1 0 2 3}$ | 39.63 |
| Unlearn by Eq. 9 (Ours) | 90.42 | 0.1051 | $\mathbf{0 . 0 5 9}$ |

Table 6: Ablation on Unlearning Strategies.

Effectiveness of Unlearn Strategies. We empirically investigate the feasibility of the unlearning mechanism on training and external samples on the Colored MNIST with bias severity 0.99. In Tab.6, we report the results of unlearning harmful training samples (Eq. 7), unlearning by replacing harmful
samples with their bias-conflict helpful samples (Eq. 87) and unlearning with external counterfactual sample pairs (Eq. 9). It can be observed that unlearning in the training dataset can achieve higher accuracy and less bias, and Eq. 8 excels on both metrics. But unlearning with training samples requires much more time and training samples might not be available in practice, while unlearning with external samples provides a satisfactory alternative.

| Attr. | Method | Acc.(\%) $\uparrow$ | Co. $\downarrow$ | De. $\downarrow$ | Eq. $\downarrow$ | Time(s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gender | EqOdd | 82.71 | 0.0247 | 0.5991 | $\mathbf{0 . 0 0 2 1}$ | $\mathbf{0 . 0 1 1 3}$ |
|  | CEqOdd | 83.22 | 0.0047 | 0.4469 | 0.0125 | 0.5583 |
|  | Reject | 74.63 | 0.0876 | 0.2744 | 0.3140 | 14.420 |
|  | Ours | $\mathbf{8 3 . 4 9}$ | $\mathbf{0 . 0 0 1 9}$ | $\mathbf{0 . 1 4 3 8}$ | 0.0460 | 0.0389 |
| Race | EqOdd | 83.22 | 0.0139 | 0.7288 | $\mathbf{0 . 0 0 2 1}$ | $\mathbf{0 . 0 1 0 5}$ |
|  | CEqOdd | 82.88 | 0.0012 | 0.6803 | 0.0054 | 3.6850 |
|  | Reject | 74.63 | 0.1156 | 0.4349 | 0.1825 | 14.290 |
|  | Ours | $\mathbf{8 3 . 1 2}$ | $\mathbf{0 . 0 0 0 6}$ | $\mathbf{0 . 4 2 1 9}$ | 0.0367 | 0.0360 |


| Method | \# Lay. | \# Para. | Acc(\%) $\uparrow$ | Bias $\downarrow$ | Time(s) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Vanilla | - | - | 51.34 | 0.4931 | - |
| Ours $^{1}$ | 1 | $1 \mathrm{~K}$ | 71.19 | $\mathbf{0 . 2 7 5 7}$ | $\mathbf{3 . 7 5}$ |
| Ours $^{2}$ | 2 | $11 \mathrm{~K}$ | $\mathbf{7 4 . 1 8}$ | 0.3134 | 432.86 |
| Ours $^{3}$ | 3 | $21 \mathrm{~K}$ | 61.45 | 0.2949 | 496.44 |

Table 8: Ablation on \# MLP Layers.

Table 7: Discussion on Post-processing Methods.

Discussion on Post-processing Methods. We compare our method to post-processing methods, i.e., Equalized Odds Post-processing (EqOdd) [35], Calibrated Equalized Odds Post-processing (CEqOdd) [35] and Reject Option Classification (Reject) [82], as shown in Tab. 7, Note these methods only apply to logistic regression. Our method outperforms them in most cases on Adult. It is also worth noting that these post-processing methods aimed at a specific fairness measure tend to exacerbate unfairness under other measures while our method consistently improves the fairness under different measures.

Ablation on the Number of Samples. Fig. 4 demonstrates the sensitivity of our unlearning performance w.r.t. number of samples on Colored MNIST with a bias ratio of 0.99 . The accuracy increases and bias decreases incrementally with more samples, and becomes steady after the number is beyond 5,000 . On the other hand, the unlearning time increases linearly with the number of samples. Additionally, constructing a large number of counterfactual samples in practice might be time-consuming as well. Practical usage of the FMD would require a trade-off based on utility requirements.

![](https://cdn.mathpix.com/cropped/2024_06_04_3c39cd930b6e5467e029g-10.jpg?height=347&width=483&top_left_y=1214&top_left_x=1233)

Figure 4: Ablation on \# Samples.

Ablation on Number of Fine-tuning Layers. We explore the impact of unlearning different numbers of layers (i.e., the last (one), two, three MLP) on the Color MNIST, with results in Tab. 8. Interestingly, the accuracy excels with two layers but decreases with three layers. Additionally, fine-tuning multiple layers takes much longer time on computation on much more parameters. It is also worth noting that our method could achieve such superior or competing performance even only by updating the last layer in deep models, which calls for more in-depth analysis in the future

## 5 Conclusion and Limitation

Biased behaviors in contemporary well-trained deep neural networks can perpetuate social biases, and also pose challenges to the models' robustness. In response, we present FDM, an all-inclusive framework for fast and effective model debiasing. We explicitly measure the influence of training samples on bias measurement and propose a removal mechanism for model debiasing. Comprehensive experiments on multiple datasets demonstrate that our method can achieve superior/competing accuracies with a significantly lower bias as well as computational cost.

Our work preliminarily explored the application of our method to large language models, as well as more analysis on model fairness from different perspectives, which will be in our future plan. In addition, our method is not applicable to black-box models, which are of high interest in real-world scenarios. Our proposed method requires generating counterfactual pairs with labeled sensitive attributes, while many datasets do not have enough labels. Research on fairness with few/no attribute labels is still in the infant stage [93], and we will further explore it.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China (Grant No. 62106222), the Natural Science Foundation of Zhejiang Province, China(Grant No. LZ23F020008) and the Zhejiang University-Angelalign Inc. R\&D Center for Intelligent Healthcare.

## References

[1] M. Du, F. Yang, N. Zou, and X. Hu, "Fairness in deep learning: A computational perspective," IEEE Intelligent Systems, vol. 36, no. 4, pp. 25-34, 2020.

[2] N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan, "A survey on bias and fairness in machine learning," ACM Computing Surveys (CSUR), vol. 54, no. 6, pp. 1-35, 2021.

[3] B. Kim, H. Kim, K. Kim, S. Kim, and J. Kim, "Learning not to learn: Training deep neural networks with biased data," in IEEE Conference on Computer Vision and Pattern Recognition, 2019 .

[4] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, "Shortcut learning in deep neural networks," Nature Machine Intelligence, vol. 2, no. 11, pp. 665673,2020 .

[5] S. Sagawa, A. Raghunathan, P. W. Koh, and P. Liang, "An investigation of why overparameterization exacerbates spurious correlations," in International Conference on Machine Learning, pp. 8346-8356, PMLR, 2020.

[6] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel, "Imagenettrained cnns are biased towards texture; increasing shape bias improves accuracy and robustness," arXiv preprint arXiv:1811.12231, 2018.

[7] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, "Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization," in International Conference on Learning Representations, 2020.

[8] E. Tartaglione, C. A. Barbano, and M. Grangetto, "End: Entangling and disentangling deep representations for bias correction," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13508-13517, June 2021.

[9] S. Sagawa, A. Raghunathan, P. W. Koh, and P. Liang, "An investigation of why overparameterization exacerbates spurious correlations," in International Conference on Machine Learning, pp. 8346-8356, PMLR, 2020.

[10] T. Brennan, W. Dieterich, and B. Ehret, "Evaluating the predictive validity of the compas risk and needs assessment system," Criminal Justice and behavior, vol. 36, no. 1, pp. 21-40, 2009.

[11] J. F. Mahoney and J. M. Mohen, "Method and system for loan origination and underwriting," Oct. 23 2007. US Patent 7,287,008.

[12] M. Bogen and A. Rieke, "Help wanted: An examination of hiring algorithms, equity, and bias," 2018.

[13] D. Pessach and E. Shmueli, "A review on fairness in machine learning," ACM Computing Surveys (CSUR), vol. 55, no. 3, pp. 1-44, 2022.

[14] O. Parraga, M. D. More, C. M. Oliveira, N. S. Gavenski, L. S. Kupssinskü, A. Medronha, L. V. Moura, G. S. Simões, and R. C. Barros, "Debiasing methods for fairer neural models in vision and language research: A survey," arXiv preprint arXiv:2211.05617, 2022.

[15] Y. Li and N. Vasconcelos, "Repair: Removing representation bias by dataset resampling," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, $\mathrm{pp}$. 9572$9581,2019$.

[16] M. Tanjim, R. Sinha, K. K. Singh, S. Mahadevan, D. Arbour, M. Sinha, G. W. Cottrell, et al., "Generating and controlling diversity in image search," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 411-419, 2022.

[17] F. Calmon, D. Wei, B. Vinzamuri, K. Natesan Ramamurthy, and K. R. Varshney, "Optimized preprocessing for discrimination prevention," Advances in neural information processing systems, vol. 30, 2017.

[18] M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and S. Venkatasubramanian, "Certifying and removing disparate impact," in proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pp. 259-268, 2015.

[19] F. Kamiran and T. Calders, "Data preprocessing techniques for classification without discrimination," Knowledge and information systems, vol. 33, no. 1, pp. 1-33, 2012.

[20] M. B. Zafar, I. Valera, M. Gomez Rodriguez, and K. P. Gummadi, "Fairness beyond disparate treatment \& disparate impact: Learning classification without disparate mistreatment," in Proceedings of the 26th international conference on world wide web, pp. 1171-1180, 2017.

[21] E. Adeli, Q. Zhao, A. Pfefferbaum, E. V. Sullivan, L. Fei-Fei, J. C. Niebles, and K. M. Pohl, "Representation learning with statistical independence to mitigate bias," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2513-2523, 2021.

[22] P. Dhar, J. Gleason, A. Roy, C. D. Castillo, and R. Chellappa, "Pass: protected attribute suppression system for mitigating bias in face recognition," in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15087-15096, 2021.

[23] M. Alvi, A. Zisserman, and C. Nellåker, "Turning a blind eye: Explicit removal of biases and variation from deep neural network embeddings," in Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pp. 0-0, 2018.

[24] H. Bahng, S. Chun, S. Yun, J. Choo, and S. J. Oh, "Learning de-biased representations with biased representations," in International Conference on Machine Learning, pp. 528-539, PMLR, 2020 .

[25] J. Lee, E. Kim, J. Lee, J. Lee, and J. Choo, "Learning debiased representation via disentangled feature augmentation," Advances in Neural Information Processing Systems, vol. 34, pp. 25123$25133,2021$.

[26] T. Wang, J. Zhao, M. Yatskar, K.-W. Chang, and V. Ordonez, "Balanced datasets are not enough: Estimating and mitigating gender bias in deep image representations," in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5310-5319, 2019.

[27] M. Hardt, E. Price, and N. Srebro, "Equality of opportunity in supervised learning," Advances in neural information processing systems, vol. 29, 2016.

[28] A. K. Menon and R. C. Williamson, "The cost of fairness in binary classification," in Conference on Fairness, accountability and transparency, pp. 107-118, PMLR, 2018.

[29] S. Corbett-Davies, E. Pierson, A. Feller, S. Goel, and A. Huq, "Algorithmic decision making and the cost of fairness," in Proceedings of the 23rd acm sigkdd international conference on knowledge discovery and data mining, pp. 797-806, 2017.

[30] P. Saleiro, B. Kuester, L. Hinkson, J. London, A. Stevens, A. Anisfeld, K. T. Rodolfa, and R. Ghani, "Aequitas: A bias and fairness audit toolkit," arXiv preprint arXiv:1811.05577, 2018.

[31] M. J. Kusner, J. Loftus, C. Russell, and R. Silva, "Counterfactual fairness," Advances in neural information processing systems, vol. 30, 2017.

[32] P. W. Koh and P. Liang, "Understanding black-box predictions via influence functions," in International conference on machine learning, pp. 1885-1894, PMLR, 2017.

[33] Z. Cao, J. Wang, S. Si, Z. Huang, and J. Xiao, "Machine unlearning method based on projection residual," arXiv preprint arXiv:2209.15276, 2022.

[34] C. Dwork, M. Hardt, T. Pitassi, O. Reingold, and R. Zemel, "Fairness through awareness," in Proceedings of the 3rd innovations in theoretical computer science conference, pp. 214-226, 2012.

[35] M. Hardt, E. Price, and N. Srebro, "Equality of opportunity in supervised learning," Advances in neural information processing systems, vol. 29, 2016.

[36] S. Jung, S. Chun, and T. Moon, "Learning fair classifiers with partially annotated group labels," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10348-10357, 2022.

[37] S. Verma and J. Rubin, "Fairness definitions explained," in Proceedings of the international workshop on software fairness, pp. 1-7, 2018.

[38] M. Kearns, S. Neel, A. Roth, and Z. S. Wu, "Preventing fairness gerrymandering: Auditing and learning for subgroup fairness," in International conference on machine learning, pp. 25642572, PMLR, 2018.

[39] C. Dwork, M. Hardt, T. Pitassi, O. Reingold, and R. Zemel, "Fairness through awareness," in Proceedings of the 3rd innovations in theoretical computer science conference, pp. 214-226, 2012.

[40] M. Joseph, M. Kearns, J. Morgenstern, S. Neel, and A. Roth, "Rawlsian fairness for machine learning," arXiv preprint arXiv:1610.09559, vol. 1, no. 2, p. 19, 2016.

[41] C. Louizos, K. Swersky, Y. Li, M. Welling, and R. Zemel, "The variational fair autoencoder," arXiv preprint arXiv:1511.00830, 2015.

[42] W. Fleisher, "What's fair about individual fairness?," in Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, pp. 480-490, 2021.

[43] R. Binns, "On the apparent conflict between individual and group fairness," in Proceedings of the 2020 conference on fairness, accountability, and transparency, pp. 514-524, 2020.

[44] S. Chiappa, "Path-specific counterfactual fairness," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, pp. 7801-7808, 2019.

[45] S. Garg, V. Perot, N. Limtiaco, A. Taly, E. H. Chi, and A. Beutel, "Counterfactual fairness in text classification through robustness," in Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pp. 219-226, 2019.

[46] Y. Wu, L. Zhang, and X. Wu, "Counterfactual fairness: Unidentification, bound and algorithm," in Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 2019 .

[47] N. Kilbertus, P. J. Ball, M. J. Kusner, A. Weller, and R. Silva, "The sensitivity of counterfactual fairness to unmeasured confounding," in Uncertainty in artificial intelligence, pp. 616-626, PMLR, 2020.

[48] F. Kamiran, A. Karim, and X. Zhang, "Decision theory for discrimination-aware classification," in 2012 IEEE 12th international conference on data mining, pp. 924-929, IEEE, 2012.

[49] C. Dwork, N. Immorlica, A. T. Kalai, and M. Leiserson, "Decoupled classifiers for group-fair and efficient machine learning," in Conference on fairness, accountability and transparency, pp. 119-133, PMLR, 2018.

[50] N. Kallus, X. Mao, and A. Zhou, "Assessing algorithmic fairness with unobserved protected class using data combination," Management Science, vol. 68, no. 3, pp. 1959-1981, 2022.

[51] T. Baumhauer, P. Schöttle, and M. Zeppelzauer, "Machine unlearning: Linear filtration for logit-based classifiers," Machine Learning, vol. 111, no. 9, pp. 3203-3226, 2022.

[52] Q. P. Nguyen, R. Oikawa, D. M. Divakaran, M. C. Chan, and B. K. H. Low, "Markov chain monte carlo-based machine unlearning: Unlearning what needs to be forgotten," arXiv preprint arXiv:2202.13585, 2022.

[53] A. Tahiliani, V. Hassija, V. Chamola, and M. Guizani, "Machine unlearning: Its need and implementation strategies," in 2021 Thirteenth International Conference on Contemporary Computing (IC3-2021), pp. 241-246, 2021.

[54] M. Magdziarczyk, "Right to be forgotten in light of regulation (eu) 2016/679 of the european parliament and of the council of 27 april 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing directive 95/46/ec," in 6th International Multidisciplinary Scientific Conference on Social Sciences and Art Sgem 2019, pp. 177-184, 2019.

[55] L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, and N. Papernot, "Machine unlearning," in 2021 IEEE Symposium on Security and Privacy $(S P)$, pp. 141-159, IEEE, 2021.

[56] J. Brophy and D. Lowd, "Machine unlearning for random forests," in International Conference on Machine Learning, pp. 1092-1104, PMLR, 2021.

[57] A. Golatkar, A. Achille, and S. Soatto, "Eternal sunshine of the spotless net: Selective forgetting in deep networks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9304-9312, 2020.

[58] A. Mahadevan and M. Mathioudakis, "Certifiable machine unlearning for linear models," arXiv preprint arXiv:2106.15093, 2021.

[59] Y. Wu, E. Dobriban, and S. Davidson, "Deltagrad: Rapid retraining of machine learning models," in International Conference on Machine Learning, pp. 10355-10366, PMLR, 2020.

[60] S. Neel, A. Roth, and S. Sharifi-Malvajerdi, "Descent-to-delete: Gradient-based methods for machine unlearning," in Algorithmic Learning Theory, pp. 931-962, PMLR, 2021.

[61] Z. Cao, J. Wang, S. Si, Z. Huang, and J. Xiao, "Machine unlearning method based on projection residual," arXiv preprint arXiv:2209.15276, 2022.

[62] Z. Izzo, M. A. Smart, K. Chaudhuri, and J. Zou, "Approximate data deletion from machine learning models," in International Conference on Artificial Intelligence and Statistics, pp. 20082016, PMLR, 2021.

[63] C. Guo, T. Goldstein, A. Hannun, and L. Van Der Maaten, "Certified data removal from machine learning models," arXiv preprint arXiv:1911.03030, 2019.

[64] R. D. Cook and S. Weisberg, Residuals and influence in regression. New York: Chapman and Hall, 1982 .

[65] X. Han, B. C. Wallace, and Y. Tsvetkov, "Explaining black box predictions and unveiling data artifacts through influence functions," arXiv preprint arXiv:2005.06676, 2020.

[66] A. Peste, D. Alistarh, and C. H. Lampert, "Ssse: Efficiently erasing samples from trained machine learning models," arXiv preprint arXiv:2107.03860, 2021.

[67] B. A. Pearlmutter, "Fast exact multiplication by the hessian," Neural computation, vol. 6, no. 1, pp. 147-160, 1994.

[68] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.

[69] Y. Li and N. Vasconcelos, "Repair: Removing representation bias by dataset resampling," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9572$9581,2019$.

[70] H. Bahng, S. Chun, S. Yun, J. Choo, and S. J. Oh, "Learning de-biased representations with biased representations," in International Conference on Machine Learning, 2020.

[71] Z. Liu, P. Luo, X. Wang, and X. Tang, "Deep learning face attributes in the wild," in IEEE International Conference on Computer Vision, 2015.

[72] A. Frank, A. Asuncion, et al., "Uci machine learning repository, 2010," URL http://archive. ics. uci. edu/ml, vol. 15, p. 22, 2011.

[73] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork, "Learning fair representations," in International conference on machine learning, pp. 325-333, PMLR, 2013.

[74] T. Kamishima, S. Akaho, and J. Sakuma, "Fairness-aware learning through regularization approach," in 2011 IEEE 11th International Conference on Data Mining Workshops, pp. 643650, IEEE, 2011.

[75] R. K. Bellamy, K. Dey, M. Hind, S. C. Hoffman, S. Houde, K. Kannan, P. Lohia, J. Martino, S. Mehta, A. Mojsilović, et al., "Ai fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias," IBM Journal of Research and Development, vol. 63, no. 4/5, pp. 4-1, 2019 .

[76] M. Nadeem, A. Bethke, and S. Reddy, "Stereoset: Measuring stereotypical bias in pretrained language models," arXiv preprint arXiv:2004.09456, 2020.

[77] P. Li and H. Liu, "Achieving fairness at no utility cost via data reweighing with influence," in International Conference on Machine Learning, pp. 12917-12930, PMLR, 2022.

[78] J. Nam, H. Cha, S. Ahn, J. Lee, and J. Shin, "Learning from failure: Training debiased classifier from biased classifier," in Advances in Neural Information Processing Systems, 2020.

[79] H. Bahng, S. Chun, S. Yun, J. Choo, and S. J. Oh, "Learning de-biased representations with biased representations," in International Conference on Machine Learning (ICML), 2020.

[80] M. Yurochkin and Y. Sun, "Sensei: Sensitive set invariance for enforcing individual fairness," arXiv preprint arXiv:2006.14168, 2020.

[81] M. Yurochkin, A. Bower, and Y. Sun, "Training individually fair ml models with sensitive subspace robustness," arXiv preprint arXiv:1907.00020, 2019.

[82] F. Kamiran, A. Karim, and X. Zhang, "Decision theory for discrimination-aware classification," in 2012 IEEE 12th International Conference on Data Mining, pp. 924-929, 2012.

[83] F. Petersen, D. Mukherjee, Y. Sun, and M. Yurochkin, "Post-processing for individual fairness," Advances in Neural Information Processing Systems, vol. 34, pp. 25944-25955, 2021.

[84] R. Zmigrod, S. J. Mielke, H. Wallach, and R. Cotterell, "Counterfactual data augmentation for mitigating gender stereotypes in languages with rich morphology," arXiv preprint arXiv:1906.04571, 2019.

[85] K. Webster, X. Wang, I. Tenney, A. Beutel, E. Pitler, E. Pavlick, J. Chen, E. Chi, and S. Petrov, "Measuring and reducing gendered correlations in pre-trained models," arXiv preprint arXiv:2010.06032, 2020.

[86] S. Ravfogel, Y. Elazar, H. Gonen, M. Twiton, and Y. Goldberg, "Null it out: Guarding protected attributes by iterative nullspace projection," arXiv preprint arXiv:2004.07667, 2020.

[87] T. Schick, S. Udupa, and H. Schütze, "Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in nlp," Transactions of the Association for Computational Linguistics, vol. 9, pp. 1408-1424, 2021.

[88] P. P. Liang, I. M. Li, E. Zheng, Y. C. Lim, R. Salakhutdinov, and L.-P. Morency, "Towards debiasing sentence representations," arXiv preprint arXiv:2007.08100, 2020.

[89] N. Nangia, C. Vania, R. Bhalerao, and S. R. Bowman, "Crows-pairs: A challenge dataset for measuring social biases in masked language models," arXiv preprint arXiv:2010.00133, 2020.

[90] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," arXiv preprint arXiv:1512.03385, 2015.

[91] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," arXiv preprint arXiv:1810.04805, 2018.

[92] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al., "Language models are unsupervised multitask learners," OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[93] S. Seo, J.-Y. Lee, and B. Han, "Unsupervised learning of debiased representations with pseudoattributes," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16742-16751, 2022.


[^0]:    Corresponding author.

</end of paper 3>


<paper 4>
# LARGE LANGUAGE Model Bias MitiGATION FROM THE PERSPECTIVE OF KNOWLEDGE EDITING 

Ruizhe Chen<br>Zhejiang University

Zuozhu Liu *

Zhejiang University

Yichen Li ${ }^{\dagger}$<br>Zhejiang University

Yang Feng<br>Angelalign Technology Inc.


#### Abstract

Existing debiasing methods inevitably make unreasonable or undesired predictions as they are designated and evaluated to achieve parity across different social groups but leave aside individual facts, resulting in modified existing knowledge. In this paper, we first establish a new bias mitigation benchmark BiasKE leveraging existing and additional constructed datasets, which systematically assesses debiasing performance by complementary metrics on fairness, specificity, and generalization. Meanwhile, we propose a novel debiasing method, Fairness Stamp (FAST), which enables editable fairness through fine-grained calibration on individual biased knowledge. Comprehensive experiments demonstrate that FAST surpasses state-of-the-art baselines with remarkable debiasing performance while not hampering overall model capability for knowledge preservation, highlighting the prospect of fine-grained debiasing strategies for editable fairness in LLMs.


## 1 INTRODUCTION

Pre-trained Large Language Models (LLMs) have demonstrated exceptional performance on many tasks (Devlin et al., 2018; Floridi \& Chiriatti, 2020, Brown et al., 2020). However, the encoded social stereotypes and human-like biases inevitably cause undesired behaviors when deploying LLMs in practice (Zhao et al., 2019, Navigli et al., 2023; Sheng et al., 2021). Existing approaches to mitigate biases in LLMs are mainly categorized into: (1) Fine-tuning (Zmigrod et al., 2019, Webster et al., 2020; He et al., 2022; Liang et al., 2020; Lauscher et al. 2021), which includes techniques such as re-balanced corpus pre-training, contrastive learning, projection methods, and efficient parameter tuning. (2) Prompt-tuning (Guo et al., 2022, Yang et al., 2023, Li et al., 2023, Dong et al., 2023), which involves creating prompts to address social biases.

![](https://cdn.mathpix.com/cropped/2024_06_04_a0775832d6ae1325d620g-01.jpg?height=352&width=1391&top_left_y=1857&top_left_x=367)

(a) Examples

(b) Existing approaches

(c) Ours

Figure 1: (a) Expression towards different groups (e.g., mom/dad) does not necessarily constitute a bias. (b) Existing debiasing approaches usually equalize different groups, resulting in unreasonable predictions. (c) Our proposed method performs fine-grained calibration with biased knowledge, while maintaining the others.

However, existing techniques treat social groups as interchangeable (Gallegos et al., 2023) and neutralize protected attributes of different social groups in model inputs or outputs, while ignoring or[^0]concealing distinct mechanisms of different social groups (Hanna et al., 2020), as shown in Figure 1 Furthermore, existing debiasing evaluation metrics mainly focus on the degree of bias, but fail to measure whether the model retains its origin knowledge (Gallegos et al., 2023) of discerning reasonable disparities among different social groups.

To address these issues, we first establish a more comprehensive debiasing benchmark BiasKE by extending existing datasets with additional constructed data and evaluation metrics on fairness, specificity, and generalization. Moreover, we propose a novel method Fairness-Stamp (FAST) for editable bias mitigation. Instead of mitigating group biases indiscriminately, FAST operates finegrained calibrations on individual biases, i.e., specific stereotyped statements toward a social group. Specifically, we first design a causal-tracing-based method to locate the decisive layer in LLMs responsible for biased predictions. Then we propose to add a lightweight modular network, which enables fine-grained and efficient debiasing of one or multiple individual biased knowledge, with objectives of bias mitigation and knowledge maintenance.

We evaluate FAST with comprehensive experiments on StereoSet (Nadeem et al. 2020b) and CrowsPairs (Nangia et al. 2020), which are further extended as BiasKE for systematic evaluation. Results show that FAST achieves remarkable debiasing performance without compromising model capability. We extend FAST to larger models such as GPT-Neo and Llama to demonstrate the scalability in real-world applications. Additional experiments showcase the effectiveness on downstream tasks, continual bias mitigation, and lightweight optimization, with results and analysis in Appendix D.

## 2 BiASKE BENCHMARK CONSTRUCTION

![](https://cdn.mathpix.com/cropped/2024_06_04_a0775832d6ae1325d620g-02.jpg?height=439&width=1393&top_left_y=1228&top_left_x=363)

Figure 2: An illustration of the construction of BiasKE.

In this section, we describe the procedures for establishing BiasKE, with an illustration in Figure 2 To better express a bias, we formalize the stereotype bias (e.g., Man is good at man) as a triplet $k=(s, r, o)$, where $s$ is the subject (i.e., Man), o is the object (i.e., math), and $r$ is the relation between them (i.e., is good at), as inspired by Petroni et al. (2019). We collect social biases related to three domains (gender, race, and religion) from six existing datasets, as detailed in Appendix A. 2

Step1. Based on these social biases, we extract biased knowledge pairs $\left(k_{1}, k_{2}\right)$. As shown in Figure 2. the sentence "black people are more likely to commit a crime" can be extracted as $k_{1}$ (Black people, are more likely to, commit a crime.). $k_{2}$ is the counterfactual of $k_{1}$, which can have an opposite $s_{2}$ (i.e., white people) or $o_{2}$ (i.e., compliance). Representative examples of different datasets can be referred to in Table 5. The set of biased knowledge pairs is denoted by $\Omega_{S}$.

Step2. Then we create $\Omega_{P}$, the set of paraphrased biased knowledge pair $\left(k_{1}^{\prime}, k_{2}^{\prime}\right)$, with the same semantic expression as $k_{1}, k_{2}$, as exemplified in Figure 2. $\Omega_{P}$ constitutes similar social biases as in $\Omega_{S}$, which is utilized to measure the generalization ability of debiased models and prevent the edited model from overfitting to a particular input.

Step3. Finally, $\Omega_{D}$ is independently created by collecting commonsense knowledge related to the subjects (e.g., man/woman, Christians/Jewish) in $\Omega_{S}$. We also confirm that pre-existing knowledge in $\Omega_{D}$ is irrelevant to the knowledge within $\Omega_{S}$, thus measuring the ability to retain unrelated knowledge. Both $\Omega_{P}$ and $\Omega_{D}$ are initially generated by prompting GPT- 4 API and manually validated.

Evaluating Metrics. Furthermore, for fair and systematic evaluation, we design three evaluating metrics, Stereotype Score (SS), Paraphrase Stereotype Score and Differentiation Score (DS), to evaluate fairness, generalization and specificity ability of debiasing methods, respectively. Specifically, in addition to using SS to measure the degree of bias, PS evaluates the generalization ability on semantically similar biased knowledge, and DS evaluates the ability to preserve existing knowledge about individuals. Detailed descriptions of these evaluating metrics are presented in Appendix A. 1 .

## 3 METHOD

![](https://cdn.mathpix.com/cropped/2024_06_04_a0775832d6ae1325d620g-03.jpg?height=458&width=1272&top_left_y=682&top_left_x=426)

Figure 3: An illustration of our FAST framework. (a) We first localize the critical layer towards biased predictions. (b) A fairness stamp is inserted within the critical layer. (c) Our FAST can finely calibrate debiasing demands with the objective of bias mitigation and knowledge maintenance.

We propose a fine-grained bias mitigation method Fairness-Stamp (FAST). FAST operates through a two-step process, as depicted in Figure 3 . In the first step, we propose to investigate if there are specific hidden states (i.e., layers) that play a more crucial role than others when recalling biased knowledge, as inspired by the knowledge localization works (Meng et al. 2022; Finlayson et al. 2021). Our biased knowledge localization is performed in three steps, biased run, counterfactual input and restoration run, with a complete description in Figure 4 in the Appendix B. 1 .

In the second step, we propose to select the layer that contributes most significantly to the bias and envelope it with a Fairness Stamp. The fairness stamp is a 2-layer Feed-Forward Network (FFN) layer, which adjusts the output of the enveloped layer with the same input. Assuming the input hidden states to be $\mathbf{h}$, the FFN layer in original LLMs can be formulated as follows: FFN $(\mathbf{h})=$ $\operatorname{Act}\left(\mathbf{h} \mathbf{K}^{\top}\right) \mathbf{V}$, where $\mathbf{K}$ and $\mathbf{V}$ denote the parameters (i.e., keys and values matrices) of the first and second linear layers in the FFN, respectively. Our fairness stamp inserts an extra intervention on the original output with a few external parameters. The new output of the modified FFN layer is:

$$
\begin{equation*}
\operatorname{FFN}^{\prime}(\mathbf{h})=\operatorname{FFN}(\mathbf{h})+\operatorname{Act}\left(\mathbf{h} \mathbf{K}^{\prime \top}\right) \mathbf{V}^{\prime} \tag{1}
\end{equation*}
$$

where $\mathbf{K}^{\prime}, \mathbf{V}^{\prime} \in R^{d_{c} \times d}$ are the new parameter matrices in our fairness stamp. The stamp is optimized for each individual biased knowledge in the set $\Omega$ with the objectives of fairness (i.e., bias mitigation) and specificity (i.e., knowledge maintenance).

Fairness. The main objective is to mitigate the biased prediction. With prompts of a biased knowledge pair, we narrow the gap between predictions on the biased object and unbiased object:

$$
\begin{equation*}
\mathcal{L}_{e}=\frac{1}{|\Omega|} \sum_{\left(k_{1}, k_{2}\right) \in \Omega}\left|\mathcal{P}_{\mathcal{G}}\left[k_{1}\right]-\mathcal{P}_{\mathcal{G}}\left[k_{2}\right]\right| \tag{2}
\end{equation*}
$$

where $k_{i}=\left(s_{i}, r_{i}, o_{i}\right)$ and $\mathcal{P}_{\mathcal{G}}\left[k_{i}\right]=\mathcal{P}_{\mathcal{G}}\left[o_{i} \mid p_{i}\right]$ denotes the probability of predicting $o_{i}$ given the prompt $p_{i}=\left(s_{i}, r_{i}\right)$.

Specificity. We propose to preserve existing knowledge in two parts. First, we maintain the predictions for the input prompts on other objects. Furthermore, we minimize the change of predictions on simple prompts $p^{\prime}$ (e.g., " $\{$ subject $\}$ is a [MASK]"), which helps preserve the perception of the
model on the subjects (e.g., man, woman). The two losses are formulated as follows:

$$
\begin{equation*}
\mathcal{L}_{s 1}=\frac{1}{|\Omega|} \sum_{p_{i} \in \Omega} \mathcal{D}_{K L}\left(\mathcal{P}_{\mathcal{G}}\left[\star \mid p_{i}\right], \mathcal{P}_{\mathcal{G}^{*}}\left[\star \mid p_{i}\right]\right), \quad \mathcal{L}_{s 2}=\frac{1}{|\Omega|} \sum_{s_{i} \in \Omega} \mathcal{D}_{K L}\left(\mathcal{P}_{\mathcal{G}}\left[\star \mid p^{\prime}\left(s_{i}\right)\right], \mathcal{P}_{\mathcal{G}^{*}}\left[\star \mid p^{\prime}\left(s_{i}\right)\right]\right) \tag{3}
\end{equation*}
$$

where $\mathcal{P}_{\mathcal{G}}\left[\star \mid p^{\prime}\right]$ is the predicted probability vector. $\mathcal{G}$ and $\mathcal{G}^{*}$ represent the origin and debiased model. $\mathcal{D}_{K L}$ represents the Kullback-Leibler Divergence. To prevent the model from overfitting to particular inputs, we also utilize prefix texts $x_{j}$ to enhance generalization ability across various contexts. These prefix texts are randomly generated by the model, for instance, "My father told me that", and are concatenated to the front of the prompts.

The overall objective is formulated as: $\mathcal{L}=\mathcal{L}_{e}+\alpha \mathcal{L}_{s 1}+\beta \mathcal{L}_{s 2}$, where $\alpha$ and $\beta$ are hyper-parameters.

## 4 EXPERIMENT

Experimental Details. Experiments are mainly conducted on BERT (Devlin et al. 2018) and GPT2 (Radford et al. 2019) compared with 8 state-of-the-art baselines. We also conduct additional experiments on larger models, i.e., GPT2-XL, GPT-Neo, and Llama-2 to further validate the scalability of FAST. We evaluate SS, PS, DS, LMS, and ICAT for comprehensive comparison, with detailed description in the Appendix A.1. We report results on StereoSet (Nadeem et al., 2020b) and Crows-Pairs (Nangia et al. 2020) datasets to keep consistent with baselines. Details of datasets, baselines, model and implementation are reported in Appendix C.1. We only report the experimental results in terms of gender, please refer to the Appendix C. 3 for race and religion.

Debiasing Results on BERT. The results are reported in Table 1 It is observed that all baseline methods fail to yield satisfactory results in knowledge maintenance (i.e., DS). This proves our claim that group-invariant methods compromise the ability to distinguish between different social groups while mitigating biases. However, our FAST can largely maintain a high DS. Furthermore, our FAST is the first to achieve near-perfect bias mitigation (i.e., SS), while SS of all baselines are still higher than 56 as for StereoSet. This demonstrates the

Table 1: Debiasing Results on BERT. The best result is indicated in bold. $\diamond$ : the closer to 50 , the better. "-": results are not reported.

| Method | SS $_{\text {S-Set }} \diamond$ | SS $_{\text {Crows }} \diamond$ | PS $\diamond$ | DS $\uparrow$ | LMS $\uparrow$ | ICAT $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BERT | 60.28 | 57.25 | 59.17 | 100.0 | 84.17 | 68.11 |
| CDA | 59.61 | 56.11 | 57.56 | 75.00 | 83.08 | 70.11 |
| Dropout | 60.68 | 55.34 | 58.65 | 87.50 | 83.04 | 66.95 |
| INLP | 56.66 | 51.15 | 54.15 | 66.67 | 80.63 | 71.40 |
| SelfDebias | 59.34 | 52.29 | 57.45 | 68.75 | 84.09 | 69.92 |
| SentDebias | 59.37 | 52.29 | 56.78 | 70.83 | 84.20 | 69.56 |
| MABEL | 56.25 | 50.76 | 54.74 | 66.67 | 84.54 | 73.98 |
| AutoDebias | 59.65 | 48.43 | 57.64 | 58.33 | 86.28 | 69.64 |
| FMD | 57.77 | - | 55.43 | 70.83 | 85.45 | 72.17 |
| Ours | $\mathbf{5 1 . 1 6}$ | $\mathbf{4 9 . 6 9}$ | $\mathbf{5 0 . 8 0}$ | $\mathbf{9 5 . 8 3}$ | $\mathbf{8 6 . 3 0}$ | $\mathbf{8 4 . 2 9}$ |

effectiveness of our FAST towards eliminating social biases in LLMs.

Debiasing Results on GPT2. As for GPT2, our method can consistently surpass all the baselines in terms of SS and DS, indicating its superiority in both bias mitigation and knowledge maintenance, as shown in Table 2 FAST also enhances the ICAT score from 68.74 to 80.38 , exceeding the secondbest result by 6.86. More debiasing results and qualitative study can be referred to Appendix C

Scalibility to Larger Models. The results on large models are reported in Table 3. After debiasing, FAST induces a significant reduction in SS, and a great improvment in ICAT. Meanwhile, FAST can also largely maintain the differentiation score for larger language models. These demonstrate the consistent effectiveness of FAST on LLMs and scalability in real-world applications.

More analysis and discussion on language modeling capability, knowledge locating, computational complexity and hyper-parameters are provided in the Appendix $D$.

Table 2: Debiasing Results on GPT2.

| Method | SS $_{\text {S-Set }} \diamond$ | SS $_{\text {Crows }} \diamond$ | PS $\diamond$ | DS $\uparrow$ | LMS $\uparrow$ | ICAT $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT2 | 62.65 | 56.87 | 60.26 | 100.0 | 91.01 | 68.74 |
| CDA | 64.02 | 56.87 | 61.12 | 67.86 | 90.36 | 65.02 |
| Dropout | 63.35 | 57.63 | 64.29 | 71.00 | $\mathbf{9 0 . 4 0}$ | 64.44 |
| INLP | 59.83 | 53.44 | 57.78 | 60.71 | 73.76 | 61.38 |
| SelfDebias | 60.84 | 56.11 | 58.97 | 64.29 | 89.07 | 70.72 |
| SentDebias | 56.05 | 56.11 | 57.67 | 71.43 | 87.43 | 73.52 |
| Ours | $\mathbf{5 4 . 9 1}$ | $\mathbf{5 1 . 6 2}$ | $\mathbf{5 3 . 8 3}$ | $\mathbf{8 2 . 1 4}$ | 89.42 | $\mathbf{8 0 . 3 8}$ |

Table 3: Debiasing Results on larger models.

| Method | SS $_{\text {S-Set }} \diamond$ | SS $_{\text {Crows }} \diamond$ | PS $\diamond$ | DS $\uparrow$ | LMS $\uparrow$ | ICAT $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT2-XL | 68.70 | 65.41 | 64.35 | 100.0 | 92.79 | 58.09 |
| Ours | 60.50 | 50.94 | 56.89 | 85.71 | 89.14 | 70.42 |
| GPT-Neo | 70.40 | 63.52 | 68.23 | 100.0 | 93.47 | 55.33 |
| Ours | 60.97 | 50.96 | 60.34 | 90.48 | 84.49 | 65.95 |
| Llama-2 | 66.28 | 65.41 | 66.16 | 100.0 | 88.83 | 59.92 |
| Ours | 55.70 | 51.57 | 54.79 | 78.57 | 86.89 | 76.98 |

## 5 CONCLUSION

In this paper, we pioneer the fine-grained bias mitigation paradigm, which specifically focuses on human-relevant individual social biases/facts rather than broad group differences. We develop a novel evaluation benchmark BiasKE and propose the first Editable Fairness framework, FAST, capable of mitigating single social biases and scalable to mitigating thousands of biases concurrently. Extensive experiments across various models and datasets demonstrate the efficacy of our approach, showcasing its generalizability, specificity, and scalability. Our findings offer significant implications for future debiasing research. The limitation and future works can be referred to Appendix E.

## REFERENCES

Marion Bartl, Malvina Nissim, and Albert Gatt. Unmasking contextual stereotypes: Measuring and mitigating bert's gender bias. arXiv preprint arXiv:2010.14534, 2020.

Sid Black, Gao Leo, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, March 2021. URL https://doi.org/ 10.5281 /zenodo.5297715. If you use this software, please cite it using these metadata.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.

Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183-186, 2017.

Ruizhe Chen, Jianfei Yang, Huimin Xiong, Jianhong Bai, Tianxiang Hu, Jin Hao, Yang Feng, Joey Tianyi Zhou, Jian Wu, and Zuozhu Liu. Fast model debias with machine unlearning. arXiv preprint arXiv:2310.12560, 2023.

Sunipa Dev, Akshita Jha, Jaya Goyal, Dinesh Tewari, Shachi Dave, and Vinodkumar Prabhakaran. Building stereotype repositories with llms and community engagement for scale and depth. CrossCultural Considerations in NLP@ EACL, pp. 84, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Xiangjue Dong, Ziwei Zhu, Zhuoer Wang, Maria Teleki, and James Caverlee. Co ${ }^{2}$ pt: Mitigating bias in pre-trained language models through counterfactual contrastive prompt tuning. arXiv preprint arXiv:2310.12490, 2023.

Matthew Finlayson, Aaron Mueller, Sebastian Gehrmann, Stuart Shieber, Tal Linzen, and Yonatan Belinkov. Causal analysis of syntactic agreement mechanisms in neural language models. arXiv preprint arXiv:2106.06087, 2021.

Luciano Floridi and Massimo Chiriatti. Gpt-3: Its nature, scope, limits, and consequences. Minds and Machines, 30:681-694, 2020.

Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. Bias and fairness in large language models: A survey. arXiv preprint arXiv:2309.00770, 2023.

Yue Guo, Yi Yang, and Ahmed Abbasi. Auto-debias: Debiasing masked language models with automated biased prompts. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1012-1023, 2022.

Alex Hanna, Emily Denton, Andrew Smart, and Jamila Smith-Loud. Towards a critical race methodology in algorithmic fairness. In Proceedings of the 2020 conference on fairness, accountability, and transparency, pp. 501-512, 2020.

Jacqueline He, Mengzhou Xia, Christiane Fellbaum, and Danqi Chen. Mabel: Attenuating gender bias using textual entailment data. arXiv preprint arXiv:2210.14975, 2022.

Anne Lauscher, Tobias Lueken, and Goran Glavaš. Sustainable modular debiasing of language models. arXiv preprint arXiv:2109.03646, 2021.

Yingji Li, Mengnan Du, Xin Wang, and Ying Wang. Prompt tuning pushes farther, contrastive learning pulls closer: A two-stage approach to mitigate social biases. arXiv preprint arXiv:2307.01595, 2023.

Paul Pu Liang, Irene Mengze Li, Emily Zheng, Yao Chong Lim, Ruslan Salakhutdinov, and LouisPhilippe Morency. Towards debiasing sentence representations. arXiv preprint arXiv:2007.08100, 2020.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in Neural Information Processing Systems, 35:17359-17372, 2022.

Moin Nadeem, Anna Bethke, and Siva Reddy. Stereoset: Measuring stereotypical bias in pretrained language models. arXiv preprint arXiv:2004.09456, 2020a.

Moin Nadeem, Anna Bethke, and Siva Reddy. Stereoset: Measuring stereotypical bias in pretrained language models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020b.

Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R Bowman. Crows-pairs: A challenge dataset for measuring social biases in masked language models. arXiv preprint arXiv:2010.00133, 2020 .

Roberto Navigli, Simone Conia, and Björn Ross. Biases in large language models: Origins, inventory and discussion. ACM Journal of Data and Information Quality, 2023.

Anaelia Ovalle, Palash Goyal, Jwala Dhamala, Zachary Jaggers, Kai-Wei Chang, Aram Galstyan, Richard Zemel, and Rahul Gupta. "i'm fully who i am": Towards centering transgender and nonbinary voices to measure biases in open language generation. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, pp. 1246-1266, 2023.

Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H Miller, and Sebastian Riedel. Language models as knowledge bases? arXiv preprint arXiv:1909.01066, 2019.

Alec Radford et al. Language models are unsupervised multitask learners. OpenAI Blog, 1(8), 2019.

Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. arXiv preprint arXiv:2004.07667, 2020.

Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. Gender bias in coreference resolution. arXiv preprint arXiv:1804.09301, 2018.

Nihar Sahoo, Himanshu Gupta, and Pushpak Bhattacharyya. Detecting unintended social bias in toxic language datasets. arXiv preprint arXiv:2210.11762, 2022.

Timo Schick, Sahana Udupa, and Hinrich Schütze. Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in nlp. Transactions of the Association for Computational Linguistics, $9: 1408-1424,2021$.

Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, and Nanyun Peng. Societal biases in language generation: Progress and challenges. arXiv preprint arXiv:2105.04054, 2021.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

Kellie Webster, Xuezhi Wang, Ian Tenney, Alex Beutel, Emily Pitler, Ellie Pavlick, Jilin Chen, Ed Chi, and Slav Petrov. Measuring and reducing gendered correlations in pre-trained models. arXiv preprint arXiv:2010.06032, 2020.

Thomas Wolf et al. Transformers: State-of-the-art natural language processing, 2020.

Ke Yang, Charles Yu, Yi R Fung, Manling Li, and Heng Ji. Adept: A debiasing prompt framework. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 10780-10788, 2023.

Xinchen Yu, Eduardo Blanco, and Lingzi Hong. Hate speech and counter speech detection: Conversational context does matter. arXiv preprint arXiv:2206.06423, 2022.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Gender bias in coreference resolution: Evaluation and debiasing methods. arXiv preprint arXiv:1804.06876, 2018.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Ryan Cotterell, Vicente Ordonez, and Kai-Wei Chang. Gender bias in contextualized word embeddings. arXiv preprint arXiv:1904.03310, 2019.

Ran Zmigrod, Sabrina J Mielke, Hanna Wallach, and Ryan Cotterell. Counterfactual data augmentation for mitigating gender stereotypes in languages with rich morphology. arXiv preprint arXiv:1906.04571, 2019.
</end of paper 4>


