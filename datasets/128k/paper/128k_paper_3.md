<paper 0>
# Deep Neural Networks Motivated by Partial Differential Equations 

Lars Ruthotto ${ }^{1,3}$ and Eldad Haber ${ }^{2,3}$<br>${ }^{1}$ Emory University, Department of Mathematics and Computer Science, Atlanta, GA, USA,<br>(lruthotto@emory.edu)<br>${ }^{2}$ Department of Earth and Ocean Science, The University of British Columbia, Vancouver, BC, Canada,<br>(ehaber@eoas.ubc.ca)<br>${ }^{3}$ Xtract Technologies Inc., Vancouver, Canada, (info@xtract.tech)

December 12, 2018


#### Abstract

Partial differential equations (PDEs) are indispensable for modeling many physical phenomena and also commonly used for solving image processing tasks. In the latter area, PDE-based approaches interpret image data as discretizations of multivariate functions and the output of image processing algorithms as solutions to certain PDEs. Posing image processing problems in the infinite dimensional setting provides powerful tools for their analysis and solution. Over the last few decades, the reinterpretation of classical image processing problems through the PDE lens has been creating multiple celebrated approaches that benefit a vast area of tasks including image segmentation, denoising, registration, and reconstruction.

In this paper, we establish a new PDE-interpretation of a class of deep convolutional neural networks (CNN) that are commonly used to learn from speech, image, and video data. Our interpretation includes convolution residual neural networks (ResNet), which are among the most promising approaches for tasks such as image classification having improved the state-of-the-art performance in prestigious benchmark challenges. Despite their recent successes, deep ResNets still face some critical challenges associated with their design, immense computational costs and memory requirements, and lack of understanding of their reasoning.

Guided by well-established PDE theory, we derive three new ResNet architectures that fall into two new classes: parabolic and hyperbolic CNNs. We demonstrate how PDE theory can provide new insights and algorithms for deep learning and demonstrate the competitiveness of three new CNN architectures using numerical experiments.


Keywords: Machine Learning, Deep Neural Networks, Partial Differential Equations, PDE-Constrained Optimization, Image Classification

## 1 Introduction

Over the last three decades, algorithms inspired by partial differential equations (PDE) have had a profound impact on many processing tasks that involve speech, image, and video data. Adapting PDE models that were traditionally used in physics to perform image processing tasks has led to ground-breaking contributions. An incomplete list of seminal works includes optical flow models for motion estimation [26, nonlinear diffusion models for filtering of images [38, variational methods for image segmentation 36, 1), and nonlinear edge-preserving denoising 42 .

A standard step in PDE-based data processing is interpreting the involved data as discretizations of multivariate functions. Consequently, many operations on the data can be modeled as discretizations of PDE operators acting on the underlying functions. This continuous data model has led to solid mathematical theories for classical data processing tasks obtained by leveraging the rich results from PDEs and variational calculus (e.g., 43]). The continuous perspective has also enabled more abstract formulations that are independent of the actual resolution, which has been exploited to obtain efficient multiscale and multilevel algorithms (e.g., 34).

In this paper, we establish a new PDE-interpretation of deep learning tasks that involve speech, image, and video data as features. Deep learning is a form of machine learning that uses neural networks with many hidden layers 4, 31. Although neural networks date back at least to the 1950s 41, their popularity soared a few years ago when deep neural networks (DNNs) outperformed other machine learning methods in speech recognition [39] and image classification 25]. Deep learning also led to dramatic improvements in computer vision, e.g., surpassing human performance in image recognition 25, 29, 31. These results ignited the recent flare of research in the field. To obtain a PDE-interpretation, we use a continuous representation of the images and extend recent works by 19, 15, which relate deep learning problems for general data types to ordinary differential equations (ODE).

Deep neural networks filter input features using several layers whose operations consist of element-wise nonlinearities and affine transformations. The main idea of convolutional neural networks (CNN) 30 is to base the affine transformations on convolution operators with compactly supported filters. Supervised learning aims at learning the filters and other parameters, which are also called weights, from training data. CNNs are widely used for solving large-scale learning tasks involving data that represent a discretization of a continuous function, e.g., voice, images, and videos 29, 30, 32. By design, each CNN layer exploits the local relation between image information, which simplifies computation 39.

Despite their enormous success, deep CNNs still face critical challenges including designing a CNN architecture that is effective for a practical learning task, which requires many choices. In addition to the number of layers, also called depth of the network, important aspects are the number of convolution filters at each layer, also called the width of the layers, and the connections between those filters. A recent trend is to favor deep over wide networks, aiming at improving generalization (i.e., the performance of the CNN on new examples that were not used
during the training) 31. Another key challenge is designing the layer, i.e., choosing the combination of affine transformations and nonlinearities. A practical but costly approach is to consider depth, width, and other properties of the architecture as hyper-parameters and jointly infer them with the network weights [23]. Our interpretation of CNN architectures as discretized PDEs provides new mathematical theories to guide the design process. In short, we obtain architectures by discretizing the underlying PDE through adequate time integration methods.

In addition to substantial training costs, deep CNNs face fundamental challenges when it comes to their interpretability and robustness. In particular, CNNs that are used in mission-critical tasks (such as driverless cars) face the challenge of being "explainable." Casting the learning task within nonlinear PDE theory allows us to understand the properties of such networks better. We believe that further research into the mathematical structures presented here will result in a more solid understanding of the networks and will close the gap between deep learning and more mature fields that rely on nonlinear PDEs such as fluid dynamics. A direct impact of our approach can be observed when studying, e.g., adversarial examples. Recent works [37] indicate that the predictions obtained by deep networks can be very sensitive to perturbations of the input images. These findings motivate us to favor networks that are stable, i.e., networks whose output are robust to small perturbations of the input features, similar to what PDE analysis suggests.

In this paper, we consider residual neural networks (ResNet) 22], a very effective type of neural networks. We show that residual CNNs can be interpreted as a discretization of a space-time differential equation. We use this link for analyzing the stability of a network and for motivating new network models that bear similarities with well-known PDEs. Using our framework, we present three new architectures. First, we introduce parabolic CNNs that restrict the forward propagation to dynamics that smooth image features and bear similarities with anisotropic filtering [38, 45, 12]. Second, we propose hyperbolic CNNs that are inspired by Hamiltonian systems and finally, a third, second-order hyperbolic CNN. As to be expected, those networks have different properties. For example, hyperbolic CNNs approximately preserve the energy in the system, which sets them apart from parabolic networks that smooth the image data, reducing the energy. Computationally, the structure of a hyperbolic forward propagation can be exploited to alleviate the memory burden because hyperbolic dynamics can be made reversible on the continuous and discrete levels. The methods suggested here are closely related to reversible ResNets [16, 9.

The remainder of this paper is organized as follows. In Section 2, we provide a brief introduction into residual networks and their relation to ordinary and, in the case of convolutional neural networks, partial differential equations. In Section 3, we present three novel CNN architectures motivated by PDE theory. Based on our continuous interpretation we present regularization functionals that enforce the smoothness of the dynamical systems, in Section 4 In Section 5 we present numerical results for image classification that indicate the competitiveness of our PDE-based architectures. Finally, we highlight some directions for future research in Section 6

## 2 Residual Networks and Differential Equations

The abstract goal of machine learning is to find a function $f: \mathbb{R}^{n} \times \mathbb{R}^{p} \rightarrow \mathbb{R}^{m}$ such that $f(\cdot, \boldsymbol{\theta})$ accurately predicts the result of an observed phenomenon (e.g., the class of an image,

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-2.jpg?height=718&width=905&top_left_y=167&top_left_x=1068)

Figure 1: Classification results of the three proposed CNN architecture for four randomly selected test images from the STL10 dataset 13. The predicted and actual class probabilities are visualized using bar plots on the right of each image. While all networks reach a competitive prediction accuracy between around $74 \%$ and $78 \%$ across the whole dataset, predictions for individual images vary in some cases.

a spoken word, etc.). The function is parameterized by the weight vector $\boldsymbol{\theta} \in \mathbb{R}^{p}$ that is trained using examples. In supervised learning, a set of input features $\mathbf{y}_{1}, \ldots, \mathbf{y}_{s} \in \mathbb{R}^{n}$ and output labels $\mathbf{c}_{1}, \ldots, \mathbf{c}_{s} \in \mathbb{R}^{m}$ is available and used to train the model $f(\cdot, \boldsymbol{\theta})$. The output labels are vectors whose components correspond to the estimated probability of a particular example belonging to a given class. As an example, consider the image classification results in Fig. 1 where the predicted and actual labels are visualized using bar plots. For brevity, we denote the training data by $\mathbf{Y}=\left[\mathbf{y}_{1}, \mathbf{y}_{2}, \ldots, \mathbf{y}_{s}\right] \in \mathbb{R}^{n \times s}$ and $\mathbf{C}=\left[\mathbf{c}_{1}, \mathbf{c}_{2}, \ldots, \mathbf{c}_{s}\right] \in \mathbb{R}^{m \times s}$

In deep learning, the function $f$ consists of a concatenation of nonlinear functions called hidden layers. Each layer is composed of affine linear transformations and pointwise nonlinearities and aims at filtering the input features in a way that enables learning. As a fairly general formulation, we consider an extended version of the layer used in [22], which filters the features $\mathbf{Y}$ as follows

$$
\begin{equation*}
\mathbf{F}(\boldsymbol{\theta}, \mathbf{Y})=\mathbf{K}_{2}\left(\boldsymbol{\theta}^{(3)}\right) \sigma\left(\mathcal{N}\left(\mathbf{K}_{1}\left(\boldsymbol{\theta}^{(1)}\right) \mathbf{Y}, \boldsymbol{\theta}^{(2)}\right)\right) \tag{1}
\end{equation*}
$$

Here, the parameter vector, $\boldsymbol{\theta}$, is partitioned into three parts where $\boldsymbol{\theta}^{(1)}$ and $\boldsymbol{\theta}^{(3)}$ parameterize the linear operators $\mathbf{K}_{1}(\cdot) \in$ $\mathbb{R}^{k \times n}$ and $\mathbf{K}_{2}(\cdot) \in \mathbb{R}^{k_{\text {out }} \times k}$, respectively, and $\boldsymbol{\theta}^{(2)}$ are the parameters of the normalization layer $\mathcal{N}$. The activation function $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ is applied component-wise. Common examples are $\sigma(x)=\tanh (x)$ or the rectified linear unit (ReLU) defined as $\sigma(x)=\max (0, x)$. A deep neural network can be written by concatenating many of the layers given in 11.

When dealing with image data, it is common to group the features into different channels (e.g., for RGB image data there are three channels) and define the operators $\mathbf{K}_{1}$ and $\mathbf{K}_{2}$ as block matrices consisting of spatial convolutions. Typically each channel of the output image is computed as a weighted sum of each of the convolved input channels. To give an example, assume that $\mathbf{K}_{1}$ has three input and two output channels and denote by
$\mathbf{K}_{1}^{(\cdot, \cdot)}(\cdot)$ a standard convolution operator 21. In this case, we can write $\mathbf{K}_{1}$ as

$$
\mathbf{K}_{1}(\boldsymbol{\theta})=\left(\begin{array}{lll}
\mathbf{K}_{1}^{(1,1)}\left(\boldsymbol{\theta}^{(1,1)}\right) & \mathbf{K}_{1}^{(1,2)}\left(\boldsymbol{\theta}^{(1,2)}\right) & \mathbf{K}_{1}^{(1,3)}\left(\boldsymbol{\theta}^{(1,3)}\right)  \tag{2}\\
\mathbf{K}_{1}^{(2,1)}\left(\boldsymbol{\theta}^{(1,2)}\right) & \mathbf{K}_{1}^{(2,2)}\left(\boldsymbol{\theta}^{(2,2)}\right) & \mathbf{K}_{1}^{(2,3)}\left(\boldsymbol{\theta}^{(2,3)}\right)
\end{array}\right)
$$

where $\boldsymbol{\theta}^{(i, j)}$ denotes the parameters of the stencil of the $(i, j)$-th convolution operator.

A common choice for $\mathcal{N}$ in 1) is the batch normalization layer 27. This layer computes the empirical mean and standard deviation of each channel in the input images across the spatial dimensions and examples and uses this information to normalize the statistics of the output images. While the coupling of different examples is counter-intuitive, its use is wide-spread and motivated by empirical evidence showing a faster convergence of training algorithms. The weights $\boldsymbol{\theta}^{(2)}$ represent scaling factors and biases (i.e., constant shifts applied to all pixels in the channel) for each output channel that are applied after the normalization.

ResNets have recently improved the state-of-the-art in several benchmarks including computer vision contests on image classification 25, 29, 31. Given the input features $\mathbf{Y}_{0}=\mathbf{Y}$, a ResNet unit with $N$ layers produces a filtered version $\mathbf{Y}_{N}$ as follows

$$
\begin{equation*}
\mathbf{Y}_{j+1}=\mathbf{Y}_{j}+\mathbf{F}\left(\boldsymbol{\theta}^{(j)}, \mathbf{Y}_{j}\right), \quad \text { for } \quad j=0,1, \ldots, N-1 \tag{3}
\end{equation*}
$$

where $\boldsymbol{\theta}^{(j)}$ are the weights (convolution stencils and biases) of the $j$ th layer. To emphasize the dependency of this process on the weights, we denote $\mathbf{Y}_{N}(\boldsymbol{\theta})$.

Note that the dimension of the feature vectors (i.e., the image resolution and the number of channels) is the same across all layers of a ResNets unit, which is limiting in many practical applications. Therefore, implementations of deep CNNs contain a concatenation of ResNet units with other layers that can change, e.g., the number of channels and the image resolution (see, e.g., 22, 9]).

In image recognition, the goal is to classify the output of (3), $\mathbf{Y}_{N}(\boldsymbol{\theta})$, using, e.g., a linear classifier modeled by a fullyconnected layer, i.e., an affine transformation with a dense matrix. To avoid confusion with the ResNet units we denote these transformations as $\mathbf{W} \mathbf{Y}_{N}(\boldsymbol{\theta})+\left(\mathbf{B}_{W} \boldsymbol{\mu}\right) \mathbf{e}_{s}^{\top}$, where the columns of $\mathbf{B}_{W}$ represent a distributed bias and $\mathbf{e}_{s} \in \mathbb{R}^{s}$ is a vector of all ones. The parameters of the network and the classifier are unknown and have to be learned. Thus, the goal of learning is to estimate the network parameters, $\boldsymbol{\theta}$, and the weights of the classifier, $\mathbf{W}, \boldsymbol{\mu}$, by approximately solving the optimization problem

$$
\begin{equation*}
\min _{\boldsymbol{\theta}, \mathbf{W}, \boldsymbol{\mu}} \frac{1}{2} S\left(\mathbf{W} \mathbf{Y}_{N}(\boldsymbol{\theta})+\left(\mathbf{B}_{W} \boldsymbol{\mu}\right) \mathbf{e}_{s}^{\top}, \mathbf{C}\right)+R(\boldsymbol{\theta}, \mathbf{W}, \boldsymbol{\mu}) \tag{4}
\end{equation*}
$$

where $S$ is a loss function, which is convex in its first argument, and $R$ is a convex regularizer discussed below. Typical examples of loss functions are the least-squares function in regression and logistic regression or cross entropy functions in classification 17 .

The optimization problem in $\sqrt{4}$ is challenging for several reasons. First, it is a high-dimensional and non-convex optimization problem. Therefore one has to be content with local minima. Second, the computational cost per example is high, and the number of examples is large. Third, very deep architectures are prone to problems such as vanishing and exploding gradients 5 that may occur when the discrete forward propagation is unstable 19 .

### 2.1 Residual Networks and ODEs

We derived a continuous interpretation of the filtering provided by ResNets in 19. Similar observations were made in 15, 10. The ResNet in (3) can be seen as a forward Euler discretization (with a fixed step size of $\delta_{t}=1$ ) of the initial value problem

$$
\begin{align*}
\partial_{t} \mathbf{Y}(\boldsymbol{\theta}, t) & =\mathbf{F}(\boldsymbol{\theta}(t), \mathbf{Y}(t)), \text { for } t \in(0, T]  \tag{5}\\
\mathbf{Y}(\boldsymbol{\theta}, 0) & =\mathbf{Y}_{0}
\end{align*}
$$

Here, we introduce an artificial time $t \in[0, T]$. The depth of the network is related to the arbitrary final time $T$ and the magnitude of the matrices $\mathbf{K}_{1}$ and $\mathbf{K}_{2}$ in 11. This observation shows the relation between the learning problem (4) and parameter estimation of a system of nonlinear ordinary differential equations. Note that this interpretation does not assume any particular structure of the layer $\mathbf{F}$.

The continuous interpretation of ResNets can be exploited in several ways. One idea is to accelerate training by solving a hierarchy of optimization problems that gradually introduce new time discretization points for the weights, $\boldsymbol{\theta}$ 20. Also, new numerical solvers based on optimal control theory have been proposed in 33. Another recent work 11 uses more sophisticated time integrators to solve the forward propagation and the adjoint problem (in this context commonly called back-propagation), which is needed to compute derivatives of the objective function with respect to the network weights.

### 2.2 Convolutional ResNets and PDEs

In the following, we consider learning tasks involving features given by speech, image, or video data. For these problems, the input features, $\mathbf{Y}$, can be seen as a discretization of a continuous function $Y(x)$. We assume that the matrices $\mathbf{K}_{1} \in \mathbb{R}^{\tilde{w} \times w_{\text {in }}}$ and $\mathbf{K}_{2} \in \mathbb{R}^{w_{\text {out }} \times \tilde{w}}$ in 1) represent convolution operators [21. The parameters $w_{\text {in }}, \tilde{w}$, and $w_{\text {out }}$ denote the width of the layer, i.e., they correspond to the number of input, intermediate, and output features of this layer.

We now show that a particular class of deep residual CNNs can be interpreted as nonlinear systems of PDEs. For ease of notation, we first consider a one-dimensional convolution of a feature with one channel and then outline how the result extends to higher space dimensions and multiple channels.

Assume that the vector $\mathbf{y} \in \mathbb{R}^{n}$ represents a one-dimensional grid function obtained by discretizing $y:[0,1] \rightarrow \mathbb{R}$ at the cellcenters of a regular grid with $n$ cells and a mesh size $h=1 / n$, i.e., for $i=1,2, \ldots, n$

$$
\mathbf{y}=\left[y\left(x_{1}\right), \ldots, y\left(x_{n}\right)\right]^{\top} \quad \text { with } \quad x_{i}=\left(i-\frac{1}{2}\right) h
$$

Assume, e.g., that the operator $\mathbf{K}_{1}=\mathbf{K}_{1}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}$ in $\left.\sqrt{1}\right)$ is parameterized by the stencil $\boldsymbol{\theta} \in \mathbb{R}^{3}$. Applying a coordinate change, we see that

$$
\begin{aligned}
\mathbf{K}_{1}(\boldsymbol{\theta}) \mathbf{y} & =\left[\boldsymbol{\theta}_{1} \boldsymbol{\theta}_{2} \boldsymbol{\theta}_{3}\right] * \mathbf{y} \\
& =\left(\frac{\boldsymbol{\beta}_{1}}{4}[121]+\frac{\boldsymbol{\beta}_{2}}{2 h}[-101]+\frac{\boldsymbol{\beta}_{3}}{h^{2}}[-12-1]\right) * \mathbf{y}
\end{aligned}
$$

Here, the weights $\boldsymbol{\beta} \in \mathbb{R}^{3}$ are given by

$$
\left(\begin{array}{crr}
\frac{1}{4} & -\frac{1}{2 h} & -\frac{1}{h^{2}} \\
\frac{1}{2} & 0 & \frac{2}{h^{2}} \\
\frac{1}{4} & \frac{1}{2 h} & -\frac{1}{h^{2}}
\end{array}\right)\left(\begin{array}{l}
\boldsymbol{\beta}_{1} \\
\boldsymbol{\beta}_{2} \\
\boldsymbol{\beta}_{3}
\end{array}\right)=\left(\begin{array}{l}
\boldsymbol{\theta}_{1} \\
\boldsymbol{\theta}_{2} \\
\boldsymbol{\theta}_{3}
\end{array}\right)
$$

which is a non-singular linear system for any $h>0$. We denote by $\boldsymbol{\beta}(\boldsymbol{\theta})$ the unique solution of this linear system. Upon taking
the limit, $h \rightarrow 0$, this observation motivates one to parameterize the convolution operator as

$$
\mathbf{K}_{1}(\boldsymbol{\theta})=\boldsymbol{\beta}_{1}(\boldsymbol{\theta})+\boldsymbol{\beta}_{2}(\boldsymbol{\theta}) \partial_{x}+\boldsymbol{\beta}_{3}(\boldsymbol{\theta}) \partial_{x}^{2}
$$

The individual terms in the transformation matrix correspond to reaction, convection, diffusion and the bias term in (1) is a source/sink term, respectively. Note that higher-order derivatives can be generated by multiplying different convolution operators or increasing the stencil size.

This simple observation exposes the dependence of learned weights on the image resolution, which can be exploited in practice, e.g., by multiscale training strategies 20. Here, the idea is to train a sequence of network using a coarse-to-fine hierarchy of image resolutions (often called image pyramid). Since both the number of operations and the memory required in training is proportional to the image size, this leads to immediate savings during training but also allows one to coarsen already trained networks to enable efficient evaluation. In addition to computational benefits, ignoring fine-scale features when training on the coarse grid can also reduce the risk of being trapped in an undesirable local minimum, which is an observation also made in other image processing applications.

Our argument extends to higher spatial dimensions. In $2 \mathrm{D}$, e.g., we can relate the $3 \times 3$ stencil parametrized by $\boldsymbol{\theta} \in \mathbb{R}^{9}$ to

$$
\begin{aligned}
\mathbf{K}_{1}(\boldsymbol{\theta})= & \boldsymbol{\beta}_{1}(\boldsymbol{\theta})+\boldsymbol{\beta}_{2}(\boldsymbol{\theta}) \partial_{x}+\boldsymbol{\beta}_{3}(\boldsymbol{\theta}) \partial_{y} \\
& +\boldsymbol{\beta}_{4}(\boldsymbol{\theta}) \partial_{x}^{2}+\boldsymbol{\beta}_{5}(\boldsymbol{\theta}) \partial_{y}^{2}+\boldsymbol{\beta}_{6}(\boldsymbol{\theta}) \partial_{x} \partial_{y} \\
& +\boldsymbol{\beta}_{7}(\boldsymbol{\theta}) \partial_{x}^{2} \partial_{y}+\boldsymbol{\beta}_{8}(\boldsymbol{\theta}) \partial_{x} \partial_{y}^{2}+\boldsymbol{\beta}_{9}(\boldsymbol{\theta}) \partial_{x}^{2} \partial_{y}^{2}
\end{aligned}
$$

To obtain a fully continuous model for the layer in 11, we proceed the same way with $\mathbf{K}_{2}$. In view of $\sqrt{21}$, we note that when the number of input and output channels is larger than one, $\mathbf{K}_{1}$ and $\mathbf{K}_{2}$ lead to a system of coupled partial differential operators.

Given the continuous space-time interpretation of CNN we view the optimization problem (4) as an optimal control problem and, similarly, see learning as a parameter estimation problem for the time-dependent nonlinear PDE (5). Developing efficient numerical methods for solving PDE-constrained optimization problems arising in optimal control and parameter estimation has been a fruitful research endeavor and led to many advances in science and engineering (for recent overviews see, e.g., 7, 24, 6). Using the theoretical and algorithmic framework of optimal control in machine learning applications has gained some traction only recently (e.g., 15, 19, 9, 33, 11).

## 3 Deep Neural Networks motivated by PDEs

It is well-known that not every time-dependent PDE is stable with respect to perturbations of the initial conditions 2. Here, we say that the forward propagation in $\sqrt[5]{5}$ is stable if there is a constant $M>0$ independent of $T$ such that

$$
\begin{equation*}
\|\mathbf{Y}(\boldsymbol{\theta}, T)-\tilde{\mathbf{Y}}(\boldsymbol{\theta}, T)\|_{F} \leq M\|\mathbf{Y}(0)-\tilde{\mathbf{Y}}(0)\|_{F} \tag{6}
\end{equation*}
$$

where $\mathbf{Y}$ and $\tilde{\mathbf{Y}}$ are solutions of (5) for different initial values and $\|\cdot\|_{F}$ is the Frobenius norm. The stability of the forward propagation depends on the values of the weights $\boldsymbol{\theta}$ that are chosen by solving 4 . In the context of learning, the stability of the network is critical to provide robustness to small perturbations of the input images. In addition to image noise, perturbations could also be added deliberately to mislead the network's prediction by an adversary. There is some recent evidence showing the existence of such perturbations that reliably mislead deep networks by being barely noticeable to a human observer (e.g., 18, 37, 35]).

To ensure the stability of the network for all possible weights, we propose to restrict the space of CNNs. As examples of this general idea, we present three new types of residual CNNs that are motivated by parabolic and first- and second-order hyperbolic PDEs, respectively. The construction of our networks guarantees that the networks are stable forward and, for the hyperbolic network, stable backward in time.

Though it is common practice to model $\mathbf{K}_{1}$ and $\mathbf{K}_{2}$ in (1) independently, we note that it is, in general, hard to show the stability of the resulting network. This is because, the Jacobian of $\mathbf{F}(\boldsymbol{\theta}, \mathbf{Y})$ with respect to the features has the form

$$
\mathbf{J}_{\mathbf{Y}} \mathbf{F}=\mathbf{K}_{2}(\boldsymbol{\theta}) \operatorname{diag}\left(\sigma^{\prime}\left(\mathbf{K}_{1}(\boldsymbol{\theta} \mathbf{Y})\right)\right) \mathbf{K}_{1}(\boldsymbol{\theta})
$$

where $\sigma^{\prime}$ denotes the derivatives of the pointwise nonlinearity and for simplicity we assume $\mathcal{N}(\mathbf{Y})=\mathbf{Y}$. Even in this simplified setting, the spectral properties of $\mathbf{J}_{\mathbf{Y}}$, which impact the stability, are unknown for arbitrary choices of $\mathbf{K}_{1}$ and $\mathbf{K}_{2}$.

As one way to obtain a stable network, we introduce a symmetric version of the layer in (1) by choosing $\mathbf{K}_{2}=-\mathbf{K}_{1}^{\top}$ in (1). To simplify our notation, we drop the subscript of the operator and define the symmetric layer

$$
\begin{equation*}
\mathbf{F}_{\text {sym }}(\boldsymbol{\theta}, \mathbf{Y})=-\mathbf{K}(\boldsymbol{\theta})^{\top} \sigma(\mathcal{N}(\mathbf{K}(\boldsymbol{\theta}) \mathbf{Y}, \boldsymbol{\theta})) \tag{7}
\end{equation*}
$$

It is straightforward to verify that this choice leads to a negative semi-definite Jacobian for any non-decreasing activation function. As we see next, this choice also allows us to link the discrete network to different types of PDEs.

### 3.1 Parabolic CNN

We define the parabolic CNN by using the symmetric layer from (7) in the forward propagation, i.e., in the standard ResNet we replace the dynamic in (5) by

$$
\begin{equation*}
\partial_{t} \mathbf{Y}(\boldsymbol{\theta}, t)=\mathbf{F}_{\text {sym }}(\boldsymbol{\theta}(t), \mathbf{Y}(t)), \quad \text { for } t \in(0, T] \tag{8}
\end{equation*}
$$

Note that (8) is equivalent to the heat equation if $\sigma(x)=x$, $\mathcal{N}(\mathbf{Y})=\mathbf{Y}$ and $\mathbf{K}(t)=\nabla$. This motivates us to refer to this network as a parabolic CNN. Nonlinear parabolic PDEs are widely used, e.g., to filter images [38, 45, 12 and our interpretation implies that the networks can be viewed as an extension of such methods.

The similarity to the heat equation motivates us to introduce a new normalization layer motivated by total variation denoising. For a single example $\mathbf{y} \in \mathbb{R}^{n}$ that can be grouped into $c$ channels, we define

$$
\begin{equation*}
\mathcal{N}_{\mathrm{tv}}(\mathbf{y})=\operatorname{diag}\left(\frac{1}{\mathbf{A}^{\top} \sqrt{\mathbf{A}\left(\mathbf{y}^{2}\right)+\epsilon}}\right) \mathbf{y} \tag{9}
\end{equation*}
$$

where the operator $\mathbf{A} \in \mathbb{R}^{n / c \times n}$ computes the sum over all $c$ channels for each pixel, the square, square root, and the division are defined component-wise, $0<\epsilon \ll 1$ is fixed. As for the batch norm layer, we implement $\mathcal{N}_{\mathrm{tv}}$ with trainable weights corresponding to global scaling factors and biases for each channel. In the case that the convolution is reduced to a discrete gradient, $\mathcal{N}_{\mathrm{tv}}$ leads to the regular dynamics in TV denoising.

Stability. Parabolic PDEs have a well-known decay property that renders them robust to perturbations of the initial conditions. For the parabolic CNN in (8) we can show the following stability result.

Theorem 1 If the activation function $\sigma$ is monotonically nondecreasing, then the forward propagation through a parabolic CNN satisfies 6.

Proof 1 For ease of notation, we assume that no normalization layer is used, i.e., $\mathcal{N}(\mathbf{Y})=\mathbf{Y}$ in (8). We then show that $\mathbf{F}_{\text {sym }}(\boldsymbol{\theta}(t), \mathbf{Y})$ is a monotone operator. Note that for all $t \in[0, T]$

$$
-\left(\sigma(\mathbf{K}(t) \mathbf{Y})-\sigma\left(\mathbf{K}(t) \mathbf{Y}_{\epsilon}\right), \mathbf{K}(t)\left(\mathbf{Y}-\mathbf{Y}_{\epsilon}\right)\right) \leq 0
$$

Where $(\cdot, \cdot)$ is the standard inner product and the inequality follows from the monotonicity of the activation function, which shows that

$$
\partial_{t}\left\|\mathbf{Y}(t)-\mathbf{Y}_{\epsilon}(t)\right\|_{F}^{2} \leq 0
$$

Integrating this inequality over $[0, T]$ yields stability as in 6 . The proof extends straightforwardly to cases when a normalization layer with scaling and bias is included.

One way to discretize the parabolic forward propagation 88 is using the forward Euler method. Denoting the time step size by $\delta_{t}>0$ this reads

$$
\mathbf{Y}_{j+1}=\mathbf{Y}_{j}+\delta_{t} \mathbf{F}_{\mathrm{sym}}\left(\boldsymbol{\theta}\left(t_{j}\right), \mathbf{Y}_{j}\right), \quad j=0,1, \ldots, N-1
$$

where $t_{j}=j \delta_{t}$. The discrete forward propagation of a given example $\mathbf{y}_{0}$ is stable if $\delta_{t}$ satisfies

$$
\max _{i=1,2, \ldots, n}\left|1+\delta_{t} \lambda_{i}\left(\mathbf{J}\left(t_{j}\right)\right)\right| \leq 1, \quad j=0,1, \ldots, N-1
$$

and accurate if $\delta_{t}$ is chosen small enough to capture the dynamics of the system. Here, $\lambda_{i}\left(\mathbf{J}\left(t_{j}\right)\right)$ denotes the $i$ th eigenvalue of the Jacobian of $\mathbf{F}_{\text {sym }}$ with respect to the features at a time point $t_{j}$. If we assume, for simplicity, that no normalization layer is used, the Jacobian is

$$
\begin{aligned}
\mathbf{J}\left(t_{j}\right)= & -\mathbf{K}^{\top}\left(\boldsymbol{\theta}^{(1)}\left(t_{j}\right)\right) \mathbf{D}\left(t_{j}\right) \mathbf{K}\left(\boldsymbol{\theta}^{(1)}\left(t_{j}\right)\right) \\
& \text { with } \quad \mathbf{D}(t)=\operatorname{diag}\left(\sigma^{\prime}\left(\mathbf{K}\left(\boldsymbol{\theta}^{(1)}(t)\right) \mathbf{y}(t)\right)\right)
\end{aligned}
$$

If the activation function is monotonically nondecreasing, then $\sigma^{\prime}(\cdot) \geq 0$ everywhere. In this case, all eigenvalues of $\mathbf{J}\left(t_{j}\right)$ are real and bounded above by zero since $\mathbf{J}\left(t_{j}\right)$ is also symmetric. Thus, there is an appropriate $\delta_{t}$ that renders the discrete forward propagation stable. In our numerical experiments, we aim at ensuring the stability of the discrete forward propagation by limiting the magnitude of elements in $\mathbf{K}$ by adding bound constraints to the optimization problem 4 .

### 3.2 Hyperbolic CNNs

Different types of networks can be obtained by considering hyperbolic PDEs. In this section, we present two CNN architectures that are inspired by hyperbolic systems. A favorable feature of hyperbolic equations is their reversibility. Reversibility allows us to avoid storage of intermediate network states, thus achieving higher memory efficiency. This is particularly important for very deep networks where memory limitation can hinder training (see [16] and 9]).

Hamiltonian CNNs. Introducing an auxiliary variable $\mathbf{Z}$ (i.e., by partitioning the channels of the original features), we consider the dynamics

$$
\begin{aligned}
\partial_{t} \mathbf{Y}(t) & =-\mathbf{F}_{\text {sym }}\left(\boldsymbol{\theta}^{(1)}(t), \mathbf{Z}(t)\right), & & \mathbf{Y}(0)=\mathbf{Y}_{0} \\
\partial_{t} \mathbf{Z}(t) & =\mathbf{F}_{\text {sym }}\left(\boldsymbol{\theta}^{(2)}(t), \mathbf{Y}(t)\right), & & \mathbf{Z}(0)=\mathbf{Z}_{0}
\end{aligned}
$$

We showed in 9 that the eigenvalues of the associated Jacobian are imaginary. When assuming that $\boldsymbol{\theta}^{(1)}$ and $\boldsymbol{\theta}^{(2)}$ change sufficiently slow in time stability as defined in 6) is obtained. A more precise stability result can be established by analyzing the kinematic eigenvalues of the forward propagation 3.

We discretize the dynamic using the symplectic Verlet integration (see, e.g., 2] for details)

$$
\begin{align*}
& \mathbf{Y}_{j+1}=\mathbf{Y}_{j}+\delta_{t} \mathbf{F}_{\mathrm{sym}}\left(\boldsymbol{\theta}^{(1)}(t), \mathbf{Z}_{j}\right) \\
& \mathbf{Z}_{j+1}=\mathbf{Z}_{j}-\delta_{t} \mathbf{F}_{\mathrm{sym}}\left(\boldsymbol{\theta}^{(2)}(t), \mathbf{Y}_{j+1}\right) \tag{10}
\end{align*}
$$

for $j=0,1, \ldots, N-1$ using a fixed step size $\delta_{t}>0$. This dynamic is reversible, i.e., given $\mathbf{Y}_{N}, \mathbf{Y}_{N-1}$ and $\mathbf{Z}_{N}, \mathbf{Z}_{N-1}$ it can also be computed backwards

$$
\begin{aligned}
\mathbf{Z}_{j} & =\mathbf{Z}_{j+1}+\delta_{t} \mathbf{F}_{\text {sym }}\left(\boldsymbol{\theta}^{(2)}(t), \mathbf{Y}_{j+1}\right) \\
\mathbf{Y}_{j} & =\mathbf{Y}_{j+1}-\delta_{t} \mathbf{F}_{\text {sym }}\left(\boldsymbol{\theta}^{(1)}(t), \mathbf{Y}_{j+1}\right)
\end{aligned}
$$

for $j=N-1, N-2, \ldots, 0$. These operations are numerically stable for the Hamiltonian CNN (see 9 for details).

Second-order CNNs. An alternative way to obtain hyperbolic CNNs is by using a second-order dynamics

$$
\begin{align*}
\partial_{t}^{2} \mathbf{Y}(t) & =\mathbf{F}_{\text {sym }}(\boldsymbol{\theta}(t), \mathbf{Y}(t)) \\
\mathbf{Y}(0) & =\mathbf{Y}_{0}, \partial_{t} \mathbf{Y}(0)=0 \tag{11}
\end{align*}
$$

The resulting forward propagation is associated with a nonlinear version of the telegraph equation 40, which describes the propagation of signals through networks. Hence, one could claim that second-order networks better mimic biological networks and are therefore more appropriate than first-order networks for approaches that aim at imitating the propagation through biological networks.

We discretize the second-order network using the Leapfrog method. For $j=0,1, \ldots, N-1$ and $\delta_{t}>0$ fixed this reads

$$
\mathbf{Y}_{j+1}=2 \mathbf{Y}_{j}-\mathbf{Y}_{j-1}+\delta_{t}^{2} \mathbf{F}_{\mathrm{sym}}\left(\boldsymbol{\theta}\left(t_{j}\right), \mathbf{Y}_{j}\right)
$$

We set $\mathbf{Y}_{-1}=\mathbf{Y}_{0}$ to denote the initial condition. Similar to the symplectic integration in 100, this scheme is reversible.

We show that the second-order network is stable in the sense of (6) when we assume stationary weights. Weaker results for the time-dependent dynamic are possible assuming $\partial_{t} \boldsymbol{\theta}(t)$ to be bounded.

Theorem 2 Let $\boldsymbol{\theta}(t)$ be constant in time and assume that the activation function satisfies $|\sigma(x)| \leq|x|$ for all $x$. Then, the forward propagation through the second-order network satisfies (6).

Proof 2 For brevity, we denote $\mathbf{K}=\mathbf{K}(\boldsymbol{\theta}(t))$ and consider the forward propagation of a single example. Let $\mathbf{y}:[0, T] \rightarrow \mathbb{R}^{n}$ be a solution to 11) and consider the energy

$$
\begin{equation*}
\mathcal{E}(t)=\frac{1}{2}\left(\left(\partial_{t} \mathbf{y}(t)\right)^{\top} \partial_{t} \mathbf{y}(t)+(\mathbf{K} \mathbf{y}(t))^{\top} \sigma(\mathbf{K} \mathbf{y}(t))\right) \tag{12}
\end{equation*}
$$

Given that $|\sigma(x)| \leq|x|$ for all $x$ by assumption, this energy can be bounded as follows

$$
\begin{aligned}
\mathcal{E}(t) & \leq \mathcal{E}_{\operatorname{lin}}(t) \\
& =\frac{1}{2}\left(\left(\partial_{t} \mathbf{u}(t)\right)^{\top} \partial_{t} \mathbf{u}(t)+(\mathbf{K u}(t))^{\top}(\mathbf{K u}(t))\right)
\end{aligned}
$$

where $\mathcal{E}_{\text {lin }}$ is the energy associated with the linear wave-like hyperbolic equation

$$
\partial_{t}^{2} \mathbf{u}(t)=-\mathbf{K}^{\top} \mathbf{K} \mathbf{u}(t), \mathbf{u}(0)=\mathbf{y}_{0}, \quad \partial_{t} \mathbf{u}(0)=0
$$

Since by assumption $\mathbf{K}$ is constant in time, we have that

$$
\partial_{t} \mathcal{E}_{\operatorname{lin}}(t)=\partial_{t} \mathbf{u}(t)^{\top}\left(\partial_{t}^{2} \mathbf{u}(t)+\mathbf{K}^{\top} \mathbf{K} \mathbf{u}(t)\right)=0
$$

Thus, the energy of the hyperbolic network in 12 is positive and bounded from above by the energy of the linear wave equation. Applying this argument to the initial condition $\mathbf{y}_{0}-\mathbf{y}_{\epsilon}$ we derive (6) and thus the forward propagation is stable.

## 4 Regularization

The proposed continuous interpretation of the CNNs also provides new perspectives on regularization. To enforce stability of the forward propagation, the linear operator $\mathbf{K}$ in (7) should not change drastically in time. This suggests adding a smoothness regularizer in time. In 19 a $H^{1}$-seminorm was used to smooth kernels over time with the goal to avoid overfitting. A theoretically more appropriate function space consists of all kernels that are piecewise smooth in time. To this end, we introduce the regularizer

$$
\begin{align*}
R(\boldsymbol{\theta}, \mathbf{W}, \boldsymbol{\mu}) & =\alpha_{1} \int_{0}^{T} \phi_{\tau}\left(\partial_{t} \boldsymbol{\theta}(t)\right) d t \\
& +\frac{\alpha_{2}}{2}\left(\int_{0}^{T}\|\boldsymbol{\theta}(t)\|^{2} d t+\|\mathbf{W}\|_{F}^{2}+\|\boldsymbol{\mu}\|^{2}\right) \tag{13}
\end{align*}
$$

where the function $\phi_{\epsilon}(x)=\sqrt{x^{2}+\tau}$ is a smoothed $\ell_{1}$-norm with conditioning parameter $\tau>0$. The first term of $R$ can be seen as a total variation [42] penalty in time that favors piecewise constant dynamics. Here, $\alpha_{1}, \alpha_{2} \geq 0$ are regularization parameters that are assumed to be fixed.

A second important aspect of stability is to keep the time step sufficiently small. Since $\delta_{t}$ can be absorbed in $\mathbf{K}$ we use the box constraint $-1 \leq \boldsymbol{\theta}^{(1)}\left(t_{j}\right) \leq 1$ for all $j$, and fix the time step size to $\delta_{t}=1$ in our numerical experiments.

## 5 Numerical Experiments

We demonstrate the potential of the proposed architectures using the common image classification benchmarks STL-10 [13, CIFAR-10, and CIFAR-100 28. Our central goal is to show that, despite their modeling restrictions, our new network types achieve competitive results. We use our basic architecture for all experiments, do not excessively tune hyperparameters individually for each case, and employ a simple data augmentation technique consisting of random flipping and cropping.

Network Architecture. Our architecture is similar to the ones in [22, 9 and contains an opening layer, followed by several blocks each containing a few time steps of a ResNet and a connector that increases the width of the CNN and coarsens the images. Our focus is on the different options for defining the ResNet block using parabolic and hyperbolic networks. To this end, we choose the same basic components for the opening and connecting layers. The opening layer increases the number of channels from three (for RGB image data) to the number of channels of the first ResNet using convolution operators with $3 \times 3$ stencils, a batch normalization layer and a ReLU activation function. We build the connecting layers using $1 \times 1$ convolution operators that increase the number of channels, a batch normalization layer, a ReLU activation, and an average pooling operator that coarsens the images by a factor of two. Finally, we obtain the output features $\mathbf{Y}(\boldsymbol{\theta})$ by averaging the features of each channel to ensure translation-invariance. The ResNet blocks use the symmetric layer (7) including the total variation normalization (9) with $\epsilon=10^{-3}$. The classifier is modeled using a fully-connected layer, a softmax transformation, and a crossentropy loss.

Training Algorithm. In order to estimate the weights, we use a standard stochastic gradient descent (SGD) method with momentum of 0.9. We use a piecewise constant step size (in this context also called learning rate), starting with 0.1 , which is decreased by a factor of 0.2 at a-priori specified epochs. For STL-10 and CIFAR-10 examples, we perform 60,20 , and 20 epochs with step sizes of $0.1,0.02,0.004$, respectively. For the more challenging CIFAR-100 data set, we use $60,40,40,40$, and 20 epochs with step sizes of $0.1,0.02,0.004,0.0008,0.00016$, respectively. In all examples, the SGD steps are computed using mini-batches consisting of 125 randomly chosen examples. For data augmentation, we apply a random horizontal flip ( $50 \%$ probability), pad the images by a factor of $1 / 16$ with zeros into all directions and randomly crop the image by $1 / 8$ of the pixels, counting from the lower-left corner. The training is performed using the opensource software Meganet on a workstation running Ubuntu 16.04 and MATLAB 2018b with two Intel(R) Xeon(R) CPU E5-2620 v4 and $64 \mathrm{~GB}$ of RAM. We use NVIDIA Titan X GPU for accelerating the computation through the frameworks CUDA 9.1 and $\mathrm{CuDNN} 7.0$.

Results for STL-10. The STL-10 dataset 13 contains 13,000 digital color images of size $96 \times 96$ that are evenly divided into ten categories, which can be inferred from Fig. 1 The dataset is split into 5,000 training and 8,000 test images. The STL-10 data is a popular benchmark test for image classification algorithms and challenging due to the relatively small number of training images.

For each dynamic, the network uses four ResNet blocks with 16, 32,64 , and 128 channels and image sizes of $96 \times 96,48 \times 48$, $24 \times 24,12 \times 12$, respectively. Within the ResNet blocks, we perform three time steps with a step size of $\delta_{t}=1$ and include a total variation normalization layer and ReLU activation. This architecture leads to 324,794 trainable weights for the Hamiltonian network and 618,554 weights for the parabolic and secondorder network, respectively. We note that our network are substantially smaller than commonly used ResNets. For example, the architectures in 99 contain about 2 million parameters. Reducing the number of parameters is important during training and, e.g., when trained networks have to be deployed on devices with limited memory. The regularization parameters are $\alpha_{1}=4 \cdot 10^{-4}$ and $\alpha_{2}=1 \cdot 10^{-4}$.

To show how the generalization improves as more training data becomes available, we train the network with an increasing number of examples that we choose randomly from the training dataset. We also randomly sample 1,000 examples from the remaining training data to build a validation set, which we use to monitor the performance after each full epoch. We use no data augmentation in this experiment. In all cases, the training accuracy was close to $100 \%$. After the training, we compute the accuracy of the networks parameterized by the weights that performed best on the validation data for all the 8,000 test images; see Fig. 2. The predictions of the three networks may vary for single examples without any apparent pattern (see also Fig. 1). However, overall their performance and convergence are comparable which leads to similarities in the confusion matrices; see Fig. 3 .

To show the overall performance of the networks, we train the networks using a random partition of the examples into 4,000

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-7.jpg?height=889&width=550&top_left_y=149&top_left_x=321)

A. Generalization for partial data
![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-7.jpg?height=854&width=546&top_left_y=180&top_left_x=320)

Figure 2: Performance of the three proposed architectures for the STL-10 dataset. Top: Improvement of test accuracy when increasing the number of training images ( $10 \%$ to $80 \%$ in increments of $10 \%$ ). Bottom: Validation accuracy on remaining $20 \%$ of training examples at every epoch of the stochastic gradient descent method. In this example, the parabolic and first-order hyperbolic architectures outperform the second-order network.

training and 1,000 validation data. For data augmentation, we use horizontal flipping and random cropping. The performance of the networks on the validation data after each epoch can be seen in the bottom plot of Fig. 2 As before, the optimization found weights that almost perfectly fit the training data. After the training, we compute the loss and classification accuracy for all the test images. For this example, the parabolic and Hamiltonian network perform slightly superior to the secondorder network $77.0 \%$ and $78.3 \%$ vs. $74.3 \%$ test classification accuracy, respectively. It is important to emphasize that the Hamiltonian network achieves the best test accuracy using only about half as many trainable weights as the other two networks. These results are competitive with the results reported, e.g., in [44, 14 . Fine-tuning of hyperparameters such as step size, number of time steps, and width of the network may achieve additional improvements for each dynamic. Using these means and performing training on all 5,000 images we achieved a test accuracy of around $85 \%$ in 9 .

Results for CIFAR 10/100. For an additional comparison of the proposed architectures we use the CIFAR-10 and CIFAR-100 datasets 28. Each of these datasets consists of 60,000 labeled RGB images of size $32 \times 32$ that are chosen from the 80 million tiny images dataset. In both cases, we use 50,000 images for training and validation and keep the remaining 10,000 to test the generalization of the trained weights. While CIFAR-10 consists of 10 categories the CIFAR-100 dataset contains 100 categories and, thus, classification is more challenging.

Our architectures contain three blocks of parabolic or hyperbolic networks between which the image size is reduced from $32 \times 32$ to $8 \times 8$. For the simpler CIFAR-10 problem, we use a narrower network with $32,64,112$ channels while for the CIFAR100 challenge we use more channels ( 32,64 , and 128) and add a final connecting layer that increases the number of channels to 256. This leads to networks whose number of trainable weights vary between 264,106 and 652,484 ; see also Table 11 As regularization parameters we use $\alpha_{1}=2 \cdot 10^{-4}$ and $\alpha_{2}=2 \cdot 10^{-4}$, which is similar to 9 .

As for the STL-10 data set, the three proposed architectures achieved comparable results on these benchmarks; see convergence plots in Figure 4 and test accuracies in Table 1 In all cases, the training loss is near zero after the training. For these datasets, the second-order network slightly outperforms the other networks. Additional tuning of the learning rate, regularization parameter, and other hyperparameters may further improve the results shown here. Using those techniques architectures with more time-steps and the entire training dataset we achieved about $5 \%$ higher accuracy on CIFAR-10 and $9 \%$ higher accuracy on CIFAR-100 in 9 .

## 6 Discussion and Outlook

In this paper, we establish a link between deep residual convolutional neural networks and PDEs. The relation provides a general framework for designing, analyzing, and training those CNNs. It also exposes the dependence of learned weights on the image resolution used in training. Exemplarily, we derive three PDE-based network architectures that are forward stable (the parabolic network) and forward-backward stable (the hyperbolic networks).

It is well-known that different types of PDEs have different properties. For example, linear parabolic PDEs have decay properties while linear hyperbolic PDEs conserve energy. Hence, it is common to choose different numerical techniques for solving and optimizing different kinds of PDEs. The type of the underlying PDE is not known a-priori for a standard convolutional ResNet as it depends on the trained weights. This renders ensuring the stability of the trained network and the choice of adequate time-integration methods difficult. These considerations motivate us to restrict the convolutional ResNet architecture a-priori to discretizations of nonlinear PDEs that are stable.

In our numerical examples, our new architectures lead to an adequate performance despite the constraints on the networks. In fact, using only networks of relatively modest size, we obtain results that are close to those of state-of-the-art networks with a considerably larger number of weights. This may not hold in general, and future research will show which types of architectures are best suited for a learning task at hand. Our intuition is that, e.g., hyperbolic networks may be preferable over parabolic ones for image extrapolation tasks to ensure the preservation of edge information in the images. In contrast to that, we anticipate parabolic networks to perform superior for tasks that require filtering, e.g., image denoising.

We note that our view of CNNs mirrors the developments in PDE-based image processing in the 1990s. PDE-based methods have since significantly enhanced our mathematical understanding of image processing tasks and opened the door to many popular algorithms and techniques. We hope that continuous models of CNNs will result in similar breakthroughs and, e.g., help streamline the design of network architectures and improve training outcomes with less trial and error.

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=606&width=1850&top_left_y=147&top_left_x=143)

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=163&width=209&top_left_y=165&top_left_x=145)
![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=496&width=790&top_left_y=168&top_left_x=343)

500

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=328&width=160&top_left_y=329&top_left_x=343)

1,000

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=358&width=176&top_left_y=328&top_left_x=495)

1,500

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=152&width=157&top_left_y=501&top_left_x=667)

2,000
3,000
![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=484&width=166&top_left_y=172&top_left_x=1120)

3,500

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=493&width=182&top_left_y=163&top_left_x=1275)

4,000
![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-8.jpg?height=574&width=534&top_left_y=152&top_left_x=1440)

Figure 3: Confusion matrices for classifiers obtained using the three proposed architectures (row-wise) for an increasing number of training data from the STL-10 dataset (column-wise). The $(i, j)$ th element of the $10 \times 10$ confusion matrix counts the number of images of class $i$ for which the predicted class is $j$. We use the entire test data set, which contains 800 images per class.

Table 1: Summary of numerical results for the STL-10, CIFAR-10, and CIFAR-100 datasets. In each experiment, we randomly split the training data into $80 \%$ used to train the weights and $20 \%$ used to validate the performance after each epoch. After training, we compute and report the classification accuracy and the value of cross entropy loss (in brackets) for the test data. We evaluate the performance using the weights with the best classification accuracy on the validation set. We also report the number of trainable weights for each network.

|  | STL-10 |  | CIFAR-10 |  | CIFAR-100 |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | number of <br> weights | test data (8,000) <br> accuracy \%(loss) | number of <br> weights | test data (10,000) <br> accuracy \%(loss) | number of <br> weights | test data (10,000) <br> accuracy \%(loss) |
| Parabolic | 618,554 | $77.0 \%(0.711)$ | 502,570 | $88.5 \%(0.333)$ | 652,484 | $64.8 \%(1.234)$ |
| Hamiltonian | 324,794 | $78.3 \%(0.789)$ | 264,106 | $89.3 \%(0.349)$ | 362,180 | $64.9 \%(1.237)$ |
| Second-order | 618,554 | $74.3 \%(0.810)$ | 502,570 | $89.2 \%(0.333)$ | 652,484 | $65.4 \%(1.232)$ |

## Acknowledgements

L.R. is supported by the U.S. National Science Foundation (NSF) through awards DMS 1522599 and DMS 1751636 and by the NVIDIA Corporation's GPU grant program. We thank Martin Burger for outlining how to show stability using monotone operator theory and Eran Treister and other contributors of the Meganet package. We also thank the Isaac Newton Institute (INI) for Mathematical Sciences for support and hospitality during the programme on Generative Models, Parameter Learning and Sparsity (VMVW02) when work on this paper was undertaken. INI was supported by EPSRC Grant Number: LNAG/036, RG91310.

## References

[1] L. Ambrosio and V. M. Tortorelli. Approximation of Functionals Depending on Jumps by Elliptic Functionals via Gamma-Convergence. Commun. Pure Appl. Math., 43(8):999-1036, 1990 .

[2] U. Ascher. Numerical methods for Evolutionary Differential Equations. SIAM, Philadelphia, USA, 2010.

[3] U. Ascher, R. Mattheij, and R. Russell. Numerical Solution of Boundary Value Problems for Ordinary Differential Equations. SIAM, Philadelphia, Philadelphia, 1995.

[4] Y. Bengio et al. Learning deep architectures for AI. Found. Trends Mach. Learn., 2(1):1-127, 2009.

[5] Y. Bengio, P. Simard, and P. Frasconi. Learning Long-Term Dependencies with Gradient Descent Is Difficult. IEEE Transactions on Neural Networks, $5(2): 157-166,1994$.

[6] L. T. Biegler, O. Ghattas, M. Heinkenschloss, D. Keyes, and B. van Bloemen Waanders, editors. Real-time PDE-constrained Optimization. Society for Industrial and Applied Mathematics (SIAM), 2007.

[7] A. Borzì and V. Schulz. Computational optimization of systems governed by partial differential equations, volume 8. Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA, 2012.

[8] T. F. Chan and L. A. Vese. Active contours without edges. IEEE Trans Image Process., 10(2):266-277, 2001

[9] B. Chang, L. Meng, E. Haber, L. Ruthotto, D. Begert, and E. Holtham. Reversible architectures for arbitrarily deep residual neural networks. In AAAI Conference on AI, 2018.
[10] P. Chaudhari, A. Oberman, S. Osher, S. Soatto, and G. Carlier. Deep Relaxation: Partial Differential Equations for Optimizing Deep Neural Networks. arXiv preprint 1704.04932 , Apr. 2017.

[11] T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud. Neural ordinary differential equations. arXiv preprint arXiv:1806.07366, 2018

[12] Y. Chen and T. Pock. Trainable Nonlinear Reaction Diffusion: A Flexible Framework for Fast and Effective Image Restoration. IEEE Trans. Pattern Anal. Mach. Intell., 39(6):1256-1272, 2017

[13] A. Coates, A. Ng, and H. Lee. An Analysis of Single-Layer Networks in Unsupervised Feature Learning. In Proceedings of the 14 th International Conference on Artificial Intelligence and Statistics, pages 215-223, June 2011.

[14] A. Dundar, J. Jin, and E. Culurciello. Convolutional Clustering for Unsupervised Learning. In $I C L R$, Nov. 2015.

[15] W. E. A Proposal on Machine Learning via Dynamical Systems. Comm. Math. Statist., 5(1):1-11, 2017 .

[16] A. N. Gomez, M. Ren, R. Urtasun, and R. B. Grosse. The reversible residual network: Backpropagation without storing activations. In Adv Neural Inf Process Syst, pages 2211-2221, 2017.

[17] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT Press, Nov. 2016.

[18] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and Harnessing Adversarial Examples. arXiv.org, Dec. 2014.

[19] E. Haber and L. Ruthotto. Stable architectures for deep neural networks. Inverse Probl., 34:014004, 2017

[20] E. Haber, L. Ruthotto, and E. Holtham. Learning across scales - A multiscale method for convolution neural networks. In AAAI Conference on AI, volume abs/1703.02009, pages 1-8, 2017.

[21] P. C. Hansen, J. G. Nagy, and D. P. O'Leary. Deblurring Images: Matrices, Spectra and Filtering. Matrices, Spectra, and Filtering. SIAM, Philadelphia, USA, 2006.

[22] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages $770-778,2016$.

[23] J. M. Hernández-Lobato, M. A. Gelbart, R. P. Adams, M. W. Hoffman, and Z. Ghahramani. A general framework for constrained bayesian optimization using information-based search. J. Mach. Learn. Res., 17:2-51, 2016.

[24] R. Herzog and K. Kunisch. Algorithms for PDE-constrained optimization. GAMM-Mitteilungen, 33(2):163-176, Oct. 2010 .

[25] G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE Signal Process. Mag., 29(6):82-97, 2012.

[26] B. K. Horn and B. G. Schunck. Determining optical flow. Artificial intelligence, $17(1-3): 185-203,1981$.

[27] S. Ioffe and C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In 32nd International Conference on Machine Learning, pages 448-456, Feb. 2015.

![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-9.jpg?height=903&width=556&top_left_y=145&top_left_x=318)

A. Convergence for CIFAR-10
![](https://cdn.mathpix.com/cropped/2024_06_04_b6631205f3b01875104bg-9.jpg?height=850&width=546&top_left_y=187&top_left_x=323)

Figure 4: Performance of the three proposed architectures for the CIFAR-10 (top) and CIFAR-100 (bottom) datasets. Validation accuracy computed on 10,000 randomly chosen images is shown at every epoch of the stochastic gradient descent method. In this example, all architectures perform comparably with the second-order network slightly outperforming the parabolic and first-order hyperbolic architectures.

[28] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. 2009

[29] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. Adv Neural Inf Process Syst, 61:10971105, 2012

[30] Y. LeCun and Y. Bengio. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361:255258 1995

[31] Y. LeCun, Y. Bengio, and G. Hinton. Deep learning. Nature, 521(7553):436444,2015 .

[32] Y. LeCun, K. Kavukcuoglu, and C. Farabet. Convolutional networks and applications in vision. IEEE International Symposium on Circuits and Systems: Nano-Bio Circuit Fabrics and Systems, page 253256, 2010.

[33] Q. Li, L. Chen, C. Tai, and E. Weinan. Maximum principle based algorithms for deep learning. The Journal of Machine Learning Research, 18(1):5998 $6026,2017$.

[34] J. Modersitzki. FAIR: Flexible Algorithms for Image Registration. Fundamentals of Algorithms. SIAM, Philadelphia, USA, 2009.

[35] S. M. Moosavi-Dezfooli, A. Fawzi, O. F. arXiv, and 2017. Universal adversarial perturbations. openaccess.thecvf.com.

[36] D. Mumford and J. Shah. Optimal Approximations by Piecewise Smooth Functions and Associated Variational-Problems. Commun. Pure Appl. Math., 42(5):577-685, 1989

[37] K. Pei, Y. Cao, J. Yang, and S. Jana. Deepxplore: Automated whitebox testing of deep learning systems. In 26th Symposium on Oper. Sys. Princ., pages 1-18. ACM Press, New York, USA, 2017

[38] P. Perona and J. Malik. Scale-space and edge detection using anisotropic diffusion. IEEE Trans. Pattern Anal. Mach. Intell., 12(7):629-639, 1990.

[39] R. Raina, A. Madhavan, and A. Y. Ng. Large-scale deep unsupervised learning using graphics processors. In 26th Annual International Conference pages 873-880, New York, USA, 2009. ACM.

[40] C. Rogers and T. Moodie. Wave Phenomena: Modern Theory and Applications. North-Holland Mathematics Studies. Elsevier Science, 1984

[41] F. Rosenblatt. The perceptron: A probabilistic model for information storage and organization in the brain. Psychological review, 65(6):386-408, 1958

[42] L. I. Rudin, S. Osher, and E. Fatemi. Nonlinear Total Variation Based Noise Removal Algorithms. Physica D, 60(1-4):259-268, 1992.

[43] O. Scherzer, M. Grasmair, H. Grossauer, M. Haltmeier, and F. Lenzen Variational methods in imaging. Springer, New York, USA, 2009

[44] P. L. C. C. L. W. K. S. X. T. Shuo Yang. Deep Visual Representation Learning with Target Coding. In AAAI Conference on AI, pages 3848-3854, Jan. 2015.

[45] J. Weickert. Anisotropic Diffusion in Image Processing. 2009

</end of paper 0>


<paper 1>
# SPHERICAL CNNS ON UNSTRUCTURED GRIDS 

Chiyu "Max" Jiang<br>UC Berkeley<br>Prabhat<br>Lawrence Berkeley Nat'l Lab

Jingwei Huang<br>Stanford University

Philip Marcus

UC Berkeley

Karthik Kashinath<br>Lawrence Berkeley Nat'l Lab

Matthias Nießner<br>Technical University of Munich


#### Abstract

We present an efficient convolution kernel for Convolutional Neural Networks (CNNs) on unstructured grids using parameterized differential operators while focusing on spherical signals such as panorama images or planetary signals. To this end, we replace conventional convolution kernels with linear combinations of differential operators that are weighted by learnable parameters. Differential operators can be efficiently estimated on unstructured grids using one-ring neighbors, and learnable parameters can be optimized through standard back-propagation. As a result, we obtain extremely efficient neural networks that match or outperform state-of-the-art network architectures in terms of performance but with a significantly smaller number of network parameters. We evaluate our algorithm in an extensive series of experiments on a variety of computer vision and climate science tasks, including shape classification, climate pattern segmentation, and omnidirectional image semantic segmentation. Overall, we (1) present a novel CNN approach on unstructured grids using parameterized differential operators for spherical signals, and (2) show that our unique kernel parameterization allows our model to achieve the same or higher accuracy with significantly fewer network parameters.


## 1 INTRODUCTION

A wide range of machine learning problems in computer vision and related areas require processing signals in the spherical domain; for instance, omnidirectional RGBD images from commercially available panorama cameras, such as Matterport (Chang et al. 2017), panaramic videos coupled with LIDAR scans from self-driving cars (Geiger et al. |2013), or planetary signals in scientific domains such as climate science (Racah et al., 2017). Unfortunately, naively mapping spherical signals to planar domains results in undesirable distortions. Specifically, projection artifacts near polar regions and handling of boundaries makes learning with 2D convolutional neural networks (CNNs) particularly challenging and inefficient. Very recent work, such as Cohen et al. (2018) and Esteves et al. (2018), propose network architectures that operate natively in the spherical domain, and are invariant to rotations in the $\mathcal{S O}(3)$ group. Such invariances are desirable in a set of problems - e.g., machine learning problems of molecules - where gravitational effects are negligible and orientation is arbitrary. However, for other different classes of problems at large, assumed orientation information is crucial to the predictive capability of the network. A good example of such problems is the MNIST digit recognition problem, where orientation plays an important role in distinguishing digits " 6 " and " 9 ". Other examples include omnidirectional images, where images are naturally oriented by gravity; and planetary signals, where planets are naturally oriented by their axis of rotation.

In this work, we present a new convolution kernel for CNNs on arbitrary manifolds and topologies, discretized by an unstructured grid (i.e., mesh), and focus on its applications in the spherical domain approximated by an icosahedral spherical mesh. We propose and evaluate the use of a new parameterization scheme for CNN convolution kernels, which we call Parameterized Differential Operators (PDOs), which is easy to implement on unstructured grids. We call the resulting convolution operator that operates on the mesh using such kernels the MeshConv operator. This parameterization scheme utilizes only 4 parameters for each kernel, and achieves significantly better performance

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-02.jpg?height=474&width=1353&top_left_y=267&top_left_x=386)

Figure 1: Illustration for the MeshConv operator using parameterized differential operators to replace conventional learnable convolutional kernels. Similar to classic convolution kernels that establish patterns between neighboring values, differential operators computes "differences", and a linear combination of differential operators establishes similar patterns.

than competing methods, with much fewer parameters. In particular, we illustrate its use in various machine learning problems in computer vision and climate science.

In summary, our contributions are as follows:

- We present a general approach for orientable CNNs on unstructured grids using parameterized differential operators.
- We show that our spherical model achieves significantly higher parameter efficiency compared to state-of-the-art network architectures for 3D classification tasks and spherical image semantic segmentation.
- We release and open-source the codes developed and used in this study for other potential extended applications ${ }^{1}$

We organize the structure of the paper as follows. We first provide an overview of related studies in the literature in Sec. 2; we then introduce details of our methodology in Sec. 3, followed by an empirical assessment of the effectiveness of our model in Sec. 4. Finally, we evaluate the design choices of our kernel parameterization scheme in Sec. 5 .

## 2 BACKGROUND

Spherical CNNs The first and foremost concern for processing spherical signals is distortions introduced by projecting signals on curved surfaces to flat surfaces. Su \& Grauman (2017) process equirectangular images with regular convolutions with increased kernel sizes near polar regions where greater distortions are introduced by the planar mapping. Coors et al. (2018) and Zhao et al. (2018) use a constant kernel that samples points on the tangent plane of the spherical image to reduce distortions. A slightly different line of literature explores rotational-equivariant implementations of spherical CNNs. Cohen et al. (2018) proposed spherical convolutions with intermediate feature maps in $\mathcal{S O}(3)$ that are rotational-equivariant. Esteves et al. (2018) used spherical harmonic basis to achieve similar results.

Reparameterized Convolutional Kernel Related to our approach in using parameterized differential operators, several works utilize the diffusion kernel for efficient Machine Learning and CNNs. Kondor \& Lafferty (2002) was among the first to suggest the use of diffusion kernel on graphs. Atwood \& Towsley (2016) propose Diffusion-Convolutional Neural Networks (DCNN) for efficient convolution on graph structured data. Boscaini et al. (2016) introduce a generalization of classic CNNs to non-Euclidean domains by using a set of oriented anisotropic diffusion kernels. Cohen \& Welling (2016) utilized a linear combination of filter banks to acquire equivariant convolution filters.[^0]

Ruthotto \& Haber (2018) explore the reparameterization of convolutional kernels using parabolic and hyperbolic differential basis with regular grid images.

Non-Euclidean Convolutions Related to our work on performing convolutions on manifolds represented by an unstructured grid (i.e., mesh), works in geometric deep learning address similar problems (Bronstein et al. 2017). Other methods perform graph convolution by parameterizing the convolution kernels in the spectral domain, thus converting the convolution step into a spectral dot product (Bruna et al., 2014, Defferrard et al., 2016, Kipf \& Welling, 2017, Yi et al., 2017). Masci et al. (2015) perform convolutions directly on manifolds using cross-correlation based on geodesic distances and Maron et al. (2017) use an optimal surface parameterization method (seamless toric covers) to parameterize genus-zero shapes into 2D signals for analysis using conventional planar CNNs.

Image Semantic Segmentation Image semantic segmentation is a classic problem in computer vision, and there has been an impressive body of literature studying semantic segmentation of planar images (Ronneberger et al., 2015; Badrinarayanan et al., 2015; Long et al., 2015, Jégou et al., 2017; Wang et al., 2018a). Song et al. (2017) study semantic segmentation of equirectangular omnidirectional images, but in the context of image inpainting, where only a partial view is given as input. Armeni et al. (2017) and Chang et al. (2017) provide benchmarks for semantic segmentation of 360 panorama images. In the 3D learning literature, researchers have looked at 3D semantic segmentation on point clouds or voxels (Dai et al., 2017a; Qi et al., 2017a; Wang et al., 2018b; Tchapmi et al. 2017, Dai et al., 2017b). Our method also targets the application domain of image segmentation by providing a more efficient convolutional operator for spherical domains, for instance, focusing on panoramic images (Chang et al., 2017).

## 3 METHOD

### 3.1 PARAMETERIZED DIFFERENTIAL OPERATORS

We present a novel scheme for efficiently performing convolutions on manifolds approximated by a given underlying mesh, using what we call Parameterized Differential Operators. To this end, we reparameterize the learnable convolution kernel as a linear combination of differential operators. Such reparameterization provides two distinct advantages: first, we can drastically reduce the number of parameters per given convolution kernel, allowing for an efficient and lean learning space; second, as opposed to the cross-correlation type convolution on mesh surfaces (Masci et al., 2015), which requires large amounts of geodesic computations and interpolations, first and second order differential operators can be efficiently estimated using only the one-ring neighborhood.

In order to illustrate the concept of PDOs, we draw comparisons to the conventional $3 \times 3$ convolution kernel in the regular grid domain. The $3 \times 3$ kernel parameterized by parameters $\boldsymbol{\theta}: \mathcal{G}_{\boldsymbol{\theta}}^{3 \times 3}$ can be written as a linear combination of basis kernels which can be viewed as delta functions at constant offsets:

$$
\begin{equation*}
\mathcal{G}_{\boldsymbol{\theta}}^{3 \times 3}(x, y)=\sum_{i=-1}^{1} \sum_{j=-1}^{1} \theta_{i j} \delta(x-i, y-j) \tag{1}
\end{equation*}
$$

where $x$ and $y$ refer to the spatial coordinates that correspond to the two spatial dimensions over which the convolution is performed. Due to the linearity of the cross-correlation operator $(*)$, the output feature map can be expressed as a linear combination of the input function cross-correlated with different basis functions. Defining the linear operator $\Delta_{i j}$ to be the cross-correlation with a basis delta function, we have:

$$
\begin{align*}
\Delta_{i j} \mathcal{F}(x, y) & :=\mathcal{F}(x, y) * \delta(x-i, y-j)  \tag{2}\\
\mathcal{F}(x, y) * \mathcal{G}_{\boldsymbol{\theta}}^{3 \times 3}(x, y) & =\sum_{i=-1}^{1} \sum_{j=-1}^{1} \theta_{i j} \Delta_{i j} \mathcal{F}(x, y) \tag{3}
\end{align*}
$$

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-04.jpg?height=637&width=1395&top_left_y=270&top_left_x=365)

Figure 2: Schematics for model architecture for classification and semantic segmentation tasks, at a level-5 input resolution. $\mathrm{L} n$ stands for spherical mesh of level- $n$ as defined in Sec. 3.2 MeshConv is implemented according to Eqn. 4. MeshConv ${ }^{\mathrm{T}}$ first pads unknown values at the next level with 0 , followed by a regular MeshConv. DownSamp samples the values at the nodes in the next mesh level. A ResBlock with bottleneck layers, consisting of Conv1x1 (1-by-1 convolutions) and MeshConv layers is detailed above. In the decoder, ResBlock is after each MeshConv ${ }^{\mathrm{T}}$ and Concat.

In our formulation of PDOs, we replace the cross-correlation linear operators $\Delta_{i j}$ with differential operators of varying orders. Similar to the linear operators resulting from cross-correlation with basis functions, differential operators are linear, and approximate local features. In contrast to crosscorrelations on manifolds, differential operators on meshes can be efficiently computed using Finite Element basis, or derived by Discrete Exterior Calculus. In the actual implementation below, we choose the identity ( $I$, 0th order differential, same as $\Delta_{00}$ ), derivatives in two orthogonal spatial dimensions $\left(\nabla_{x}, \nabla_{y}, 1\right.$ st order differential $)$, and the Laplacian operator $\left(\nabla^{2}, 2\right.$ nd order differential $)$ :

$$
\begin{equation*}
\mathcal{F}(x, y) * \mathcal{G}_{\boldsymbol{\theta}}^{d i f f}=\theta_{0} I \mathcal{F}+\theta_{1} \nabla_{x} \mathcal{F}+\theta_{2} \nabla_{y} \mathcal{F}+\theta_{3} \nabla^{2} \mathcal{F} \tag{4}
\end{equation*}
$$

The identity $(I)$ of the input function is trivial to obtain. The first derivative $\left(\nabla_{x}, \nabla_{y}\right)$ can be obtained by first computing the per-face gradients, and then using area-weighted average to obtain per-vertex gradient. The dot product between the per-vertex gradient value and the corresponding $x$ and $y$ vector fields are then computed to acquire $\nabla_{x} \mathcal{F}$ and $\nabla_{y} \mathcal{F}$. For the sphere, we choose the eastwest and north-south directions to be the $x$ and $y$ components, since the poles naturally orient the spherical signal. The Laplacian operator on the mesh can be discretized using the cotangent formula:

$$
\begin{equation*}
\nabla^{2} \mathcal{F} \approx \frac{1}{2 \mathcal{A}_{i}} \sum_{j \in \mathcal{N}(i)}\left(\cot \alpha_{i j}+\cot \beta_{i j}\right)\left(\mathcal{F}_{i}-\mathcal{F}_{j}\right) \tag{5}
\end{equation*}
$$

where $\mathcal{N}(i)$ is the nodes in the neighboring one-ring of $i, \mathcal{A}_{i}$ is the area of the dual face corresponding to node $i$, and $\alpha_{i j}$ and $\beta_{i j}$ are the two angles opposing edge $i j$. With this parameterization of the convolution kernel, the parameters can be similarly optimized via backpropagation using standard stochastic optimization routines.

### 3.2 ICOSAHEDRAL SPHERICAL MESH

The icosahedral spherical mesh (Baumgardner \& Frederickson, 1985) is among the most uniform and accurate discretizations of the sphere. A spherical mesh can be obtained by progressively subdividing each face of the unit icosahedron into four equal triangles and reprojecting each node to unit distance from the origin. Apart from the uniformity and accuracy of the icosahedral sphere, the subdivision scheme for the triangles provides a natural coarsening and refinement scheme for the

| Model | Accuracy(\%) | Number of Parameters |
| :--- | :---: | ---: |
| S2CNN (Cohen et al. 2018) | 96.00 | $58 \mathrm{k}$ |
| SphereNet (Coors et al. 2018) | 94.41 | $196 \mathrm{k}$ |
| Ours | $\mathbf{9 9 . 2 3}$ | $62 \mathrm{k}$ |

Table 1: Results on the Spherical MNIST dataset for validating the use of Parameterized Differential Operators. Our model achieves state-of-the-art performance with comparable number of training parameters.

grid that allows for easy implementations of pooling and unpooling routines associated with CNN architectures. See Fig. 1 for a schematic of the level-3 icosahedral spherical mesh.

For the ease of discussion, we adopt the following naming convention for mesh resolution: starting with the unit icosahedron as the level-0 mesh, each progressive mesh resolution is one level above the previous. Hence, for a level- $l$ mesh:

$$
\begin{equation*}
n_{f}=20 \cdot 4^{l} ; n_{e}=30 \cdot 4^{l} ; n_{v}=n_{e}-n_{f}+2 \tag{6}
\end{equation*}
$$

where $n_{f}, n_{e}, n_{v}$ stands for the number of faces, edges, and vertices of the spherical mesh.

### 3.3 MODEL ARCHITECTURE DESIGN

A detailed schematic for the neural architectures in this study is presented in Fig. 2. The schematic includes architectures for both the classification and regression network, which share a common encoder architecture. The segmentation network consists of an additional decoder which features transpose convolutions and skip layers, inspired by the U-Net architecture (Ronneberger et al., 2015). Minor adjustments are made for different tasks, mainly surrounding adjusting the number of input and output layers to process signals at varied resolutions. A detailed breakdown for model architectures, as well as training details for each task in the Experiment section (Sec. 4), is provided in the appendix (Appendix Sec. B).

## 4 EXPERIMENTS

### 4.1 SPHERICAL MNIST

To validate the use of parameterized differential operators to replace conventional convolution operators, we implemented such neural networks towards solving the classic computer vision benchmark task: the MNIST digit recognition problem (LeCun, 1998).

Experiment Setup We follow Cohen et al. (2018) by projecting the pixelated digits onto the surface of the unit sphere. We further move the digits to the equator to prevent coordinate singularity at the poles. We benchmark our model against two other implementations of spherical CNNs: a rotational-invariant model by Cohen et al. (2018) and an orientable model by Coors et al. (2018). All models are trained and tested with non-rotated digits to illustrate the performance gain from orientation information.

Results and Discussion Our model outperforms its counterparts by a significant margin, achieving the best performance among comparable algorithms, with comparable number of parameters. We attribute the success in our model to the gain in orientation information, which is indispensable for many vision tasks. In contrast, S2CNN (Cohen et al. 2018) is rotational-invariant, and thus has difficulties distinguishing digits " 6 " and " 9 ".

### 4.2 3D OBJECT ClaSSIFICATION

We use the ModelNet40 benchmark (Wu et al., 2015), a 40-class 3D classification problem, to illustrate the applicability of our spherical method to a wider set of problems in 3D learning. For this study, we look into two aspects of our model: peak performance and parameter efficiency.

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-06.jpg?height=570&width=669&top_left_y=295&top_left_x=381)

Figure 3: Parameter efficiency study on ModelNet40, benchmarked against representative 3D learning models consuming different input data representations: PointNet++ using point clouds as input, VoxNet consuming binary-voxel inputs, S2CNN consuming the same input structure as our model (spherical signal). The abscissa is drawn based on log scale.

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-06.jpg?height=566&width=680&top_left_y=297&top_left_x=1075)

Figure 4: Parameter efficiency study on 2D3DS semantic segmentation task. Our spherical segmentation model outperforms the planar and point-based counterparts by a significant margin across all parameter regimes.

Experiment Setup To use our spherical CNN model for the object classification task, we preprocess the 3D geometries into spherical signals. We follow Cohen et al. (2018) for preprocessing the 3D CAD models. First, we normalize and translate each mesh to the coordinate origin. We then encapsulate each mesh with a bounding level- 5 unit sphere and perform ray-tracing from each point to the origin. We record the distance from the spherical surface to the mesh, as well as the sin, cos of the incident angle. The data is further augmented with the 3 channels corresponding to the convex hull of the input mesh, forming a total of 6 input channels. An illustration of the data preprocessing process is presented in Fig. 55. For peak performance, we compare the best performance achievable by our model with other 3D learning algorithms. For the parameter efficiency study, we progressively reduce the number of feature layers in all models without changing the overall model architecture. Then, we evaluate the models after convergence in 250 epochs. We benchmark our results against PointNet++ (Qi et al., 2017a), VoxNet (Qi et al. 2016), and S2CNN2

Results and Discussion Fig. 3 shows a comparison of model performance versus number of parameters. Our model achieves the best performance across all parameter ranges. In the lowparameter range, our model is able to achieve approximately $60 \%$ accuracy for the 40 -class 3D classification task with a mere 2000+ parameters. Table 2 shows a comparison of peak performance between models. At peak performance, our model is on-par with comparable state-of-the-art models, and achieves the best performance among models consuming spherical input signals.

### 4.3 OMNIDIRECTIONAL IMAGE SEGMENTATION

We illustrate the semantic segmentation capability of our network on the omnidirectional image segmentation task. We use the Stanford 2D3DS dataset (Armeni et al. 2017) for this task. The 2D3DS dataset consists of 1,413 equirectangular images with RGB+depth channels, as well as semantic labels across 13 different classes. The panoramic images are taken in 6 different areas, and the dataset is officially split for a 3-fold cross validation. While we are unable to find reported results on the semantic segmentation of these omnidirectional images, we benchmark our spherical segmentation algorithm against classic 2D image semantic segmentation networks as well as a 3D point-based model, trained and evaluated on the same data.[^1]

| Model | Input | Accu. <br> $(\%)$ |
| :---: | :---: | :---: |
| 3DShapeNets (Wu et al. 2015) | voxels | 84.7 |
| VoxNet (Maturana \& Scherer 2015) | voxels | 85.9 |
| PointNet (Qi et al. 2017a) | points | 89.2 |
| PointNet++ Q1 et al. $2017 \mathrm{~b})$ | points | 91.9 |
| DGCNN (Wang et al. 2018b) | points | 92.2 |
| S2CNN (Cohen et al., 2018) | spherical | 85.0 |
| SphericalCNN (Esteves et al., 2018) | spherical | 88.9 |
| Ours | spherical | 90.5 |

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-07.jpg?height=236&width=230&top_left_y=321&top_left_x=1224)

(a) Original CAD model and spherical mesh.

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-07.jpg?height=233&width=227&top_left_y=323&top_left_x=1491)

(b) Resulting surface distance signal.
Table 2: Results on ModelNet40 dataset. Our method compares favorably with state-of-the-art, and achieves best performance among networks utilizing spherical input signals.
Figure 5: Illustration of spherical signal rendering process for a given 3D CAD model.

Experiment Setup First, we preprocess the data into a spherical signal by sampling the original rectangular images at the latitude-longitudes of the spherical mesh vertex positions. Input RGB$\mathrm{D}$ channels are interpolated using bilinear interpolation, while semantic labels are acquired using nearest-neighbor interpolation. We input and output spherical signals at the level-5 resolution. We use the official 3-fold cross validation to train and evaluate our results. We benchmark our semantic segmentation results against two classic semantic segmentation networks: the U-Net (Ronneberger et al. 2015) and FCN8s (Long et al., 2015). We also compared our results with a modified version of spherical S2CNN, and 3D point-based method, PointNet++ (Qi et al., 2017b) using ( $x, y, z, \mathrm{r}, \mathrm{g}, \mathrm{b})$ inputs reconstructed from panoramic RGBD images. We provide additional details toward the implementation of these models in Appendix E. We evaluate the network performance under two standard metrics: mean Intersection-over-Union (mIoU), and pixel-accuracy. Similar to Sec. 4.2, we evaluate the models under two settings: peak performance and a parameter efficiency study by varying model parameters. We progressively decimate the number of feature layers uniformly for all models to study the effect of model complexity on performance.

Results and Discussion Fig. 4 compares our model against state-of-the-art baselines. Our spherical segmentation outperforms the planar baselines for all parameter ranges, and more significantly so compared to the 3D PointNet++. We attribute PointNet++'s performance to the small amount of training data. Fig. 6 shows a visualization of our semantic segmentation performance compared to the ground truth and the planar baselines.

### 4.4 Climate Pattern SeGmentation

To further illustrate the capabilities of our model, we evaluate our model on the climate pattern segmentation task. We follow Mudigonda et al. (2017) for preprocessing the data and acquiring the ground-truth labels for this task. This task involves the segmentation of Atmospheric Rivers (AR) and Tropical Cyclones (TC) in global climate model simulations. Following Mudigonda et al. (2017), we analyze outputs from a 20 -year run of the Community Atmospheric Model v5 (CAM5) (Neale et al. 2010). We benchmark our performance against Mudigonda et al. (2017) for the climate segmentation task to highlight our model performance. We preprocess the data to level-5 resolution.

| Model | Background (\%) | TC (\%) | AR (\%) | Mean (\%) |
| :--- | :---: | :---: | :---: | :---: |
| Mudigonda et al. (2017) | 97 | 74 | 65 | 78.67 |
| Ours | 97 | $\mathbf{9 4}$ | $\mathbf{9 3}$ | $\mathbf{9 4 . 6 7}$ |

Table 3: We achieves better accuracy compared to our baseline for climate pattern segmentation.
![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-08.jpg?height=350&width=1216&top_left_y=278&top_left_x=367)

Legend: board - bookcase
![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-08.jpg?height=164&width=1172&top_left_y=462&top_left_x=380)
ceiling
![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-08.jpg?height=180&width=896&top_left_y=610&top_left_x=366)

- clutter

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-08.jpg?height=166&width=309&top_left_y=611&top_left_x=1252)
column door - floor
![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-08.jpg?height=316&width=1052&top_left_y=756&top_left_x=662)

Figure 6: Visualization of semantic segmentation results on test set. Our results are generated on a level-5 spherical mesh and mapped to the equirectangular grid for visualization. Model underperforms in complex environments, and fails to predict ceiling lights due to incomplete RGB inputs.

Results and Discussion Segmentation accuracy is presented in Table 3. Our model achieves better segmentation accuracy as compared to the baseline models. The baseline model (Mudigonda et al. 2017) trains and tests on random crops of the global data, whereas our model inputs the entire global data and predicts at the same output resolution as the input. Processing full global data allows the network to acquire better holistic understanding of the information, resulting in better overall performance.

## 5 ABLATION STUDY

We further perform an ablation study for justifying the choice of differential operators for our convolution kernel (as in Eqn. 4). We use the ModelNet40 classification problem as a toy example and use a $250 \mathrm{k}$ parameter model for evaluation. We choose various combinations of differential operators, and record the final classification accuracy. Results for the ablation study is presented in Table 4. Our choice of differential operator combinations in Eqn. 4 achieves the best performance

| Convolution kernel | Accuracy |
| :--- | ---: |
| $I+\frac{\partial}{\partial y}+\nabla^{2}$ | 0.8748 |
| $I+\frac{\partial}{\partial x}+\nabla^{2}$ | 0.8809 |
| $I+\nabla^{2}$ | 0.8801 |
| $I+\frac{\partial}{\partial x}+\frac{\partial}{\partial y}$ | 0.8894 |
| $I+\frac{\partial}{\partial x}+\frac{\partial}{\partial y}+\nabla^{2}$ | $\mathbf{0 . 8 9 7 9}$ |

Table 4: Results for the ablation study. The choice of kernel that includes all differential operator components achieve the best accuracy, validating our choice of kernel in Eqn.

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-08.jpg?height=263&width=282&top_left_y=1991&top_left_x=1062)

(a) Ground Truth

![](https://cdn.mathpix.com/cropped/2024_06_04_dd80ce1c1865fd1086cag-08.jpg?height=266&width=268&top_left_y=1992&top_left_x=1427)

(b) Predictions

Figure 7: Visualization of segmentation for Atmospheric River (AR). Plotted in the background is Integrated Vapor Transport (IVT), whereas red masks indicates the existance of AR.
among other choices, and the network performance improves with increased differential operators, thus allowing for more degrees of freedom for the kernel.

## 6 CONCLUSION

We have presented a novel method for performing convolution on unstructured grids using parameterized differential operators as convolution kernels. Our results demonstrate its applicability to machine learning problems with spherical signals and show significant improvements in terms of overall performance and parameter efficiency. We believe that these advances are particularly valuable with the increasing relevance of omnidirectional signals, for instance, as captured by real-world 3D or LIDAR panorama sensors.

## ACKNOWLEDGEMENTS

We would like to thank Taco Cohen for helping with the S2CNN comparison, Mayur Mudigonda, Ankur Mahesh, and Travis O'Brien for helping with the climate data, and Luna Huang for ETT $_{E}$ Xmagic. Chiyu "Max" Jiang is supported by the National Energy Research Scientific Computer (NERSC) Center summer internship program at Lawrence Berkeley National Laboratory. Prabhat and Karthik Kashinath are partly supported by the Intel Big Data Center. The authors used resources of NERSC, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231. In addition, this work is supported by a TUM-IAS Rudolf Mößbauer Fellowship and the ERC Starting Grant Scan2CAD (804724).

## REFERENCES

Iro Armeni, Sasha Sax, Amir R Zamir, and Silvio Savarese. Joint 2d-3d-semantic data for indoor scene understanding. arXiv preprint arXiv:1702.01105, 2017.

James Atwood and Don Towsley. Diffusion-convolutional neural networks. In Advances in Neural Information Processing Systems, pp. 1993-2001, 2016.

Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoderdecoder architecture for image segmentation. arXiv preprint arXiv:1511.00561, 2015.

John R Baumgardner and Paul O Frederickson. Icosahedral discretization of the two-sphere. SIAM Journal on Numerical Analysis, 22(6):1107-1115, 1985.

Davide Boscaini, Jonathan Masci, Emanuele Rodolà, and Michael Bronstein. Learning shape correspondence with anisotropic convolutional neural networks. In Advances in Neural Information Processing Systems, pp. 3189-3197, 2016.

Mario Botsch, Leif Kobbelt, Mark Pauly, Pierre Alliez, and Bruno Lévy. Polygon mesh processing. AK Peters/CRC Press, 2010.

Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine, 34(4):18-42, 2017.

Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann Lecun. Spectral networks and locally connected networks on graphs. In International Conference on Learning Representations (ICLR2014), CBLS, April 2014, 2014.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017.

Taco S Cohen and Max Welling. Steerable cnns. arXiv preprint arXiv:1612.08498, 2016.

Taco S. Cohen, Mario Geiger, Jonas Khler, and Max Welling. Spherical CNNs. In International Conference on Learning Representations, 2018. URLhttps://openreview. net/forum? $i d=H k . b d 5 \times 2 R b$.

Benjamin Coors, Alexandru Paul Condurache, and Andreas Geiger. Spherenet: Learning spherical representations for detection and classification in omnidirectional images. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 518-533, 2018.

Keenan Crane. Discrete differential geometry: An applied introduction, 2015.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas A Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, volume 2, pp. $10,2017 a$.

Angela Dai, Daniel Ritchie, Martin Bokeloh, Scott Reed, Jürgen Sturm, and Matthias Nießner. Scancomplete: Large-scale scene completion and semantic segmentation for 3d scans. In Proc. Conference on Computer Vision and Pattern Recognition (CVPR), 2017b.

Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in Neural Information Processing Systems, pp. 3844-3852, 2016.

Carlos Esteves, Kostas Daniilidis, Ameesh Makadia, and Christine Allec-Blanchette. Learning so (3) equivariant representations with spherical cnns. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 52-68, 2018.

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR), 2013.

Simon Jégou, Michal Drozdzal, David Vazquez, Adriana Romero, and Yoshua Bengio. The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation. In Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on, pp. 1175-1183. IEEE, 2017.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.

Risi Imre Kondor and John Lafferty. Diffusion kernels on graphs and other discrete structures. In Proceedings of the 19th international conference on machine learning, volume 2002, pp. 315$322,2002$.

Yann LeCun. The mnist database of handwritten digits. http://yann. lecun. com/exdb/mnist/, 1998.

Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3431-3440, 2015.

Haggai Maron, Meirav Galun, Noam Aigerman, Miri Trope, Nadav Dym, Ersin Yumer, Vladimir G Kim, and Yaron Lipman. Convolutional neural networks on surfaces via seamless toric covers. ACM Trans. Graph, 36(4):71, 2017.

Jonathan Masci, Davide Boscaini, Michael Bronstein, and Pierre Vandergheynst. Geodesic convolutional neural networks on riemannian manifolds. In Proceedings of the IEEE international conference on computer vision workshops, pp. 37-45, 2015.

Daniel Maturana and Sebastian Scherer. Voxnet: A 3d convolutional neural network for real-time object recognition. In Intelligent Robots and Systems (IROS), 2015 IEEE/RSJ International Conference on, pp. 922-928. IEEE, 2015.

Mayur Mudigonda, Sookyung Kim, Ankur Mahesh, Samira Kahou, Karthik Kashinath, Dean Williams, Vincen Michalski, Travis O'Brien, and Mr Prabhat. Segmenting and tracking extreme climate events using neural networks. In First Workshp Deep Learning for Physical Sciences. Neural Information Processing Systems (NIPS), 2017.

Richard B Neale, Chih-Chieh Chen, Andrew Gettelman, Peter H Lauritzen, Sungsu Park, David L Williamson, Andrew J Conley, Rolando Garcia, Doug Kinnison, Jean-Francois Lamarque, et al. Description of the ncar community atmosphere model (cam 5.0). NCAR Tech. Note NCAR/TN$486+\operatorname{STR}, 1(1): 1-12,2010$.

Charles R Qi, Hao Su, Matthias Nießner, Angela Dai, Mengyuan Yan, and Leonidas J Guibas. Volumetric and multi-view cnns for object classification on 3d data. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5648-5656, 2016.

Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. Proc. Computer Vision and Pattern Recognition (CVPR), $\operatorname{IEEE}, 1(2): 4,2017 \mathrm{a}$.

Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in Neural Information Processing Systems, pp. 5099-5108, 2017b.

Evan Racah, Christopher Beckham, Tegan Maharaj, Samira Ebrahimi Kahou, Mr Prabhat, and Chris Pal. Extremeweather: A large-scale climate dataset for semi-supervised detection, localization, and understanding of extreme weather events. In Advances in Neural Information Processing Systems, pp. 3402-3413, 2017.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computerassisted intervention, pp. 234-241. Springer, 2015.

Lars Ruthotto and Eldad Haber. Deep neural networks motivated by partial differential equations. arXiv preprint arXiv:1804.04272, 2018.

Shuran Song, Andy Zeng, Angel X Chang, Manolis Savva, Silvio Savarese, and Thomas Funkhouser. Im2pano3d: Extrapolating 360 structure and semantics beyond the field of view. arXiv preprint arXiv:1712.04569, 2017.

Yu-Chuan Su and Kristen Grauman. Learning spherical convolution for fast features from 360 imagery. In Advances in Neural Information Processing Systems, pp. 529-539, 2017.

Lyne Tchapmi, Christopher Choy, Iro Armeni, JunYoung Gwak, and Silvio Savarese. Segcloud: Semantic segmentation of 3d point clouds. In 3D Vision (3DV), 2017 International Conference on, pp. 537-547. IEEE, 2017.

Panqu Wang, Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, Xiaodi Hou, and Garrison Cottrell. Understanding convolution for semantic segmentation. In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 1451-1460. IEEE, 2018a.

Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. arXiv preprint arXiv:1801.07829, 2018b.

Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1912-1920, 2015.

Li Yi, Hao Su, Xingwen Guo, and Leonidas J Guibas. Syncspeccnn: Synchronized spectral cnn for 3d shape segmentation. In CVPR, pp. 6584-6592, 2017.

Qiang Zhao, Chen Zhu, Feng Dai, Yike Ma, Guoqing Jin, and Yongdong Zhang. Distortion-aware cnns for spherical images. In IJCAI, pp. 1198-1204, 2018.
</end of paper 1>


<paper 2>
# Pseudocylindrical Convolutions for Learned Omnidirectional Image Compression 

Mu Li, Kede Ma, Member, IEEE, Jinxing Li, and David Zhang, Life Fellow, IEEE


#### Abstract

Although equirectangular projection (ERP) is a convenient form to store omnidirectional images (also known as $360^{\circ}$ images), it is neither equal-area nor conformal, thus not friendly to subsequent visual communication. In the context of image compression, ERP will over-sample and deform things and stuff near the poles, making it difficult for perceptually optimal bit allocation. In conventional $360^{\circ}$ image compression, techniques such as region-wise packing and tiled representation are introduced to alleviate the over-sampling problem, achieving limited success. In this paper, we make one of the first attempts to learn deep neural networks for omnidirectional image compression. We first describe parametric pseudocylindrical representation as a generalization of common pseudocylindrical map projections. A computationally tractable greedy method is presented to determine the (sub)-optimal configuration of the pseudocylindrical representation in terms of a novel proxy objective for rate-distortion performance. We then propose pseudocylindrical convolutions for $360^{\circ}$ image compression. Under reasonable constraints on the parametric representation, the pseudocylindrical convolution can be efficiently implemented by standard convolution with the so-called pseudocylindrical padding. To demonstrate the feasibility of our idea, we implement an end-to-end $360^{\circ}$ image compression system, consisting of the learned pseudocylindrical representation, an analysis transform, a non-uniform quantizer, a synthesis transform, and an entropy model. Experimental results on 19, 790 omnidirectional images show that our method achieves consistently better rate-distortion performance than the competing methods. Moreover, the visual quality by our method is significantly improved for all images at all bitrates.


Index Terms-Omnidirectional image compression, pseudocylindrical representation, pseudocylindrical convolution, map projection

## 1 INTRODUCTION

OMNIDIRECTIONAL images, also referred to as spherical and $360^{\circ}$ images, provide $360^{\circ} \times 180^{\circ}$ panoramas of natural scenes, and enable free view direction exploration. Recent years have witnessed a dramatic increase in the volume of $360^{\circ}$ image data being generated. On the one hand, average users have easy access to $360^{\circ}$ imaging and display devices, and are getting used to play with this format of virtual reality content on a daily basis. On the other hand, there is a trend to capture ultra-high-definition panoramas to provide an excellent immersive experience, pushing the spatial resolution to be exceedingly high (e.g., $8 \mathrm{~K}$ ). The increasing need for storing and transmitting the enormous amount of panoramic data calls for novel effective $360^{\circ}$ image compression methods.

Currently, the prevailing scheme for $360^{\circ}$ image compression takes a two-step approach. First, select (or create) a map projection [1] with the optimized hyperparameter setting for the sphere-to-plane mapping. Second, pick (or

- This project is supported by China Postdoctoral Science Foundation (2020TQ0319, 2020M682034), NSFC Foundation (61906162, 62102339), and Shenzhen Science and Technology Program (RCBS20200714114910193).
- Mu Li is with the School of Data Science, The Chinese University of Hong Kong (Shenzhen), Shenzhen, 518172, China, and also with the School of Information Science and Technology, University of Science and Technology of China, Hefei, 230026, China (e-mail: limuhit@gmail.com).
- Kede Ma is with the Department of Computer Science, City University of Hong Kong, Kowloon, Hong Kong (e-mail: kede.ma@cityu.edu.hk).
- Jinxing Li is with the School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, 518055, China (e-mail: lijinxing158@gmail.com).
- David Zhang is with the School of Data Science, The Chinese University of Hong Kong (Shenzhen), and also with the Shenzhen Institute of Artificial Intelligence and Robotics for Society, Shenzhen, 518172, China (e-mail: davidzhang@cuhk.edu.cn). adapt) a standard image codec that is compatible with central-perspective images for compression. In differential geometry, the Theorema Egregium by Gauss states that all planar projections of a sphere will necessarily be distorted [1]. Among the three major projection desiderata: equalarea, conformal, and equidistant the most widely used equirectangular projection (ERP) does not satisfy the former two, and thus is not friendly to subsequent visual communication applications. Regional resampling [2], [3], [4], [5] and adaptive quantization [6], [7], [8], |9] techniques have been proposed to mitigate the sampling problem of ERP. Compression-friendly projections, such as the tiled representation [5], [10] and hybrid cubemap projection [11], [12], [13] have also been investigated. In $360^{\circ}$ content streaming, viewport-based format is often preferred for coding and transmission \14], [15]. Other projection methods for image display [16], |17], [18] and visual recognition (e.g., icosahedron [19| and tangent images [20]) also emerge in the field of computer vision. With many possible projections at hand, it remains unclear which one is the best choice for learned $360^{\circ}$ image compression in terms of rate-distortion performance, computation and implementation complexity, and compatibility with standard deep learning-based analysis/synthesis transform, and entropy model.

Deep neural networks (DNNs) have been proved effective in many low-level vision tasks, including centralperspective image compression [21], [22], [23], [24], [25], [26], [27]. Following a transform coding scheme, the raw RGB image is first transformed to a latent code representa-

1. Equal-area, conformal, and equidistant map projections preserve relative scales of things and stuff, local angles, and great-circle distances between points, respectively, on the sphere.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-02.jpg?height=691&width=1770&top_left_y=145&top_left_x=164)

Fig. 1. Illustration of the non-uniform sampling problem of ERP. We first project the same image patch to different latitudes of the ERP images (padded with zeros), and compress them by the HEVC intra coding with the identical hyperparameter. The performance is given in the format of bytes / peak signal-to-noise ratio (PSNR in dB).

tion, quantized to the discrete code, and final transformed back to the RGB domain, all with DNNs that can be end-toend optimized with respect to rate-distortion performance. Recently, a growing research trend is to enable DNNs for $360^{\circ}$ computer vision, which is broadly sorted into three categories depending on how they address the sphere-toplane distortion: spatially adaptive convolution [28], [29], knowledge distillation [30], [31], and reparameterization [32], [33], [34], [35]. Nevertheless, it is highly nontrivial to directly adapt these techniques to learned $360^{\circ}$ image compression. This is because these methods typically require to modify convolution filters, and cannot benefit from years of sophisticated code optimization of standard convolution. As a result, compressing a high-resolution omnidirectional image would be painfully slow. Moreover, non-uniform sampling (especially over-sampling at high latitudes) is a more urgent issue in $360^{\circ}$ image compression than geometric deformation (see Fig. 1), because the latter can be handled by adopting a perceptual image quality metric as the learning objective [36]. From this perspective, reparameterization methods that directly work with spherical signals and are independent of projection methods seem to be more appropriate for rate reduction. But reparameterization comes with its own problem apart from computational complexity: the spherical representation is orderless, which may hinder the context-based entropy modeling for accurate rate estimation [24], [27].

In this paper, we take initial steps towards learned omnidirectional image compression based on DNNs. Our main contributions are three-fold.

- We describe parametric pseudocylindrical representation as a generalization of common pseudocylindrical map projections and the tiled representation by Yu et al. [5]. We propose a computationally tractable greedy algorithm to determine the (sub)-optimal parameter configuration in terms of the rate-distortion performance, estimated by a novel proxy objective. Interestingly, we find that the optimized representa- tion does not correspond to pseudocylindrical projections with the equal-area property (e.g., sinusoidal projection). Empirically, the rate-distortion performance will benefit from slight over-sampling at midlatitudes.
- We propose pseudocylindrical convolutions that work seamlessly with the parametric pseudocylindrical representation for $360^{\circ}$ image compression. Under reasonable constraints on the representation (i.e., the tiled representation), the pseudocylindrical convolution can be efficiently implemented by standard convolution with pseudocylindrical padding. In particular, given the current tile, we pad the latitudinal side with adjacent tiles resized to the same width, and pad the longitudinal side circularly to respect the spherical structure. The manipulation on feature representation instead of convolution leads to a significant advantage of our approach: we are able to transfer the large zoo of DNN-based compression methods for central-perspective images to omnidirectional images.
- We build an end-to-end $360^{\circ}$ image compression system, which is composed of the optimized pseudocylindrical representation, an analysis transform, a non-uniform quantizer, a synthesis transform, and a context-based entropy model. Extensive experiments show that our method outperforms compression standards and DNN-based methods for centralperspective images with region-wise packing (RWP). More importantly, the visual quality of the compressed images is much better for all images at all bitrates.


## 2 RELATED WORK

In this section, we provide a short overview of learned image compression methods for planar images and standards (and tricks) for compressing omnidirectional images.

Relevant techniques for $360^{\circ}$ computer vision will also be briefly summarized.

### 2.1 Learned Planar Image Compression

Learned image compression learns to trade off the rate and distortion, in which DNNs are commonly used to build the analysis transform (i.e., encoder) and the synthesis transform (i.e., decoder), and to model the rate of the codes.

For rate estimation, the discrete entropy serves as a general choice, which requires keeping track of the joint probability of the discrete codes that varies with changes in the network parameters. Side information in the form of hyper-prior [23] and code context [24], [27] can be introduced to boost the accuracy of entropy modeling. Ballé et al. |21| adopted a parametric piece-wise probability distribution function for codes of the same channel; they [23] later assumed a univariate Gaussian distribution for codes of the same spatial location, whose mean and variance are estimated using a hyper-prior. Minnen et al. [24] modeled the code distribution with a mixture of Gaussians (MoG). A $5 \times 5$ code context was adopted as an auto-regressive prior to better predict the MoG parameters. Similarly, a MoG distribution was used in [22]. Li et al. [27] proposed a context-based DNN for efficient entropy modeling.

Besides precise estimation of the code rate using the discrete entropy, various upper-bounds (e.g., the number and dimension of codes) have been derived. Toderici et al. [37|, [38] proposed a progressive compression scheme, in which the rate was controlled by the number of iterations. Johnston et al. $|39|$ took a step further, and presented a content-adaptive bit allocation strategy. Ripple et al. [40] implemented pyramid networks for the encoder and the decoder, with an adaptive code length regularization for rate control. In a similar spirit, Li et al. [41] described a spatially adaptive bit allocation scheme, where the rate was estimated as the total number of codes allocated to different regions. They [42] further designed better relaxation strategies for learning optimal bit allocation.

For distortion quantification, conventional metrics, including mean squared error (MSE), peak signal-to-noise ratio (PSNR), structural similarity (SSIM) [43], and multiscale SSIM (MS-SSIM) [44], were employed to evaluate the "perceptual" distance between the original and compressed images in learned image compression. Other metrics were also incorporated for special considerations. For instance, to boost the visual quality of low-bitrate images, the adversarial loss [45] was introduced in [26], [40]. Torfason et al. [25] suggested to utilize task-dependent losses (e.g., the classification and segmentation error) for task-driven image compression.

## $2.2360^{\circ}$ Image Compression

Most $360^{\circ}$ image compression methods were designed on top of widely used compression standards such as HEVC [3], [46], [47] for central-perspective content. Thus sphereto-plane projections are inevitable for compatibility purposes [48]. One popular branch of work is to introduce practical tricks to tackle the non-uniform sampling problem, such as regional resampling and adaptive quantization. Budagavi et al. 49| adopted Gaussian blurring to smooth high-latitude regions of the ERP image to ease subsequent compression. Regional down-sampling [2], [3], [4], [5] partitions the ERP image into several regions according to the latitude, and resamples and assembles them into a new image of reduced size for compression. Of particular interest, RWP, which repacks the ERP image by reducing the sizes of polar regions, is adopted in HEVC when dealing with $360^{\circ}$ content [3]. Close to our work, Yu et al. [5] introduced the tile representation for $360^{\circ}$ images, and suggested to optimize the height and width of each tile for the sampling rate and bitrate. Adaptive quantization |6], [7], [8], |9], [50], on the other hand, adjusts quantization parameters (QPs) for different regions in ERP with respect to spherical "perceptual" metrics such as S-PSNR [51] and WS-PSNR [52].

Apart from ERP, cubemap projection is also commonly seen in $360^{\circ}$ compression. Su et al. [53] learned to rotate the $360^{\circ}$ images to boost the coding performance. Other variants of cubemap formats, such as hybrid equi-angular cubemap projection (HEC) [11], hybrid cubemap projection (HСР) [12], and hybrid angular cubemap projection (HAP) [13|, were investigated for uniform and content adaptive sampling. In addition, viewport-based [14], [46], [54], [55] and saliency-aware methods [56], 577], [58] were proposed to spend most of the bits on coding the viewports of interest, when streaming $360^{\circ}$ content.

Taking inspirations from Yu et al. |5|, we propose parametric pseudocylindrical representation for learned $360^{\circ}$ image compression. The optimal parameter configuration is determined by a greedy algorithm optimized for a proxy rate-distortion objective. With reasonable constraints, the parametric representation supports an efficient implementation of the proposed pseudocylindrical convolutions.

## $2.3360^{\circ}$ Computer Vision

Recently, there has been a surge of interest to develop DNNs for $360^{\circ}$ computer vision with four main types of techniques. The first is spatially adaptive convolution. Designed around ERP, the most straightforward implementation is to expand the receptive field horizontally via rectangular filters or dilated convolutions [28]. A more advanced version is to design distortion-aware and deformable convolution kernels |29], in combination with spiral spherical sampling [59]. This type of approach is less scalable to deeper networks, which is necessary to achieve satisfactory rate-distortion performance in learned image compression. The second is knowledge distillation, with the goal of training DNNs on ERP images to predict the responses of a target model on viewport images. As one of the first attempts to learn "spherical convolution" for $360^{\circ}$ vision, Su and Grauman |30| tied the kernel weights along each row of the ERP image to accommodate the over-sampling issue. Due to the incorporation of secondary DNNs, this type of approach is often computationally expensive, whose performance may also be limited as the sphere-to-plane distortion is not explicitly modeled. The third is a family of reparameterization methods that are rooted in spherical harmonics and spectral analysis. These often define spherical convolution mathematically to seek rotational equivariance and invariance for dense and global prediction problems. Cohen et al. [32] defined spherical correlation with an efficient implementation
based on the generalized fast Fourier transform $\sqrt{60 \mid}$. Concurrently, Esteves et al. [33] defined spherical convolution [61] as a specific case of group convolution [62], which admits a spectral domain implementation. To avoid the computational cost of the spherical Fourier transform, Perraudin

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-04.jpg?height=43&width=892&top_left_y=347&top_left_x=150)
pixelization (HEALPix) to formulate graph convolutions for cosmological applications. However, it is only practical to apply the method to part of the sphere. Jiang et al. [35] reparameterized convolution filters as linear combinations of first-order and second-order differential operators with learnable weights. Despite being mathematically appealing, rotational equivariance and invariance are less relevant to $360^{\circ}$ image compression, and may cause inconvenience in context-based entropy modeling because of the orderless nature of the spherical representation.

The above-mentioned methods typically require modifying or re-designing the convolution operation, which are generally computationally expensive. Moreover, they may not enable the desired transferability of existing DNNs for central-perspective images, which enjoy years of research into optimal architecture design and efficient convolution implementation. In contrast, the proposed pseudocylindrical convolution resolves the over-sampling problem and can be efficiently implemented by standard convolution with pseudocylindrical padding.

## 3 PROPOSEd METHOD

In this section, we first introduce parametric pseudocylindrical representation, and describe a greedy algorithm to determine the parameter configuration for a proxy ratedistortion objective. We then propose pseudocylindrical convolutions as the main building block for our learned $360^{\circ}$ image compression system.

### 3.1 Parametric Pseudocylindrical Representation

From the practical standpoint, we start with a $360^{\circ}$ image $\boldsymbol{x} \in \mathbb{R}^{H \times W}$ stored in ERP format, where $H$ and $W$ are the maximum numbers of samples in each column and row, respectively. The plane-to-sphere coordinate conversion can be calculated by

$$
\begin{align*}
\theta_{i} & =\left(0.5-\frac{i+0.5}{H}\right) \times \pi, \quad i=\{0, \ldots, H-1\}  \tag{1}\\
\phi_{j} & =\left(\frac{j+0.5}{W}-0.5\right) \times 2 \pi, \quad j=\{0, \ldots, W-1\} \tag{2}
\end{align*}
$$

where $\theta$ and $\phi$ index the latitude and the longitude, respectively. Bilinear interpolation is used as the optional resampling filter if necessary. As a generalization of ERP, the proposed representation is also defined over a $2 \mathrm{D}$ grid $\Omega=\{0, \ldots, H-1\} \times\{0, \ldots, W-1\}$, and is parameterized by $\left\{W_{i}\right\}_{i=0}^{H-1}$, where $W_{i} \in\{1, \ldots, W\}$ is the width of the $i$ th row (with the starting point fixed to zero). By varying $W_{i}$, our representation offers a precise control over the sampling density of each row. For visualization purposes, we may reparameterize the representation by $\left\{W_{i}, S_{i}\right\}_{i=0}^{H-1}$, where $S_{i}$ denotes the starting point:

$$
\begin{equation*}
S_{i}=\left\lfloor\left(W-W_{i}\right) / 2\right\rfloor \tag{3}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-04.jpg?height=253&width=440&top_left_y=150&top_left_x=1081)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-04.jpg?height=192&width=364&top_left_y=457&top_left_x=1138)

(c)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-04.jpg?height=250&width=446&top_left_y=146&top_left_x=1509)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-04.jpg?height=196&width=377&top_left_y=455&top_left_x=1573)

(d)
Fig. 2. Comparison of different omnidirectional image representations. (a) ERP. (b) Sinusoidal projection. (c) Tiled sinusoidal representation. (d) Optimized pseudocylindrical representation.

We refer to this data structure as parametric pseudocylindrical representation, since it generalizes several pseudocylindrical map projections. Specifically:

- Choosing $W_{i}=W$ and Eqs. (1) and (2) as the planeto-sphere mapping yields the standard ERP;
- Choosing $W_{i}=\cos \left(\theta_{i}\right) W$ and replacing Eq. (2) to

$$
\begin{equation*}
\phi_{j}=\left(\frac{j-S_{i}+0.5}{W_{i}}-0.5\right) \times 2 \pi \tag{4}
\end{equation*}
$$

for $j=\left\{S_{i}, \ldots, S_{i}+W_{i}-1\right\}$ as the longitude mapping yields sinusoidal projection (see Fig. 2 (b));

- Choosing $W_{i}=\left(2 \cos \left(2 \theta_{i} / 3\right)-1\right) W$, where

$$
\begin{equation*}
\theta_{i}=3 \arcsin \left(0.5-\frac{i+0.5}{H}\right) \tag{5}
\end{equation*}
$$

is the latitude mapping and Eq. 4) is the longitude mapping yields the Craster parabolic projection. This is used in the objective quality metric - CPP-PSNR 633.

- Another special case of interest arises when setting adjacent rows to the same width, leading to the tiled representation proposed by Yu et al. [5], which plays a crucial role in accelerating pseudocylindrical convolutions, as will be immediately clear.

With different combinations of the width configuration and the plane-to-sphere mapping, our pseudocylindrical representation not only includes a broad class of pseudocylindrical map projections as special cases but also opens the door of novel data structures that may be more suitable for $360^{\circ}$ image compression. Without loss of generality, in the remainder of the paper, we use Eqs. (1) and (4) for the plane-tosphere coordinate conversion, and assume $S_{i}=0$.

### 3.2 Pseudocylindrical Convolutions

Based on the pseudocylindrical representation, we define the pseudocylindrical convolution operation ${ }^{2}$ by first specifying a neighboring grid:

$$
\begin{equation*}
\mathcal{N}=\{(i, j) \mid i, j \in\{-K, \ldots, K\}\} \tag{6}
\end{equation*}
$$

2. In fact, nearly all DNNs implement cross-correlation instead of convolution. Here we assume the convolution filter has already been reflected around the center.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-05.jpg?height=526&width=751&top_left_y=146&top_left_x=275)

Fig. 3. Illustration of the neighborhood in the (a) planar and (b) spherical representation.

where $2 K+1$ is the spread of the convolution kernel. For a central-perspective image $\boldsymbol{x}$, it is straightforward to define the neighbors using the Manhattan distance (see Fig. 3). The response $\boldsymbol{y}$ of the convolution on $\boldsymbol{x}$ at the position $(p, q)$ is computed by

$$
\begin{equation*}
\boldsymbol{y}(p, q)=\sum_{(i, j) \in \mathcal{N}} \boldsymbol{w}(i, j) \boldsymbol{x}\left(p_{i}, q_{j}\right) \tag{7}
\end{equation*}
$$

where $\boldsymbol{w}$ denotes the convolution filter, $p_{i}=p+i$, and $q_{j}=q+j$. To convolve the filter over the pseudocylindrical representation, we also need to define the neighbors, with the goal of approaching the uniform sampling density over the sphere. Relying on a variant of Manhattan distance, we start from $\left(\theta_{p}, \phi_{q}\right)$ corresponding to $(p, q) \in \Omega$, and retrieve the neighbor at $(i, j) \in \mathcal{N}$ by first moving $i \delta_{\theta}$ along the longitude and then $j \delta_{\phi}$ along the latitud ${ }^{3}$, where $\delta_{\theta}=\pi / H$ and $\delta_{\phi}=2 \pi \cos \left(\theta_{p}\right) / W_{p}$. Positive (negative) values of $i$ head towards the north (south) pole. Similarly, positive (negative) values of $j$ mean anticlockwise (clockwise) movement from a bird's-eye view. We obtain the neighbors on the pseudocylindrical representation via the sphere-to-plane projection:

$$
\begin{align*}
p_{i} & =p+i  \tag{8}\\
q_{j} & =\frac{W_{p_{i}}}{W_{p}}(q+0.5)-0.5+\frac{\cos \theta_{p}}{\cos \theta_{p_{i}}} \frac{W_{p_{i}}}{W_{p}} j \\
& =\frac{W_{p_{i}}}{W_{p}}\left(q+\frac{\cos \theta_{p}}{\cos \theta_{p_{i}}} j+0.5\right)-0.5 \tag{9}
\end{align*}
$$

We assume circular boundary condition, and give a careful treatment of the boundary handling near the two poles (i.e., $p_{i}<0$ and $\left.p_{i}>H\right)$ :

$$
\begin{align*}
& p_{i}=\left(-1-p_{i}\right) \bmod H \\
& q_{j}=\left(q_{j}+0.5 W_{p_{i}}\right) \bmod W_{p_{i}} \tag{10}
\end{align*}
$$

Fig. 3 (b) shows an example of the adjustment of $q_{j}$ over the sphere when $p_{i}<0$ (i.e., crossing the north pole). For fractional $q_{j}$, we compute $\boldsymbol{x}\left(p_{i}, q_{j}\right)$ via linear interpolation:

$$
\begin{equation*}
\boldsymbol{x}\left(p_{i}, q_{j}\right)=\sum_{k \in \mathcal{N}} \boldsymbol{b}\left(q_{j}, k\right) \boldsymbol{x}\left(p_{i},\left\lfloor q_{j}\right\rfloor+k\right) \tag{11}
\end{equation*}
$$

3. We assume a unit sphere. where $\boldsymbol{b}$ is the linear kernel. Last, the pseudocylindrical convolution is computed by plugging Eqs. 88, (9), and 11 into Eq. 77.

We take a close look at the computational complexity of the pseudocylindrical convolution, which mainly comes from three parts: neighbor search, linear interpolation, and inner product. Searching one neighbor requires calling the transcendental function, $\cos (\cdot)$, twice with four multiplications and three additions. For linear interpolation, one modulo operation and one addition are needed to create the bilinear kernel, and two multiplications and one addition are used to compute interpolated value. For a kernel size of $(2 K+1) \times(2 K+1)$, we need $28 K^{2}+14 K$ and $20 K^{2}+10 K$ operations for neighbor search and linear interpolation, respectively, but only $8 K^{2}+8 K+1$ operations for inner product.

To reduce the computational complexity of neighbor search, we make a mild approximation to Eq. 9):

$$
\begin{equation*}
q_{j} \approx \frac{W_{p_{i}}}{W_{p}}(q+j+0.5)-0.5 \tag{12}
\end{equation*}
$$

where we assume $\cos \left(\theta_{p}\right) \approx \cos \left(\theta_{p_{i}}\right)$. This is reasonably true because $\theta_{p}$ and $\theta_{p_{i}}$ correspond to adjacent rows, and are very close provided that $H$ is large. From Eq. (12, it is clear $q_{j}=(q+k)_{j-k}$, meaning that adjacent samples in a row are neighbors of one another with no computation. Moreover, searching for neighbors in an adjacent row amounts to scaling it to the width of the current row ${ }^{4}$. Furthermore, we perform circular padding for $q_{j}<0$ and $q_{j} \geq W_{p}$ with $q_{j}=q_{j} \bmod W_{p}$. We refer to this process as $p$ seudocylindrical padding (see Fig. 4 (b) and (c)). For a row with width $W_{p}$, we greatly reduce the computation ${ }^{5}$ from $\left(48 K^{2}+24 K\right) W_{p}$ to $20 K W_{p}+(2 K+1) 2 K$, where $W_{p} \gg K$.

We may further simplify Eq. 12 to 7 by enforcing $W_{p}=W_{p_{i}}$. By doing so, the $p$-th and $(p+i)$-th rows become neighborhood of each other with no computation. The neighboring rows with the same width can be viewed as a tile, and the pseudocylindrical representation reduces gracefully to the tiled representation [5] (see Fig. 22 (d)). Pseudocylindrical padding occurs only at the boundaries of each tile. In summary, the tiled representation $\boldsymbol{z}$ of $\boldsymbol{x}$ is composed of $\left\{\boldsymbol{z}_{t}\right\}_{t=0}^{T-1}$, where $\boldsymbol{z}_{t} \in \mathbb{R}^{H_{t} \times W_{t}}$ is the $t$-th tile. The set of free parameters are $\left\{T,\left\{H_{t}\right\}_{t=0}^{T-1},\left\{W_{t}\right\}_{t=0}^{T-1}\right\}$. With these two steps of simplifications, the pseudocylindrical convolution can be implemented in parallel on $\left\{\boldsymbol{z}_{t}\right\}$ by standard convolution with pseudocylindrical padding. On one hand, this offers the opportunity to build DNN-based compression methods for omnidirectional images upon those for central-perspective images with minimal modifications. On the other hand, this enables fast implement of the proposed pseudocylindrical convolution. For a tile $\boldsymbol{z}_{t}$, the computation operations used for pseudocylindrical padding are $20 K W_{t}+\left(2 K+H_{t}\right) 2 K$, which are much smaller than the operations for convolution, i.e., $\left(8 K^{2}+8 K+1\right) W_{t} H_{t}$, when $H_{t} \gg 1$. In such case, our pseudocylindrical convolution should achieve nearly the same running speed as the standard convolution with zero padding.

4. More precisely, we first shift the adjacent row by half pixel, and scale it to the width of the current row, and shift it back by half pixel.
5. $(2 K+1) 2 K$ arises from the circular boundary handling along the longitude.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-06.jpg?height=566&width=1778&top_left_y=145&top_left_x=171)

Shestars,

(b) (c)

Fig. 4. Illustration of the pseudocylindrical representation and pseudocylindrical padding. (a) Intermediate pseudocylindrical representations in the DNN. (b) Pseudocylindrical padding. (c) Pseudocylindrical padding for the pole tile.

### 3.3 Pseudocylindrical Representation Optimization

In general, different $360^{\circ}$ images may have different parameter configurations of the pseudocylindrical representation for optimal compression performance, which depend on the image content. The corresponding combinatorial optimization problem can be formulated as

$$
\begin{align*}
& \min _{T,\left\{H_{t}\right\},\left\{W_{t}\right\}} \operatorname{RD}\left(\boldsymbol{x}, \operatorname{compress}_{\boldsymbol{\alpha}}(\boldsymbol{x}) ; T,\left\{H_{t}\right\},\left\{W_{t}\right\}\right) \\
& \text { s.t. } T \in\{0, \ldots, H-1\} \\
& H_{t} \in\{0, \ldots, H-1\}, t \in\{0, \ldots, T-1\} \\
& \sum_{t} H_{t}=H-1 \\
& W_{t} \in\{0, \ldots, W-1\}, t \in\{0, \ldots, T-1\} \tag{13}
\end{align*}
$$

where $\boldsymbol{x}$ is the given $360^{\circ}$ image, $\operatorname{RD}(\cdot)$ is a quantitative measure for rate-distortion performance, and compress $\left.\boldsymbol{\alpha}^{(} \cdot\right)$ is a generic compression method with a learnable parameter vector $\boldsymbol{\alpha}$. As noted by Yu et al. [5], Problem (13) is essentially a multiple-dimensional, multiplechoice knapsack problem, which prohibits exhaustive search when $H$ and $W$ are large. We choose to simplify the problem in the follow ways.

1) Treat $T$ and $\left\{H_{t}\right\}_{t=1}^{T-1}$ as hyperparameters, and preset them, where $H_{t}=H_{0}$ and $T=H / H_{0}$. The general guideline is to set $H_{0}$ large enough to enjoy the fast computation of standard convolution, while making Eq. 12, approximately hold.
2) Quantize the width to $L$ levels, where $L \ll W$. Thus enumeration of the possible widths is performed in a much reduced search space.
3) Enforce the symmetry of the pseudocylindrical representation with respect to the equator, which further halves the search space.
4) Discourage oversampling at higher latitudes by adding the constraint $W_{t} \leq W_{t^{\prime}}$ for $t \leq t^{\prime}$ and $t, t^{\prime} \in\left\{0, \ldots, T_{\text {half }}\right\}$, where $T_{\text {half }}=\lfloor(T-1) / 2\rfloor-1$ (see the coordinate system in Fig. 2).
5) Solve the problem at the dataset level instead of the image level for two reasons. First, even with the above simplifications, it still takes quite some time to obtain the sub-optimal configuration in practice.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-06.jpg?height=149&width=881&top_left_y=874&top_left_x=1077)

Fig. 5. Estimation of the bits used to code a single tile.

Second, the content-dependent configuration may render the training of DNN-based $360^{\circ}$ image compression unstable.

Putting together, Problem 13) is simplified to

$$
\begin{align*}
\min _{\left\{W_{t}\right\}} & \frac{1}{|\mathcal{D}|} \sum_{\boldsymbol{x} \in \mathcal{D}} \operatorname{RD}\left(\boldsymbol{x}, \operatorname{compress}_{\boldsymbol{\alpha}}(\boldsymbol{x}) ;\left\{W_{t}\right\}\right) \\
\text { s.t. } & W_{t} \in\left\{\bar{W}_{0}, \ldots, \bar{W}_{L-1}\right\}, t \in\{0, \ldots, T-1\}  \tag{14}\\
& W_{t}=W_{T-1-t}, t \in\left\{0, \ldots, T_{\text {half }}\right\} \\
& W_{t} \leq W_{t^{\prime}}, \text { for } t \leq t^{\prime} \text { and } t, t^{\prime} \in\left\{0, \ldots, T_{\text {half }}\right\}
\end{align*}
$$

where $\bar{W}_{t}=(t+1)\left\lfloor\frac{W}{L}\right\rfloor$. Now, an exhaustive search which evaluates all possible configurations may be feasible. Alternatively, we propose a divide-and-conquer greedy solver to Problem (14) in case $L$ and $T$ are still large. We first initialize all $\left\{W_{t}\right\}$ to $\bar{W}_{T-1}$. From the pole to the equator, we enumerate the possible widths of the current tile while holding higher-latitude tiles to the estimated widths and lower-latitude tiles to the initialized widths. This procedure is repeated until all tiles are visited, which is summarized in Algorithm 1

It remains to instantiate the objective function in Problem 14, which quantifies the rate-distortion trade-off given a particular parameter configuration $\left\{W_{t}\right\}$. To obtain an accurate estimate, it is preferable but impractical to optimize a set of DNN-based image compression models (i.e., optimize $\boldsymbol{\alpha}$ in compress $\boldsymbol{\alpha}^{(}(\cdot)$ ) for each configuration. A workaround is to apply existing codecs on each tile and "sum" the results. In our implementation, we use JPEG2000 as the offthe-shelf compression method. As context is crucial in image compression for bitrate reduction, a naїve compression of the tile without considering adjacent ones would lead to an inaccurate parameter estimation of our representation that admits pseudocylindrical padding. To alleviate this issue, we introduce a proxy rate-distortion objective. For the rate

```
Algorithm 1 Greedy Estimation of the Pseudocylindrical
Representation Parameters
    Input: ERP image set $\mathcal{D}=\left\{\boldsymbol{x}^{(0)}, \ldots, \boldsymbol{x}^{(|\mathcal{D}|-1)}\right\}$, and the
quantized width set $\left\{\bar{W}_{0}, \ldots, \bar{W}_{L-1}\right\}$.
    Output: The optimized parameter set $\left\{W_{t}^{\star}\right\}$.
    for $t \leftarrow 0$ to $T-1$ do
        $W_{t}^{\star} \leftarrow \bar{W}_{L-1} ; \quad \triangleright$ Initialization
    end for
    for $t \leftarrow 0$ to $T_{\text {half }}$ do
        $V_{\text {best }} \leftarrow \infty, \quad W_{\text {best }} \leftarrow 0$;
        if $t=0$ then
            start_width $\leftarrow \bar{W}_{0}$;
        else
            start_width $\leftarrow W_{t-1}^{\star} ;$
        end if
        for $W_{t}^{\star} \leftarrow$ start_width to $\bar{W}_{L-1}$ do
            $W_{T-t-1}^{\star} \leftarrow W_{t}^{\star}$;
            $V_{\text {temp }} \leftarrow \mathbb{E}_{\boldsymbol{x} \in \mathcal{D}} \operatorname{RD}\left(\boldsymbol{x}, \operatorname{compress}_{\boldsymbol{\alpha}}(\boldsymbol{x}) ;\left\{W_{t}^{\star}\right\}\right)$;
            if $V_{\text {temp }}<V_{\text {best }}$ then
                $V_{\text {best }} \leftarrow V_{\text {temp }}, \quad W_{\text {best }} \leftarrow W_{t} ;$
            end if
        end for
        $W_{t}^{\star} \leftarrow W_{\text {best }}, \quad W_{T-t-1}^{\star} \leftarrow W_{\text {best }} ;$
    end for
```

estimation of the $t$-th tile, $\boldsymbol{z}_{t}$, we resize the width of $\boldsymbol{x}$ to $W_{t}$, and crop two subimages such that the intersection is $\boldsymbol{z}_{t}$ and the union is the resized image. The rate of $\boldsymbol{z}_{t}$ is calculated as the difference between the number of bits of the subimages and the resized image. This process is better illustrated in Fig. 5 .

For the distortion, we suggest to use viewport-based objective quality metrics, which correctly reflect how humans view $360^{\circ}$ images. According to [36], viewport-based metrics deliver so far the best quality prediction performance on $360^{\circ}$ images. In our implementation, we use MSE as the base quality metric. We sample 14 viewports by first mapping the pseudocylindrical representation back to the unit sphere followed by rectilinear projections ${ }^{6}$ Each viewport is a $H_{v} \times W_{v}$ rectangle, where $H_{v}=\left\lceil\frac{H}{3}\right\rceil$ and $W_{v}=\left\lceil\frac{W}{4}\right\rceil$, with a field of view (FoV) of $\frac{\pi}{3} \times \frac{\pi}{2}$. Together, they cover all spherical content.

By varying the QP values of JPEG2000, we produce a rate-distortion curve for each image $\boldsymbol{x} \in \mathcal{D}$, and we average all curves ${ }^{7}$ as the rate-distortion performance of the current parameter configuration of the proposed pseudocylindrical representation. We compare two curves using the Bjontegaard delta bitrate saving (BD-BR) metric [64] to identify a better configuration.

We conclude this section by illustrating two interesting properties of our pseudocylindrical representation and convolution. First, Fig. 2 shows the optimized pseudocylindrical configuration in comparison to sinusoidal projection and its tiled version, where we set $H=512, W=1024$,

6. The centers of the 14 viewports correspond to $\left(0,-\frac{\pi}{2}\right),(0,0)$, $\left(0, \frac{\pi}{2}\right),(0, \pi),\left(-\frac{\pi}{4},-\frac{\pi}{2}\right),\left(-\frac{\pi}{4}, 0\right),\left(-\frac{\pi}{4}, \frac{\pi}{2}\right),\left(-\frac{\pi}{4}, \pi\right),\left(\frac{\pi}{4},-\frac{\pi}{2}\right),\left(\frac{\pi}{4}, 0\right)$, $\left(\frac{\pi}{4}, \frac{\pi}{2}\right),\left(\frac{\pi}{4}, \pi\right),\left(\frac{\pi}{2}, 0\right)$ and $\left(-\frac{\pi}{2}, 0\right)$, respectively, on the unit sphere.
7. We average the rates and the distortion scores separately over images compressed with identical QP values.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-07.jpg?height=455&width=702&top_left_y=149&top_left_x=1167)

Fig. 6. The optimized width configuration of the pseudocylindrical representation in comparison to ERP and the sinusoidal projection.

$H_{0}=32$, and $L=64$. The key observation is that the optimized configuration does not completely resolve the over-sampling problem in ERP (see also Fig. 6. Surprisingly, to achieve better compression performance, the mid-latitude regions are still over-sampled than the sinusoidal tiles, which is also supported by the ablation experiments in Sec. 5.5 Second, we build a six-layer DNN with $3 \times 3$ pseudocylindrical convolutions, and visualize the receptive field at different latitudes in Fig. 7. With the optimized structure as input, the proposed pseudocylindrical convolution is able to produce similar receptive fields for different locations on the sphere, which are latitude-adaptive in the ERP domain. Although the receptive field is slightly deformed, it should not be a problem in $360^{\circ}$ image compression because the kernel weights can be learned to adapt to such geometric distortions (if necessary) by optimizing viewport-based perceptual quality metrics.

## 4 LEARNED $360^{\circ}$ IMAGE COMPRESSION SYSTEM

In this section, we design a learned $360^{\circ}$ image compression system based on the proposed pseudocylindrical representation and convolution. At a high level, our system consists of an analysis network $g_{a}$, a non-uniform quantizer $g_{q}$, a synthesis network $g_{s}$, and a context-based entropy network $g_{e}$, which are jointly optimized for rate-distortion performance.

### 4.1 Analysis, Synthesis, and Entropy Networks

The analysis transform $g_{a}$ takes the ERP image $\boldsymbol{x}$ as input and maps it to the proposed pseudocylindrical representation $\boldsymbol{z}$, based on which the code representation $\boldsymbol{c}$ is produced by the network. $g_{a}$ is made of ERP to pseudocylindrical representation transform, four down-sampling blocks, four residual blocks, two attention blocks, a back-end pseudocylindrical convolution, and a sigmoid layer, whose computational graph with detailed specifications is shown in Fig 8 . The down-sampling block processes and down-samples the pseudocylindrical feature maps by a factor of two. The residual block [65| has two convolution layers with a skip connection, following each down-sampling block. A simplified attention block $\mid 66$ is added right after the second and fourth residual block to increase the model capacity and expand the receptive field. A final convolution layer with $C$ filters followed by a sigmoid activation is used to produce

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-08.jpg?height=564&width=1783&top_left_y=141&top_left_x=171)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-08.jpg?height=231&width=445&top_left_y=172&top_left_x=184)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-08.jpg?height=219&width=445&top_left_y=427&top_left_x=184)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-08.jpg?height=501&width=699&top_left_y=156&top_left_x=713)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-08.jpg?height=526&width=458&top_left_y=146&top_left_x=1492)

(c)

Fig. 7. Illustration of the receptive field of the standard and pseudocylindrical convolution. (a) ERP domain. (b) Spherical domain. (c) Viewport domain.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-08.jpg?height=554&width=1684&top_left_y=834&top_left_x=226)

Fig. 8. Analysis transform $g_{a}$. P-Conv: proposed pseudocylindrical convolution with filter support $(S \times S)$ and number of channels (output $\times$ input).

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-08.jpg?height=445&width=881&top_left_y=1515&top_left_x=167)

Up-sampling Block

Fig. 9. Synthesis transform $g_{s}$. P-Conv: proposed pseudocylindrical convolution with filter support $(S \times S)$ and number of channels (output $\times$ input).

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-08.jpg?height=326&width=832&top_left_y=2168&top_left_x=186)

Fig. 10. Entropy network $g_{e}$. Masked P-Conv: masked pseudocylindrical convolution adapted from [27. the code representation $\boldsymbol{c}$ with a desired operating range of $[0,1]$.

The synthesis transform $g_{s}$ (see Fig. 9) is a mirror of the analysis transform where the down-sampling blocks are replaced by the up-sampling blocks. Instead of performing deconvolution for upsampling, we expand the feature representation by a factor of four in the channel dimension and reshape it such that the height and width grow by a factor of two [38], [67]. $g_{s}$ ends with a pseudocylindrical representation to ERP transform to reconstruct the ERP from the proposed pseudocylindrical representation. Generalized divisive normalization (GDN) and inverse GDN as bio-inspired nonlinearities [21] are separately adopted after the last convolution of the down-sampling and upsampling blocks. For other convolution layers, unless stated otherwise, the parametric rectified linear unit (ReLU) is used as the nonlinear activation function.

As for the entropy network $g_{e}$ (see Fig. 10), we model the probability distribution of the quantized code $\overline{\boldsymbol{c}}$ as a MoG, whose accuracy can be improved by considering the code context, also known as the auto-regressive prior. Thus we employ the group contex $8^{8}$ proposed by Li et al. [27], and train three context-based DNNs to predict the mean,

8. When performing pseudocylindrical convolution, the specified code order following [27] should be respected, which requires careful treatment of linear interpolation and pseudocylindrical padding.
variance, and mixing weights, respectively. Each DNN comprises a front-end up-sampling layer and two masked pseudocylindrical convolutions [27], followed by three masked residual blocks and a back-end masked pseudocylindrical convolution. No activation function is added as the output layer in the mean branch; ReLU activation is added in the variance branch to ensure that the output is nonnegative; the softmax activation is added in the mixing weight branch to produce a probability vector as output.

### 4.2 Quantizer

The quantizer $g_{q}$ is parameterized by $\boldsymbol{\omega}=$ $\left\{\omega_{k, 0}, \ldots, \omega_{k, L_{q}-1}\right\}$, where $k=0, \ldots, C-1$ is the channel index, and $L_{q}$ is the number of quantization centers in each channel. We compute the $l$-th quantization center for the $k$-th channel by

$$
\begin{equation*}
\Omega_{k, l}=\sum_{l^{\prime}=0}^{l} \exp \left(\omega_{k, l^{\prime}}\right) \tag{15}
\end{equation*}
$$

The code $c_{k, i, j}$, where $i, j$ are spatial indices, is quantized to its nearest quantization center:

$$
\begin{equation*}
\bar{c}_{k, i, j}=g_{q}\left(c_{k, i, j}\right)=\underset{\left\{\Omega_{k, l}\right\}}{\operatorname{argmin}}\left\|c_{k, i, j}-\Omega_{k, l}\right\|_{2}^{2} \tag{16}
\end{equation*}
$$

$g_{q}$ has zero gradients almost everywhere, which hinders training via back-propagation. Inspired by [22], [68], we approximate $g_{q}$ by the identify function $\hat{g}_{q}\left(c_{k, i, j}\right)=c_{k, i, j}$. That is, we use $g_{d}$ and $\hat{g}_{d}$ in the forward and backward passes, respectively. The parameters $\omega$ can be optimized by minimizing the quantization error, i.e., $\|\boldsymbol{c}-\overline{\boldsymbol{c}}\|_{2}^{2}$, on the fly.

### 4.3 Rate-Distortion Objective

As with general data compression, our objective function is a weighted sum of the rate and distortion:

$$
\begin{equation*}
\ell=\mathbb{E}_{\boldsymbol{x} \in \mathcal{D}}\left[\ell_{r}\left(g_{q}\left(g_{a}(\boldsymbol{x})\right)\right)+\lambda \ell_{d}\left(\boldsymbol{x}, g_{s}\left(g_{q}\left(g_{a}(\boldsymbol{x})\right)\right)\right)\right] \tag{17}
\end{equation*}
$$

where $\lambda$ is the trade-off parameter, and $\mathcal{D}$ is the training set. The rate of the quantized code $\overline{\boldsymbol{c}}$ is computed by

$$
\begin{equation*}
\ell_{r}(\overline{\boldsymbol{c}})=-\mathbb{E}_{\overline{\boldsymbol{c}}}[\log p(\overline{\boldsymbol{c}})]=-\mathbb{E}\left[\sum_{k, i, j} \log p\left(\bar{c}_{k, i, j}\right)\right] \tag{18}
\end{equation*}
$$

where we omit the conditional dependency to make the notation uncluttered. The discretized probability, $p\left(\bar{c}_{k, i, j}\right)$, can be computed by integrating the continuous MoG with three components:

$p\left(\bar{c}_{k, i, j}\right)=\int_{\frac{\Omega_{k, l-1}+\Omega_{k, l}}{2}}^{\frac{\Omega_{k, l+1}+\Omega_{k, l}}{2}} \sum_{m=0}^{2} \pi_{k, i, j}^{m} \mathcal{N}\left(\xi ; \mu_{k, i, j}^{m},\left(\sigma_{k, i, j}^{m}\right)^{2}\right) d \xi$,

where we assume $\Omega_{k, l}=\bar{c}_{k, i, j}$, and $\pi_{k, i, j}^{m}, \mu_{k, i, j}^{m}$, and $\left(\sigma_{k, i, j}^{m}\right)^{2}$ are the mixing weight, mean and variance of $m$ th Gaussian component for $\bar{c}_{k, i, j}$, respectively.

As previously discussed in Sec. 3.3, we adopt viewportbased MSE and structure similarity (SSIM) as the quality measures, which are denoted by VMSE and VSSIM. It is

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-09.jpg?height=217&width=892&top_left_y=146&top_left_x=1077)

Fig. 11. Region-wise packing (RWP) for the ERP image.

noteworthy that VSSIM is a quality metric, and we need to convert it into a loss function:

$$
\begin{equation*}
\ell_{\operatorname{VSSIM}}(\boldsymbol{x}, \hat{\boldsymbol{x}})=1-\operatorname{VSSIM}(\boldsymbol{x}, \hat{\boldsymbol{x}}) \tag{20}
\end{equation*}
$$

where $\hat{\boldsymbol{x}}$ is the reconstructed image of $\boldsymbol{x}$ by our model.

## 5 EXPERIMENTS

In this section, we first describe the experimental setup, and then augment existing codecs by RWP with the height parameter optimized for rate-distortion performance (see Fig. 11. We compare our method to the augmented codecs in terms of quantitative metrics and visual quality. Last, we conduct comprehensive ablation studies to single out the contributions of the proposed techniques. The codes and the trained models will be available at: https:// github.com/ limuhit/pseudocylindrical_convolution

### 5.1 Experimental Setup

We collect 19, 790 ERP images from Flickr that carry Creative Commons licenses and save them losslessly. All images are down-sampled to the size of $512 \times 1024$ to further counteract potential compression artifacts. We split the dataset into the training set with 19,590 images and the test set with 200 images.

In main experiments, we set the height of each tile to $H_{0}=32$ such that it can be down-sampled by 16 times in the analysis transform. Correspondingly, the number of tiles in our pseudocylindrical representation is 16 . The quantization level $L$ is set to 64 in Problem 133). The tradeoff parameter $\lambda$ in Eq. 17) is in the range of $[1 / 16,1 / 3]$. For the number of channels of the latent code, we choose $C=192,112,56$ for high-bitrate, mid-bitrate, and lowbitrate models, respectively. The number of quantization levels adopted by $g_{q}$ is $L_{q}=8$.

Stochastic optimization is carried out by minimizing Eq. (17) using the Adam method |69| with a learning rate of $10^{-4}$. We decay the learning rate by a factor of 10 whenever the training plateaus. We leverage the favorable transferability of the proposed method, and pre-train it using a large set of central-perspective images with standard convolution (i.e., no pseudocylindrical padding). We then fine-tune the entire method with pseudocylindrical convolutions on the full-size $360^{\circ}$ images in the training set.

Compression methods are commonly evaluated from two aspects - rate and distortion. In this paper, we adopt the bits per pixel (bpp), calculated as the total number of bits used to code the image divided by the number of pixels, for the rate, and VMSE, viewport-based PSNR (VPSNR) and VSSIM for the distortion. Then, we are able to draw the ratedistortion (RD) curve, and compute BD-BR and Bjontegaard

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-10.jpg?height=789&width=1825&top_left_y=137&top_left_x=150)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-10.jpg?height=705&width=594&top_left_y=154&top_left_x=164)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-10.jpg?height=713&width=610&top_left_y=150&top_left_x=755)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-10.jpg?height=710&width=597&top_left_y=152&top_left_x=1363)

(c)

Fig. 12. Rate-distortion curves of different compression methods. (a) VMSE. (b) VPSNR. (c) VSSIM.

delta distortion (BD-distortion) [64]. BD-BR in the unit of percentage (\%) calculates the average bitrate saving between two rate-distortion curves, while BD-distortion calculates the average distortion improvement between two curves. A negative BD-BR value and a positive BD-distortion value represent that the test method is better than the anchor method.

### 5.2 Optimization of RWP for Existing Codecs

To the best of our knowledge, there are no learned compression methods specifically designed for $360^{\circ}$ images. Thus, we choose to augment five codecs for central-perspective images with the RWP strategy [3] - JPEG [70|, JPEG2000 [71|, BPG |72] (i.e., HEVC intra coding), Minnen18 [24], and Ballé18 |23|. As shown in Fig. 11. RWP partitions the images into three parts with a parameter to control the height of the north (and south) pole region. The assembled image is used as the input for compression. RWP can relieve the nonuniform sampling problem to a certain extent. However, the simple split-and-merge operation will break the image continuity and context, which may have a negative impact on compression performance.

We optimize over a set of the height parameter of RWP, $\{8,16,24,32,40,48,56,64,72,80,100\}$, for each of the three compression standards (JPEG, JPEG2000, and BPG) in terms of BD-VPSNR and BD-BR metrics using the original method without RWP as anchor. We find that RWP indeed improves the compression performance for all three codecs. For example, RWP helps improve $0.59 \mathrm{~dB}$ and saves $10.2 \%$ bits for JPEG. It turns out the best heights for JPEG, JPEG2000, and BPG are 64,40 , and 48 , respectively. As for DNN-based methods, we choose the height to be 48 , which offers satisfactory performance for all three compression standards.

### 5.3 Quantitative Evaluation

We compare our methods optimized for VMSE and VSSIM, with the five chosen codes and their augmented versions
TABLE 1

Performance comparison of different compression methods in terms of BD-VPSNR, BD-VSSIM, and BD-BR

| Method | RATE-VPSNR |  | RATE-VSSIM |  |
| :---: | :---: | :---: | :---: | :---: |
|  | BD-VPSNR <br> (dB) | BD-BR (\%) | BD-VSSIM | BD-BR (\%) |
| JPEG | -3.504 | 140.08 | -0.057 | 79.92 |
| JPEG + RWP | -3.080 | 112.04 | -0.048 | 59.34 |
| JPEG2000 | -1.385 | 43.86 | -0.028 | 33.67 |
| JPEG2000 + RWP | -1.300 | 40.85 | -0.025 | 31.34 |
| BPG | 0.000 | 0.00 | 0.000 | 0.00 |
| $\mathrm{BPG}+\mathrm{RWP}$ | 0.092 | -2.44 | 0.003 | -3.04 |
| Ballé18 | -0.110 | 2.90 | 0.012 | -12.41 |
| Ballé18 + RWP | -0.077 | 2.02 | 0.012 | -13.02 |
| Minnen18 | 0.187 | -4.73 | 0.016 | -16.99 |
| Minnen18 + RWP | 0.183 | -4.62 | 0.016 | -16.68 |
| Ours (VMSE) | 0.547 | -13.97 | 0.025 | -25.86 |
| Ours (VSSIM) | -0.958 | 27.99 | 0.043 | -41.84 |

with optimized RWP. Both the JPEG and JPEG2000 are based on the internal implementations in OpenCV 4.2. As for BPG, we adopt the official codec from https://bellard.org/bpg/ with the 4:2:0 chroma format. For the two DNN-based methods, we test on codes released by the respective authors.

Fig. 12 draws the rate-distortion curves, where we find that our VMSE-optimized method clearly outperforms BPG(-RWP), Minnen18(-RWP) Ballé18(-RWP), and overwhelms JPEG(-RWP) and JPEG2000(-RWP) under VMSE and VPSNR. Similar observations can be drawn for the proposed method under VSSIM. Our VMSE-optimized method ranks second in terms of VSSIM, outperforming the competing methods by a clear margin. It is interesting to note that all DNN-based compression methods optimized for VMSE achieve better VSSIM performance compared with the three compression standards. As SSIM is widely regarded as a more perceptual quality metric, it is expected DNN-based compression methods to deliver better visual quality. Table 1 lists the BD-BR and BD-distortion metrics computed from the rate-distortion curves in Fig. 12, where BPG is the anchor method. We observe that our VMSE-optimized

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-11.jpg?height=2342&width=1789&top_left_y=141&top_left_x=165)

Fig. 13. Visual quality comparison of JPEG-RWP, JPEG2000-RWP, BPG-RWP, Minnen18, and our VSSIM-optimized method using 14 viewports indexed by $(\theta, \phi)$. We quantify the distortion in the form of PSNR (dB) / SSIM under each viewport. The bitrates of the ERP image produced by JPEG-RWP, JPEG2000-RWP, BPG-RWP, Minnen18 and ours are separately $0.165 \mathrm{bpp}, 0.171 \mathrm{bpp}, 0.188 \mathrm{bpp}, 0.169 \mathrm{bpp}$, and $0.161 \mathrm{bpp}$.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-12.jpg?height=569&width=889&top_left_y=149&top_left_x=152)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-12.jpg?height=512&width=420&top_left_y=153&top_left_x=167)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-12.jpg?height=515&width=429&top_left_y=152&top_left_x=604)

(b)
Fig. 14. Rate-distortion curves of different input representations and convolutions. P-Conv: pseudocylindrical convolution.

method improves VPSNR by $0.547 \mathrm{~dB}$ and saves $13.97 \%$ bitrate on average. Our VSSIM-optimized method increases VSSIM by 0.043 and saves $41.84 \%$ bitrate on average. One caveat should be pointed out: although RWP is beneficial to JPEG, JPEG2000, and BPG in compressing $360^{\circ}$ images, it has no clear contribution to DNN-based image compression methods. For example, RWP boosts Ballé 18 by 0.033 dB, but hurts Minnen18 by $0.004 \mathrm{~dB}$. This result arises because RWP breaks image context and creates image discontinuity, to which the learned compression methods fail to adapt.

### 5.4 Qualitative Evaluation

Fig. 13 visually compares our VSSIM-optimized method against JPEG-RWP, JPEG2000-RWP, BPG-RWP, and Minnen18 at similar bitrates, from which we have several interesting observations. First, JPEG-RWP, JPEG2000-RWP, and BPG-RWP suffer from common visual artifacts such as blocking, ringing, and blurring, which are further geometrically distorted at high latitudes. Second, $360^{\circ}$ specific distortions such as the "radiation artifacts" in Viewports $\left(\frac{\pi}{2}, 0\right)$ and $\left(-\frac{\pi}{2}, 0\right)$ begin to emerge for all but the proposed method due to the over-sampling issue at poles even with RWP. This provides strong justifications of the proposed pseudocylindrical representation. Meanwhile, we also see different levels of color cast produced by all competing methods in Viewport $\left(\frac{\pi}{4}, \pi\right)$, centered at the right boundary of the ERP image (see Fig. 2). That is, the viewport is made up of pixels that are far apart in the ERP image, which may be compressed in substantially different ways. In contrast, our method with pseudocylindrical convolutions does not suffer from this problem. Third, compared to the three compression standards, Minnen18 appears to have better visual quality with less visible distortions, especially in flat regions. Overall, the proposed method offers the best quality, generating flat regions with little artifacts (see Viewport $\left(\frac{\pi}{2}, 0\right)$ ), reconstructing structures in a sharp way (see Viewport $\left(-\frac{\pi}{4}, 0\right)$ ), and producing plausible textures that are close to the original (see Viewport $\left(-\frac{\pi}{4}, \frac{\pi}{2}\right)$ ).

### 5.5 Ablation Studies

We conduct three sets of ablation experiments to single out two core contributions of the proposed method - the
TABLE 2

Performance comparison of different input representations and convolutions in terms of BD-VPSNR, BD-VSSIM, and BD-BR. The BPG is the anchor method. P-Conv: pseudocylindrical convolution

| Method | RATE-VPSNR |  | RATE-VSSIM |  |
| :--- | ---: | ---: | ---: | ---: |
|  | BD-VPSNR <br> (dB) | BD-BR <br> $(\%)$ | BD-VSSIM | BD-BR <br> $(\%)$ |
| ERP (P-Conv) | -0.161 | 4.30 | 0.013 | -14.22 |
| Sinusoidal (P-Conv) | 0.145 | -4.19 | 0.018 | -18.34 |
| Optimized (Conv) | 0.116 | -3.09 | 0.017 | -18.21 |
| Optimized (P-Conv) | $\mathbf{0 . 5 4 7}$ | $\mathbf{- 1 3 . 9 7}$ | $\mathbf{0 . 0 2 5}$ | $\mathbf{- 2 5 . 8 6}$ |

TABLE 3

The running speed in seconds of the pseudocylindrical convolution and the standard convolution on the optimized pseudocylindrical representation

| Convolution | Analysis <br> Network $\left(g_{a}\right)$ | Synthesis <br> Network $\left(g_{s}\right)$ | Entropy <br> Network $\left(g_{e}\right)$ |
| :---: | :---: | :---: | :---: |
| Conv | 0.104 | 0.105 | 0.040 |
| P-Conv | 0.107 | 0.108 | 0.045 |

pseudocylindrical representation and convolution. First, we compare the optimized pseudocylindrical representation (by solving Problem (14) using Alg. 1 with the ERP format ${ }^{9}$ and the sinusoidal tiles by setting $\bar{W}_{t}=\cos \left(\theta_{t}\right) W$ (see Fig. 2). For each input structure, we retrain a DNN with the same network architecture illustrated in Fig. 8., 9 and 10 using the same training strategy and VMSE as the distortion measure. Next, we fix the optimized pseudocylindrical representation and compare the standard convolution with zero padding to the proposed pseudocylindrical convolution with the same network structure and training strategy.

The rate-distortion curves and the corresponding BD-BR and BD-Distortion metrics are given in Fig. 14 and Tab. 2 . respectively, where BPG is the anchor. We find that both sinusoidal and the optimized pseudocylindrical tiles deliver better compression performance than ERP. Moreover, the optimized representation offers more perceptual gains (and bitrate savings) than the sinusoidal tiles, validating the effectiveness of the proposed greedy search algorithm. Our results convey a somewhat counterintuitive message: slightly oversampling mid-latitude regions is preferred over uniform sampling everywhere for $360^{\circ}$ image compression. Meanwhile, the proposed pseudocylindrical convolution significantly outperforms the standard convolution with zero padding. Last, we compare the computational speed of the proposed pseudocylindrical convolution to the standard convolution, and report the running time of the analysis, synthesis, entropy networks in Tab. 3. As can be seen, the pseudocylindrical convolution has nearly the same running speed as the standard convolution. These demonstrate the promise of pseudocylindrical convolutions in modeling $360^{\circ}$ images.[^0]

## 6 CONCLUSION AND DISCUSSION

In this paper, we have introduced a new data structure for representing $360^{\circ}$ images - pseudocylindrical representation. We also proposed the pseudocylindrical convolution that can be efficiently implemented by standard convolutions with pseudocylindrical padding. Relying on the proposed techniques, we implemented one of the first DNNbased $360^{\circ}$ image compression system that offers favorable perceptual gains at similar bitrates. In the future, we will try to perform joint optimization of the parameters of the front-end pseudocylindrical representation and the backend image compression method at the image level. This may be achievable by training another DNN that takes a $360^{\circ}$ image as the input for parameter estimation of the corresponding representation [53]. We will also explore the possibility of combining the proposed techniques with existing video codecs (e.g., HEVC, VVC, VP9, and AV1) to improve $360^{\circ}$ video compression.

The application scope of the proposed pseudocylindrical representation and convolution is far beyond $360^{\circ}$ image compression. In fact, it may serve as a canonical building block for general $360^{\circ}$ image modeling, and is particular useful for $360^{\circ}$ applications that expect efficiency, scalability, and transferability. For example, in $360^{\circ}$ image editing and enhancement, the pseudocylindrical representation may be optimized to under-sample certain parts of the image to better account for global image context. As another example, our representation with uniform sampling density (i.e., the sinusoidal tiles) may be preferable in $360^{\circ}$ computer vision tasks to localize and track objects moving from lowlatitude to high-latitude places. In either $360^{\circ}$ application, the proposed pseudocylindrical convolution enables reusing existing methods trained on central-perspective images, and requires only a small set of (labeled) $360^{\circ}$ images for efficient adaptation.

## REFERENCES

[1] J. P. Snyder, Map Projections - A Working Manual. US Government Printing Office, 1987.

[2] S.-H. Lee, S.-T. Kim, E. Yip, B.-D. Choi, J. Song, and S.-J. Ko, "Omnidirectional video coding using latitude adaptive downsampling and pixel rearrangement," Electronics Letters, vol. 53, no. 10, pp. 655-657, 2017.

[3] J. Boyce, A. Ramasubramanian, R. Skupin, G. J. Sullivan, A. Tourapis, and Y. Wang, "HEVC additional supplemental enhancement information (draft 4)," Joint Collaborative Team on Video Coding of ITU-T SG, vol. 16, 2017

[4] R. G. Youvalari, A. Aminlou, and M. M. Hannuksela, "Analysis of regional down-sampling methods for coding of omnidirectional video," in Picture Coding Symposium, 2016.

[5] M. Yu, H. Lakshman, and B. Girod, "Content adaptive representations of omnidirectional videos for cinematic virtual reality," in International Workshop on Immersive Media Experiences, 2015, pp. 16.

[6] Y. Li, J. Xu, and Z. Chen, "Spherical domain rate-distortion optimization for 360-degree video coding," in IEEE International Conference on Multimedia and Expo, 2017, pp. 709-714.

[7] Y. Liu, L. Yang, M. Xu, and Z. Wang, "Rate control schemes for panoramic video coding," Journal of Visual Communication and Image Representation, vol. 53, pp. 76-85, 2018.

[8] M. Tang, Y. Zhang, J. Wen, and S. Yang, "Optimized video coding for omnidirectional videos," in IEEE International Conference on Multimedia and Expo, 2017, pp. 799-804.

[9] X. Xiu, Y. He, and Y. Ye, "An adaptive quantization method for 360-degree video coding," in Applications of Digital Image Processing XLI, vol. 10752, 2018.
[10] J. Li, Z. Wen, S. Li, Y. Zhao, B. Guo, and J. Wen, "Novel tile segmentation scheme for omnidirectional video," in IEEE International Conference on Image Processing, 2016, pp. 370-374.

[11] J.-L. Lin, Y.-H. Lee, C.-H. Shih, S.-Y. Lin, H.-C. Lin, S.-K. Chang, P. Wang, L. Liu, and C.-C. Ju, "Efficient projection and coding tools for 360 video," IEEE Journal on Emerging and Selected Topics in Circuits and Systems, vol. 9, no. 1, pp. 84-97, 2019.

[12] Y. He, X. Xiu, P. Hanhart, Y. Ye, F. Duanmu, and Y. Wang, "Contentadaptive 360-degree video coding using hybrid cubemap projection," in Picture Coding Symposium, 2018, pp. 313-317.

[13] P. Hanhart, X. Xiu, Y. He, and Y. Ye, "360 video coding based on projection format adaptation and spherical neighboring relationship," IEEE Journal on Emerging and Selected Topics in Circuits and Systems, vol. 9, no. 1, pp. 71-83, 2018.

[14] C. Ozcinar, A. De Abreu, and A. Smolic, "Viewport-aware adaptive 360 video streaming using tiles for virtual reality," in IEEE International Conference on Image Processing, 2017, pp. 2174-2178.

[15] X. Corbillon, G. Simon, A. Devlic, and J. Chakareski, "Viewportadaptive navigable 360-degree video delivery," in IEEE International Conference on Communications, 2017.

[16] L. Zelnik-Manor, G. Peters, and P. Perona, "Squaring the circle in panoramas," in IEEE International Conference on Computer Vision, vol. 1, 2005.

[17] C.-H. Chang, M.-C. Hu, W.-H. Cheng, and Y.-Y. Chuang, "Rectangling stereographic projection for wide-angle image visualization," in IEEE International Conference on Computer Vision, 2013, pp. 2824-2831

[18] Y. W. Kim, C.-R. Lee, D.-Y. Cho, Y. H. Kwon, H.-J. Choi, and K.J. Yoon, "Automatic content-aware projection for 360 videos," in IEEE International Conference on Computer Vision, 2017, pp. 47534761.

[19] J. A. Kimerling, K. Sahr, D. White, and L. Song, "Comparing geometrical properties of global grids," Cartography and Geographic Information Science, vol. 26, no. 4, pp. 271-288, 1999.

[20] M. Eder, M. Shvets, J. Lim, and J.-M. Frahm, "Tangent images for mitigating spherical distortion," in IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp. 12 426-12 434.

[21] J. Ballé, V. Laparra, and E. P. Simoncelli, "End-to-end optimized image compression," in International Conference Learning Representations, 2017

[22] L. Theis, W. Shi, A. Cunningham, and F. Huszár, "Lossy image compression with compressive autoencoders," in International Conference Learning Representations, 2017.

[23] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, "Variational image compression with a scale hyperprior," in International Conference Learning Representations, 2018.

[24] D. Minnen, J. Ballé, and G. D. Toderici, "Joint autoregressive and hierarchical priors for learned image compression," in Neural Information Processing System, 2018.

[25] R. Torfason, F. Mentzer, E. Agustsson, M. Tschannen, R. Timofte, and L. Van Gool, "Towards image understanding from deep compression without decoding," in International Conference Learning Representations, 2018.

[26] E. Agustsson, M. Tschannen, F. Mentzer, R. Timofte, and L. V. Gool, "Generative adversarial networks for extreme learned image compression," in IEEE International Conference on Computer Vision, 2019, pp. 221-231.

[27] M. Li, K. Ma, J. You, D. Zhang, and W. Zuo, "Efficient and effective context-based convolutional entropy modeling for image compression," IEEE Transactions on Image Processing, vol. 29, pp. $5900-5911,2020$

[28] N. Zioulis, A. Karakottas, D. Zarpalas, and P. Daras, "Omnidepth: Dense depth estimation for indoors spherical panoramas," in European Conference on Computer Vision, 2018, pp. 448-465.

[29] K. Tateno, N. Navab, and F. Tombari, "Distortion-aware convolutional filters for dense prediction in panoramic images," in European Conference on Computer Vision, 2018, pp. 707-722.

[30] Y.-C. Su and K. Grauman, "Learning spherical convolution for fast features from 360 imagery," in Neural Information Processing Systems, vol. 30, 2017, pp. 529-539

[31] Y.-C. Su and K. Grauman, "Kernel transformer networks for compact spherical convolution," in IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 9442-9451.

[32] T. S. Cohen, M. Geiger, J. Köhler, and M. Welling, "Spherical CNNs," International Conference Learning Representations, 2018.

[33] C. Esteves, C. Allen-Blanchette, A. Makadia, and K. Daniilidis, "Learning SO(3) equivariant representations with spherical CNNs," in European Conference on Computer Vision, 2018, pp. 52-68.

[34] N. Perraudin, M. Defferrard, T. Kacprzak, and R. Sgier, "Deepsphere: Efficient spherical convolutional neural network with healpix sampling for cosmological applications," Astronomy and Computing, vol. 27, pp. 130-146, 2019.

[35] C. Jiang, J. Huang, K. Kashinath, P. Marcus, M. Niessner et al., "Spherical cnns on unstructured grids," arXiv preprint arXiv:1901.02039, 2019.

[36] X. Sui, K. Ma, Y. Yao, and Y. Fang, "Perceptual quality assessment of omnidirectional images as moving camera videos," IEEE Transactions on Visualization and Computer Graphics, 2021.

[37] G. Toderici, S. M. O'Malley, S. J. Hwang, D. Vincent, D. Minnen, S. Baluja, M. Covell, and R. Sukthankar, "Variable rate image compression with recurrent neural networks," arXiv:1511.06085, 2015.

[38] G. Toderici, D. Vincent, N. Johnston, S. J. Hwang, D. Minnen, J. Shor, and M. Covell, "Full resolution image compression with recurrent neural networks," in IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 5306-5314.

[39] N. Johnston, D. Vincent, D. Minnen, M. Covell, S. Singh, T. Chinen, S. J. Hwang, J. Shor, and G. Toderici, "Improved lossy image compression with priming and spatially adaptive bit rates for recurrent networks," in IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 4385-4393.

[40] O. Rippel and L. Bourdev, "Real-time adaptive image compression," in International Conference on Machine Learning, 2017, pp. 2922-2930.

[41] M. Li, W. Zuo, S. Gu, D. Zhao, and D. Zhang, "Learning convolutional networks for content-weighted image compression," in IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 3214-3223.

[42] M. Li, W. Zuo, S. Gu, J. You, and D. Zhang, "Learning contentweighted deep image compression," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 10, pp. 3446-3461, 2021.

[43] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

[44] Z. Wang, E. P. Simoncelli, and A. C. Bovik, "Multiscale structural similarity for image quality assessment," in Asilomar Conference on Signals, Systems, and Computers, 2003, pp. 1398-1402.

[45] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in Neural Information Processing System, 2014, pp. 2672-2680.

[46] R. Ghaznavi-Youvalari, A. Zare, H. Fang, A. Aminlou, Q. Xie, M. M. Hannuksela, and M. Gabbouj, "Comparison of HEVC coding schemes for tile-based viewport-adaptive streaming of omnidirectional video," in IEEE International Workshop on Multimedia Signal Processing, 2017, pp. 1-6.

[47] G. J. Sullivan, J.-R. Ohm, W.-J. Han, T. Wiegand et al., "Overview of the high efficiency video coding(HEVC) standard," IEEE Transactions on Circuits and Systems for Video Technology, vol. 22, no. 12, pp. 1649-1668, 2012.

[48] M. Xu, C. Li, S. Zhang, and P. Le Callet, "State-of-the-art in 360 video/image processing: Perception, assessment and compression," IEEE Journal of Selected Topics in Signal Processing, vol. 14, no. 1, pp. 5-26, 2020.

[49] M. Budagavi, J. Furton, G. Jin, A. Saxena, J. Wilkinson, and A. Dickerson, " 360 degrees video coding using region adaptive smoothing," in IEEE International Conference on Image Processing, 2015, pp. 750-754.

[50] Y. Liu, M. Xu, C. Li, S. Li, and Z. Wang, "A novel rate control scheme for panoramic video coding," in IEEE International Conference on Multimedia and Expo, 2017, pp. 691-696.

[51] M. Yu, H. Lakshman, and B. Girod, "A framework to evaluate omnidirectional video coding schemes," in IEEE International Symposium on Mixed and Augmented Reality, 2015, pp. 31-36.

[52] Y. Sun, A. Lu, and L. Yu, "AHG8: WS-PSNR for 360 video objective quality evaluation," in Joint Video Exploration Team of ITU-T SG16 WP3 and ISO/IEC JTC1/SC29/WG11, JVET-D0040, 4th Meeting, 2016.

[53] Y.-C. Su and K. Grauman, "Learning compressible $360^{\circ}$ video isomers," in IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 7824-7833.
[54] A. Zare, A. Aminlou, and M. M. Hannuksela, "Virtual reality content streaming: Viewport-dependent projection and tile-based techniques," in IEEE International Conference on Image Processing, 2017, pp. 1432-1436.

[55] C. Ozcinar, J. Cabrera, and A. Smolic, "Visual attention-aware omnidirectional video streaming using optimal tiles for virtual reality," IEEE Journal on Emerging and Selected Topics in Circuits and Systems, vol. 9, no. 1, pp. 217-230, 2019.

[56] H. Hadizadeh and I. V. Bajić, "Saliency-aware video compression," IEEE Transactions on Image Processing, vol. 23, no. 1, pp. 19-33, 2013.

[57] G. Luz, J. Ascenso, C. Brites, and F. Pereira, "Saliency-driven omnidirectional imaging adaptive coding: Modeling and assessment," in IEEE International Workshop on Multimedia Signal Processing, 2017, pp. 1-6.

[58] V. Sitzmann, A. Serrano, A. Pavel, M. Agrawala, D. Gutierrez, B. Masia, and G. Wetzstein, "Saliency in VR: How do people explore virtual environments?" IEEE Transactions on Visualization and Computer Graphics, vol. 24, no. 4, pp. 1633-1642, 2018

[59] E. B. Saff and A. B. Kuijlaars, "Distributing many points on a sphere," The Mathematical Intelligencer, vol. 19, no. 1, pp. 5-11, 1997.

[60] P. J. Kostelec and D. N. Rockmore, "FFTs on the rotation group," Journal of Fourier Analysis and Applications, vol. 14, no. 2, pp. 145179,2008

[61] J. R. Driscoll and D. M. Healy, "Computing Fourier transforms and convolutions on the 2-sphere," Advances in Applied Mathematics, vol. 15, no. 2, pp. 202-250, 1994

[62] A. Weinstein, "Groupoids: unifying internal and external symmetry," Notices of the AMS, vol. 43, no. 7, pp. 744-752, 1996.

[63] V. Zakharchenko, E. Alshina, A. Singh, and A. Dsouza, "AHG8: Suggested testing procedure for 360-degree video," in Joint Video Exploration Team of ITU-T SG16 WP3 and ISO/IEC JTC1/SC29/WG11, JVET-D0027, 4th Meeting, 2016.

[64] G. Bjontegaard, "Calculation of average PSNR differences between RD-curves," VCEG-M33, 2001.

[65] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770-778.

[66] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, "Learned image compression with discretized Gaussian mixture likelihoods and attention modules," in IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp. 7939-7948.

[67] W. Shi, J. Caballero, F. Huszar, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang, "Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network," in IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 1874-1883.

[68] M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio, "Binarized neural networks: Training deep neural networks with weights and activations constrained to +1 or -1, ," arXiv:1602.02830, 2016.

[69] D. Kingma and J. Ba, "Adam: A method for stochastic optimization," in International Conference Learning Representations, 2015

[70] G. K. Wallace, "The JPEG still picture compression standard," IEEE Transactions on Consumer Electronics, vol. 38, no. 1, pp. 18-34, 1992.

[71] A. Skodras, C. Christopoulos, and T. Ebrahimi, "The JPEG 2000 still image compression standard," IEEE Signal Processing Magazine, vol. 18, no. 5, pp. 36-58, 2001

[72] F. Bellard, "BPG image format," 2019. [Online]. Available: https://bellard.org/bpg/

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-14.jpg?height=330&width=239&top_left_y=2185&top_left_x=1084)

Mu Li received his BCS in Computer Science and Technology in 2015 from Harbin Institute of Technology, and the Ph.D. degree from the Department of Computing, the Hong Kong Polytechnic University, Hong Kong, China, in 2020 $\mathrm{He}$ is the owner of Hong Kong PhD Fellowship. He is currently a postdoctoral researcher at School of Data Science, the Chinese University of Hong Kong, Shenzhen. His research interests include deep learning, image processing, and image compression.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-15.jpg?height=333&width=271&top_left_y=148&top_left_x=149)

Kede Ma (S'13-M'18) received the B.E. degree from the University of Science and Technology of China, Hefei, China, in 2012, and the M.S. and Ph.D. degrees in electrical and computer engineering from the University of Waterloo, Waterloo, ON, Canada, in 2014 and 2017, respectively. He was a Research Associate with the Howard Hughes Medical Institute and New York University, New York, NY, USA, in 2018. He is currently an Assistant Professor with the Department of Computer Science, City University of Hong Kong. His research interests include perceptual image processing, computational vision, and computational photography.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-15.jpg?height=336&width=254&top_left_y=886&top_left_x=163)

Jinxing Li received the B.Sc. degree from the department of Automation, Hangzhou Dianzi University, Hangzhou, China, in 2012, the M.Sc. degree from the department of Automation, Chongqing University, Chongqing, China, in 2015, and the PhD degree from the department of Computing, Hong Kong Polytechnic University, Hong Kong, China, in 2019. Dr. Li worked at The Chinese University of Hong Kong, Shenzhen, from 2019 to 2021 . He is currently with Harbin Institute of Technology, Shenzhen, China. His research interests are pattern recognition, deep learning, medical biometrics and machine learning.

![](https://cdn.mathpix.com/cropped/2024_06_04_fea49e45d9256423f656g-15.jpg?height=334&width=271&top_left_y=1622&top_left_x=149)

David Zhang (Life Fellow, IEEE) graduated in Computer Science from Peking University. He received his MSc in 1982 and his PhD in 1985 in both Computer Science from the Harbin Institute of Technology (HIT), respectively. From 1986 to 1988 he was a Postdoctoral Fellow at Tsinghua University and then an Associate Professor at the Academia Sinica, Beijing. In 1994 he received his second $\mathrm{PhD}$ in Electrical and Computer Engineering from the University of Waterloo, Ontario, Canada. He has been a Chair Professor at the Hong Kong Polytechnic University where he is the Founding Director of Biometrics Research Centre (UGC/CRC) supported by the Hong Kong SAR Government since 1998. Currently he is Presidential Chair Professor in Chinese University of Hong Kong (Shenzhen). So far, he has published over 20 monographs, 500+ international journal papers and 40+ patents from USA/Japan/HK/China. He has been continuously 8 years listed as a Global Highly Cited Researchers in Engineering by Clarivate Analytics during 2014-2021. He is also ranked about 85 with $\mathrm{H}$-Index 120 at Top 1,000 Scientists for international Computer Science and Electronics. Professor Zhang is selected as a Fellow of the Royal Society of Canada. He also is a Croucher Senior Research Fellow, Distinguished Speaker of the IEEE Computer Society, and an IEEE Life Fellow and an IAPR Fellow.


[^0]:    9. We may as well consider the ERP tiles by setting $W_{t}=W$ as input to our pseudocylindrical DNN-based compression system, which is equivalent to feeding the entire ERP image to a standard DNN for compression.
</end of paper 2>


<paper 3>
# Scanpath Prediction in Panoramic Videos via Expected Code Length Minimization 

Mu Li, Kanglong Fan, and Kede Ma, Senior Member, IEEE


#### Abstract

Predicting human scanpaths when exploring panoramic videos is a challenging task due to the spherical geometry and the multimodality of the input, and the inherent uncertainty and diversity of the output. Most previous methods fail to give a complete treatment of these characteristics, and thus are prone to errors. In this paper, we present a simple new criterion for scanpath prediction based on principles from lossy data compression. This criterion suggests minimizing the expected code length of quantized scanpaths in a training set, which corresponds to fitting a discrete conditional probability model via maximum likelihood. Specifically, the probability model is conditioned on two modalities: a viewport sequence as the deformation-reduced visual input and a set of relative historical scanpaths projected onto respective viewports as the aligned path input. The probability model is parameterized by a product of discretized Gaussian mixture models to capture the uncertainty and the diversity of scanpaths from different users. Most importantly, the training of the probability model does not rely on the specification of "ground-truth" scanpaths for imitation learning. We also introduce a proportional-integral-derivative (PID) controller-based sampler to generate realistic human-like scanpaths from the learned probability model. Experimental results demonstrate that our method consistently produces better quantitative scanpath results in terms of prediction accuracy (by comparing to the assumed "ground-truths") and perceptual realism (through machine discrimination) over a wide range of prediction horizons. We additionally verify the perceptual realism improvement via a formal psychophysical experiment, and the generalization improvement on several unseen panoramic video datasets.


Index Terms-Panoramic videos, scanpath prediction, expected code length, maximum likelihood

## 1 INTRODUCTION

PANORAMIC videos (also known as omnidirectional, spherical, and $360^{\circ}$ videos) are gaining increasing popularity owing to their ability to provide a more immersive viewing experience. However, streaming and rendering $360^{\circ}$ videos with minimal delay for real-time immersive and interactive experiences remains a challenge due to the big data volume involved. To address this, viewport-adaptive streaming solutions have been developed, which transmit portions of the video in the user's field of view (FoV) at the highest possible quality while streaming the rest at a lower quality to save bandwidth. These solutions rely exclusively on accurate prediction of the user's future scanpath [1], [2], which is a time series of head/eye movement coordinates. Generally, scanpath prediction is an effective computational means of studying and summarizing human viewing behaviors when watching $360^{\circ}$ videos with a broad range of applications, including panoramic video production |3|, [4], compression [5], [6], processing |7], [8], and rendering [9], 10 .

In the past ten years, many scanpath prediction methods in $360^{\circ}$ videos have been proposed, differing mainly in three aspects: 1) the input formats and modalities, 2) the computational prediction mechanisms, and 3) the loss functions. For

- This project was supported in part by the National Natural Scientific Foundation of China (NSFC) under Grant No. 62102339, and Shenzhen Science and Technology Program, PR China under Grant No. RCBS20221008093121052 and GXWD20220811170130002.
- Mu Li is with the School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, China, 518055 (e-mail: limuhit@gmail.com).
- Kanglong Fan and Kede Ma are with the Department of Computer Science, City University of Hong Kong, Kowloon, Hong Kong (e-mail: kanglofan2c@my.cityu.edu.hk, kede.ma@cityu.edu.hk). the input formats and modalities, Rondón et al. [11] revealed that the user's past scanpath solely suffices to inform the prediction for time horizons shorter than two to three seconds. Nevertheless, the majority of existing methods take $360^{\circ}$ video frames as an "indispensable" form of visual input for improved scanpath prediction. Among numerous $360^{\circ}$ video representations, the equirectangular projection (ERP) format is the most widely adopted, which however exhibits noticeable geometric deformations, especially for objects at high latitudes. For the computational prediction mechanisms, existing methods are inclined to rely on external algorithms for saliency detection |11], [12], [13| or optical flow estimation [12], [13| for visual feature analysis, whose performance is inevitably upper-bounded by these external methods, which are often trained on planar rather than $360^{\circ}$ videos. After multimodal feature extraction and aggregation, a sequence-to-sequence (seq2seq) predictor, implemented by an unfolded recurrent neural network (RNN) or a transformer, is adopted to gather historical information. For the loss functions in guiding the optimization, some form of "ground-truth" scanpaths is commonly specified to gauge the prediction accuracy. A convenient choice is the mean squared error (MSE) [11], [13|, [14] or its spherical derivative [15], which assumes the underlying probability distribution to be unimodal Gaussian. Such "imitation learning" is weak at capturing the scanpath uncertainty of an individual user and the scanpath diversity of different users. The binary cross entropy (BCE) [12], [16| between the predicted probability map of the next viewpoint and the normalized (multimodal) saliency map (aggregated from multiple ground-truth scanpaths) alleviates the diversity problem in a short term, but may lead to unnatural and inconsistent long-term predictions. In addition, auxiliary

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-02.jpg?height=382&width=415&top_left_y=153&top_left_x=172)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-02.jpg?height=376&width=374&top_left_y=167&top_left_x=604)

(b)
Fig. 1. Comparison of different coordinate systems used in $360^{\circ}$ video processing. (a) Spherical coordinates $(\phi, \theta)$ and 3D Eculidean coordinates $(x, y, z)$. (b) Relative $u v$ coordinates $(u, v)$.

tasks such as fixation duration prediction [17] and adversarial training [18], [19] may be incorporated, which further complicate the overall loss calculation and optimization.

In this paper, we formulate the problem of scanpath prediction from the perspective of lossy data compression |20|. We identify a simple new criterion-minimizing the expected coding length-to learn a good discrete conditional probability model for quantized scanpaths in a training set, from which we are able to sample realistic humanlike scanpaths for a long prediction horizon. Specifically, we condition our probability model on two modalities: the historical $360^{\circ}$ video frames and the associated scanpath. To conform to the spherical nature of $360^{\circ}$ videos, we choose to sample, along the historical scanpath, a sequence of rectilinear projections of viewports as the geometric deformation-reduced visual input compared to the ERP format. We further align the visual and positional modalities by projecting the scanpath (represented by spherical coordinates) onto each of the viewports (represented by relative $u v$ coordinates, see Fig. 11. This allows us to better represent and combine the multimodal features [21] and to make easier yet better scanpath prediction in relative $u v$ space than in absolute spherical or 3D Euclidean space. To capture the uncertainty and diversity of scanpaths, we parameterize the conditional probability model by a product of discretized Gaussian mixture models (GMMs), whose weight, mean, and variance parameters are estimated using feed-forward deep neural networks (DNNs). As a result, the expected code length can be approximated by the empirical expectation of the negative log probability.

Given the learned conditional probability model of visual scanpaths, we need a computational procedure to draw samples from it to complete the scanpath prediction procedure. We propose a variant of ancestral sampling based on a proportional-integral-derivative (PID) controller [22]. We assume a proxy viewer who starts exploring the $360^{\circ}$ video from some initial viewpoint with some initial speed and acceleration. We then randomly sample a position from the learned probability distribution as the next viewpoint, and feed it to the PID controller as the new target to adjust the acceleration. The proxy viewer is thus guided to view towards the sampled viewpoint. By repeatedly sampling future viewpoints and adjusting the acceleration, we are able to generate human-like scanpaths of arbitary length.

In summary, the current work has fourfold contributions.

- We identify a neat criterion for scanpath predictionexpected code length minimization-which establishes the conceptual equivalence between scanpath prediction and lossy data compression.
- We propose to represent both visual and path contexts in the relative $u v$ coordinate system. This effectively reduces the problem of panoramic scanpath prediction to a planar one, the latter of which is more convenient for computational modeling.
- We develop a PID controller-based sampler to draw realistic, diverse, and long-term scanpaths from the learned probability model, which shows clear advantages over existing scanpath samplers.
- We conduct extensive experiments to quantitatively demonstrate the superiority of our method in terms of prediction accuracy (by comparing to "groundtruths") and perceptual realism (through machine discrimination and psychophysical testing) across different prediction horizons. We additionally verify the generalization of our method on several unseen panoramic video datasets.


## 2 RELATED WORK

In this section, we review current scanpath prediction methods in planar images, $360^{\circ}$ images, and $360^{\circ}$ videos, respectively, and put our work in the proper context.

### 2.1 Scanpath Prediction in Planar Images

Scanpath prediction has been first investigated in planar images as a generalization of non-ordered prediction of eye fixations in the form of a 2D saliency map. Ngo and Manjunath |23| used a long short-term memory (LSTM) to process the features extracted from a DNN for saccade sequence prediction. Wloka et al. [24] extracted and combined saliency information in a biologically plausible paradigm for the next fixation prediction together with a history map of previous fixations. In contrast, Xia et al. [25| constrained the input of the DNN to be localized to the current predicted fixation with no historical fixations as input. Sun et al. |17| explicitly modeled the inhibition of return 1 (IOR) mechanism when predicting the fixation location and duration. The GMM was adopted for probabilistic modeling of the next fixation. A similar work was presented in [27], where the IOR mechanism was inspired by the Guided Search 6 (GS6) [28|, a theoretical model of visual search in cognitive neuroscience.

### 2.2 Scanpath Prediction in $360^{\circ}$ Images and Videos

For scanpath prediction in $360^{\circ}$ images, Assens et al. [32| advocated the concept of the "saliency volume" as a sequence of time-indexed saliency maps in the ERP format as the prediction target. Scanpaths can be sampled from the predicted saliency volume based on maximum likelihood with IOR. Zhu et al. |33| clustered and organized the most

1. Inhibition of return is defined as the relative suppression of processing of (detection of, orienting toward, responding to) stimuli (object and events) that had recently been the focus of attention 26 .

TABLE 1

Comparison of current scanpath prediction methods in terms of the input format and modality, the computational prediction mechanism and capability, and the loss function for optimization. NLL: Negative log likelihood. BCE: Binary cross entropy loss. DTW: Dynamic time warping loss. "-" means not available for the Horizon column or not needed for the Loss column, respectively

| Method | Input Format \& Modality | External Algorithm | Sampling Method | Horizon | $\overline{\text { GT }}$ | $\overline{\text { Loss }}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ngo17 [23] | planar image | - | beam search | - | No | ![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-03.jpg?height=35&width=225&top_left_y=369&top_left_x=1713) |
| Xia19 [25] | planar image, past scanpath | - | maximizing likelihood | - | Yes | BCE |
| Sun21 | planar image, past scanpath | instance segmentation $[31]$ | beam search | - | No | NLL |
| Assens1/ $\frac{1}{32}$ | $360^{\circ}$ image in ERP | - | maximizing likelihood | - | Yes | BCE |
| Zhu18 33. | $360^{\circ}$ image in viewport \& ERP | object detection 34] | clustering \& graph cut | - | No | - |
| Assens18 18] | planar $/ 360^{\circ}$ image in ERP | - | random sampling | - | Yes | MSE \& GAN |
| Martin22 19$]$ | $360^{\circ}$ image in ERP | - | feed-forward generation | - | Yes | DTW \& GAN |
| Li18 16] | $360^{\circ}$ video in ERP, past scanpath | saliency 397, optical flow 38] | probability thresholding | $1 \mathrm{~s}$ | Yes | BCE |
| Nguyen18 [14] | $360^{\circ}$ video in ERP, past scanpath | sallency [14] | maximizing likelihood | $2.5 \mathrm{~s}$ | Yes | MSE |
| $\mathrm{Xu} 18$ | $360^{\circ}$ video in ERP, past scanpath | saliency 40], optical flow 41] | maximizing likelihood | $1 \mathrm{~s}$ | Yes | MSE |
| Xu19 | $360^{\circ}$ video in viewport, past scanpath |  | maximizing reword (likelihood) | $30 \mathrm{~ms}$ | Yes | MSE |
| Li19 43] | past \& future scanpaths (from others) | saliency [44] (optional) | maximizing likelihood | $10 \mathrm{~s}$ | Yes | MSE |
| TRACK 111 | $360^{\circ}$ video in ERP, past scanpath | saliency 14] | maximizing likelihood | $5 \mathrm{~s}$   <br> 5   | Yes | MSE |
| VPT360 45 | past scanpath | - | maximizing likelihood | $5 \mathrm{~s}$ | Yes | MSE |
| Xu22 15] | $360^{\circ}$ video in ERP, past scanpath | saliency 40], optical flow 41] | maximizing likelihood | $1 \mathrm{~s}$ | Yes | spherical MSE |

salient areas into a graph. Scanpaths were generated by maximizing the graph weights. Assens et al. [18] combined the mean squared error (MSE) with an adversarial loss to encourage realistic scanpath generation. Similarly, Martin et al. [19| trained a generative adversarial network (GAN) with the MSE replaced by a loss term based on dynamic time warping [46]. Kerkouri et al. [35] adopted a differentiable version of the argmax operation to sample fixations memorylessly, and leveraged saliency prediction as an auxiliary task.

For scanpath prediction in $360^{\circ}$ videos, Fan et al. [12] combined the saliency map, the optical flow map, and historical viewing data (in the form of scanpaths or tiles ${ }^{2}$ to calculate the probability of tiles in future frames. Built upon [12], Li et al. [16] added a correction module to check and correct outlier tiles. Nguyen et al. [14] improved panoramic saliency detection performance for scanpath prediction with the creation of a new $360^{\circ}$ video saliency dataset. $\mathrm{Xu}$ et al. |13| improved saliency detection performance from a multi-scale perspective, and advocated relative viewport displacement prediction but applying Euclidean geometry to spherical coordinates. Xu et al. [42] used deep reinforcement learning to imitate human scanpaths, but the prediction horizon is limited to $30 \mathrm{~ms}$ (i.e., one frame). $\mathrm{Li}$ et al. [43| made use of not only the historical scanpath of the current user but also the full scanpaths of other users who had previously explored the same $360^{\circ}$ video (also known as cross-user behavior analysis). Rondón et al. [11] performed a thorough root-cause analysis of existing scanpath prediction methods. They identified that visual features only start contributing to scanpath prediction for horizons longer than two to three seconds, and an RNN to process the visual features is crucial before concatenating them with positional features. To respect the spherical nature of $360^{\circ}$ videos, spherical convolution [47], [48], [49] has been adopted to process visual features [15], [50] in combination with

2. Typically, an ERP image can be divided into a set of nonoverlapping rectangular patches, i.e., tiles. Any FoV can be covered by a subset of consecutive tiles. spherical MSE as the loss function. Additionally, Chao et

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-03.jpg?height=44&width=878&top_left_y=1062&top_left_x=1079)
scanpath using its history as the sole input.

In Table 1. we contrast our scanpath prediction method with existing representative ones in terms of the input format and modality, whether to rely on external algorithms, the sampling method for the next viewpoint, the prediction horizon, whether to specify ground-truth scanpaths, and the loss function. From the table, we see that most existing panoramic scanpath predictors work directly with the ERP format for computational simplicity. Like [42], we choose to sample along the scanpath a sequence of $2 \mathrm{D}$ viewports as the visual input, and further project the scanpath onto each of the viewports for relative scanpath prediction, both of which are beneficial for mitigating geometric deformations induced by ERP. Moreover, nearly all panoramic scanpath predictors take a supervised learning approach: first specify ground-truth scanpaths, and then adopt the MSE to quantify the prediction error, which essentially corresponds to sampling the next viewpoint by maximizing unimodal Gaussian likelihood. Such a supervised learning formulation is limited to capturing the uncertainty and diversity of scanpaths. Interestingly, early work on planar scanpath prediction suggests taking an unsupervised learning approach: first specify a parametric probability model of scanpaths, and then estimate the parameters by minimizing the negative log likelihood (NLL). In a similar spirit, we optimize a probability model of panomaric visual scanpaths, as specified by a product of discretized GMMs, by minimizing the expected code length. The proposed sampling strategy is also different and physics-driven. Additionally, our method is end-to-end trainable, and does not rely on any external algorithms for visual feature analysis.

## 3 DisCRETIZEd ProbabiLitY ModeL FOR Panoramic Scanpath Prediction

In this section, we first model scanpath prediction from a probabilistic perspective, and connect it to lossy data
compression. Then, we build our probability model on the historical visual and path contexts in the relative $u v$ space. Finally, we introduce the expected code length of future scanpaths as the optimization objective for scanpath prediction.

### 3.1 Problem Formulation

Panoramic scanpath prediction aims to learn a seq2seq mapping $f:\{\mathcal{X}, s\} \mapsto r$, in which a sequence of seen $360^{\circ}$ video frames $\mathcal{X}=\left\{\boldsymbol{x}_{0}, \ldots, \boldsymbol{x}_{t}, \ldots, \boldsymbol{x}_{T-1}\right\}$ and a sequence of past viewpoints (i.e., the historical scanpath) $s=\left\{\left(\phi_{0}, \theta_{0}\right), \ldots,\left(\phi_{t}, \theta_{t}\right), \ldots,\left(\phi_{T-1}, \theta_{T-1}\right)\right\}$ are used to predict a sequence of future viewpoints (i.e., the future scanpath) $\boldsymbol{r}=\left\{\left(\phi_{T}, \theta_{T}\right), \ldots,\left(\phi_{T+S-1}, \theta_{T+S-1}\right)\right\}$. Here, $S$ indexes the discrete prediction horizon; $\left(\phi_{t}, \theta_{t}\right)$ specifies the $t$-th viewpoint in the format of (latitude, longitude), and can be transformed to other coordinate systems as well (see Fig 1 ; $\boldsymbol{x}_{t}$ denotes the $t$-th $360^{\circ}$ video frame in any format, and in this paper, we adopt the viewport representation by first inversing the plane-to-sphere mapping followed by rectilinear projection centered at $\left(\phi_{t}, \theta_{t}\right)$.

A supervised learning formulation of panoramic scanpath prediction relies on the specification of the groundtruth scanpath $\boldsymbol{r}$, and aims to optimize the predictor $f$ by

$$
\begin{equation*}
\min D(f(\mathcal{X}, \boldsymbol{s}), \boldsymbol{r})) \tag{1}
\end{equation*}
$$

where $D(\cdot, \cdot)$ is a distance measure between the predicted and ground-truth scanpaths. It is clear that Problem (1) encourages deterministic prediction, which may not adequately model the scanpath uncertainty and diversity.

Inspired by early work on planar scanpath prediction [17], [23] and optical flow estimation [53], we argue that it is preferred to formulate panoramic scanpath prediction as an unsupervised density estimation problem:

$$
\begin{equation*}
\max p(f(\mathcal{X}, \boldsymbol{s}))=\max p(\boldsymbol{r} \mid \mathcal{X}, \boldsymbol{s}) \tag{2}
\end{equation*}
$$

Generally, estimating the probability distribution in highdimensional space can be challenging due to the curse of dimensionality. Nevertheless, we can decompose $p(\boldsymbol{r} \mid \mathcal{X}, \boldsymbol{s})$ into the product of conditional probabilities of each viewpoint using the chain rule in probability theory:

$$
\begin{equation*}
p(\boldsymbol{r} \mid \mathcal{X}, \boldsymbol{s})=\prod_{t=0}^{S-1} p\left(\phi_{T+t}, \theta_{T+t} \mid \mathcal{X}, \boldsymbol{s}, \boldsymbol{c}_{t}\right) \tag{3}
\end{equation*}
$$

where $\boldsymbol{c}_{t}=\left\{\left(\phi_{T}, \theta_{T}\right),\left(\phi_{T+1}, \theta_{T+1}\right), \ldots,\left(\phi_{T+t-1}, \theta_{T+t-1}\right)\right\}$ for $t=1, \ldots, S-1$, is the set of all preceding viewpoints, and $\boldsymbol{c}_{0}=\emptyset$. The set of $\left\{\mathcal{X}, s, \boldsymbol{c}_{t}\right\}$ constitutes the contexts of $\left(\phi_{T+t}, \theta_{T+t}\right)$, among which $\mathcal{X}$ is the historical visual context, $s$ is the historical path context, and $c_{t}$ is the causal path context. For computational reasons, we may as well keep track of only the most recent visual and path contexts by placing a context window of size $R$. As a result, $\mathcal{X}$ and $s$ become $\left\{\boldsymbol{x}_{T-R}, \ldots, \boldsymbol{x}_{T-1}\right\}$ and $\left\{\left(\phi_{T-R}, \theta_{T-R}\right), \ldots,\left(\phi_{T-1}, \theta_{T-1}\right)\right\}$, respectively. As for the causal path context, we adopt human scanpaths during training, and sample them from the learned probability model during testing.

Often, viewpoints in a visual scanpath are represented by continuous values, which are amenable to lossy compression. A typical lossy data compression system consists of three major steps: transformation, quantization, and entropy coding. The transformation step maps spherical coordinates in the form of $(\phi, \theta)$ to other corrdinate systems such as 3D Eculidean coordinates used in [19] and the relaitve $u v$ coordinates advocated in the paper. The quantization step truncates input values from a larger set (e.g., a continuous set) to output values in a smaller countable set with a finite number of elements. The uniform quantizer is the most widely used:

$$
\begin{equation*}
Q(\xi)=\Delta\left\lfloor\frac{\xi}{\Delta}+\frac{1}{2}\right\rfloor \tag{4}
\end{equation*}
$$

where $\xi \in\{\phi, \theta\}$ indicates viewpoint coordinates, $\Delta$ is the quantization step size, and $\lfloor\cdot\rfloor$ denotes the floor function.

After quantization, we compute the discrete probability mass of $\left(\phi_{T+t}, \bar{\theta}_{T+t}\right)=\left(Q\left(\phi_{T+t}\right), Q\left(\theta_{T+t}\right)\right)$ by accumulating the probability density defined in the righthand side of Eq. (3) over the area $\Omega=\left[\bar{\phi}_{T+t}-1 / 2 \Delta, \bar{\phi}_{T+t}+1 / 2 \Delta\right] \times$ $\left[\bar{\theta}_{T+t}-1 / 2 \Delta, \bar{\theta}_{T+t}+1 / 2 \Delta\right]:$

$$
\begin{equation*}
P\left(\bar{\phi}_{T+t}, \bar{\theta}_{T+t} \mid \mathcal{X}, \boldsymbol{s}, \boldsymbol{c}_{t}\right)=\int_{\Omega} p\left(\bar{\phi}_{T+t}, \bar{\theta}_{T+t} \mid \mathcal{X}, \boldsymbol{s}, \boldsymbol{c}_{t}\right) d \Omega \tag{5}
\end{equation*}
$$

Finally, given a minibatch of human scanpaths $\mathcal{B}=$ $\left\{\mathcal{X}^{(i)}, \boldsymbol{s}^{(i)}\right\}_{i=1}^{B}$, where $\mathcal{X}^{(i)}=\left\{\boldsymbol{x}_{0}^{(i)}, \ldots, \boldsymbol{x}_{T-1}^{(i)}\right\}$ and $\boldsymbol{s}^{(i)}=$ $\left\{\left(\phi_{0}^{(i)}, \theta_{0}^{(i)}\right), \ldots,\left(\phi_{T-1}^{(i)}, \theta_{T-1}^{(i)}\right)\right\}$, we may use stochastic optimizers |54| to minimize the negative log-likelihood of the parameters in the discretized probability model:

$$
\begin{equation*}
\min -\frac{1}{B S} \sum_{i=1}^{B} \sum_{t=0}^{S-1} \log _{2}\left(P\left(\bar{\phi}_{T+t}^{(i)}, \bar{\theta}_{T+t}^{(i)} \mid \mathcal{X}^{(i)}, \boldsymbol{s}^{(i)}, \boldsymbol{c}_{t}^{(i)}\right)\right) \tag{6}
\end{equation*}
$$

It can be shown that this optimization is equivalent to minimizing the expected code length of training scanpaths, where $-\log _{2}\left(P\left(\bar{\phi}_{T+t}^{(i)}, \bar{\theta}_{T+t}^{(i)} \mid \mathcal{X}^{(i)}, \boldsymbol{s}^{(i)}, \boldsymbol{c}_{t}^{(i)}\right)\right)$ provides a good approximation to the code length (i.e., the number of bits) used to encode $\left(\bar{\phi}_{T+t}^{(i)}, \bar{\theta}_{T+t}^{(i)}\right)$.

We conclude this subsection by pointing out the advantages of optimizing the discretized probability model defined in Eq. (5) over its continuous counterpart in Eq. (3). From the probabilistic perspective, estimating a continuous probability density function in a high-dimensional space with a small finte sample set (as in the case of panoramic scanpath prediction) can easily lead to overfitting [55]. Discretization (Eqs. (4) and (5) introduces a regularization effect that encourages the esimated probability to be less spiky. From the computational perspective, we introduce an important hyperparameter-the quantization step size $\Delta$ that includes the continuous probability density esimation as a special case (i.e., $\Delta \rightarrow 0$ ). Thus, with a proper setting of $\Delta$, a better probability model for scanpath prediction can be obtained (see the ablation experiment in Sec. 5.4). From the conceptual perspective, optimizing a discretized probality model is deeply rooted in the well-established theory of lossy data compression, which gives us a great opportunity to transfer the recent advances in learned-based image compression [56], [57], |58] to scanpath prediction.

### 3.2 Context Modeling

### 3.2.1 Historical Visual Context Modeling

Representing panoramic content in a plane is a longstanding challenging problem that has been extensively studied in cartography. Unfortunately, there is no perfect sphere-to-plane projection, as stated in Gauss's Theorem Egregium. Therefore, the question boils down to finding a panoramic representation that is less distorted and meanwhile more convenient to work with computationally. Instead of directly adopting the ERP sequence as the historical visual context, we resort to the viewport representation [42], [59], which is less distorted and better reflects how viewers experience $360^{\circ}$ videos.

Specifically, a viewport $\boldsymbol{x} \in \mathbb{R}^{H_{v} \times W_{v}}$ with an FoV of $\phi_{v} \times$ $\theta_{v}$ is defined as the tangent plane of a sphere, centered at the tangent point (i.e., the current viewpoint in the scanpath). To simplify the parameterization, we place the viewport (in $u v$ coordinates) on the plane $x=r$ centered at $(r, 0,0)$, where $r=0.5 W_{v} \cot \left(0.5 \theta_{v}\right)$ is the radius of the sphere. As a result, a pixel location $(u, v)$ in the viewport can be conveniently represented by $(r, y, z)$ in the 3D Euclidean space, where $y=u-0.5 W_{v}+0.5$ and $z=0.5 H_{v}-v-0.5$. We rotate the center of the viewport to the current viewpoint $(\phi, \theta)$ using the Rodrigues' rotation formula, which is an efficient method for rotating an arbitrary vector $\boldsymbol{q} \in \mathbb{R}^{3}$ in $3 \mathrm{D}$ space given an axis (described by a unit-length vector $k \in \mathbb{R}^{3}=$ $\left(k_{x}, k_{y}, k_{z}\right)^{\top}$ ) and an angle of rotation $\omega$ (using the righthand rule):

$$
\begin{align*}
\boldsymbol{q}^{\text {rot }} & =\text { Rodrigues }(\boldsymbol{q} ; \boldsymbol{k}, \omega) \\
& =\left(\mathbf{I}+\sin (\omega) \mathbf{K}+\left(1-\cos (\omega) \mathbf{K}^{2}\right) \boldsymbol{q}\right. \tag{7}
\end{align*}
$$

where

$$
\mathbf{K}=\left(\begin{array}{ccc}
0 & -k_{z} & k_{y}  \tag{8}\\
k_{z} & 0 & -k_{x} \\
-k_{y} & k_{x} & 0
\end{array}\right)
$$

We use Eq. (7) to first rotate a pixel location $\boldsymbol{q}=(r, y, z)^{\top}$ in the viewport with respect to the $z$-axis by $\theta$ :

$$
\begin{equation*}
\boldsymbol{q}^{\prime}=\text { Rodrigues }\left(\boldsymbol{q} ;(0,0,1)^{\top}, \theta\right) \tag{9}
\end{equation*}
$$

and then rotate it with respect to the rotated $y$-axis

$$
\begin{equation*}
\boldsymbol{y}^{\prime}=\text { Rodrigues }\left((0,1,0)^{\top} ;(0,0,1)^{\top}, \theta\right) \tag{10}
\end{equation*}
$$

by $-\phi$ :

$$
\begin{equation*}
\boldsymbol{q}^{\text {rot }}=\text { Rodrigues }\left(\boldsymbol{q}^{\prime} ; \boldsymbol{y}^{\prime},-\phi\right) \tag{11}
\end{equation*}
$$

The rotation process described in Eqs. (9) to 11) can be compactly expressed as

$$
\begin{equation*}
\boldsymbol{q}^{\mathrm{rot}}=\mathbf{R}(\phi, \theta) \boldsymbol{q} \tag{12}
\end{equation*}
$$

where $\mathbf{R}(\phi, \theta) \in \mathbb{R}^{3 \times 3}$ is the rotation matrix. Finally, we transform $\boldsymbol{q}^{\text {rot }}=\left(q_{x}^{\text {rot }}, q_{y}^{\text {rot }}, q_{z}^{\text {rot }}\right)^{\top}$ back to the spherical coordinates:

$$
\begin{equation*}
\phi_{q}=\arcsin \left(q_{z}^{\mathrm{rot}} / r\right) \quad \text { and } \quad \theta_{q}=\arctan 2\left(q_{y}^{\mathrm{rot}} / q_{x}^{\mathrm{rot}}\right) \tag{13}
\end{equation*}
$$

where $\arctan 2(\cdot)$ is the 2 -argument arctangen ${ }^{3}$, and relate $\left(\phi_{q}, \theta_{q}\right)$ to the discrete sampling position $\left(m_{q}, n_{q}\right)$ in the ERP format:

$$
\begin{equation*}
m_{q}=\left(0.5-\phi_{q} / \pi\right) H-0.5 \tag{14}
\end{equation*}
$$[^0]

and

$$
\begin{equation*}
n_{q}=\left(\theta_{q} / 2 \pi+0.5\right) W-0.5 \tag{15}
\end{equation*}
$$

With that, we complete the mapping from the $(u, v)$ coordinates in the viewport to $(m, n)$ coordinates in the ERP format. In case the computed $(m, n)$ according to Eqs. (14) and (15) are non-integers, we interpolate its values with bilinear kernels. For each viewpoint in the historical scanpath, we generate a viewport to represent the visual content the user has viewed. The resulting viewport sequence $\mathcal{X}=\left\{\boldsymbol{x}_{T-R}, \ldots, \boldsymbol{x}_{T-1}\right\}$ can be seen as a standard planar video clip.

To extract visual features from $\left\{\mathcal{X}^{(i)}\right\}_{i=1}^{B}$, we use a variant of ResNet50 [60] by replacing the last global average pooling layer and the fully connected (FC) layer with a $1 \times 1$ convolution layer for channel dimension adjustment. We stack the $R$ historical viewports in the batch dimension to parallelize spatial feature extraction, leading to an output representation of size $(B \times R) \times C \times H \times W$, where $C, H, W$ are the channel number, the spatial height, and the spatial width, respectively. We then reshape the features to $B \times R \times(C \times H \times W)$, where we split the batch and time dimensions and flatten spatial and channel dimensions. A 1D convolution is applied to adjust the time dimension to $S$ (i.e., the prediction horizon). We last reshape the features to $(B \times S) \times(C \times H \times W)$, and adopt a multilayer perceptron (consisting of one front-end FC layer, three FC residual blocks ${ }^{4}$, and one back-end FC layer) to compute the final visual features of size $B \times S \times C_{v}$.

### 3.2.2 Historical Path Context Modeling

In previous work, panoramic scanpaths have commonly been represented using spherical coordinates $(\phi, \theta)$ (or their discrete versions ( $m, n$ ) by Eqs. 14 and (15) or 3D Euclidean coordinates $(x, y, z)$. However, these absolute coordinates are neither user-centric, meaning that the historical and future viewpoints are not relative to the current viewpoint, nor well aligned with the visual context. To remedy both, we propose to represent the scanpath in the relative $u v$ coordinates. Given the anchor time stamp $t$, we extract the viewport tangent at $\left(\phi_{t}, \theta_{t}\right)$, and project the scanpath onto it, which can be conveniently done by inverse mapping from $(\phi, \theta)$ to $(u, v)$. Specifically, we first cast $\left(\phi_{k}, \theta_{k}\right) \in s$ to 3D Euclidean coordinates:

$$
\left(\begin{array}{c}
x_{k}  \tag{16}\\
y_{k} \\
z_{k}
\end{array}\right)=\left(\begin{array}{c}
r \cos \left(\phi_{k}\right) \cos \left(\theta_{k}\right) \\
r \cos \left(\phi_{k}\right) \sin \left(\theta_{k}\right) \\
r \sin \left(\phi_{k}\right)
\end{array}\right)
$$

rotate $\left(x_{k}, y_{k}, z_{k}\right)$ by the transpose of $\mathbf{R}\left(\phi_{t}, \theta_{t}\right)$ :

$$
\left(\begin{array}{c}
x_{t k}  \tag{17}\\
y_{t k} \\
z_{t k}
\end{array}\right)=\mathbf{R}^{\top}\left(\phi_{t}, \theta_{t}\right)\left(\begin{array}{c}
x_{k} \\
y_{k} \\
z_{k}
\end{array}\right)
$$

4. The FC residual block is composed of two FC layers followed by layer normalization and leaky ReLU activation.

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-06.jpg?height=762&width=1765&top_left_y=134&top_left_x=169)

Fig. 2. Visualization of the projected scanpaths onto different viewports. The top raw shows the procedure of projecting the same scanpath $s$ onto different viewports indexed by the time stamp $t$. The orange (green) dots indicate the viewpoints before (after) the anchor blue viewpoint, from which we extract the anchor viewport for projection. The bottom row overlaps the corresponding viewport and scanpath, where we center the anchor viewpoint by Eq. 20.

and project $\left(x_{t k}, y_{t k}, z_{t k}\right)$ onto the plane $x=r$ :

$$
\left(\begin{array}{c}
x_{t k}^{\prime}  \tag{18}\\
y_{t k}^{\prime} \\
z_{t k}^{\prime}
\end{array}\right)=\left(\begin{array}{c}
r \\
y_{t k} \cdot r / x_{t k} \\
z_{t k} \cdot r / x_{t k}
\end{array}\right)
$$

where we add a subscript " $t$ " to emphasize that the historical scanpath $s$ is projected onto the anchor viewport at the $t$-th time stamp. We further convert $\left(x_{t k}^{\prime}, y_{t k}^{\prime}, z_{t k}^{\prime}\right)$ to $u v$ coordinates:

$$
\begin{equation*}
\binom{u_{t k}^{\prime}}{v_{t k}^{\prime}}=\binom{y_{t k}^{\prime}+0.5 W_{v}-0.5}{0.5 H_{v}-z_{t k}^{\prime}-0.5} \tag{19}
\end{equation*}
$$

We last shift the $u v$ plane by moving the center of viewport from $\left(0.5 W_{v}-0.5,0.5 H_{v}-0.5\right)$ to $(0,0)$. The projection of $\left(\phi_{k}, \theta_{k}\right)$ onto the $t$-th viewport is then represented using the relative $u v$ coordinates:

$$
\begin{equation*}
\binom{u_{t k}}{v_{t k}}=\binom{y_{t k}^{\prime}}{-z_{t k}^{\prime}} \tag{20}
\end{equation*}
$$

where $\left(u_{t t}, v_{t t}\right)=(0,0)$.

As shown in Fig. 2. by projecting the historical scanpath $s$ onto each of the viewports, we model the current viewpoint of interest and future viewpoints that are likely to be oriented from the viewer's perspective. Meanwhile, aligning data from different modalities has been shown to be effective in multimodal computer vision tasks [21]. Similarly, we align the visual and path contexts in the same $u v$ coordinate system, which is beneficial for scanpath prediction (as will be clear in Sec. 5.4. This also bridges the computational modeling gap between scanpath prediction in planar and panoramic videos.

To extract historical path features from $\left\{\boldsymbol{s}^{(i)}\right\}_{i=1}^{B}$, we first reshape the input from $B \times R \times(2 R+1) \times 2$, where $B, R,(2 R+1)$ are, respectively, the minibatch size, the historical viewport number, and the context window size of the projected scanpaths, to $(B \times R) \times 2(2 R+1)$, and process it with an FC layer and an FC residual block to obtain an intermediate output of size $(B \times R) \times C_{h}$. We then split the first two dimensions (i.e., $(B \times R) \times C_{h} \rightarrow B \times R \times C_{h}$ ), and append a 1D convolution layer and four 1D convolution residual block 5 .5 to produce the final historical path features of size $B \times S \times C_{h}$.

### 3.2.3 Causal Path Context Modeling

Similarly, we model the causal path context $c_{t}$ by projecting it onto the anchor viewport $\boldsymbol{x}_{T-1}$. The computational difference here is that we need to use masked computation to ensure causal modeling. Specifically, we first reshape the input from $B \times S \times 2$ to $(B \times S) \times 2$, and use an FC layer to transform the two-dimensional coordinates to a $C$ dimensional feature representation. We then stack the last two dimensions (i.e., $(B \times S) \times C \rightarrow B \times(S \times C)$ ), and apply a masked multilayer perceptron, consisting of a frontend masked FC layer, four masked FC residual blocks, and a back-end masked FC layer to compute the causal path features of size $B \times S \times C_{c}$. The masked FC layer is defined as

$$
\begin{equation*}
\boldsymbol{h}^{\top}=(\mathbf{M} \otimes \mathbf{W}) \boldsymbol{g}^{\top} \tag{21}
\end{equation*}
$$

where $\otimes$ is the Hadamard product, and $\boldsymbol{g} \in \mathbb{R}^{B \times\left(S \times C_{i}\right)}$ and $h \in \mathbb{R}^{B \times\left(S \times C_{o}\right)}$ are the input and output features, respectively. $\mathbf{W}, \mathbf{M} \in \mathbb{R}^{\left(S \times C_{o}\right) \times\left(S \times C_{i}\right)}$ are the weight and mask matrices, respectively, in which

$$
M_{i j}= \begin{cases}1 & \text { if }\left\lfloor j / C_{i}\right\rfloor<\left\lfloor i / C_{i}\right\rfloor  \tag{22}\\ 0 & \text { otherwise }\end{cases}
$$

5. The 1D convolution residual block consists of two convolutions followed by batch normalization and leaky ReLU activation.

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-07.jpg?height=255&width=1003&top_left_y=187&top_left_x=171)

Historical Visual Context

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-07.jpg?height=829&width=1765&top_left_y=453&top_left_x=169)

Fig. 3. System diagram of the proposed discretized probability model.

for the front-end layer and

$$
M_{i j}= \begin{cases}1 & \text { if }\left\lfloor j / C_{i}\right\rfloor \leq\left\lfloor i / C_{i}\right\rfloor  \tag{23}\\ 0 & \text { otherwise }\end{cases}
$$

for the hidden and back-end layers. We summarize the proposed probabilistic scanpath prediction method in Fig. 3

### 3.3 Objective Function

Inspired by the entropy modeling in the field of learned image compression [56], [57], we construct the probability model of $\overline{\boldsymbol{\eta}}_{T-1, t}=\left(\bar{u}_{T-1, t}, \bar{v}_{T-1, t}\right)^{\top}$, the quantized version of $\boldsymbol{\eta}_{T-1, t}=\left(u_{T-1, t}, v_{T-1, t}\right)^{\top}$, using a GMM with $K$ components. Our GMM is conditioned on the historical visual context $\mathcal{X}$, the historical path context $s$, and the causal path context $\boldsymbol{c}_{t}$ :

$$
\begin{align*}
& \operatorname{GMM}\left(\overline{\boldsymbol{\eta}}_{t} \mid \mathcal{X}, \boldsymbol{s}, \boldsymbol{c}_{t} ; \boldsymbol{\alpha},\left\{\boldsymbol{\mu}_{i}\right\}_{i=1}^{K},\left\{\boldsymbol{\Sigma}_{i}\right\}_{i=1}^{K}\right)= \\
& \sum_{i=1}^{K} \frac{\alpha_{i}}{2 \pi \sqrt{\left|\boldsymbol{\Sigma}_{i}\right|}} \exp \left(-\frac{1}{2}\left(\overline{\boldsymbol{\eta}}_{t}-\boldsymbol{\mu}_{\boldsymbol{i}}\right)^{\top} \boldsymbol{\Sigma}_{i}^{-1}\left(\overline{\boldsymbol{\eta}}_{t}-\boldsymbol{\mu}_{i}\right)\right) \tag{24}
\end{align*}
$$

where we omit the subscript $T-1$ in $\overline{\boldsymbol{\eta}}_{t}$ to make the notations uncluttered. Due to the fact the gradients of the quantizer in Eq. (4) are zeros almost everywhere, we follow the method in [56], and approximate the quantizer during training by adding a random noise $\epsilon$ uniformly sampled from $[-\Delta, \Delta]$ to the continuous value:

$$
\begin{equation*}
Q_{\epsilon}(\xi)=\xi+\epsilon \tag{25}
\end{equation*}
$$

$\left\{\alpha_{i}, \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right\}$ in Eq. 24) represent the mixture weight, the mean vector, and the covariance of the $i$-th Gaussian component, respectively, to be estimated. Such estimation can be made by concatenating the visual and path features (with the size of $\left.B \times S \times\left(C_{v}+C_{h}+C_{c}\right)\right)$ followed by three prediction heads to produce the mixture weight vector, $K$ mean vectors, and $K$ covariance matrices, respectively. We assume the horizontal direction $u$ and the vertical direction $v$ to be independent, resulting in diagonal covariance matrices. Each prediction head consists of a front-end FC layer, two FC residual blocks, and a back-end FC layer. We append a softmax layer at the end of the weight prediction head to ensure that the output is a probability vector. Similarly, we add the ReLU nonlinearity at the end of the covariance prediction head to ensure nonnegative outputs on the diagonals.

We then discretize the GMM model by integrating the probability density over the area $\Omega=\left[\bar{u}_{t}-1 / 2 \Delta, \bar{u}_{t}+\right.$ $1 / 2 \Delta] \times\left[\bar{v}_{t}-1 / 2 \Delta, \bar{v}_{t}+1 / 2 \Delta\right]:$

$$
\begin{equation*}
P\left(\overline{\boldsymbol{\eta}}_{t} \mid \mathcal{X}, \boldsymbol{s}, \boldsymbol{c}_{t}\right)=\int_{\Omega} \operatorname{GMM}\left(\overline{\boldsymbol{\eta}}_{t} \mid \mathcal{X}, \boldsymbol{s}, \boldsymbol{c}_{t}\right) d \Omega \tag{26}
\end{equation*}
$$

Finally, we end-to-end optimize the entire model by minimizing the expected code length of the scanpaths in a minibatch:

$$
\begin{equation*}
\min -\frac{1}{B S} \sum_{i=1}^{B} \sum_{t=0}^{S-1} \log _{2}\left(P\left(\overline{\boldsymbol{\eta}}_{t}^{(i)} \mid \mathcal{X}^{(i)}, \boldsymbol{s}^{(i)}, \boldsymbol{c}_{t}^{(i)}\right)\right) \tag{27}
\end{equation*}
$$

## 4 PID ControlLER FOR Scanpath SAMPLING

Probabilistic scanpath prediction needs to have a sampler, drawing future viewpoints from the learned probability model. Being causal (i.e., autoregressive), our probability model as a product of discretized GMMs fits naturally to ancestral sampling. That is, we start by initializing the causal path context to be an empty set and conditioning on historical visual and path contexts to draw the first viewpoint using the given sampler. We put the previously sampled viewpoint into the causal path context for the next viewpoint generation. By repeating this step, we are able to predict an $S$-length scanpath, which completes a sampling round. We then update the historical visual context by extracting a sequence of $S$ viewports along the newly sampled scanpath, which is used to override the historical path context. The causal path context is also cleared for the next round of scanpath prediction. By completing multiple rounds, our scanpath prediction method supports very long-term (and in theory arbitrary-length) scanpath generation.

It remains to specify the sampler for the next viewpoint generation based on the learned discretized probability model in Eq. 26. One straightforward instantiation is to draw a random sample from the distribution as the next viewpoint by inverse transform sampling [61]. Empirically, this sampler tends to produce less smooth scanpaths, which correspond to shaky viewport sequences. Another option is to sample the next viewpoint that has the maximum probability mass. This sampler is closely related to directly regressing the next viewpoint in the supervised learning setting, and is thus likely to produce similar repeated scanpaths and may even get stuck in one position for a long period of time.

To address these issues, we propose to use a PID con-

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-08.jpg?height=48&width=892&top_left_y=1521&top_left_x=150)
troller is a widely used feedback mechanism that allows for continuous modulation of control signals to achieve stable control. Here we assume a proxy viewer based on Newton's laws of motion. At the beginning, the proxy viewer is placed at the starting point $\hat{\boldsymbol{\eta}}_{-1}=(0,0)$ in the $u v$ coordinate system, with the given initial speed $\boldsymbol{b}_{-1}$ and acceleration $\boldsymbol{a}_{-1}$. The $t$-th predicted viewpoint is given by

$$
\begin{equation*}
\hat{\boldsymbol{\eta}}_{t}=\hat{\boldsymbol{\eta}}_{t-1}+\Delta \tau \boldsymbol{b}_{t-1}+\frac{1}{2}(\Delta \tau)^{2} \boldsymbol{a}_{t-1}, t \in\{0, \ldots, S-1\} \tag{28}
\end{equation*}
$$

where the speed $\boldsymbol{b}_{t-1}$ is updated by

$$
\begin{equation*}
\boldsymbol{b}_{t}=\boldsymbol{b}_{t-1}+\Delta \tau \boldsymbol{a}_{t-1} \tag{29}
\end{equation*}
$$

and $\Delta \tau$ is the sampling interval (i.e., the inverse of the sampling rate). To update the accelartion $\boldsymbol{a}_{t-1}$, we first provide a reference viewpoint $\overline{\boldsymbol{\eta}}_{t}$ for $\hat{\boldsymbol{\eta}}_{t}$ by drawing a sample from $P\left(\overline{\boldsymbol{\eta}}_{t} \mid \mathcal{X}, \boldsymbol{s}, \boldsymbol{c}_{t}\right)$, where $\boldsymbol{c}_{t}=\left\{\hat{\boldsymbol{\eta}}_{0}, \ldots, \hat{\boldsymbol{\eta}}_{t-1}\right\}$, using inverse transform sampling. An error signal can then be generated:

$$
\begin{equation*}
\boldsymbol{e}_{t}=\overline{\boldsymbol{\eta}}_{t}-\hat{\boldsymbol{\eta}}_{t} \tag{30}
\end{equation*}
$$

6. It is worth noting that our independence assumption between the horizontal direction $u$ and the vertical direction $v$ admits efficient inverse transform sampling in 1D.
TABLE 2

Summary of panoramic video datasets for scanpath prediction. In the last column, NP indicates natural photographic videos, while CG stands for computer-generated videos

| Dataset | \# of Videos | \# of Scanpaths | Duration | Type |
| :--- | :---: | :---: | :---: | :--- |
| NOSSDAV17 [12] | 10 | 250 | $60 \mathrm{~s}$ | NP/CG |
| ICBD16 63 | 16 | 976 | $30 \mathrm{~s}$ | NP |
| MMSys17 64 | 18 | 864 | $164-655 \mathrm{~s}$ | NP |
| MMSy18 65 | 19 | 1,083 | $20 \mathrm{~s}$ | NP |
| PAMI19 42 | 76 | 4,408 | $10-80 \mathrm{~s}$ | NP/CG |
| CVPR18 13 | 208 | 6,672 | $20-60 \mathrm{~s}$ | NP |
| VRW23 66 | 502 | 20,080 | $15 \mathrm{~s}$ | NP/CG |

which is fed to the PID controller for acceleration adjustment through

$$
\begin{equation*}
\boldsymbol{a}_{t}=K_{p} \boldsymbol{e}_{t}+K_{i} \sum_{\tau=0}^{t} \boldsymbol{e}_{\tau}+K_{d}\left(\boldsymbol{e}_{t}-\boldsymbol{e}_{t-1}\right) \tag{31}
\end{equation*}
$$

where $K_{p}, K_{i}$, and $K_{d}$ are the proportional, integral, and derivative gains, respectively. One subtlety is that when we move to the next sampling round, we need to transfer and represent $\boldsymbol{b}_{t}$ and $\boldsymbol{a}_{t}$ in the new $u v$ space defined on the viewport at the time stamp $T+S-1$ (instead of the time stamp $T-1$ ). This can be done by keeping track of one more computed viewpoint using Eq. 28 for speed and acceleration computation in the new $u v$ space. In practice, it suffices to transfer only the average speed (i.e., $\left(\hat{\boldsymbol{\eta}}_{S}-\hat{\boldsymbol{\eta}}_{S-1}\right) / \Delta \tau$ ), and reset the acceleration to zero because the latter is usually quite small.

## 5 EXPERIMENTS

In this section, we first describe the panoramic video datasets used as evaluation benchmarks, followed by the experimental setups. We then compare our method with existing panoramic scanpath predictors in terms of prediction accuracy, perceptual realism, and generalization on unseen datasets. We last conduct comprehensive ablation studies to single out the contributions of the proposed method. The trained models and the accompanying code will be made available at https://github.com/limuhit/panoramic_ video_scanpath

### 5.1 Datasets

Panoramic video datasets typically contain eye-tracking data, in the form of eye movements and head orientations, collected from human participants. We list some basic information of commonly used $360^{\circ}$ video datasets in Table 2

Based on the dataset scale, we have selected the CVPR18 dataset [13] and the VRW23 dataset [66] for the main experiments, and leave some of the remaining for cross-dataset generalization testing. To illustrate the diversity of the scanpaths in the two datasets, we evaluate the consistency of two scanpaths of the same length using the temporal correlation:

$\operatorname{TC}\left(\boldsymbol{s}^{(i)}, \boldsymbol{s}^{(j)}\right)=\frac{1}{2}\left(\operatorname{PCC}\left(\boldsymbol{\phi}^{(i)}, \boldsymbol{\phi}^{(j)}\right)+\operatorname{PCC}\left(\boldsymbol{\theta}^{(i)}, \boldsymbol{\theta}^{(j)}\right)\right)$

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-09.jpg?height=610&width=811&top_left_y=150&top_left_x=191)

Fig. 4. meanTC histograms of the CVPR and VRW23 datasets.

where $\mathrm{PCC}(\cdot)$ is the function to compute the Pearson correlation coefficient. The mean temporal correlation over $N$ scanpaths for the same $360^{\circ}$ video can be computed by

$$
\begin{equation*}
\operatorname{meanTC}\left(\left\{\boldsymbol{s}^{(i)}\right\}_{i=1}^{N}\right)=\frac{\sum_{i=1}^{N} \sum_{j=i+1}^{N} \mathrm{TC}\left(\boldsymbol{s}^{(i)}, \boldsymbol{s}^{(j)}\right)}{N(N-1) / 2} \tag{33}
\end{equation*}
$$

meanTC ranges from $[-1,1]$, with a larger value indicating higher temporal consistency.

We visualize the meanTC histograms of the CVPR18 and VRW23 datasets in Fig. 4, from which we observe that scanpaths in both datasets exhibit considerable diversity, which shall be computationally modeled. Moreover, the scanpaths with longer horizons in the CVPR18 dataset (e.g., more than 30 seconds) are even less consistent, showing the difficulty of the long-term scanpath prediction.

### 5.2 Experimental Setups

In the main experiments, we inherit the same sampling rate in previous studies [11], [13] to downsample both the video (and the corresponding) scanpaths to five frames (and viewpoints) per second. We use one second as the context window size to create the visual and path history (i.e., $R=5$ ), and produce one-second future scanpath (i.e., the prediction horizon $S=5$ ). As for predicting scanpaths that are longer than one second, we just apply our PID controller-based sampling strategy multiple rounds, as described in Sec. 4 .

We set the quantization step size $\Delta$ in Eq. (4) to 0.2 (with a quantization error $<0.034^{\circ}$ ). The resolution of the extracted viewport is set to $H_{v} \times W_{v}=252 \times 484$, covering an $\mathrm{FoV}$ of $\phi_{v} \times \theta_{v}=63^{\circ} \times 112^{\circ}$. As shown in Fig. 3. for the historical visual context, we set $H=8$, $W=14, C=16$, and $C_{v}=128$; for the historical path context, we set $C_{h}=128$; for the causal patch context, we set $C_{c}=32$. The number of Gaussian components $K$ in GMM in Eq. (24) is set to 3. For the PID controller, we set the sampling interval $\Delta \tau$ in Eq. (28) as the inverse of the sampling rate, i.e., $\Delta \tau=0.2$ second. The set of parameters in the PID controller to adjust the accelaration in Eq. (31) are set using the Ziegler-Nichols method [67] to $K_{p}=0.6 K_{u}$, $K_{i}=2 K_{u} / P_{u}$, and $K_{d}=K_{u} P_{u} / 8$, respectively. For the CVPR18 dataset, we set $K_{u}=60$ and $P_{u}=0.29$, while for the VRW23 dataset, we set $K_{u}=90$ and $P_{u}=0.29$.

During model training, we first initialize the convolution layers in ResNet-50 for visual feature extraction with the pre-trained weights on ImageNet, and initialize the remaining parameters by He's method |68|. We then optimize the entire method by minimizing Eq. (27) using Adam [54] with an initial learning rate of $10^{-4}$ and a minibatch size of $B=48$ (where we parallelize data on 4 NVIDIA A100 cards). We decay the learning rate by a factor of 10 whenever the training plateaus. We train two separate models, one for the CVPR18 dataset by following the suggestion of the training/test set splitting in [13| and the other for the VRW23 dataset, in which we use the first 400 videos for training and the rest 102 videos for testing.

We evaluate panoramic video scanpath predictors from three perspectives: 1) prediction accuracy, 2) perceptual realism, and 3) generalization to unseen datasets. For prediction accuracy evaluation, we introduce two quantitative metrics: minimum orthodromic distance ${ }^{7}$ and maximum temporal correlation. Specifically, given a panoramic video, we define the set of scanpaths, $\mathcal{S}=\left\{\boldsymbol{s}^{(i)}\right\}_{i=1}^{|\mathcal{S}|}$, corresponding to $|\mathcal{S}|$ different viewers as the ground-truths. The minimum orthodromic distance between $\mathcal{S}$ and the set of predicted $\hat{\mathcal{S}}=\left\{\hat{\boldsymbol{s}}^{(i)}\right\}_{i=1}^{|\hat{\mathcal{S}}|}$ can be computed by

$$
\begin{equation*}
\operatorname{minOD}(\mathcal{S}, \hat{\mathcal{S}})=\min _{s \in \mathcal{S}, \hat{s} \in \hat{\mathcal{S}}} \mathrm{OD}(s, \hat{s}) \tag{34}
\end{equation*}
$$

where the $\operatorname{OD}(\cdot, \cdot)$ between two scanpaths of the same length is defined as

$$
\begin{gather*}
\mathrm{OD}(\boldsymbol{s}, \hat{\boldsymbol{s}})=\frac{1}{T} \sum_{t=0}^{T-1} \arccos \left(\cos \left(\phi_{t}\right) \cos \left(\hat{\phi}_{t}\right) \cos \left(\theta_{t}-\hat{\theta}_{t}\right)\right.  \tag{35}\\
\left.+\sin \left(\phi_{t}\right) \sin \left(\hat{\phi}_{t}\right)\right)
\end{gather*}
$$

Similarly, the maximum temporal correlation between $\mathcal{S}$ and $\hat{\mathcal{S}}$ is calculated by

$$
\begin{equation*}
\operatorname{maxTC}(\mathcal{S}, \hat{\mathcal{S}})=\max _{s \in \mathcal{S}, \hat{\boldsymbol{s}} \in \hat{\mathcal{S}}} \mathrm{TC}(\boldsymbol{s}, \hat{\boldsymbol{s}}) \tag{36}
\end{equation*}
$$

It is noteworthy that we intentionally opt for best-case setto-set distance metrics to avoid specifying, for each predicted scanpath from $\hat{\mathcal{S}}$, one ground-truth from $\mathcal{S}$. Moreover, such distances have the advantage over path-to-path distances in terms of quantifying the prediction accuracy without penalizing the generation diversity.

Additionally, inspired by the time-delay embedding technique in dynamical systems [69], [70], we introduce the sliced versions of the minimum orthodromic distance and the maximum temporal correlation, respectively. We first slice each ground-truth scanpath $s^{(i)} \in \mathcal{S}$, for $i \in 1, \ldots,|\mathcal{S}|$, into $N_{s}$ overlapping sub-paths of length $T_{s},\left\{\boldsymbol{s}_{t}^{(i)}\right\}_{t=1}^{N_{s}}$, in which the overlap between two consecutive sub-paths is set to $\left\lfloor T_{s} / 2\right\rfloor$. This gives rise to $N_{s}$ sets of sliced scanpaths $\left\{\mathcal{S}_{t}\right\}_{t=1}^{N_{s}}$, where $\mathcal{S}_{t}=\left\{\boldsymbol{s}_{t}^{(i)}\right\}_{i=1}^{|\mathcal{S}|}$. Similarly, for the predicted scanpath set $\hat{\mathcal{S}}$, we create $N_{s}$ sets of sliced scanpaths

7. The orthodromic distance is also known as the great-circle or the spherical distance.

TABLE 3

Comparison results in terms of minOD and maxTC, and their sliced versions SminOD and SmaxTC on the CVPR18 dataset. The slice length,

$T_{s}$, is set to one of the three values $\{5,10,15\}$, corresponding to predicted scanpaths of one-second, two-second, and three-second long, respectively. The prediction horizon, $S$, is set to the entire duration of each test video, excluding the initial that severs as the historical context. The top two results are highlighted in bold

| Model | minOD $\downarrow$ | SminOD-5 $\downarrow$ | SminOD-10 $\downarrow$ | SminOD-15 $\downarrow$ | maxTC $\uparrow$ | SmaxTC-5 $\uparrow$ | SmaxTC-10 $\uparrow$ | SmaxTC-15 $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Path-Only | 0.629 | 0.282 | 0.334 | 0.367 | 0.382 | 0.854 | 0.812 |  |
| Nguyen18 (CB-sal) | 0.779 | 0.418 | 0.468 | 0.508 | 0.293 | 0.800 | 0.641 |  |
| Nguyen18 (GT-sal) | 0.808 | 0.466 | 0.521 | 0.566 | 0.277 | 0.814 | 0.653 |  |
| Xu18 (CB-sal) | 0.977 | 0.626 | 0.688 | 0.724 | 0.395 | 0.776 | 0.544 |  |
| Xu18 (GT-sal) | $\mathbf{0 . 5 2 2}$ | 0.236 | 0.280 | 0.309 | 0.467 | 0.894 | 0.798 |  |
| TRACK (CB-sal) | 0.852 | 0.407 | 0.473 | 0.519 | 0.392 | 0.931 | 0.864 |  |
| TRACK (GT-sal) | $\mathbf{0 . 4 5 6}$ | $\mathbf{0 . 1 9 7}$ | $\mathbf{0 . 2 3 4}$ | $\mathbf{0 . 2 6 1}$ | 0.498 | 0.898 | 0.750 |  |
| Ours-5 | 0.773 | 0.215 | 0.268 | 0.311 | $\mathbf{0 . 6 4 4}$ | 0.812 |  |  |
| Ours-20 | 0.627 | $\mathbf{0 . 1 1 9}$ | $\mathbf{0 . 1 5 7}$ | $\mathbf{0 . 1 9 0}$ | $\mathbf{0 . 7 0 8}$ | $\mathbf{0 . 9 8 8}$ | 0.805 |  |

TABLE 4

Comparison results in terms of minOD and maxTC, and their sliced versions SminOD and SmaxTC on the VRW23 dataset

| Model | minOD $\downarrow$ | SminOD-5 $\downarrow$ | SminOD-10 $\downarrow$ | SminOD-15 $\downarrow$ | maxTC $\uparrow$ | SmaxTC-5 $\uparrow$ | SmaxTC-10 $\uparrow$ | SmaxTC-15 $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Path-Only | 1.072 | 0.309 | 0.360 | 0.386 | 0.676 | 0.949 | 0.902 |  |
| Nguyen18 (CB-sal) | 1.141 | 0.770 | 0.868 | 0.923 | 0.425 | 0.718 | 0.590 |  |
| Nguyen18 (GT-sal) | 1.063 | 0.726 | 0.804 | 0.851 | 0.415 | 0.709 | 0.587 |  |
| Xu18 (CB-sal) | 1.185 | 0.437 | 0.494 | 0.527 | 0.637 | 0.795 | 0.718 |  |
| Xu18 (GT-sal) | 1.215 | 0.397 | 0.460 | 0.511 | 0.618 | 0.773 | 0.727 |  |
| TRACK (CB-sal) | 1.067 | 0.348 | 0.400 | 0.430 | 0.699 | 0.953 | 0.914 |  |
| TRACK (GT-sal) | 0.966 | 0.259 | 0.307 | 0.335 | 0.686 | 0.907 | 0.862 |  |
| Ours-5 | $\mathbf{0 . 6 4 5}$ | $\mathbf{0 . 1 7 1}$ | $\mathbf{0 . 2 4 1}$ | $\mathbf{0 . 2 9 6}$ | $\mathbf{0 . 7 3 8}$ | $\mathbf{0 . 9 8 9}$ | 0.728 |  |
| Ours-20 | $\mathbf{0 . 5 4 2}$ | $\mathbf{0 . 1 1 8}$ | $\mathbf{0 . 1 7 7}$ | $\mathbf{0 . 2 2 6}$ | $\mathbf{0 . 7 9 6}$ | $\mathbf{0 . 9 9 5}$ | $\mathbf{0 . 9 6 6}$ |  |

$\left\{\hat{\mathcal{S}}_{t}\right\}_{t=1}^{N_{s}}$, where $\hat{\mathcal{S}}_{t}=\left\{\hat{\boldsymbol{s}}_{t}^{(j)}\right\}_{j=1}^{|\hat{\mathcal{S}}|}$, and compute the sliced minimum orthodromic distance and the sliced maximum temporal correlation by

$$
\begin{equation*}
\operatorname{SminOD}(\mathcal{S}, \hat{\mathcal{S}})=\frac{1}{N_{s}} \sum_{t=1}^{N_{s}} \operatorname{minOD}\left(\mathcal{S}_{t}, \hat{\mathcal{S}}_{t}\right) \tag{37}
\end{equation*}
$$

and

$$
\begin{equation*}
\operatorname{SmaxTC}(\mathcal{S}, \hat{\mathcal{S}})=\frac{1}{N_{s}} \sum_{t=1}^{N_{s}} \operatorname{maxTC}\left(\mathcal{S}_{t}, \hat{\mathcal{S}}_{t}\right) \tag{38}
\end{equation*}
$$

respectively. In the experiments, $T_{s}$ is set to $\{5,10,15\}$, corresponding to one-second, two-second, and three-second sliced scanpaths, respectively. We will append the corresponding number to the evaluate metric (e.g., SminOD-5) to differentiate the three different settings. After determining $T_{s}, N_{s}$ can be set accordingly. Generally, the OD metric family focuses more on the pointwise local comparison, while the TC metric family emphasizes more on global covariance measurement.

For perceptual realism evaluation, we first train a separate classifier for each scanpath predictor to discriminate its predicted scanpaths from those generated by humans. The underlying idea is conceptually similar to that in GANs |71|, except that we perform post hoc training of the classifier as the discriminator. A higher classification accuracy indicates poorer perceptual realism. Rather than solely relying on machine discrimination, we also perform a formal psychophysical experiment to quantify the perceptual realism of scanpaths. We reserve the details on how to train the classifiers and how to perform the psychophysical experiment in later subsections.
For generalization evaluation, we resort to the MMSys18 [65] and the PAMI19 [42] datasets, which consist of 19 and 76 distinct paromanic scenes, respectively (see Table 2).

### 5.3 Main Experiments

### 5.3.1 Prediction Accuracy Results

We compare the proposed method with several panoramic scanpath predictors, including a path-only seq2seq model [11], Nguyen18 [14], Xu18 [13|, and TRACK [11]. Nguyen18, Xu18, and TRACK rely on external saliency models for scanpath prediction. We follow the experimental setting in [11], and exploit two types of saliency maps. The first type is the content-based saliency map produced by a panoramic saliency model ||14|, denoted by CB-sal. The second type is the ground-truth saliency map aggregated spatiotemporally from multiple human viewers, denoted by GT-sal. Nevertheless, we point out two caveats when using ground-truth saliency maps. First, future scanpaths may be unavoidable to participate in the computation of the saliency map at the current time stamp. Second, the saliency prediction module is ahead of the scanpath prediction module for some competing methods such as TRACK [11]. Both cases violate the causal assumption in scanpath prediction if the ground-truth saliency map is exploited.

We re-train all competing models, following the respective training procedures. The prediction horizon $S$ for the path-only model, Nguyen18, Xu18, and TRACK during training is set to $25,15,5$, and 25 , respectively. All competing methods are deterministic, producing a single scanpath for each test panoramic video (i.e., $|\mathcal{S}|=1$ in Eqs. (34), 36), 37) and (38)). In stark contrast, our method is designed to be probabilistic as a natural way of capturing the uncertainty and diversity of scanpaths. Thus, we report

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-11.jpg?height=1249&width=1569&top_left_y=194&top_left_x=251)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-11.jpg?height=545&width=724&top_left_y=215&top_left_x=272)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-11.jpg?height=558&width=713&top_left_y=846&top_left_x=270)

(c)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-11.jpg?height=548&width=724&top_left_y=214&top_left_x=1080)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-11.jpg?height=558&width=745&top_left_y=846&top_left_x=1059)

(d)

Fig. 5. Scanpath prediction performance in terms of minOD, maxTC, SminOD-5, and SmaxTC-5 on the CVPR18 dataset as a function of the prediction horizon.

the results of two variants of the proposed method, one samples 5 scanpaths for each test video (i.e., $|\hat{\mathcal{S}}|=5$ ), denoted by Ours-5, and the other samples 20 scanpaths (i.e., $|\hat{\mathcal{S}}|=20$ ), denoted by Ours-20.

We report the minOD, maxTC, SminOD, and SmaxTC results of all methods on the CVPR18 dataset in Table 3 and on the VRW23 dataset in Table 4, respectively. The prediction horizon $S$ is set to 150 (corresponding to a 30 second scanpath) for CVPR18 dataset and 50 (corresponding to a 10-second scanpath) for VRW23 dataset. The slice length, $T_{s}$, for computing SminOD and SmaxTC is set to one of the three values, $\{5,10,15\}$. From the tables, we make several interesting observations. First, the path-only model provides a highly nontrivial solution to panoramic scanpath prediction, consistent with the observation in [11]. This also explains the emerging but possibly "biased" view that the historical scanpath is all you need [45]. In particular, the path-only model performs better (or at least on par with) Xu18 (CB sal) and TRACK (CB sal) under the OD metric family. Second, the performance of saliency-based scanpath predictors improve when the ground-truth saliency maps are allowed on the CVPR18 dataset. This provides evidence that in our experimental setting, the (historical) visual context can be beneficial, if it is extracted and incorporated properly. Nevertheless, such visual information may be less useful when the prediction horizon is relatively short, or even harmful with inapt incorporation, as evidenced by the temporal correlation results on the VRW23 dataset. Third, the proposed methods provide consistent performance improvements on both datasets and under all evaluation metrics (except for minOD on the CVPR18 dataset).

We take a closer look at the performance variations of scanpath predictors by varying the prediction horizon in the unit of second. Figs. 5 and 6 show the results under minOD, maxTC, SminOD-5, and SmaxTC-5 on the CVPR18 and VRW23 datasets, respectively. We find that initially our methods underperform slightly but quickly catch up and significantly outperform the competing methods in the long run. This makes sense because deterministic methods are typically optimized for pointwise distance losses, and thus perform more accurately at the beginning with highly consistent viewpoints. As the prediction horizon increases, different viewers tend to explore the panoramic virtual scene in rather different ways, leading to diverse scanpaths that cause deterministic methods to degrade. Meanwhile, we also make a similar "counterintuitive" observation: models with predicted saliency show noticeably better temporal correlation but poorer orthodromic distance than those with ground-truth saliency on the VRW23 dataset (not on the CVPR18 dataset). We believe these may arise because of the

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-12.jpg?height=1244&width=1569&top_left_y=197&top_left_x=251)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-12.jpg?height=542&width=721&top_left_y=217&top_left_x=274)

(a)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-12.jpg?height=561&width=724&top_left_y=844&top_left_x=272)

(c)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-12.jpg?height=545&width=724&top_left_y=215&top_left_x=1080)

(b)

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-12.jpg?height=558&width=740&top_left_y=846&top_left_x=1061)

(d)

Fig. 6. Scanpath prediction performance in terms of minOD, maxTC, SminOD-5, and SmaxTC-5 on the VRW23 dataset as a function of the prediction horizon.

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-12.jpg?height=266&width=879&top_left_y=1580&top_left_x=165)

Fig. 7. The structure of the classifier to test the perceptual realism of predicted scanpaths.

interplay of the differences in dataset characteristics (e.g., the duration of panoramic videos) and in metric emphasis (i.e., local pointwise versus global listwise comparison). In addition, our methods are fairly stable under sliced metrics.

### 5.3.2 Perceptual Realism Results

Machine Discrimination. Apart from prediction accuracy, we also evaluate the perceptual realism of the predicted scanpaths. We first take a machine discrimination approach: train DNN-based binary classifiers to discriminate whether input viewport sequences are real (i.e., sampled along human scanpaths) or fake (i.e., sampled along machinepredicted scanpaths). As shown in Fig. 7. we adopt a variant of ResNet-50 (the same as used in Sec. 3.2.1) to extract the visual features from $B$ input viewport sequences with $L$ frames, leading to the intermediate representation of size $B \times L \times C \times H \times W$. We then reshape it to $(B \times L) \times(C \times H \times W)$, and process the representation with four residual blocks and a back-end FC layer to produce an output representation of size $B \times L$. Inspired by the multihead attention in $[51$, our residual block consists of a frontend FC layer, a transposing operation, a 2D convolution with a kernel size of $1 \times 9$, a second transposing operation, and a back-end FC layer with a skip connection. After the front-end FC layer, we split the representation into $D$ parts, with the size of $(B \times L) \times(D \times E)$, which is transposed to $B \times D \times E \times L$. We then apply $2 \mathrm{D}$ convolution, and transpose the convolved representation back to $(B \times L) \times(D \times E)$. We further process it with the back-end FC layer to generate the output of size $(B \times L) \times(C \times H \times W)$, which is added to the input as the final feature representation. Last, we take the average of the output features along the time dimension, and add a sigmoid activation to estimate the probabilities.

We train the classifiers to minimize the cross-entropy loss, following the training procedures described in Sec. 5.2 We test the classifiers using the classification accuracy, the $F_{1}$ score, and the cross-entropy objective. It is clear from Table 5 that, our method outperforms the others on both datasets. Moreover, all methods have better results on the

TABLE 5

Perceptual realism results through machine discrimination on the CVPR18 and VRW23 datasets. CE stands for the cross entropy objective

| Model | CVPR18 Dataset |  |  | VRW23 Dataset |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Acc $\downarrow$ | $F_{1} \downarrow$ | CE $\uparrow$ | Acc $\downarrow$ | $F_{1} \downarrow$ | CE $\uparrow$ |
| Path-Only | 0.992 | 0.992 | 0.027 | 0.962 | 0.962 | 0.110 |
| Nguyen18 (CB-sal) | 0.999 | 0.999 | 0.005 | 0.996 | 0.996 | 0.007 |
| Nguyen18 (GT-sal) | 0.999 | 0.999 | 0.002 | 0.994 | 0.994 | 0.024 |
| Xu18 (CB-sal) | 0.980 | 0.981 | 0.061 | 0.978 | 0.978 | 0.094 |
| Xu18 (GT-sal) | 0.999 | 0.999 | 0.008 | 0.995 | 0.995 | 0.022 |
| TRACK (CB-sal) | 0.993 | 0.993 | 0.023 | 0.949 | 0.950 | 0.154 |
| TRACK (GT-sal) | 0.970 | 0.971 | 0.094 | 0.955 | 0.955 | 0.162 |
| Ours-5 | $\mathbf{0 . 9 4 9}$ | $\mathbf{0 . 8 5 4}$ | $\mathbf{0 . 1 4 4}$ | $\mathbf{0 . 8 6 8}$ | $\mathbf{0 . 5 9 7}$ | $\mathbf{0 . 3 2 9}$ |

VRW23 dataset, which is attributed to the overall shorter video durations. After all, the longer you predict, the more possible mistakes you would make, which are easier spotted by the classifiers.

Psychophysical Experiment. We next take a psychophysical approach: invite human subjects to judge whether the viewed viewport sequences are real or not. We select 11 and 12 panoramic videos from the CVPR18 and VRW23 test sets, respectively. For each test video, we generate 7 viewport sequences by sampling along different scanpaths produced by the path-only model, Xu18 (CB-sal), Xu18 (GT-sal), TRACK (CB-sal), TRACK (GT-sal), the proposed method, and one human viewer (as the real instance). Fig. 8 shows the graphical user interface customized for this experiment. All viewport videos are shown in the actual resolution of $252 \times 448$, with a framerate of $30 \mathrm{fps} 8$ and in a randomized temporal order. The "Real" and "Fake" bottoms are utilized to collect the perceptual realism judgment for each video, both of which serve as the "Next" bottom for the next video playback. Each video can be replayed multiple times until the subject is confident with her/his rating, but we encourage her/him to make the judgment at the earliest convenience. We also allow the subject to go back to the previous video with the "Back" bottom in case s/he would like to change the rating for some reason, as a way of mitigating the serial dependence between adjacent videos [73|. For each viewport sequence, we gather human data from 10 subjects with normal and correct-to-normal visual acuity. They have general knowledge of image processing and computer vision, but do not know the detailed purpose of the study. We include a training session to familiarize them with the user interface and the moving patterns of real viewport sequences. Each subject is asked to give judgments to all viewport sequences.

The perceptual realism of each model is defined as the number of viewport sequences labeled as real divided by the total number of sequences corresponding to that model. As shown in Fig. 9. the perceptual realism of scanpaths by our model is very close to the ground-truth scanpaths, and is much better than scanpaths by the competing methods on both datasets. This is due primarily to the accurate probabilistic modeling of the uncertainty and diversity of scanpaths and the PID controller-based sampler that takes

8. We upconvert the framerate from the default $5 \mathrm{fps}$ to $30 \mathrm{fps}$ using spherical linear interpolation $|72|$.

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-13.jpg?height=694&width=876&top_left_y=149&top_left_x=1080)

Fig. 8. Graphical user interface for the psychophysical experiment.

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-13.jpg?height=564&width=770&top_left_y=976&top_left_x=1138)

Fig. 9. Perceptual realism results through a psychophysical experiment on the CVPR18 and VRW23 datasets.

into account Newton's laws of motion during sampling. It is also interesting to note that TRACK (CB-sal) ranks third in the psychophysical experiment, which is consistent with the results in Fig. 5 (d) and Fig. 6 (d). This indicates the TC metric family is more in line with human visual perception.

From a qualitative perspective, we find that the deterministic saliency-based methods are easier to swing between two objects when there are multiple salient objects in the scene. Meanwhile, Xu18 exhibits a tendency to remain fixated on one position. This phenomenon may be attributed to the original model design for short-term scanpath prediction. As delineated in [11], the past scanpath suffices to serve as the historical context for short-term prediction. Consequently, it is likely that when the initial viewpoints are situated on some objects, it is easy for Xu18 to get trapped in such bad "local optima." On the contrary, our method does not suffer from any of the above problems.

### 5.3.3 Cross-Dataset Generalization Results

To test the generalizability of CVPR18-trained and VRW23trained models, we conduct cross-dataset experiments

TABLE 6

Comparison results in terms of minOD and maxTC, and their sliced versions SminOD and SmaxTC on the MMSys18 dataset

| Model | CVPR18-Trained |  |  |  |  |  |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | minOD $\downarrow$ | SminOD-5 $\downarrow$ | maxTC $\uparrow$ | SmaxTC-5 $\uparrow$ | minOD $\downarrow$ | SminOD-5 $\downarrow$ | $\operatorname{maxTC} \uparrow$ | SmaxTC-5 $\uparrow$ |  |
| Path-Only | 0.441 | 0.179 | 0.795 | 0.914 | 0.577 | 0.267 | 0.791 |  |  |
| TRACK (CB-sal) | 0.578 | 0.258 | 0.773 | 0.967 | 0.617 | 0.299 | 0.790 |  |  |
| TRACK (GT-sal) | 0.493 | 0.212 | 0.714 | 0.949 | 0.595 | 0.283 | 0.729 |  |  |
| Ours-5 | $\mathbf{0 . 4 1 6}$ | $\mathbf{0 . 1 4 1}$ | $\mathbf{0 . 8 8 2}$ | $\mathbf{0 . 9 9 6}$ | $\mathbf{0 . 4 3 5}$ | $\mathbf{0 . 1 4 8}$ | 0.971 |  |  |
| Ours-20 | $\mathbf{0 . 3 2 2}$ | $\mathbf{0 . 0 9 3}$ | $\mathbf{0 . 9 1 9}$ | $\mathbf{0 . 9 9 8}$ | $\mathbf{0 . 3 4 4}$ | $\mathbf{0 . 0 9 8}$ | $\mathbf{0 . 8 8 7}$ |  |  |

TABLE 7

Comparison results in terms of minOD and maxTC, and their sliced versions SminOD and SmaxTC on the PAMI19 dataset

| Model | CVPR18-Trained |  |  |  |  |  |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | minOD $\downarrow$ | SminOD-5 $\downarrow$ | maxTC $\uparrow$ | SmaxTC-5 $\uparrow$ | minOD $\downarrow$ | SminOD-5 $\downarrow$ | $\operatorname{maxTC} \uparrow$ | SmaxTC-5 $\uparrow$ |  |
| Path-Only | $\mathbf{0 . 1 2 5}$ | $\mathbf{0 . 0 6 4}$ | 0.636 | 0.855 | $\mathbf{0 . 5 9 3}$ | $\mathbf{0 . 3 5 3}$ | $\mathbf{0 . 7 2 9}$ |  |  |
| TRACK (CB-sal) | 0.538 | 0.294 | 0.635 | 0.951 | 0.986 | 0.577 | 0.718 |  |  |
| TRACK (GT-sal) | $\mathbf{0 . 1 7 4}$ | $\mathbf{0 . 0 6 8}$ | 0.645 | 0.922 | 0.646 | 0.387 | 0.702 | 0.964 |  |
| Ours-5 | 0.584 | 0.408 | $\mathbf{0 . 8 0 1}$ | $\mathbf{0 . 9 9 4}$ | 0.824 | 0.499 | 0.624 |  |  |
| Ours-20 | 0.346 | 0.180 | $\mathbf{0 . 8 9 8}$ | $\mathbf{0 . 9 9 9}$ | $\mathbf{0 . 5 6 4}$ | $\mathbf{0 . 2 1 1}$ | $\mathbf{0 . 7 4 7}$ | $\mathbf{0 . 9 9 6}$ |  |

on two relatively smaller datasets - MMSys18 [65] and PAMI19 [42]. Tables 6 and 7 show the results, in which we omit Nguyen18 and Xu18 as they are inferior to the pathonly and TRACK models. Consistent with the results in the main experiments, our methods outperform the others on both datasets in terms of temporal correlation metrics (except for Ours-5 trained on VRW23 and tested on PAMI19). For the orthodromic distance metrics, our methods achieve the best results on MMSys18, but are worse than the pathonly method on PAMI19. Interestingly, the path-only model always performs better than TRACK. Moreover, our methods trained on CVPR18 have better performance than those trained on VRW23 when tested on PAMI19, while both perform similarly when tested on MMSys18. This implies that the scanpath distribution of PAMI19 is closer to that of CVPR18.

### 5.4 Ablation Experiments

We conduct a series of ablation experiments to justify the rationality of our model design. For experiments that need no scanpath sampling, we report the expected code length in Eq. (27). As for experiments that require scanpath sampling, we set the prediction horizon $S=5$, sample 20 scanpaths (i.e., $|\hat{\mathcal{S}}|=20$ ), and report the maxTC results.

Input Component. We first probe the contribution of the three input components in our model, i.e., the historical visual context, the historical path context, and the causal path context, by training three variants: 1 ) the model with only the historical visual context, 2) the model with the historical visual and path contexts, and 3) the full model with all three input components. We report the maxTC results in Table 8 (see the PID Controller columns). Our results show that adding the historical path context clearly increases the maximum temporal correlation, particularly on VRW23. Moreover, the causal path context also contributes substantially, validating its effectiveness as an autoregressive prior. Scanpath Representation. We next probe different scanpath representations: 1) spherical coordinates $(\phi, \theta), 2) 3 \mathrm{D}$ Euclidean coordinates $(x, y, z)$, and 3) relative $u v$ coordinates $(u, v)$. Table 9 reports the expected code length results, in which we find that our relative $u v$ representation performs the best, followed by the 3D Euclidean coordinates.

Quantization Step Size. We further study the effect of the quantization step size on the probabilistic modeling of our method. Specifically, we test four different quantization step sizes of $\{0.02,0.2,2,20\}$, which, respectively, correspond to the largest quantization errors of $\{0.01,0.1,1,10\}$. We report the maxTC results in Fig. 10, from which we find that a proper quantization step size is crucial to the final scanpath prediction performance. A very large quantization step size would induce a noticeable quantization error, which impairs the diversity modeling. Conversely, a very small quantization step size would hinder the training of smooth entropy models. This provides strong justification for the use of the discretized probability model (in Eq. 26) over its continuous counterpart (in Eq. (24)).

Sampler. We last compare our PID controller-based sampler to three counterparts: the naive random sampler, the max sampler, and the beam search sampler (with a beam width of 20). Table 8 shows the maxTC results. Our PID controllerbased sampler outperforms all three competing samplers by a large margin for the three model variants and on the two datasets. We also observe that the causal path context increases the performance of the random sampler and our PID controller-based sampler, but decreases the performance of the max and beam search samplers. This suggests that the causal path context is a double-edged sword: conditioning on an inaccurate causal path context would lead to degraded performance.

## 6 ConcLUSION AND DISCUSSION

We have described a new probabilistic approach to panoramic scanpath prediction from the perspective of lossy data compression. We explored a simple criterion-expected code length minimization-to train a discrete conditional probability model for quantized scanpaths. We also pre-

TABLE 8

Ablation analysis of different samplers for three model variants with different input components in terms of maxTC. H-Path and C-Path stand for the historical and causal path contexts, respectively

| Model | CVPR18 Dataset |  |  |  |  |  |  |  |  |  |  |  | VRW23 Dataset |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Random | Max | Beam Search | PID Controller | Random | Max | Beam Search | PID Controller |  |  |  |  |  |
| Visual | 0.007 | 0.124 | 0.159 | 0.551 | -0.001 | 0.115 | 0.163 | 0.515 |  |  |  |  |  |
| Visual + H-Path | 0.133 | 0.451 | 0.418 | 0.786 | 0.232 | 0.469 | 0.483 | 0.799 |  |  |  |  |  |
| Visual + H-Path + C-Path | 0.147 | 0.360 | 0.349 | 0.844 | 0.245 | 0.446 | 0.437 | 0.825 |  |  |  |  |  |

TABLE 9

Ablation analysis of different scanpath representation in terms of expected code length

| Representation | CVPR18 Dataset | VRW23 Dataset |
| :--- | :---: | :---: |
| Spherical $(\phi, \theta)$ | 17.99 | 18.67 |
| 3D Eculidean $(x, y, z)$ | 17.61 | 18.41 |
| Relative $(u, v)$ | 17.32 | 18.20 |

![](https://cdn.mathpix.com/cropped/2024_06_04_e3c2765764afcf14c33fg-15.jpg?height=513&width=705&top_left_y=901&top_left_x=249)

Fig. 10. Ablation analysis of different quantization step sizes in terms of $\operatorname{maxTC}$.

sented a PID controller-based sampler to generate realistic scanpaths from the learned probability model.

Our method is rooted in density estimation, the mother of all unsupervised learning problems. While the question of how to reliably assess the performance of unsupervised learning methods on finite data remains open in the general sense, we provide a quantitative measure, expected code length, in the context of scanpath prediction. We have carefully designed ablation experiments to point out the importance of the quantization step during probabilistic modeling. A similar idea that optimizes the coding rate reduction has been explored previously in image segmentation |74] and recently in representation learning [75].

We have advocated the adoption of best-case set-to-set distances to quantitatively compare the set of predicted scanpaths to the set of human scanpaths. Our set-to-set distances can be easily generalized by first finding an optimal bipartite matching between predicted and ground truth scanpaths (for example, using the Hungarian algorithm $[76 \mid$ ), and then comparing pairs of matched scanpaths. We have experimented with this variant of set-to-set distances, and arrive at similar conclusions in Sec. 5.3

One goal of scanpath prediction is to model and understand how humans explore different panoramic virtual scenes. Thus, we have emphasized on testing the perceptual realism of predicted scanpaths via machine discrimination and human verification. Although it is relatively easy for the trained classifiers to identify predicted scanpaths, our method performs favorably in "fooling" human subjects, with a matched perceptual realism level to human scanpaths. Thus, our method appears promising for a number of panoramic video processing applications, including panoramic video compression [58|, streaming [43], and quality assessment [66].

Finally, we have introduced a relative $u v$ representation of scanpaths in the viewport domain. This scanpath representation is well aligned with the viewport sequence, and simplifies the computational modeling of panoramic videos, and transforms the panoramic scanpath prediction problem to a planar one. We believe our relative $u v$ representation has great potential in a broader $360^{\circ}$ computer vision tasks, including panoramic video semantic segmentation, object detection, and object tracking.

## REFERENCES

[1] D. Noton and L. Stark, "Scanpaths in saccadic eye movements while viewing and recognizing patterns," Vision Research, vol. 11, no. 9, pp. 929-942, 1971.

[2] , "Scanpaths in eye movements during pattern perception," Science, vol. 171, no. 3968, pp. 308-311, 1971.

[3] F. Perazzi, A. Sorkine-Hornung, H. Zimmer, P. Kaufmann, O. Wang, S. Watson, and M. Gross, "Panoramic video from unstructured camera arrays," Computer Graphics Forum, vol. 34, no. 2, pp. 57-68, 2015.

[4] G. Zoric, L. Barkhuus, A. Engström, and E. Önnevall, "Panoramic video: Design challenges and implications for content interaction," in European Conference on Interactive TV and Video, 2013, pp. 153162 .

[5] K.-T. Ng, S.-C. Chan, and H.-Y. Shum, "Data compression and transmission aspects of panoramic videos," IEEE Transactions on Circuits and Systems for Video Technology, vol. 15, no. 1, pp. 82-95, 2005.

[6] Y. Cai, X. Li, Y. Wang, and R. Wang, "An overview of panoramic video projection schemes in the IEEE 1857.9 standard for immersive visual content coding," IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 9, pp. 6400-6413, 2022.

[7] M. Xu, C. Li, Y. Liu, X. Deng, and J. Lu, "A subjective visual quality assessment method of panoramic videos," in IEEE International Conference on Multimedia and Expo, 2017, pp. 517-522.

[8] V. Sitzmann, A. Serrano, A. Pavel, M. Agrawala, D. Gutierrez, B. Masia, and G. Wetzstein, "Saliency in VR: How do people explore virtual environments?" IEEE Transactions on Visualization and Computer Graphics, vol. 24, no. 4, pp. 1633-1642, 2018.

[9] T. Rhee, L. Petikam, B. Allen, and A. Chalmers, "MR360: Mixed reality rendering for $360^{\circ}$ panoramic videos," IEEE Transactions on Visualization and Computer Graphics, vol. 23, no. 4, pp. 1379-1388, 2017.

[10] W.-T. Lee, H.-I. Chen, M.-S. Chen, I.-C. Shen, and B.-Y. Chen, "High-resolution 360 video foveated stitching for real-time VR," Computer Graphics Forum, vol. 36, no. 7, pp. 115-123, 2017.

[11] M. F. R. Rondón, L. Sassatelli, R. Aparicio-Pardo, and F. Precioso, "TRACK: A new method from a re-examination of deep architectures for head motion prediction in $360^{\circ}$ videos," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 9, pp. 5681$5699,2022$.

[12] C.-L. Fan, J. Lee, W.-C. Lo, C.-Y. Huang, K.-T. Chen, and C.H. Hsu, "Fixation prediction for $360^{\circ}$ video streaming in headmounted virtual reality," in Workshop on Network and Operating Systems Support for Digital Audio and Video, 2017, pp. 67-72.

[13] Y. Xu, Y. Dong, J. Wu, Z. Sun, Z. Shi, J. Yu, and S. Gao, "Gaze prediction in dynamic $360^{\circ}$ immersive videos," in IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 5333-5342.

[14] A. Nguyen, Z. Yan, and K. Nahrstedt, "Your attention is unique: Detecting 360-degree video saliency in head-mounted display for head movement prediction," in ACM International Conference on Multimedia, 2018, pp. 1190-1198.

[15] Y. Xu, Z. Zhang, and S. Gao, "Spherical DNNs and their applications in $360^{\circ}$ images and videos," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7235-7252, 2022.

[16] Y. Li, Y. Xu, S. Xie, L. Ma, and J. Sun, "Two-layer FOV prediction model for viewport dependent streaming of 360-degree videos," in International Conference on Communications and Networking in China, 2018, pp. 501-509.

[17] W. Sun, Z. Chen, and F. Wu, "Visual scanpath prediction using IOR-ROI recurrent mixture density network," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 6, pp. 21012118, 2021.

[18] M. Assens, X. Giro-i Nieto, K. McGuinness, and N. E. O'Connor, "PathGAN: Visual scanpath prediction with generative adversarial networks," in European Conference on Computer Vision Workshops, 2018, pp. 406-422.

[19] D. Martin, A. Serrano, A. W. Bergman, G. Wetzstein, and B. Masia, "ScanGAN360: A generative model of realistic scanpaths for $360^{\circ}$ images," IEEE Transactions on Visualization and Computer Graphics, vol. 28, no. 5, pp. 2003-2013, 2022.

[20] T. Cover and J. Thomas, Elements of Information Theory. Wiley, 2012.

[21] T. Baltrušaitis, C. Ahuja, and L.-P. Morency, "Multimodal machine learning: A survey and taxonomy," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423-443, 2018.

[22] R. Bellman, Adaptive Control Processes: A Guided Tour. Princeton University Press, 2015.

[23] T. Ngo and B. Manjunath, "Saccade gaze prediction using a recurrent neural network," in IEEE International Conference on Image Processing, 2017, pp. 3435-3439.

[24] C. Wloka, I. Kotseruba, and J. K. Tsotsos, "Active fixation control to predict saccade sequences," in IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 3184-3193.

[25] C. Xia, J. Han, F. Qi, and G. Shi, "Predicting human saccadic scanpaths based on iterative representation learning," IEEE Transactions on Image Processing, vol. 28, no. 7, pp. 3502-3515, 2019.

[26] R. Klein, "Inhibitory tagging system facilitates visual search," Nature, vol. 334, no. 6181, pp. 430-431, 1988.

[27] R. A. J. de Belen, T. Bednarz, and A. Sowmya, "ScanpathNet: A recurrent mixture density network for scanpath prediction," in IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2022, pp. 5010-5020.

[28] J. M. Wolfe, "Guided search 6.0: An updated model of visual search," Psychonomic Bulletin E Review, vol. 28, no. 4, pp. 1060$1092,2021$.

[29] N. D. B. Bruce and J. K. Tsotsos, "Saliency, attention, and visual search: An information theoretic approach," Journal of Vision, vol. 9, no. 3, pp. 1-24, 2009

[30] X. Huang, C. Shen, X. Boix, and Q. Zhao, "SALICON: Reducing the semantic gap in saliency prediction by adapting deep neural networks," in IEEE International Conference on Computer Vision, 2015, pp. 262-270.

[31] K. He, G. Gkioxari, P. Dollar, and R. Girshick, "Mask R-CNN," in IEEE International Conference on Computer Vision, 2017, pp. 29612969.

[32] M. Assens, X. Giro-i Nieto, K. McGuinness, and N. E. O'Connor, "SaltiNet: Scan-path prediction on 360 degree images using saliency volumes," in IEEE International Conference on Computer Vision Workshops, 2017, pp. 2331-2338.
[33] Y. Zhu, G. Zhai, and X. Min, "The prediction of head and eye movement for 360 degree images," Signal Processing: Image Communication, vol. 69, pp. 15-25, 2018.

[34] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan, "Object detection with discriminatively trained part-based models," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 9, pp. 1627-1645, 2010.

[35] M. A. Kerkouri, M. Tliba, A. Chetouani, and M. Sayeh, "SalyPath360: Saliency and scanpath prediction framework for omnidirectional images," in Electronic Imaging Symposium, 2022, pp. 168-1 $-168-7$.

[36] Y. Dahou, M. Tliba, K. McGuinness, and N. O'Connor, "ATSal: An attention based architecture for saliency prediction in $360^{\circ}$ videos," in International Conference on Pattern Recognition Workshops, 2020, pp. 305-320.

[37] M. Cornia, L. Baraldi, G. Serra, and R. Cucchiara, "A deep multilevel network for saliency prediction," in International Conference on Pattern Recognition, 2016, pp. 3488-3493.

[38] B. D. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision," in International Joint Conference on Artificial Intelligence, 1981, pp. 674-679.

[39] A. De Abreu, C. Ozcinar, and A. Smolic, "Look around you: Saliency maps for omnidirectional images in VR applications," in International Conference on Quality of Multimedia Experience, 2017, pp. 1-6.

[40] J. Pan, E. Sayrol, X. Giro-i Nieto, K. McGuinness, and N. E. O'Connor, "Shallow and deep convolutional networks for saliency prediction," in IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 598-606.

[41] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, and T. Brox, "FlowNet 2.0: Evolution of optical flow estimation with deep networks," in IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2462-2470.

[42] M. Xu, Y. Song, J. Wang, M. Qiao, L. Huo, and Z. Wang, "Predicting head movement in panoramic video: A deep reinforcement learning approach," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 11, pp. 2693-2708, 2019.

[43] C. Li, W. Zhang, Y. Liu, and Y. Wang, "Very long term field of view prediction for 360-degree video streaming," in IEEE Conference on Multimedia Information Processing and Retrieval, 2019, pp. 297-302.

[44] M. Cornia, L. Baraldi, G. Serra, and R. Cucchiara, "Predicting human eye fixations via an LSTM-based saliency attentive model," IEEE Transactions on Image Processing, vol. 27, no. 10, pp. 5142-5154, 2018.

[45] F.-Y. Chao, C. Ozcinar, and A. Smolic, "Transformer-based longterm viewport prediction in $360^{\circ}$ video: Scanpath is all you need," in IEEE International Workshop on Multimedia Signal Processing, 2021, pp. 1-6.

[46] M. Müller, Information Retrieval for Music and Motion. Springer Berlin Heidelberg, 2007.

[47] T. S. Cohen, M. Geiger, J. Köhler, and M. Welling, "Spherical CNNs," in International Conference on Learning Representations, 2018.

[48] C. Esteves, C. Allen-Blanchette, A. Makadia, and K. Daniilidis, "Learning SO(3) equivariant representations with spherical CNNs," in European Conference on Computer Vision, 2018, pp. 52-68.

[49] C. Jiang, J. Huang, K. Kashinath, Prabhat, P. Marcus, and M. Niessner, "Spherical CNNs on unstructured grids," in International Conference on Learning Representations, 2019

[50] C. Wu, R. Zhang, Z. Wang, and L. Sun, "A spherical convolution approach for learning long term viewport prediction in $360 \mathrm{im}-$ mersive video," in AAAI Conference on Artificial Intelligence, 2020, pp. 14003-14040.

[51] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems, 2017.

[52] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pretraining of deep bidirectional transformers for language understanding," in Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019.

[53] E. P. Simoncelli, "Distributed representation and analysis of visual motion," Ph.D. dissertation, Massachusetts Institute of Technology, 1993.

[54] D. Kingma and J. Ba, "Adam: A method for stochastic optimization," in International Conference for Learning Representations, 2015.

[55] C. M. Bishop and N. M. Nasrabadi, Pattern Recognition and Machine Learning. Springer, 2006.

[56] J. Ballé, V. Laparra, and E. P. Simoncelli, "End-to-end optimized image compression," in International Conference on Learning Representations, 2016.

[57] M. Li, K. Ma, J. You, D. Zhang, and W. Zuo, "Efficient and effective context-based convolutional entropy modeling for image compression," IEEE Transactions on Image Processing, vol. 29, pp. 5900-5911, 2020.

[58] M. Li, K. Ma, J. Li, and D. Zhang, "Pseudocylindrical convolutions for learned omnidirectional image compression," arXiv preprint arXiv:2112.13227, 2021.

[59] X. Sui, K. Ma, Y. Yao, and Y. Fang, "Perceptual quality assessment of omnidirectional images as moving camera videos," IEEE Transactions on Visualization and Computer Graphics, vol. 28, no. 8, pp. 3022-3034, 2022.

[60] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770-778.

[61] L. Devroye, Handbooks in Operations Research and Management Science. Elsevier, 2006.

[62] K. H. Ang, G. Chong, and Y. Li, "PID control system analysis, design, and technology," IEEE Transactions on Control Systems Technology, vol. 13, no. 4, pp. 559-576, 2005.

[63] Y. Bao, H. Wu, T. Zhang, A. A. Ramli, and X. Liu, "Shooting a moving target: Motion-prediction-based transmission for 360degree videos," in IEEE International Conference on Big Data, 2016, pp. 1161-1170.

[64] C. Wu, Z. Tan, Z. Wang, and S. Yang, "A dataset for exploring user behaviors in VR spherical video streaming," in ACM Multimedia Systems Conference, 2017, pp. 193-198.

[65] E. J. David, J. Gutiérrez, A. Coutrot, M. P. Da Silva, and P. L. Callet, "A dataset of head and eye movements for $360^{\circ}$ videos," in ACM Multimedia Systems Conference, 2018, pp. 432-437.

[66] Y. Fang, Y. Yao, X. Sui, and K. Ma, "Subjective quality assessment of user-generated $360^{\circ}$ videos," in IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops, 2023, pp. $74-83$.

[67] J. G. Ziegler and N. B. Nichols, "Optimum settings for automatic controllers," Transactions of the American Society of Mechanical Engineers, vol. 64, no. 8, pp. 759-765, 1942.

[68] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 1026-1034.

[69] T. Sauer, J. A. Yorke, and M. Casdagli, "Embedology," Journal of Statistical Physics, vol. 65, no. 3, pp. 579-616, 1991.

[70] W. Wang, C. Chen, Y. Wang, T. Jiang, F. Fang, and Y. Yao, "Simulating human saccadic scanpaths on natural images," in IEEE Conference on Computer Vision and Pattern Recognition, 2011, pp. $441-448$.

[71] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial networks," Communications of the ACM, vol. 63, no. 11, pp. 139$144,2020$.

[72] K. Shoemake, "Animating rotation with quaternion curves," in Proceedings of the 12th annual conference on Computer graphics and interactive techniques, 1985, pp. 245-254.

[73] J. Fischer and D. Whitney, "Serial dependence in visual perception," Nature Neuroscience, vol. 17, no. 5, pp. 738-743, 2014.

[74] Y. Ma, H. Derksen, W. Hong, and J. Wright, "Segmentation of multivariate mixed data via lossy data coding and compression," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 29, no. 9, pp. 1546-1562, 2007.

[75] X. Dai, S. Tong, M. Li, Z. Wu, K. H. R. Chan, P. Zhai, Y. Yu, M. Psenka, X. Yuan, and H. Y. Shum, "CTRL: Closed-loop transcription to an LDR via minimaxing rate reduction," Entropy, vol. 24, no. 4 , p. 456, 2022.

[76] H. W. Kuhn, "The Hungarian method for the assignment problem," Naval Research Logistics Quarterly, vol. 2, no. 1-2, pp. 83-97, 1955 .


[^0]:    3. https://en.wikipedia.org/wiki/Atan2
</end of paper 3>


<paper 4>
# Impact of Design Decisions in Scanpath Modeling 

PARVIN EMAMI, University of Luxembourg, Luxembourg<br>YUE JIANG and ZIXIN GUO, Aalto University, Finland<br>LUIS A. LEIVA, University of Luxembourg, Luxembourg


#### Abstract

Modeling visual saliency in graphical user interfaces (GUIs) allows to understand how people perceive GUI designs and what elements attract their attention. One aspect that is often overlooked is the fact that computational models depend on a series of design parameters that are not straightforward to decide. We systematically analyze how different design parameters affect scanpath evaluation metrics using a state-ofthe-art computational model (DeepGaze++). We particularly focus on three design parameters: input image size, inhibition-of-return decay, and masking radius. We show that even small variations of these design parameters have a noticeable impact on standard evaluation metrics such as DTW or Eyenalysis. These effects also occur in other scanpath models, such as UMSS and ScanGAN, and in other datasets such as MASSVIS. Taken together, our results put forward the impact of design decisions for predicting users' viewing behavior on GUIs.


CCS Concepts: $\bullet$ Human-centered computing $\rightarrow$ Empirical studies in ubiquitous and mobile computing; $\cdot$ Computing methodologies $\rightarrow$ Computer vision.

Additional Key Words and Phrases: Visual Saliency; Interaction Design; Computer Vision; Deep Learning; Eye Tracking

ACM Reference Format:

Parvin Emami, Yue Jiang, Zixin Guo, and Luis A. Leiva. 2024. Impact of Design Decisions in Scanpath Modeling. Proc. ACM Hum.-Comput. Interact. 8, ETRA, Article 228 (May 2024), 16 pages. https://doi.org/10.1145/3655602

## 1 INTRODUCTION

Understanding how user attention is allocated in graphical user interfaces (GUIs) is an important research challenge, considering that many different GUI elements (e.g. buttons, headers, cards, etc.) may stand out and engage users effectively [33]. By modeling eye movement patterns of visual saliency, we can gain invaluable insights into how users perceive and interact with GUIs, without having to recruit users early in the GUI design process. When presented with a GUI screenshot, a saliency model can predict how users would spend their attention, typically over short periods of time during free-viewing scenarios (bottom-up saliency) or over longer periods in task-based scenarios (top-down saliency). We can use these predictions to quantify the impact of a visual change in the GUI (e.g. after rescaling an element or changing its position), optimize some design components so that it can grab less or more user attention, or understand whether users notice some element after some time of exposure.

Saliency models can predict either (static) saliency maps [41] or (dynamic) scanpaths [19]. Most research has focused on predicting saliency maps [ $2,9,15,17,20]$, overlooking key temporal aspects[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-01.jpg?height=119&width=268&top_left_y=2040&top_left_x=149)

This work is licensed under a Creative Commons Attribution International 4.0 License.

© 2024 Copyright held by the owner/author(s).

ACM 2573-0142/2024/5-ART228

https://doi.org/10.1145/3655602
like fixation timing and duration. Saliency maps show aggregated fixation locations, i.e., areas that users will pay attention to, where the eye remains relatively static. In contrast, scanpaths contain sequential information on individual fixations and sometimes also saccades, i.e., eye movements between those points, thus retaining information about fixation order and their temporal dynamics In other words, scanpaths comprise rich data from which second-order representations such as saliency maps can be computed. Critically, scanpaths can inform about the users' visual flows, through which one can better assess how attention deploys over time. This is vital for understanding how individual users, instead of a group of users, would perceive the GUI and for design adjustments that encourage viewing GUI elements in a desired order for different people [12]. For these reasons, in this paper we focus on scanpath models of visual saliency.

One problem that is often overlooked is that many computational models of visual saliency rely on a set of design parameters that must be defined beforehand. Some of them can be inferred from the collected data, such as deciding the number of fixations to predict on average. However other parameters must be established by the researcher, such as deciding the resolution of the GUI screenshots for model input. These design parameters cannot be learned e.g. through backpropagation, so researchers have to rely on their own expertise, trial and error, previous work, or best practices. To the best of our knowledge, their impact on downstream performance has not been systematically analyzed. We believe this kind of analysis is very much needed because any evaluation depends on the quality of the model predictions, so it may be the case that small variations on some parameters produce different performance results. In this context, we pose the following research question: To what extent do saliency model predictions depend on the choices made in their design parameters?

We use DeepGaze++ [17] as a reference model to investigate the potential impact that different design parameters may have in scanpath prediction. DeepGaze++ is a state-of-the-art scanpath model for visual saliency prediction that has shown promising results. However, like other models, it relies on "hardcoded" design parameters such as the aforementioned input screenshot size or, more interestingly, the masking radius used for inhibition of return (IOR) mechanisms. IOR, which refers to the phenomenon where attention is less likely to return to previously attended locations, plays a crucial role in visual perception. If the masking radius is not appropriately calibrated, there is a possibility of omitting potentially salient areas within a GUI [18]. Also, as explained later, DeepGaze++ relies on a sub-optimal IOR weighting mechanism limited to 12 fixation points, so we propose a new weighting scheme to overcome this limitation as an aside research contribution.

While using hardcoded design parameters may be the most straightforward approach, determining their optimal values for each type of GUI remains an open question. For example, if we were to have a masking radius equivalent to the whole image size, the model could only predict one fixation, as the whole GUI would have been masked out (i.e. no other GUI parts could be fixated on because, by definition, there is nothing left to fixate on if everything is masked out). In this paper, we focus on three key design parameters common to every scanpath model:

Input image size, which determines the granularity of the predicted fixations (higher resolution gives more room to fixate on more GUI parts).

IOR decay, which implements the importance that previous fixations have in successive fixation predictions

Masking radius, which allows to prevent that previously fixated GUI parts are fixated on again.

We study the impact of these parameters on different GUI types, including web-based, mobile, and desktop applications. Then, we show that the optimized parameters we discovered for DeepGaze++ [17]
help improve the performance of other scanpath models and also generalize to other datasets. In sum, this paper makes the following contributions:

(1) A comparative study on model design parameters on scanpath prediction performance, across different types of GUIs.

(2) An optimized set of design parameters for scanpath models, evaluated across multiple models and datasets.

(3) A new IOR decay parameter, designed to work with an arbitrary number of fixation points.

## 2 RELATED WORK

Visual saliency prediction in GUIs has witnessed substantial growth in recent years [39, 40, 42], driven by the increasing demand for accurate models that can anticipate where human attention is likely to be directed within digital displays. As previously hinted, existing research has predominantly focused on the development and refinement of visual saliency prediction models, often overlooking the crucial impact that different design parameters may have. For example, de Belen et al. [11] proposed ScanpathNet, a deep learning model inspired by neuroscience, and noted that model performance was influenced by the choice of the number of Gaussian components. This highlights the significant impact that the choice of design parameters may have.

Previous research considered different image sizes to predict visual saliency, informed by the datasets they used for training the models $[3,8,19]$. It became apparent that higher image resolutions are not desired, as drifting errors tend to increase [24], but no systematic examination was provided in this regard. In addition, Parmar et al. [29] demonstrated that generative models are particularly sensible to image resizing artifacts such as quantization and compression. Therefore, we decided to examine the impact of image resizing in visual saliency prediction.

Generating fixation sequences accurately, while promoting a coherent and natural order, remains as the main challenge in scanpath modeling. Itti et al. [16] introduced the Inhibition of Return (IOR) mechanism as a way to ensure that predicted fixations do not bounce back and forth around previously visited areas. This was later exploited in scanpath modeling [3, 32], although all scanpath models that incorporate IOR employ a fixed masking radius [7, 10]. Furthermore, there is no discussion about how this radius affects model predictions. As mentioned in the previous section, when this radius is too large, the model's ability to predict multiple fixation points will be severely limited. Therefore, we decided to examine the impact of masking radius and IOR decay in visual saliency prediction.

Several methods, including deep neural networks and first-principle models, have been proposed to predict scanpaths in natural images [9], videos [23], and, more recently, GUIs [17]. Ngo and Manjunath [28] developed a recurrent neural network to predict sequences of saccadic eye movements and Wloka et al. [36] predicted fixation sequences by relying on a "history map" of previously observed fixations. These works were evaluated on small datasets and using a limited set of metrics, therefore it remains unclear whether these models can compare favorably in GUIs.

Later on, Xia et al. [37] introduced an iterative representation learning framework to predict saccadic movements. More recently, Jiang et al. [17] developed DeepGaze++ based on DeepGaze III [20]. DeepGaze III takes both an input image and the positions of the previous fixation points to predict a probabilistic density map for the next fixation point. It frequently tends to predict clusters of nearby points, potentially leading to stagnation within those clusters. To address this problem, DeepGaze++ recurrently chooses the position with the highest probability from the density map, concurrently implementing a custom IOR decay to suppress the selected position in the saliency map. (As explained in the next section, this decay only works for a relatively small number of
fixation points, therefore we propose a new IOR decay to address this limitation.) Nevertheless, DeepGaze++ is a state-of-the-art scanpath model so we use it in our investigation.

## 3 METHODOLOGY

The goal of scanpath prediction is to generate a plausible sequence of fixations, where fixations refer to distinct focal points during visual exploration. As previously mentioned, our study leverages the advanced capabilities of DeepGaze++ to answer our research question.

### 3.1 Dataset

In our study, we use the UEyes dataset [17], a collection of eye-tracking data over 1,980 screenshots covering four GUI types (495 screenshots per type): posters, desktop UIs, mobile UIs, and webpages. This dataset was collected from 66 participants ( 23 male, 43 female) aged 27.25 years (SD=7.26) via a high-fidelity in-lab eye tracker Gazepoint GP3. Participants had normal vision (43) or wore either glasses (18) or contact lenses (5). No participant was colorblind. Eye-tracking data were recorded after participant-specific calibration, a step that accounts for variables such as eye-display distance and visual angle, to ensure accurate recording of eye data. Participants were given 7 seconds to freely view each GUI screenshot in a $1920 \times 1200$ px monitor. For our study, we considered the same data partitions as in the UEyes dataset: 1,872 screenshots for training and 108 for testing. Our experiments are performed over the training partition of the UEyes dataset. The testing partition simulates unseen data, therefore it is used for final model evaluation.

### 3.2 Design parameters

In the following, we describe the parameters we have considered for our study. Note that they are all common to every scanpath model, they cannot be inferred from data, and they cannot be learned automatically. In our experiments, whenever we modify the values of each parameter, everything else remains constant. This way, it is easy to understand the concrete influence of each parameter in model performance.

3.2.1 Image size. Each GUI type has a preferred size (e.g. desktop applications are usually designed for FullHD monitors) or proportion (e.g. mobile apps have around 9:16 aspect ratio). When GUI images are resized (downsampled), to speed up computations, the models may perform differently. Therefore, it is unclear which image resolution should be used as model input. To shed light in this regard, we tested different input sizes and aspect ratios.

3.2.2 IOR decay. Initially introduced by Posner et al. [30], IOR is a neural mechanism that suppresses visual processing within recently attended locations. In the context of scanpath modeling, DeepGaze ++ uses an IOR decay of $1-0.1(n-i-1)$, for the $i$-th fixation point when predicting $n$ fixation points, to prevent that older fixation points are likely to be revisited. As can be noticed, this is limited to a maximum number of 12 fixation points, after which the decay values may become negative; e.g., for $n=13$ the first fixation point $i=1$ gets an IOR of -0.1 . Consequently, we have developed a new IOR decay designed to accommodate any number of fixation points. We propose $\gamma^{(n-i-1)}$, in which $\gamma$ is a design parameter, between 0 and 1 , that we also analyze systematically.

3.2.3 Masking radius. To implement any IOR mechanism, we need to mask some areas around the previous fixation points. However, determining the optimal size of the masked areas is unclear. Therefore, we consider the masking radius as a third design parameter and, in line with the previous discussions, examine how various masking radii impact the scanpath prediction results.

### 3.3 Evaluation metrics

We employ a set of four metrics that, together, provide a holistic assessment about the predictive performance of scanpath models [1, 13, 26]: Dynamic Time Warping (DTW), Eyenalysis, Determinism, and Laminarity. These metrics are well-established in the research literature [14]. While DTW measures the location and sequence of fixations in temporal order, Eyenalysis measures only locations and Determinism measures only the order of fixation points. Conversely, Laminarity is a measure of repeated fixations on a particular region, without considering their location or order. Table 1 provides an overview of these metrics.

| Metric | Location | Order |
| :--- | :--- | :--- |
| DTW | Yes | Yes |
| Determinism | No | Yes |
| Eyeanalysis | Yes | No |
| Laminarity | No | No |

Table 1. Overview of the chosen scanpath evaluation metrics [14].

3.3.1 Dynamic Time Warping (DTW). First introduced by Berndt and Clifford [4], DTW is a method for comparing time series with varying lengths. It involves creating a distance matrix between two sequences and finding the optimal path that respects boundary, continuity, and monotonicity conditions. The optimal solution is the minimum path from the starting point to the endpoint of the matrix. DTW identifies such an optimal match between two scanpaths in an iteratively manner, ensuring the inclusion of critical features [27, 34].

3.3.2 Eyenalysis. This is a technique that involves double mapping of fixations between two scanpaths, aiming to reduce positional variability [26]. Like in DTW, this approach may result in multiple points from one scanpath being assigned to a single point in the other. Eyeanalysis performs dual mapping by finding spatially closest fixation points between two scanpaths, measuring average distances for these corresponding pairs.

3.3.3 Determinism. This metric gauges diagonal alignments among cross-recurrent points, representing shared fixation trajectories [14]. With a minimum line length of $L=2$ for diagonal elements, Determinism measures the congruence of fixation sequences. Computed as the percentage of recurrent fixation points in sub-scanpaths, Determinism considers pairs of distinct fixation points from two scanpaths, enhancing the original (unweighted) Determinism metric for subscanpath evaluation.

3.3.4 Laminarity. It measures the percentage of fixation points on sub-scanpaths in which all the pairs of corresponding fixation points are recurrences but all such recurrent fixation point pairs contain the same fixation point from one of the scanpaths [1, 14]. In sum, Laminarity indicates the tendency of scanpath fixations to cluster on one or a few specific locations.

## 4 EXPERIMENTS

In the following, we report the experiments aimed at finding the optimal set of design parameters. For the sake of conciseness, we consider DTW for determining the best result for each design parameter, as this metric accounts for both location and order of fixations (Table 1).
![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-06.jpg?height=262&width=1396&top_left_y=289&top_left_x=171)

Fig. 1. Impact of resizing to square or non-square image on different GUI types. The height is always fixed to $225 \mathrm{px}$. We consider widths of 128,225 , and $512 \mathrm{px}$. The best results are observed for widths of $225 \mathrm{px}$ (resulting in a square aspect ratio).
![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-06.jpg?height=258&width=1408&top_left_y=786&top_left_x=172)

Fig. 2. Impact of resizing to different square image sizes on different GUI types. We consider sizes of 128, 225, and $512 \mathrm{px}$. The best results are usually observed for the $128 \mathrm{px}$ cases.

### 4.1 Sensitivity to input image size

We analyzed the impact of resizing under different aspect ratios (square and non-square images). In the first experiment (Figure 1), the height of the resized images remained constant at $225 \mathrm{px}$, as suggested in previous studies [17], while we modified their width. The other width values were chosen as the closest powers of two around this baseline value of 225 , for convenience.

The results, presented in Figure 1, indicate that resizing any input image to a square aspect ratio consistently yields superior performance across all GUI types. An intriguing observation is that mobile GUIs are particularly sensible to this parameter as compared to other GUI types. We attribute this effect to the fact that mobile GUIs, despite having the largest aspect ratio, make heavy use of icons and usually icons have a square aspect ratio.

In the second experiment (Figure 2), we resized images down to various dimensions, while ensuring a square aspect ratio, as per the results of our previous experiment. The results in this case indicate that resizing images to smaller dimensions has a positive impact on the prediction of both scanpaths and fixation points in mobile UIs. However, the opposite holds true for desktop UIs, as they typically have smaller elements as compared to mobile UIs.

### 4.2 Sensitivity to IOR decay

In this experiment, we varied the $\gamma$ parameter of our proposed IOR decay to assess its impact on scanpath prediction. As a reminder, a larger $\gamma$ indicates a higher probability of revisiting previously observed fixation points. Figure 3 illustrates the findings of this experiment, indicating that smaller $\gamma$ values lead to improved scanpath prediction performance. This suggests that when the likelihood of revisiting a previously observed fixation point is low, the model performs better in predicting subsequent fixation points. Conversely, when the likelihood of revisiting a fixation point is high, the model excels in predicting individual fixation points.
![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-07.jpg?height=250&width=1390&top_left_y=294&top_left_x=174)

Fig. 3. Impact of different $\gamma$ values on different GUI types. Lower $\gamma$ means a high probability of revisiting fixation points. The best results are observed when $\gamma=0.1$.

### 4.3 Sensitivity to masking radius

In this experiment, we examined how altering the masking radius impacts scanpath prediction performance. The results are provided in Figure 4. We observed a negative correlation between the masking radius and the quality of the scanpath predictions, indicating that, as the radius increases, scanpath prediction quality decreases. However, we observed a sweet spot when the radius is set between 0.1 and 0.2 , as better results are obtained according to the three non-DTW metrics.
![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-07.jpg?height=250&width=1388&top_left_y=1031&top_left_x=172)

Fig. 4. Impact of different masking radius on different GUI types. Masking radii are relative to the input image size (e.g. 0.2 means $20 \%$ of the size). The best results are observed when the radius is set to 0.05 .

### 4.4 Putting it all together

With the optimal parameters in place, we conducted an additional experiment on the test partition of UEyes to understand the impact of an improved scanpath model. Figure 5 illustrates the results. The "DeepGaze++" cases represent the baseline model implementation [17]. The "Baseline IOR" cases represent DeepGaze++ using the original IOR decay and the optimal parameters derived from our experiments, whereas the "Improved IOR" cases represent DeepGaze++ with our proposed IOR decay and the optimal parameters derived from our experiments. The results highlight that adopting the new IOR decay addresses the challenge of the limited number of fixation points and contributes to enhanced prediction performance as compared with the baseline DeepGaze++ model, although the baseline IOR with optimal parameters is comparable in predicting fixation points.

The results indicate significant improvements when using these optimal parameters, underscoring their substantial impact on prediction performance. Figure 6 provides additional evidence by showing the ratios of visited-revisited elements for three types of GUI elements (image, text, face) following previous work [17, 22].

Table 2 shows that, by setting all the optimized parameters, the results of DeepGaze++ improve in all the metrics except Eyenalysis. The table presents the results of DeepGaze++ with baseline parameters, as described in [17], and with the set of optimized parameters. According to the two-sample paired $t$-test, differences are statistically significant for all metrics except Eyenalysis: DTW: $t(107)=5.36, p<.0001, d=0.367$; Eyenalysis: $t(107)=5.36, p=.4503$ (n.s.), $d=0.074$;
![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-08.jpg?height=254&width=1392&top_left_y=293&top_left_x=172)

Fig. 5. Impact of different IOR mechanisms, using optimal parameters, on different GUI types. "Baseline IOR" uses DeepGaze++ with the original IOR decay and the optimal parameters. "Improved IOR" uses DeepGaze++ with our proposed IOR decay and the optimal parameters. The best results are usually observed with the baseline IOR.

| DeepGaze ++ | DTW $\downarrow$ | Eyenalysis $\downarrow$ | Determinism $\uparrow$ | Laminarity $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: |
| Baseline | $5.118 \pm 0.482$ | $\mathbf{0 . 0 4 0} \pm \mathbf{0 . 0 0 4}$ | $1.101 \pm 0.536$ | $16.908 \pm 5.900$ |
| Improved | $4.669 \pm \mathbf{0 . 6 6 7}$ | $0.044 \pm 0.010$ | $2.529 \pm 2.059$ | $24.557 \pm 4.705$ |

Table 2. Evaluation of baseline (original) and improved DeepGaze++ model (using the optimized parameters), showing Mean $\pm$ SD results for each metric. Arrows denote the direction of the importance; e.g., $\downarrow$ means "lower is better." Each column's best result is highlighted in boldface.

Determinism: $t(107)=3.60, p<.001, d=0.432$; Laminarity: $t(107)=8.98, p<.0001, d=0.580$. Effect sizes (Cohen's $d$ ) suggest a moderate practical importance of the results [21].

### 4.5 Analysis of visited and revisited patterns

In line with previous research that quantified the impact of scanpath models in GUI elements [17], we categorized the GUI elements in UEyes into three categories (image, text, and face) using an enhanced version of the UIED model [38], which is designed to detect images and text on GUIs. We then analyzed the number of elements in each category that were initially visited and subsequently revisited. An element is considered revisited if the element gets a fixation again after at least three fixations on other elements. The findings are presented in Figure 6. We observed that text elements have a higher fixation probability than images in our improved model, which is better aligned with the ground-truth cases. The improved model is also more aligned with the ground-truth cases in terms revisited fixations. For visited fixations, no differences between models were observed.

### 4.6 Example gallery

Figure 7 illustrates our qualitative comparison of different scanpath models across various GUI types. The baseline DeepGaze++ model can predict fixation points but the resulting scanpaths are not very realistic. The improved DeepGaze++ model is able to predict realistic trajectories, with more accurate fixation points overall. It is worth noting that both models tend to have a center bias and tend to generate clusters of fixation points. The scanpaths shown in Figure 7 follow a color gradient from red (beginning of trajectory) to blue (end of trajectory).

### 4.7 Comparison against other scanpath models

To show the generalizability of our findings, we further conducted evaluations on a more diverse set of scanpath prediction models: Itti-Koch model [16], UMSS [35], ScanGAN [25], ScanDMM [31], and the model by Chen et al. [6]. We applied the same set of optimized parameters obtained from

![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-09.jpg?height=445&width=1096&top_left_y=278&top_left_x=292)

![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-09.jpg?height=369&width=276&top_left_y=289&top_left_x=306)

(a) Ground-truth

![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-09.jpg?height=369&width=280&top_left_y=289&top_left_x=656)

(b) Baseline DeepGaze++

![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-09.jpg?height=369&width=281&top_left_y=289&top_left_x=1051)

(c) Improved DeepGaze++

Fig. 6. Visit vs. revisit bias analysis, showing the ratios of visited-revisited elements for three element categories. According to the ground-truth data, text elements are more likely to be visited and revisited than images. The improved model is better aligned with this observation.

| Model |  | DTW $\downarrow$ | Eyenalysis $\downarrow$ | Determinism $\uparrow$ | Laminarity $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Itti-Koch | Baseline | $7.023 \pm 0.261$ | $0.075 \pm 0.014$ | $0.363 \pm 0.154$ | $5.823 \pm \mathbf{1 . 1 6 9}$ |
|  | Improved | $5.824 \pm \mathbf{0 . 2 1 9}$ | $\mathbf{0 . 0 5 3} \pm \mathbf{0 . 0 1 2}$ | $\mathbf{0 . 3 7 8} \pm \mathbf{0 . 1 4 1}$ | $4.943 \pm 1.034$ |
| Chen et al. | Baseline | $4.298 \pm 0.225$ | $0.028 \pm 0.003$ | $\mathbf{1 . 5 9 7} \pm \mathbf{1 . 5 5 6}$ | $7.028 \pm 0.344$ |
|  | Improved | $4.111 \pm \mathbf{0 . 1 8 7}$ | $\mathbf{0 . 0 2 5} \pm \mathbf{0 . 0 0 1}$ | $1.483 \pm 0.188$ | $7.724 \pm \mathbf{1 . 2 5 9}$ |
| UMSS | Baseline | $4.567 \pm 0.394$ | $\mathbf{0 . 0 3 1} \pm \mathbf{0 . 0 0 6}$ | $2.390 \pm 1.044$ | $10.541 \pm 1.764$ |
|  | Improved | $4.537 \pm \mathbf{0 . 4 3 2}$ | $0.033 \pm 0.003$ | $3.123 \pm 1.028$ | $\mathbf{1 2 . 3 0 2} \pm 1.815$ |
| ScanGAN | Baseline | $4.001 \pm 0.379$ | $\mathbf{0 . 0 2 6} \pm \mathbf{0 . 0 0 3}$ | $1.306 \pm 1.025$ | $7.311 \pm 1.817$ |
|  | Improved | $3.973 \pm \mathbf{0 . 2 0 4}$ | $0.027 \pm 0.002$ | $\mathbf{1 . 3 3 1} \pm \mathbf{0 . 7 9 8}$ | $7.900 \pm \mathbf{1 . 4 2 3}$ |
| ScanDMM | Baseline | $4.584 \pm 0.336$ | $0.033 \pm 0.002$ | $\mathbf{0 . 5 9 7} \pm \mathbf{0 . 4 9 9}$ | $5.123 \pm 1.042$ |
|  | Improved | $4.452 \pm \mathbf{0 . 3 7 8}$ | $\mathbf{0 . 0 2 9} \pm \mathbf{0 . 0 0 2}$ | $0.472 \pm 0.546$ | $5.596 \pm \mathbf{0 . 9 9 6}$ |

Table 3. Evaluation of baseline and improved models, showing Mean $\pm$ SD for each metric. Arrows denote the direction of the importance; e.g., $\downarrow$ means "lower is better." Each column's best result is highlighted in boldface.

DeepGaze++ to these other models. Figure 8 and Table 3 show the results. This approach allowed us to assess the performance of the parameters across different scanpath models, providing valuable insights into their applicability beyond a single model.

All the models show improved results on most metrics after employing the optimized set of parameters on most types of GUIs. Similar to DeepGaze++, the Itti-Koch model also incorporates the IOR mechanism. Therefore, adjusting the masking radius to its optimal value has a notable impact on prediction performance. According to our findings, it is clear that there is potential to enhance the performance of scanpath prediction models by utilizing a set of optimal design parameters that cannot be learned from the data. This highlights the importance of considering and optimizing these design parameters to achieve improved performance in scanpath models.

### 4.8 Comparison against other datasets

To further demonstrate the generalization of our optimal parameters and the improved DeepGaze++ model, we evaluate it on MASSVIS [5], one of the largest real-world visualization databases, scraped

![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-10.jpg?height=1272&width=1333&top_left_y=287&top_left_x=173)

Fig. 7. Qualitative comparison between scanpaths. The scanpaths follow a color gradient from red (beginning of trajectory) to blue (end of trajectory).

from various online sources including government reports, infographic blogs, news media websites, and scientific journals. MASSVIS includes scanpaths from 393 screenshots observed by 33 viewers, with an average of 16 viewers per visualization. Each viewer spent 10 seconds examining each visualization, resulting in an average of 37 fixation points. To accommodate the limitation of the baseline DeepGaze++ model of 12 fixation points, we considered the first 15 fixation points in each scanpath.

Table 4 shows that the improved DeepGaze++ model consistently outperforms the baseline model on the four MASSVIS datasets across all scanpath metrics except Laminarity. When Laminarity is high but Determinism is low, it means that the scanpath model quantifies the number of locations that were fixated in detail in the ground-truth scanpath, but were only fixated briefly in the predicted scanpath [1]. In this regard, we can see that the improved model has always a smaller difference between these two metrics, suggesting thus a better alignment with the ground-truth scanpaths.
![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-11.jpg?height=1384&width=1334&top_left_y=280&top_left_x=172)

Fig. 8. Comparison against other scanpath models across GUI types. Each model (baseline, light blue bars) is re-trained with the optimized parameters (improved, dark blue bars) and evaluated on the testing partition. Error bars denote standard deviations

Notably, the best improvements were observed on the InfoVis dataset and the best performance overall was observed on the Science dataset.

### 4.9 Understanding the role of the number of fixation points

All scanpath models are ultimately used to produce a number of fixation points. While we do not consider this to be a design parameter, since it is actually a model outcome, we do find it interesting to study their role in downstream performance. Therefore, we conducted an additional analysis across all the scanpath models considered in our previous experiment. We systematically varied the

| Dataset |  | DTW $\downarrow$ | Eyenalysis $\downarrow$ | Determinism $\uparrow$ | Laminarity $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Government | Baseline | $8.073 \pm 1.932$ | $0.171 \pm 0.102$ | $0.253 \pm 4.057$ | $39.555 \pm 24.446$ |
|  | Improved | $\mathbf{6 . 6 7 4} \pm 2.294$ | $\mathbf{0 . 1 2 5} \pm \mathbf{0 . 1 1 1}$ | $\mathbf{1 . 6 8 0} \pm \mathbf{1 0 . 1 9 2}$ | $33.980 \pm 26.814$ |
| InfoVis | Baseline | $7.318 \pm 1.782$ | $0.147 \pm 0.094$ | $1.418 \pm 10.334$ | $49.564 \pm 25.672$ |
|  | Improved | $5.726 \pm \mathbf{1 . 5 5 8}$ | $\mathbf{0 . 0 8 8} \pm \mathbf{0 . 0 6 6}$ | $3.268 \pm \mathbf{1 4 . 2 6 6}$ | $40.855 \pm 27.360$ |
| Science | Baseline | $5.844 \pm 1.425$ | $0.074 \pm 0.040$ | $4.555 \pm 17.859$ | $49.658 \pm 26.855$ |
|  | Improved | $5.323 \pm \mathbf{1 . 4 7 5}$ | $\mathbf{0 . 0 6 4} \pm \mathbf{0 . 0 4 7}$ | $5.611 \pm \mathbf{1 8 . 5 2 1}$ | $45.203 \pm 25.027$ |
| News | Baseline | $8.103 \pm 2.214$ | $\mathbf{0 . 1 6 3} \pm \mathbf{0 . 1 3 4}$ | $0.110 \pm 2.926$ | $28.426 \pm 25.567$ |
|  | Improved | $7.648 \pm 2.776$ | $0.168 \pm 0.169$ | $\mathbf{0 . 8 7 5} \pm 7.100$ | $29.744 \pm 24.629$ |
| Averaged | Baseline | $7.334 \pm 1.058$ | $0.139 \pm 0.044$ | $1.584 \pm 2.065$ | $41.801 \pm \mathbf{1 0 . 0 9 8}$ |
|  | Improved | $\mathbf{6 . 3 4 3} \pm \mathbf{1 . 0 3 8}$ | $\mathbf{0 . 1 1 1} \pm \mathbf{0 . 0 4 5}$ | $\mathbf{2 . 8 5 9} \pm 2.087$ | $37.445 \pm 6.907$ |

Table 4. Evaluation of baseline and improved DeepGaze++ in the MASSVIS datasets, showing Mean $\pm$ SD for each metric. Arrows denote the direction of the importance; e.g., $\downarrow$ means "lower is better." The best result in each case is highlighted in boldface.

number of fixation points from 5 to 10 and evaluated model performance on the testing partition. The results are shown in Figure 9.

We can observe that an increase in the number of fixation points correlates with improved Determinism and Laminarity values across all models. In addition, Eyenalysis exhibits enhancement in predictive accuracy for more fixation points except the Itti-Koch model. Thus, scanpaths with a larger number of fixation points are more likely to simulate the human's real scanpaths.

## 5 DISCUSSION

Despite the development of scanpath prediction models for GUIs, the extent to which the design parameter choices influence saliency predictions performance has remained underexplored. We have conducted comprehensive experiments in this regard, using a state-of-the-art scanpath model as a reference. By understanding the significance of these parameters, we contribute to the body of knowledge of how people look at GUIs and how to better develop models to predict it.

To what extent do saliency predictions depend on the choices made in design parameters? Our findings draw attention to the considerable influence of design parameters in determining the accuracy of predicting scanpaths in GUIs. Specifically, the role of input image size, masking radius, and IOR decay is significant in assessing user attention and eye movement patterns in GUIs. As shown in Figure 8, optimizing these parameters can substantially enhance scanpath prediction performance. In summary, our research has led to the following findings:

(1) Image size has a large impact on model predictions. Resizing images to smaller dimensions positively impacts prediction performance The best results were observed for images resized to $225 \mathrm{px}$.

(2) Resizing any input image to a square aspect ratio consistently yields superior performance across all GUI types. Mobile GUIs are particularly sensible to the image aspect ratio.

(3) IOR is essential to reduce the likelihood of a user revisiting earlier seen GUI points. Our proposed decay $\gamma=0.1$ addresses an important limitation in DeepGaze++ and leads to improved prediction performance.

![](https://cdn.mathpix.com/cropped/2024_06_04_52a37ebf1823dbcbf7f4g-13.jpg?height=1665&width=1330&top_left_y=283&top_left_x=175)

Fig. 9. Impact of different numbers of fixation points on different models on different GUI types.

(4) When the masking radius increases, prediction quality decreases. The masking radius should find a balance between repetition of viewed areas (small radius) and blocking out too large parts of the GUI (large radius). A sensible value is 0.1 , i.e. $10 \%$ of the available image size.

(5) All the studied GUI types follow a similar trend in terms of optimal parameter settings, although some GUIs may be affected slightly differently, as reported by the four evaluation metrics considered.

Understanding the effect of design decisions on scanpath predictive models allows researchers to be aware of the fact that even small variations can lead to more accurate results. By examining how different design elements gauge users' attention, researchers can identify effective design strategies that promote user engagement and optimize information presentation. This knowledge can be applied to various domains, including website design, multimedia content creation, and advertising, enabling designers to create more visually appealing and user-friendly interfaces. Furthermore, the evaluation on multiple scanpath models shows the generalizability of our findings. We hope the insights presented in this paper could serve as a reference for future researchers working on saliency prediction in GUIs.

### 5.1 Limitations and future work

While our findings offer valuable new knowledge to optimize scanpath model performance, our experiments examined the impact of design parameters in isolation (i.e. we studied one design parameter at a time), therefore future work should consider a joint optimization procedure. It may be the case that an automatically optimized set (e.g. with Bayesian optimization) can lead to more accurate performance results.

In principle, the proposed values of the design parameters we studied are meant to be applicable to every scanpath model. We found that this is the case for the 5 models we evaluated, most of them offering state-of-the-art performance, but we also acknowledge that there might be other sets of values that could work better for a particular model.

Future work should also consider more fixations points as model output. In all our experiments, DeepGaze++ was used to predict trajectories of 10 fixations each, to facilitate comparisons against previous work [17, 20]. However, it may be the case that predicting more fixation points would result in more (or less) informed trajectories, which may in turn affect the performance evaluation metrics. For example, if far-distant points (in time) tend to be more dispersed, the DTW values will increase. Such exploration can help develop better computational scanpath models.

### 5.2 Privacy and ethics statement

On the positive side, our research focuses on providing optimal parameters for scanpath models, enabling more accurate predictions. But this enhanced prediction goes beyond mere gaze direction, offering valuable insights into an individual's perceptual and cognitive processes. Our advancements open up opportunities for innovative applications, particularly in the realm of designing or adapting user interfaces. However, it is important to consider that the use of these optimal parameters and more accurate models can also be exploited for unforeseen purposes, such as optimizing advertisements placement on websites or enabling "dark patterns" such as making the user click on some content as a result of some GUI adaptation that optimized the interface elements for quick scannability. Overall, we should note that striking a balance between harnessing the technology's potential benefits and safeguarding individuals' rights is crucial for responsible development and deployment.

## 6 CONCLUSION

Scanpath models rely on a series of design parameters that are usually taken for granted. We have shown that even small variations of these parameters have a noticeable impact on several evaluation metrics. As a result, we have found a set of optimal parameters that improve the state of
the art in scanpath modeling. These parameters have resulted in an improved DeepGaze++ model that can better capture both the spatial and temporal characteristics of scanpaths. These parameters are replicable to other computational models and datasets, showing the generalizability of our findings. The community can use therefore this improved set of model parameters (or even the improved models themselves) to get a better understanding of how users are likely to perceive GUIs. Ultimately, this work provides invaluable insights for designers and researchers interested in predicting users' viewing strategies on GUIs. Our software and models are publicly available https://github.com/prviin/scanpath-design-decisions.

## ACKNOWLEDGMENTS

Research supported by the Horizon 2020 FET program of the European Union (grant CHIST-ERA20-BCI-001) and the European Innovation Council Pathfinder program (SYMBIOTIK project, grant 101071147).

## REFERENCES

[1] Nicola C Anderson, Fraser Anderson, Alan Kingstone, and Walter F Bischof. 2015. A comparison of scanpath comparison methods. Behavior research methods 47 (2015), 1377-1392.

[2] Marc Assens, Xavier Giro-i Nieto, Kevin McGuinness, and Noel E O'Connor. 2018. PathGAN: Visual scanpath prediction with generative adversarial networks. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops. $0-0$.

[3] Wentao Bao and Zhenzhong Chen. 2020. Human scanpath prediction based on deep convolutional saccadic model. Neurocomputing 404 (2020), 154-164.

[4] Donald J Berndt and James Clifford. 1994. Using dynamic time warping to find patterns in time series. In Proceedings of the 3rd international conference on knowledge discovery and data mining. 359-370.

[5] Michelle A. Borkin, Zoya Bylinskii, Nam Wook Kim, Constance May Bainbridge, Chelsea S. Yeh, Daniel Borkin, Hanspeter Pfister, and Aude Oliva. 2016. Beyond Memorability: Visualization Recognition and Recall. IEEE Transactions on Visualization and Computer Graphics 22, 1 (2016).

[6] Xianyu Chen, Ming Jiang, and Qi Zhao. 2021. Predicting human scanpaths in visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10876-10885.

[7] Zhenzhong Chen and Wanjie Sun. 2018. Scanpath Prediction for Visual Attention using IOR-ROI LSTM.. In I7CAI. 642-648.

[8] Aladine Chetouani and Leida Li. 2020. On the use of a scanpath predictor and convolutional neural network for blind image quality assessment. Signal Processing: Image Communication 89 (2020), 115963.

[9] Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, and Rita Cucchiara. 2018. SAM: Pushing the limits of saliency prediction models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. $1890-1892$.

[10] Erwan Joel David, Pierre Lebranchu, Matthieu Perreira Da Silva, and Patrick Le Callet. 2019. Predicting artificial visual field losses: A gaze-based inference study. Journal of Vision 19, 14 (2019), 22-22.

[11] Ryan Anthony Jalova de Belen, Tomasz Bednarz, and Arcot Sowmya. 2022. Scanpathnet: A recurrent mixture density network for scanpath prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. $5010-5020$.

[12] Sukru Eraslan, Yeliz Yesilada, and Simon Harper. 2016. Eye tracking scanpath analysis on web pages: how many users?. In Proceedings of the ninth biennial ACM symposium on eye tracking research \& applications. 103-110.

[13] Ramin Fahimi. 2018. Sequential selection, saliency and scanpaths. (2018).

[14] Ramin Fahimi and Neil DB Bruce. 2021. On metrics for measuring scanpath similarity. Behavior Research Methods 53 (2021), 609-628.

[15] Camilo Fosco, Vincent Casser, Amish Kumar Bedi, Peter O’Donovan, Aaron Hertzmann, and Zoya Bylinskii. 2020. Predicting visual importance across graphic design types. In Proceedings of the 33rd Annual ACM Symposium on User Interface Software and Technology. 249-260.

[16] Laurent Itti, Christof Koch, and Ernst Niebur. 1998. A model of saliency-based visual attention for rapid scene analysis. IEEE Transactions on pattern analysis and machine intelligence 20, 11 (1998), 1254-1259.

[17] Yue Jiang, Luis A Leiva, Hamed Rezazadegan Tavakoli, Paul RB Houssel, Julia Kylmälä, and Antti Oulasvirta. 2023 UEyes: Understanding Visual Saliency across User Interface Types. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems. 1-21.

[18] Raymond M Klein. 2000. Inhibition of return. Trends in cognitive sciences 4,4 (2000), 138-147.

[19] Matthias Kümmerer and Matthias Bethge. 2021. State-of-the-art in human scanpath prediction. arXiv preprint arXiv:2102.12239 (2021).

[20] Matthias Kümmerer, Matthias Bethge, and Thomas SA Wallis. 2022. DeepGaze III: Modeling free-viewing human scanpaths with deep learning. Journal of Vision 22, 5 (2022), 7-7.

[21] Daniël Lakens. 2013. Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs. Front. Psychol. 4, 863 (2013).

[22] Luis A. Leiva, Yunfei Xue, Avya Bansal, Hamed R. Tavakoli, Tuğçe Köroğlu, Jingzhou Du, Niraj R. Dayama, and Antti Oulasvirta. 2020. Understanding Visual Saliency in Mobile User Interfaces. In Proceedings of the Intl. Conf. on Human-computer interaction with mobile devices and services (MobileHCI).

[23] Mu Li, Kanglong Fan, and Kede Ma. 2023. Scanpath Prediction in Panoramic Videos via Expected Code Length Minimization. arXiv preprint arXiv:2305.02536 (2023).

[24] Yue Li, Dong Liu, Houqiang Li, Li Li, Zhu Li, and Feng Wu. 2018. Learning a convolutional neural network for image compact-resolution. IEEE Transactions on Image Processing 28, 3 (2018), 1092-1107.

[25] Daniel Martin, Ana Serrano, Alexander W Bergman, Gordon Wetzstein, and Belen Masia. 2022. ScanGAN360: A generative model of realistic scanpaths for 360 images. IEEE Transactions on Visualization and Computer Graphics 28, 5 (2022), 2003-2013.

[26] Sebastiaan Mathôt, Filipe Cristino, Iain D Gilchrist, and Jan Theeuwes. 2012. A simple way to estimate similarity between pairs of eye movement sequences. Journal of Eye Movement Research 5, 1 (2012), 1-15.

[27] Meinard Müller. 2007. Dynamic time warping. Information retrieval for music and motion (2007), 69-84.

[28] Thuyen Ngo and BS Manjunath. 2017. Saccade gaze prediction using a recurrent neural network. In 2017 IEEE International Conference on Image Processing (ICIP). IEEE, 3435-3439.

[29] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. 2022. On aliased resizing and surprising subtleties in GAN evaluation In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11410-11420.

[30] Michael I Posner, Yoav Cohen, et al. 1984. Components of visual orienting. Attention and performance X: Control of language processes 32 (1984), 531-556.

[31] Xiangjie Sui, Yuming Fang, Hanwei Zhu, Shiqi Wang, and Zhou Wang. 2023. ScanDMM: A Deep Markov Model of Scanpath Prediction for $360^{\circ}$ Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6989-6999.

[32] Wanjie Sun, Zhenzhong Chen, and Feng Wu. 2019. Visual scanpath prediction using IOR-ROI recurrent mixture density network. IEEE transactions on pattern analysis and machine intelligence 43, 6 (2019), 2101-2118.

[33] Nađa Terzimehić, Renate Häuslschmid, Heinrich Hussmann, and MC Schraefel. 2019. A review \& analysis of mindfulness research in HCI: Framing current lines of research and future opportunities. In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems. 1-13.

[34] Xiaowei Wang, Xubo Li, Haiying Wang, Wenning Zhao, and Xia Liu. 2023. An Improved Dynamic Time Warping Method Combining Distance Density Clustering for Eye Movement Analysis. Journal of Mechanics in Medicine and Biology 23, 02 (2023), 2350031.

[35] Yao Wang, Andreas Bulling, et al. 2023. Scanpath prediction on information visualisations. IEEE Transactions on Visualization and Computer Graphics (2023).

[36] Calden Wloka, Iuliia Kotseruba, and John K Tsotsos. 2018. Active fixation control to predict saccade sequences. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 3184-3193.

[37] Chen Xia, Junwei Han, Fei Qi, and Guangming Shi. 2019. Predicting human saccadic scanpaths based on iterative representation learning. IEEE Transactions on Image Processing 28, 7 (2019), 3502-3515.

[38] Mulong Xie, Sidong Feng, Zhenchang Xing, Jieshan Chen, and Chunyang Chen. 2020. UIED: a hybrid tool for GUI element detection. In Proceedings of the 28th ACM foint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 1655-1659.

[39] Fei Yan, Cheng Chen, Peng Xiao, Siyu Qi, Zhiliang Wang, and Ruoxiu Xiao. 2021. Review of visual saliency prediction: Development process from neurobiological basis to deep models. Applied Sciences 12, 1 (2021), 309.

[40] Jiawei Yang, Guangtao Zhai, and Huiyu Duan. 2019. Predicting the visual saliency of the people with VIMS. In 2019 IEEE Visual Communications and Image Processing (VCIP). IEEE, 1-4.

[41] Yucheng Zhu, Guangtao Zhai, Xiongkuo Min, and Jiantao Zhou. 2019. The prediction of saliency map for head and eye movements in 360 degree images. IEEE Transactions on Multimedia 22, 9 (2019), 2331-2344.

[42] Yucheng Zhu, Guangtao Zhai, Yiwei Yang, Huiyu Duan, Xiongkuo Min, and Xiaokang Yang. 2021. Viewing behavior supported visual saliency predictor for 360 degree videos. IEEE Transactions on Circuits and Systems for Video Technology 32,7 (2021), 4188-4201.

Received November 2023; revised January 2024; accepted March 2024


[^0]:    Authors' addresses: Parvin Emami, parvin@emami@uni.lu, University of Luxembourg, Luxembourg; Yue Jiang, yue.jiang@ aalto.fi; Zixin Guo, zixin.guo@aalto.fi, Aalto University, Finland; Luis A. Leiva, name.surname@uni.lu, University of Luxembourg, Luxembourg.

</end of paper 4>


