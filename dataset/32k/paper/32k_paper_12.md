<paper 0>
# A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data 

Abien Fred M. Agarap<br>abienfred.agarap@gmail.com


#### Abstract

Gated Recurrent Unit (GRU) is a recently-developed variation of the long short-term memory (LSTM) unit, both of which are variants of recurrent neural network (RNN). Through empirical evidence, both models have been proven to be effective in a wide variety of machine learning tasks such as natural language processing[23], speech recognition[4], and text classification[24]. Conventionally, like most neural networks, both of the aforementioned RNN variants employ the Softmax function as its final output layer for its prediction, and the cross-entropy function for computing its loss. In this paper, we present an amendment to this norm by introducing linear support vector machine (SVM) as the replacement for Softmax in the final output layer of a GRU model. Furthermore, the cross-entropy function shall be replaced with a margin-based function. While there have been similar studies[2, 22], this proposal is primarily intended for binary classification on intrusion detection using the 2013 network traffic data from the honeypot systems of Kyoto University. Results show that the GRU-SVM model performs relatively higher than the conventional GRU-Softmax model. The proposed model reached a training accuracy of $\approx 81.54 \%$ and a testing accuracy of $\approx 84.15 \%$, while the latter was able to reach a training accuracy of $\approx 63.07 \%$ and a testing accuracy of $\approx 70.75 \%$. In addition, the juxtaposition of these two final output layers indicate that the SVM would outperform Softmax in prediction time - a theoretical implication which was supported by the actual training and testing time in the study.


## CCS CONCEPTS

- Computing methodologies $\rightarrow$ Supervised learning by classification; Support vector machines; Neural networks; - Security and privacy $\rightarrow$ Intrusion detection systems;


## KEYWORDS

artificial intelligence; artificial neural networks; gated recurrent units; intrusion detection; machine learning; recurrent neural networks; support vector machine[^0]

## ACM Reference Format:

Abien Fred M. Agarap. 2018. A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data. In ICMLC 2018: 2018 10th International Conference on Machine Learning and Computing, February 26-28, 2018, Macau, China. ACM, New York, NY, USA, 5 pages. https: $/ /$ doi.org $/ 10.1145 / 3195106.3195117$

## 1 INTRODUCTION

By 2019, the cost to the global economy due to cybercrime is projected to reach $\$ 2$ trillion[10]. Among the contributory felonies to cybercrime is intrusions, which is defined as illegal or unauthorized use of a network or a system by attackers[7]. An intrusion detection system (IDS) is used to identify the said malicious activity[7]. The most common method used for uncovering intrusions is the analysis of user activities[7, 13, 17]. However, the aforementioned method is laborious when done manually, since the data of user activities is massive in nature $[6,14]$. To simplify the problem, automation through machine learning must be done.

A study by Mukkamala, Janoski, \& Sung (2002)[17] shows how support vector machine (SVM) and artificial neural network (ANN) can be used to accomplish the said task. In machine learning, SVM separates two classes of data points using a hyperplane[5]. On the other hand, an ANN is a computational model that represents the human brain, and shows information is passed from a neuron to another[18].

An approach combining ANN and SVM was proposed by Alalshekmubarak \& Smith[2], for time-series classification. Specifically, they combined echo state network (ESN, a variant of recurrent neural network or RNN) and SVM. This research presents a modified version of the aforementioned proposal, and use it for intrusion detection. The proposed model will use recurrent neural network (RNNs) with gated recurrent units (GRUs) in place of ESN. RNNs are used for analyzing and/or predicting sequential data, making it a viable candidate for intrusion detection[18], since network traffic data is sequential in nature.

## 2 METHODOLOGY

### 2.1 Machine Intelligence Library

Google TensorFlow[1] was used to implement the neural network models in this study - both the proposed and its comparator.

### 2.2 The Dataset

The 2013 Kyoto University honeypot systems' network traffic data[20] was used in this study. It has 24 statistical features[20]; (1) 14 features from the KDD Cup 1999 dataset[21], and (2) 10 additional features, which according to Song, Takakura, \& Okabe (2006)[20],
might be pivotal in a more effective investigation on intrusion detection. Only 22 dataset features were used in the study.

### 2.3 Data Preprocessing

For the experiment, only $25 \%$ of the whole 16.2 GB network traffic dataset was used, i.e. $\approx 4.1 \mathrm{~GB}$ of data (from January 1, 2013 to June 1, 2013). Before using the dataset for the experiment, it was normalized first - standardization (for continuous data, see Eq. 1) and indexing (for categorical data), then it was binned (discretized).

$$
\begin{equation*}
z=\frac{X-\mu}{\sigma} \tag{1}
\end{equation*}
$$

where $X$ is the feature value to be standardized, $\mu$ is the mean value of the given feature, and $\sigma$ is its standard deviation. But for efficiency, the StandardScaler().fit_transform() function of Scikit-learn[19] was used for the data standardization in this study.

For indexing, the categories were mapped to $[0, n-1]$ using the LabelEncoder().fit_transform() function of Scikit-learn[19]

After dataset normalization, the continuous features were binned (decile binning, a discretization/quantization technique). This was done by getting the $10^{t h}, 20^{t h}, \ldots, 90^{t h}$, and $100^{t h}$ quantile of the features, and their indices served as their bin number. This process was done using the qcut () function of pandas[16]. Binning reduces the required computational cost, and improves the classification performance on the dataset[15]. Lastly, the features were one-hot encoded, making it ready for use by the models.

### 2.4 The GRU-SVM Neural Network Architecture

Similar to the work of Alalshekmubarak \& Smith (2013)[2] and Tang (2013)[22], the present paper proposes to use SVM as the classifier in a neural network architecture. Specifically, a Gated Recurrent Unit (GRU) RNN (see Figure 1).

![](https://cdn.mathpix.com/cropped/2024_06_04_2b10739b8d40ed1e866fg-2.jpg?height=423&width=767&top_left_y=1762&top_left_x=213)

Figure 1: The proposed GRU-SVM architecture model, with $n-1$ GRU unit inputs, and SVM as its classifier.
For this study, there were 21 features used as the model input. Then, the parameters are learned through the gating mechanism of GRU[3] (Equations (2) to (5)).

$$
\begin{gather*}
z=\sigma\left(\mathbf{W}_{z} \cdot\left[h_{t-1}, x_{t}\right]\right)  \tag{2}\\
r=\sigma\left(\mathbf{W}_{r} \cdot\left[h_{t-1}, x_{t}\right]\right)  \tag{3}\\
\tilde{h}_{t}=\tanh \left(\mathbf{W} \cdot\left[r_{t} * h_{t-1}, x_{t}\right]\right)  \tag{4}\\
h_{t}=\left(1-z_{t}\right) * h_{t-1}+z_{t} * \tilde{h}_{t} \tag{5}
\end{gather*}
$$

But with the introduction of SVM as its final layer, the parameters are also learned by optimizing the objective function of SVM (see Eq. 6). Then, instead of measuring the network loss using crossentropy function, the GRU-SVM model will use the loss function of SVM (Eq. 6).

$$
\begin{equation*}
\min \frac{1}{2}\|\mathbf{w}\|_{1}^{2}+C \sum_{i=1}^{n} \max \left(0,1-y_{i}^{\prime}\left(\mathbf{w}^{T} \mathbf{x}_{i}+b_{i}\right)\right) \tag{6}
\end{equation*}
$$

Eq. 6 is known as the unconstrained optimization problem of L1SVM. However, it is not differentiable. On the contrary, its variation, known as the L2-SVM is differentiable and is more stable[22] than the L1-SVM:

$$
\begin{equation*}
\min \frac{1}{2}\|\mathbf{w}\|_{2}^{2}+C \sum_{i=1}^{n} \max \left(0,1-y_{i}^{\prime}\left(\mathbf{w}^{T} \mathbf{x}_{i}+b_{i}\right)\right)^{2} \tag{7}
\end{equation*}
$$

The L2-SVM was used for the proposed GRU-SVM architecture. As for the prediction, the decision function $f(x)=\operatorname{sign}(\mathbf{w x}+b)$ produces a score vector for each classes. So, to get the predicted class label $y$ of a data $x$, the argmax function is used:

$$
\text { predicted_class }=\operatorname{argmax}(\operatorname{sign}(\mathbf{w} \mathbf{x}+b))
$$

The argmax function will return the index of the highest score across the vector of the predicted classes.

The proposed GRU-SVM model may be summarized as follows:

(1) Input the dataset features $\left\{\mathbf{x}_{i} \mid \mathbf{x}_{i} \in \mathbb{R}^{m}\right\}$ to the GRU model.

(2) Initialize the learning parameters weights and biases with arbitrary values (they will be adjusted through training).

(3) The cell states of GRU are computed based on the input features $\mathbf{x}_{i}$, and its learning parameters values.

(4) At the last time step, the prediction of the model is computed using the decision function of SVM: $f(x)=\operatorname{sign}(\mathbf{w} \mathbf{x}+b)$.

(5) The loss of the neural network is computed using Eq. 7.

(6) An optimization algorithm is used for loss minimization (for this study, the Adam[12] optimizer was used). Optimization adjusts the weights and biases based on the computed loss.

(7) This process is repeated until the neural network reaches the desired accuracy or the highest accuracy possible. Afterwards, the trained model can be used for binary classification on a given data.

The program implementation of the proposed GRU-SVM model is available at https://github.com/AFAgarap/gru-svm.

Table 1: Hyper-parameters used in both neural networks.

| Hyper-parameters | GRU-SVM | GRU-Softmax |
| :---: | :---: | :---: |
| Batch Size | 256 | 256 |
| Cell Size | 256 | 256 |
| Dropout Rate | 0.85 | 0.8 |
| Epochs | 5 | 5 |
| Learning Rate | $1 \mathrm{e}-5$ | $1 \mathrm{e}-6$ |
| SVM C | 0.5 | N/A |

### 2.5 Data Analysis

The effectiveness of the proposed GRU-SVM model was measured through the two phases of the experiment: (1) training phase, and (2) test phase. Along with the proposed model, the conventional GRU-Softmax was also trained and tested on the same dataset

The first phase of the experiment utilized $80 \%$ of total data points ( $\approx 3.2 \mathrm{~GB}$, or $14,856,316$ lines of network traffic log) from the $25 \%$ of the dataset. After normalization and binning, it was revealed through a high-level inspection that a duplication occurred. Using the DataFrame.drop_duplicates() of pandas[16], the 14, 856, 316-line data dropped down to 1, 898, 322 lines ( $\approx 40 \mathrm{MB}$ ).

The second phase of the experiment was the evaluation of the two trained models using $20 \%$ of total data points from the $25 \%$ of the dataset. The testing dataset also experienced a drastic shrinkage in size - from 3, 714, 078 lines to 420,759 lines ( $\approx 9 \mathrm{MB}$ ).

The parameters for the experiments are the following: (1) Accuracy, (2) Epochs, (3) Loss, (4) Run time, (5) Number of data points, (6) Number of false positives, (7) Number of false negatives. These parameters are based on the ones considered by Mukkamala, Janoski, \& Sung (2002)[17] in their study where they compared SVM and a feed-forward neural network for intrusion detection. Lastly, the statistical measures for binary classification were measured (true positive rate, true negative rate, false positive rate, and false negative rate).

## 3 RESULTS

All experiments in this study were conducted on a laptop computer with Intel Core(TM) i5-6300HQ CPU @ 2.30GHz x 4, 16GB of DDR3 RAM, and NVIDIA GeForce GTX 960M 4GB DDR5 GPU. The hyperparameters used for both models were assigned by hand, and not through hyper-parameter optimization/tuning (see Table 1).

Both models were trained on $1,898,240$ lines of network traffic data for 5 epochs. Afterwards, the trained models were tested to classify 420,608 lines of network traffic data for 5 epochs. Only the specified number of lines of network traffic data were used for the experiments as those are the values that are divisble by the batch size of 256. The class distribution of both training and testing dataset is specified in Table 2.

The experiment results are summarized in Table 3. Although the loss for both models were recorded, it will not be a topic of further discussion as they are not comparable since they are in different scales. Meanwhile, Tables $4 \& 5$ show the statistical measures for binary classification by the models during training and testing.

Figure 2 shows that for 5 epochs on the 1,898,240-line network traffic data (a total exposure of $9,491,200$ to the training dataset), the
Table 2: Class distribution of training and testing dataset.

| Class | Training data | Testing data |
| :---: | :---: | :---: |
| Normal | 794,512 | 157,914 |
| Intrusion detected | $1,103,728$ | 262,694 |

Table 3: Summary of experiment results on both GRU-SVM and GRU-Softmax models.

| Parameter | GRU-SVM | GRU-Softmax |
| :---: | :---: | :---: |
| No. of data points - Training | $1,898,240$ | $1,898,240$ |
| No. of data points - Testing | 420,608 | 420,608 |
| Epochs | 5 | 5 |
| Accuracy - Training | $\approx 81.54 \%$ | $\approx 63.07 \%$ |
| Accuracy - Testing | $\approx 84.15 \%$ | $\approx 70.75 \%$ |
| Loss - Training | $\approx 131.21$ | $\approx 0.62142$ |
| Loss - Testing | $\approx 129.62$ | $\approx 0.62518$ |
| Run time - Training | $\approx 16.72 \mathrm{mins}$ | $\approx 17.18 \mathrm{mins}$ |
| Run time - Testing | $\approx 1.37 \mathrm{mins}$ | $\approx 1.67 \mathrm{mins}$ |
| No. of false positives - Training | 889,327 | $3,017,548$ |
| No. of false positives - Testing | 192,635 | 32,255 |
| No. of false negatives - Training | 862,419 | 487,175 |
| No. of false negatives - Testing | 140,535 | 582,105 |

Table 4: Statistical measures on binary classification: Training performance of the GRU-SVM and GRU-Softmax models.

| Parameter | GRU-SVM | GRU-Softmax |
| :---: | :---: | :---: |
| True positive rate | $\approx 84.3726 \%$ | $\approx 91.1721 \%$ |
| True negative rate | $\approx 77.6132 \%$ | $\approx 24.0402 \%$ |
| False positive rate | $\approx 22.3867 \%$ | $\approx 75.9597 \%$ |
| False negative rate | $\approx 15.6273 \%$ | $\approx 8.82781 \%$ |

Table 5: Statistical measures on binary classification: Testing performance of the GRU-SVM and GRU-Softmax models.

| Parameter | GRU-SVM | GRU-Softmax |
| :---: | :---: | :---: |
| True positive rate | $\approx 89.3005 \%$ | $\approx 55.6819 \%$ |
| True negative rate | $\approx 75.6025 \%$ | $\approx 95.9149 \%$ |
| False positive rate | $\approx 10.6995 \%$ | $\approx 4.08513 \%$ |
| False negative rate | $\approx 24.3975 \%$ | $\approx 44.3181 \%$ |

GRU-SVM model was able to finish its training in 16 minutes and 43 seconds. On the other hand, the GRU-Softmax model finished its training in 17 minutes and 11 seconds.

Figure 3 shows that for 5 epochs on the 420,608-line network traffic data (a total test prediction of $2,103,040$ ), the GRU-SVM model was able to finish its testing in 1 minute and 22 seconds. On the other hand, the GRU-Softmax model finished its testing in 1 minute and 40 seconds.

![](https://cdn.mathpix.com/cropped/2024_06_04_2b10739b8d40ed1e866fg-4.jpg?height=434&width=846&top_left_y=301&top_left_x=192)

Figure 2: Training accuracy of the proposed GRU-SVM model, and the conventional GRU-Softmax model.

![](https://cdn.mathpix.com/cropped/2024_06_04_2b10739b8d40ed1e866fg-4.jpg?height=442&width=848&top_left_y=885&top_left_x=194)

Figure 3: Testing accuracy of the proposed GRU-SVM model, and the conventional GRU-Softmax model.

## 4 DISCUSSION

The empirical evidence presented in this paper suggests that SVM outperforms Softmax function in terms of prediction accuracy, when used as the final output layer in a neural network. This finding corroborates the claims by Alalshekmubarak \& Smith (2013)[2] and Tang (2013)[22], and supports the claim that SVM is a more practical approach than Softmax for binary classification. Not only did the GRU-SVM model outperform the GRU-Softmax in terms of prediction accuracy, but it also outperformed the conventional model in terms of training time and testing time. Thus, supporting the theoretical implication as per the respective algorithm complexities of each classifier.

The reported training accuracy of $\approx 81.54 \%$ and testing accuracy of $\approx 84.15 \%$ posits that the GRU-SVM model has a relatively stronger predictive performance than the GRU-Softmax model (with training accuracy of $\approx 63.07 \%$ and testing accuracy of $\approx 70.75 \%$ ). Hence, we propose a theory to explain the relatively lower performance of Softmax compared to SVM in this particular scenario. First, SVM was designed primarily for binary classification[5], while Softmax is best-fit for multinomial classification[11]. Building on the premise, SVM does not care about the individual scores of the classes it predicts, it only requires its margins to be satisfied[11]. On the contrary, the Softmax function will always find a way to improve its predicted probability distribution by ensuring that the correct class has the higher/highest probability, and the incorrect classes

![](https://cdn.mathpix.com/cropped/2024_06_04_2b10739b8d40ed1e866fg-4.jpg?height=545&width=833&top_left_y=302&top_left_x=1123)

Figure 4: Image from [9]. Graph of a sigmoid $\sigma$ function.

have the lower probability. This behavior of the Softmax function is exemplary, but excessive for a problem like binary classification. Given that the sigmoid $\sigma$ function is a special case of Softmax (see Eq. 8-9), we can refer to its graph as to how it classifies a network output.

$$
\begin{gather*}
\sigma(y)=\frac{1}{1+e^{-y}}=\frac{1}{1+\frac{1}{e^{y}}}=\frac{1}{\frac{e^{y}+1}{e^{y}}}=\frac{e^{y}}{1+e^{y}}=\frac{e^{y}}{e^{0}+e^{y}}  \tag{8}\\
\operatorname{softmax}(y)=\frac{e^{y_{i}}}{\sum_{i=0}^{n=1} e^{y_{i}}}=\frac{e^{y_{i}}}{e^{y_{0}}+e^{y_{1}}} \tag{9}
\end{gather*}
$$

It can be inferred from the graph of sigmoid $\sigma$ function (see Figure 4) that $y$ values tend to respond less to changes in $x$. In other words, the gradients would be small, which gives rise to the "vanishing gradients" problem. Indeed, one of the problems being solved by LSTM, and consequently, by its variants such as GRU[3, 8]. This behavior defeats the purpose of GRU and LSTM solving the problems of a traditional RNN. We posit that this is the cause of misclassifications by the GRU-Softmax model.

The said erroneous manner of the GRU-Softmax model reflects as a favor for the GRU-SVM model. But the comparison of the exhibited predictive accuracies of both models is not the only reason for the practicality in choosing SVM over Softmax in this case. The amount of training time and testing time were also considered. As their computational complexities suggest, SVM has the upper hand over Softmax. This is because the algorithm complexity of the predictor function in SVM is only $O(1)$. On the other hand, the predictor function of Softmax has an algorithm complexity of $O(n)$. As results have shown, the GRU-SVM model also outperformed the GRU-Softmax model in both training time and testing time. Thus, it corroborates the respective algorithm complexities of the classifiers.

## 5 CONCLUSION AND RECOMMENDATION

We proposed an amendment to the architecture of GRU RNN by using SVM as its final output layer in a binary/non-probabilistic classification task. This amendment was seen as viable for the fast prediction time of SVM compared to Softmax. To test the model,
we conducted an experiment comparing it with the established GRU-Softmax model. Consequently, the empirical data attests to the effectiveness of the proposed GRU-SVM model over its comparator in terms of predictive accuracy, and training and testing time.

Further work must be done to validate the effectiveness of the proposed GRU-SVM model in other binary classification tasks. Extended study on the proposed model for a faster multinomial classification would prove to be prolific as well. Lastly, the theory presented to explain the relatively low performance of the Softmax function as a binary classifier might be a pre-cursor to further studies.

## 6 ACKNOWLEDGMENT

An appreciation to the open source community (Cross Validated, GitHub, Stack Overflow) for the virtually infinite source of information and knowledge; to the Kyoto University for their intrusion detection dataset from their honeypot system.

## REFERENCES

[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2015. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. (2015). http://tensorflow.org/ Software available from tensorflow.org.

[2] A. Alalshekmubarak and L.S. Smith. 2013. A Novel Approach Combining Recurrent Neural Network and Support Vector Machines for Time Series Classification. In Innovations in Information Technology (IIT), 2013 9th International Conference on. IEEE, 42-47.

[3] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078 (2014)

[4] Jan K Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio. 2015. Attention-based models for speech recognition. In Advances in Neural Information Processing Systems. 577-585.

[5] C. Cortes and V. Vapnik. 1995. Support-vector Networks. Machine Learning 20.3 (1995), 273-297. https://doi.org/10.1007/BF00994018

[6] Jeremy Frank. 1994. Artificial intelligence and intrusion detection: Current and future directions. In Proceedings of the 17th national computer security conference, Vol. 10. Baltimore, USA, 1-12.

[7] Anup K Ghosh, Aaron Schwartzbard, and Michael Schatz. 1999. Learning Program Behavior Profiles for Intrusion Detection.. In Workshop on Intrusion Detection and Network Monitoring, Vol. 51462. 1-13.

[8] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9,8 (1997), 1735-1780.

[9] Yohan Grember (https://stackoverflow.com/users/7672928/yohan grember). [n. d.]. Binary classification with Softmax. Stack Overflow. ([n. d.]). arXiv:https://stackoverflow.com/questions/45793856 https://stackoverflow.com/ questions/45793856 URL:https://stackoverflow.com/questions/45793856 (version: 2017-08-21).

[10] Juniper. May 12, 2015. Cybercrime will cost Businesses over \$2 Trillion by 2019. https://www.juniperresearch.com/press/press-releases/ cybercrime-cost-businesses-over-2trillion. (May 12, 2015). Accessed: May 6, 2017.

[11] Anrej Karpathy. [n. d.]. CS231n Convolutional Neural Networks for Visual Recognition. http://cs231n.github.io/. ([n. d.]).

[12] Diederik Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[13] Sandeep Kumar and Eugene H Spafford. 1994. An application of pattern matching in intrusion detection. (1994).

[14] MIT Lincoln Laboratory. 1999. 1999 DARPA Intrusion Detection Evaluation Data Set. https://www.ll.mit.edu/ideval/data/1999data.html. (1999).

[15] Jonathan L Lustgarten, Vanathi Gopalakrishnan, Himanshu Grover, and Shyam Visweswaran. 2008. Improving classification performance with discretization on biomedical datasets. In AMIA annual symposium proceedings, Vol. 2008. American Medical Informatics Association, 445.

[16] Wes McKinney. 2010. Data Structures for Statistical Computing in Python. In Proceedings of the 9th Python in Science Conference, Stéfan van der Walt and Jarrod Millman (Eds.). $51-56$.

[17] Srinivas Mukkamala, Guadalupe Janoski, and Andrew Sung. 2002. Intrusion detection: support vector machines and neural networks. In proceedings of the IEEE International 7oint Conference on Neural Networks (ANNIE), St. Louis, MO. $1702-1707$.

[18] M. Negnevitsky. 2011. Artificial Intelligence: A Guide to Intelligent Systems (3rd ed.). Pearson Education Ltd., Essex, England.

[19] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12 (2011), 2825-2830.

[20] Jungsuk Song, Hiroki Takakura, and Yasuo Okabe. 2006. Description of kyoto university benchmark data. Available at link: http://www.takakura.com/Kyoto_data/BenchmarkData-Descriptionv5.pdf.[Accessed on 15 March 2016] (2006).

[21] J Stolfo, Wei Fan, Wenke Lee, Andreas Prodromidis, and Philip K Chan. 2000. Cost-based modeling and evaluation for data mining with application to fraud and intrusion detection. Results from the JAM Project by Salvatore (2000).

[22] Yichuan Tang. 2013. Deep learning using linear support vector machines. arXiv preprint arXiv:1306.0239 (2013).

[23] Tsung-Hsien Wen, Milica Gasic, Nikola Mrksic, Pei-Hao Su, David Vandyke, and Steve Young. 2015. Semantically conditioned lstm-based natural language generation for spoken dialogue systems. arXiv preprint arXiv:1508.01745 (2015).

[24] Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alexander J Smola, and Eduard H Hovy. 2016. Hierarchical Attention Networks for Document Classification.. In HLT-NAACL. 1480-1489.


[^0]:    Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

    ICMLC 2018, February 26-28, 2018, Macau, China

    ๑ 2018 Association for Computing Machinery.

    ACM ISBN 978-1-4503-6353-2/18/02...\$15.00

    https://doi.org/10.1145/3195106.3195117

</end of paper 0>


<paper 1>
# Deep Learning using Rectified Linear Units (ReLU) 

Abien Fred M. Agarap<br>abienfred.agarap@gmail.com


#### Abstract

We introduce the use of rectified linear units (ReLU) as the classification function in a deep neural network (DNN). Conventionally, ReLU is used as an activation function in DNNs, with Softmax function as their classification function. However, there have been several studies on using a classification function other than Softmax, and this study is an addition to those. We accomplish this by taking the activation of the penultimate layer $h_{n-1}$ in a neural network, then multiply it by weight parameters $\theta$ to get the raw scores $o_{i}$. Afterwards, we threshold the raw scores $o_{i}$ by 0 , i.e. $f(o)=\max \left(0, o_{i}\right)$, where $f(o)$ is the ReLU function. We provide class predictions $\hat{y}$ through arg max function, i.e. arg max $f(x)$.


## CCS CONCEPTS

- Computing methodologies $\rightarrow$ Supervised learning by classification; Neural networks;


## KEYWORDS

artificial intelligence; artificial neural networks; classification; convolutional neural network; deep learning; deep neural networks; feed-forward neural network; machine learning; rectified linear units; softmax; supervised learning

## 1 INTRODUCTION

A number of studies that use deep learning approaches have claimed state-of-the-art performances in a considerable number of tasks such as image classification[9], natural language processing[15], speech recognition[5], and text classification[18]. These deep learning models employ the conventional softmax function as the classification layer.

However, there have been several studies[2,3, 12] on using a classification function other than Softmax, and this study is yet another addition to those.

In this paper, we introduce the use of rectified linear units (ReLU) at the classification layer of a deep learning model. This approach is the novelty presented in this study, i.e. ReLU is conventionally used as an activation function for the hidden layers in a deep neural network. We accomplish this by taking the activation of the penultimate layer in a neural network, then use it to learn the weight parameters of the ReLU classification layer through backpropagation.

We demonstrate and compare the predictive performance of DL-ReLU models with DL-Softmax models on MNIST[10], FashionMNIST[17], and Wisconsin Diagnostic Breast Cancer (WDBC)[16] classification. We use the Adam[8] optimization algorithm for learning the network weight parameters.

## 2 METHODOLOGY

### 2.1 Machine Intelligence Library

Keras[4] with Google TensorFlow[1] backend was used to implement the deep learning algorithms in this study, with the aid of other scientific computing libraries: matplotlib[7], numpy[14], and scikit-learn[11].

### 2.2 The Datasets

In this section, we describe the datasets used for the deep learning models used in the experiments.

2.2.1 MNIST. MNIST[10] is one of the established standard datasets for benchmarking deep learning models. It is a 10-class classification problem having 60,000 training examples, and 10,000 test cases - all in grayscale, with each image having a resolution of $28 \times 28$.

2.2.2 Fashion-MNIST. Xiao et al. (2017)[17] presented the new Fashion-MNIST dataset as an alternative to the conventional MNIST. The new dataset consists of $28 \times 28$ grayscale images of 70,000 fashion products from 10 classes, with 7,000 images per class.

2.2.3 Wisconsin Diagnostic Breast Cancer (WDBC). The WDBC dataset[16] consists of features which were computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. There are 569 data points in this dataset: (1) 212 - Malignant, and (2) 357 - Benign.

### 2.3 Data Preprocessing

We normalized the dataset features using Eq. 1,

$$
\begin{equation*}
z=\frac{X-\mu}{\sigma} \tag{1}
\end{equation*}
$$

where $X$ represents the dataset features, $\mu$ represents the mean value for each dataset feature $x^{(i)}$, and $\sigma$ represents the corresponding standard deviation. This normalization technique was implemented using the StandardScaler[11] of scikit-learn.

For the case of MNIST and Fashion-MNIST, we employed Principal Component Analysis (PCA) for dimensionality reduction. That is, to select the representative features of image data. We accomplished this by using the PCA[11] of scikit-learn.

### 2.4 The Model

We implemented a feed-forward neural network (FFNN) and a convolutional neural network (CNN), both of which had two different classification functions, i.e. (1) softmax, and (2) ReLU.

2.4.1 Softmax. Deep learning solutions to classification problems usually employ the softmax function as their classification function (last layer). The softmax function specifies a discrete probability distribution for $K$ classes, denoted by $\sum_{k=1}^{K} p_{k}$.

If we take $\mathbf{x}$ as the activation at the penultimate layer of a neural network, and $\theta$ as its weight parameters at the softmax layer, we have $\boldsymbol{o}$ as the input to the softmax layer,

$$
\begin{equation*}
o=\sum_{i}^{n-1} \theta_{i} x_{i} \tag{2}
\end{equation*}
$$

Consequently, we have

$$
\begin{equation*}
p_{k}=\frac{\exp \left(o_{k}\right)}{\sum_{k=0}^{n-1} \exp \left(o_{k}\right)} \tag{3}
\end{equation*}
$$

Hence, the predicted class would be $\hat{y}$

$$
\begin{equation*}
\hat{y}=\underset{i \in 1, \ldots, N}{\arg \max } p_{i} \tag{4}
\end{equation*}
$$

2.4.2 Rectified Linear Units (ReLU). ReLU is an activation function introduced by [6], which has strong biological and mathematical underpinning. In 2011, it was demonstrated to further improve training of deep neural networks. It works by thresholding values at 0 , i.e. $f(x)=\max (0, x)$. Simply put, it outputs 0 when $x<0$, and conversely, it outputs a linear function when $x \geq 0$ (refer to Figure 1 for visual representation).

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-2.jpg?height=407&width=653&top_left_y=1206&top_left_x=281)

Figure 1: The Rectified Linear Unit (ReLU) activation function produces 0 as an output when $x<0$, and then produces a linear with slope of 1 when $x>0$.

We propose to use ReLU not only as an activation function in each hidden layer of a neural network, but also as the classification function at the last layer of a network.

Hence, the predicted class for ReLU classifier would be $\hat{y}$,

$$
\begin{equation*}
\hat{y}=\underset{i \in 1}{\arg \max } \max (0, o) \tag{5}
\end{equation*}
$$

2.4.3 Deep Learning using ReLU. ReLU is conventionally used as an activation function for neural networks, with softmax being their classification function. Then, such networks use the softmax cross-entropy function to learn the weight parameters $\theta$ of the neural network. In this paper, we still implemented the mentioned loss function, but with the distinction of using the ReLU for the prediction units (see Eq. 6). The $\theta$ parameters are then learned by backpropagating the gradients from the ReLU classifier. To accomplish this, we differentiate the ReLU-based cross-entropy function (see Eq. 7) w.r.t. the activation of the penultimate layer,

$$
\begin{equation*}
\ell(\theta)=-\sum y \cdot \log (\max (0, \theta x+b)) \tag{6}
\end{equation*}
$$

Let the input $\mathbf{x}$ be replaced the penultimate activation output $\mathbf{h}$,

$$
\begin{equation*}
\frac{\partial \ell(\theta)}{\partial \mathbf{h}}=-\frac{\theta \cdot y}{\max (0, \theta h+b) \cdot \ln 10} \tag{7}
\end{equation*}
$$

The backpropagation algorithm (see Eq. 8) is the same as the conventional softmax-based deep neural network.

$$
\begin{equation*}
\frac{\partial \ell(\theta)}{\partial \theta}=\sum_{i}\left[\frac{\partial \ell(\theta)}{\partial p_{i}}\left(\sum_{k} \frac{\partial p_{i}}{\partial o_{k}} \frac{\partial o_{k}}{\partial \theta}\right)\right] \tag{8}
\end{equation*}
$$

Algorithm 1 shows the rudimentary gradient-descent algorithm for a DL-ReLU model.

## Algorithm 1: Mini-batch stochastic gradient descent training of neural network with the rectified linear unit (ReLU) as its classification function.

Input: $\left\{x^{(i)} \in \mathbb{R}^{m}\right\}_{i=1}^{n}, \theta$

Output: W

for number of training iterations do

for $i=1,2, \ldots n$ do

$$
\nabla_{\theta}=\nabla_{\theta}-\frac{\theta \cdot y}{\max (0, \theta h+b) \cdot \ln 10}
$$

$$
\theta=\theta-\alpha \cdot \nabla_{\theta} \ell\left(\theta ; x^{(i)}\right)
$$

Any standard gradient-based learning algorithm may be used. We used adaptive momentum estimation (Adam) in our experiments.

In some experiments, we found the DL-ReLU models perform on par with the softmax-based models.

### 2.5 Data Analysis

To evaluate the performance of the DL-ReLU models, we employ the following metrics:

(1) Cross Validation Accuracy \& Standard Deviation. The result of 10 -fold CV experiments.

(2) Test Accuracy. The trained model performance on unseen data.

(3) Recall, Precision, and F1-score. The classification statistics on class predictions.

(4) Confusion Matrix. The table for describing classification performance.

## 3 EXPERIMENTS

All experiments in this study were conducted on a laptop computer with Intel Core(TM) i5-6300HQ CPU @ 2.30GHz x 4, 16GB of DDR3 RAM, and NVIDIA GeForce GTX 960M 4GB DDR5 GPU.

Table 1 shows the architecture of the VGG-like CNN (from Keras[4]) used in the experiments. The last layer, dense_2, used the softmax classifier and ReLU classifier in the experiments.

The Softmax- and ReLU-based models had the same hyperparameters, and it may be seen on the Jupyter Notebook found in the project repository: https://github.com/AFAgarap/relu-classifier.

Table 1: Architecture of VGG-like CNN from Keras[4].

| Layer (type) | Output Shape | Param \# |
| :---: | :---: | :---: |
| conv2d_1 (Conv2D) | (None, 14, 14, 32) | 320 |
| conv2d_2 (Conv2D) | (None, 12, 12, 32) | 9248 |
| max_pooling2d_1 (MaxPooling2) | (None, 6, 6, 32) | 0 |
| dropout_1 (Dropout) | (None, 6, 6, 32) | 0 |
| conv2d_3 (Conv2D) | (None, 4, 4, 64) | 18496 |
| conv2d_4 (Conv2D) | (None, 2, 2, 64) | 36928 |
| max_pooling2d_2 (MaxPooling2) | (None, 1, 1, 64) | 0 |
| dropout_2 (Dropout) | (None, 1, 1, 64) | 0 |
| flatten_1 (Flatten) | (None, 64) | 0 |
| dense_1 (Dense) | (None, 256) | 16640 |
| dropout_3 (Dropout) | (None, 256) | 0 |
| dense_2 (Dense) | (None, 10) | 2570 |

Table 2 shows the architecture of the feed-forward neural network used in the experiments. The last layer, dense_6, used the softmax classifier and ReLU classifier in the experiments.

Table 2: Architecture of FFNN.

| Layer (type) | Output Shape | Param \# |
| :---: | :---: | :---: |
| dense_3 (Dense) | (None, 512) | 131584 |
| dropout_4 (Dropout) | (None, 512) | 0 |
| dense_4 (Dense) | (None, 512) | 262656 |
| dropout_5 (Dropout) | (None, 512) | 0 |
| dense_5 (Dense) | (None, 512) | 262656 |
| dropout_6 (Dropout) | (None, 512) | 0 |
| dense_6 (Dense) | (None, 10) | 5130 |

All models used Adam[8] optimization algorithm for training, with the default learning rate $\alpha=1 \times 10^{-3}, \beta_{1}=0.9, \beta_{2}=0.999$, $\epsilon=1 \times 10^{-8}$, and no decay.

### 3.1 MNIST

We implemented both CNN and FFNN defined in Tables 1 and 2 on a normalized, and PCA-reduced features, i.e. from $28 \times 28$ (784) dimensions down to $16 \times 16$ (256) dimensions.

In training a FFNN with two hidden layers for MNIST classification, we found the results described in Table 3.

Despite the fact that the Softmax-based FFNN had a slightly higher test accuracy than the ReLU-based FFNN, both models had 0.98 for their F1-score. These results imply that the FFNN-ReLU is on par with the conventional FFNN-Softmax.

Figures 2 and 3 show the predictive performance of both models for MNIST classification on its 10 classes. Values of correct prediction in the matrices seem to be balanced, as in some classes, the ReLU-based FFNN outperformed the Softmax-based FFNN, and vice-versa.

In training a VGG-like CNN[4] for MNIST classification, we found the results described in Table 4.

The CNN-ReLU was outperformed by the CNN-Softmax since it converged slower, as the training accuracies in cross validation were
Table 3: MNIST Classification. Comparison of FFNNSoftmax and FFNN-ReLU models in terms of $\%$ accuracy. The training cross validation is the average cross validation accuracy over 10 splits. Test accuracy is on unseen data. Precision, recall, and F1-score are on unseen data.

| Metrics / Models | FFNN-Softmax | FFNN-ReLU |
| :---: | :---: | :---: |
| Training cross validation | $\approx 99.29 \%$ | $\approx 98.22 \%$ |
| Test accuracy | $97.98 \%$ | $97.77 \%$ |
| Precision | 0.98 | 0.98 |
| Recall | 0.98 | 0.98 |
| F1-score | 0.98 | 0.98 |

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-3.jpg?height=726&width=724&top_left_y=824&top_left_x=1161)

Figure 2: Confusion matrix of FFNN-ReLU on MNIST classification.

Table 4: MNIST Classification. Comparison of CNN-Softmax and CNN-ReLU models in terms of \% accuracy. The training cross validation is the average cross validation accuracy over 10 splits. Test accuracy is on unseen data. Precision, recall, and $\mathrm{F} 1$-score are on unseen data.

| Metrics / Models | CNN-Softmax | CNN-ReLU |
| :---: | :---: | :---: |
| Training cross validation | $\approx 97.23 \%$ | $\approx 73.53 \%$ |
| Test accuracy | $95.36 \%$ | $91.74 \%$ |
| Precision | 0.95 | 0.92 |
| Recall | 0.95 | 0.92 |
| F1-score | 0.95 | 0.92 |

inspected (see Table 5). However, despite its slower convergence, it was able to achieve a test accuracy higher than $90 \%$. Granted, it is lower than the test accuracy of $\mathrm{CNN}$-Softmax by $\approx 4 \%$, but further optimization may be done on the CNN-ReLU to achieve an on-par performance with the CNN-Softmax.

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-4.jpg?height=721&width=715&top_left_y=282&top_left_x=247)

Figure 3: Confusion matrix of FFNN-Softmax on MNIST classification.

Table 5: Training accuracies and losses per fold in the 10 -fold training cross validation for CNN-ReLU on MNIST Classification.

| Fold \# | Loss | Accuracy $(\times 100 \%)$ |
| :---: | :---: | :---: |
| 1 | 1.9060128301398311 | 0.32963837901722315 |
| 2 | 1.4318902588488513 | 0.5091768125718277 |
| 3 | 1.362783239967884 | 0.5942213337366827 |
| 4 | 0.8257899198037331 | 0.7495911319797827 |
| 5 | 1.222473526516734 | 0.7038720233118376 |
| 6 | 0.4512576775334098 | 0.8729090907790444 |
| 7 | 0.49083630082824015 | 0.8601818182685158 |
| 8 | 0.34528968995411613 | 0.9032199380288064 |
| 9 | 0.30161443973038743 | 0.912663755545276 |
| 10 | 0.279967466075669 | 0.9171823807790317 |

Figures 4 and 5 show the predictive performance of both models for MNIST classification on its 10 classes. Since the CNN-Softmax converged faster than CNN-ReLU, it has the most number of correct predictions per class.

### 3.2 Fashion-MNIST

We implemented both CNN and FFNN defined in Tables 1 and 2 on a normalized, and PCA-reduced features, i.e. from $28 \times 28$ (784) dimensions down to $16 \times 16$ (256) dimensions. The dimensionality reduction for MNIST was the same for Fashion-MNIST for fair comparison. Though this discretion may be challenged for further investigation.

In training a FFNN with two hidden layers for Fashion-MNIST classification, we found the results described in Table 6.

Despite the fact that the Softmax-based FFNN had a slightly higher test accuracy than the ReLU-based FFNN, both models had

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-4.jpg?height=718&width=708&top_left_y=281&top_left_x=1164)

Figure 4: Confusion matrix of CNN-ReLU on MNIST classification.

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-4.jpg?height=721&width=708&top_left_y=1187&top_left_x=1164)

Figure 5: Confusion matrix of CNN-Softmax on MNIST classification.

0.89 for their F1-score. These results imply that the FFNN-ReLU is on par with the conventional FFNN-Softmax.

Figures 6 and 7 show the predictive performance of both models for Fashion-MNIST classification on its 10 classes. Values of correct prediction in the matrices seem to be balanced, as in some classes, the ReLU-based FFNN outperformed the Softmax-based FFNN, and vice-versa.

In training a VGG-like CNN[4] for Fashion-MNIST classification, we found the results described in Table 7.

Table 6: Fashion-MNIST Classification. Comparison of FFNN-Softmax and FFNN-ReLU models in terms of $\%$ accuracy. The training cross validation is the average cross validation accuracy over 10 splits. Test accuracy is on unseen data. Precision, recall, and F1-score are on unseen data.

| Metrics / Models | FFNN-Softmax | FFNN-ReLU |
| :---: | :---: | :---: |
| Training cross validation | $\approx 98.87 \%$ | $\approx 92.23 \%$ |
| Test accuracy | $89.35 \%$ | $89.06 \%$ |
| Precision | 0.89 | 0.89 |
| Recall | 0.89 | 0.89 |
| F1-score | 0.89 | 0.89 |

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-5.jpg?height=732&width=721&top_left_y=832&top_left_x=236)

Figure 6: Confusion matrix of FFNN-ReLU on Fashion-MNIST classification.

Table 7: Fashion-MNIST Classification. Comparison of CNNSoftmax and CNN-ReLU models in terms of $\%$ accuracy. The training cross validation is the average cross validation accuracy over 10 splits. Test accuracy is on unseen data. Precision, recall, and F1-score are on unseen data.

| Metrics / Models | CNN-Softmax | CNN-ReLU |
| :---: | :---: | :---: |
| Training cross validation | $\approx 91.96 \%$ | $\approx 83.24 \%$ |
| Test accuracy | $86.08 \%$ | $85.84 \%$ |
| Precision | 0.86 | 0.86 |
| Recall | 0.86 | 0.86 |
| F1-score | 0.86 | 0.86 |

Similar to the findings in MNIST classification, the CNN-ReLU was outperformed by the CNN-Softmax since it converged slower, as the training accuracies in cross validation were inspected (see Table 8). Despite its slightly lower test accuracy, the CNN-ReLU

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-5.jpg?height=731&width=708&top_left_y=285&top_left_x=1164)

Figure 7: Confusion matrix of FFNN-Softmax on Fashion-MNIST classification.

had the same F1-score of 0.86 with CNN-Softmax - also similar to the findings in MNIST classification.

Table 8: Training accuracies and losses per fold in the 10 -fold training cross validation for CNN-ReLU for Fashion-MNIST classification.

| Fold \# | Loss | Accuracy $(\times 100 \%)$ |
| :---: | :---: | :---: |
| 1 | 0.7505188028133193 | 0.7309229651162791 |
| 2 | 0.6294445606858231 | 0.7821584302325582 |
| 3 | 0.5530192871624917 | 0.8128293656488342 |
| 4 | 0.468552251288519 | 0.8391494002614356 |
| 5 | 0.4499297190579501 | 0.8409090909090909 |
| 6 | 0.45004472223195163 | 0.8499999999566512 |
| 7 | 0.4096944159454683 | 0.855610110994295 |
| 8 | 0.39893951664539995 | 0.8681098779960613 |
| 9 | 0.37760543597664203 | 0.8637190683266308 |
| 10 | 0.34610279169377683 | 0.8804367606156083 |

Figures 8 and 9 show the predictive performance of both models for Fashion-MNIST classification on its 10 classes. Contrary to the findings of MNIST classification, CNN-ReLU had the most number of correct predictions per class. Conversely, with its faster convergence, CNN-Softmax had the higher cumulative correct predictions per class.

### 3.3 WDBC

We implemented FFNN defined in Table 2, but with hidden layers having 64 neurons followed by 32 neurons instead of two hidden layers both having 512 neurons. For the WDBC classfication, we only normalized the dataset features. PCA dimensionality reduction might not prove to be prolific since WDBC has only 30 features.

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-6.jpg?height=737&width=716&top_left_y=282&top_left_x=244)

Figure 8: Confusion matrix of CNN-ReLU on Fashion-MNIST classification.

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-6.jpg?height=732&width=707&top_left_y=1187&top_left_x=248)

Figure 9: Confusion matrix of CNN-Softmax on Fashion-MNIST classification.

In training the FFNN with two hidden layers of $[64,32]$ neurons, we found the results described in Table 9 .

Similar to the findings in classification using CNN-based models, the FFNN-ReLU was outperformed by the FFNN-Softmax in WDBC classification. Consistent with the CNN-based models, the FFNNReLU suffered from slower convergence than the FFNN-Softmax. However, there was only 0.2 F1-score difference between them. It stands to reason that the FFNN-ReLU is still comparable with FFNN-Softmax.
Table 9: WDBC Classification. Comparison of CNN-Softmax and $\mathrm{CNN}$-ReLU models in terms of \% accuracy. The training cross validation is the average cross validation accuracy over 10 splits. Test accuracy is on unseen data. Precision, recall, and $F 1$-score are on unseen data.

| Metrics / Models | FFNN-Softmax | FFNN-ReLU |
| :---: | :---: | :---: |
| Training cross validation | $\approx 91.21 \%$ | $\approx 87.96 \%$ |
| Test accuracy | $\approx 92.40 \%$ | $\approx 90.64 \%$ |
| Precision | 0.92 | 0.91 |
| Recall | 0.92 | 0.91 |
| F1-score | 0.92 | 0.90 |

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-6.jpg?height=716&width=702&top_left_y=840&top_left_x=1167)

Figure 10: Confusion matrix of FFNN-ReLU on WDBC classification.

Figures 10 and 11 show the predictive performance of both models for WDBC classification on binary classification. The confusion matrices show that the FFNN-Softmax had more false negatives than FFNN-ReLU. Conversely, FFNN-ReLU had more false positives than FFNN-Softmax.

## 4 CONCLUSION AND RECOMMENDATION

The relatively unfavorable findings on DL-ReLU models is most probably due to the dying neurons problem in ReLU. That is, no gradients flow backward through the neurons, and so, the neurons become stuck, then eventually "die". In effect, this impedes the learning progress of a neural network. This problem is addressed in subsequent improvements on ReLU (e.g. [13]). Aside from such drawback, it may be stated that DL-ReLU models are still comparable to, if not better than, the conventional Softmax-based DL models. This is supported by the findings in DNN-ReLU for image classification using MNIST and Fashion-MNIST.

Future work may be done on thorough investigation of DL-ReLU

![](https://cdn.mathpix.com/cropped/2024_06_04_77a75aa0b1b9d2334b60g-7.jpg?height=735&width=713&top_left_y=283&top_left_x=245)

Figure 11: Confusion matrix of FFNN-Softmax on WDBC classification.

models through numerical inspection of gradients during backpropagation, i.e. compare the gradients in DL-ReLU models with the gradients in DL-Softmax models. Furthermore, ReLU variants may be brought into the table for additional comparison.

## 5 ACKNOWLEDGMENT

An appreciation of the VGG-like Convnet source code in Keras[4], as it was the CNN model used in this study.

## REFERENCES

[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2015. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. (2015). http://tensorflow.org/ Software available from tensorflow.org.

[2] Abien Fred Agarap. 2017. A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data. arXiv preprint arXiv:1709.03082 (2017).

[3] Abdulrahman Alalshekmubarak and Leslie S Smith. 2013. A novel approach combining recurrent neural network and support vector machines for time series classification. In Innovations in Information Technology (IIT), 2013 9th International Conference on. IEEE, 42-47.

[4] François Chollet et al. 2015. Keras. https://github.com/keras-team/keras. (2015).

[5] Jan K Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio. 2015. Attention-based models for speech recognition. In Advances in Neural Information Processing Systems. 577-585.

[6] Richard HR Hahnloser, Rahul Sarpeshkar, Misha A Mahowald, Rodney J Douglas, and H Sebastian Seung. 2000. Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. Nature 405, 6789 (2000), 947.

[7] J. D. Hunter. 2007. Matplotlib: A 2D graphics environment. Computing In Science \& Engineering 9, 3 (2007), 90-95. https://doi.org/10.1109/MCSE. 2007.55

[8] Diederik Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[9] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems. $1097-1105$

[10] Yann LeCun, Corinna Cortes, and Christopher JC Burges. 2010. MNIST handwritten digit database. AT\&T Labs [Online]. Available: http://yann. lecun. com/exd$b / m n i s t 2$ (2010).

[11] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12 (2011), 2825-2830.

[12] Yichuan Tang. 2013. Deep learning using linear support vector machines. arXiv preprint arXiv:1306.0239 (2013).

[13] Ludovic Trottier, Philippe Gigu, Brahim Chaib-draa, et al. 2017. Parametric exponential linear unit for deep convolutional neural networks. In Machine Learning and Applications (ICMLA), 2017 16th IEEE International Conference on. IEEE, $207-214$.

[14] Stéfan van der Walt, S Chris Colbert, and Gael Varoquaux. 2011. The NumPy array: a structure for efficient numerical computation. Computing in Science \& Engineering 13, 2 (2011), 22-30.

[15] Tsung-Hsien Wen, Milica Gasic, Nikola Mrksic, Pei-Hao Su, David Vandyke, and Steve Young. 2015. Semantically conditioned lstm-based natural language generation for spoken dialogue systems. arXiv preprint arXiv:1508.01745 (2015).

[16] William H Wolberg, W Nick Street, and Olvi L Mangasarian. 1992. Breast cancer Wisconsin (diagnostic) data set. UCI Machine Learning Repository [http://archive. ics. uci. edu/ml/] (1992).

[17] Han Xiao, Kashif Rasul, and Roland Vollgraf. 2017. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. (2017). arXiv:cs.LG/1708.07747

[18] Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alexander J Smola, and Eduard H Hovy. 2016. Hierarchical Attention Networks for Document Classification.. In HLT-NAACL. 1480-1489.

</end of paper 1>


<paper 2>
# BEYOND TRADITIONAL MAGNETIC RESONANCE PROCESSING WITH ARTIFICIAL INTELLIGENCE * 

Amir Jahangiri ${ }^{\text {à }}$ and Vladislav Orekhov ${ }^{\text {ał }}$<br>andepartment of Chemistry and Molecular Biology, Swedish NMR Centre, University of Gothenburg, Box 465,<br>Gothenburg, 40530, Sweden ${ }^{\dagger}$

May 14, 2024


#### Abstract

Smart signal processing approaches using Artificial Intelligence are gaining momentum in NMR applications. In this study, we demonstrate that AI offers new opportunities beyond tasks addressed by traditional techniques. We developed and trained several artificial neural networks in our new toolbox Magnetic Resonance with Artificial intelligence (MR-Ai) to solve three "impossible" problems: quadrature detection using only Echo (or Anti-Echo) modulation from the traditional Echo/Anti-Echo scheme; accessing uncertainty of signal intensity at each point in a spectrum processed by any given method; and defining a reference-free score for quantitative access of NMR spectrum quality. Our findings highlight the potential of AI techniques to revolutionize NMR processing and analysis.


Keywords NMR $\cdot$ NUS $\cdot \mathrm{AI} \cdot \mathrm{WNN} \cdot$ Quadrature detection[^0]

NMR spectroscopy is a powerful analytical technique widely used to acquire atomic-level information about molecular structure, dynamics, and interactions [1,2]. To derive meaningful insights from the acquired spectra, NMR data processing plays a vital role. Artificial Intelligence (AI), and specifically Deep Learning (DL), presents a compelling alternative to traditional methods in NMR processing [3]. Although early demonstrations of machine learning in NMR date back to the 1970s [4], practical applications have evolved significantly with recent advancements in algorithms and computer hardware. In most cases, DL in NMR data processing focuses on surpassing the existing algorithmic techniques for fast and high-quality solving of traditional tasks such as spectra reconstruction from Non-Uniformly Sampled (NUS) time domain signals [5]-8], virtual homo-decoupling [8--11], spectra denoising [12, 13], and automating peak picking [14, 15]. In this study, we address an intriguing question of whether DL can go beyond the traditional problems and offer new ways of spectra processing and analysis [16], and possibly give us insights for designing new signal processing algorithms [8].

We demonstrate a Magnetic Resonance processing with Artificial intelligence (MR-Ai) solution for the seemingly impossible task of recovering a high-quality spectrum from an incomplete phase-modulated quadrature detection experiment, where only one of the two $\mathrm{P}$ - and $\mathrm{N}$-type parts of phase-modulated quadrature detection experiments is available. Furthermore, we show that MR-Ai is able to perform valuable statistical analyses of spectra reconstructed by other methods and thus provides a new reference-free metric of the spectrum's quality.

Phase-twist lineshape in incomplete quadrature detection as a pattern recognition problem: Traditionally, in multidimensional ( $n \mathrm{D}$ ) NMR experiments, frequency discrimination and obtaining pure, absorptive phase signals rely on quadrature detection. This involves acquiring two data points per time increment and per spectral dimension. For a 2D experiment, where the signal evolves in two-time dimensions $t_{1}$ and $t_{2}$, the amplitude-modulated quadrature detection [17, 18] is implemented by acquiring two separate data sets in the form of cosine and sine modulation:

$$
\begin{aligned}
& \text { Data_set }_{1}\left(\cos _{-} \text {modulated }\right): \cos \left(\Omega_{1} t_{1}\right) \exp \left(i \Omega_{2} t_{2}\right) \\
& \text { Data_set }_{2}(\text { sin_modulated }): \sin \left(\Omega_{1} t_{1}\right) \exp \left(i \Omega_{2} t_{2}\right)
\end{aligned}
$$

where $\Omega_{n}$ is the signal frequency in the $n$th dimension.

In contrast, the phase-modulated data usually obtained from gradient coherence order selection experiments are encoded with frequency as either Echo (P-type data) or Anti-Echo (N-type data) coherence:

$$
\begin{aligned}
& \text { Data_set }_{1}\left(P_{-} \text {type }\right): \exp \left(+i \Omega_{1} t_{1}\right) \exp \left(i \Omega_{2} t_{2}\right) \\
& \text { Data_set }_{2}\left(N_{-} \text {type }\right): \exp \left(-i \Omega_{1} t_{1}\right) \exp \left(i \Omega_{2} t_{2}\right)
\end{aligned}
$$

Individually, each of these datasets produces frequencydiscriminated spectra but exhibits a phase-twist lineshape of the peaks (Fig. 1.a), which is not amenable for normal analysis. Until now, it has been understood that the only way to obtain pure absorptive phase signals in Echo-AntiEcho experiments is by using both $\mathrm{P}$ - and $\mathrm{N}$-type data [19-21]. In this work, we demonstrate that MR-Ai can effectively recognize the twisted lineshapes and convert them into the pure absorption form (Fig. 1.b). To the best of our knowledge, none of the traditional methods demonstrated this capability so far.
![](https://cdn.mathpix.com/cropped/2024_06_04_ba858378e2b4ea3de27dg-02.jpg?height=732&width=804&top_left_y=627&top_left_x=1076)

Figure 1: Illustration of the Echo and normal spectrum. (a) Echo spectrum with a phase-twist lineshape, (b) a normal spectrum with a pure, absorptive phase in the frequency domain, and (c and d) their corresponding Virtual Echo presentations in the time domain respectively. In the figures, $P$ and $N$ represent the P-type and N-type data sets, while $\widetilde{P}$ and $\widetilde{N}$ indicate the time reverse and conjugation of P-type and N-type data sets respectively - procedures for transition between the presentations are indicated by arrows.

In our recent publication, we introduced a DNN architecture called WNN, specifically designed to grasp 1D patterns over the entire NMR spectrum in the frequency domain, such as specific patterns of NUS aliasing artifacts and peak multiples in homo-decoupling [8]. Here, we utilize an updated version of our WNN architecture (see Appendix A for more details) capable of capturing 2D patterns, including the phase twisted peaks associated with the $\mathrm{P}$ - (or $\mathrm{N}$-) type data, as a pattern recognition problem in the frequency domain. Fig. 2 a demonstrates the excellent performance of Echo and Anti-Echo reconstructions of ${ }^{1} \mathrm{H}-{ }^{15} \mathrm{~N}$ correlation spectra by MR-Ai on MALT1 ( $\left.45 \mathrm{kDa}\right)$ protein. Similar results for Ubiquitin (7 kDa), Azurin (14 $\mathrm{kDa}$ ), and Tau (disordered, 441 amino acids) are found in Appendix B Figures B. 1 to B. 4

Echo and Anti-Echo reconstruction as a data completion problem: Figure 11illustrates that the task to rectify the phase twist problem encountered in Echo (or Anti-

![](https://cdn.mathpix.com/cropped/2024_06_04_ba858378e2b4ea3de27dg-03.jpg?height=740&width=808&top_left_y=237&top_left_x=236)

Figure 2: Performance of Echo and Anti-Echo reconstruction by using MR-Ai and CS on real data. (a) $2 \mathrm{D}$ ${ }^{1} \mathrm{H}-{ }^{15} \mathrm{~N}$ - TROSY spectra of MALT1 [22] Echo reconstruction using MR-Ai with predicted uncertainty in pink color - The insets show zooming part and corresponding reference with the actual absolute error between reference and reconstruction in red color. Bar graphs (b) represent $R M S D$ as a traditional reference-based evaluation metric and Boxplots (c) represent normalized uncertainty as the intelligent reference-free evaluation metric for comparison reconstructed spectra using MR-Ai and CS for Malt as described in the Appendix A section for additional details.

Echo) spectra can be viewed as a specific case of NUS reconstruction. Indeed, Fig. 1 c and Fig. 1.d show the time domain equivalents of the twisted (Fig. 1.a) and absorptive (Fig. 1 b) line shapes. The time domain presentation, obtained by the two-dimensional inverse Fourier Transform of the real signal shown in Fig. $1, \mathrm{a}, 1, \mathrm{~b}$ and in the following called the Virtual Echo (VE) [23], clearly shows the roles of the P-type and N-type data. There are four regions defined by the signs of $t_{1}$ and $t_{2}$. P-type data corresponds to the upper-right region where both $t_{1}$ and $t_{2}$ are positive $(P)$, and the time-reversed conjugated $\mathrm{P}$ type data corresponds to the lower-left region where both $t_{1}$ and $t_{2}$ are negative $(\widetilde{P})$. Similarly, N-type data corresponds to the region where $t_{1}$ is negative and $t_{2}$ is positive $(N)$, while the time-reversed and conjugated $\mathrm{N}$-type data is in the region where $t_{1}$ is positive and $t_{2}$ is negative $(\widetilde{N})$. Therefore, rectification of the twisted line shape in the frequency domain is equivalent to completing the missing half of the signal in the VE time domain. The problem is similar to spectral reconstruction from NUS data and can be performed using the Compressed Sensing Iterative Soft Thresholding algorithm (CS-IST) [24, 25], a representative traditional NUS reconstruction technique. The CS algorithm maximizes sparsity of the spectrum, thus filling in the missing data to produce the most compact absorptive form of the signal while suppressing the wider and less sparse dispersive features of the twisted signal. Previously, we demonstrated that for performing data completion in the NUS spectra reconstruction, the MR-Ai utilized pattern recognition [8]. In this work, we point out that the seemingly pattern reconstruction problem of rectifying the Echo (or Anti-Echo) twisted line shapes is akin to the data completion in the time domain and can be addressed by traditional algorithms such as CS.

Both reconstruction methods, MR-Ai and CS-IST, reproduce the spectrum with high quality using either P- or N-type data. Fig. 2b (and Fig. A.3) shows a simple spectra quality metric point-to-point $R M S D$ (and $R_{2}^{s}$ ) between the reconstruction and reference spectra for MALT1 (Tau, Azurin, and Ubiquitin). It was demonstrated that the results from the $R M S D$ (and $R_{2}^{s}$ ) metric correspond well to the results obtained using a more extended and advanced set of the NUScon metrics [8, 26]. With its lower $R M S D$ (and higher $R_{2}^{s}$ ), the MR-Ai displays visibly better results compared to the CS for both Echo and Anti-Echo reconstruction. The even better quality score obtained for the reconstruction from time equivalent $50 \%$ NUS experiment (Fig. 2.b) indicates that well-randomized NUS is a better time-saving strategy than acquiring only $\mathrm{N}$ - or P-type data. Fig. 2 b shows that the quality scores for the Anti-Echo reconstruction are higher than for the Echo regardless of the reconstruction method MR-Ai or CS, which is also reproduced for three other proteins shown in Fig. A.3. At first glance, this is a surprising result since from the theory we expect the quality of the reconstructions to be the same for Echo and Anti-Echo. However, in practice, we should note that these are separate experiments with somewhat different pulse sequences [27], which may lead to imbalances between the two spectra. We reproduced the result in simulations (Fig. A.5 b) where the Echo signal had a somewhat lower amplitude than the Anti-Echo. Then, the better result for the Anti-Echo can be explained by the residual unbalanced contribution of the Anti-Echo part in the traditionally processed reference spectrum. This result underlines the value of the reconstruction approach from the individual Echo or Anti-Echo parts in cases of imbalance between the two or if only one can be practically obtained.

Predicting uncertainty of spectrum intensities with DNN: For any physical measurement, such as the intensity at a point in a reconstructed spectrum, estimating the error is equally important for quantitative analysis and distinguishing true signals from noise and artifacts. This task is particularly challenging when using nonlinear processing methods such as CS and DNN, since the RMSD of the baseline noise can no longer be used as a reliable error estimate. Traditionally, this problem is solved by the brute force approach of repeated measures or postexperiment resampling of the data [26, 28]. DNN offers a much more efficient alternative [29]. It is possible to train a network to predict the quality of the results generated by any method [30] by employing the negative log-likelihood (NLL) as the loss function during the training stage.

$$
\begin{aligned}
N L L\left(y_{i} \mid \mu_{i}, \sigma_{i}\right)= & -\log \left(P D F\left(y_{i} \mid \mu_{i}, \sigma_{i}\right)\right) \\
& \approx \frac{\left(y_{i}-\mu_{i}\right)^{2}}{2 \sigma_{i}^{2}}+\log \left(\sigma_{i}\right)
\end{aligned}
$$

where for point $i$ in the spectrum, $y_{i}$, represents the ground truth value; $\mu_{i}$ is the value produced by the applied method, and $\sigma_{i}$ is the uncertainty represented as the standard deviation of a normal distribution. During training, $\sigma_{i}$ is learned, while $y_{i}$ and $\mu_{i}$ are known.

![](https://cdn.mathpix.com/cropped/2024_06_04_ba858378e2b4ea3de27dg-04.jpg?height=363&width=807&top_left_y=740&top_left_x=239)

Figure 3: Illustration of training and predicting MR-Ai for estimation of the uncertainty of the reconstruction generated by any method

The modified MR-Ai architecture employing the NLL as the loss function is illustrated in Fig. 3. In this work, we trained three MR-Ai to predict intensity uncertainties in the Echo (or Anti-Echo) spectra reconstructed by MR-Ai and CS, as well as NUS spectra reconstructed by CS. In Fig. 2 a (and Figures B.5 to B.8), the pink color represents the estimated uncertainty with a $95 \%$ confidence interval (CI) overlaid on the reconstructed spectrum, while the true error is superimposed on the reference spectrum in red for visual comparison.

Reference-free spectrum quality score, pSQ: The predicted uncertainty can be used as a reference-free score of the spectrum quality, akin to the predicted protein structure accuracy pTM-score calculated by AlphaFold [31]. In Fig. 2.c (and Fig. A.4, box plots display the estimated normalized sigma for all points with the mean (green triangles) and median (orange bars). The box plots correlate well with the reference-based scores shown in Fig. 2 b. Moreover, the box plot sores for the Echo and Anti-Echo reconstructions are nearly the same, as they should be, whereas the reference-based scores depicted in Fig. 2.b wrongly prefer the Anti-Echo because of the biased reference as described above.

In this work, we introduce the MR-Ai toolbox, which offers intelligent NMR processing capabilities beyond traditional techniques. The reconstruction of spectra using the incomplete Echo/Anti-Echo quadrature detection pair can be advantageous for saving experimental time and in cases where obtaining a complete and intensity-balanced signal for traditional quadrature is problematic. The predicted uncertainties of spectral intensities and quantitative reference-free spectrum quality metric will aid in the development of new spectrum processing algorithms and may become a crucial component in methods utilizing targeted acquisitions approaches [32, 33]. Our results demonstrate the potential of AI to expand the scope of traditional NMR signal processing and analysis.

## Acknowledgements

The work was supported by the Swedish Research Council grants to 2019-03561 and 2023-03485 to V.O. This study used NMRbox: National Center for Biomolecular NMR Data Processing and Analysis, a Biomedical Technology Research Resource (BTRR), which is supported by NIH grant P41GM111135 (NIGMS).

## References

[1] J. Cavanagh, W. J. Fairbrother, A. G. Palmer III, and N. J. Skelton, Protein NMR spectroscopy: principles and practice. Academic press, 1996.

[2] T. D. Claridge, High-resolution NMR techniques in organic chemistry, vol. 27. Elsevier, 2016.

[3] D. Chen, Z. Wang, D. Guo, V. Orekhov, and X. Qu, "Review and prospect: Deep learning in nuclear magnetic resonance spectroscopy," Chemistry - A European Journal, vol. 26, no. 46, pp. 10391-10401, 2020.

[4] C. Reilly and B. Kowalski, "Nuclear magnetic resonance spectral interpretation by pattern recognition," The Journal of Physical Chemistry, vol. 75, no. 10, pp. 1402-1411, 1971.

[5] X. Qu, Y. Huang, H. Lu, T. Qiu, D. Guo, T. Agback, V. Orekhov, and Z. Chen, "Accelerated nuclear magnetic resonance spectroscopy with deep learning," Angewandte Chemie, vol. 132, no. 26, pp. 1038310386, 2020.

[6] D. Hansen, "Using deep neural networks to reconstruct non-uniformly sampled $\mathrm{nmr}$ spectra," Journal of Biomolecular NMR, vol. 73, 112019.

[7] G. Karunanithy and D. Hansen, "Fid-net: A versatile deep neural network architecture for $\mathrm{nmr}$ spectral reconstruction and virtual decoupling," Journal of Biomolecular NMR, vol. 75, 052021.

[8] A. Jahangiri, X. Han, D. Lesovoy, T. Agback, P. Agback, A. Achour, and V. Orekhov, "Nmr spectrum reconstruction as a pattern recognition problem," Journal of Magnetic Resonance, vol. 346, p. 107342, 2023.

[9] G. Karunanithy, H. W. Mackenzie, and D. F. Hansen, "Virtual homonuclear decoupling in direct detection nuclear magnetic resonance experiments using deep neural networks," Journal of the American Chemical Society, vol. 143, no. 41, pp. 16935-16942, 2021.

Magnetic Resonance processing with Artificial intelligence (MR-Ai)

[10] K. Kazimierczuk, P. Kasprzak, P. S. Georgoulia, I. Matečko-Burmann, B. M. Burmann, L. Isaksson, E. Gustavsson, S. Westenhoff, and V. Y. Orekhov, "Resolution enhancement in nmr spectra by deconvolution with compressed sensing reconstruction," Chemical Communications, vol. 56, no. 93, pp. 14585-14588, 2020.

[11] T. Qiu, A. Jahangiri, X. Han, D. Lesovoy, T. Agback, P. Agback, A. Achour, X. Qu, and V. Orekhov, "Resolution enhancement of $\mathrm{nmr}$ by decoupling with the low-rank hankel model," Chemical Communications, vol. 59, no. 36, pp. 5475-5478, 2023.

[12] H. Lee and H. Kim, "Intact metabolite spectrum mining by deep learning in proton magnetic resonance spectroscopy of the brain," Magnetic Resonance in Medicine, 032019.

[13] D. Chen, W. Hu, H. Liu, Y. Zhou, T. Qiu, Y. Huang, Z. Wang, M. Lin, L. Lin, Z. Wu, et al., "Magnetic resonance spectroscopy deep learning denoising using few in vivo data," IEEE Transactions on Computational Imaging, 2023.

[14] P. Klukowski, M. Augoff, M. Zieba, M. Drwal, A. Gonczarek, and M. Walczak, "Nmrnet: A deep learning approach to automated peak picking of protein nmr spectra," Bioinformatics (Oxford, England), vol. 34, 032018.

[15] D.-W. Li, A. L. Hansen, L. Bruschweiler-Li, C. Yuan, and R. Brüschweiler, "Fundamental and practical aspects of machine learning for the peak picking of biomolecular nmr spectra," Journal of Biomolecular NMR, pp. 1-9, 2022.

[16] V. K. Shukla, G. T. Heller, and D. F. Hansen, "Biomolecular nmr spectroscopy in the era of artificial intelligence," Structure, 2023.

[17] D. States, R. Haberkorn, and D. Ruben, "A twodimensional nuclear overhauser experiment with pure absorption phase in four quadrants," Journal of Magnetic Resonance (1969), vol. 48, no. 2, pp. 286-292, 1982.

[18] D. Marion, M. Ikura, R. Tschudin, and A. Bax, "Rapid recording of $2 \mathrm{~d} \mathrm{nmr}$ spectra without phase cycling. application to the study of hydrogen exchange in proteins," Journal of Magnetic Resonance (1969), vol. 85, no. 2, pp. 393-399, 1989.

[19] A. L. Davis, J. Keeler, E. D. Laue, and D. Moskau, "Experiments for recording pure-absorption heteronuclear correlation spectra using pulsed field gradients," Journal of Magnetic Resonance (1969), vol. 98, no. 1, pp. 207-216, 1992.

[20] G. Kontaxis, J. Stonehouse, E. Laue, and J. Keeler, "The sensitivity of experiments which use gradient pulses for coherence-pathway selection," Journal of Magnetic Resonance, Series A, vol. 111, no. 1, pp. 7076, 1994 .
[21] M. Bostock, D. Holland, and D. Nietlispach, "Improving resolution in multidimensional $\mathrm{nmr}$ using random quadrature detection with compressed sensing reconstruction," Journal of Biomolecular NMR, vol. 68, pp. 67-77, 2017.

[22] X. Han, M. Levkovets, D. Lesovoy, R. Sun, J. Wallerstein, T. Sandalova, T. Agback, A. Achour, P. Agback, and V. Y. Orekhov, “Assignment of ivl-methyl side chain of the ligand-free monomeric human malt1 paracaspase-ig13 domain in solution," Biomolecular NMR Assignments, vol. 16, no. 2, pp. 363-371, 2022.

[23] M. Mayzel, K. Kazimierczuk, and V. Y. Orekhov, "The causality principle in the reconstruction of sparse nmr spectra," Chemical Communications, vol. 50, no. 64, pp. 8947-8950, 2014.

[24] K. Kazimierczuk and V. Y. Orekhov, "Accelerated nmr spectroscopy by using compressed sensing," Angewandte Chemie International Edition, vol. 50, no. 24, pp. 5556-5559, 2011.

[25] D. J. Holland, M. J. Bostock, L. F. Gladden, and D. Nietlispach, "Fast multidimensional $\mathrm{nmr}$ spectroscopy using compressed sensing," Angewandte Chemie International Edition, vol. 50, no. 29, pp. 6548-6551, 2011.

[26] Y. Pustovalova, F. Delaglio, D. L. Craft, H. Arthanari, A. Bax, M. Billeter, M. J. Bostock, H. Dashti, D. F. Hansen, S. G. Hyberts, et al., "Nuscon: a communitydriven platform for quantitative evaluation of nonuniform sampling in $\mathrm{nmr}$," Magnetic Resonance, vol. 2, no. 2, pp. 843-861, 2021.

[27] T. J. Horne and G. A. Morris, "P-type gradientenhanced cosy experiments show lower $t 1$ noise than n-type," Magnetic resonance in chemistry, vol. 35, no. 10, pp. 680-686, 1997.

[28] M. Mayzel, A. Ahlner, P. Lundström, and V. Y. Orekhov, "Measurement of protein backbone 13 co and $15 \mathrm{n}$ relaxation dispersion at high resolution," Journal of Biomolecular NMR, vol. 69, pp. 1-12, 2017.

[29] M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi, U. R. Acharya, V. Makarenkov, and S. Nahavandi, "A review of uncertainty quantification in deep learning: Techniques, applications and challenges," Information Fusion, vol. 76, pp. 243297, 2021.

[30] G. Scalia, C. A. Grambow, B. Pernici, Y.-P. Li, and W. H. Green, "Evaluating scalable uncertainty estimation methods for deep learning-based molecular property prediction," Journal of chemical information and modeling, vol. 60, no. 6, pp. 2697-2717, 2020 .

[31] J. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool,

R. Bates, A. Žídek, A. Potapenko, et al., "Highly accurate protein structure prediction with alphafold," Nature, vol. 596, no. 7873, pp. 583-589, 2021.

[32] V. A. Jaravine and V. Y. Orekhov, "Targeted acquisition for real-time $\mathrm{nmr}$ spectroscopy," Journal of the American Chemical Society, vol. 128, no. 41, pp. 13421-13426, 2006.

[33] L. Isaksson, M. Mayzel, M. Saline, A. Pedersen, J. Rosenlöw, B. Brutscher, B. G. Karlsson, and V. Y. Orekhov, "Highly efficient nmr assignment of intrinsically disordered proteins: application to b-and $t$ cell receptor domains," PLos one, vol. 8, no. 5, p. e62947, 2013.

[34] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, "Wavenet: A generative model for raw audio," 2016.

[35] A. F. Agarap, "Deep learning using rectified linear units (relu)," arXiv preprint arXiv:1803.08375, 2018.

[36] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, et al., "Tensorflow: Large-scale machine learning on heterogeneous distributed systems," arXiv preprint arXiv:1603.04467, 2016.

[37] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.

[38] M. W. Maciejewski, A. D. Schuyler, M. R. Gryk, I. I. Moraru, P. R. Romero, E. L. Ulrich, H. R. Eghbalnia, M. Livny, F. Delaglio, and J. C. Hoch, "Nmrbox: a resource for biomolecular nmr computation," Biophysical journal, vol. 112, no. 8, pp. 1529-1534, 2017.

[39] F. Delaglio, S. Grzesiek, G. W. Vuister, G. Zhu, J. Pfeifer, and A. Bax, "Nmrpipe: a multidimensional spectral processing system based on unix pipes," Journal of biomolecular NMR, vol. 6, pp. 277-293, 1995.

[40] J. J. Helmus and C. P. Jaroniec, "Nmrglue: an open source python package for the analysis of multidimensional nmr data," Journal of biomolecular NMR, vol. 55, pp. 355-367, 2013.

[41] S. G. Hyberts, K. Takeuchi, and G. Wagner, "Poissongap sampling and forward maximum entropy reconstruction for enhancing the resolution and sensitivity of protein nmr data," Journal of the American Chemical Society, vol. 132, no. 7, pp. 2145-2147, 2010.

[42] P. S. Brzovic, A. Lissounov, D. E. Christensen, D. W. Hoyt, and R. E. Klevit, "A ubch5/ubiquitin noncovalent complex is required for processive brca1directed ubiquitination," Molecular cell, vol. 21, no. 6, pp. 873-880, 2006.
[43] D. M. Korzhnev, B. G. Karlsson, V. Y. Orekhov, and M. Billeter, "Nmr detection of multiple transitions to low-populated states in azurin," Protein Science, vol. 12, no. 1, pp. 56-65, 2003.

[44] S. Unnerståle, M. Nowakowski, V. Baraznenok, G. Stenberg, J. Lindberg, M. Mayzel, V. Orekhov, and T. Agback, "Backbone assignment of the malt1 paracaspase by solution nmr," Plos one, vol. 11, no. 1, p. e0146496, 2016.

[45] D. M. Lesovoy, P. S. Georgoulia, T. Diercks, I. Matečko-Burmann, B. M. Burmann, E. V. Bocharov, W. Bermel, and V. Y. Orekhov, "Unambiguous tracking of protein phosphorylation by fast high-resolution fosy nmr," Angewandte Chemie International Edition, vol. 60, no. 44, pp. 23540-23544, 2021.
</end of paper 2>


