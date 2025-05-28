<paper 0>
# Churn Prediction with Sequential Data and Deep Neural Networks A Comparative Analysis* 

C. Gary Mena ${ }^{1} \quad$ Arno De Caigny ${ }^{2,3} \quad$ Kristof Coussement ${ }^{2,3}$<br>Koen W. De Bock ${ }^{4} \quad$ Stefan Lessmann ${ }^{1}$<br>${ }^{1}$ School of Business and Economics, Humboldt University of Berlin<br>${ }^{2}$ IÉSEG School of Management<br>${ }^{3}$ LEM - CNRS 9221<br>${ }^{4}$ Audencia Business School

September 2019


#### Abstract

Off-the-shelf machine learning algorithms for prediction such as regularized logistic regression cannot exploit the information of time-varying features without previously using an aggregation procedure of such sequential data. However, recurrent neural networks provide an alternative approach by which time-varying features can be readily used for modeling. This paper assesses the performance of neural networks for churn modeling using recency, frequency, and monetary value data from a financial services provider. Results show that RFM variables in combination with LSTM neural networks have larger top-decile lift and expected maximum profit metrics than regularized logistic regression models with commonly-used demographic variables. Moreover, we show that using the fitted probabilities from the LSTM as feature in the logistic regression increases the out-of-sample performance of the latter by 25 percent compared to a model with only static features.


[^0]
## 1 Introduction

One of the core tasks in customer relationship management is customer retention. Predicting the probability that a customer will churn has an important role in the design and implementation of customer retention strategies, especially in saturated industries like the financial or telecommunications sectors. One of the reasons is that in such industries the potential customer base of the (relevant) market is close to be fully allocated between the different competitors. Therefore, the value associated with customer retention tends to be larger than the value obtained from acquiring new customers, which in turn fosters the development of churn management strategies Hadden et al. (2007).

The improvement of the predictive performance of churn models is important for targeting and the design of marketing strategies that aim to reduce churn. With the increase in computational power and new data sources, deep learning methods have exploited previously unused features such as social network (Óskarsdóttir et al. (2017)) or textual information (De Caigny et al. (2019)) to enhance the predictive performance of customer churn models. However, sequential features are an important group of features that cannot be readily incorporated in off-the-shelf machine learning algorithms without suitable transformations. For example, Ładyżyński et al. (2019) use moving averages as well as recent lagged values of sequential data as features for random forest and neural networks. Although there are attempts to incorporate sequential information directly in churn prediction models as in Chen et al. (2012), the documented performance of recurrent neural networks provides an alternative approach that can enhance the predictive power of churn models by directly exploiting the sequential nature of behavioral features.

In direct marketing, for example, three sequential variables that can impact the predictive power of churn models are recency, frequency, and monetary variables (henceforth RFM variables). As defined in general terms in Zhang et al. (2014), recency is the time period since the customer's last purchase to the time of data collection, frequency is the number of purchases made by individual customers within a specified time period, and the monetary value variable represents the amount of money a customer spent during a specified time period. Intuitively, in contractual markets RFM variables help to characterize the relative behavior of customers with the firm over time. Thus, these variables help to determine which customer are more prone to churn since the customer might alter her behavior during the period when she is about to churn in contrast to non-churning customers.

The main questions that we aim to answer in this paper are i) what is the relative performance of LSTM models with RFM variables compared to off-the-shelf models with static features, and ii) what is the best way to incorporate sequential information into static models. Accordingly, we contribute to the empirical churn modeling literature by assessing the predictive performance of Long-Short Term Memory architectures to model RFM variables and by providing an empirical application using data from an European provider of financial services. Concretely, we show that i) the predictive performance of RNNs and RFM data is higher than the performance of regularized logistic regression without RFM data, and ii) using RNNs to summarize the information contained in RFM variables is an effective alternative to incorporate the latter in off-the-shelf machine learning methods.

The document is organized as follows. In section 2 we provide a succinct literature review. In section 3 we describe the data and our experimental approach. Section 4 reports the main results of the document, and the final section concludes.

## 2 Literature Review

The literature on empirical churn modeling in static settings using cross-sections of data is well studied and Verbeke et al. (2012) provide an extensive benchmark of classification techniques using several real-life data sets. Alternative techniques to analyze churn modeling includes survival analysis (van den Poel and Larivière (2004)) however we focus on classification analysis given the data that we have available.

When there are time-varying features then different aggregation procedure are available so that their information can be used with machine learning classification methods (see Wei and Chiu (2002), and Song et al. (2006)). The reason why they cannot be directly used is that a majority of the classification methods requires one observation per customer but when there are time-varying features one can usually follow the behavior of the same customer over time and the estimation classification methods cannot directly exploit this type of information. One important exception is the work of Chen et al. (2012) who explicitly consider and exploit the information from longitudinal data for customer churn prediction. Specifically, they propose a novel framework called hierarchical multiple kernel support vector machine that without transformation of time-varying features improves the performance of customer churn prediction compared to SVM and other classification algorithms in terms of AUC and Lift using data sets from the Telecom, and other nonfinancial industries.

The rising popularity of deep neural networks methods for sequential data has fostered an increase in their applications to churn modeling as the overview in Table 1 shows. To synthesize the results from this literature, Martins (2017) shows that the performance of LSTM taking into account the time-varying features performs as well as aggregating this information using their average and a random forest algorithm. Tan et al. (2018) propose a network architecture that combines CNN and LSTM networks that outperforms them individually as well as other algorithms that do not use sequential data in terms of AUC, precision-recall, F1-score, and Mathews correlation coefficient. Wangperawong et al. (2016) process the time-varying features such that they can be used as images and then apply a CNN architecture, but offer no comparative performance of their approach. In a similar vein, Zaratiegui et al. (2015) encodes the sequential information as images and then applies a CNN and shows that after encoding the CNN performs better than random forests and extreme gradient boost classifiers. Finally, the study of Zhou et al. (2019) combines different network architectures to leverage the sequential data and show that this combination outperforms CNN, LSTM and classifiers that do not use the time-varying information like random forest and extreme gradient boosting.

Table 1: Previous Studies on Customer Churn Prediction Using Longitudinal Data and Neural Networks

| Study | Neural Network Architecture | Sample Size (customers) | Industry | Type | Sequential Features |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Tan et al. (2018) | BLA $(=$ LSTM + CNN $)$ | $120 x 10^{3}$ and $156 x 10^{3}$ | MOOC, Online Services | IECD proceeding | Subscription characteristics |
| Wangperawong et al. (2016) | CNN | $1 x 10^{6}$ | Telecom | Working Paper | Data, voice, sms usage |
| Zaratiegui et al. (2015) | CNN | $132 x 10^{3}$ | Telecom | Working Paper | Call record, Topup |
| Zhou et al. (2019) | DLCNN $(=\mathrm{LSTM}+\mathrm{CNN})$ | $1 x 10^{6}$ | Music Streaming | Conf. Proceedings | Transaction and log activity |

To conclude this section, note that the last three columns show that the previous works that look to improve the performance of churn models using time-varying information is still in an early stage. Moreover, different industries offer different types of time-varying features which a priori does not tell us whether incorporating such type of information leads to a higher performance of the models. Thus, this paper makes a contribution by
assessing the performance of different classification algorithms and alternative ways of incorporating the time-varying features in the financial services industry. Specifically, we use recency, frequency, and monetary value longitudinal information, in addition to demographic data, to evaluate the prediction of recurrent neural networks as well as the well-established logistic regression that require aggregating the information for training the model.

## 3 Data and Experimental Setup

### 3.1 Data: Definitions and Processing

An important step in our analysis is to choose a suitable definition of the target variable. The information available in the data set makes it possible to construct churn measures using data related to contract closure of all products. ${ }^{1}$ Concretely, the observation period that we use to determine whether a customer churns starts in 01/04/2018 and we use a time window of 12 months. ${ }^{2}$ Thus, a customer is labeled as having churned if she closes all contracts during 01/04/2018 and 01/04/2019. Although we could define churn using a six month period, the number of people who actually churned would be greatly decreased which could potentially impact the predictive performance of the estimators. Moreover, a few customers are characterized as churners using the six month definition but are clients again under the 12 month definition, which implies that by using a 12 month the observed churn behavior is not a temporary phenomenon but rather permanent.

Regarding the features, the dataset contains i) demographic variables such as age, gender, social and marital status, ii) customer-company characteristics such as length of relationship, and iii) variables related to customer behavior other than RFM variables like customer-company communication information. In the raw data, features other than RFM data were already normalized in the 0-1 interval and we refer the interested reader to De Caigny et al. (2019) for details about processing of missing values and outliers. Note that we constrain the observations to customers that have available RFM and target information for each of the 36 months before the 12 month observation period, and focus on the treatment of time-varying features.

While recurrent and convolutional neural networks can make use of time-varying features, off-the-shelf methods like regularized logistic regression or random forests require aggregating the time-varying information into a single observation. Hence, the treatment of such variables plays an important role behind the results of the experiments. One option to aggregate the time-varying variables is taking their average value per customer, expecting that differences in the levels between churners and non-churners is salient after the transform. A second transformation is by taking the average of first differences of the sequential data. Assuming the data is sorted by customer and date, this effectively reduces to taking the difference between the last and first observation for each customer and dividing by the number of available periods ${ }^{3}$. For our final aggregation procedure we use the last six values of each RFM feature, normalized by its average value of the last quarter. The motivation is that departures from the average behavior are more important[^1]than the levels of the observed RFM data. Notice that this average value is not estimated as a moving average.

Figure 1 presents descriptive statistics of the mean RFM data evolution over time by type of customer. Panel (a) clearly shows that the average value of the frequency feature decreases for churners as we approach the observation window. Such behavior of the feature should be well captured by our aggregation procedures and thus provide a good comparison when compared to the results without aggregation from the neural networks. Panel (b) reveals that churners have on average a lower recency value compared to non-churners. Finally, panel (c) shows no substantial differences of the average monetary value for churners and non-churners beyond a decreasing trend for churners during the months close to the observation period.

Figure 1: Mean RFM data by churn status over time

(a) Frequency

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-05.jpg?height=494&width=532&top_left_y=964&top_left_x=248)

(b) Recency

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-05.jpg?height=491&width=514&top_left_y=965&top_left_x=771)

(c) Monetary value

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-05.jpg?height=485&width=514&top_left_y=968&top_left_x=1274)

Source: Own calculations.

Notes: Sample includes only customers who where observed during all the 36 months prior to the target observation window.

Figure 2 presents the average values of RFM variables when they are normalized by its average value of the last quarter of the year. In this case, panel (a) is the only one where there are clear differences between the descriptive statistics of churners and non-churners. The figures also highlight how due to the normalization the average difference in levels between groups disappears and only differences in trends remain.

### 3.2 Experimental Design

The main question that we aim to answer with the experimental design is i) what is the relative performance of LSTM models with RFM variables compared to standard models with static features, and ii) what is the best way to incorporate sequential information into static models. In this subsection we document our experimental design in terms of the chosen algorithms, hyper-parameter tuning strategy, handling of class imbalance, and evaluation metrics.

We choose to use the regularized logistic regression as a baseline model. This model is regularly used in industry because it can offer an interpretable measure of the effect of the features on the predicted probabilities of churn. Furthermore, one needs to tune only one hyper-parameter which is the regularization term. We use a Lasso-type of regularization and the set of considered hyper-parameters are given in table 2 .

Regarding the neural networks, we choose a LSTM architecture to work with the RFM variables. Table 3 presents the set of hyper-parameters to tune and the considered values

Figure 2: Mean of RFM data relative to the mean value of the previous three months by churn status over time

(a) Frequency

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-06.jpg?height=486&width=532&top_left_y=425&top_left_x=248)

(b) Recency

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-06.jpg?height=486&width=516&top_left_y=425&top_left_x=770)

(c) Monetary value

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-06.jpg?height=491&width=514&top_left_y=420&top_left_x=1274)

Source: Own calculations.

Notes: Sample includes only customers who where observed during all the 36 months prior to the target observation window. To obtain the mean for each customer the RFM is divided by the average of the RFM indicator of the previous quarter. For example, for months 10,11 , and 12 the, say, recency indicator is divided by the average of recency during months 7,8 , and 9 , and then the mean is estimated over all customers for a given date.

Table 2: Hyper Parameters Considered for Non-sequential Models

| Learner | Meta parameter | Broad Tune |
| :--- | :--- | :--- |
| Regularized Logistic Reg. | l-measure | 11 (lasso) |
|  | regularization C | $[0.001,0.01,0.02,0.03,0.04,0.05,0.1,1]$ |

Notes: All algorithms implemented in Python. Intercept is not penalized.

for the search.

Table 3: Parameter Setting for Experiments with Deep Neural Networks

| Architecture | Layers | Hidden Units | Filter Size | Optimizer | Learning rate | Epochs | Batch size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LSTM | 1 | $[5,10,25,30]$ | da | adam | $[0.001]$ | $[10,25,50,75]$ | $[10,25,50,100,250]$ |

Notes: Algorithms are implemented with Keras using Tensorflow as backend. da $=$ does not apply

To tune the hyper-parameters of the algorithms we use a nested cross-validation procedure. We choose to work with three outer folds, and four inner folds. We use the inner folds to tune the hyper-parameters (based on the AUC metric), and use the outer folds to compute the full set of evaluation metrics. Notice that after running the inner cross-validation there is the chance of obtaining different sets of tuning parameters for each model, where a model is defined based on the algorithm and the features used. For instance, since there are three outer folds, there can be up to 3 set of parameters for any model after running the four-fold inner cross-validations. In the case that these three sets of hyper-parameters are non-repeated we report the results of the model with the highest AUC metric. When the set of hyper-parameters is repeated we report the evaluation metrics for the model that appeared more times. In the best case scenario the three set of tuning parameters for any given model coincide.

Since we also consider using the LSTM fitted probabilities as feature in the logistic model, we need to take measures to reduce over-fitting. To do so, we divide the training sample of each inner and outer fold into k-folds. Then we train the LSTM with the optimal
set of hyper-parameters on k-1 folds and estimate the probability in the fold that was left out. This way, the fitted probability from the LSTM is free from leaking information. For the test folds we use the corresponding training fold to train the LSTM model and then obtain the fitted probabilities.

To evaluate the performance of the models we report the area under the receiver operating curve (AUC), top-decile lift, and expected maximum profit. ${ }^{4}$ Each metric is based on a different rationale. The AUC is a common metric applied in the literature that offers a performance measure of a classifier but it does not take into account misclassification costs, which are important from a marketing perspective. Top-decile lift, on the other hand, allows to evaluate the performance of a model within the top ten percent of customers with the highest probability of attrition which is useful when we consider that marketing campaigns are meant to be targeted, but is still not a profit-based metric for evaluation. Thus, by measuring the expected maximum profit we can evaluate the performance of the models from a marketing-relevant perspective since this metric is defined as the expected maximum profit of a classifier with respect to the distribution of classification costs (?).

One of the most salient stylized facts when modeling churn is that churners tend to represent a small proportion of the observed sample. To illustrate this, in the full training data set the ratio of churners to non-churners is 938 / 384733, which translates to a 0.243 percent of churners. This imbalance in the target variable can have a detrimental effect in the performance of the algorithms if not handled appropriately. Burez and van den Poel (2009) analyze the problem of class imbalance and show that under-sampling leads to better accuracy compared to more complex sampling techniques. Based on this, we choose an under-sampling approach. Given the high number of observations in the data we do not expect under-sampling to introduce a problem in the estimation due to a decreased sample size. This will also help us to reduce the training time of the algorithms. The under-sampling strategy consists of obtaining two observations from the non-churners sample at random for each customer that churned. Note that we apply this strategy only after the data is split for the nested cross-validation. Finally, features in the training data set are always standardized. The parameters for the standardization are obtained from the training data before under-sampling and then applied to the corresponding test set.

## 4 Results

Table 4 reports our main empirical results. The columns present the average value of the respective evaluation metric over the outer folds, whereas the rows show each of the models that we consider. The first row shows that a regularized logistic regression without RFM information and only static features performs the worst compared to the other models according to top-decile lift and EMPC measures. A salient feature is that the predictive performance of the LSTM, which only has RFM features, is higher than that of the logistic model when we use top-decile and EMPC as evaluation metrics.

When we use the LSTM to extract the probability of churning and use this probability as feature in the logistic model with static features (row 5) we obtain better results for each evaluation metric as compared to the logistic model with aggregated RFM features (rows 2, 3, and 4). This is the main result of the experiment since it provides evidence that[^2]using raw RFM data that is commonly available in the financial sector helps to improve the performance of the model without relying on further aggregation procedures. This result also highlights the importance of including RFM information in this study case where the firm operates in the sector of financial services because the magnitude of the improvement in some metrics is not marginal. For example, note that the logistic model with LSTM probabilities has a top-decile lift metric of 4.211 which means that we improve by this amount the number of identified churners as compared with randomly selecting ten percent of the sample. This represents a 25.7 percent $^{5}$ improvement for the lift metric compared to the logistic model without LSTM probabilities. Furthermore, the logistic model with LSTM probabilities has an EMPC measure that is three times larger than a model with only static features.

The next group of rows $(6,7,8)$ uses the LSTM fitted probabilities and the aggregated RFM features in the logistic model. This allows us to assess whether the aggregated time-varying features still help to enhance the performance of the logistic model after we summarized the RFM information through the probability from the LSTM. Results show that the lift metric in these rows is not higher than that of row (5). Similarly, the EMPC metric for these models is at most as large as that of row (5). Thus, this indicates that the fitted probabilities from the LSTM summarizes well the information of the time-varying features. Since we use many aggregations of the RFM variables, this is an important result because it tells us that instead of having to decide which of the aggregations to use, one could just rely on a single summary measure such as the fitted probability from the LSTM.

Table 4: Mean evaluation metrics

| Model | AUC | Lift | EMPC |
| :--- | ---: | ---: | ---: |
| Only static | 0.746 | 3.350 | 0.004 |
| Static + agg. RFM | 0.768 | 3.964 | 0.006 |
| Static + norm. lagged RFM | 0.763 | 3.913 | 0.007 |
| Static + agg. RFM + norm. lagged RFM | 0.774 | 4.083 | 0.009 |
| Static + LSTM prob. | 0.775 | 4.211 | $\mathbf{0 . 0 1 2}$ |
| Static + LSTM prob. + norm. lagged RFM | 0.770 | 4.160 | 0.012 |
| Static + LSTM prob. + agg. RFM | $\mathbf{0 . 7 7 9}$ | 4.186 | 0.009 |
| Static + LSTM prob. + agg. RFM + norm. lagged RFM | 0.774 | 4.177 | 0.009 |
| LSTM Neural network | 0.741 | 4.101 | 0.006 |

Notes: Evaluation metrics estimated as the mean over the 3-outer folds based on cross-validation. Lift refers to top decile lift, while EMPC refers to the Expected Maximum Profit Measure for Customer Churn (EMPC). All model results are based on 11-regularized logistic regression, except the LSTM neural network which uses the RFM sequential data. Static features refers to customer demographics, customer behavior, and customer contact variables

## 5 Final Comments

In this document we assess the predictive performance of neural network architectures for sequential data using RFM variables from a provider of financial services. We use the LSTM model to predict the probability of churning and evaluate its performance against the results of regularized logistic model. Results show that top-decile lift and EMPC[^3]measures of the LSTM model with RFM data are higher than the values of the logistic model with only static features. Moreover, when we use the LSTM fitted probabilities and standard static features with the logistic model we obtain the best results as measured by lift and EMPC.

Our results have important implications for churn modeling in the specific case of the financial industry because RFM data is likely to be readily available in this sector and its inclusion in predictive churn models is facilitated through deep learning model such as the LSTM as we show in this paper. This also highlights the importance of incorporating different types of dynamic behavioral data in churn modeling in combination with deep learning methods, which is an open area for future research.

While the use of deep learning models comes at the cost of increasing the number of tuning parameters and in some cases of training time, we argue that this limitations will be less constraining with the continue increase of computational power.

## Appendices

Figure 3 shows the lift curves of the first 20 percentiles for each of the outer folds. The figures clearly show that the lift metric of the LSTM model and the logistic model plus fitted probabilities tend to give the best results (a higher lift).

Figure 3: Lift curves for LSTM and Logistic models

(a) Outer Fold 1

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-10.jpg?height=360&width=514&top_left_y=688&top_left_x=263)

(b) Outer Fold 2

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-10.jpg?height=354&width=512&top_left_y=691&top_left_x=772)

(c) Outer Fold 3

![](https://cdn.mathpix.com/cropped/2024_06_04_a1030606b8477fb81b32g-10.jpg?height=357&width=505&top_left_y=690&top_left_x=1278)

Source: Own calculations.

Notes: LSTM refers to model where churn is the target and RFM data the features. Only static refers to l1-regularized logistic regression that uses only customer demographics, customer behavior, and customer contact variables. Static is l1-regularized logistic regression that includes static features as well as the estimated out-of-sample probability from a LSTM model. To estimate the percentiles the data is sorted based on the probabilities of the model from larger to smaller, thus percentile 1 is the 1 percent with the highest estimated churn probability. Figure shows only the first 20 percentiles.

## References

ŁadyżyŃski, P., K. ŻBikowski, and P. Gawrysiak (2019): "Direct marketing campaigns in retail banking with the use of deep learning and random forests," Expert Systems with Applications, 134, 28-35.

Burez, J. and D. van den Poel (2009): "Handling class imbalance in customer churn prediction," Expert Systems with Applications, 36, 4626 - 4636.

Chen, Z.-Y., Z.-P. FAN, and M. Sun (2012): "A hierarchical multiple kernel support vector machine for customer churn prediction using longitudinal behavioral data," European Journal of Operational Research, 223, $461-472$.

De Caigny, A., K. Coussement, K. W. D. Bock, and S. Lessmann (2019): "Incorporating textual information in customer churn prediction models based on a convolutional neural network," International Journal of Forecasting.

Hadden, J., A. Tiwari, R. Roy, and D. Ruta (2007): "Computer assisted customer churn management: State-of-the-art and future trends," Computers and Operations Research, 34, 2902 - 2917.

Martins, H. (2017): "Predicting user churn on streaming services using recurrent neural networks," .

Óskarsdóttir, M., C. Bravo, W. Verbeke, C. Sarraute, B. Baesens, and J. Vanthienen (2017): "Social network analytics for churn prediction in telco: Model building, evaluation and network architecture," Expert Systems with Applications, 85, $204-220$.

Song, G., D. Yang, L. Wu, T. Wang, and S. Tang (2006): "A Mixed Process Neural Network and its Application to Churn Prediction in Mobile Communications," in Sixth IEEE International Conference on Data Mining - Workshops (ICDMW'06), $798-802$.

Tan, F., Z. Wei, J. He, X. Wu, B. Peng, H. Liu, and Z. Yan (2018): "A Blended Deep Learning Approach for Predicting User Intended Actions," 2018 IEEE International Conference on Data Mining (ICDM), 487-496.

VAN DEN Poel, D. and B. LaRIviÈRe (2004): "Customer attrition analysis for financial services using proportional hazard models," European Journal of Operational Research, 157, 196 - 217, smooth and Nonsmooth Optimization.

Verbeke, W., K. Dejaeger, D. Martens, J. Hur, and B. Baesens (2012): "New insights into churn prediction in the telecommunication sector: A profit driven data mining approach," European Journal of Operational Research, 218, 211 - 229.

Wangperawong, A., C. Brun, O. Laudy, and R. Pavasuthipaisit (2016): "Churn analysis using deep convolutional neural networks and autoencoders," CoRR, abs $/ 1604.05377$.

WeI, C.-P. and I.-T. ChIU (2002): "Turning telecommunications call details to churn prediction: a data mining approach," Expert Systems with Applications, 23, 103 - 112.

Zaratiegui, J., A. Montoro, and F. Castanedo (2015): "Performing Highly Accurate Predictions Through Convolutional Networks for Actual Telecommunication Challenges," CoRR, abs/1511.04906.

Zhang, Y., E. T. Bradlow, and D. S. Small (2014): "Predicting customer value using clumpiness: From RFM to RFMC," Marketing Science, 34, 195-208.

Zhou, J., J.-F. Yan, L. Yang, M. Wang, and P. Xia (2019): "Customer Churn Prediction Model Based on LSTM and CNN in Music Streaming," DEStech Transactions on Engineering and Technology Research.


[^0]:    *All comments, conclusions, and errors are those of the authors only. Please send comments to gary.mena@bdpems.de

[^1]:    ${ }^{1}$ Customers can have access to four different type of financial products. To preserve the anonymity of the financial institution we refrain from mentioning the names of the products.

    ${ }^{2} \mathrm{~A} 12$ month time window is also used in Burez and van den Poel (2009) for one of their datasets.

    ${ }^{3}$ For example, focusing on the numerator the following holds: $\left(x_{t}-x_{t-1}\right)+\left(x_{t-1}-x_{t-2}\right)=x_{t}-x_{t-2}$

[^2]:    ${ }^{4}$ The code for the expected maximum profit is taken from https://github.com/estripling/ proflogit-python/blob/master/proflogit/empc.py. We take the customer lifetime value to estimate the expected maximum profit based on the values in De Caigny et al. (2019), who use this information to compute the profit of a customer retention campaign.

[^3]:    ${ }^{5}=(4.211-3.350) / 3.350 * 100$

</end of paper 0>


<paper 1>
# MUlTimodAL DeEP LEARNING OF Word-OF-MOUTH TEXT AND DEMOGRAPHICS TO PREDICT CUSTOMER RATING: HANDLING CONSUMER HETEROGENEITY IN MARKETING 

A PREPRINT<br>Junichiro Niimi ${ }^{*}$ 1,2<br>${ }^{1}$ Meijo University<br>${ }^{2}$ RIKEN AIP


#### Abstract

In the marketing field, understanding consumer heterogeneity, which is the internal or psychological difference among consumers that cannot be captured by behavioral logs, has long been a critical challenge. However, a number of consumers today usually post their evaluation on the specific product on the online platform, which can be the valuable source of such unobservable differences among consumers. Several previous studies have shown the validity of the analysis on text modality, but on the other hand, such analyses may not necessarily demonstrate sufficient predictive accuracy for text alone, as they may not include information readily available from cross-sectional data, such as consumer profile data. In addition, recent advances in machine learning techniques, such as large-scale language models (LLMs) and multimodal learning have made it possible to deal with the various kind of dataset simultaneously, including textual data and the traditional cross-sectional data, and the joint representations can be effectively obtained from multiple modalities. Therefore, this study constructs a product evaluation model that takes into account consumer heterogeneity by multimodal learning of online product reviews and consumer profile information. We also compare multiple models using different modalities or hyper-parameters to demonstrate the robustness of multimodal learning in marketing analysis.


Keywords Multimodal Learning $\cdot$ BERT $\cdot$ Customer Relationship Management $\cdot$ Word-of-Mouth $\cdot$ LLM

## 1 Introduction

Deep learning is currently used in various fields. In the field such as image recognition, it is already common to obtain features that affect the target variable by the feature extraction of deep learning [1], and it is considered one of the most powerful advantages of deep learning [2]. In the field of customer relationship management (CRM) in marketing, it also has been widely employed in various ways of prediction and demonstrated the validity of utilizing deep learning, such as customer segmentation [3], customer lifetime value (CLV) [4, 5], purchases (in the future or current session) [6, 7, 8], churn [9, 10], and other tasks [11].

However, within the vast field of data science, a problematic factor, referred as consumer heterogeneity, has been identified, particularly in marketing. In marketing analysis, we combine a variety of behavioral log data; however, all the data recorded in the behavioral logs are the results of behaviors. The Differences in the psychological attributes among consumers that cause those behaviors cannot be obseerved by such logs. Many previous studies have highlighted the importance of considering heterogeneity [12, 13].

In terms of combining various data, multimodal learning [14, 15] has become widely popular in machine learning applications in general. It combines and learns different types of multiple data (i.e., modality), for example, audio and its corresponding text, in a state close to the original data. This method allows modeling that considers the relationships between modalities. However, in the marketing field, analysis using such a variety of data is generally conducted by variablizing each data into a single dataset [16].[^0]

This paper is organized as follows. The previous studies are reviewed in Section 2, and the methodology (proposed model and used data) is shown in Section 3. An overview of the analysis is described in Section 4. The results of the analysis are presented in Section 5. Finally, we present the discussion and conclusions in Section 6.

## 2 Related Work

### 2.1 Text Analysis in Marketing

Natural Language Processing (NLP) techniques have been proposed for analyzing text modalities using deep learning. Among them, bidirectional encoder representations from transformers (BERT) [17] and their extensions [18, 19] are some of the most popular methods because of their wide applicability, which is not limited to NLP tasks, such as machine translation and question answering. For pre-training, it has learned a contextual language representations using large text corpus; therefore, it can be further applied to various downstream tasks through fine-tuning. Many models have already undergone pretraining with extensive data, enabling effective analysis with relatively small datasets. In recent years, an extension of BERT into multiple languages, known as multilingual BERT (mBERT), has emerged, and several models for the Japanese version, which is the focus of this study, have also been proposed (e.g., BERT models trained by Cyber Agent [20] and Tohoku NLP Group [21]). Before the advent of BERT, vectorization methods such as word2 vec [22] and doc2vec [23] had been used to map sentences to feature vectors. Compared with techniques that produce a single- word embedding representation, the advantage of BERT is that it is context-wise, which means that it produces a representation based on other terms in the sentence [17, 24].

For the actual use of these methods in marketing, especially regarding word-of-mouth texts, this study [24] predicts user review (that is, sentiment) for smartphone games using review texts collected from the Google Play Store, which compare several techniques to obtain word representations, including RNN [25], CNN [26], LSTM [27], BERT, mBERT, DistilBERT [18], and RoBERTa [19]. In addition, a study [28] adopted BERT to predict customers' ratings of hotels in seven criteria (e.g., overall ratings, value, and service) simultaneously, using online review texts for the recommender system. They indicated that BERT could predict a more accurate rating by considering the context of a review text. Another study regarding social media marketing [29] adopts BERT to capture the social media engagements and comments of influencers of eight categories on Instagram. Thus, several studies have used BERT to map word-of-mouth documents to feature maps to predict customer evaluations.

In addition, while BERT has two different scale: Base (approx. 110 million parameters) and Large (approx. 340 million parameters), the extent to which such scales of BERT affect the result of the analysis for relatively small text data is yet to be clarified, particularly in non-English model $s^{2}$

### 2.2 Multimodal Learning in Marketing

Multimodal learning involves learning representations from multiple modalities such as images, videos, audio, and text. A combination of these enables the construction of a robust learner based on the relationships among modalities that cannot be obtained by learning a single modality [15]. There are a variety of applications, including the integration of information from multiple sources and interconversion between modalities, applied to a wide range of academic fields such as medicine, human-computer interaction, biometrics, and remote sensing [30]. In contrast to this rise in multimodal learning, research on multimodal learning in the marketing field is relatively scarce. One possible reason is that marketing data analysis involves a mixture of various data in different formats, such as server logs, ID-POS, GPS, survey responses, and customer information. Although analyses combining various datasets are widely practiced, they are typically conducted by variablizing multiple data and merging them into a single set for analysis 3

However, there are a few notable instances, such as studies that construct a multimodal deep learning model to predict consumer loyalty with the source-target attention mechanism [31], which datasets with different dimensionality are input simultaneously; however, by using bidirectional LSTM, the time-series data is converted to two-dimensional and unified to a single representation with cross-sectional data by feature fusion ${ }^{4}$.

### 2.3 Consumer Heterogeneity in Marketing

Both the studies mentioned above highlight that it is possible to enhance discriminative power by considering demographic variables as a context affecting actual behavior. In the CRM context, the problem of consumer heterogeneity has long been highlighted, where even consumers who perform same behavior have unobservable differences [12],[^1]such as demographic and psychographic differences. In general, this kind of difference is unobservable in behavioral logs such as ID-POS data. Several studies have attempted to capture such differences using statistical modeling, such as structural equation modeling (SEM) and Bayesian mixture models [32, 33]. When it comes to the evaluation of the results addressed in this study, even among customers who have the same rating (number of stars), the reasons for the rating must differ; however, this difference cannot be observed in the cross-sectional data. In recent years, with the availability of a variety of large-scale data, document data such as word-of-mouth on online platforms have been used in the aforementioned studies as an important source for understanding consumer preferences [34].

Thus, several studies have utilized BERT and text data to make predictions that account for consumer heterogeneity. In general, the review text has more or less amount of reason why they evaluate the venue in such rating. In other words, the data, although it is behavioral log, can be the important source of understanding consumer heterogeneity. Therefore, this study utilize product review as a means to capture consumer heterogeneity to predict with better performance.

On the other hand, relying sorely on the feature extraction of machine learning is not advisable because domain knowledge is not incorporated into the analysis. In fact, several studies have shown that in multimodal learning, the combination of extracted features and handcrafted variables achieves the best prediction accuracy [1, 16]. Therefore, this study constructs a multimodal deep learning model that combines the review text with handcrafted user profile variables to achieve a robust and precise model.

## 3 Proposed Model

Based on previous studies of multimodal learning using time series and cross-sectional data [17], we present a conceptual model of multimodal deep learning that integrates three smaller neural network components, referred to as subnetworks or subnets. In this study, they are called text-specific subnetworks (X1-subnet), cross-sectional-data-specific subnetworks (X2-subnet), and output subnetworks (output-subnet). The construction of each subnet is described in the following subsections.

### 3.1 Text-Specific Subnetworks (X1-subnet)

First, we describe the structure of the X1-subnet, which specializes in processing text data. The purpose of this subnet is to map word-of-mouth texts (whose lengths differ among users) onto a two-dimensional single-feature map. The actual component included two layers: a tokenization layer and a BERT layer. In this study, BERT was used to obtain embedded representations of text data; therefore, the X1-subnet (tokenizer and BERT layers) is freezed throughout the actual training process, which means that the parameters of the subnet are held fixed in the pre-trained state.

![](https://cdn.mathpix.com/cropped/2024_06_04_a75d6c2d33a2963d18b0g-3.jpg?height=49&width=1645&top_left_y=1515&top_left_x=240)
(len $\max$ represents the maximum number of tokens of the text among $N$ sets of data). After processing in the BERT layer with batch size $n$, we obtained the feature map as the pooler output of BERT as the $(n$, len max $)$ tensor 5

### 3.2 Cross-Sectional-Data-Specific Subnetworks (X2-subnet)

Next, we describe the structure of the X2-subnet handling two-dimensional cross-sectional data. This subnet consists of a typical deep neural network with feed-forward layers (FFLs). An input layer for the X2-subnet receives a 2dimensional tensor $\left(n, J_{\text {in }}\right)$, with batch size $n$ and $J_{\text {in }}$ variables in the cross-sectional data. After processing, $\left(n, J_{\text {out }}\right)$ feature map is obtained as the output.

### 3.3 Output Subnetworks (output-subnet)

Textual and cross-sectional data, processed in parallel in each dedicated subnet are sent to the output subnet. In this subnet, both the feature maps, stemming from $\mathrm{X} 1$ and $\mathrm{X} 2$, are unified as a joint representation by feature fusion laye 6 . and the obtained feature map would be ( $n$, len $\max +J_{\text {out }}$ ) tensor. After the feature fusion, the joint representation is processed through one or more FFLs, and finally classification is conducted with Softmax layer. All the FFLs throughout the model employed a hyperbolic tangent function (tanh) for layer activation.

## 4 Analysis

### 4.1 Dataset

To conduct this analysis, we randomly selected one product from the women's cosmetics market. The target product had to be well-recognized in Japan and already out of production. We collected data from 1040 participants online.[^2]

![](https://cdn.mathpix.com/cropped/2024_06_04_a75d6c2d33a2963d18b0g-4.jpg?height=1314&width=1001&top_left_y=270&top_left_x=562)

Figure 1: Model Architecture

The actual data contains three kinds of modalities: rating for the product (7-point Likert scale), word-of-mouth texts for the product, and demographic information 7 Finally, the sample size is 1532 (i.e., $N=1532$ ).

Next, we process the survey text in the general way of preprocessing in NLP. The text contained line breaks, pictographs, emoticons, and other characters that were not appropriate for analysis. Therefore, in the preprocessing stage, these elements were replaced with periods only when they were placed at the end of the sentence; otherwise, the elements were removed. Subsequently, all successive punctuation periods are merged into a single period. Consequently, the maximum length of the text data was set as $l e n_{\max }=200$.

### 4.2 Model Evaluation

This study adopts a simple binary classification for the task, similar to previous studies [24]. User ratings were dichotomized into two classes based on the rating scale. Six and seven stars were classified as Loyalty $=1$ (loyalty is high) and Loyalty $=0$ (loyalty is not high), respectively. The obtained dataset was divided into a training set (75\%) and a test set $(25 \%)$. The model performance was evaluated using both training and test accuracies and the number of epochs to converge.

This study aims to validate several key points to compare the models. First, to validate the usefulness of multimodal learning in marketing, we construct three basic models according to their modalities: X1-modal, X2-modal, and[^3]

Table 1: Model Settings

| Parameters | Candidates |
| :--- | :--- |
| Model Parameters |  |
| Number of Epochs | 200 (with Early-Stopping in 50 Epochs) |
| Batchsize $(n)$ | 64 |
| Optimizer | \{Adam, Adamax, Nadam \} |
| Loss Function | Binary Cross-entropy |
| X1-subnet | mBERT (Japanese) |
| Structure | \{bert-base-japanese-v3, |
| Model | bert-base-japanese-char-v3, |
|  | bert-large-japanesese-v2, |
|  |  |
| X2-subnet | 2 |
| Number of Hidden Layers | 10 |
| Number of Neurons in the Layer | tanh |
| Activation Function | 2 |
| output-subnet | 10 |
| Number of Hidden Layers | tanh |
| Number of Neurons in the Layer |  |
| Activation Function |  |

Note. tanh stands for hyperbolic tangent function.

multimodal. This comparison allowed for the verification of changes in prediction accuracy by combining multiple modalities. Especially in marketing analysis, review texts can be a valuable source for comprehending consumer heterogeneity in user ratings.

Next, we examined the change in prediction accuracy using multiple pre-trained models within BERT. Regarding the Japanese language model, several models of different scales (that is, numbers of parameters), different training datasets, and tokenization methods (in particular, with regard to Japanese models, some are trained on a word-by-word basis, whereas others are trained on a character-by-character basis) have already been proposed, and the extent to which the prediction accuracy differs depending on the use of these models is yet to be clarified. Nowadays, we can easily switch the pre-trained model in BERT by changing one line of the code, which makes it easy to compare the accuracies of different pre-trained models. As it has already been shown in the literatures [24, 28, 29] that models using BERT-like architectures achieve higher accuracy compared to those using Collaborative Filtering, LSTM, CNN, and other benchmark models, this study sticks to comparing among multiple models using BERT.

The model settings are shown in Table 11 In the case where several candidates are shown in the setting, we utilize a grid search to explore the settings that maximize the test accuracy. For example, we compared the prediction accuracy among the four pre-trained models in BERT (bert base/large in word/char) and among three optimizers (Adaptive Moment Estimation, Adam [35], Adamax [35], and Nesterov-accelerated Adaptive Moment Estimation, Nadam [36]).

The training process was conducted with a maximum of 200 epochs and 64 batch sizes. Early Stopping [37] was employed with a patience of 50 epochs, which terminates the training if no improvement in the accuracy of the validation data was observed within 50 epochs.

## 5 Results

The best models for each modality are listed in Table 2. First, in both the training and test results, the prediction accuracy improved the most with multimodal learning. Although these results do not allow us to evaluate whether multimodal learning immediately improves the prediction accuracy, multimodal learning with the bert-base-japanesev3 model shows the highest prediction accuracy for the test data, which indicates that the extension to multimodal learning alone does not improve the prediction accuracy. For multimodal learning, we need to carefully consider factors such as the task to be solved with the multimodal model, the relationship between the modalities, and the quality of the data, because several previous studies have shown that the prediction accuracy in multimodal learning can be influenced by such factors [38].

Table 2: Result I (Accuracy in Train and Test Data)

| BERT Model / Modality | Train |  |  | Test |  |  | Epochs |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Both | $\mathrm{X} 1$ | $\mathrm{X} 2$ | Both | $\mathrm{X} 1$ | $\mathrm{X} 2$ | Both | $\mathrm{X} 1$ | $\mathrm{X} 2$ |
| cl-tohoku/bert-base-japanese-v3 | 0.705 | 0.708 | -1 | 0.711 | 0.703 | - | 36 | 38 | - |
| cl-tohoku/bert-base-japanese-char-v3 | 0.683 | 0.686 | - | 0.690 | 0.678 | - | 41 | 42 | - |
| cl-tohoku/bert-large-japanese-v2 | $0.712 \quad$ | 0.698 | - | 0.695 | 0.699 | - | 95 | 190 | - |
| cl-tohoku/bert-large-japanese-char-v2 | 0.681 | 0.658 | - | 0.690 | 0.703 | - | 58 | 18 | - |
| None | - | - | 0.555 | - | - | 0.623 | - | - | 29 |

Note. Numbers in bold represent the best accuracy in training and testing and the best epochs.

Table 3: Result II (Group Average)

| Optimizer | Train | Test | Epochs |  | Modality | Train | Test | Epochs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Adam | 0.646 | 0.659 | 58.4 |  | Both | 0.681 | $\mathbf{0 . 6 8 3}$ | 70.2 |
| Adamax | 0.627 | 0.651 | 61.7 |  | X1 | $\mathbf{0 . 6 8 5}$ | 0.681 | 58.9 |
| Nadam | $\mathbf{0 . 6 4 6}$ | $\mathbf{0 . 6 6 2}$ | $\mathbf{4 6 . 2}$ |  | X2 | 0.546 | 0.604 | $\mathbf{3 5 . 5}$ |

Note. Numbers in bold represent the best accuracy in training and testing and the best epochs.

Second, in the comparison of four pre-trained models, note that upgrading to the BERT Large model did not always lead to a significant improvement in accuracy. Regarding the scale of the model, there is a tendency for an increase in the number of epochs required for model training with bert-large-japanese-v2, and the X1-modality has reached 190 epochs. This result might indicate the potential for further improvement with an increase in the number of epochs; however, large model is not always necessary because other models have shown higher prediction accuracy in fewer epochs.

In addition, a comparison of the mean accuracies among certain conditions is listed in Table 3 In terms of the optimizer, although the training accuracy is almost the same between Adam and Nadam, the latter is better in terms of both test accuracy and the best epoch, which means that, on average, Nadam achieved a higher generalization performance in a shorter tim 8 . In terms of modality, the X1-modal shows high accuracy, as listed in Table 2, for the training process; however, multimodal learning, on average, still shows the highest accuracy for the test data. This result suggests that, on average, multimodal learning improves generalization performance. As expected, the accuracy of the analysis using only X2-modal remains low for both training and testing, although the model converged early.

## 6 Conclusions

This study attempts to construct a multimodal deep learning model that predicts the user ratings of a product using both review text and user profile data simultaneously to account for consumer heterogeneity. First, as academic implications, even when both review and demographic data are relatively small, both the best model and the average score by modality, the prediction accuracy is the best when they are combined, which indicates that multimodal learning that accounts for consumer heterogeneity allows analysis with high robustness and generalizability. Second, it can be shown that, at least when dealing with relatively short sentences such as those used in this study (len $\max ^{2}=$ 200), a larger BERT model does not necessarily contribute to an improvement in prediction accuracy. This implies that, particularly in small datasets like those used in this study, converting sentences into word embeddings with BERT is important while the scale of the BERT model is not necessarily critical. Next, the conceptual model presented in this study, as a way to extend review data with cross-sectional data or as a way to extend cross-sectional data with review data, has a potential to be extended to various prediction models in marketing analysis with higher prediction accuracy, compared to conventional methods.

Finally, owing to the constraints of data collection, this study relies on consumer ratings as a proxy for behavioral loyalty and predicts whether it is high or not using the proposed model. However, this methodology can be extended to purchase prediction models by incorporating data that include purchase history and more demographics. A few models in previous studies [31] fused modalities twice within a model through the use of attention mechanisms and feature fusion, which aims to enhance prediction accuracy and robustness. Moreover, regarding the actual analysis, further improvements in accuracy can be expected by adopting techniques such as dropout [39]. In addition, although[^4]the multimodal learning model developed in this study, which utilizes actual review texts as a source of information for understanding consumer heterogeneity, is based on the assumption that consumer heterogeneity is embedded in the review texts, the actual causal relationships need to be carefully examined for the presence of potential endogeneity between the variables.

## Acknowledgements

All computations in this study were conducted using RAIDEN, which is a computational infrastructure hosted by RIKEN AIP. We would like to express our gratitude to all the members of AIP who maintain the system.

## References

[1] Loris Nanni, Stefano Ghidoni, and Sheryl Brahnam. Handcrafted vs. non-handcrafted features for computer vision classification. Pattern Recognition, 71:158-172, 2017.

[2] Yoshua Bengio. Deep learning of representations: Looking forward. In International Conference on Statistical Language and Speech Processing, pages 1-37. Springer, 2013.

[3] Licheng Zhao, Yi Zuo, and Katsutoshi Yada. Sequential classification of customer behavior based on sequenceto-sequence learning with gated-attention neural networks. Advances in Data Analysis and Classification, pages $1-33,2022$.

[4] Rafet Sifa, Julian Runge, Christian Bauckhage, and Daniel Klapper. Customer lifetime value prediction in noncontractual freemium settings: Chasing high-value users using deep neural networks and smote. 2018.

[5] Pei Pei Chen, Anna Guitart, Ana Fernández del Río, and Africa Periánez. Customer lifetime value in video games using deep learning and parametric models. In 2018 IEEE international conference on big data (big data), pages 2134-2140. IEEE, 2018.

[6] Jan Valendin, Thomas Reutterer, Michael Platzer, and Klaudius Kalcher. Customer base analysis with recurrent neural networks. International Journal of Research in Marketing, 39(4):988-1018, 2022.

[7] Arthur Toth, Louis Tan, Giuseppe Di Fabbrizio, and Ankur Datta. Predicting shopping behavior with mixture of rnns. In $e C O M @$ SIGIR, 2017.

[8] Long Guo, Lifeng Hua, Rongfei Jia, Binqiang Zhao, Xiaobo Wang, and Bin Cui. Buying or browsing?: Predicting real-time purchasing intent using attention-based deep network with multiple behavior. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery \& data mining, pages 1984-1992, 2019.

[9] C Gary Mena, Arno De Caigny, Kristof Coussement, Koen W De Bock, and Stefan Lessmann. Churn prediction with sequential data and deep neural networks. a comparative analysis. arXiv preprint arXiv:1909.11114, 2019.

[10] Philip Spanoudes and Thomson Nguyen. Deep learning in customer churn prediction: unsupervised feature learning on abstract company independent feature vectors. arXiv preprint arXiv:1703.03869, 2017.

[11] Mainak Sarkar and Arnaud De Bruyn. Lstm response models for direct marketing analytics: Replacing feature engineering with deep learning. Journal of Interactive Marketing, 53(1):80-95, 2021.

[12] Peter E Rossi, Robert E McCulloch, and Greg M Allenby. The value of purchase history data in target marketing. Marketing Science, 15(4):321-340, 1996.

[13] Werner J Reinartz and Vita Kumar. The impact of customer relationship characteristics on profitable lifetime duration. Journal of marketing, 67(1):77-99, 2003.

[14] Nitish Srivastava and Russ R Salakhutdinov. Multimodal learning with deep boltzmann machines. Advances in neural information processing systems, 25, 2012.

[15] Jiquan Ngiam, Aditya Khosla, Mingyu Kim, Juhan Nam, Honglak Lee, and Andrew Y Ng. Multimodal deep learning. In ICML, 2011.

[16] Junichiro Niimi and Takahiro Hoshino. Predicting purchases with using the variety of customer behaviors analysis of the purchase history and the browsing history by deep learning-. Transactions of the Japanese Society for Artificial Intelligence, 32(2):B-G63_1-9, 2017.

[17] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[18] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv, 2019.

[19] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

[20] Alex Andonian, Quentin Anthony, Stella Biderman, Sid Black, Preetham Gali, Leo Gao, Eric Hallahan, Josh Levy-Kramer, Connor Leahy, Lucas Nestler, Kip Parker, Michael Pieler, Shivanshu Purohit, Tri Songz, Wang Phil, and Samuel Weinbach. GPT-NeoX: Large Scale Autoregressive Language Modeling in PyTorch, 82021.

[21] Tohoku NLP Group. cl-tohoku/bert-japanese (github: https://github.com/cl-tohoku/bert-japanese), 2023.

[22] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.

[23] Quoc Le and Tomas Mikolov. Distributed representations of sentences and documents. In International conference on machine learning, pages 1188-1196. PMLR, 2014.

[24] Zeynep Hilal Kilimci. Prediction of user loyalty in mobile applications using deep contextualized word representations. Journal of Information and Telecommunication, 6(1):43-62, 2022.

[25] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533-536, 1986.

[26] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.

[27] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.

[28] Yuanyuan Zhuang and Jaekyeong Kim. A bert-based multi-criteria recommender system for hotel promotion management. Sustainability, 13(14):8039, 2021.

[29] Seungbae Kim, Xiusi Chen, Jyun-Yu Jiang, Jinyoung Han, and Wei Wang. Evaluating audience loyalty and authenticity in influencer marketing via multi-task multi-relational learning. In Proceedings of the International AAAI Conference on Web and Social Media, volume 15, pages 278-289, 2021.

[30] Dhanesh Ramachandram and Graham W Taylor. Deep multimodal learning: A survey on recent advances and trends. IEEE signal processing magazine, 34(6):96-108, 2017.

[31] Maher Ala'raj, Maysam F Abbod, and Munir Majdalawieh. Modelling customers credit card behaviour using bidirectional lstm neural networks. Journal of Big Data, 8(1):1-27, 2021.

[32] Carsten Hahn, Michael D Johnson, Andreas Herrmann, and Frank Huber. Capturing customer heterogeneity using a finite mixture pls approach. Schmalenbach Business Review, 54:243-269, 2002.

[33] Thomas Otter, Regina Tüchler, and Sylvia Frühwirth-Schnatter. Capturing consumer heterogeneity in metric conjoint analysis using bayesian mixture models. International Journal of Research in Marketing, 21(3):285297, 2004.

[34] Silvana Aciar, Debbie Zhang, Simeon Simoff, and John Debenham. Recommender system based on consumer product reviews. In 2006 IEEE/WIC/ACM International Conference on Web Intelligence (WI 2006 Main Conference Proceedings)(WI'06), pages 719-723. IEEE, 2006.

[35] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[36] Timothy Dozat. Incorporating nesterov momentum into adam. 2016.

[37] Lutz Prechelt. Early stopping-but when? In Neural Networks: Tricks of the trade, pages 55-69. Springer, 1998.

[38] Douwe Kiela and Léon Bottou. Learning image embeddings using convolutional neural networks for improved multi-modal semantics. In Proceedings of the 2014 Conference on empirical methods in natural language processing (EMNLP), pages 36-45, 2014.

[39] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929$1958,2014$.


[^0]:    *jniimi@meijo-u.ac.jp

[^1]:    ${ }^{2}$ The results are comparable between the model scales since both BERT models of Base and Large are designed to handle the same number of input tokens.

    ${ }^{3}$ The information loss occurring in the process of converting behavioral log into cross-sectional data may lead to a decrease in the accuracy of marketing analysis using deep learning. In other words, the manually variablized features cannot be sufficient statistics for the task on their own.

    ${ }^{4}$ Notably, in both studies, data fusion is conducted twice in one network structure: source-target attention and feature fusion.

[^2]:    ${ }^{5}$ The maximum length of the text $l e n_{\max } \leq 512$ since it cannot be exceeded the maximum number of tokens that BERT can deal with.

    ${ }^{6}$ However, a more rigorous call of feature fusion in this study should be "intermediate fusion" [30].

[^3]:    ${ }^{7}$ Since the cosmetic we focus on is a product intended for use by females, the survey was limited to females who purchased the products themselves.

[^4]:    ${ }^{8}$ Note that the time required for training one epoch did not differ significantly among optimizers.

</end of paper 1>


<paper 2>
# An Efficient Multimodal Learning Framework to Comprehend Consumer Preferences Using BERT and Cross-Attention 

Junichiro Niimi ${ }^{1,2}$<br>(jniimi@meijo-u.ac.jp)<br>1. Faculty of Business Management, Meijo University<br>2. RIKEN Center for Advanced Intelligence Project (AIP)


#### Abstract

Today, the acquisition of various behavioral log data has enabled deeper understanding of customer preferences and future behaviors in the marketing field. In particular, multimodal deep learning has achieved highly accurate predictions by combining multiple types of data. Many of these studies utilize with feature fusion to construct multimodal models, which combines extracted representations from each modality. However, since feature fusion treats information from each modality equally, it is difficult to perform flexible analysis such as the attention mechanism that has been used extensively in recent years. Therefore, this study proposes a context-aware multimodal deep learning model that combines Bidirectional Encoder Representations from Transformers (BERT) and cross-attention Transformer, which dynamically changes the attention of deep-contextualized word representations based on background information such as consumer demographic and lifestyle variables. We conduct a comprehensive analysis and demonstrate the effectiveness of our model by comparing it with six reference models in three categories using behavioral logs stored on an online platform. In addition, we present an efficient multimodal learning method by comparing the learning efficiency depending on the optimizers and the prediction accuracy depending on the number of tokens in the text data.


Index Terms-Deep Learning, Multimodal Learning, electronic Word-of-Mouth, BERT, Cross-Attention, LLM, Transformer.

## I. INTRODUCTION

Nowadays, social media and other online platforms play an important role in shaping consumer behaviors and aiding decision-making. However, amidst the burgeoning amount of online information, users often face difficulties in discovering preferred content and suitable services [1]. To address this information overload, recommender systems have recently found application not only in social networking services (SNSs) and electronic commerce (EC) but also in wider domains such as tourism, healthcare, and education [2]. To optimize personalized content for each user, these systems must accurately discern the preferences of consumers with various sets of values to offer tailored recommendations.

With the evolution of machine learning techniques, contemporary models can handle a wide array of data, including text. Notably, Transformer [3] has made substantial contributions to the field of natural language processing (NLP). Bidirectional Encoder Representations from Transformers (BERT)
[4] is known as one of the significant models in this regard. Leveraging large language models (LLMs), BERT enables the prediction and classification of consumers based on the texts they contribute to the platform. Furthermore, many recommender systems leverage review texts posted on platforms, [5], commonly referred to as electronic word-of-mouth (eWOM).

In addition, multimodal learning, which combines multiple types of data to derive joint representations for classification and regression tasks, has gained widespread adoption. In particular, simultaneous analysis of data such as text and images, previously difficult to analyze individually, is now being undertaken in conjunction with other modalities. Nonetheless, despite these advancements, multimodal learning in marketing studies remains relatively limited, primarily due to the specificity of the data and marketing-specific issues, notably consumer heterogeneity [6].

Both the development of an optimal recommender system and the acquisition of review data are crucial on online platforms; however, the use of these data remains limited despite their potential value in understanding customer preferences. Therefore, in this study, we construct a novel multimodal deep learning model to assess user preferences on social platforms. The paper is structured as follows: First, we review prior studies relevant to our research. Next, we formulate hypotheses to address our research question. Subsequently, we outline the model architecture and provide an overview of the dataset. Then, we conduct several analyses to demonstrate the performance of the proposed model. Finally, we summarize the results and discuss the implications and challenges of our study.

## II. RELATEd StUDY

## A. Attention Mechanism

First, it is essential to discuss the attention mechanism (Fig. (1a) [7], which has had a significant impact on the field of machine learning. This mechanism operates by selectively focusing on relevant parts of the input sequence, thereby enabling models to prioritize and process these significant elements with greater emphasis. For example, a scaled-dot

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-02.jpg?height=401&width=217&top_left_y=195&top_left_x=214)

(a) Attention mechanism

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-02.jpg?height=393&width=265&top_left_y=194&top_left_x=453)

(b) Multihead attention

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-02.jpg?height=404&width=249&top_left_y=194&top_left_x=735)

(c) Transformer encoder
Fig. 1. From Attention to Transformer

attention (Att) is computed using query $(Q)$, key $(K)$, value $(V)$, and the softmax function (softmax) as follows:

$$
\begin{equation*}
\operatorname{Att}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{K}}}\right) V \tag{1}
\end{equation*}
$$

It adjusts the focus by computing attention weights within the softmax function, assigning relative importance to each element within the sequence. Particularly advantageous in handling large representations, attention can effectively train the model through layer-wise concatenation of multiple representations [8]. Two widely recognized variations of this mechanism are self-attention (SA) and source-target attention (STA), both obtained through the same calculation. In terms of differences, SA involves query, key, and value for the source and computes relationships between elements within the source sequence. On the other hand, STA uses query and key for the source, and value for the target, computing relationships between the source and the target. For example, in the field of NLP, SA is utilized to identify word-to-word relationships within a sentence, thus providing contextual understanding.

## B. Transformer

Based on the attention mechanism, Transformer distributes multiple attentions with weighted $Q, K$, and $V$ in parallel, a concept known as multihead attention (MHAtt, see Fig. 1b), which is expressed with $m$-th attention head $\left(\mathrm{Head}_{m}\right.$, where $m=1,2, \ldots, M)$ and Att as [3]:

$$
\begin{align*}
\operatorname{Head}_{m}(Q, K, V)= & \operatorname{Att}\left(Q W_{m}^{Q}, K W_{m}^{K}, V W_{m}^{V}\right)  \tag{2}\\
\operatorname{MHAtt}(Q, K, V)= & \operatorname{concat}\left(\operatorname{Head}_{1}, \text { Head }_{2}, \cdots\right.  \tag{3}\\
& \left.\operatorname{Head}_{M}\right) W^{O} \tag{4}
\end{align*}
$$

Transformer consists of an encoder and a decoder. Encoder's output (TransEnc, see Fig. 1c) is obtained using layernormalization $(L N)$ [9], feed-forward layer (FFL), residual network [10], and MHAtt as:

$$
\begin{equation*}
\operatorname{TransEnc}(Q, K, V)=L N(u+F F L(u)) \tag{5}
\end{equation*}
$$

$$
\begin{equation*}
\text { where } u=L N(Q+\operatorname{MHAtt}(Q, K, V)) \tag{6}
\end{equation*}
$$

While prior research has proposed integrating the attention mechanism into recurrent models [7], prior studies have shown that a single Transformer outperforms the combination of attention and recurrent structures [11].

Whether discussing the attention mechanism or Transformer, some studies [12], [13] have highlighted the utility of STA in capturing contextual information (i.e., the background) of the sequential data. Specifically, by setting tabular data (including demographic information) as the target, STA dynamically weighs the attention given to time-series data (including user behavior) as the source, based on demographic and other tabular variables. In addition, other study [14] highlighted the utility of cross-attention of Transformer which integrates both visual and textual post about the same event on social media to evaluate whether the post is informative or not.

## C. BERT

BERT stands out as one of the most significant pre-trained language models, which consists of Transformer encoder [4]. In the NLP field, the problem of ambiguity, where the meaning of a word changes depending on context, has long been recognized when handling textual data [15]. Within the BERT architecture, SA plays an important role in obtaining distributed representations of textual data known as deepcontextualized word representations. This mechanism overcomes the ambiguity problem by adjusting embeddings based on context, i.e., the word's relationship to other words in the sentence, unlike traditional word-embedding methods such as word2vec [16], which assign context-independent unique vectors [4], 17].

BERT utilizes fixed-length tokenization with padding and truncation. The full output shape of BERT is a 3-dimensional tensor with dimensions $\left(b s\right.$, len $\max$, param $_{B E R T}$ ), where len $\max$ represents the aligned length of the tokenized sequence, and param ${ }_{B E R T}$ depends on the scale of the BERT model (e.g., 768 for BERT-Base and 1024 for BERT-Large). In addition, BERT has a pooler-output which is the 2-dimensional tensor with shape ( $b s$, param $_{\text {BERT }}$ ) obtained by applying a tanh activation to a weighted sum of the [CLS] token. Pooler-output has been adopted in many downstream tasks due to its simplicity, effectively addressing ambiguity in natural language. For example, in marketing applications, a study [17] utilized BERT to obtain deep-contextualized word representations from review text about mobile applications on online platforms, enabling the prediction of user loyalty.

In addition, various models based on BERT have been proposed such as a robustly optimized BERT pre-training approach (RoBERTa) [18] and DistilBERT [19]. In particular, RoBERTa [18] is the improved model of BERT, which performance through pre-training on a larger dataset and longer training steps.

## D. Multimodal Learning

Originally, multimodal learning has made significant strides in computer science fields such as machine translation and computer vision [20], [21]. Multimodal learning involves extracting attributes from multiple data streams with different
shapes and dimensions, then learning to fuse these heterogeneous features and project them into a common representation space [22]. Two widely recognized approaches to conducting multimodal learning are early fusion and late fusion [23]. In late fusion, multiple decisions of classifiers are combined, while in early fusion, multiple representations from different inputs share a single hidden layer as a joint representation. In early fusion [21], feature fusion, typically achieved through layer-wise concatenation, forms a single feature map $\mathrm{H}_{3}$ by horizontally combining multiple input features $H_{1}$ and $H_{2}$ as $H_{3}=\left[H_{1}, H_{2}\right]$. In many cases where multimodal learning enhances accuracy, it does so by obtaining additional information beyond a single modality or leveraging information based on relationships between modalities. Prior study [24] shows that models perform optimally when combining representations from feature extraction with human-generated features.

The success of multimodal learning in these domains has spurred its application in wider domains, such as the classification of social media activity [25]-[27], the prediction of stock prices and credit scores in finance [12], [28], forecasting the usage amount of smartphone games [13], and evaluating customer product reviews [6]. Many of these studies emphasize the importance of multimodal learning that considers relationships between multiple modalities. It should be noted that some studies employ multimodal learning using attention mechanisms and developed models such as Memory Fusion Network (MFN) [29], [30], while others combine mechanisms and LSTM [12], [31], STA-Transformer [13], and crossattention between image and text [14]. As mentioned, since STA and cross-attention can model relationships between the source (input) and target (output), enabling the adjustment of attention weights based on features from modalities such as tabular data by placing different modalities at the source and target. It is shown to have the better performance than feature fusion.

## E. Consumer Heterogeneity and UGCs

In marketing literature, there has long been an acknowledgement of consumer heterogeneity, defined as latent differences in behaviors among multiple consumers. These differences, stemming from unobservable attributes such as demographic variables, life stage, and purpose of visit, significantly affect observable behaviors. The problem addressed in this study is that even when multiple users rate a restaurant similarly, understanding their preferences is hindered by the inability to discern latent differences. However, user-generated contents (UGCs), including review texts, offer potential insights into these differences.

Prior studies on electronic word-of-mouth (eWOM) [32], [33] and UGCs [34], [35] have predominantly focused on their impact on other consumers' brand attitudes, purchase intentions, and similar factors. However, UGCs also provide valuable information about consumer's own perceptions and attitudes toward products or services, enabling partial identification of heterogeneity without additional surveys typically required for cross-sectional data such as user profiles. Several studies [5], [6], [17] have enhanced the accuracy of product recommendations by analyzing customer evaluation data.
Nonetheless, a significant challenge for these studies is their reliance on single-modality textual data. As discussed in the Multimodal Learning subsection, extending these studies to multimodal learning models holds promise for further accuracy improvements. Moreover, one study [36] has constructed the crossmodal transfer learning model with considering heterogeneity in image and text.

## F. Research Gap and the Objective

Based on previous studies, we formulate hypotheses. As mentioned above, machine learning has drastically advanced so far; however, research gap exists especially with regard to applications of machine learning in marketing.

For example, while BERT is capable of acquiring deepcontextualized word representations based on literal context, marketing context encompasses broader aspects, including consumer demographics and life stage. Regarding the consumer behaviors such as posting the review on the online platforms, the meaning of the word could depend not only on text context but also on the background information of the consumers. Surprisingly, there are no studies addressing this broader context of textual and tabular data within a single model. Therefore, we construct a context-aware multimodal deep learning model using BERT and cross-attention, considering consumer context to predict behavior. Therefore, our first hypothesis is as follows.

H1 The prediction accuracy improves significantly using the context-aware model compared to reference models.

Additionally, we assess the effectiveness of our model across diverse sample groups: Restaurants, Nightlife, and Café (cf. Data Description subsection). Given the diverse characteristics of the Nightlife category, which may include entertainment factors such as shows, music, and alcohol, predicting ratings in this category is expected to be more challenging. Thus, we propose the following hypothesis.

H2 The prediction accuracy decreases on average in Nightlife category.

Moreover, multimodal learning models often contend with numerous parameters and complex architectures. In such sparse training scenarios, determining which parameters to update can be challenging, leading to vanishing gradient problem. While Adam optimization is a common choice for deep learning, Adamax may prove more effective in training such models since the original paper [37] highlights out the advantage in sparse gradients. Therefore, we establish another hypothesis as follows.

H3 In comparing the performance of multiple optimizers, Adamax achieves the highest average test score.

Although prior studies have compared different forms of LLMs in terms of prediction accuracy, the research landscape in applied domains remains somewhat lacking. In particular, BERT comes in various forms, distinguished by scale (params ${ }_{B E R T}$ : number of parameters in BERT) and advanced models (e.g., RoBERTa and DistilBERT). The question arises regarding BERT's impact on performance: whether it merely
serves as a means of acquiring deep-contextualized word representations, or if prediction accuracy can be further improved by employing more advanced or larger-scale BERT models. Therefore, we establish two hypotheses, respectively.

H4-1 The prediction accuracy improves on average with larger-scale pre-training models.

H4-2 The prediction accuracy improves on average with newer pre-training models.

Lastly, unlike tabular data, textual data differs significantly in the amount of information conveyed in each post (e.g., while one post contains only one-word impressions, another might provide detailed information about the user's situation and background). This variability poses a challenge for prediction accuracy, as illustrated by the differing amounts of information that the post retains. For example, regarding the two review texts shown in Fig. [210, even though they are all text data, the amount of information each holds is completely different. Similar issues have been pointed out in marketing fields,

```
[User A] Rate: 1/5
    Review: Disappointed
[User B] Rate: 1/5
    Review: Disappointed with this place as we were
    treated horribly. Although it was recommended as
    an ideal spot for couples, the staff members seemed
    noticeably unprepared for the sufficient service.
```

Fig. 2. Amount of information in the review text

for example, one study [38] pointed out that the prediction using behavioral logs may vary in accuracy depending on the quantity of services used. This problem is anticipated to arise in multimodal learning with textual data as well. Therefore, in this study, we also assess the change in prediction accuracy based on the number of tokens in the textual input. Thus, we propose the following hypothesis.

H5 Prediction accuracy decreases with fewer tokens in multimodal learning.

## III. MODEL

## A. Architecture

This study addresses both textual and tabular data which needs multiple inputs. The network is divided into three subnets based on their roles: X1-, X2-, and Output-subnet. $\mathrm{X} 1$ - and X2-subnet process each modality with appropriate structures, and Output-subnet concatenates them and predicts the values in the upper layers (Fig. 3, $b s$ : batch size, $J$ : number of tabular variables).

First, in the X1-subnet handling textual data, BERT and a tokenizer are employed to acquire deep-contextualized word representations. As discussed, using the pooler-output in multimodal learning may not always be optimal as it could lead to dimensionality reduction based on the [CLS] token in BERT.[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-04.jpg?height=913&width=702&top_left_y=210&top_left_x=1167)

Fig. 3. Context-Aware Model

Despite the possible for selecting necessary features through multihead attentions within cross-attention, in this model, we opt for the state of the final hidden layer in BERT as the BERT

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-04.jpg?height=46&width=876&top_left_y=1408&top_left_x=1080)
in the X2-subnet managing tabular data, while feed-forward layers can be incorporated, the input data should not be overly processed prior to feature fusion. Therefore, we choose to directly feed the input into the Output-subnet.

The Output-subnet receives these two representations, fuses them, and generates outputs. While prior studies [12], [13] have utilized both STA and layer-wise concatenation, if the STA mechanism adequately captures the features of two modalities, it is uncertain whether early fusion is necessary to obtain a joint representation. Hence, our proposed model adopts the cross-attention Transformer encoder with eight attention heads. This mechanism is anticipated to yield high accuracy without feature fusion, as it captures the relationship between the two modalities from multiple perspectives, which can be challenging with a single attention mechanism. In this study, this proposed model is called the context-aware model.

## B. Evaluation

To evaluate the effectiveness of our proposed model, we construct two multimodal learning models and two monomodal models as reference points. First, for the multimodal approach, we introduce a context-fusion model (referred to Fig. 4a), which integrates feature fusion into the contextaware model. Additionally, we implement a typical multimodal model with feature fusion using pooler-output (refer to Fig.

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-05.jpg?height=559&width=423&top_left_y=195&top_left_x=165)

(a) Context-fusion model

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-05.jpg?height=561&width=440&top_left_y=191&top_left_x=604)

(b) Feature-fusion model
Fig. 4. Reference models (multimodal)

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-05.jpg?height=558&width=425&top_left_y=911&top_left_x=167)

(a) Textual model

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-05.jpg?height=556&width=420&top_left_y=909&top_left_x=619)

(b) Tabular model
Fig. 5. Reference models (monomodal)

4b) 2 . Since the feature-fusion model directly receives the highdimensional representation from the BERT output, the number of hidden layers in the output layer post feature fusion is increased to three, with each layer comprising 512,256 , and 128 units.

Second, for the monomodal models, we introduce the textual and tabular models (referred to Fig. 5a, 5b, which process modality-specific layers and transmit them to the output layer without traversing through Transformer or feature-fusion architectures. Moreover, we incorporate two benchmarks: a linear regression model that captures linear relationships and a random model that generates random predictions within the range of $[0,1]$, and. These six reference models allows for a comprehensive comparison of the performance of the proposed models.

In terms of optimization, many existing studies have adopted Adam [37] as an optimizer; however, as described in H2, it has yet to be clarified what optimizer is effective for a complex architecture of multimodal learning. Hence, this study delves into the impact of different optimizers on prediction accuracy,[^1]

including Adam, Nesterov-accelerated Adaptive Moment Estimation (Nadam) [39], and Adamax.

Furthermore, regarding the pre-trained BERT model, we initially employ bert-base-uncased among several pre-trained models to demonstrate the superior prediction accuracy of our proposed model's architecture compared to others (Study 1). Subsequently, we explore changes in accuracy by replacing bert-base-uncased with different pre-trained models (Study 2).

TABLE I

MODEL SETTINGS

| Parameters | Values |
| :--- | :--- |
| Hyper-Parameters | 500 |
| Number of Epochs | 256 |
| Batchsize | Adamax |
| Optimizer | mean squared error (MSE) |
| Loss Function |  |
| X1-subnet | BERT |
| Sert-base-uncased |  |
| Pre-trained Model | 768 |
| params $B E R T$ | 0 |
| X2-subnet | 15 |
| Number of Hidden Layers |  |
| Number of Input Features $(J)$ | used in 'Context-Aware' and 'Context-Fusion' |
| Feature Fusion | 8 |
| Cross-Attention | used in 'Context-Fusion' and 'Feature Fusion' |
| Layer-wise Concatenation |  |
| Output-subnet | tanh |
| Activation (hidden layers) | $2-3$ (cf. Fig. 1 |
| Activation (output) |  |
| Hidden Layers |  |

Note. tanh stands for hyperbolic tangent function.

## C. Data Description

To validate the efficacy of the proposed model, we require behavioral log data containing both textual and tabular information. For this purpose, we utilize the Yelp Open Dataset [40]. Yelp, an online platform, offers a wealth of information about various venues including restaurants, stores, and public facilities, alongside user ratings and reviews. The dataset comprises user review texts, profiles, and venue details. Each location is associated with one or more category tags, facilitating the extraction of target locations by specifying these tags. In our study, we focus on three business categories to demonstrate the robustness of the model: Restaurants (tagged with "Restaurants", but not neither with "Fast Food", "Food Truck", "Nightlife", and "Bar"), Nightlife (tagged with both "Restaurants" and "Nightlife", but not neither with "Fast Food" and "Food Truck"), and Café (tagged with both "Cafes" and "Coffee and Tea", but not neither with "Fast Food" and "Food Truck"). In particular, Nightlife category encompasses various types of establishments such as bars and nightclubs, posing challenges in evaluation solely based on store information.

For the sake of data acquisition convenience, we predict the ratings (i.e., the number of stars) of restaurants using review texts, user profile information, and restaurant information. While the target variable can be readily obtained from the app, its accurate prediction by our proposed model signifies
its suitability in understanding consumer preferences and its potential extension into an effective recommender system [5].

We randomly sample 10,000 posts of ratings and reviews from each category containing one or more English words in year 2018. In cases where a user reviews the same location multiple times, we consider only the latest post. For textual data preprocessing, we replace line breaks, emojis, icons, and other symbols with periods and merge continuous sequences of periods into a single period. A summary of the dataset statistics is provided in Table II Notably, there are no duplications for the location in the Restaurants and Nightlife categories.

TABLE II

StATISTICS OF THE CATEGORIES

| Category | \#Users | \#Spots | \#Stars |  |  | \#Tokens |  |  |  |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  | Mean | Std |  | Mean | Std | Min | Max |
| Restaurants | 9567 | 1387 | 3.9 | 1.4 |  | 110.3 | 91.6 | 9 | 512 |
| Nightlife | 9491 | 2097 | 3.9 | 1.4 |  | 119.9 | 100.1 | 11 | 512 |
| Café | 9189 | 665 | 4.2 | 1.2 |  | 109.9 | 90.5 | 10 | 512 |

Note. \#Users and \#Spots indicate the unique numbers of users and restaurants in each category, respectively.

The dataset of each category consists of $D=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{n}$ with a sample size of $n=10000$, where each input $x_{i}$ comprises one textual variable and $J$ tabular variables, denoted as $x_{i}=\left(x_{i}^{(t e x t)}, x_{i}^{(t a b)}\right)=\left(x_{1 i}^{(t e x t)}, x_{1 i}^{(t a b)}, x_{2 i}^{(t a b)}, \ldots, x_{J i}^{(t a b)}\right) \in$ $\mathbb{X}$. The target variable $y_{i} \in \mathbb{Y}=[0,1]$ represents a normalized value of the ratings, which is scaled between 0 and 1 from a range of 1 to 5 stars. The variables are shown in Table III.

The dataset of 10,000 observations is split into training $(70 \%)$, validation ( $15 \%$ ), and test (15\%) subsets. During training, the loss function employed is mean squared error (MSE), while model performance is evaluated using root mean squared error (RMSE) using actual and predicted values $\left(y_{i}, \hat{y}_{i}\right)$ as follows:

$$
\begin{align*}
\mathrm{MSE} & =\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}  \tag{7}\\
\mathrm{RMSE} & =\sqrt{\mathrm{MSE}} \tag{8}
\end{align*}
$$

Detailed model settings are provided in Table $\square$

## IV. ReSUlts AND DisCUSSION

## A. Study 1: Comparison Across the Model Architectures

The results are presented in Table IV, reveal a similar pattern across all categories. The proposed model consistently achieves the highest prediction accuracy in the test scores across all categories. The context-fusion model follows closely behind, while the performance of the feature-fusion model sometimes lags behind that of the textual and even linear regression models. In particular, despite the context-fusion model having the largest number of parameters in this study. Context-Fusion model fuses the representations twice with STA-Transformer and feature fusion, but the contribution on the performance is actually limited. Conversely, the random model exhibits the lowest accuracy, followed by the linear regression model in most cases.
TABLE III

| Variable Name | Description |
| :---: | :---: |
| $\mathbf{Y}$ : Target Variable $(b s, 1)$ |  |
| - rating | Rating value posted on Yelp ${ }^{\dagger}$ |
| X1: Textual Variable $(b s, 1)$ |  |
| - review | Review text posted on Yelp tokenized <br> with fixed-length of $l e n_{\max }=512$ tokens |
| X2: Tabular Variables shape $(b s, 15)$ |  |
| Location <br> - open_dow <br> - <br> open_hours | ![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-06.jpg?height=152&width=561&top_left_y=644&top_left_x=1346) |
| - open_mon <br> . . <br> - open_sun | A number of opening hours in Monday ${ }^{\dagger}$ <br> A number of opening hours in Sunday ${ }^{\dagger}$ |
| User <br> $-n \_f r i e n d s$ <br> $-n \_f a n s$ <br> $-n \_e l i t e s$ | A number of friends <br> A number of getting fans <br> A number of getting elite |
| Post <br> - n_useful <br> - n_funny <br> - n_cool | A number of getting useful <br> A number of getting funny <br> A number of getting cool |

MODEL VARIABLES

${ }^{\dagger}$ Variables are normalized in $[0,1]$.

Although some reference models stopped training in fewer epochs than the proposed model, this trend does not necessarily indicate early convergence due to the absence of early stopping [41]. Rather, it suggests that these models struggled to escape local convergence in the early stages 3 . These results guarantee the generalized performance of the proposed model, and thus, $\mathbf{H 1}$ is supported.

Second, Table $\mathrm{V}$ provides an overview of the average performance considering various perspectives: such as target categories, modalities, and optimizers. As anticipated, the Nightlife category exhibits slightly lower test performance than the Restaurants category, possibly due to the diverse nature of establishments in the Nightlife category. Nonetheless, the mean score by modality indicates a high level of predictability, underscoring the usefulness of analyzing multiple modalities. This result supports $\mathbf{H 2}$.

Third, the results in Table $\square$ further demonstrate the effectiveness of adamax as an optimizer, particularly in handling the complex structure of neural networks dealing with sparse textual representations. Despite taking longer for training, adamax proves considerably more effective. Notably, even with an enormous number of parameters, adamax demonstrates superior performance in effectively updating the weights. A comparison of the change in losses of the context-aware model among different optimizers in Fig. 6, corroborates these findings. Adamax shows outstanding effectiveness over the training epochs. The progression of learning in the three[^2]

TABLE IV

RESULTS (WITH ADAMAX OPTIMIZER, ASCENDING IN TEST RMSE)

|  | Model | Modality | BERT Model | Optimizer | Train | Validation | Test | Epochs | Training Time | \#Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-07.jpg?height=242&width=49&top_left_y=349&top_left_x=203) | Multimodal | context-aware | bert-base-uncased | adamax | 0.085 | 0.135 | 0.132 | 316 | 04:01:19 | $119,122,520$ |
|  | Multimodal | context-fusion | bert-base-uncased | adamax | 0.109 | 0.130 | 0.134 | 125 | $01: 50: 58$ | $119,123,080$ |
|  | X1-modal | textual | bert-base-uncased | adamax | 0.151 | 0.149 | 0.143 | 499 | 05:53:48 | $109,712,129$ |
|  | Multimodal | feature-fusion | bert-base-uncased | adamax | 0.152 | 0.152 | 0.155 | 290 | 03:33:04 | $110,048,001$ |
|  | X2-modal | tabular | bert-base-uncased | adamax | 0.258 | 0.260 | 0.261 | 313 | $00: 01: 05$ | 281 |
|  |  |  | Linear regression: |  | 0.259 | 0.261 | 0.262 |  |  |  |
|  |  |  |  | Random: | 0.494 | 0.496 | 0.503 |  |  |  |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-07.jpg?height=242&width=46&top_left_y=596&top_left_x=203) | Multimodal | context-aware | bert-base-uncased | adamax | 0.084 | 0.127 | 0.140 | 401 | 05:22:41 | $119,122,520$ |
|  | Multimodal | context-fusion | bert-base-uncased | adamax | 0.093 | 0.129 | 0.141 | 406 | $05: 25: 48$ | $119,123,080$ |
|  | X1-modal | textual | bert-base-uncased | adamax | 0.150 | 0.141 | 0.150 | 476 | $05: 33: 54$ | $109,712,129$ |
|  | Multimodal | feature-fusion | bert-base-uncased | adamax | 0.141 | 0.144 | 0.161 | 423 | $05: 02: 02$ | $110,048,001$ |
|  | X2-modal | tabular | bert-base-uncased | adamax | 0.255 | 0.254 | 0.257 | 471 | $00: 01: 31$ | 281 |
|  |  |  | Linear regression: |  | 0.262 | 0.259 | 0.260 |  |  |  |
|  |  |  |  | Random: | 0.481 | 0.480 | 0.482 |  |  |  |
| Uّש | Multimodal | context-aware | bert-base-uncased | adamax | 0.076 | 0.127 | 0.125 | 475 | $06: 08: 27$ | $119,122,520$ |
|  | Multimodal | context-fusion | bert-base-uncased | adamax | 0.074 | 0.125 | 0.127 | 480 | 06:02:39 | $119,123,080$ |
|  | Multimodal | feature-fusion | bert-base-uncased | adamax | 0.137 | 0.147 | 0.142 | 213 | $02: 43: 10$ | $110,048,001$ |
|  | X1-modal | textual | bert-base-uncased | adamax | 0.137 | 0.141 | 0.147 | 491 | $06: 10: 34$ | $109,712,129$ |
|  | X2-modal | tabular | bert-base-uncased | adamax | 0.231 | 0.241 | 0.228 | 234 | $00: 01: 03$ | 281 |
|  |  |  | Linear regression: |  | 0.231 | 0.236 | 0.223 |  |  |  |
|  |  |  |  | Random: | 0.509 | 0.500 | 0.516 |  |  |  |

Note. Bold type represents the best model for the indices. Training Time shows the actual duration for the best validation score in $h h: m m$ :ss using the same environment (GPU: NVIDIA A100-SXM4-40GB).

TABLE V

AVERAGE PERFORMANCES BY THE GROUP

| Group | Train | Validation | Test | Epochs | Training Time | \#Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Category |  |  |  |  |  |  |
| Café | $\mathbf{0 . 1 5 9}$ | $\mathbf{0 . 1 7 8}$ | $\mathbf{0 . 1 7 5}$ | 340.5 | $03: 14: 07$ | $91,601,202.2$ |
| Restaurants | 0.193 | 0.207 | 0.209 | $\mathbf{2 7 8 . 5}$ | $\mathbf{0 2 : 4 8 : 3 3}$ | $91,601,202.2$ |
| Nightlife | 0.208 | 0.218 | 0.226 | 321.4 | $03: 07: 13$ | $91,601,202.2$ |
| Modality |  |  |  |  |  |  |
| context-aware | $\mathbf{0 . 1 0 0}$ | $\mathbf{0 . 1 3 3}$ | $\mathbf{0 . 1 4 0}$ | 394.7 | $05: 01: 51$ | $119,122,520.0$ |
| context-fusion | 0.108 | 0.133 | 0.141 | 400.2 | $05: 02: 31$ | $119,123,080.0$ |
| Multimodal | 0.153 | 0.174 | 0.179 | 320.9 | $04: 02: 31$ | $116,097,867.0$ |
| X1-modal | 0.228 | 0.230 | 0.229 | 312.7 | $03: 08: 03$ | $109,712,129.0$ |
| X2-modal | 0.249 | 0.253 | 0.251 | 292.0 | $\mathbf{0 0 : 0 0 : 5 3}$ | 281.0 |
| feature-fusion | 0.250 | 0.255 | 0.257 | $\mathbf{1 6 7 . 8}$ | $02: 03: 12$ | $110,048,001.0$ |
| Optimizer |  |  |  |  |  |  |
| Adamax | $\mathbf{0 . 1 4 2}$ | $\mathbf{0 . 1 6 0}$ | $\mathbf{0 . 1 6 3}$ | 374.2 | $03: 29: 54$ | $91,601,202.2$ |
| Nadam | 0.202 | 0.214 | 0.217 | 309.1 | $\mathbf{0 2 : 4 8 : 0 2}$ | $91,601,202.2$ |
| Adam | 0.217 | 0.228 | 0.231 | $\mathbf{2 5 7 . 1}$ | $02: 51: 57$ | $91,601,202.2$ |

Note. Bold type represents the best model for the indices. Training Time shows the actual duration for the best validation score in hh:mm:ss using the same environment (GPU: NVIDIA A100-SXM4-40GB)

categories shows that adamax does not always train efficiently from the early stages; however, as it proceeds to the later stages, only adamax continues to reduce the loss while the other optimizers converges locally. Thus, H3 is supported.

## B. Study 2: Impact of Replacing Pre-Trained Models

The results from Study 1 demonstrate the effectiveness of our proposed architecture; however, even with its high accuracy in multimodal learning, the model relies on BERTBase-Uncased component. To further investigate the impact of different pre-trained models, we conducted additional analyses by replacing the BERT component with BERT-Large-Uncased,

![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-07.jpg?height=648&width=876&top_left_y=1237&top_left_x=1080)

Fig. 6. Training process by different optimizers

RoBERTa-Base, and RoBERTa-Large within context-aware model.

The findings, presented in Table VI confirm a significant improvement in test performance on average with BERTLarge-Uncased and RoBERTa-Base compared to Bert-BaseUncased across all three categories. Both RoBERTa-Base and BERT-Large-Uncased contribute to the accuracy, while RoBERTa-Large does not exhibit the same level of improvement. The average test scores suggest that both RoBERTaBase and BERT-Large-Uncased demonstrate comparable generalization capabilities, with RoBERTa-Base outperforming in terms of convergence time. The lower accuracy observed with the RoBERTa-Large component could be attributed to

TABLE VI

IMPACT OF THE PRE-TRAINED MODELS (WITH ADAMAX OPTIMIZER, ASCENDING IN TEST RMSE)

|  | Modality | BERT Model | Optimizer | Train | Validation | Test | Epochs | Training Time | \#Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-08.jpg?height=151&width=47&top_left_y=358&top_left_x=210) | context-aware | bert-large-uncased | adamax | 0.079 | 0.129 | 0.121 | 496 | 16:16:07 | $347,927,896$ |
|  | context-aware | roberta-base | adamax | 0.091 | 0.133 | 0.131 | 319 | 04:11:54 | $134,285,912$ |
|  | context-aware | bert-base-uncased | adamax | 0.085 | 0.135 | 0.132 | 316 | 04:01:19 | $119,122,520$ |
|  | context-aware | roberta-large | adamax | 0.076 | 0.144 | 0.150 | 490 | 16:00:06 | $368,145,752$ |
| $\frac{9}{7}$ <br> $\bar{z}$ <br> $\bar{z}$ | context-aware | roberta-b: | adamax | 0.095 | 0.120 | 0.130 | 176 | 02:23:58 | $134,285,912$ |
|  | context-aware | bert-large-uncased | adamax | 0.085 | 0.123 | 0.135 | 416 | 16:29:20 | $347,927,896$ |
|  | context-aware | roberta-large | adamax | 0.084 | 0.125 | 0.136 | 401 | 13:14:30 | $368,145,752$ |
|  | context-aware | bert-base-uncased | adamax | 0.084 | 0.127 | 0.140 | 401 | 05:22:41 | $119,122,520$ |
| نֶ | context-aware | roberta-base | adamax | 0.077 | 0.118 | 0.120 | 377 | 04:51:41 | $134,285,912$ |
|  | context-aware | bert-large-uncased | adamax | 0.071 | 0.122 | 0.124 | 494 | 16:16:29 | $347,927,896$ |
|  | context-aware | bert-base-uncased | adamax | 0.076 | 0.127 | 0.125 | 475 | 06:08:27 | $119,122,520$ |
|  | context-aware | roberta-large | adamax | 0.082 | 0.128 | 0.137 | 238 | 07:58:23 | $368,145,752$ |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-08.jpg?height=137&width=64&top_left_y=831&top_left_x=203) | context-aware | bert-large-uncased | adamax | 0.078 | 0.125 | 0.127 | 468.7 | 16:20:39 | $347,927,896$ |
|  | context-aware | roberta-base | adamax | 0.088 | 0.123 | 0.127 | 290.7 | 03:49:11 | $134,285,912$ |
|  | context-aware | bert-base-uncased | adamax | 0.082 | 0.130 | 0.132 | 397.3 | 05:10:49 | $119,122,520$ |
|  | context-aware | roberta-large | adamax | 0.081 | 0.132 | 0.141 | 376.3 | 12:24:20 | $368,145,752$ |

Note. Bold type represents the best model for the indices. Training Time shows the actual duration for the best validation score in $h h: m m$ :ss using the same environment (GPU: NVIDIA A100-SXM4-40GB).

the insufficient sample size relative to the complexity of the architecture. Previous studies have indicated that largescale models like RoBERTa-Large require a larger sample size for optimal performance. Thus, both $\mathbf{H} 4$-1 and $\mathbf{H} 4-2$ are supported, respectively.

## C. Study 3: Impact of the Number of Tokens

Finally, we examine the impact of the amount of information in the review text on prediction accuracy, as described in H41 and H4-2. We regard the number of tokens in the review as a measure of information and investigate whether accuracy varies with the number of tokens. The best model from Study 1 (context-aware model with bert-base-uncased and the adamax optimizer) is utilized for each category. We set up the token strata by dividing three subsets of training, validation, and test data into $20 \%$ according to the number of tokens. Then, we predict and compute the average RMSE by strata.

The results, categorized by the number of tokens and by stratum are shown in Table VIH. For the test data alone, the prediction accuracy is highest when the number of tokens is lowest in the Restaurants category, and medium in the other two categories. This suggests that while multimodal learning of textual and tabular data is expected to improve prediction accuracy, it does not always require a large amount of textbased information. For two categories other than Restaurants, the prediction accuracy is also best in the lowest tokens strata in training and validation. However, in the Nightlife and Cafécategories, which exhibit wide variation in location attributes, higher numbers of tokens ensure generalizability in test performance, whereas the model for the Restaurants category demonstrates high generalizability with fewer tokens. In addition, regarding the observed decrease in accuracy with[^3]

particularly large numbers of tokens, several possible reasons exist. First, we cut off sentences with more than 512 tokens due to the size of BERT's context window, which may not convey enough information to the model. Second, excessively long texts may contain redundant information unrelated to the user ratings, leading to that the model has not properly discerned the information. Thus, $\mathbf{H 5}$ is not supported.

TABLE VII

IMPACT OF THE NUMBER OF TOKENS (WITH ADAMAX OPTIMIZER, ASCENDING IN THE NUMBER OF TOKENS)

|  | Train |  | Validation |  | Test |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | $M$ | $\overline{R M S E}$ | $\bar{M}$ | RMSE | $\bar{M}$ | RMSE |
| ![](https://cdn.mathpix.com/cropped/2024_06_04_0a3699d89150659e1cd1g-08.jpg?height=158&width=38&top_left_y=1750&top_left_x=1107) | 31.5 | 0.067 | 30.4 | 0.109 | 31.8 | 0.115 |
|  | 54.1 | 0.068 | 54.3 | 0.130 | 55.5 | 0.125 |
|  | 81.8 | 0.069 | 81.8 | 0.130 | 82.9 | 0.127 |
|  | 127.1 | 0.075 | 125.7 | 0.135 | 131.8 | 0.147 |
|  | 258.2 | 0.086 | 262.7 | 0.165 | 260.3 | 0.154 |
| $\underbrace{0}_{Z}$ | 31.8 | 0.125 | 33.3 | 0.118 | 33.0 | 0.137 |
|  | 56.2 | 0.132 | 56.7 | 0.130 | 59.1 | 0.137 |
|  | 88.4 | 0.139 | 88.8 | 0.141 | 90.3 | 0.102 |
|  | 139.1 | 0.140 | 140.6 | 0.140 | 136.6 | 0.147 |
|  | 284.7 | 0.161 | 288.8 | 0.152 | 282.4 | 0.156 |
| نَّ | 31.1 | 0.112 | 32.4 | 0.131 | 31.0 | 0.114 |
|  | 53.5 | 0.121 | 56.0 | 0.120 | 53.7 | 0.115 |
|  | 80.5 | 0.128 | 86.2 | 0.125 | 84.6 | 0.109 |
|  | 123.7 | 0.128 | 133.8 | 0.130 | 130.7 | 0.123 |
|  | 255.9 | 0.147 | 261.3 | 0.154 | 261.2 | 0.170 |

Note. $M$ represents the mean number of tokens in the strata. Bold type represents the best score in the each dataset.

## V. CONCLUSION

## A. Contribution

In this study, we propose a novel multimodal deep learning model that integrates posted review texts with tabular data,
including user profiles and location information. This model effectively captures consumer heterogeneity to predict user ratings on locations with high accuracy. In addition, we conduct a comprehensive analysis of different pre-trained models and the effect of token count on prediction accuracy.

Our proposed model consistently outperforms reference models on test data across all categories. This result indicates the superiority of contextual understanding facilitated by the cross-attention over mere feature fusion for joint representation. Despite prior studies confirming the efficacy of multimodal learning in the various field, in this study, feature fusion which is a simple form of multimodal learning does not overtake of single-modality models. This limitation may stem from the complexity of features, as even with a substantial number of units in the Output-subnet, the large-scale deepcontextualized word representations may overwhelm the upper hidden layers. This result indicates the limitations of simple feature-fusion methods, and as the complexity of the features to be combined reflects, sophisticated mechanisms are needed to understand them.

In addition, our proposed model exclusively utilizes the cross-attention, unlike previous research that emphasizes the combination of features through both attention and feature fusion [12]. Our results demonstrate that achieving higher accuracy is feasible with the cross-attention alone. By establishing causality between different modalities as the source and target, the model can effectively attend to large and sparse features. Although our study focuses on predicting ratings due to data availability, it highlights the potential to construct models based on an accurate understanding of user preferences.

Extending the proposed model presented in this study opens the door to addressing various advanced tasks, such as a model that recommends the appropriate content based on user's past posts and profile and another model that predicts future repeated purchases based on a consumer's past product reviews and purchase history on the EC platforms.

## B. Challenges

Our model still encounters challenges in improving prediction accuracy, primarily due to computational limitations. All BERT layers in our study remain frozen (i.e., parameters are set to non-trainable) during the training process due to these limitations. In addition, newly developed LLMs are proposed one after another. That is, the model can be further improved through structural refinements, such as selecting different LLMs, fine-tuning BERT layers, incorporating additional dropout, adjusting the number and shape of hidden layers, and optimizing other hyper-parameters.

Finally, despite the use of LLMs, the handling a large number of tokens remains difficult. Our study suggests that an excessive number of tokens may actually decrease prediction accuracy. To address the issue, appropriate measures must be taken, such as pre-summarizing large amounts of text data or using LLMs with larger context windows. It is worth noting that such analyses require additional computational resources and training time, which is a problem to be balanced with prediction accuracy.

## ETHICAL STATEMENT

This study only uses academic open data and does not additionally collect personally identifiable information. We observe the terms of use of the dataset and manage the data in a secure environment.

## ACKNOWLEDGMENT

Our comprehensive analyses were implemented on RAIDEN, a computing infrastructure hosted by RIKEN AIP. We would like to express our gratitude to all the members of the center who maintain the system. Additionally, we extend our gratitude to Yelp which enriched our study by providing the open data.

## REFERENCES

[1] S. Zhang, L. Yao, A. Sun, and Y. Tay, "Deep learning based recommender system: A survey and new perspectives," ACM computing surveys (CSUR), vol. 52, no. 1, pp. 1-38, 2019.

[2] H. Ko, S. Lee, Y. Park, and A. Choi, "A survey of recommendation systems: recommendation models, techniques, and application fields," Electronics, vol. 11, no. 1, p. 141, 2022.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," Advances in neural information processing systems, vol. 30, 2017.

[4] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," arXiv preprint arXiv:1810.04805, 2018.

[5] Y. Zhuang and J. Kim, "A bert-based multi-criteria recommender system for hotel promotion management," Sustainability, vol. 13, no. 14, p. 8039,2021

[6] J. Niimi, "Multimodal deep learning of word-of-mouth text and demographics to predict customer rating: Handling consumer heterogeneity in marketing," arXiv preprint arXiv:2401.11888, 2024.

[7] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," arXiv preprint arXiv:1409.0473, 2014.

[8] R. A. Baten, Y. Liu, H. Peters, F. Barbieri, N. Shah, L. Neves, and M. W. Bos, "Predicting future location categories of users in a large social platform," in Proceedings of the International AAAI Conference on Web and Social Media, vol. 17, 2023, pp. 47-58.

[9] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," arXiv preprint arXiv:1607.06450, 2016.

[10] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770-778.

[11] D. Soydaner, "Attention mechanism in neural networks: where it comes and where it goes," Neural Computing and Applications, vol. 34, no. 16, pp. $13371-13385,2022$.

[12] M. Ala'raj, M. F. Abbod, and M. Majdalawieh, "Modelling customers credit card behaviour using bidirectional 1stm neural networks," Journal of Big Data, vol. 8, no. 1, pp. 1-27, 2021.

[13] J. Niimi, "Prediction of behavioral loyalty using different dimensionality data: Multimodal deep learning with transformer encoder and serial feature fusion," Japanese Journal of Applied Statistics, vol. 53, no. 1, 2024.

[14] A. Khattar and S. Quadri, "Camm: Cross-attention multimodal classification of disaster-related tweets," IEEE Access, vol. 10, pp. 92 889-92 902, 2022.

[15] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, "Deep contextualized word representations," in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), M. Walker, H. Ji, and A. Stent, Eds. New Orleans, Louisiana: Association for Computational Linguistics, 2018, pp. 2227-2237. [Online]. Available: https://aclanthology.org/N18-1202

[16] T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," arXiv preprint arXiv:1301.3781, 2013.

[17] Z. H. Kilimci, "Prediction of user loyalty in mobile applications using deep contextualized word representations," Journal of Information and Telecommunication, vol. 6, no. 1, pp. 43-62, 2022.

[18] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "Roberta: A robustly optimized bert pretraining approach," arXiv preprint arXiv:1907.11692, 2019.

[19] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv, 2019.

[20] N. Srivastava and R. R. Salakhutdinov, "Multimodal learning with deep boltzmann machines," Advances in neural information processing systems, vol. 25, 2012.

[21] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng, "Multimodal deep learning," in Proceedings of the 28th international conference on machine learning (ICML-11), 2011, pp. 689-696.

[22] K. Bayoudh, R. Knani, F. Hamdaoui, and A. Mtibaa, "A survey on deep multimodal learning for computer vision: advances, trends, applications, and datasets," The Visual Computer, vol. 38, no. 8, pp. 2939-2970, 2022.

[23] L. I. Kuncheva, Combining pattern classifiers: methods and algorithms. John Wiley \& Sons, 2014.

[24] L. Nanni, S. Ghidoni, and S. Brahnam, "Handcrafted vs. non-handcrafted features for computer vision classification," Pattern Recognition, vol. 71, pp. 158-172, 2017.

[25] F. S. Abousaleh, W.-H. Cheng, N.-H. Yu, and Y. Tsao, "Multimodal deep learning framework for image popularity prediction on social media," IEEE Transactions on Cognitive and Developmental Systems, vol. 13, no. 3, pp. 679-692, 2020

[26] F. Ofli, F. Alam, and M. Imran, "Analysis of social media data using multimodal deep learning for disaster response," arXiv preprint arXiv:2004.11838, 2020.

[27] L. Zhang, J. Shen, J. Zhang, J. Xu, Z. Li, Y. Yao, and L. Yu, "Multimodal marketing intent analysis for effective targeted advertising," IEEE Transactions on Multimedia, vol. 24, pp. 1830-1843, 2021.

[28] S. I. Lee and S. J. Yoo, "Multimodal deep learning for finance: integrating and forecasting international stock markets," The Journal of Supercomputing, vol. 76, pp. 8294-8312, 2020.

[29] T. Mittal, U. Bhattacharya, R. Chandra, A. Bera, and D. Manocha, "M3er: Multiplicative multimodal emotion recognition using facial, textual, and speech cues," in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 02, 2020, pp. 1359-1367.

[30] -, "M3er: Multiplicative multimodal emotion recognition using facial, textual, and speech cues," in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 02, 2020, pp. 1359-1367.

[31] M. G. Huddar, S. S. Sannakki, and V. S. Rajpurohit, "Attention-based multimodal contextual fusion for sentiment and emotion classification using bidirectional 1stm," Multimedia Tools and Applications, vol. 80, pp. 13 059-13 076, 2021.

[32] P. C. Wu and Y.-C. Wang, "The influences of electronic word-of-mouth message appeal and message source credibility on brand attitude," Asia Pacific Journal of Marketing and Logistics, vol. 23, no. 4, pp. 448-472, 2011.

[33] A. N. Albarq, "Measuring the impacts of online word-of-mouth on tourists' attitude and intentions to visit jordan: An empirical study," International Business Research, vol. 7, no. 1, p. 14, 2014.

[34] J. Mohammad, F. Quoquab, R. Thurasamy, and M. N. Alolayyan, "The effect of user-generated content quality on brand engagement: The mediating role of functional and emotional values," Journal of Electronic Commerce Research, vol. 21, no. 1, pp. 39-55, 2020.

[35] A. J. Kim and K. K. Johnson, "Power of consumers using social media: Examining the influences of brand-related user-generated content on facebook," Computers in human behavior, vol. 58, pp. 98-108, 2016.

[36] L. Zhen, P. Hu, X. Peng, R. S. M. Goh, and J. T. Zhou, "Deep multimodal transfer learning for cross-modal retrieval," IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 2, pp. 798-810, 2020.

[37] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.

[38] P. S. Fader, B. G. Hardie, and K. L. Lee, "Rfm and clv: Using isovalue curves for customer base analysis," Journal of marketing research, vol. 42, no. 4, pp. 415-430, 2005.

[39] T. Dozat, "Incorporating nesterov momentum into adam," 2016.

[40] Yelp, Yelp Open Dataset, An all-purpose dataset for learning (https://www.yelp.com/dataset, accessed Nov. 20th, 2023), 2022.

[41] L. Prechelt, "Early stopping-but when?" in Neural Networks: Tricks of the trade. Springer, 1998, pp. 55-69.


[^0]:    ${ }^{1}$ Since the dataset employed in this study prohibits the disclosure of actual review sentences, the texts shown in the figure are fictitious ones written by the authors.

[^1]:    ${ }^{2}$ We employ pooler-output for the feature-fusion model due to the requirement of two-dimensional representations for layer-wise concatenation

[^2]:    ${ }^{3}$ A similar tendency is confirmed in Fig. 6 Noted that it is about a different analysis.

[^3]:    ${ }^{4}$ Note that the values shown in the table represent the prediction accuracy for trained, validated, and test samples without any further training.

</end of paper 2>


