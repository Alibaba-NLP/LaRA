<paper 0>
# Influence of Neighborhood on the Preference of an Item in eCommerce Search 

$1^{\text {st }}$ Saratchandra Indrakanti<br>eBay Inc.<br>San Jose, California, USA<br>sindrakanti@ebay.com<br>$2^{\text {nd }}$ Svetlana Strunjas<br>eBay Inc.<br>San Jose, California, USA<br>sstrunjas @ebay.com<br>$3^{\text {rd }}$ Shubhangi Tandon<br>eBay Inc.<br>San Jose, California, USA<br>shtandon@ebay.com<br>$4^{\text {th }}$ Manojkumar Kannadasan<br>eBay Inc.<br>San Jose, California, USA<br>mkannadasan@ebay.com

## Influence of Neighborhood on the Preference of an Item in eCommerce Search


#### Abstract

Surfacing a ranked list of items for a search query to help buyers discover inventory and make purchase decisions is a critical problem in eCommerce search. Typically, items are independently predicted with a probability of sale with respect to a given search query. But in a dynamic marketplace like eBay, even for a single product, there are various different factors distinguishing one item from another which can influence the purchase decision for the user. Users have to make a purchase decision by considering all of these options. Majority of the existing learning to rank algorithms model the relative relevance between labeled items only at the loss functions like pairwise or list-wise losses [1]-[3]. But they are limited to point-wise scoring functions where items are ranked independently based on the features of the item itself. In this paper, we study the influence of an item's neighborhood to its purchase decision. Here, we consider the neighborhood as the items ranked above and below the current item in search results. By adding delta features comparing items within a neighborhood and learning a ranking model, we are able to experimentally show that the new ranker with delta features outperforms our baseline ranker in terms of Mean Reciprocal Rank (MRR) [4]. The ranking models with proposed delta features result in $3-5 \%$ improvement in MRR over the baseline model. We also study impact of different sizes for neighborhood. Experimental results show that neighborhood size 3 perform the best based on MRR with an improvement of $4-5 \%$ over the baseline model.


Index Terms-eCommerce, search, ranking, information retrieval, list-wise, group-wise

## I. INTRODUCTION

Search ranking is a widely studied problem in both academia and industry. A lot of research has been performed in improving the learning to rank frameworks employed in different applications like web search, eCommerce search, question answering systems, recommendation systems [5], [6]. In eCommerce, given a query $q$, a typical search system retrieves all items $I_{n} \in I$ matching the query, ranks the items based on a ranking function $f\left(q, I_{n}\right)$ and returns the top $N$ documents. The ranking function $f\left(q, I_{n}\right)$ usually provides the probability of click or sale [7], [8] of an item, independent of other items in $I$, which in turn is used to sort items.

On the other hand, shoppers on eCommerce sites tend to compare and evaluate the list of items presented in search results, considering different options/selections available while making their purchase decision. This is somewhat different from web search, where the goal is to satisfy a single informational need. The comparative evaluation of eCommerce search results indicates that a shopper's perception of an item may be influenced by neighboring items presented along with it in the ranked results. However, the ranking functions learnt and applied in most eCommerce sites today score items independently and do not take the neighborhood into consideration. To that end, in this paper we study the influence of neighboring items on a user's preference of a given item in eCommerce search. Specifically, we aim to evaluate if incorporating the knowledge of neighborhood can help us better predict the preference of an item in the context of eCommerce search.

For learning the ranking function, training data can be collected in 2 ways. One approach is to obtain human judged labels for items matching a query, to annotate a binary decision of relevant or not for a given item [7]. Second approach is to extract implicit relevance feedback based on user behavior logs [9]-[11]. In web search as well as in eCommerce search, one of the widely used relevance feedback is clicks. In addtion to that, eCommerce search systems have the advantage of using more relevance feedback signals like bids, add to carts, purchases, revenue etc [12]. The basic assumption in implicit relevance feedback is, users scan the items in topdown manner. Existing literature study the impact of items that were viewed and not clicked as negative samples in relevance feedback [11]. Other studies have focused on the impact of a document's relevance based on the documents ranked above it with the focus on search result diversity [13], [14]. In this paper, we study the effect of the neighboring items, i.e. items ranked above and below a particular item $I_{n}$ on the preference of $I_{n}$ in eCommerce search results. To evaluate the impact, we quantify neighborhood by means of features that compare items ranked at different positions above and below the current item. These comparative features are denoted as delta features.

Our study highlights different delta features we tried on top of our current baseline model, and the improvements in offline metrics they result in. We also evaluate the effect of different neighborhood sizes $m$ used in constructing the delta features, and experimentally show that the neighborhood of an item has an impact on the item's preference in the ranked results through offline metrics.

The rest of the paper is organized as follows. Section II discusses some of the related work in the literature. In Section III we describe our methodology. In Section IV we describe our datasets and experiments. We summarize our work and discuss possible future research in Section V

## II. RELATED WORK

Lichtenstein et. al presented some early work on how people make decisions under uncertainty in [15], where the key insight is that the decisions are different when choices are presented separately vs. when they are presented together. Importance of a context (neighborhood) for a given item to its clickability has been extensively researched in the past. Previous studies of users' clicks as implicit feedback in search found out that clicking decision on a web document is affected by both rank and other documents in the presentation [16],

[17]. Craswell et al. [18] introduced the cascade click model where the probability of click for a given document at a given rank is influenced by probability of click for documents at higher ranks.

Dupret et al. [19] introduced a new browsing behavior model, where the probability of a subsequent click for a given document is affected by a distance between that document and the most recently clicked document. The probability gets lower if the previously clicked document is further away, i.e. if a user has to scroll through numerous irrelevant documents. Our approach extends this research to model preference of items in e-commerce search.

## III. OUR APPROACH

Our hypothesis is that whenever users make a decision to buy an item on an eCommerce platform, it is not in isolation. The decision is made by comparing the item to other items in its vicinity. Most ranking models use a single item's features to determine the probability of sale. To understand how the neighboring items affect an item's preference by a user, we define delta features that represent how the item differs from it neighboring items.

We focus on features that could be potentially distinguishing factors of an item and those that can identify user behavior. Since we want to model user behavior, these features are derived from elements users are likely to see on the search results page when making a purchase, for e.g. shipping time, product title, product price etc. We identified the set of features which users are likely to perceive while buying an item as the candidate set $\left(F: f_{1}, f_{2} . f_{n}\right)$ from which we can generate delta features.

![](https://cdn.mathpix.com/cropped/2024_06_04_b6629bef00e21586460fg-3.jpg?height=411&width=543&top_left_y=1543&top_left_x=162)

Fig. 1. Illustration of previous and next delta features constructed based on a ranked list of items. Here the neighborhood size is 2 .

We experiment with three different neighborhood sizes ( size $=1,3,5$ ) to study how the influence of the delta features changes as the neighborhood size changes. For each of these candidate features $F$, we generated two types of delta features each, namely next and prev; next represents the delta features based on the items ranked below the current item, while prev represents the delta features based on the items ranked above the current item. Fig 1 represents an example of a neighborhood of size 2 . For the item $I_{4}$, next features are calculated by comparing features of $I_{4}$ with $I_{5}$ and $I_{6}$. Similarly, prev features are calculated by comparing features of $I_{4}$ with $I_{2}$ and $I_{3}$. Note that neighborhood size refers to the number of items considered in computing the delta features above and below the current item. The delta features are denoted as,

$$
\begin{aligned}
D & :\left[d_{1 m \_p r e v}, d_{1 m \_n e x t}, d_{2 m \_p r e v}, d_{2 m \_n e x t}\right. \\
& \left.\ldots, d_{n m \_p r e v}, d_{n m \_n e x t}\right]
\end{aligned}
$$

where $m$ represents the neighborhood size. We further define a distance weighted decay function $\gamma(j)$, where $j$ is the number of positions a neighbor is away from the current item. $\gamma(j)$ captures varying distance adjusted contributions to the delta feature by different neighbors, based on the intuition that items that are farther may have a different influence on a users' perception of an item than a closer one. There are three different categories of delta features defined :

1) Numerical Delta Features : Numerical delta features are defined as the difference between the previous/next item's features and the current item's features:

$$
\begin{aligned}
& D_{k m \_p r e v}=\frac{1}{m} * \sum_{j=1}^{m} \frac{f_{k-j}-f_{k}}{\gamma(j)} \\
& D_{k m \_n e x t}=\frac{1}{m} * \sum_{j=1}^{m} \frac{f_{k+j}-f_{k}}{\gamma(j)}
\end{aligned}
$$

2) Categorical Delta Features : For categorical features with discrete values, the delta features are defined as the distance weighted average of matching discrete feature values occurring in the neighborhood of the current item. This can be represented as:

$$
\begin{aligned}
D_{k m \_p r e v} & =\frac{1}{m} * \sum_{j=1}^{m} \frac{\operatorname{diff}\left(f_{k-j}, f_{k}\right)}{\gamma(j)} \\
D_{k m \_n e x t} & =\frac{1}{m} * \sum_{j=1}^{m} \frac{\operatorname{diff}\left(f_{k+j}, f_{k}\right)}{\gamma(j)}
\end{aligned}
$$

where $\operatorname{diff}(a, b)=1$ if $a=b$, and 0 otherwise. Note that, boolean delta features are a special case of categorical ones, where there are only 2 possible feature values.

3) Vector based Delta Features : Delta features can be computed based on vector based representations of items. For instance, item embeddings learnt based on specific properties and subsequent user interactions can be used as representations to effectively capture similarities and differences between items.

$$
\begin{aligned}
& D_{k m \_p r e v}=\frac{1}{m} * \sum_{j=1}^{m} \frac{\operatorname{Vdiff}\left(v_{k-j}, v_{k}\right)}{\gamma(j)} \\
& D_{k m \_n e x t}=\frac{1}{m} * \sum_{j=1}^{m} \frac{V \operatorname{diff}\left(v_{k+j}, v_{k}\right)}{\gamma(j)}
\end{aligned}
$$

where $v_{k}$ is the vector representing the item at position $k$ and $\operatorname{V} \operatorname{dif} f(\alpha, \beta)$ is a distance measures between vectors $\alpha$ and $\beta$ of the same dimensionality. A measure such as cosine similarity may be used for this purpose where $V \operatorname{diff}(\alpha, \beta)$ can be defined as $1-\cos (\alpha, \beta)$.

## IV. EXPERIMENTS

We build several offline ranking models with varying neighborhood sizes and selection of delta features to evaluate the incremental improvement produced by these features in the performance of the ranking models, and subsequently observe the effect of neighborhood on the preference of an item. In this section, we will describe the dataset used, the various feature sets employed in the experiments that follow, and the models built as part of the experiments.

## A. Dataset, Features and Experiment Setting

We conduct our ranking experiments on a large-scale dataset sampled from eBay search logs. The dataset consists of about 20000 unique search queries sampled based on user search sessions which resulted in an item's sale, along with the ranked list of top items impressed for the query. The labels for the items in the dataset are obtained via implicit relevance feedback. In this paper, we consider the sale of an item as the target. We constructed delta features as described in Section III based on features that are perceivable by the users such as price, popularity and retail standards associated with the item. While, embedding based delta features can be constructed using item embeddings, we limit delta features to either numerical or categorical in the experiments that follow. Further, we use a distance weighted decay function $\gamma(j)=1$ in constructing delta features. In other words, we treat farther neighbors the same as closer ones while computing delta features. $80 \%$ of the dataset was used for training and $20 \%$ for validation.

We trained several learning to rank models on the dataset described above. We use the state-of-the-art LambdaMART model [1] for our experiments. The baseline model, Model_Base is trained on the same dataset without any delta features. Model_Base is the production ranking model for eBay. The proposed ranking models use features from Model_Base and delta features. We train ranking models with different neighborhood sizes and different neighborhood types namely, prev and next. We experimented with 3 neighborhood sizes in this paper, $m=1,3,5$. We trained three different models for each neighborhood size, $m$ :

1) Model_Prev_Wm : Models with prev delta features, calculated based on items ranked above the current item
2) Model_Next_Wm : Models with next delta features, calculated based on items ranked below the current item
3) Model_Prev_Next_Wm : Models with prev and next delta features, calculated based on items ranked above and below the current item

The hyperparameters are tuned based on Model_Base and the same parameters are used to train all the proposed ranking models with delta features.

## B. Results

We trained models with both previous and next delta features constructed based on neighborhood sizes 1,3 and 5 respectively. The trained models were evaluated offline on the test dataset with the aim being observing incremental ranking improvements to the models introduced by delta features. Mean reciprocal sale rank (MRR) was chosen as the metric to evaluate and compare the performance of the various models relative to the baseline model Model_Base. MRR, in this case captures the first result that involves an item sale. We employed MRR as the evaluation metric to capture the notion of preference in a ranked list via sale of an item.

![](https://cdn.mathpix.com/cropped/2024_06_04_b6629bef00e21586460fg-4.jpg?height=498&width=876&top_left_y=537&top_left_x=1080)

Fig. 2. MRR difference with respect to Model_Base for neighborhood sizes 1,3 and 5 using prev features.

![](https://cdn.mathpix.com/cropped/2024_06_04_b6629bef00e21586460fg-4.jpg?height=512&width=892&top_left_y=1273&top_left_x=1083)

Fig. 3. MRR difference with respect to Model_Base for neighborhood sizes 1,3 and 5 using both prev_next features.

The prev and next features which capture the neighborhood above and below an item in the ranked list of results, show significant improvements in MRR compared to the baseline model. The figures show MRR difference with respect to Model_Base and the error bars are computed using 1000 bootstrap samples of the test dataset.

First, we used only prev features constructed based on neighborhood sizes 1,3 and 5 in addition to baseline features. prev features lead to MRR improvements as can be seen from Fig 2, with neighborhood size 3 outperforming others. Similarly, Fig 4 shows the relative MRR improvements when only next features constructed based on neighborhood sizes 1 , 3 and 5 in addition to baseline features. Neighborhood size 3 leads to the most significant improvements in MRR. Further, varying neighborhood sizes has a measurable effect on MRR,

![](https://cdn.mathpix.com/cropped/2024_06_04_b6629bef00e21586460fg-5.jpg?height=496&width=894&top_left_y=191&top_left_x=168)

Fig. 4. MRR difference with respect to Model_Base for neighborhood sizes 1,3 and 5 using next features.

indicating that the choice of neighborhood size is an important decision. Lastly, by combining prev and next features on top of the baseline features also resulted in significant improvements in MRR with neighborhood size 3, performing the best as shown in Fig 3 .

The percentage gains in MRR resulting from each of the models relative to Model_Base is tabulated in Table I As evident from the table, using prev_next features constructed using a neighborhood size, 3 , results in $5.01 \%$ improvement in MRR, thereby supporting the intuition that the neighborhood consisting of both items ranked above and below an item together influence preference of an item.

TABLE I

PERCENTAGE CHANGE IN MRR

| Neighborhood size | prev | next | prev_next |
| :--- | :--- | :--- | :--- |
| 1 | -1.32 | 0.07 | 1.81 |
| $\mathbf{3}$ | 4.65 | 4.45 | $\mathbf{5 . 0 1}$ |
| 5 | 3.05 | 3.55 | 4.52 |

Percentage change in MRR relative to Model_Base
resulting from the various models.

Since neighborhood size 3 resulted in the most observable MRR improvements, we compared prev, next, and prev_next models trained on delta features constructed with neighborhood size 3 in addition to the baseline features. From Fig 5 we can observe that while both prev and next models lead to improvements, prev_next models have the most pronounced MRR gains, indicating that the neighborhood of an item does influence its preference in a measurable way. Further, the observation that larger neighborhood sizes don't necessarily contribute to more effective models suggests applying a distance weighted decay in constructing delta features. We plan to explore the effects of such a function in future work.

## V. Summary and Future Work

Learning to rank techniques are widely used to learn a ranking function and furnish a ranked order of items across a broad range of domains and are a critical component of eCommerce domain specifically. In practice, items are usually ranked independently, without taking into account the influence of neighboring items on the placement of a given item. However, when users view a ranked list of items, their

![](https://cdn.mathpix.com/cropped/2024_06_04_b6629bef00e21586460fg-5.jpg?height=515&width=892&top_left_y=198&top_left_x=1083)

Fig. 5. MRR difference with respect to Model_Base for neighborhood size 3 using prev, next, and prev_next features.

perception of a given item is influenced by its neighborhood. This effect is even more pronounced in eCommerce, where users have a selection of relevant items and tend to make comparative decisions. This raises the question of investigating the influence of neighborhood on the placement of an item in a ranking list. List-wise loss functions and group-wise scoring functions have been studied in literature, and methods to place an item in a ranked list based on its predecessors have been proposed. However, the influence of neighborhood on a user's perception of an item in a ranked list has been seldom investigated, specifically in the eCommerce domain. To that end, we investigated the influence of neighboring items on users' perception of a given item by studying the effect of neighborhood within a ranked list of items.

We constructed delta features that capture how a given item differs from those in its neighborhood in terms of attributes that can be perceived by the user on a search result page. We then trained learning to rank models based on a pairwise loss function and conversion ( sale ) as a target to study the effect of these delta features on understanding the preference of an item. By employing a feature set that consisted of the newly constructed delta features in addition to features that are already being used in models that are on site, we examined the incremental benefits of the delta features. From our experiments, we find that delta features consistently rank high in terms of feature importance. Further, including delta features contributes positively to ranking metrics such as mean reciprocal sale rank. Including previous and next features outperforms using either previous or next individually. In addition to this, we discovered that the choice of the size of neighborhood influences the performance of these features. In summary, the key takeaways from this work are :

- The neighborhood of an item effects users' perception of it and its preference within a ranked list, specifically in eCommerce domain. Hence neighborhood must be accounted for while placing an item in a raked search result page.
- The choice of the size of the neighborhood influences the performance of delta features, and subsequently the ability to model neighborhood.

As a next step, we plan to investigate the applicability of item embeddings and the effect of introducing a distance
weighted decay in the construction of delta features, as part of work focused on constructing more effective representations of neighborhoods. Another application of the learning of this work is incorporating the idea of neighborhood and delta features into ranking models. This would require designing efficient methods to determine the placement of a candidate item based on its potential neighbors, in contrast to an independent decision. Further, by identifying discriminating delta features, we may be able to understand diversity as perceived by eCommerce users. While diversity in a ranked list has been well studied in web search, a nuanced study of what attributes describe diversity in the context of eCommerce can be useful to the domain. Building up on the idea of delta features, we will study the features and attributes that can explain diversity in eCommerce as future work.

## ACKNOWLEDGMENT

We would like to thank Alex Cozzi for the insightful discussions and valuable guidance he provided during the course of this work.

## REFERENCES

[1] C. J. Burges, "From ranknet to lambdarank to lambdamart: An overview," Learning, vol. 11, no. 23-581, p. 81, 2010.

[2] Z. Cao, T. Qin, T.-Y. Liu, M.-F. Tsai, and H. Li, "Learning to rank: from pairwise approach to listwise approach," in Proceedings of the 24th international conference on Machine learning. ACM, 2007, pp. $129-136$.

[3] F. Xia, T.-Y. Liu, J. Wang, W. Zhang, and H. Li, "Listwise approach to learning to rank: theory and algorithm," in Proceedings of the 25th international conference on Machine learning. ACM, 2008, pp. 11921199 .

[4] N. Craswell, Mean Reciprocal Rank. Boston, MA: Springer US, 2009, pp. 1703-1703.

[5] T.-Y. Liu et al., "Learning to rank for information retrieval," Foundations and Trends $®$ in Information Retrieval, vol. 3, no. 3, pp. 225-331, 2009.

[6] H. Li, "A short introduction to learning to rank," IEICE TRANSACTIONS on Information and Systems, vol. 94, no. 10, pp. 1854-1862, 2011.

[7] F. Radlinski and T. Joachims, "Query chains: learning to rank from implicit feedback," in Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining. ACM, 2005, pp. 239-248.

[8] T. Joachims, L. A. Granka, B. Pan, H. Hembrooke, and G. Gay, "Accurately interpreting clickthrough data as implicit feedback," in Sigir, vol. 5, 2005, pp. 154-161.

[9] E. Agichtein, E. Brill, S. Dumais, and R. Ragno, "Learning user interaction models for predicting web search result preferences," in Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 2006, pp. 3-10.

[10] W. W. Cohen, R. E. Schapire, and Y. Singer, "Learning to order things," in Advances in Neural Information Processing Systems, 1998, pp. 451457.

[11] T. Joachims, "Optimizing search engines using clickthrough data," in Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD '02. New York, NY, USA: ACM, 2002, pp. 133-142. [Online]. Available: http://doi.acm.org/10.1145/775047.775067

[12] S. K. Karmaker Santu, P. Sondhi, and C. Zhai, "On application of learning to rank for e-commerce search," in Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, ser. SIGIR '17. New York, NY, USA: ACM, 2017, pp. 475-484. [Online]. Available: http://doi.acm.org/10.1145/3077136.3080838

[13] Y. Zhu, Y. Lan, J. Guo, X. Cheng, and S. Niu, "Learning for search result diversification," in Proceedings of the 37th international ACM SIGIR conference on Research \& development in information retrieval. ACM, 2014, pp. 293-302.
[14] R. Agrawal, S. Gollapudi, A. Halverson, and S. Ieong, "Diversifying search results," in Proceedings of the Second ACM International Conference on Web Search and Data Mining, ser. WSDM '09. New York, NY, USA: ACM, 2009, pp. 5-14. [Online]. Available: http://doi.acm.org/10.1145/1498759.1498766

[15] S. Lichtenstein and P. Slovic, "Reversals of preference between bids and choices in gambling decisions." Journal of experimental psychology, vol. 89, no. 1, p. 46, 1971.

[16] T. Joachims, L. Granka, B. Pan, H. Hembrooke, and G. Gay, "Accurately interpreting clickthrough data as implicit feedback," in Proceedings of the 28th annual international ACM SIGIR conference on Research and development in information retrieval, 2005, pp. 154-161.

[17] T. Joachims, L. Granka, B. Pan, H. Hembrooke, F. Radlinski, and G. Gay, "Evaluating the accuracy of implicit feedback from clicks and query reformulations in web search," ACM Transactions on Information Systems, vol. 25, no. 2, 2007.

[18] N. Craswell, O. Zoeter, M. Taylor, and B. Ramsey, "An experimental comparison of click position-bias models," in Proceedings of the 2008 International Conference on Web Search and Data Mining, 2008, pp. 87-94.

[19] G. Dupret and B. Piwowarski, "A user browsing model to predict search engine click data from past observations," in Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval, 2008, pp. 331-338.

</end of paper 0>


<paper 1>
# Context-Aware Learning to Rank with Self-Attention 

Przemysław Pobrotyn<br>ML Research at Allegro.pl<br>przemyslaw.pobrotyn@allegro.pl

Tomasz Bartczak<br>ML Research at Allegro.pl<br>tomasz.bartczak@allegro.pl

Mikołaj Synowiec<br>ML Research at Allegro.pl<br>mikolaj.synowiec@allegro.pl

Radosław Białobrzeski<br>ML Research at Allegro.pl<br>radoslaw.bialobrzeski@allegro.pl

Jarosław Bojar<br>ML Research at Allegro.pl<br>jaroslaw.bojar@allegro.pl


#### Abstract

Learning to rank is a key component of many e-commerce search engines. In learning to rank, one is interested in optimising the global ordering of a list of items according to their utility for users. Popular approaches learn a scoring function that scores items individually (i.e. without the context of other items in the list) by optimising a pointwise, pairwise or listwise loss. The list is then sorted in the descending order of the scores. Possible interactions between items present in the same list are taken into account in the training phase at the loss level. However, during inference, items are scored individually, and possible interactions between them are not considered. In this paper, we propose a context-aware neural network model that learns item scores by applying a self-attention mechanism. The relevance of a given item is thus determined in the context of all other items present in the list, both in training and in inference. We empirically demonstrate significant performance gains of self-attention based neural architecture over Multi-Layer Perceptron baselines, in particular on a dataset coming from search logs of a large scale e-commerce marketplace, Allegro.pl. This effect is consistent across popular pointwise, pairwise and listwise losses. Finally, we report new state-of-the-art results on MSLR-WEB30K, the learning to rank benchmark.


## CCS CONCEPTS

- Information systems $\rightarrow$ Learning to rank;


## KEYWORDS

learning to rank, self-attention, context-aware ranking

## ACM Reference Format:

Przemysław Pobrotyn, Tomasz Bartczak, Mikołaj Synowiec, Radosław Białobrzeski, and Jarosław Bojar. 2020. Context-Aware Learning to Rank with Self-Attention. In Proceedings of ACM SIGIR Workshop on eCommerce (SIGIR eCom'20). ACM, New York, NY, USA, 8 pages.

## 1 INTRODUCTION

Learning to rank (LTR) is an important area of machine learning research, lying at the core of many information retrieval (IR) systems. It arises in numerous industrial applications like e-commerce search[^0]

engines, recommender systems, question-answering systems, and others.

A typical machine learning solution to the LTR problem involves learning a scoring function, which assigns real-valued scores to each item of a given list, based on a dataset of item features and human-curated or implicit (e.g. clickthrough logs) relevance labels. Items are then sorted in the descending order of scores [23]. Performance of the trained scoring function is usually evaluated using an IR metric like Mean Reciprocal Rank (MRR) [33], Normalised Discounted Cumulative Gain (NDCG) [19] or Mean Average Precision (MAP) [4].

In contrast to other classic machine learning problems like classification or regression, the main goal of a ranking algorithm is to determine relative preference among a group of items. Scoring items individually is a proxy of the actual learning to rank task. Users' preference for a given item on a list depends on other items present in the same list: an otherwise preferable item might become less relevant in the presence of other, more relevant items. For example, in the context of an e-commerce search engine, the relative desirability of an item might depend of the relation of its price to the prices of other items displayed in the results list. Common learning to rank algorithms attempt to model such inter-item dependencies at the loss level. That is, items in a list are still scored individually, but the effect of their interactions on evaluation metrics is accounted for in the loss function, which usually takes a form of a pairwise (RankNet [6], LambdaLoss [34]) or a listwise (ListNet [9], ListMLE [35]) objective. For example, in LambdaMART [8] the gradient of the pairwise loss is rescaled by the change in NDCG of the list which would occur if a pair of items was swapped. Pointwise objectives, on the other hand, do not take such dependencies into account.

In this work, we propose a learnable, context-aware, self-attention [31] based scoring function, which allows for modelling of interitem dependencies not only at the loss level but also in the computation of items' scores. Self-attention is a mechanism first introduced in the context of natural language processing. Unlike RNNs [16], it does not process the input items sequentially but allows the model to attend to different parts of the input regardless of their distance from the currently processed item. We adapt the Transformer [31], a popular self-attention based neural machine translation architecture, to the ranking task. Since the self-attention operation is permutation-equivariant (scores items the same way irrespective of their input order), we obtain a permutation-equivariant scoring function suitable for ranking. If we further refine the model with positional encodings, the obtained model becomes suitable for re-ranking setting. We demonstrate that the obtained (re)ranking
model significantly improves performance over Multi-Layer Perceptron (MLP) baselines across a range of pointwise, pairwise and listwise ranking losses. Evaluation is conducted on MSLR-WEB30K [28], the benchmark LTR dataset with multi-level relevance judgements, as well as on clickthrough data coming from Allegro.pl, a large-scale e-commerce search engine. We also establish the new state-of-the-art results on WEB30K in terms of NDCG@5.

We provide an open-source Pytorch [26] implementation of our self-attentive context-aware ranker available at https://github.com/ allegro/allRank.

The rest of the paper is organised as follows. In Section 2 we review related work. In Section 3 we formulate the problem solved in this work. In Section 4 we describe our self-attentive ranking model. Experimental results and their discussion are presented in Section 5. In Section 6 we conduct an ablation study of various hyperparameters of our model. Finally, a summary of our work is given in Section 7.

## 2 RELATED WORK

Learning to rank has been extensively studied and there is a plethora of resources available on classic pointwise, pairwise and listwise approaches. We refer the reader to [23] for the overview of the most popular methods.

What the majority of LTR methods have in common is that their scoring functions score items individually. Inter-item dependencies are (if at all) taken into account at the loss level only. Previous attempts at modelling context of other items in a list in the scoring function include:

- a pairwise scoring function [12] and Groupwise Scoring Function (GSF) [2], which incorporates the former work as its special case. However, the proposed GSF method simply concatenates feature vectors of multiple items and passes them through an MLP. To desensitize the model to the order of concatenated items, Monte-Carlo sampling is used, which yields an unscalable algorithm,
- a seq2slate model [5] uses an RNN combined with a variant of Pointer Networks [32] in an encoder-decoder type architecture to both encode items in a context-aware fashion and then produce the optimal list by selecting items one-by-one. Authors evaluate their approach only on clickthrough data (both real and simulated from WEB30K). A similar, simpler approach known as Deep Listwise Context Model (DLCM) was proposed in [1]: an RNN is used to encode a set of items for re-ranking, followed by a single decoding step with attention,
- in [17], authors attempt to capture inter-item dependencies by adding so-called delta features that represent how differ ent a given item is from items surrounding it in the list. It can be seen as a simplified version of a local self-attention mechanism. Authors evaluate their approach on proprietary search logs only,
- authors of [20] formulate the problem of re-ranking of a list of items as that of a whole-list generation. They introduce ListCVAE, a variant of Conditional Variational Auto-Encoder [29] which learns the joint distribution of items in a list conditioned on users' relevance feedback and uses it to directly generate a ranked list of items. Authors claim NDCG unfairly favours greedy ranking methods and thus do not use that metric in their evaluation,
- similarly to our approach, Pei et al. [27] use the self-attention mechanism to model inter-item dependencies. Their approach, however, was not evaluated on a standard WEB30K dataset and the only loss function considered was ListNet.

Our proposed solution to the problem of context-aware ranking makes use of the self-attention mechanism. It was first introduced as intra-attention in [11] and received more attention after the introduction of the Transformer architecture [31]. Our model can be seen as a special case of the encoder part of the Transformer.

Our model, being a neural network, can be trained with gradientbased optimisation methods to minimise any differentiable loss function. Loss functions suitable for the ranking setting have been studied extensively in the past [23]. In order to demonstrate that our context-aware model provides performance boosts irrespectively of the loss function used, we evaluate its performance when tasked with optimising several popular ranking losses.

We compare the proposed approach with those of the aforementioned methods which provided an evaluation on WEB30K in terms of NDCG@5 and NDCG@10. These include GSF of [2] and DLCM of [1]. We outperform both competing methods.

## 3 PROBLEM FORMULATION

In this section, we formulate problem at hand in learning to rank setting. Let $X$ be the training set. It consists of pairs $(x, y)$ of a list $x$ of $d_{f}$-dimensional real-valued vectors $x_{i}$ together with a list $y$ of their relevance labels $y_{i}$ (multi-level or binary). Note that lists $x$ in the training set may be of varying length. The goal is to find a scoring function $f$ which maximises an IR metric of choice (e.g. NDCG) on the test set. Since IR metrics are rank based (thus, nondifferentiable), the scoring function $f$ is trained to minimise the average of a surrogate loss $l$ over the training data.

$$
\mathcal{L}(f)=\frac{1}{|X|} \sum_{(\boldsymbol{x}, \boldsymbol{y}) \in X} l((\boldsymbol{x}, \boldsymbol{y}), f)
$$

while controlling for overfitting (e.g. by using dropout [30] in the neural network based scoring function $f$ or adding $L_{1}$ or $L_{2}$ penalty term [24] to the loss function $l$ ). Thus, two crucial choices one needs to make when proposing a learning to rank algorithm are that of a scoring function $f$ and loss function $l$. As discussed earlier, typically, $f$ scores elements $x_{i} \in x$ individually to produce scores $f\left(x_{i}\right)$, which are then input to loss function $l$ together with ground truth labels $y_{i}$. In subsequent sections, we describe our construction of context-aware scoring function $f$ which can model interactions between items $x_{i}$ in a list $x$. Our model is generic enough to be applicable with any of standard pointwise, pairwise or listwise loss. We thus experiment with a variety of popular ranking losses $l$.

## 4 SELF-ATTENTIVE RANKER

In this section, we describe the architecture of our self-attention based ranking model. We modify the Transformer architecture to work in the ranking setting and obtain a scoring function which, when scoring a single item, takes into account all other items present in the same list.

### 4.1 Self-Attention Mechanism

The key component of our model is the self-attention mechanism introduced in [31]. The attention mechanism can be described as taking the query vector and pairs of key and value vectors as input and producing a vector output. The output of the attention mechanism for a given query is a weighted sum of the value vectors, where weights represent how relevant to the query is the key of the corresponding value vector. Self-attention is a variant of attention in which query, key and value vectors are all the same - in our case, they are vector representations of items in the list. The goal of the self-attention mechanism is to compute a new, higher-level representation for each item in a list, by taking a weighted sum over all items in a list according to weights representing the relevance of these items to the query item.

There are many ways in which one may compute the relevance of key vectors to query vectors. We use the variant of selfattention known as Scaled Dot-Product Attention. Suppose $Q$ is a $d_{\text {model }}$-dimensional matrix representing all items (queries) in the list. Let $K$ and $V$ be the keys and values matrices, respectively. Then

$$
\text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{\text {model }}}}\right) V
$$

The scaling factor of $\frac{1}{\sqrt{d_{\text {model }}}}$ is added to avoid small gradients in the softmax operation for large values of $d_{\text {model }}$.

### 4.2 Multi-Headed Self-Attention

As described in [31], it is beneficial to perform the self-attention operation multiple times and concatenate the outputs. To avoid growing the size of the resulting output vector, matrices $Q, K$ and $V$ are first linearly projected $H$ times to $d_{q}, d_{k}$ and $d_{v}$ dimensional spaces, respectively. Usually, $d_{q}=d_{k}=d_{v}=d_{\mathrm{model}} / H$. Each of $H$ computations of a linear projection of $Q, K, V$, followed by a self-attention mechanism is referred to as a single attention head. Note that each head has its own learnable projection matrices. The outputs of each head are concatenated and once again linearly projected, usually to the vector space of the same dimension as that of input matrix $Q$. Similarly to the Transformer, our model also uses multiple attention heads. Thus

$$
\operatorname{MultiHead}(\mathrm{Q}, \mathrm{K}, \mathrm{V})=\operatorname{Concat}\left(\text { head }_{1}, \ldots, \text { head }_{\mathrm{H}}\right) W^{O}
$$

$$
\text { where head }_{i}=\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
$$

and the projections are given by matrices

$$
\begin{gathered}
W_{i}^{Q} \in \mathbb{R}^{d_{\text {model }} \times d_{q}}, W_{i}^{K} \in \mathbb{R}^{d_{\text {model }} \times d_{k}} \\
W_{i}^{V} \in \mathbb{R}^{d_{\text {model }} \times d_{v}}, W^{O} \in \mathbb{R}^{H d_{v} \times d_{\text {model }}}
\end{gathered}
$$

### 4.3 Permutation-equivariance

The key property of the proposed context-aware model making it suitable for the ranking setting is that it is permutation-equivariant, i.e. the scores of items do not depend on the original ordering of the input. Recall the definition of permutation equivariance:

Definition 4.1. Let $x \in \mathbb{R}^{n}$ be a real-valued vector and $\pi \in S_{n}$ be a permutation of $n$ elements. A function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$ is called permutation-equivariant iff

$$
f(\pi(x))=\pi(f(x))
$$

That is, a function is permutation-equivariant if it commutes with any permutation of the input elements.

It is a trivial observation that the self-attention operation is permutation-equivariant.

### 4.4 Positional Encodings

Transformer architecture was designed to solve a neural machine translation (NMT) task. In NMT, the order of input tokens should be taken into account. Unlike RNNs, self-attention based encoder has no way of discerning the order of input tokens, because of its permutation-equivariance. Authors of the original Transformer paper proposed to solve the problem by adding either fixed or learnable positional encodings to the input embeddings. Fixed positional encodings use sine and cosine functions of different frequencies, as follows:

$$
\begin{gathered}
P E_{(\text {pos }, 2 i)}=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(p o s, 2 i+1)}=\cos \left(\text { pos } / 10000^{2 i / d_{\text {model }}}\right)
\end{gathered}
$$

where pos is the position and $i$ is the dimension.

The ranking problem can be viewed as either ordering a set of (unordered) items or as re-ranking, where the input list has already been sorted according to a weak ranking model. In the former case, the use of positional encodings is not needed. In the latter, they may boost the model's performance. We experiment with both ranking and re-ranking settings and when positional encodings are used, we test the fixed encodings variant ${ }^{1}$. Details can be found in Section 5 .

### 4.5 Model Architecture

We adapt the Transformer model to the ranking setting as follows. Items on a list are treated as tokens and item features as input token embeddings. We denote the length of an input list as $l$ and the number of features as $d_{f}$. Each item is first passed through a shared fully connected layer of size $d_{f c}$. Next, hidden representations are passed through an encoder part of Transformer architecture with $N$ encoder blocks, $H$ heads and hidden dimension $d_{h}$. Recall that an encoder block in the Transformer consists of a multi-head attention layer with a skip-connection [14] to the input, followed by layer normalisation [3], time-distributed feed-forward layer, and another skip connection followed by layer normalisation. Dropout is applied before performing summation in residual blocks. Finally, after $N$ encoder blocks, a fully-connected layer shared across all items in the list is used to compute a score for each item. The model can be seen as an encoder part of the Transformer with extra linear projection on the input (see Figure 1 for a schematic of the architecture).

Thus, the model can be expressed as

$$
f(x)=\mathrm{FC}(\underbrace{\text { Encoder }(\operatorname{Encoder}(\ldots(\operatorname{Encoder}}_{N \text { times }}(\mathrm{FC}(x))))))
$$

where

$$
\operatorname{Encoder}(x)=\operatorname{LayerNorm}(z+\operatorname{Dropout}(\operatorname{FC}(z)))
$$

$$
z=\operatorname{LayerNorm}(x+\operatorname{Dropout}(\operatorname{MultiHead}(x)))
$$

and FC, Dropout are fully-connected and dropout layers, respectively.[^1]

By using self-attention in the encoder, we ensure that in the computation of a score of a given item, hidden representation of all other items were accounted for. Obtained scores, together with ground truth labels, can provide input to any ranking loss of choice. If the loss is a differentiable function of scores (and thus, of model's parameters), one can use SGD to optimise it. We thus obtain a general, context-aware model for scoring items on a list that can readily be used with any differentiable ranking loss. Since all the components used in the construction of the model (self-attention, layer normalisation, feed-forward layers) are permutation-equivariant, the entire model is permutation-equivariant (unless positional encodings are used).

![](https://cdn.mathpix.com/cropped/2024_06_04_2479b797d5347d973056g-4.jpg?height=295&width=832&top_left_y=774&top_left_x=191)

Figure 1: Schematic of the proposed model architecture. Input is a list of real-valued vectors. Output is the list of realvalued scores.

## 5 EXPERIMENTS

### 5.1 Datasets

Learning to rank datasets come in two flavours: they can have either multi-level or binary relevance labels. Usually, multi-level relevance labels are human-curated, whereas binary labels are derived from clickthrough logs and are considered implicit feedback. We evaluate our context-aware ranker on both types of data.

For the first type, we use the popular WEB30K dataset, which consists of more than 30,000 queries together with lists of associated search results. Every search result is encoded as a 136-dimensional real-valued vector and has associated with it a relevance label on the scale from 0 (irrelevant) to 4 (most relevant). We standardise the features before inputting them into a learning algorithm. The dataset comes partitioned into five folds with roughly the same number of queries per fold. We perform 5-fold cross-validation by training our models on three folds, validating on one and testing on the final fold. All results reported are averages across five folds together with the standard deviation of results. Since lists in the dataset are of unequal length, we pad or subsample to equal length for training, but use full length (i.e. pad to maximum length present in the dataset) for validation and testing. Note that there are 982 queries for which the associated search results list contains no relevant documents (i.e. all documents have label 0). For such lists, the NDCG can be arbitrarily set to either 0 or 1 . To allow for a fair comparison with the current state-of-the-art, we followed LightGBM [21] implementation of setting NDCG of such lists to 1 during evaluation.

For a dataset with binary labels, we use clickthrough logs of a large scale e-commerce search engine from Allegro.pl. The search engine already has a ranking model deployed, which is trained using
XGBoost [10] with rank: pairwise loss. We thus treat learning on this dataset as a re-ranking problem and use fixed positional encodings in context-aware scoring functions. This lets the models leverage items' positions returned by the base ranker. The search logs consist of $1 \mathrm{M}$ lists, each of length at most 60. Nearly all lists $(95 \%)$ have only one relevant item with label 1; remaining items were not clicked and are deemed irrelevant (label 0). Each item in a list is represented by a 45-dimensional, real-valued vector. We do not perform cross-validation on this set, but we use the usual train, validation and test splits of the data (using $100 \mathrm{k}$ lists for validation, $100 \mathrm{k}$ for test and the remaining lists for training).

### 5.2 Loss Functions

To evaluate the performance of the proposed context-aware ranking model, we use several popular ranking losses. Pointwise losses used are RMSE of predicted scores and ordinal loss [25] (with minor modification to make it suitable for ranking). For pairwise losses, we use NDCGLoss 2++ (one of the losses of LambdaLoss framework) and its special cases, RankNet and LambdaRank [7]. Listwise losses used consist of ListNet and ListMLE.

Below, we briefly describe all of the losses used. For a more thorough treatment, please refer to the original papers. Throughout, $X$ denotes the training set, $x$ denotes an input list of items, $s=f(x)$ is a vector of scores obtained via the ranking function $f$ and $\boldsymbol{y}$ is the vector of ground truth relevancy labels.

5.2.1 Pointwise RMSE. The simplest baseline is a pointwise loss, in which no interaction between items is taken into account. We use RMSE loss:

$$
l(s, \boldsymbol{y})=\sqrt{\sum_{i}\left(y_{i}-s_{i}\right)^{2}}
$$

In practice, we used sigmoid activation function on the outputs of the scoring function $f$ and rescaled them by multiplying by maximum relevance value (e.g. 4 for WEB30K).

5.2.2 Ordinal Loss. We formulated ordinal loss as follows. Multilevel ground truth labels were converted to vectors as follows:

$$
\begin{aligned}
0 & \mapsto[0,0,0,0] \\
1 & \mapsto[1,0,0,0] \\
2 & \mapsto[1,1,0,0] \\
3 & \mapsto[1,1,1,0] \\
4 & \mapsto[1,1,1,1]
\end{aligned}
$$

The self-attentive scoring function was modified to return four outputs and each output was passed through a sigmoid activation function. Thus, each neuron of the output predicts a single relevancy level, but by the reformulation of ground truth, their relative order is maintained, i.e. if, say, label 2 is predicted, label 1 should be predicted as well (although it is not strictly enforced and model is allowed to predict label 2 without predicting label 1). The final loss value is the mean of binary cross-entropy losses for each relevancy level. During inference, the outputs of all output neurons are summed to produce the final score of an item. Note that this is the classic ordinal loss, simply used in the ranking setting.

5.2.3 LambdaLoss, RankNet and LambdaRank. We used NDCGLoss2++ of [34], formulated as follows:

$l(s, y)=-\sum_{y_{i}>y_{j}} \log _{2} \sum_{\pi}\left(\frac{1}{1+e^{-\sigma\left(s_{i}-s_{j}\right)}}\right)^{\left(\rho_{i j}+\mu \delta_{i j}\right)\left|G_{i}-G_{j}\right|} H(\pi \mid s)$

where

$$
\begin{gathered}
G_{i}=\frac{2^{y_{i}}-1}{\operatorname{maxDCG}} \\
\rho_{i j}=\left|\frac{1}{D_{i}}-\frac{1}{D_{j}}\right| \\
\delta_{i j}=\left|\frac{1}{D_{|i-j|}}-\frac{1}{D_{|i-j|}+1}\right| \\
D_{i}=\log _{2}(1+i)
\end{gathered}
$$

and $H(\pi \mid s)$ is a hard assignment distribution of permutations, i.e.

$$
H(\hat{\pi} \mid s)=1 \text { and } H(\pi \mid s)=0 \text { for all } \pi \neq \hat{\pi}
$$

where $\hat{\pi}$ is the permutation in which all items are sorted by decreasing scores $\boldsymbol{s}$. Fixed parameter $\mu$ is set to 10.0 .

By removing the exponent in $l(s, y)$ formula we obtain the RankNet loss function, weighing each score pair identically. Similarly, we may obtain differently weighted RankNet variants by changing the formula in the exponent.

To obtain a LambdaRank formula, replace the exponent with

$$
\Delta \operatorname{NDCG}(i, j)=\left|G_{i}-G_{j}\right| \rho_{i j}
$$

5.2.4 ListNet and ListMLE. ListNet loss [9] is given by the following formula:

$$
l(s, y)=-\sum_{j} \operatorname{softmax}(\boldsymbol{y})_{j} \times \log \left(\operatorname{softmax}(s)_{j}\right)
$$

In binary version, softmax of ground truth $\boldsymbol{y}$ is omitted for singleclick lists and replaced with normalisation by the number of clicks for multiple-click lists.

ListMLE [35] is given by:

$$
l(\boldsymbol{s}, \boldsymbol{y})=-\log P(\boldsymbol{y} \mid \boldsymbol{s})
$$

where

$$
P(\boldsymbol{y} \mid \boldsymbol{s})=\prod_{i}^{n} \frac{\exp \left(f\left(x_{y(i)}\right)\right)}{\sum_{k=i}^{n} \exp \left(f\left(x_{y(k)}\right)\right)}
$$

and $y(i)$ is the index of object which is ranked at position $i$.

Table 1: Results from the literature on WEB30K

| Method | NDCG@5 | NDCG@ 10 |
| :--- | :---: | :---: |
| GSF $^{2}$ | 44.46 | 46.77 |
| DLCM | 45.00 | 46.90 |
| NDCGLoss 2++ (LightGBM) | 51.21 | - |
| Context-Aware Ranker (this work) | $\mathbf{5 3 . 0 0}$ | $\mathbf{5 4 . 8 8}$ |

[^2]
### 5.3 Experimental setup

We train both our context-aware ranking models and MLP models on both datasets, using all loss functions discussed in Section $5.2^{3}$. We also train XGBoost models with rank: pairwise loss similar to the production model of the e-commerce search engine for both datasets. Hyperparameters of all models (number of encoder blocks, number of attention heads, dropout, etc.) are tuned on the validation set of Fold 1 for each loss separately. MLP models are constructed to have a similar number of parameters to context-aware ranking models. For optimisation of neural network models, we use Adam optimiser [22] with the learning rate tuned separately for each model. Details of hyperparameters used can be found in Appendix A. In Section 6 we provide an ablation study of the effect of various hyperparameters on the model's performance.

Table 2: Relative percentage NDCG@60 improvement on ecommerce search logs dataset

| Loss | Self-attention | MLP |
| :--- | :---: | :---: |
| NDCGLoss 2++ | $\mathbf{3 . 0 0}$ | $\mathbf{1 . 5 1}$ |
| LambdaRank | 2.97 | 1.39 |
| ListNet | 2.93 | 1.24 |
| RankNet | 2.68 | 1.19 |
|  | XGBoost |  |
| rank: pairwise | 1.83 |  |

### 5.4 Results

On WEB30K, models' performance is evaluated using NDCG@5 ${ }^{4}$, which is the usual metric reported for this dataset, as well as NDCG at rank cutoffs 10 and 30. Results are reported in Table 3. On ecommerce search logs, we report a relative percentage increase in NDCG@60 ${ }^{5}$ over production XGBoost model, presented in Table 2. We observe consistent and significant performance improvement of the proposed self-attention based model over MLP baseline across all types of loss functions considered and any of the chosen rank cutoffs. In particular, for ListNet we observe a $7.3 \%$ performance improvement over MLP baseline on WEB30K in terms of NDCG@5. Note also that the best performing MLP model is outperformed even by the worst-performing self-attention based model on both datasets in all the metrics reported. We thus observe that incorporating context-awareness into the model architecture has a more pronounced effect on the performance of the model than varying the underlying loss function. Surprisingly, ordinal loss outperforms more established and better-studied losses like ListNet, ListMLE or NDCGLoss 2++ on multi-level relevancy data. In particular, we improve on the previous state-of-the-art NDCG@5 result of 51.21 by $2.27 \%$, obtaining 52.37 . The previous state-of-the-art result wasobtained using NDCGLoss 2++ trained using LightGBM. Another surprising finding is a good performance of models trained with RMSE loss, especially as compared to models trained to optimise RankNet and ListMLE. For comparison with the other methods, we[^3]

Table 3: Test results on WEB30K

| Loss | Self-attention |  |  | MLP |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | NDCG@5 | NDCG@10 | NDCG@30 | NDCG@5 | NDCG@10 | NDCG@30 |
| Ordinal loss | $53.00 \pm 0.35$ | $54.88 \pm 0.21$ | $60.19 \pm 0.13$ | $48.84 \pm 0.42$ | $51.02 \pm 0.33$ | $56.98 \pm 0.15$ |
| NDCGLoss 2++ | $52.65 \pm 0.37$ | $54.49 \pm 0.27$ | $59.80 \pm 0.08$ | $49.15 \pm 0.44$ | $51.22 \pm 0.34$ | $57.14 \pm 0.23$ |
| ListNet | $52.33 \pm 0.33$ | $54.26 \pm 0.20$ | $59.63 \pm 0.14$ | $47.81 \pm 0.36$ | $50.20 \pm 0.26$ | $56.41 \pm 0.17$ |
| LambdaRank | $52.29 \pm 0.31$ | $54.08 \pm 0.19$ | $59.48 \pm 0.12$ | $48.77 \pm 0.38$ | $50.85 \pm 0.28$ | $56.72 \pm 0.17$ |
| RMSE | $51.74 \pm 0.48$ | $53.40 \pm 0.38$ | $58.89 \pm 0.31$ | $48.24 \pm 0.52$ | $50.31 \pm 0.41$ | $56.29 \pm 0.19$ |
| RankNet | $50.79 \pm 0.48$ | $52.90 \pm 0.39$ | $58.75 \pm 0.23$ | $47.54 \pm 0.47$ | $49.78 \pm 0.35$ | $55.91 \pm 0.17$ |
| ListMLE | $50.20 \pm 0.40$ | $52.19 \pm 0.23$ | $57.94 \pm 0.18$ | $46.98 \pm 0.45$ | $49.14 \pm 0.36$ | $55.24 \pm 0.2$ |
|  | XGBoost |  |  |  |  |  |
|  | NDCG@5 |  | NDCG@10 |  | NDCG@30 |  |
| rank:pairw | 46.8 |  | 49.17 |  | 55.33 |  |

provide results on WEB30K reported in other works in Table 1. For models with multiple variants, we cite the best result reported in the original work. In all tables, boldface is the best value column-wise.

### 5.5 Re-ranking

All experiments on WEB30K described above were conducted in the ranking setting - input lists of items were treated as unordered, thus positional encoding was not used. To verify the effect of positional encoding on the model's performance, we conduct the following experiments on WEB30K. To avoid information leak, training data ${ }^{6}$ is divided into five folds and five XGBoost models are trained, each on four folds. Each model predicts scores for the remaining fold, and the entire dataset is sorted according to these scores.

Finally, we train the same models ${ }^{7}$ as earlier on the sorted dataset but use fixed positional encoding. Results are presented in Table 4. For brevity, we focus on NDCG@5. As expected, the models are able to learn positional information and demonstrate improved performance over the plain ranking setting.

Table 4: NDCG@5 on re-ranking task

| Loss | With PE | Self-attention w/o PE |
| :--- | :---: | :---: |
| Ordinal loss | $\mathbf{5 2 . 6 7}$ | $\mathbf{5 2 . 2 0}$ |
| NDCGLoss 2++ | 52.24 | 51.40 |
| RMSE | 51.85 | 50.23 |
| ListNet | 51.77 | 51.34 |
| LambdaRank | 51.51 | 51.22 |
| ListMLE | 50.90 | 49.19 |
| RankNet | 50.58 | 49.90 |

### 5.6 Usage in latency-sensitive conditions

The results discussed so far were concerned with the offline evaluation of the proposed model. Its favourable performance suggests an online test in a live search engine, say that of Allegro.pl, is justified. Indeed such a test is planned for further study. It needs to be taken[^4]

into account though that the complexity of the proposed model is $O\left(n^{2}\right)$, due to the usage of the self-attention operation. The inference speed is further influenced by the number $N$ of encoder blocks used. To allow the usage of the model in latency-sensitive scenarios, one might consider using it as a final stage ranker on shorter lists of input elements, reducing the number of encoder blocks, or applying one or more of model distillation [15], quantisation [18] or pruning [13] techniques known from the literature.

Table 5: Ablation study

| Parameter | Value | Params | WEB30K NDCG@5 |
| :--- | :---: | :---: | :---: |
| baseline |  | $950 \mathrm{~K}$ | 52.64 |
| $H$ | 1 | $950 \mathrm{~K}$ | 52.44 |
|  | 4 | $950 \mathrm{~K}$ | 52.59 |
| $N$ | 1 | $250 \mathrm{~K}$ | 50.62 |
|  | 2 | $490 \mathrm{~K}$ | 52.15 |
| $d_{h}$ | 64 | $430 \mathrm{~K}$ | 51.58 |
|  | 128 | $509 \mathrm{~K}$ | 51.96 |
|  | 256 | $650 \mathrm{~K}$ | 52.19 |
|  | 1024 | $1540 \mathrm{~K}$ | 52.86 |
| $p_{\text {drop }}$ | 0.0 | $950 \mathrm{~K}$ | 42.18 |
|  | 0.1 | $950 \mathrm{~K}$ | 51.22 |
|  | 0.2 | $950 \mathrm{~K}$ | 52.58 |
|  | 0.3 | $950 \mathrm{~K}$ | $\mathbf{5 2 . 8 6}$ |
|  | 0.5 | $950 \mathrm{~K}$ | 52.26 |
|  | 30 | $950 \mathrm{~K}$ | 50.94 |
|  | 60 | $950 \mathrm{~K}$ | 51.78 |

## 6 ABLATION STUDY

To gauge the effect of various hyperparameters of self-attention based ranker on its performance, we performed the following ablation study. We trained the context-aware ranker with the ordinal loss on Fold 1 of WEB30K dataset, evaluating the ranker on the validation subset. We experimented with a different number $N$ of encoder blocks, $H$ attention heads, length $l$ of longest list used in training, dropout rate $p_{d r o p}$ and size $d_{h}$ of hidden dimension.

Results are summarised in Table 5. Baseline model (i.e. the best performing context-aware ranker trained with ordinal loss) had the following values of hyperparameters: $N=4, H=2, l=240$, $p_{\text {drop }}=0.4$ and $d_{h}=512$. We observe that a high value of dropout is essential to prevent overfitting but setting it too high results in performance degradation. A similar statement is true of the number of the attention heads - even though it is better to use multiple attention heads as opposed to a single one, we notice a decline in performance when using more than two attention heads. Finally, stacking multiple encoder blocks and increasing the hidden dimension size increases performance. However, we did not test the effect of stacking more than 4 encoder blocks or using hidden dimension sizes larger than 1024 due to GPU memory constraints.

## 7 CONCLUSIONS

In this work, we addressed the problem of constructing a contextaware scoring function for learning to rank. We adapted the selfattention based Transformer architecture from the neural machine translation literature to propose a new type of scoring function for LTR. We demonstrated considerable performance gains of proposed neural architecture over MLP baselines across different losses and types of data, both in ranking and re-ranking setting. In particular, we established the new state-of-the-art performance on WEB30K. These experiments provide strong evidence that the gains are due to the ability of the model to score items simultaneously. As a result of our empirical study, we observed the strong performance of models trained to optimise ordinal loss function. Such models outperformed models trained with well-studied losses like NDCGLoss $2++$ or LambdaRank, which were previously shown to provide tight bounds on IR metrics like NDCG. On the other hand, we observed the surprisingly poor performance of models trained to optimise RankNet and ListMLE losses. In future work, we plan to investigate the reasons for both good and poor performance of the aforementioned losses, in particular, the relation between ordinal loss and NDCG.

## REFERENCES

[1] Qingyao Ai, Keping Bi, Jiafeng Guo, and W. Bruce Croft. 2018. Learning a Deep Listwise Context Model for Ranking Refinement. CoRR abs/1804.05936 (2018) arXiv:1804.05936 http://arxiv.org/abs/1804.05936

[2] Qingyao Ai, Xuanhui Wang, Sebastian Bruch, Nadav Golbandi, Mike Bendersky, and Marc Najork. 2019. Learning Groupwise Multivariate Scoring Functions Using Deep Neural Networks. In Proceedings of the 5th ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR).

[3] Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer Normalization. ArXiv abs/1607.06450 (2016).

[4] Ricardo A. Baeza-Yates and Berthier Ribeiro-Neto. 1999. Modern Information Retrieval. Addison-Wesley Longman Publishing Co., Inc., Boston, MA, USA.

[5] Irwan Bello, Sayali Kulkarni, Sagar Jain, Craig Boutilier, Ed Huai-hsin Chi, Elad Eban, Xiyang Luo, Alan Mackey, and Ofer Meshi. 2018. Seq2Slate: Re-ranking and Slate Optimization with RNNs. CoRR abs/1810.02019 (2018). arXiv:1810.02019 http://arxiv.org/abs/1810.02019

[6] Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005. Learning to Rank Using Gradient Descent. In Proceedings of the 22Nd International Conference on Machine Learning (ICML '05). ACM, New York, NY, USA, 89-96. https://doi.org/10.1145/1102351.1102363

[7] Christopher J. Burges, Robert Ragno, and Quoc V. Le. 2007. Learning to Rank with Nonsmooth Cost Functions. In Advances in Neural Information Processing Systems 19, B. Schölkopf, J. C. Platt, and T. Hoffman (Eds.). MIT Press, 193200. http://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-costfunctions.pdf

[8] Christopher J. C. Burges. 2010. From RankNet to LambdaRank to LambdaMART An Overview. Technical Report. Microsoft Research. http://research.microsoft com/en-us/um/people/cburges/tech_reports/MSR-TR-2010-82.pdf
[9] Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007. Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th International Conference on Machine Learning (ICML '07). ACM, New York, NY, USA, 129-136. https://doi.org/10.1145/1273496.1273513

[10] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). ACM, New York, NY, USA, 785-794. https://doi.org/10.1145/2939672.2939785

[11] Jianpeng Cheng, Li Dong, and Mirella Lapata. 2016. Long Short-Term MemoryNetworks for Machine Reading. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Austin, Texas, 551-561. https://doi.org/10.18653/v1/D16-1053

[12] Mostafa Dehghani, Hamed Zamani, Aliaksei Severyn, Jaap Kamps, and W. Bruce Croft. 2017. Neural Ranking Models with Weak Supervision. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '17). ACM, New York, NY, USA, 65-74. https: //doi.org/10.1145/3077136.3080832

[13] Song Han, Jeff Pool, John Tran, and William J. Dally. 2015. Learning Both Weights and Connections for Efficient Neural Networks. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1 (NIPS'15). MIT Press, Cambridge, MA, USA, 1135-1143.

[14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015), 770-778.

[15] Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the Knowledge in a Neural Network. In NIPS Deep Learning and Representation Learning Workshop. http://arxiv.org/abs/1503.02531

[16] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-Term Memory. Neural Comput. 9, 8 (Nov. 1997), 1735-1780. https://doi.org/10.1162/neco.1997.9. 8.1735

[17] Saratchandra Indrakanti, Svetlana Strunjas, Shubhangi Tandon, and Manojkumar Rangasamy Kannadasan. 2019. Exploring the Effect of an Item's Neighborhood on its Sellability in eCommerce. arXiv:cs.IR/1908.03825

[18] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. 2017. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. (12 2017).

[19] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated Gain-based Evaluation of IR Techniques. ACM Trans. Inf. Syst. 20, 4 (Oct. 2002), 422-446. https://doi. org $/ 10.1145 / 582415.582418$

[20] Ray Jiang, Sven Gowal, Yuqiu Qian, Timothy Mann, and Danilo J. Rezende. 2019. Beyond Greedy Ranking: Slate Optimization via List-CVAE. In International Conference on Learning Representations. https://openreview.net/forum?id= r1xX42R5Fm

[21] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems 30, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.). Curran Associates, Inc., 3146-3154. http://papers.nips.cc/paper/6907lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf

[22] Diederik P. Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Optimization. http://arxiv.org/abs/1412.6980 cite arxiv:1412.6980Comment: Published as a conference paper at the 3rd International Conference for Learning Representations, San Diego, 2015.

[23] Tie-Yan Liu. 2009. Learning to Rank for Information Retrieval. Found. Trends Inf. Retr. 3, 3 (March 2009), 225-331. https://doi.org/10.1561/1500000016

[24] Andrew Y. Ng. 2004. Feature Selection, L1 vs. L2 Regularization, and Rotational Invariance. In Proceedings of the Twenty-first International Conference on Machine Learning (ICML '04). ACM, New York, NY, USA, 78-. https://doi.org/10.1145/ 1015330.1015435

[25] Zhenxing Niu, Mo Zhou, Le Wang, and Xinbo Gao. 2016. Ordinal Regression with Multiple Output CNN for Age Estimation. 4920-4928. https://doi.org/10. 1109/CVPR.2016.532

[26] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. 2017. Automatic Differentiation in PyTorch. In NIPS Autodiff Workshop.

[27] Changhua Pei, Yi Zhang, Yongfeng Zhang, Fei Sun, Xiao Lin, Hanxiao Sun, Jian Wu, Peng Jiang, and Wenwu Ou. 2019. Personalized Re-ranking for Recommendation. arXiv:cs.IR/1904.06813

[28] Tao Qin and T. M. Liu. 2013. Introducing LETOR 4.0 Datasets. ArXiv abs/1306.2597 (2013).

[29] Kihyuk Sohn, Honglak Lee, and Xinchen Yan. 2015. Learning Structured Output Representation using Deep Conditional Generative Models. In Advances in Neural Information Processing Systems 28, C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett (Eds.). Curran Associates, Inc., 3483-3491. http://papers.nips.cc/paper/5775-learning-structured-outputrepresentation-using-deep-conditional-generative-models.pdf

[30] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. 7. Mach. Learn. Res. 15, 1 (Jan. 2014), 1929-1958. http://dl.acm.org/ citation.cfm?id=2627435.2670313

[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems 30, I. Guyon, U. V Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.) Curran Associates, Inc., 5998-6008. http://papers.nips.cc/paper/7181-attentionis-all-you-need.pdf

[32] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer Networks. In Advances in Neural Information Processing Systems 28, C. Cortes, N. D. Lawrence D. D. Lee, M. Sugiyama, and R. Garnett (Eds.). Curran Associates, Inc., 2692-2700. http://papers.nips.cc/paper/5866-pointer-networks.pdf

[33] Ellen M. Voorhees. 1999. The TREC-8 Question Answering Track Report. In In Proceedings of TREC-8. 77-82.

[34] Xuanhui Wang, Cheng Li, Nadav Golbandi, Mike Bendersky, and Marc Najork 2018. The LambdaLoss Framework for Ranking Metric Optimization. In Proceed ings of The 27th ACM International Conference on Information and Knowledge Management (CIKM '18). 1313-1322.

[35] Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise Approach to Learning to Rank: Theory and Algorithm. In Proceedings of the 25th International Conference on Machine Learning (ICML '08). ACM, New York, NY, USA, 1192-1199. https://doi.org/10.1145/1390156.1390306
</end of paper 1>


