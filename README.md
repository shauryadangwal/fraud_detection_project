# fraud_detection_project
The project gives an overview of fraud detection that are done in credit card transaction. It is necessary for the owner and the for the company from whose gateway payment is being done.
The dataset can be downloaded from https://www.kaggle.com/mlg-ulb/creditcardfraud/download
Supervised learning techniques are widely employed in credit card fraud detection, as they make use of the assumption that fraudulent patterns can be learned from an analysis of past transactions. The task becomes challenging, however, when it has to take account of changes in customer behavior and fraudsters’ ability to invent novel fraud patterns. In this context, unsupervised learning techniques can help the fraud detection systems to find anomalies. In this project unsupervised learning is applied. Unsupervised techniques improve the fraud detection accuracy. Unsupervised outlier scores, computed at different levels of granularity, are compared and tested on a real, annotated, credit card fraud detection dataset.
Unsupervised outlier detection was done by using these ML algorithms
Local Outlier Factor (LOF)
The anomaly score of each sample is called Local Outlier Factor. It measures the local deviation of density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood.

Isolation Forest Algorithm
The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.

Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.


