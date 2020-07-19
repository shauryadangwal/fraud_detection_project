#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dataset is laoded
dataset=pd.read_csv("C:\\Users\\Shaurya\\Desktop\\fraud_detection\\creditcard.csv")

#plotting histograms of each pparameter
dataset.hist(figsize=(20,20))
plt.show()

fraud=dataset[dataset['Class']==1]
valid=dataset[dataset['Class']==0]

print('fraud transaction: {}'.format(len(fraud)))
print('fraud transaction: {}'.format(len(valid)))

#outlier fraction
print(len(fraud)/len(valid))
outlier_fraction = len(fraud)/float(len(valid))

#correlation matrix
corelation=dataset.corr()
fig1=plt.figure(figsize=(12,9))
sns.heatmap(corelation,vmax= .8, square=True)
plt.show()
# Get all the columns from the dataFrame
columns = dataset.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# Store the variable we'll be predicting on
target = "Class"

x= dataset[columns]
y = dataset[target]

# Print shapes
print(x.shape)
print(y.shape)

#unsupervised outlier detectiion
#Local outlier factor 
#isolation forest algo


from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#random state is defined
state=1

#defining outlier detection tool to be compared
classifiers={
        "Isolation Forest": IsolationForest(max_samples=len(x),contamination=outlier_fraction,random_state=state),
        "Local Outlier Factor": LocalOutlierFactor( n_neighbors=20,contamination=outlier_fraction)
        }
# Fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(x)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(x)
        scores_pred = clf.decision_function(x)
        y_pred = clf.predict(x)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))


