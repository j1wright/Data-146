To complete this project I imported the following libraries and functions 

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split as tts

from sklearn.preprocessing import StandardScaler as SS 

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn import tree
```

I then defined the DoKFold and other functions below. 

```python
def DoKFold(model, X, y, k, scaler = None):

  kf = KFold(n_splits = k, shuffle = True)



  train_scores = []

  test_scores = []

  train_mse = []

  test_mse = []



  for idxTrain, idxTest in kf.split(X):

​    Xtrain = X[idxTrain, :]

​    Xtest = X[idxTest, :]

​    ytrain = y[idxTrain]

​    ytest = y[idxTest]



​    if scaler != None:

​      Xtrain = scaler.fit_transform(Xtrain)

​      Xtest = scaler.transform(Xtest)



​    model.fit(Xtrain, ytrain)



​    train_scores.append(model.score(Xtrain,ytrain))

​    test_scores.append(model.score(Xtest, ytest))



​    ytrain_pred = model.predict(Xtrain)

​    ytest_pred = model.predict(Xtest)

​    train_mse.append(np.mean((ytrain-ytrain_pred)**2))

​    test_mse.append((np.mean(ytest-ytest_pred)**2))



  return train_scores, test_scores, train_mse, test_mse
```

```python
def GetData(scale=False):

  Xtrain, Xtest, ytrain, ytest = tts(X, y, test_size=0.4)

  ss = StandardScaler()

  if scale:

​    Xtrain = ss.fit_transform(Xtrain)

​    Xtest = ss.transform(Xtest)

  return Xtrain, Xtest, ytrain, ytest
```

```python
def CompareClasses(actual, predicted, names=None):

  accuracy = sum(actual == predicted) / actual.shape[0]

  classes = pd.DataFrame(columns=['Actual', 'Predicted'])

  classes['Actual'] = actual

  classes['Predicted'] = predicted

  conf_mat = pd.crosstab(classes['Predicted'], classes['Actual'])

  \# Relabel the rows/columns if names was provided

  if type(names) != type(None):

​    conf_mat.index = y_names

​    conf_mat.index.name = 'Predicted'

​    conf_mat.columns = y_names

​    conf_mat.columns.name = 'Actual'

  print('Accuracy = ' + format(accuracy, '.2f'))

  return conf_mat, accuracy
```

I then imported and cleaned the data by deleting any NaN values and then setting the dataframe to their respective X and y variables. 

- Execute a K-nearest neighbors classification method on the data. What model specification returned the most accurate results? Did adding a distance weight help?

  When I used KNN the best result was the range 20 , 80 with a test score of .541231495

  Adding distance to the weight did not help with the solution and you can see that with the graphs below. 

  **Without adding distance** 

  ![](C:\Users\Jason\OneDrive\Desktop\Data Science\5b\no distance weight.PNG)

  **With adding distance** 

  ![](C:\Users\Jason\OneDrive\Desktop\Data Science\5b\distance weight.PNG)

- Execute a logistic regression method on the data. How did this model fair in terms of accuracy compared to K-nearest neighbors?

When I ran the logistic regression I received a testing score of .5466081015. The logistic regression score was slightly higher than the KNN testing score so in this scenario logistic regression is a better method than usage of KNN. 

- Next execute a random forest model and produce the results. See the number of estimators (trees) to 100, 500, 1000 and 5000 and determine which specification is most likely to return the best model. Also test the minimum number of samples required to split an internal node with a range of values. Also produce results for your four different estimator values by both comparing both standardized and non-standardized (raw) results.

When running the random forest model the results I got are below 

| Trees | Training | Testing |
| ----- | -------- | ------- |
| 100   | .7893    | .5115   |
| 500   | .7893    | .5105   |
| 1000  | .7893    | .5159   |
| 5000  | .7893    | .5154   |

The best model to use is the one with 1000 trees because it has the highest testing score while the training score is all the same.  I then tested for the minimum number of samples required to split an internal node within a range of values. I found the values range around 20 to 30 would help get the training and testing scores closer to one another. The average training score was around .64290 and the average testing score was around .559785

I then standardized the data and got the following results 

| Trees | Training | Testing |
| ----- | -------- | ------- |
| 100   | 0.7848   | .5012   |
| 500   | 0.7848   | .5051   |
| 1000  | 0.7848   | .5071   |
| 5000  | 0.7848   | .5007   |

As you can see the training score decreased and testing scores also decreased. The best option was still 1000 trees, the un standardized data seamed to be slightly better but both models were relatively close.

- Repeat the previous steps after recoding the wealth classes 2 and 3 into a single outcome. Do any of your models improve? Are you able to explain why your results have changed

When  I reduced the categorical data by combining class 2 and 3 into one class 2 the KNN test score was 0.535033 which is a slightly weaker correlation then from KNN. The results are plotted below. 

![](C:\Users\Jason\OneDrive\Desktop\Data Science\5b\KNN reduced.PNG)

The logistic regression of the new data has a test score of 0.557833 which is actually an improvement of the logistic.

For the random forest the table was the best compared to the other two as you can see below. 

| Trees | Training | Testing |
| ----- | -------- | ------- |
| 100   | .7913    | .4929   |
| 500   | .7913    | .4924   |
| 1000  | .7913    | .498    |
| 5000  | .7913    | .499    |

The best trees for this model would be 5000 trees.

- Which of the models produced the best results in predicting wealth of all persons throughout the large West African capital city being described? Support your results with plots, graphs and descriptions of your code and its implementation. You are welcome to incorporate snippets to illustrate an important step, but please do not paste verbose amounts of code within your project report. Avoiding setting a seed essentially guarantees the authenticity of your results. You are welcome to provide a link in your references at the end of your (part 2) Project 5 report.

The model that produced the best result for predicting wealth for all person in the large West African capital would be logistic regression after combining the categorical data. 