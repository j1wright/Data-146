- Download the dataset [charleston_ask.csv](https://raw.githubusercontent.com/tyler-frazier/intro_data_science/main/data/charleston_ask.csv) and import it into your PyCharm project workspace. Specify and train a model the designates the asking price as your target variable and beds, baths and area (in square feet) as your features. Train and test your target and features using a linear regression model. Describe how your model performed. What were the training and testing scores you produced? How many folds did you assign when partitioning your training and testing data? Interpret and assess your output.

  ​	This project involved numerous amount of libraries and packages. I used the following code to inset the packages 

  ```python
  import pandas as pd 
  import math as md
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.linear_model import LinearRegression 
  from sklearn.linear_model import Ridge
  from sklearn.model_selection import KFold
  from sklearn.preprocessing import StandardScaler 
  ```

  I then imported the data as follows.

  ```python
  #homes = pd.read_csv('charleston_ask.csv')
  homes = pd.read_csv('charleston_act.csv')
  ```

  The reason that one of the csv files has a comment on it is because that would allow me to more effectively run code on the different files rather than having to recode everything for both files. 

  To separate the target variable from the features when running the training and testing I ran the following. 

  ```python
  X = np.array(homes.iloc[:,1:4]) 
  
  y = np.array(homes.iloc[:,0])
  ```

  I then ran the linear regression with the following code. 

  ```python
  kf = KFold(n_splits = 10, shuffle=True)
  
  train_scores = []
  test_scores = []
  
  for idxTrain, idxTest in kf.split(X):
      Xtrain = X[idxTrain, :]
      Xtest = X[idxTest,:]
      ytrain = y[idxTrain]
      ytest = y[idxTest]
      lin_reg.fit(Xtrain, ytrain)
      train_scores.append(lin_reg.score(Xtrain, ytrain))
      test_scores.append(lin_reg.score(Xtest, ytest))
  
  print('Training: ' + format(np.mean(train_scores), '.3f'))
  print('Testing: ' + format(np.mean(test_scores), '.3f'))
  ```

  With this model I was able to get a training score of 0.019 and a testing score of -0.014 . These Rsquared scores are not good values, a good Rsquared value would be close to one, none of these values are anywhere near that criteria. 

- Now standardize your features (again beds, baths and area) prior to training and testing with a linear regression model (also again with asking price as your target). Now how did your model perform? What were the training and testing scores you produced? How many folds did you assign when partitioning your training and testing data? Interpret and assess your output.

  To attempt to improve the data, I tried standardizing the asking data. To do that I ran the following code. 

  ```python
  def DoKFold(model, X, y, k, standardize=False, random_state=146):
  from sklearn.model_selection import KFold
  if standardize:
      from sklearn.preprocessing import StandardScaler as SS
      ss = SS()
  kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
  # kf = KFold(n_splits=k, shuffle=True)
  train_scores = []
  test_scores = []
  
  for idxTrain, idxTest in kf.split(X):
      Xtrain = X[idxTrain, :]
      Xtest = X[idxTest, :]
      ytrain = y[idxTrain]
      ytest = y[idxTest]
   
      if standardize:
          Xtrain = ss.fit_transform(Xtrain)
          Xtest = ss.transform(Xtest)
      
      model.fit(Xtrain, ytrain)
      
      train_scores.append(model.score(Xtrain, ytrain))
      test_scores.append(model.score(Xtest, ytest))
  
  return train_scores, test_scores
  ```

  ```python
  train_scores, test_scores = DoKFold(lin_reg,X,y,10)
  print('Training: ' + format(np.mean(train_scores), '.3f'))
  print('Testing: ' + format(np.mean(test_scores), '.3f'))
  ```

  This lead to a training score of .020 and a testing score of -.038. These scores are of slight improvement, but still not very good scores to work with when trying to analyze data. 

  - Then train your dataset with the asking price as your target using a ridge regression model. Now how did your model perform? What were the training and testing scores you produced? Did you standardize the data? Interpret and assess your output.

  To try and work for a better Rsquared value, I then went to a ridge regression model to try and get better values for training and testing. 

  To run the ridge regression I ran the following code. 

  ```
  a_range = np.linspace(0, 100, 100)
  k = 10
  avg_tr_score=[]
  avg_te_score=[]
  for a in a_range:
      rid_reg = Ridge(alpha=a)
      train_scores, test_scores = DoKFold(rid_reg,X,y,k, standardize = True)
      avg_tr_score.append(np.mean(train_scores))
      avg_te_score.append(np.mean(test_scores))
      
  idx = np.argmax(avg_te_score)
  print('Optimal alpha value: ' + format(a_range[idx], '.3f'))
  print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
  print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
  
  ```

  The result for the ridge regression was 

  Optimal alpha value: 100.000
  Training score for this value: 0.017
  Testing score for this value: -0.015

  The scores from the ridge regression model was actually worse than the result from standardizing the data. 

- Next, go back, train and test each of the three previous model types/specifications, but this time use the dataset [charleston_act.csv](https://raw.githubusercontent.com/tyler-frazier/intro_data_science/main/data/charleston_act.csv) (actual sale prices). How did each of these three models perform after using the dataset that replaced asking price with the actual sale price? What were the training and testing scores you produced? Interpret and assess your output.

  This part of the project was actually quite easy, as state in the beginning using the comment ability from python I was able to just change the comments and then re-run the whole code. The results were as follows 

  - Linear Regression 
    - actual training = .004
    - actual testing = -.011
  - Standardization 
    - actual training = .004
    - actual testing = .062 
  - Ridge Regression 
    - actual training = .004 
    - actual testing = -.055 

  What is unique about the actual price data is that the Rsquared value for the three different training models is the same. However standardization offered the best actual testing Rsquared value making standardization the best route. None of these Rsquared values are good however and need to be improved. 

- Go back and also add the variables that indicate the zip code where each individual home is located within Charleston County, South Carolina. Train and test each of the three previous model types/specifications. What was the predictive power of each model? Interpret and assess your output.

  To add the zip codes efficiently without messing up the code, I did the same thing with the comment ability of python to make it so all I would have to do is change comments. 

  ```
  y = np.array(homes.iloc[:,0])
  
  X = np.array(homes.iloc[:,1:])
  ```

  The results are as follows 

- Asking with Zip 

  - Linear Regression 
    - training = .280
    - testing = .157
  - Standardization 
    - training = .281
    - testing = .169
  - Ridge Regression 
    - training = .276
    - testing = .193

- Actual with Zip 

  - Linear Regression 
    - training = .340
    - testing = .246
  - Standardization 
    - training = .339
    - testing = .208
  - Ridge Regression 
    - training = .333
    - testing = .219

- consider the model that produced the best results. Would you estimate this model as being overfit or underfit? If you were working for Zillow as their chief data scientist, what action would you recommend in order to improve the predictive power of the model that produced your best results from the approximately 700 observations (716 asking / 660 actual)?

The model that produced the best results would be the linear Regression with Zip codes. That is because of having the highest testing score. The training score is quite high though which would make this model overfit. If I were working for Zillow as their chief data scientist, I would recommend finding even more features to add to the data frames, adding zip codes alone drastically increased the Rsquared value of each model. 