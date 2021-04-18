**Download the anonymized dataset describing [persons.csv](https://raw.githubusercontent.com/tyler-frazier/intro_data_science/main/data/persons.csv) from a West African county and import it into your PyCharm project workspace (right click and download from the above link or you can also find the data pinned to the slack channel). First set the variable `wealthC` as your target. It is not necessary to set a seed.**

When running a linear regression of the persons.csv data with wealthC set as the target data I got the following results 

- Train Score = .73582
- Test Score = .73498
- Train MSE = .44282
- Test MSE = .44387

I then standardized the data and got the following results 

- Train Score = .73551
- Test Score = .73448
- Train MSE = .44334
- Test MSE = .44474

Standardizing actually created a weaker training and testing and a small increase in MSE so standardizing the data was unsuccessful in finding a better result. 

I than ran a ridge regression with the data to try and find better results. 

- Train Score = .73584
- Test Score = .73505
- Train MSE = .44279
- Test MSE = .44376

The results from the ridge regression are currently the best results. 

I then ran a lasso regression 

- Train Score = .73583
- Test Score = .73506
- Train MSE = .44279
- Test MSE = .44375

In conclusion the Lasso and Ridge regression are equally good and more predictive then a simple linear regression. 

I then ran the code with the variable wealth I 

- Train Score = .82582
- Test Score = .82499
- Train MSE = .44282
- Test MSE = .44387

I then changed the target value from the ordinal data of wealthC to the continuous data of wealthI I got the following results 

- Train Score = .8258220621779602
- Test Score = .8249892516516357
- Train MSE = 1750318361.3776894
- Test MSE = 1754946098.4297833

I then standardized the data and got the following 

- Train Score = .8255764394674567
- Test Score = .8245760767201565
- Train MSE = 1752785711.6359572
- Test MSE = 1759125775.5920272

I then ran a ridge regression on the data and received the following. 

- Train Score = 0.8258368584473411
- Test Score = .8250203378510275
- Train MSE = 1750169674.1450438
- Test MSE = 1754636027.9290862

Lastly I ran a Lasso Regression and received the following results. 

- Train Score = 0.8258372542786979
- Test Score = 0.82501982545465
- Train MSE = 1750165696.4335837
- Test MSE = 1754641136.1647427