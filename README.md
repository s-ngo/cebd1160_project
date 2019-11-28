# cebd1160: predicting house pricing

| Name | Date |
|:-------|:---------------|
|Stephane Ngo|November 30, 2019|

-----

### Resources

- Python script : `Boston_prediction.py`
- Results figure/saved file :  `figures/`
- Dockerfile : `Dockerfile`
- runtime-instructions in a file named RUNME.md

-----

## Research Question

Using the Boston Housing Dataset, can we build a machine learning model to predict the value of a house ?

### Abstract

We have the Boston Housing dataset, a well known toy dataset from 1978 with 506 samples with 13 features of homes as well as the price.
Using this dataset, we can attempt to train a machine learning model to predict the price of a house based on certain given variables. 
Firstly, thought data exploration, we verify the integrity of the dataset and the correlation between a variable versus the median value of a home (MDEV). 
Secondly, finding that some variables are correlated, we trained and tested a linear regression model to predict the price. 
With the results of the chosen estimator (GradientsBoostingRegression), we concluded that the results are not strong enough to use as a machine learning prediction. 

### Introduction

The Boston dataset for this project originates from the UCI Machine Learning Repository. The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. (Ref. [1]) 

### Methods

- Starting the project by doing data exploration into the dataset. 
- As this data was from 1978, I adjusted the price value to account for 1978 to 2019 inflation. [1]
- For each feature in the dataset, I created scatter graph of the features vs PRICE because we are trying to predict price. A correlation heatmap and PRICE distribution graph was also created.
- After examining these graphs the following modifications we done : 
    - Dropping 14 lines with PRICE = 211.5 because possible false values. [1]
    ![Dristribution Plot of PRICE](figures/boston_displot_PRICE.png)
    - The most correlated features with PRICE that can be used to train our model: RM and LSTAT. A new dataframe created.
    ![heatmap](figures/boston_heatmap.png)
- Training the linear regression with different estimators (LinearRegression, Lasso, ElasticNet, GradientBoostingRegressor) and using MSE as the performance metric for predicting house price.
- Plotting the results of the real vs predicted values.

### Results

1. Correlation in the Boston dataset

There's is strong positive correlation, as the number of rooms increase, the house price increases.
![scatter_RM_to_PRICE](figures/boston_scatter_RM_to_PRICE.png)

There is a strong negative correlation, as the percentage of lower status of population, the house price decreases.
![scatter_LSTAT_to_PRICE](figures/boston_scatter_LSTAT_to_PRICE.png)

- Testing different estimators and using Mean Squared Error as the most appropriate performance metric we are predicting a numeric value (a regression problem)

| Estimator                 | MSE no drop | MSE with drop    |
| ------------------------- | ---------------- | -------------
| LinearRegression          | 510.50 | 339.95 |
| Lasso                     | 521.00 | 346.75 |
| ElasticNet                | 605.93 | 387.29 | 
| GradientBoostingRegressor | 351.67 | 234.09 |

Conclusion: Based on the lowest MSE, GradientBoostRegressor was the best estimator for this dataset. Dropping the false values of 50.00 also lowered our MSE.
![Real_vs_Residual](figures//Real_vs_Residual.png)

### Discussion

Brief (no more than 1-2 paragraph) description about what you did. Include:

interpretation of whether your method "solved" the problem
suggested next step that could make it better.

more data
more features

### References

[1] ritchieng.com, Boston Home Prices Prediction and Evaluation, https://www.ritchieng.com/machine-learning-project-boston-home-prices/, site visited on November 23, 2019.

[2] MEDIUM.COM, Learning Data Science: Day 9 - Linear Regression on Boston Housing Dataset https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef, site visited on November 23, 2019.

[3] KAGGLE.COM, Boston house price prediction https://www.kaggle.com/shreayan98c/boston-house-price-prediction, site visited on November 23, 2019.

[4] TOWARDSDATASCIENCE.COM, Linear Regression on Boston Housing Dataset https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155, site visited on November 23, 2019

[5] SCIKIT-LEARN.ORG, Choosing the right estimator, https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html, site visited on November 23, 2019




-------
