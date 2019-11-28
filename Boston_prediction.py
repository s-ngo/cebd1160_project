import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

# Loading the Boston dataset and exploring
print('Loading the dataset and exploring..')
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston.keys())
# print(boston.DESCR)
# print(boston.data.shape)
# print(boston.feature_names)
# print(boston.target)

# Building the data frame and adding the target "MEDV" as the "PRICE" column at the end
print('Building the Boston data frame and adding "PRICE" column..')
bostondf = pd.DataFrame(boston.data, columns=boston.feature_names)
bostondf['PRICE'] = boston.target
print(bostondf.head())

# To make the price more revelant, I accounted for inflation since 1978,
# I multiply the PRICE by 423%
bostondf['PRICE'] = round(bostondf['PRICE'] * 4.23, 2)
print(bostondf.head())

# Making scatter plots to see all the feature vs PRICE to check correlationS
print('Creating scatter plots with all the features vs PRICE..')
os.makedirs('figures/', exist_ok=True)

for feature_name in bostondf.columns:
   plt.scatter(bostondf[feature_name], bostondf['PRICE'], color='g', marker=".", s=5)
   plt.title(feature_name + ' to PRICE')
   plt.xlabel(feature_name)
   plt.ylabel('PRICE')
   plots_file = 'figures/' + '/boston_scatter_' + feature_name + '_to_PRICE' + '.' + 'png'
   plt.savefig(plots_file, format=None)
   plt.clf()
   plt.close()

plt.figure(figsize=(12,12))
sns.heatmap(bostondf.corr(), annot=True, cmap='ocean')
plt.title('Correlation Heatmap of the Boston dataframe')
plt.savefig('figures/boston_corr_heatmap.png')
plt.clf()
plt.close()

sns.distplot(bostondf['PRICE'], bins=30, color='g', label='Distribution of PRICE')
plt.title('Distribution Plot of PRICE')
plt.savefig('figures/boston_displot_PRICE.png', dpi=300)
plt.clf()
plt.close()


# Seeing that we have many values of 50.00 * 4.23 in MDEV,
# I decided to drop those lines because I found out they might be missing or false values.
# To ultimately have a better non-skewed result.
# print('After looking at the plots, dropping the lines with MEDV=50.00 because possible false values..')
# indexNames = bostondf[bostondf['PRICE'] == 211.5].index
# print(bostondf[bostondf['PRICE'] == 211.5])
# bostondf.drop(indexNames, inplace=True)
# print(bostondf.shape)

# After looking at the scatter plots, I see the strongest correlation is between
# the RM (average number of rooms per dwelling) and the LSTAT (% lower status of the population) features
# vs the PRICE (Median value of owner-occupied homes in $10000's) target
print('RM and LSTAT are the most correlated vs PRICE so I make a new dataset with 2 features only..')
correlated_bostondf = bostondf[['RM', 'LSTAT']]


# Splitting correlated_bostondf and target datasets into train and test
print('Splitting the correlated dataset and target into train and test to test different estimators.. ')
X_train, X_test, y_train, y_test = train_test_split(correlated_bostondf, bostondf['PRICE'], test_size=0.35, random_state=1)

# After testing different estimators (LinearRegression, Lasso, ElasticNet, GradientBoostingRegressor),
# I concluded that GradientBoostingRegressor was the best performing
# lm = LinearRegression()
# lm = Lasso(random_state=1)
lm = ElasticNet(random_state=1)
# lm = GradientBoostingRegressor(random_state=1)

lm.fit(X_train, y_train)

# Predicting and printing the results for our test dataset
print('Predicting and printing the results..')
predicted_values = lm.predict(X_test)
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

# Plotting the residuals: difference between real and predicted
sns.set(palette="ocean")
residuals = y_test - predicted_values

print('Plotting the results..')
sns.scatterplot(y_test, predicted_values)
plt.plot([0, 50], [0, 50], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Real Value vs Predicted Value plot')
plots_file = 'figures/' + '/Real_vs_Predicted.' + 'png'
plt.savefig(plots_file, format=None)
plt.clf()
plt.close()

sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.title('Real Value vs Residual (difference) plot')
plots_file = 'figures/' + '/Real_vs_Residual.' + 'png'
plt.savefig(plots_file, format=None)
plt.clf()
plt.close()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution')
plots_file = 'figures/' + '/Residual_Distribution.' + 'png'
plt.savefig(plots_file, format=None)
plt.clf()
plt.close()

# Understanding the error that we want to minimize
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
