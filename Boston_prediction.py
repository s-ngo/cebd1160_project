import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

# Loading the Boston dataset, adding MEDV to the dataframe
print('Loading and building the Boston data frame..')
from sklearn.datasets import load_boston
bostonds = load_boston()
bostondf = pd.DataFrame(bostonds.data, columns=bostonds.feature_names)
bostondf['MEDV'] = bostonds.target
print(bostondf)
# print(bostondf.describe())

# Making scatter plots to see all the feature vs MEDV to check correlation
print('Creating scatter plots with all the features vs MEDV..')
os.makedirs('figures/', exist_ok=True)
for feature_name in bostondf.columns:
   plt.scatter(bostondf[feature_name], bostonds.target, color='g', marker=".", s=5)
   plt.title(feature_name + 'to MEDV')
   plt.xlabel(feature_name)
   plt.ylabel('MEDV')
   plots_file = 'figures/' + '/scatter_' + feature_name + '_to_MEDV' + '.' + 'png'
   plt.savefig(plots_file, format=None)
   plt.clf()
   plt.close()

sns.distplot(bostondf['MEDV'], bins=30, color='g', label='Distribution of MEDV')
plt.savefig('figures/displot_MEDV.png', dpi=300)
plt.clf()
plt.close()

# Seeing that we have many values of 50.00 in MDEV,
# I decided to drop those lines because I found out they might be missing or false values.
# To ultimately have a better result.
print('After looking at the plots, dropping the lines with MEDV=50.00 because possible false values..')
indexNames = bostondf[bostondf['MEDV'] == 50.00].index
print(bostondf[bostondf['MEDV'] == 50.00])
bostondf.drop(indexNames, inplace=True)
print(bostondf.shape)

# After looking at the scatter plots, I see the strongest correlation is between
# the RM (average number of rooms per dwelling) and the LSTAT (% lower status of the population) features
# vs the MDEV (Median value of owner-occupied homes in $1000's) target
print('RM and LSTAT are the most correlated vs MDEV so I make a new dataset with 2 features only..')
correlated_bostondf = bostondf[['RM', 'LSTAT']]


# Splitting correlated_bostondf and target datasets into train and test
print('Splitting the correlated dataset and target into train and test to test different estimators.. ')
X_train, X_test, y_train, y_test = train_test_split(correlated_bostondf, bostondf['MEDV'], test_size=0.35, random_state=1)

# After testing different estimators (LinearRegression, Lassor, GradientBoostingRegressor),
# I concluded that GradientBoostingRegressor was the best performing
lm = GradientBoostingRegressor(random_state=1)
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
