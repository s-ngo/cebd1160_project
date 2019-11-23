import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
# %matplolib inline

from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target
df = pd.DataFrame(X, columns= boston.feature_names)

f, ax = plt.subplots(figsize=(12, 12))
corr = df.select_dtypes(exclude=["object", "bool"]).corr()

# TO display diagonal matrix instead of full matrix.
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap.
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio.
g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, annot=True, fmt='.2f', \
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

# plt.subplots_adjust(top=0.99)
plt.title("Diagonal Correlation HeatMap")
os.makedirs('plots/', exist_ok=True)
plt.savefig('plots/boston_heatmap.png')
plt.close()

print(f'data: {boston.data}')
print(f'target: {boston.target}')
print(f'feature_names: {boston.feature_names}')

