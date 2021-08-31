# (12)****************************Pairplot Seaborn**************************

# -----------if we want to plotting the many graphs is showing the one graphs than we use pairplot
# -----------pairplot is automatically take 2 columns and plotting the graphs

# /////we use these parameters
parameters="""(data, *, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind="scatter", diag_kind="auto", markers=None, height=2.5, aspect=1, corner=False, dropna=False, plot_kws=None, diag_kws=None, grid_kws=None, size=None)"""

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl

# /////importing the sklearn files on github
from sklearn.datasets import load_breast_cancer

# //////we import the file from sklearn github
cancer_dataset=load_breast_cancer()
# print(cancer_dataset)

# ////now we convered the dictionry to DataFrame
cancer_df=pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],columns=np.append(cancer_dataset['feature_names'],['target']))

# print(cancer_df)
# ///////vars------if we wantt to plotting the graph some spacific columns than we use vars parameters
# /////hue-------when we want to show the properly graph than we use hue function
# /////hue_order------if we want to order the hue than we use hue_order parameter
# ////palette-----when we want to change the color
# ax=sns.pairplot(cancer_df,vars=['mean smoothness', 'mean compactness', 'mean concavity','mean concave points', 'mean symmetry'],hue='target',palette='hot')
# ////x_vars,y_vars------if we want to take the spacific variables on x and y axis than we use x_vars and y_vars
# /////kind-----if we want to chect that the graph is linear or not than we use kind parameter
# ////diag_kind-----if we want to plote the spacific kind of graph than we use diag_kind parameter
# //////marker-----when we plotting the graph is showing the stats or other marker than we use marker parameter
# /////height-----if we want to increase the height of the pairplot than we use height parameter
ax=sns.pairplot(cancer_df,hue='target',palette='hot',x_vars=['mean radius','mean texture'],y_vars=['mean radius','mean texture'],kind='reg',diag_kind='hist',markers=['*','<'])
plt.show()