# (11)**********************Correlation Heatmap part 3 in seaboern*******************

# -------Definition-----A correlation is a statistical measure of the relationship between two variables(x,y)

# ///////////Range------The correlation coefficient is value from -1 to 1
# /////-1:perfect negative correlation(Ex. X-increases than y-decreases)
# ///// 0:No correlation(Ex. X-increses no effect on Y and vice versa)
# ///// 1:perfect positive correlation(Ex. Xincreses than Y-increses)

# /////Correlation denoted by 'r'
# ----formula of correlation is-------
# r =\frac{\sum\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum\left(x_{i}-\bar{x}\right)^{2} \sum\left(y_{i}-\bar{y}\right)^{2}}}
# --------r	=	correlation coefficient
# --------x_{i}	=	values of the x-variable in a sample
# --------\bar{x}	=	mean of the values of the x-variable
# --------y_{i}	=	values of the y-variable in a sample
# --------\bar{y}	=	mean of the values of the y-variable

# /////importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ///////importing the csv file 
gw=pd.read_csv('datasets\\Who_is_responsible_for_global_warming.csv')
gw1=gw.drop(columns=['Country Code','Indicator Name','Indicator Code'],axis=1).set_index('Country Name')

# print(gw1)

# ///if we want to relation 2 different matrix than we use correlation 
# print(gw1.corr())
gw2=gw1.corr()

# /////plt.figure(figsize=(13,13))when we want to change the size of the figure we use
# plt.figure(figsize=(13,13))


# ax=sns.heatmap(gw2,cmap='coolwarm',annot=True,linewidths=3)
# ax=sns.heatmap(gw2,annot=True,linewidths=3)
# /////if we change the shape of graph than we use tick_params function
# ax.tick_params(size=10,color='y',labelsize=10,labelcolor='y')
# /////creating title,xlabel and ylabel of the graph
# plt.title("Heatmap of correlation",fontsize=25)
# plt.xlabel('Years')
# plt.ylabel('years')

# plt.show()




# ///////we import scikit-learn as the loading of the breast_cancer from github
from sklearn.datasets import load_breast_cancer

# //////we import the file from sklearn github
cancer_dataset=load_breast_cancer()
# print(cancer_dataset)

# ////now we convered the dictionry to DataFrame
cancer_df=pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],columns=np.append(cancer_dataset['feature_names'],['target']))
# print(cancer_df)
plt.figure(figsize=(13,13))
ax=sns.heatmap(cancer_df.corr(),annot=True,linewidths=3)
ax.tick_params(size=10,color='k',labelsize=10,labelcolor='k')
plt.title("Correlation Heatmap of 'Breast Cancer Patients'",fontsize=25)
plt.show()