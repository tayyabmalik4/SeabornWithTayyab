# (02)******************line plot using seaborn****************
# /////we creating the line ploting graph as the help of seaborn
# we inport seaborn as well as matplotlib as well as pandas as well as numpy-----we use all the librarys

# /////importing the librarys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ////we creat array with the help of numpy
days=np.arange(1,16)
np.random.seed(100)
temp=np.random.randint(30,45, 15)

# ////starting the plotting line graph with the help of seaborn
# ////we convered the data to dataFrame by the use of pandas if we plotting the graph in seaborn
tem_df=pd.DataFrame({'days':days,'temp':temp}) 
# print(tem_df)
# ------------these are the functions or parameters of line plotting 
line_func="""(*, x=None, y=None, hue=None, size=None, style=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, units=None, estimator="mean", ci=95, n_boot=1000, seed=None, sort=True, err_style="band", err_kws=None, legend="auto", ax=None, **kwargs)"""
# sns.lineplot(x=days,y='temp',data=tem_df)
# plt.show()

# /////importinh the data
# /////load_dataset is working online because it direct import in github seaborn repoositry but we also use pd library to read and load the file
tips_df=sns.load_dataset('tips')
print(tips_df)
# tips_df=pd.read_csv('C:\\Tayyab Work\\seabornWithTayyab\\dataset_tips_seaborn.csv')
# tips_df=pd.read_csv('dataset_tips_seaborn.csv')

# /////plotting the graph using total_bills and tips
# sns.lineplot(x='total_bill',y='tip', data=tips_df)
# sns.lineplot(x='tip', y='total_bill', data=tips_df)

# /////plottingg the graph using tip and size(no of customers)
# sns.lineplot(x='size',y='tip', data=tips_df)
# sns.lineplot(x='tip', y='size', data=tips_df)

# ////plotting the graph using size and total_bills
sns.lineplot(x='size',y='total_bill', data=tips_df)
# sns.lineplot(x='total_bill', y='size', data=tips_df)
# print(tips_df)
plt.show()
