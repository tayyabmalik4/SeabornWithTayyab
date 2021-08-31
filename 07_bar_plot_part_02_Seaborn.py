# (07)*************************Bar plot part 2 in Seaborn******************

# /////we use the differnt kewords in barplot
# /////**kwags means that key-word-arguments(kwags)


# importing the libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# /////importing the file with the help of seaborn
tip_df=sns.load_dataset('tips')
# print(tip_df)

# /////////these are the parameters of barplot graph in seaborn
parameters_of_barplot="""(*, x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=np.mean, ci=95, n_boot=1000, units=None, seed=None, orient=None, color=None, palette=None, saturation=0.75, errcolor=".26", errwidth=None, capsize=None, dodge=True, ax=None, **kwargs)"""

# /////if we want to change the background color than we use sns.set() function
sns.set()

# ////starting the barplot functions
# ////when we use kwags parameter we use various functions in the kwags function by the use of dictionary
kwargs={'alpha':0.9,'linestyle':'--','linewidth':5,'edgecolor':'r'}
aa=sns.barplot(x='day',y='total_bill',data=tip_df,hue='sex',**kwargs)

# //////if we want to creating the title and xlabel and ylabel using seaborn than we use seaborn_variable.set() function and in this function we assign the values
aa.set(title='Bar chart of Seaborn',xlabel='Days',ylabel='Total Bills')

plt.show()
