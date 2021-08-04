# (04)**********************Distplot in seaborn******************
# ////histogram and dist function is similar we use as we wish

# ------------importing the libraries
from re import T
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kde, norm


# -----------importing the dataset in seaborn github repositry
tip_df=sns.load_dataset('tips')
# print(tip_df)

# ////we use these parameters in dist function
parameters="""(a=None, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None, x=None)"""

# /////ploting the dist graph
# sns.distplot(tip_df['size'])
# sns.distplot(tip_df['tip'])
# sns.distplot(tip_df['total_bill'])


# /////if we use bins function its means we breaks into parts 
bins=np.arange(5,60,12)

# //////if we don't show the graph lines than we use hist=False function and we use just kernal density ensidator 
# sns.distplot(tip_df['total_bill'],bins=bins,hist=False)

# /////if we want to remove the kernal density encidator than we use kde=False function
# sns.distplot(tip_df['total_bill'],bins=bins,kde=False)

# /////if we want to show the some spacific graph then we use rug function
# sns.distplot(tip_df['total_bill'],rug=True)

# ////if we to show the normilize line plot thab we use fit function but if we use this function we from scipy.stats import norm library and if we remove the kernal density line than also use jde=False function

# ////if we want to change the color than we use color function

# ////if we want to convered the histogram vertical to horizontal than we use vertical=True function

# ////we don't use the hist_norm functin maximum

# ////if we want to change the xlabel name than we use axlabel function

# ////if we want to creat the label than we use label function and if we use this function we also use plt.legend function
sns.distplot(tip_df['total_bill'],fit=norm,kde=False,color='r',vertical=False, axlabel='Hacked',label='Total Bill')
plt.legend()
# ////we creating the title and xlabel and ylabel function
plt.title('Tips Calculated')
plt.xlabel('ToTal BILL')
plt.ylabel('labels')
plt.show()

