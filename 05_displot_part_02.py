# (05)*****************distplot part 02 in seaborn***************************

# ------------importing the libraries
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

# /////this is the code of seaborn
# ////if we want to increase the figure size than we use plt.figure function
# plt.figure(figsize=(13,13))

bins=[5,10,15,20,25,30,35,40,45,50,55]
# /////if we want to change the line background color than we use set function
sns.set() 
# sns.distplot(tip_df['total_bill'],bins=bins)

# /////if we use xtrick than we shown the x-label indexs
plt.xticks(bins)

# /////when we want to change the line colors as we wish than we use hist-kws(histpgram-keywords) functions and we input in dictionary
# //////when we want to change the edgecolor than we use edge color function
# /////when we want to increse the linewidth than we use linewidth function
# ////when we want to change the stple of the line then we use linestyle function
# ////if we want to increrase or decrese the opacity of the line color than we use alpha function and we input value(0-1)
# sns.distplot(tip_df['total_bill'],bins=bins,hist_kws={'color':'red','edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.9})

# //////if we want to change the kernal-density-estimate(kde) color, opasity, linewidth, linestyle than we use kde_kws(kernal-density-estimate keywords) and it is also input in dictionary 
# sns.distplot(tip_df['total_bill'],bins=bins,hist_kws={'color':'red','edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.9}, kde_kws={'color':'g','linewidth':5,'linestyle':'--','alpha':0.9})

# /////if we want to change the rug color,linewidth, linestyle,opasity than we use rug_kws function
# sns.distplot(tip_df['total_bill'],bins=bins,hist_kws={'color':'red','edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.9}, kde_kws={'color':'g','linewidth':5,'linestyle':'--','alpha':0.9},rug=True,rug_kws={'color':'k','linewidth':3,'linestyle':'--','alpha':0.9})

# //////if we want to change the normalize line color,linewidth,linestyle and opasity than we use fit_kws function and we also kde=False
# /////if we want to show the label than we use label function
# sns.distplot(tip_df['total_bill'],bins=bins,hist_kws={'color':'red','edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.9},kde=False,rug=True,rug_kws={'color':'k','linewidth':3,'linestyle':'--','alpha':0.9},fit=norm,fit_kws={'color':'m','linewidth':3,'linestyle':'--','alpha':0.9},label="Sultan's Production")

# /////if we want to multiple graph is showing gthe one graph than we use this function
sns.distplot(tip_df['size'],label='size')
sns.distplot(tip_df['tip'],label='tip')
sns.distplot(tip_df['total_bill'],label='Total_Bill')


# /////Creating the title, xlabel and also ylabel function
plt.title("Histogram of Resturent",fontsize=25)
plt.xlabel("Total Bills",fontsize=15)
plt.ylabel("Indexs",fontsize=15)
plt.legend(loc=2)

# /////if we want to sorting the values than we use sort function
# print(tip_df.total_bill.sort_values())

# ////to showing the graph we use plt.show() FUNCTION and this is the matplotli function
plt.show()
