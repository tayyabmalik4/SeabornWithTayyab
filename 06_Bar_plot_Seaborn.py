# (06)*************************Bar Plot Seaborn in python***************************

# /////when we comparing and alalysis the data with the help of seaborn than we use Bar plot function

# /////importing the librarys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# /////importing the file with the help of seaborn
tip_df=sns.load_dataset('tips')
# print(tip_df)

# /////////these are the parameters of barplot graph in seaborn
parameters_of_barplot="""(*, x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=np.mean, ci=95, n_boot=1000, units=None, seed=None, orient=None, color=None, palette=None, saturation=0.75, errcolor=".26", errwidth=None, capsize=None, dodge=True, ax=None, **kwargs)"""

# ////starting the barplot function
# ////when comparing that who pay mostly than we use hue function
# sns.barplot(x=tip_df.day, y=tip_df.total_bill, hue=tip_df.sex)

# ////if we don't repeat the variable than we use data parameter
# sns.barplot(x='day',y='total_bill',data=tip_df,hue='sex')

# //////if we want to order the days than we use order function
order=['Sun','Thur','Fri','Sat']
# ///////if we want to change the hue order than we use hue_order
hue_order=['Female','Male']
# /////if we want to change the mean value than we use estimate parameter and this is using in numpy

# /////if we change the confidence-interval(ci) level than we use ci parameter

# ////if we want to change the confidace_interval(ci) than we use n_boot parameter

# ////if we want to convert the vertical to horizontal graph than we use orient parameter and if we use this parameter when we use not a int variable than it accures error-----vertial(v),horizontal(h)\

# ////if we want to change the color of the graph lines than we use color parameters

# /////if we want to change the barplot whole colors than we use palette parameter

# //////if we want to max or min the opacity than we use saturation parameter(0-1)

# /////if we want to change the color of the middle line than we use errcolor(0-1)

# /////if we want to change the size of the err line than we use errwidth

# ////if we want to max the err of the cap than we use capsize parameter

# /////if we want to showing the data male and female in one line than we use dodge parameter
sns.barplot(x='day',y='total_bill',data=tip_df,hue='sex',order=order,hue_order=hue_order,estimator=np.max,ci=50,n_boot=5, orient='v',color='g',palette='hot',saturation=0.3,errcolor='0.3',errwidth=12,capsize=0.5,dodge=True)



# ////to showing the graph
plt.show()