# (03)********************line Plot part 2 Seaborn**************
# ////////we use diferent functions and parameters in the line plotting

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# //////we se the privios data continue
# ////these are the parameters which we use in the lineplot
line_func="""(*, x=None, y=None, hue=None, size=None, style=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, units=None, estimator="mean", ci=95, n_boot=1000, seed=None, sort=True, err_style="band", err_kws=None, legend="auto", ax=None, **kwargs)"""
tips_df=pd.read_csv('datasets\\dataset_tips_seaborn.csv')

# ////if we want to increase the figure size than we use figure function
plt.figure(figsize=(13,13))

# /////if we want to change the background color than we use sns.set() function
sns.set(style='darkgrid')
plt.grid(color='red')
# /////data function use because seaborn take a input as a dataframe 
# ////hue function use when we plotting the graph as a string And when we use hue function lagend function as a default
# ///size function is use when we want to spacific size of the line plotting
# ////style function is use when we spacific the style of line plotting
# ////we use palette function when we want to change the color of lines 
# /////if we want to remove the dashes of the line than we use dashes=False function
# ////marker function is use when we denoted the points
# /////if we want to increase of decrease the size of marker than we use markersize
# ////if we want to remove or full the lagend values than we use lagend function
# sns.lineplot(x='size',y='total_bill',data=tips_df,hue='sex',size=40 ,style='sex',palette='hot',dashes=False,markers=['o','*'],markersize=14,legend='full')

# ////if we want to plotting the data as a day than and at a time we take markers than we take the no of lines and markers are same otherwise it gone a error accure
sns.lineplot(x='size',y='total_bill',data=tips_df,hue='day',size=40 ,style='day',palette='hot',dashes=False,markers=['o','*','<','>'],markersize=14,legend='full')

# ////this is very intersting graph
# x, y = np.random.normal(size=(2, 5000)).cumsum(axis=1)
# sns.lineplot(x=x, y=y, sort=False, lw=1)

# ////we creat the tile and xlabel and ylabel and we also increasing or decreasing the fontsize than we use fontsize function
plt.title("Temperature Karachi",fontsize=20)
plt.xlabel("Size",fontsize=15)
plt.ylabel('total_bill',fontsize=15)
plt.show()