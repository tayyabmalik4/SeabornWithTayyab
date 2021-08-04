# (08)*************************Scatter plot in seaborn*************************

# //////we plot the graph using scatter plot 
# /////we use these parameters to show and handsom graph
parameters="""(*, x=None, y=None, hue=None, style=None, size=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, markers=True, style_order=None, x_bins=None, y_bins=None, units=None, estimator=None, ci=95, n_boot=1000, alpha=None, x_jitter=None, y_jitter=None, legend="auto", ax=None, **kwargs)"""
# //////////////////importing the librarys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# //////importing the dataset of seaborn in seaborn github repositry titanic
titanic_df=sns.load_dataset('titanic')
# print(titanic_df)

# /////hue---we use hue parameter when we canalysis different colums
# ////style----we use style function when we wish to different shapes of the graph elements
# ////size-----we use size function when we want to spacific the size of the graph elements
# ////sizes-----if we want to sizes all of the elements manually than we use sizes parameter
# ////palette-------if we want to change the variables color than we use palette function
# ////hue_order-----if we want to order the hue than we use hue_order function
# ////hue_norm-----if we want to normalize the hue than we use hue_norm function
# ////x_bins------if we want to create indexes on the x_axis than we use x_bins parameter
# ////y_bins-----if we want to creat indexes on the y-axis along than we use y_bins function
# ////estimator-----if we want to creating the graph as the estimate function when we use estimate parameter and we find it in the numpy library
# ////alpha-----if we want to change opacity than we use alpha function

# ////lagend---if we want to use labels than we use legend parameter

# ////**kwargs----if we want to many vlaues as a dictionry than we use **kwargs parameter
# sns.scatterplot(x='age',y='fare',data=titanic_df, hue='sex',style='who',palette='hot',alpha=0.9)
# ////using other values as a x and y labels
# sns.scatterplot(x='who',y='fare',data=titanic_df, hue='alive',style='alive',palette='hot',alpha=0.9)

# /////this is the very intersting scatterplote graph
sns.scatterplot(x='age',y='fare',data=titanic_df)
sns.lineplot(x='age',y='fare',data=titanic_df)
sns.barplot(x='age',y='fare',data=titanic_df)

plt.show()
