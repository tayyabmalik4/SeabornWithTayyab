# (10)**********************Heatmap part 2 using Seaborn**********************


Parameters="""(data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt=".2g", annot_kws=None, linewidths=0, linecolor="white", cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels="auto", yticklabels="auto", mask=None, ax=None, **kwargs)"""


# //////////////////importing the librarys
from matplotlib import lines
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ////creating the DataFrame using numpy
df_2d=np.linspace(1,5,12).reshape(4,3)
# print(df_2d) 
annot_arr=np.array([['a00','a01','a02'],['a10','a11','a12'],['a20','a21','a22'],['a30','a31','a32']])
# print(annot_arr)

# ///////Realword Examples
global_warming=pd.read_csv('datasets\\Who_is_responsible_for_global_warming.csv')
# print(global_warming)
# /////drop----------we drop some colums which is string format because heatmap not take string
# ////index_name----------we also give index_name which is according to show our graph
gw=global_warming.drop(columns=['Country Code','Indicator Name','Indicator Code'],axis=1).set_index('Country Name')

# ////if we want to chang ethe size of the figure than we use this parameter
plt.figure(figsize=(13,13))

# ////starting the heatmap using seaborn library
# /////annot_kws--------if we want to change something in thhe annotation like fontsize,style linewidth etc than we use annot_kws(annotation_keywords)
# //////fontsize-----when we want to change the font size of annotation
# /////fontstyle-----when we want to change the style of font 
# ////color-------when we want to change the color of the font of annot
# ////alpha----when we want to change the opacity of the annot
# ////rotation-----when we want to change the rotation of annot
# /////verticalalignment----when we want to chane the alignment od the annot
# /////backgroundcolor-----when we want to change the background color of annot
annot_kws={'fontsize':15,'fontstyle':'italic','color':'g','alpha':0.6,'rotation':'vertical','verticalalignment':'center','backgroundcolor':'w'}
# sns.heatmap(df_2d,annot=annot_arr, fmt='s',annot_kws=annot_kws)

# ////linewidth------when we want to change the linewidth we use this function linewidth
# ////linecolor-----when we wantt to change the line color than we use linecolor parameter
# ////cbar------when we dont show the color bar than we use cbar function
# //////xticklabel,yticklabel-----if we don't show the x and y labels than we use xticklable and yticklabel False
# sns.heatmap(df_2d,annot=annot_arr, fmt='s',annot_kws=annot_kws,linewidths=4,linecolor='k',cbar=False,xticklabels=False,yticklabels=False, )

# /////cbar_kws-----------if we want to use the many parameters as we want to change it in the colorbar than we use cbar_kws
# /////orientation-----if we want to change the orientations of the colorbar we use orientation
# ////shrink------if we want to ling or short the color bars than we use shrink parameter
# ////extend-----when we extend the nock of the  colorbar we use this
# ////ticks------if we want to min or max the color bar than we use ticks parameter
# ////drawedge------if we want to draw the edges than we use drawedge parameter
cbar_kws={'orientation':'horizontal','shrink':1,'extend':'min','ticks':np.arange(1,22),'drawedges':True}
# sns.heatmap(df_2d,annot=annot_arr, fmt='s',annot_kws=annot_kws,linewidths=4,linecolor='k',cbar_kws=cbar_kws )
ax=sns.heatmap(gw,cmap='hot',annot_kws=annot_kws,linewidths=4,linecolor='k',cbar_kws=cbar_kws,xticklabels=np.arange(1,16))
# /////if we want to set the title,xlabel and ylabels than we use variable.set(title=?,xlabel=?,ylabel=?)
# //////sns.set(font_scale=4)we also use font_scale to change the fontsize
ax.set(title="HeatMap using Seaborn",
        xlabel="years",
        ylabel='Countrys')
sns.set(font_scale=4)
plt.show()