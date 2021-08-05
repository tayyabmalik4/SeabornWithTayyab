# (09)*****************************Heatmap plotting using Seaborn**********************************
# //////heatmap is a graphically represented of 2d array in colored formate


# ////these are the parameters of heatmat which we use in the library
from inspect import Parameter


Parameters="""(data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt=".2g", annot_kws=None, linewidths=0, linecolor="white", cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels="auto", yticklabels="auto", mask=None, ax=None, **kwargs)"""


# //////////////////importing the librarys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ////creating the DataFrame using numpy
df_2d=np.linspace(1,5,12).reshape(4,3)
# print(df_2d)
annot_arr=np.array([['a00','a01','a02'],['a10','a11','a12'],['a20','a21','a22'],['a30','a31','a32']])
print(annot_arr)


# /////realworld DataFrame
global_warming=pd.read_csv('datasets\\Who_is_responsible_for_global_warming.csv')
# print(global_warming)
# /////drop----------we drop some colums which is string format because heatmap not take string
# ////index_name----------we also give index_name which is according to show our graph
gw=global_warming.drop(columns=['Country Code','Indicator Name','Indicator Code'],axis=1).set_index('Country Name')

# ////////creating heatmap using seaborn library
# ////cmap------if we want to change the color of the graph than we use cmap
# ////parameters-------we use these colors in the cmap parameter
cmap_colors="""'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 
'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 
'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 
'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'"""
# ////vmax,vmin--------if we want to change minmum or maximum value than we use vmax or vmin
# /////annot-------if we want to show the values of all the colums than we use annot(annotation)
# sns.heatmap(gw,cmap='coolwarm',vmin=0,vmax=25,alpha=1,annot=True)

# ////if we want to show the annotation which we wish than we use fmt='format_like_string' and also annot=annotation_name which we created
sns.heatmap(df_2d,annot=annot_arr,fmt='s')
plt.show()