# (1)***********************Seaborn introduction****************************************
# https://seaborn.pydata.org/

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kde, norm
from inspect import Parameter
from sklearn.datasets import load_breast_cancer
import sklearn as skl




# (02)******************line plot using seaborn****************

days=np.arange(1,16)
np.random.seed(100)
temp=np.random.randint(30,45, 15)
tem_df=pd.DataFrame({'days':days,'temp':temp})
line_func="""(*, x=None, y=None, hue=None, size=None, style=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, units=None, estimator="mean", ci=95, n_boot=1000, seed=None, sort=True, err_style="band", err_kws=None, legend="auto", ax=None, **kwargs)"""
sns.lineplot(x=days,y='temp',data=tem_df)
plt.show()
tips_df=sns.load_dataset('tips')
print(tips_df)
tips_df=pd.read_csv('C:\\Tayyab Work\\seabornWithTayyab\\datasets\\dataset_tips_seaborn.csv')
tips_df=pd.read_csv('datasets\\dataset_tips_seaborn.csv')
sns.lineplot(x='total_bill',y='tip', data=tips_df)
sns.lineplot(x='tip', y='total_bill', data=tips_df)
sns.lineplot(x='size',y='tip', data=tips_df)
sns.lineplot(x='tip', y='size', data=tips_df)
sns.lineplot(x='size',y='total_bill', data=tips_df)
sns.lineplot(x='total_bill', y='size', data=tips_df)
print(tips_df)
plt.show()


# (03)********************line Plot part 2 Seaborn**************

line_func="""(*, x=None, y=None, hue=None, size=None, style=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, units=None, estimator="mean", ci=95, n_boot=1000, seed=None, sort=True, err_style="band", err_kws=None, legend="auto", ax=None, **kwargs)"""
tips_df=pd.read_csv('datasets\\dataset_tips_seaborn.csv')
plt.figure(figsize=(13,13))
sns.set(style='darkgrid')
plt.grid(color='red')
sns.lineplot(x='size',y='total_bill',data=tips_df,hue='day',size=40 ,style='day',palette='hot',dashes=False,markers=['o','*','<','>'],markersize=14,legend='full')
x, y = np.random.normal(size=(2, 5000)).cumsum(axis=1)
sns.lineplot(x=x, y=y, sort=False, lw=1)
plt.title("Temperature Karachi",fontsize=20)
plt.xlabel("Size",fontsize=15)
plt.ylabel('total_bill',fontsize=15)
plt.show()


# (04)**********************Distplot in seaborn******************

tip_df=sns.load_dataset('tips')
print(tip_df)
parameters="""(a=None, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None, x=None)"""
sns.distplot(tip_df['size'])
sns.distplot(tip_df['tip'])
sns.distplot(tip_df['total_bill'])
bins=np.arange(5,60,12)
sns.distplot(tip_df['total_bill'],bins=bins,hist=False)
sns.distplot(tip_df['total_bill'],bins=bins,kde=False)
sns.distplot(tip_df['total_bill'],rug=True)
sns.distplot(tip_df['total_bill'],fit=norm,kde=False,color='r',vertical=False, axlabel='Hacked',label='Total Bill')
plt.legend()
plt.title('Tips Calculated')
plt.xlabel('ToTal BILL')
plt.ylabel('labels')
plt.show()


# (05)*****************distplot part 02 in seaborn***************************

tip_df=sns.load_dataset('tips')
print(tip_df)
parameters="""(a=None, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None, x=None)"""
bins=[5,10,15,20,25,30,35,40,45,50,55]
plt.figure(figsize=(13,13))
sns.set() 
sns.distplot(tip_df['total_bill'],bins=bins)
plt.xticks(bins)
sns.distplot(tip_df['total_bill'],bins=bins,hist_kws={'color':'red','edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.9})
sns.distplot(tip_df['total_bill'],bins=bins,hist_kws={'color':'red','edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.9}, kde_kws={'color':'g','linewidth':5,'linestyle':'--','alpha':0.9})
sns.distplot(tip_df['total_bill'],bins=bins,hist_kws={'color':'red','edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.9}, kde_kws={'color':'g','linewidth':5,'linestyle':'--','alpha':0.9},rug=True,rug_kws={'color':'k','linewidth':3,'linestyle':'--','alpha':0.9})
sns.distplot(tip_df['total_bill'],bins=bins,hist_kws={'color':'red','edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.9},kde=False,rug=True,rug_kws={'color':'k','linewidth':3,'linestyle':'--','alpha':0.9},fit=norm,fit_kws={'color':'m','linewidth':3,'linestyle':'--','alpha':0.9},label="Sultan's Production")
sns.distplot(tip_df['size'],label='size')
sns.distplot(tip_df['tip'],label='tip')
sns.distplot(tip_df['total_bill'],label='Total_Bill')
plt.title("Histogram of Resturent",fontsize=25)
plt.xlabel("Total Bills",fontsize=15)
plt.ylabel("Indexs",fontsize=15)
plt.legend(loc=2)
print(tip_df.total_bill.sort_values())
plt.show()


# (06)*************************Bar Plot Seaborn in python***************************

tip_df=sns.load_dataset('tips')
print(tip_df)
parameters_of_barplot="""(*, x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=np.mean, ci=95, n_boot=1000, units=None, seed=None, orient=None, color=None, palette=None, saturation=0.75, errcolor=".26", errwidth=None, capsize=None, dodge=True, ax=None, **kwargs)"""
sns.barplot(x=tip_df.day, y=tip_df.total_bill, hue=tip_df.sex)
sns.barplot(x='day',y='total_bill',data=tip_df,hue='sex')
order=['Sun','Thur','Fri','Sat']
hue_order=['Female','Male']
sns.barplot(x='day',y='total_bill',data=tip_df,hue='sex',order=order,hue_order=hue_order,estimator=np.max,ci=50,n_boot=5, orient='v',color='g',palette='hot',saturation=0.3,errcolor='0.3',errwidth=12,capsize=0.5,dodge=True)
plt.show()


# (07)*************************Bar plot part 2 in Seaborn******************

tip_df=sns.load_dataset('tips')
print(tip_df)
parameters_of_barplot="""(*, x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=np.mean, ci=95, n_boot=1000, units=None, seed=None, orient=None, color=None, palette=None, saturation=0.75, errcolor=".26", errwidth=None, capsize=None, dodge=True, ax=None, **kwargs)"""
sns.set()
kwargs={'alpha':0.9,'linestyle':'--','linewidth':5,'edgecolor':'r'}
aa=sns.barplot(x='day',y='total_bill',data=tip_df,hue='sex',**kwargs)
aa.set(title='Bar chart of Seaborn',xlabel='Days',ylabel='Total Bills')
plt.show()


# (08)*************************Scatter plot in seaborn*************************

parameters="""(*, x=None, y=None, hue=None, style=None, size=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, markers=True, style_order=None, x_bins=None, y_bins=None, units=None, estimator=None, ci=95, n_boot=1000, alpha=None, x_jitter=None, y_jitter=None, legend="auto", ax=None, **kwargs)"""
titanic_df=sns.load_dataset('titanic')
print(titanic_df)
sns.scatterplot(x='age',y='fare',data=titanic_df, hue='sex',style='who',palette='hot',alpha=0.9)
sns.scatterplot(x='who',y='fare',data=titanic_df, hue='alive',style='alive',palette='hot',alpha=0.9)
sns.scatterplot(x='age',y='fare',data=titanic_df)
sns.lineplot(x='age',y='fare',data=titanic_df)
sns.barplot(x='age',y='fare',data=titanic_df)
plt.show()


# (09)*****************************Heatmap plotting using Seaborn**********************************

Parameters="""(data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt=".2g", annot_kws=None, linewidths=0, linecolor="white", cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels="auto", yticklabels="auto", mask=None, ax=None, **kwargs)"""
df_2d=np.linspace(1,5,12).reshape(4,3)
print(df_2d)
annot_arr=np.array([['a00','a01','a02'],['a10','a11','a12'],['a20','a21','a22'],['a30','a31','a32']])
print(annot_arr)
global_warming=pd.read_csv('datasets\\Who_is_responsible_for_global_warming.csv')
print(global_warming)
gw=global_warming.drop(columns=['Country Code','Indicator Name','Indicator Code'],axis=1).set_index('Country Name')
cmap_colors="""'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 
'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 
'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 
'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'"""
sns.heatmap(gw,cmap='coolwarm',vmin=0,vmax=25,alpha=1,annot=True)
sns.heatmap(df_2d,annot=annot_arr,fmt='s')
plt.show()


# (10)**********************Heatmap part 2 using Seaborn**********************

Parameters="""(data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt=".2g", annot_kws=None, linewidths=0, linecolor="white", cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels="auto", yticklabels="auto", mask=None, ax=None, **kwargs)"""
df_2d=np.linspace(1,5,12).reshape(4,3)
print(df_2d) 
annot_arr=np.array([['a00','a01','a02'],['a10','a11','a12'],['a20','a21','a22'],['a30','a31','a32']])
print(annot_arr)
global_warming=pd.read_csv('datasets\\Who_is_responsible_for_global_warming.csv')
print(global_warming)
gw=global_warming.drop(columns=['Country Code','Indicator Name','Indicator Code'],axis=1).set_index('Country Name')
plt.figure(figsize=(13,13))
annot_kws={'fontsize':15,'fontstyle':'italic','color':'g','alpha':0.6,'rotation':'vertical','verticalalignment':'center','backgroundcolor':'w'}
sns.heatmap(df_2d,annot=annot_arr, fmt='s',annot_kws=annot_kws)
sns.heatmap(df_2d,annot=annot_arr, fmt='s',annot_kws=annot_kws,linewidths=4,linecolor='k',cbar=False,xticklabels=False,yticklabels=False, )
cbar_kws={'orientation':'horizontal','shrink':1,'extend':'min','ticks':np.arange(1,22),'drawedges':True}
sns.heatmap(df_2d,annot=annot_arr, fmt='s',annot_kws=annot_kws,linewidths=4,linecolor='k',cbar_kws=cbar_kws )
ax=sns.heatmap(gw,cmap='hot',annot_kws=annot_kws,linewidths=4,linecolor='k',cbar_kws=cbar_kws,xticklabels=np.arange(1,16))
ax.set(title="HeatMap using Seaborn",
        xlabel="years",
        ylabel='Countrys')
sns.set(font_scale=4)
plt.show()


# (11)**********************Correlation Heatmap part 3 in seaboern*******************

gw=pd.read_csv('datasets\\Who_is_responsible_for_global_warming.csv')
gw1=gw.drop(columns=['Country Code','Indicator Name','Indicator Code'],axis=1).set_index('Country Name')
print(gw1)
print(gw1.corr())
gw2=gw1.corr()
plt.figure(figsize=(13,13))
ax=sns.heatmap(gw2,cmap='coolwarm',annot=True,linewidths=3)
ax=sns.heatmap(gw2,annot=True,linewidths=3)
ax.tick_params(size=10,color='y',labelsize=10,labelcolor='y')
plt.title("Heatmap of correlation",fontsize=25)
plt.xlabel('Years')
plt.ylabel('years')
plt.show()

from sklearn.datasets import load_breast_cancer
cancer_dataset=load_breast_cancer()
print(cancer_dataset)
cancer_df=pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],columns=np.append(cancer_dataset['feature_names'],['target']))
print(cancer_df)
plt.figure(figsize=(13,13))
ax=sns.heatmap(cancer_df.corr(),annot=True,linewidths=3)
ax.tick_params(size=10,color='k',labelsize=10,labelcolor='k')
plt.title("Correlation Heatmap of 'Breast Cancer Patients'",fontsize=25)
plt.show()


# (12)****************************Pairplot Seaborn**************************

parameters="""(data, *, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind="scatter", diag_kind="auto", markers=None, height=2.5, aspect=1, corner=False, dropna=False, plot_kws=None, diag_kws=None, grid_kws=None, size=None)"""
from sklearn.datasets import load_breast_cancer
cancer_dataset=load_breast_cancer()
print(cancer_dataset)
cancer_df=pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],columns=np.append(cancer_dataset['feature_names'],['target']))
print(cancer_df)
ax=sns.pairplot(cancer_df,vars=['mean smoothness', 'mean compactness', 'mean concavity','mean concave points', 'mean symmetry'],hue='target',palette='hot')
ax=sns.pairplot(cancer_df,hue='target',palette='hot',x_vars=['mean radius','mean texture'],y_vars=['mean radius','mean texture'],kind='reg',diag_kind='hist',markers=['*','<'])
plt.show()




















