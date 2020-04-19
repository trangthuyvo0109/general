# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:14:04 2019

@author: tvo
"""
'''
Creating boxplot and violin(or beanplot) plotting fuction to test the distribution of data. This is an important test prior to more 
advanced statistical steps

Note that although violin plots are closely related to Tukey's (1977)
box plots, they add useful information such as the distribution of the
sample data (density trace).

By default, box plots show data points outside 1.5 * the inter-quartile
range as outliers above or below the whiskers whereas violin plots show
the whole range of the data.

The testing data is the fraction of land cover map e.g. tree, shrub, roofs,....
'''




from read_csv import read_csv_file




def violin_plot(file):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    ##Read data as dataframe pandas
    data = pd.read_csv(file)
    pd.set_option('display.max_columns', None)
    
    
    ##Define values of xticks 
    data_keys = ['Tree Canopy', 'Grass\\Shrubs', 'Bare Soil','Water'
                ,'Buildings', 'Roads', 'Other Impervious', 'Railroads',
                'Water (Ocean)']
    
    
    ###Dictionarz of font style
    font = {'family':'DejaVu Sans',
            'style':'oblique',
            'size':25
            }

    
    fig, ax = plt.subplots()
    ax = sns.violinplot(x=data.Class, y=data.SUM_fracti,
                        showmeans=True, showmedians=True,
                        palette="Set3")
    sns.set(style="whitegrid")
    ax.tick_params(axis='x', labelsize=20, rotation=15)
    ax.tick_params(axis='y', labelsize=25, rotation=25)
    ax.set_ylabel('Fraction of land cover per pixel unit',size=25,
                  fontdict=font, weight='bold')
    ax.set_xlabel('Land Cover Class', size=25,
                  fontdict=font, weight='bold')
    
    fig.savefig('boxplot_fraction_land_cover.png')
    


    
    



    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

