# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:14:51 2020

@author: tvo
"""

'''
# Step 0: Running this step first, for each data

# Import code for date
date_code = '2019-09-21T063150UTC/'

# Define path containing LST and Emis files for interested date:
path_sub = 'C:/Users/tvo/Documents/urban_project/Test_NYC/'+date_code

# Save temporary file to a folder 
path_save_pickle_sub = '//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/'+date_code

# Create a new directory if not exists yet
import os
import pandas as pd
try:
    os.mkdir(path_save_pickle_sub)
except:
    pass
    
# Import linearalg_np_test_ver01
from linearalg_np_test_ver01 import * 

# Run function to red dbf files:
df_lst, df_emiswb = dbf_to_df_date(path_sub,cloud_masked=True)

df_lst.to_pickle(path_save_pickle_sub+'df_lst')
df_emiswb.to_pickle(path_save_pickle_sub+'df_emiswb')

# Emis representative 
df_landcover_concat_nyc = pd.read_pickle('//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/df_landcover_concat')
df_emiswb_nyc = pd.read_pickle(path_save_pickle_sub+'df_emiswb')


df_emis_repre = emis_pure_pixel(df_landcover_concat_nyc, df_emiswb_nyc)

df_emis_repre.to_pickle(path_save_pickle_sub+'df_emis_repre')




'''
# This script outline the procedure , not for running
# ***********************************************




# Step 1: Let it run parallelizing on matrix 
# Open the script: linearalg_np_test_ver02_linux

# Adjust the kernel size and moving pixel

# Let it run on matrix

# python linearalg_np_test_ver02_linux.py


# ***********************************************
# Step 2: Reading output files, running read_out_value_list function 


python

print('Conducting Step 2: Reading output files, running read_out_value_list function ')

import linearalg_np_test_ver01
from linearalg_np_test_ver01 import *

# Define list of kernel for testing 
kernel_list = [10]
moving_pixel = [5]


#Define max, min bounds

#date_code = '2019-09-21T063150UTC/'
#bound_min = 280
#bound_max = 297





## Define max, min bounds
date_code = '2019-07-26T222034UTC/'
bound_min = 295
bound_max = 310

# Linux:
path = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/'+date_code

# Window:
# path = '//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/' + date_code


out_value_list = read_out_value_list(path,kernel_list,moving_pixel)


# ***********************************************
# Step 3: Analyzing the output data on matrix, running groupby_outvalue function 

# Define row and column for test site
row = 884
col = 786

path_fig = path+'fig/'


print('Conducting Step 3: Analyzing the output data on matrix, running groupby_outvalue function')

out_value_modi_list = []
for i in range(len(kernel_list)):
    print(kernel_list[i])
    out_value_modi = groupby_outvalue(out_value_list[i],row,col,kernel=kernel_list[i],path_fig = path_fig)
    out_value_modi_list.append(out_value_modi)



# ***********************************************
# Step 4: Mapping canopy temperature, running canopy_temp function 
import os
import numpy as np
print('Conducting Step 4: Mapping canopy temperature, running canopy_temp function ')

path_fig = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/'+date_code+'fig/'

#class_code = ['canopy','road']
#frac_index = [0,5]

class_code = ['Tree Canopy','Grass_Shrubs','Bare Soil',
                    'Water', 'Building','Road','Other Impervious',
                    'Railways']
frac_index = np.arange(0,8,1)

out_value_modi_member_list = []
for j in range(len(class_code)):
    
    try:
        os.mkdir(path_fig)
    except:
        pass
    
    # Create new folder to store canopy temperature data
    try:
        os.mkdir(path+class_code[j]+'/')
    except:
        pass
    
    # out_value_modi_member_list = []
    for i in range(len(kernel_list)):    
       
        try:
            os.mkdir(path_fig)
            
        except:
            pass
        out_value_modi_list[0] = out_value_modi_list[0].astype(np.float64)
        out_value_modi_member = canopy_temp(out_value_modi_list[0],tick_min=bound_min,tick_max=bound_max,
        tick_interval=0.1,bin_range=0.05,frac_index=frac_index[j],frac_name=class_code[j],kernel=kernel_list[i],moving=moving_pixel[i],
        date=date_code.split('/')[0],path_fig = path_fig,canopy_temp=False,only_end=False)
        
        # Export to pickple
        out_value_modi_member.to_pickle(path+class_code[j]+'/canopy_kernel_'+str(kernel_list[i])+'_moving_'+str(moving_pixel[i]))
            
    
    out_value_modi_member_list.append(out_value_modi_member)
    
   
     

for i in range(len(out_value_modi_member_list)):
    out_value_modi_member_list[i]['class'] = str(class_code[i]+' Temperature')






# ***********************************************
# Step 6: Plotting correlation between fraction of canopy and LST, running cut_value_frac_range function   
import numpy as np
import pandas as pd

out_value_modi_member_cut_list = []
for j in range(len(out_value_modi_member_list)): 
    for i in range(len(kernel_list)):
         
        out_value_modi_cut = cut_value_frac_range(out_value_modi_member_list[j],frac_name=class_code[j],color="green",
            fraction_range = np.arange(0,1.1,0.05),kernel=kernel_list[i],moving=moving_pixel[i],path_fig=path_fig)
    
        out_value_modi_member_cut_list.append(out_value_modi_cut)


df_concat = pd.concat(out_value_modi_member_cut_list)

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(30,15))
g = sns.pointplot(x='frac_index',y='out_value',hue='class',data=df_concat)
g.set_xticklabels(np.round(np.arange(0.05,1.05,0.05),3))
plt.tick_params(axis='x',labelsize=20,rotation=25)
plt.tick_params(axis='y',labelsize=20)
plt.xlabel('Fraction of Land Cover Class',size=20,weight='bold')
plt.ylabel('End member Temperature',size=20,weight='bold')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=2,prop={'size':20})
fig.savefig(path_fig+'fraction/'+str(class_code[0])+'_'+str(class_code[-1]))



# ***********************************************
# Step 4. b): Mapping output LST temperature, running extract_out_temp function 
    
out_value_modi_output_list = []
path_fig_LST = path_fig + 'output_LST/'
try:
    os.mkdir(path_fig_LST)
    
except:
    pass

for i in range(len(kernel_list)):
    out_value_modi_final = extract_out_temp(out_value_modi_list[i],kernel=kernel_list[i],tick=True,tick_max = bound_max, tick_min = bound_min,
                                            path_fig=path_fig_LST)
    
    out_value_modi_output_list.append(out_value_modi_final)    
    
    



# ***********************************************
# Step 5: Sensitivity test among different kernel sizes, running test_sensitivity function   

test_sensitivity(out_value_modi_member_list[0],kernel_list=kernel_list,moving_pixel=moving_pixel,class_name='Canopy ',path_fig=path_fig,bounds_min=bound_min,bounds_max=bound_max)


sns.set(style="whitegrid")

test_sensitivity(out_value_modi_member_list[-1],kernel_list=kernel_list,moving_pixel=moving_pixel,class_name='Building ',path_fig=path_fig,bounds_min=bound_min,bounds_max=bound_max)







# ***********************************************
# Step 7: (Optional) Comparing different kernel and moving    
    
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


color = ['red','green','blue','pink','yellow','black','cyan']
kernel_moving = [[kernel,moving] for kernel,moving in zip(kernel_list,moving_pixel)]

for i in range(len(out_value_modi_member_cut_list)):
    
    out_value_modi_member_cut_list[i].insert(len(out_value_modi_member_cut_list[i].columns),column='kernel_moving',
                                  value=[str(kernel_moving[i])]*len(out_value_modi_member_cut_list[i]))
    
    

fig, ax = plt.subplots()
sns.pointplot(x="frac_index",y="out_value",hue='kernel_moving',data=pd.concat(out_value_modi_member_cut_list))
plt.tick_params(axis='x',rotation=45)

try:
    os.mkdir(path_fig+'fraction/')
    
except:
    pass

fig.savefig(path_fig+'fraction/'+'fraction_merge'+str(kernel_moving))


# Creating residual map btw 2 dataframe df_1 and df_2
import pandas as pd
import os

path = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/'
path_resi = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/residuals/'
date_1 = '2019-07-26T222034UTC/'
date_2 = '2019-09-21T063150UTC/'

df_1 = pd.read_pickle(path+date_1+'canopy/'+os.listdir(path+date_1+'canopy/')[0])
df_2 = pd.read_pickle(path+date_2+'canopy/'+os.listdir(path+date_2+'canopy/')[0])

title_1 = 'Canopy Temperature on '+date_1.split('/')[0]
title_2 = 'Canopy Temperature on '+date_2.split('/')[0]


residual_map(df_1, df_2, tick_min=-10,
        tick_max = 20, tick_interval=0.1, title_1 = title_1,
        title_2 = title_2, date= date_code[:-1], path_fig=path_resi)


residual_map(df_1, df_2, tick_min=-3,
        tick_max = 3, tick_interval=0.1, title_1 = title_1,
        title_2 = title_2, date= date_code[:-1], path_fig=path_fig)






# Export all ECOSTRESS images to images file, to easily detect the potential image


folder = '/nas/rhome/tvo/py_code/aes509/data/ECOSTRESS/'
import rasterio
import cmocean  
import os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma



for file in os.listdir(folder):
    try:
        if 'SDS_LST_doy' in file:
            src = rasterio.open(folder+file)
            array = src.read(1)     
            # Define levels of ticks and tick intervals
            array_masked = ma.masked_where(array == 0, array)
            tick_min=np.nanmin(array_masked*0.02)
            tick_max=np.nanmax(array_masked*0.02)
            tick_interval = (tick_max - tick_min)/100
            levels = MaxNLocator(nbins=(tick_max - tick_min)/tick_interval).tick_values(tick_min, tick_max)
            cmap = plt.get_cmap(cmocean.cm.thermal)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            
            
            fig, ax = plt.subplots()
            map = plt.imshow(array_masked*0.02,cmap=cmap,norm=norm)
            fig.colorbar(map)
            fig.savefig(folder+'/images/'+file)
            
            src.close()
            
        else:
            pass
        
    except:
        print(file)
         





'''