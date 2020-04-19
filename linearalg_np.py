# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:29:46 2019

@author: tvo
"""
'''
This a a script for solving linear regression equation of multiple variables
For the purpose of solving equations of Land Surface Temperature with known variables of fractions of land cover
The results is the Temperature of land cover properties (e.g. Temp(tree), Tem(roof), ....)


should be in this format: 
    Unknown variables: a,b,c 
    Known variables: coefficient or fractions (e.g. 0.1,0.9) and idenpendent values (e.g. 305, 307) or Land Surface Temperature
        
    0.1 * a + 0.9 * b + 0.4 * c = 305
    0.4 * a + 0.5 * b + 0.7 * c = 307
    0.7 * a + 0.9 * b + 0.4 * c = 302
    
'''

def dbf_to_df(path,bad_value):
    
    '''
    Function to read the .vat.dbf files containes Raster Attribute Table 
    and convert to pandas Dataframe for easier data manipulating. 
    
    Input: 
        path: containing target .vat.dbf files
        bad_value: the lim value for ignoring calculation due to resampling errors of zonal statistics in ArcGIS
        
        
        
    Output:
        df_landcover_concat: pandas Dataframe containing all dataframes of each land cover class 
        df_lst: pandas Dataframe containing ECOSTRESS LST value for the domain.
        
    Example:
        # Define path containing interested files 
        path = 'C:/Users/tvo/Documents/urban_project/Test_Staten/test/'
        
        # Read .vat.dbf files and convert to pandas Dataframe
        df_landcover_concat, df_lst, df_emiswb = dbf_to_df(path,bad_value=4500)
        
    
    '''
    import pandas as pd
    from dbfread import DBF
    from pandas import DataFrame
    import os
    import numpy as np

    # Define empty list contains list of fiels in directory
    listdir = []
    
    # Only search for files with extension .vat.dbf and append to listdir
    for file in os.listdir(path):
        if file.endswith('.tif.vat.dbf'):
            listdir.append(file)
    
    # Read .vat.dbf files:
    dbf_dataset = []
    for file in listdir:
        dbf = DBF(path + file)
        dbf_dataset.append(dbf)
        
    # Define a dictionary to hold the class key value for each land cover class
    land_cover_class = {'canopy':1,
                            'grass':2,
                            'bsoil':3,
                            'water':4,
                            'building':5,
                            'road':6,
                            'impervious':7,
                            'railway':8}
    
        
    # Convert .dbf to pandas DataFrame:
    df_landcover = []
    for item in dbf_dataset: # Loop over dbf_dataset
        for key in land_cover_class.keys(): # Loop over dictionary of land cover class
            if key in str(item): # Search if any key matches the dbf_dataset file 
                
                try:
                    df = DataFrame(item) # Covert dbf to pandas DataFrame
                    print("succesfull"+str(item))
                except:
                    print("fail"+str(item))
                    pass
                
                # Trying to insert the "sum" and "fraction" columns if not exit yet:
                try:
                    # sum = VALUE_0 + VALUE_12
                    df.insert(len(df.columns),column="sum",value=df["VALUE_0"] + df["VALUE_12"])
                    
                    # fraction = VALUE_12 / sum, checking if sum = 0 --> ignore
                    df.insert(len(df.columns),column="fraction",value=0)
                    df["fraction"] = df["VALUE_12"]/df["sum"] 

                    
                # If already exits, ignore
                except:
                    pass
                
                # Insert Class code column by corresponding class code 
                df.insert(len(df.columns),column='Class',value=land_cover_class[key]) 
                print(df[:5])
                print(key)

                
                
                
                
                df_landcover.append(df) # Append to empty list of dataframe
                
    # Concating or Merging all dataframe into one dataframe
    df_landcover_concat = pd.concat(df_landcover, axis=0, sort=False)    

    # Creating a separate dataFrame for LST
    for item in dbf_dataset:
        if 'LST' in str(item):         
            df_lst = DataFrame(item)
            
    # Creating a separate dataframe for Emis2
    for item in dbf_dataset:
        if "emiswb" in str(item):
            df_emiswb = DataFrame(item)
    
    # Replacing non value e.g. 0 by -9999 and later ignore for calculation:
    # Using column "Count_1" to idenity -9999 values
    try:
        df_lst.loc[df_lst["COUNT_1"] == 0,"COUNT_1"] = -9999
    
        df_emiswb.loc[df_emiswb["COUNT_1"] == 0, "COUNT_1"]= -9999
    except:
        pass
    
    
    # Set a lim to ignore the bad values due to resampling errors 
    # in zonal statistics e.g. 4500 m2 is the total area of a good pixel
    # which can be use for calculation
    # Perhaps for a small test size, we can reduce this number to 3500 e.g. 
    # Using "fraction" as an idenity for nan values, for it could be feasible for the following function
    df_landcover_concat.loc[df_landcover_concat["sum"] < bad_value,"fraction"] = -9999
    
    # Check if number of pixels for each land cover class equals to each other
    len_list = []
    for i in range(8):
        
        length = len(df_landcover_concat.groupby("Class").get_group(i + 1))
        len_list.append(length)
        
    if all(x == len_list[0] for x in len_list):
        print("ALL number of pixels for each land cover equal! The data is now ready to go! ")
    else:
        print("number of pixels for each land cover NOT equal!!!!! Check the data please !")
        
        
        
        
            
    return df_landcover_concat, df_lst, df_emiswb

def emis_pure_pixel(df_landcover, df_emiswb):
    '''
    Function to plot the distribution of Emissitvty for whole NYC for pure pixels (fraction >= 90%)
    **** Note: this function is only applied for whole NYC, DO NOT TEST with small domain *****
    Input:
        df_landcover: original df of all land cover class
        df_emiswb: original df of emis ECOSTRESS
        
    Output:
        
        df_emis_repre: the representative Emis values for each land cover class
    
    Example:
        path_nyc = 'C:/Users/tvo/Documents/urban_project/Test_NYC/'
        df_landcover_concat_nyc, df_lst_nyc, df_emiswb_nyc = dbf_to_df(path_nyc,bad_value=4500)
        df_emis_repre = emis_pure_pixel(df_landcover_concat_nyc, df_emiswb_nyc)
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    
    # Insert a new column into df_landcover with emis value
    try:
        df_landcover.insert(len(df_landcover.columns), column="Emis",value=df_emiswb["MEAN"])
    except:
        pass
    
    # Ignoreing nan values
    df_landcover_nonnan = df_landcover.loc[df_landcover["fraction"] != -9999]
    
    # Insert column "pure_pixel_index"
    df_landcover_nonnan.insert(len(df_landcover_nonnan.columns),column="pure_pixel_index",
                                        value=np.nan)
    

    
    # Groupby value fraction > 0.9 to get pure pixels
    df_landcover_nonnan_pure_list = []
    pure_pixel_indexes = np.arange(0.1,1,0.2)
    fig, ax = plt.subplots()
    for index in pure_pixel_indexes:
        df_landcover_nonnan_pure = df_landcover_nonnan.loc[df_landcover_nonnan["fraction"]>index]
        df_landcover_nonnan_pure["pure_pixel_index"] = index
        df_landcover_nonnan_pure_list.append(df_landcover_nonnan_pure)
        
        num_ob_list = []
        for i in range(9):
            print(index)
            num_ob = len(df_landcover_nonnan_pure.groupby("Class").get_group(i+1))
            num_ob_list.append(num_ob)
        ax.plot(num_ob_list,label="Fraction greater than "+str(index)[:3])
        ax.set_title("Testing sensitivity of fraction for each land cover class",size=30)
        ax.set_xticklabels(['Tree Canopy','Grass/Shrubs','Bare Soil',
                        'Water', 'Building','Road','Other Impervious',
                        'Railways'],size=30,rotation=25)             
        ax.set_xlabel('Land Cover Class',size=30,weight='bold')
        plt.tick_params(axis='y',labelsize=25)
        plt.tick_params(axis="x", labelsize=25)
        ax.set_ylabel('Number of observations',size=30,weight='bold')
        plt.legend(prop={"size":30})
    
    

    df_landcover_nonnan_pure_list_c = pd.concat(df_landcover_nonnan_pure_list)
    # Plotting distribution of "Emis" for each land cover class
    # Selecting certain indexes e.g. fraction greater than 0.7 and 0.9 to compare
    pure_index_1 = pure_pixel_indexes[2]
    pure_index_2 = pure_pixel_indexes[-1]
    df_violion = df_landcover_nonnan_pure_list_c.loc[(df_landcover_nonnan_pure_list_c["pure_pixel_index"] == pure_index_2)|
            (df_landcover_nonnan_pure_list_c["pure_pixel_index"] == pure_index_1)]
    
    
    fig, ax1 = plt.subplots()
    ax1 = sns.violinplot(x="Class",y="Emis",
                         hue="pure_pixel_index",
                         data=df_violion,
                         scale="width",
                         split=True)
    
    
    ax1.set_title("Distribution of Emissivity for Fraction greater than "+str(pure_index_1)[:3]+" and "+str(pure_index_2)[:3]
    ,size=30)
    ax1.set_xticklabels(['Tree Canopy','Grass/Shrubs','Bare Soil',
                        'Water', 'Building','Road','Other Impervious',
                        'Railways'],size=30,rotation=25)             
    ax1.set_xlabel('Land Cover Class',size=30,weight='bold')
    plt.tick_params(axis='y',labelsize=25)
    plt.tick_params(axis="x", labelsize=25)
    ax1.set_ylabel('Emissivity',size=30,weight='bold')
    plt.legend(prop={"size":30})
    
    
    # Calculate the representative Emis for each class by take the mean value of pure pixels (fraction > 90%)
    df_emis_repre = df_violion.loc[df_violion["pure_pixel_index"] >= 0.9].groupby("Class")["Emis"].mean().reset_index()
    
    return df_emis_repre
    
def remove_river_distance(df_landcover_concat):
    '''
    Function to look up the pixels 500 m from the river bodies
    
    Example: 
        df_landcover_concat_river = remove_river_distance(df_landcover_concat)
    '''
    path_river_indices = 'C:/Users/tvo/Documents/urban_project/Test_Staten/river_500m_buffer_indices.txt'

    
    import pandas as pd
    
    df_river = pd.read_csv(path_river_indices)
    
    # Replace values of these indexes with nan values, to ignore for calculation 
    df_landcover_concat.loc[df_landcover_concat["Value"].isin(df_river["Value"].tolist()),'fraction']=-9999
    
    df_landcover_concat_river = df_landcover_concat
    
    return df_landcover_concat_river
    
    
    

def cal_linear_alg_lstsq(df_landcover, df_lst, df_emiswb, df_emis_repre, row, col, kernel_list,  _type, radiance=False,bounds=None, 
                         moving_pixel=4,):
    '''
    This function :
        - convert to fraction map for each land cover 
    There will be 8 matrix corresponsing to 8 land cover class 
        - calculate using linear algrabra 
        
    Input:
        df_landcover: pandas Dataframe holds fraction values of each land cover class
        df_lst: pandas Dataframe holds ECOSTRESS LST values per each pixel
        df_emiswb: pandas Dataframe holds ECOSTRESS EmisWB values per each pixel
        df_emis_repre: the representative Emis values for each land cover class
        row: number of row of the whole image
        col: number of col of the whole image
        kerner_list: list of kernel window size (e.g. [10,20,30,40])
        bounds: upper and lower bound values for constaint optimization, optional, default is None
            which means no bounds
        type: measured using "radiance" or "emissivity" functions
        radiance: define whether calculating using "radiance" or not, by default is False which means
            calculating using "emissivity" function
        
    Output:
        out_value: pandas Dataframe contains the output of Linear Algebra for each pixel
            columns=['value','nrow','ncol','indep_value','out_value']:
                'value': list of fraction of land cover properties, extracting from coeff_matrix_df
                'nrow','ncol': indexes of row and col
                'indep_value': ECOSTRESS LST and Emissivity values, extracting from indep_matrix_df
                'out_value': list of temperture of land cover properties, as results of Linear Algebra 
        
        
    Example:
        # Kernel Test:
        row = 47
        col = 54
        
        # Staten Island
        row = 303
        col = 243
        
        # Staten Island (490 m resolution)
        row = 43
        col = 34
        
        # Testing with different kernel sizes
        # Radiance function
        out_value_list = cal_linear_alg_lstsq(df_landcover_concat, df_lst, df_emiswb, df_emis_repre,  
            row, col, kernel_list=[10], _type="radiance", 
            bounds=(290**4,310**4), moving_pixel = 4,  radiance=True)
        
        
        # Emissivity function
        out_value_list = cal_linear_alg_lstsq(df_landcover_concat, df_lst, df_emiswb, df_emis_repre,
            row, col, kernel_list=[25], _type="Emis", 
            bounds=(290**4,310**4), moving_pixel = 5,  radiance=False)
        

    
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy.linalg as la
    import pandas as pd
    from scipy import linalg
    from scipy.optimize import lsq_linear
    

    
    # Create an empy 3D array-like e.g. numpy array or list which holds
    # 8 fraction map
    fraction_map = np.empty((8,row,col),dtype=np.float64) 
    indices_map = np.empty((8,row,col),dtype=np.float64) 
    
    
    # Groupping the dataframe by Class, thus results in 8 classes
    df_grp = df_landcover.groupby('Class')
    
    # Looping over 8 land cover class
    for i in range(9): 
        # Pass the i=0 as the class code starts from 1
        if i == 0:
            pass
        else:
            fraction_map[i-1] = df_grp.get_group(i).fraction.values.reshape(row,col)
            indices_map[i-1] = df_grp.get_group(i).Value.values.reshape(row,col)
            
            
            


    
    # Reading df_lst contains LST value for each pixel and assign as independent value
    indepent_matrix = df_lst.MEAN.values.reshape(row,col)
    emis_matrix = df_emiswb.MEAN.values.reshape(row,col)
    

                    
    '''
    Trying with a new approach
    '''
    # Create an empty pandas Dataframe with columns = 'value','nrow','ncol'
    # with the purpose of indexing each pixel with old row and column indexes
    coeff_matrix_df = pd.DataFrame(data=[], columns=['index','value_fraction','nrow','ncol','indep_value','value_emis','value_emis_sum','out_value'])
    indep_matrix_df = pd.DataFrame(data=[], columns=['index','value_lst','value_emis_sum','nrow','ncol'])
    
    
    
    # Looping over the whole domain to assign new coeff_matrix and independt value dataframes
    for nrow in range(row):
        for ncol in range(col):        
            # Ingnoring NoData value
            if fraction_map[:,nrow,ncol].mean() == -9999:
                pass
            else:
                
                
                
                coeff_matrix_df = coeff_matrix_df.append({'index':indices_map[:,nrow,ncol],
                                                          'value_fraction':fraction_map[:,nrow,ncol], # value of fraction fj
                                                          'nrow':nrow,
                                                          'ncol':ncol,
                                                          'indep_value':indepent_matrix[nrow,ncol],
                                                          'value_emis':list(df_emis_repre["Emis"].values),
                                                          'value_emis_sum':emis_matrix[nrow,ncol]},ignore_index=True) # value of emiss ei
        
                indep_matrix_df = indep_matrix_df.append({'index':indices_map[:,nrow,ncol],
                                                          'value_lst':indepent_matrix[nrow,ncol],
                                                          'value_emis_sum':emis_matrix[nrow,ncol],
                                                          'nrow':nrow,
                                                          'ncol':ncol},ignore_index=True)
            print(nrow,ncol)

        
        
        
    coeff_df = []
    indep_df = []
    out_value = pd.DataFrame(data=[], columns=['index','value','nrow','ncol','indep_value','out_value',"type"])
    out_value_list = []

    # Testing with jumping kernel windows, e.g. it is not neccessary to moving every 1 pixel but instead of moving every 
    # 4 pixels. Doing so would speed up much time calculation especially when we consider the whole NYC domain.   
    
    for kernel in kernel_list: # Looping over the kernel list for testing on different kernel size
        nrow = -moving_pixel # Starting from nrow index -movingpixe, so it will make up with moving 4 pixels
        count = 0

        while nrow < row:
            nrow = nrow + moving_pixel

            
            ncol = -moving_pixel # Starting from nrow index -movingpixel
            while ncol < col:
                ncol = ncol + moving_pixel
                
                
                # Applying linear algebra function for each kernel window:
                    
                # Extracting coeff_matrix values for each kernel window and assign it to a new dataframe
                coeff_df = coeff_matrix_df.loc[(coeff_matrix_df['nrow'] >= nrow) & 
                (coeff_matrix_df['nrow'] < nrow + kernel) & 
                (coeff_matrix_df['ncol'] >= ncol) & 
                (coeff_matrix_df['ncol'] < ncol + kernel)]
                
                # Extracting independent values for each kernel window and assign it to a new dataframe
                indep_df = indep_matrix_df.loc[(indep_matrix_df['nrow'] >= nrow) & 
                (indep_matrix_df['nrow'] < nrow + kernel) & 
                (indep_matrix_df['ncol'] >= ncol) & 
                (indep_matrix_df['ncol'] < ncol + kernel)]
                
                # Ignoring kernel windows does not have the same size with kernel*kernel
                # It could happend when moving window close to the edge
                if len(coeff_df) < 9: # As we consider 8 elements of land cover class 
                    pass
                else:
                    
                    # Applying with radience formula instead of direct LST function: 
                    # LST^4 = sum(fraction*Temp^4) + residuals
                    if radiance is True: 
                        # Independent values
                        indep_df_list = list(map(lambda x:pow(x,4),indep_df["value_lst"].tolist()))
                        
                        # Coefficient matrix values 
                        coeff_df["value"] = coeff_df["value_fraction"]
                    
                    # Applying with Emis formula instead of direct LST function:emis_sum * LST^4 = sum(e*fraction*Temp^4) + residuals
                    else:

                        # LST^4:
                        lst4 = list(map(lambda x:pow(x,4),indep_df["value_lst"].tolist()))
                        # emissivity:
                        emis_sum = indep_df["value_emis_sum"].tolist()
                        # Element-wise multiplication 
                        indep_df_list = [a * b for a, b in zip(lst4,emis_sum)]

                        # fraction i * emis i 
                        coeff_df["value"] = coeff_df["value_fraction"] * coeff_df["value_emis"]


                    
                    # Applying function:
                    x, sum_res, rank, s = la.lstsq(coeff_df['value'].tolist(),
                                           indep_df_list)
                    
                    # Applying optimze function: Testing with Scipy package 
                    if bounds is not None:
                        res = lsq_linear(coeff_df['value'].tolist(),
                                           indep_df_list, bounds=bounds)
                    else:
                        res = lsq_linear(coeff_df['value'].tolist(),
                                         indep_df_list)
                        

                    
                    # Multiplication of a list 
                    # x_df = [ x for i in range(kernel*kernel)]
                    # x_df = [ x for i in range(len(coeff_df))]
                    x_df = [ res.x for i in range(len(coeff_df))]
                    
                    
                    
                    # Adding values of x to column 'out_value':
                    if radiance is True:
                        # Solution: x = 4sqrt(x)
                        coeff_df["out_value"] = list(map(lambda x:x**(1/4),x_df))
                        
                    else:
                        # Solution: x = 4sqrt(x) / emis_sum
                        out_x = list(map(lambda x:x**(1/4),x_df))
                        #coeff_df["out_value"] = [c / d for c,d in zip(out_x,emis_sum)]
                        coeff_df['out_value'] = list(map(lambda x:x**(1/4),x_df))
                        

                        
                    
                    
                    # Adding optimality value
                    coeff_df["optimality"] = [ res.optimality for i in range(len(coeff_df))]
                    
                    # Adding nit value
                    # Number of iterations. Zero if the unconstrained solution is optimal.
                    coeff_df["nit"] = [ res.nit for i in range(len(coeff_df))]
                    
                    
                    # Adding count column contains the order of kernel window
                    coeff_df["count"] = count + 1
                    # print(coeff_df)
                    
                    # Adding type colum such as radiance or temperature
                    coeff_df["type"] = _type
 
                    
                    # Append new dataframe to the existing dataframe
                    out_value = out_value.append(coeff_df)
                    
                    
                count = count + 1
                print(count)
            print(nrow,ncol,kernel)
        out_value_list.append(out_value)
            

    
    return out_value_list
    


def groupby_outvalue(out_value,row,col,kernel):
    '''
    Function to groupby (nrow, ncol) and aggregate by mean value of out_value temperature
    
    Input:
        out_value: pandas Dataframe contains results of def cal_linear_alg_lstsq
        row: number of rows
        col: number of cols
        kernel: kernel window size
        
    Output:
        out_value_modi: modified pandas Dataframe contains:
            'nrow': index of row
            'ncol': index of col
            'indep_value': ECOSTRESS LST value
            'out_value': temperature of land cover properties extracted by Linear Algebra
            'class': order of land cover class 
            'fraction': fraction of coressponding land cover class
            'value_emis_sum': emissivity of corresponding pixel
    
    Example:
        # For Emis funct:
        out_value_modi_list = []
        for i in range(len(kernel_list)):
            print(kernel_list[i])
            out_value_modi = groupby_outvalue(out_value_list[i],row,col,kernel=kernel_list[i])
            out_value_modi_list.append(out_value_modi)
            
        out_value_modi_emis = out_value_modi_list[0]
        out_value_modi_emis["type"] = "emissivity"

            
        
        
        
        
        # For Radiance funct:
        out_value_modi_list = []
        kernel_list = [9]
        for i in range(len(kernel_list)):
            print(kernel_list[i])
            out_value_modi = groupby_outvalue(out_value_list[i],row,col,kernel=kernel_list[i])
            out_value_modi_list.append(out_value_modi)
        
        
        out_value_modi_radi = out_value_modi_list[0]
        
        
        
        out_value_modi_radi.loc[(out_value_modi_radi["class"] == 3.0),"out_value"]=302
        out_value_modi_radi["type"] = "radiance"
        
        out_value_modi_emis.loc[(out_value_modi_emis["class"] == 3.0),"out_value"]=302
        out_value_modi_emis["type"] = "emissivity"
        
        # Paired violinplot to compare "temperature" and "radiance" function:
        out_value_modi_combine = pd.concat([out_value_modi_radi, out_value_modi_emis])
        
        fig, ax = plt.subplots()
        ax = sns.violinplot(x=out_value_modi_combine["class"],y=out_value_modi_combine["out_value"],
        hue=out_value_modi_combine["type"], split=True)
        ax.set_ylim(280,320)     
        ax.set_title('Kernel'+str(kernel_list[0]),size=30)
        ax.set_xticklabels(['Tree Canopy','Grass/Shrubs','Bare Soil',
                        'Water', 'Building','Road','Other Impervious',
                        'Railways'],size=30,rotation=25)             
        ax.set_xlabel('Land Cover Class',size=30,weight='bold')
        plt.tick_params(axis='y',labelsize=25)
        plt.tick_params(axis="x", labelsize=25)
        ax.set_ylabel('Temperature (K)',size=30,weight='bold')
        plt.legend(prop={"size":30})
    
    
    
    
    '''
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    out_value_grp = out_value.groupby(['nrow','ncol']) # Groupying by nrow, ncol 
    
    # Create a new dataframe to hold new groupyed and aggreated by mean values
    out_value_modi = pd.DataFrame(columns=['index','nrow','ncol','indep_value','out_value','class','value_fraction','value_emis_sum'])
    
    # Looping over the grouped dataframe and aggregated by mean of column out_value
    for nrow in range(row):
        for ncol in range(col):
            try:
                for i in range(8):
                    out_value_modi = out_value_modi.append({'index':out_value_grp.get_group((nrow,ncol))['index'].mean()[i],
                                                            'nrow':nrow,
                                                            'ncol':ncol,
                                                            'kernel':kernel,
                                                            'indep_value':out_value_grp.get_group((nrow,ncol))['indep_value'].mean(),
                                                            'out_value':out_value_grp.get_group((nrow,ncol))['out_value'].mean()[i],
                                                            'class':i,
                                                            'value_fraction':out_value_grp.get_group((nrow,ncol))['value_fraction'].mean()[i],
                                                            'value_emis_sum':out_value_grp.get_group((nrow,ncol))['value_emis_sum'].mean()},
                    ignore_index=True)
            except:
                pass
            print(nrow,ncol,"Kernel= ",kernel)
                
    # Plotting boxplot for grouped dataframe
    fig, ax = plt.subplots()
    # Boxplot grouped by fraction  with value of column out_value
    ax = sns.violinplot(x=out_value_modi["class"],y=out_value_modi["out_value"],showmeans=True, showmedians=True,
                        palette="Set3")
    # ax.set_ylim(280,340)
    ax.set_ylim(280,310)
    ax.set_title('Kernel'+str(kernel),size=30)
    ax.set_xticklabels(['Tree Canopy','Grass/Shrubs','Bare Soil',
                        'Water', 'Building','Road','Other Impervious',
                        'Railways'],size=30,rotation=25)
                
    ax.set_xlabel('Land Cover Class',size=30,weight='bold')
    plt.tick_params(axis='y',labelsize=30)
    ax.set_ylabel('Temperature (K)',size=30,weight='bold')
    
    

    
    
    # Trying boxplot without outliers
    '''
    fig, ax = plt.subplots()
    # Boxplot grouped by fraction  with value of column out_value
    out_value_modi_outlier = out_value_modi_list[-1].loc[(out_value_modi_list[-1]["out_value"] >= 280) & (out_value_modi_list[-1]["out_value"] <= 320)]
  
            
    # Trying to plot violinplot
    import seaborn as sns
    fig, ax = plt.subplots()
    kernel=10
    ax = sns.violinplot(x=out_value_modi_outlier["class"],y=out_value_modi_outlier["out_value"],showmeans=True, showmedians=True,
                        palette="Set3")
    ax.set_title("Violin Plot of Canopy Temperature Distribution for Kernel = "+str(kernel),size=30)
    ax.set_xticklabels(['Tree Canopy','Grass/Shrubs','Bare Soil',
                    'Water', 'Building','Road','Other Impervious',
                    'Railways'],size=30,rotation=25)
    ax.set_xlabel('Land Cover Class',size=30,weight='bold')
    ax.set_ylabel('Temperature (K)',size=30,weight='bold')
    plt.tick_params(axis="x",labelsize=30)
    plt.tick_params(axis="y",labelsize=30)
    
    
    
    '''
    
                
    return out_value_modi


'''
## With parallel ##
import multiprocessing as mp
import time 

start_time = time.time()
pool = mp.Pool(mp.cpu_count())

out_value_modi_para = pool.starmap(groupby_outvalue, [(out_value_list[0], row,col,3)])

pool.close()

print(time.time() - start_time  )


## Without parallel ##
start_time = time.time()
out_value_modi = groupby_outvalue(out_value_list[0],row,col,kernel=3)
print(time.time() - start_time)
'''



def canopy_temp(out_value_modi,tick_min,tick_max,tick_interval,frac_index,frac_name,canopy_temp=False):
    '''
    Function to generate canopy temperature map to discover the spatial correlation of canopy temp:
        Input:
            out_value_modi: results from func groupby_landcover...
            tick_min: min value of color ramp
            tick_max: max value of color ramp
            tick_interval: interval of color ramp
            frac_index: index of land cover class e.g. canopy = 0
            frac_name: name of land cover class e.g. "canopy"
            canopy_temp: index to map either end member temperature or fraction of land cover. By default is False, which
            means mapping End member canopy temperature
    
    Example: 
        # Mapping end member  temperature
        out_value_modi_emis_member = canopy_temp(out_value_modi_emis,tick_min=290,tick_max=304,tick_interval=0.1,frac_index=0,frac_name="Canopy")
        
        # Mapping fraction of land cover
        out_value_modi_emis_member = canopy_temp(out_value_modi_emis,tick_min=0,tick_max=1,tick_interval=0.1,frac_index=0,frac_name="Canopy",
        canopy_temp=True)
    '''
    import cmocean  
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
    
    
    # Define levels of ticks and tick intervals
    tick_min=tick_min
    tick_max=tick_max
    levels = MaxNLocator(nbins=(tick_max - tick_min)/tick_interval).tick_values(tick_min, tick_max)
    cmap = plt.get_cmap(cmocean.cm.thermal)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    
    # Get value only for canopy (fraction i = 0)
    out_value_modi_emis_member = out_value_modi_emis.groupby('class').get_group(frac_index).groupby(('nrow','ncol')).agg({"out_value":"mean",
                                                            "value_fraction":"mean"}).reset_index()
    
#    # Define a range for fraction 
#    fraction_range = np.arange(0,1.1,0.1)
#    
#    
#    # Look up over the fraction_range to get the value of end member temperature at that fraction limit
#    out_value_modi_emis_canopy["out_value_frac"]=np.nan
#    for i in range(len(fraction_range)):
#        try:
#            # Calculate mean end member temp in the certain fraction range
#            mean_value =  out_value_modi_emis_canopy.loc[(out_value_modi_emis_canopy["fraction"]>fraction_range[i])
#            &(out_value_modi_emis_canopy["fraction"]<fraction_range[i+1])]["out_value"].mean()
#            
#            
#            out_value_modi_emis_canopy.loc[(out_value_modi_emis_canopy["fraction"]>fraction_range[i])
#            &(out_value_modi_emis_canopy["fraction"]<fraction_range[i+1]),"out_value_frac"]= mean_value
#        
#        except:
#            pass
       
    if canopy_temp is not False:
        df_pivot_result = out_value_modi_emis_member.pivot_table(index='nrow',
                                                   columns='ncol',
                                                   values='value_fraction')
    else: 
        df_pivot_result = out_value_modi_emis_member.pivot_table(index='nrow',
                                                   columns='ncol',
                                                   values='out_value')

    
    fig, ax = plt.subplots()
    plot_out_temp = ax.pcolormesh(df_pivot_result,norm=norm, cmap=cmap)
    ax.set_aspect(aspect='equal')
    
    # Modify colorbar 
    cbar = fig.colorbar(plot_out_temp,ax=ax, shrink=.7)
    # access to cbar tick labels:
    cbar.ax.tick_params(labelsize=20) 
    
    
    if canopy_temp is not False:
        plt.title("Fraction of "+str(frac_name)+" Land Cover Class",size=30)
        cbar.set_label('Fraction of Land cover',size=30)
    else:
        plt.title("End member Temperature of "+str(frac_name)+" Land Cover Class",size=30)
        cbar.set_label('LST (K)',size=30)

    
    return out_value_modi_emis_member


def test_sensitivity(out_value_list,kernel_list,class_code):
    '''
    This function is for testing sensitivity of canopy temperature for 
    differenet kernel size
    
    Example:
    kernel_list = [10,15,20,25]
    out_value_modi_list,out_value_modi_list_grp = test_sensitivity(out_value_list,kernel_list=kernel_list,class_code=0)
    
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    out_value_modi_list = []
    out_value_modi_list_grp = []
    for i in range(len(out_value_list)):
        out_value_modi = groupby_outvalue(out_value_list[i],row,col,kernel=kernel_list[i])
        out_value_modi_list.append(out_value_modi)
        out_value_modi_list_grp.append(out_value_modi.groupby('class').get_group(class_code))
        
    # Plotting canopy temperature for each kernel size, recommended to create a boxplot
    fig, ax = plt.subplots()
    for item in range(len(out_value_modi_list_grp)):
        ax.plot([kernel_list[item] for i in range(len(out_value_modi_list_grp[item]))],
                out_value_modi_list_grp[item]['out_value'],
                'o',
                label='Kernel Size '+str(kernel_list[item]))
        
        ax.set_xlabel('Kernel Window',size=30,weight='bold')
        ax.set_ylabel('Canopy Temperature (K)', size=30, weight='bold')
        plt.legend(loc='upper right', prop={'size': 25})
        plt.tick_params(axis='y',labelsize=30)
        ax.set_xticks(np.arange(3,10,2))
        ax.set_xticklabels(['3x3','5x5','7x7','9x9'],size=30,rotation=25)
        ax.set_ylim(280,340)
        
        
    '''
    # Trying to extract only Fraction of Canopy and plot together with out_temp 
    # to test the spatial relationship
    # Here is an example of kernel = 10 corresponding to first element of out_value_modi_list
    out_value_modi_list[0]["out_temp"]=out_value_modi_list[0]["out_value"]*out_value_modi_list[0]["fraction"]

    out_value_modi_list[0]["fraction_canopy"] = out_value_modi_list[0].loc[out_value_modi_list[0]["class"] == 0.0]["fraction"]

    
    out_value_modi_grp_10_canopy = out_value_modi_list[0].groupby(("nrow","ncol")).agg({"out_temp":"sum",
    "fraction_canopy":"max",'indep_value':'mean'}).reset_index()
    
    # Using seaborn to plot and calculate pearmanr coeff
    import seaborn as sns
    from scipy import stats
    
    g = sns.jointplot(x=out_value_modi_grp_10_canopy["fraction_canopy"],
    y=out_value_modi_grp_10_canopy["out_temp"],kind="reg",joint_kws={"color":"green"}).annotate(stats.pearsonr)
    
    
    '''
        
    return out_value_modi_list,out_value_modi_list_grp

def cut_value(out_value_modi, _cut,landcoverclass,landcoverclass_code,color):
    '''
    This function to is cut the fraction of land cover class
    into smaller range and aggregated by mean for each range
    The result then will be ploted to calculate correlation coefficient
    between out_temp and fraction of land cover 
    Inputs:
        out_value_modi: pandas dataframe 
        _cut: cut range
    
    Example:
        out_value_modi_grp_cut_grp = cut_value(out_value_modi_emis,
         _cut = np.arange(0,1.1,0.005), landcoverclass='canopy', landcoverclass_code=0.0,color="green")
        
        out_value_modi_grp_cut_grp_grass = cut_value(out_value_modi_emis,
         _cut = np.arange(0,1.1,0.005), landcoverclass='grass', landcoverclass_code=1.0,color="red")
        
        
        # Testing with combined dataframe 
        cut_value(out_value_modi=out_value_modi_combine,
         _cut = np.arange(0,1.1,0.05), landcoverclass='canopy', landcoverclass_code=0.0, 
         color='green')
        
        
        
        
    '''
    import numpy as np
    import pandas as pd
    
    
    # Add new columns for fraction of land cover and corresponding LST
    out_value_modi[str("fraction_"+str(landcoverclass))] = np.nan
    out_value_modi["out_temp"] = np.nan
    
    try:
        # Insert new column 'out_temp' by multipying 'out_value' by 'fraction'    
    
        # Insert new column only contains value of fraction of canopy, which pixels does not contains 
        # Canopy will be resulted in Nan values
        
        out_value_modi.loc[out_value_modi["class"]==landcoverclass_code,
                           str("fraction_"+str(landcoverclass))] = out_value_modi.loc[out_value_modi["class"]==landcoverclass_code]["value_fraction"]
        
    except:
        pass


    # New dataframe group by 
    out_value_modi_grp = out_value_modi.groupby(("nrow","ncol","type")).agg({"out_temp":"sum",
    "fraction_"+str(landcoverclass):"max",'indep_value':'mean',
    'out_value':'mean'}).reset_index()
                    

    
    # Cut the current dataframe 
    out_value_modi_grp_cut = pd.cut(out_value_modi_grp[str("fraction_"+str(landcoverclass))],bins=_cut,labels=False,
                                                       right=True)
    
    # Groupy by the cutted dataframe
    out_value_modi_grp_cut_grp = []
    out_value_modi_grp_cut_grp = out_value_modi_grp.groupby(by=[out_value_modi_grp_cut,out_value_modi_grp["type"]],
                                                            as_index=False).mean()
    

    # Plotting new dataframe
    import seaborn as sns
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # Cal regression coeff
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=out_value_modi_grp_cut_grp["fraction_"+str(landcoverclass)], 
                                                                   y=out_value_modi_grp_cut_grp['out_value'])
    

#    # Generating jointplot 
#    grid = sns.JointGrid(x="fraction_"+str(landcoverclass), y='out_temp', 
#                         data=out_value_modi_grp_cut_grp)
#    
#    g = grid.plot_joint(sns.scatterplot,  
#                    alpha=0.7,
#                    s=200,
#                    data=out_value_modi_grp_cut_grp)
#    
#    g = grid.plot_joint(sns.kdeplot, zorder=0, n_levels=6)
#    
#    g.set_axis_labels(xlabel='Fraction of '+str(landcoverclass),ylabel='Land Surface Temperature (K)',size=30)
#    

    
    # Plotting regression:
    g = sns.jointplot(x="fraction_"+str(landcoverclass),
                      y='out_value', 
                      data = out_value_modi_grp_cut_grp,
                      color=color,
                      kind="reg")

    # Add an annotation e.g. remember to change
    xy = (0.5, 303) # Coordinate of the equation (x,y)
    rota = 0 # Angle for rotating the equation 
    plt.annotate(r"y=${0:.4f}$x+${1:.4f}$".format(slope,intercept) + "\n" + 
                 r"r_value=${0:.4f}$".format(r_value),
                 xy=xy,rotation=rota,size=30)
    
    
    
    plt.xlabel('Fraction of '+str(landcoverclass), size=30)
    plt.ylabel('End member '+str(landcoverclass)+" Temperature (K)", size=30)
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.xlim(0,1)
    


    
    


    return out_value_modi_grp_cut_grp

def cut_value_frac_range(out_value_modi,frac_index,frac_name,color):
        # Define a range for fraction 
    '''
    Example:
        out_value_modi_element = cut_value_frac_range(out_value_modi_emis,frac_index=0,frac_name="Canopy",color="green")
        
        # Buffer 500 m river and water bodies
        out_value_modi_element_river = cut_value_frac_range(out_value_modi_emis_river,frac_index=0,frac_name="Canopy",color="green")



        # Compare agg and non agg values
        out_value_modi_element_canopy_non_agg = cut_value_frac_range(out_value_modi_emis_non_agg,frac_index=0,frac_name="Canopy",color="green")
        out_value_modi_element_canopy_agg = cut_value_frac_range(out_value_modi_emis,frac_index=0,frac_name="Canopy",color="green")

        
        
        out_value_modi_element_canopy_non_agg["agg_index"] = "None Aggregated"
        out_value_modi_element_canopy_agg["agg_index"] = "Aggregated"
        
        df_concat = pd.concat([out_value_modi_element_canopy_non_agg,out_value_modi_element_canopy_agg])
        
        fig, ax = plt.subplots()
        title="Canopy"
        sns.pointplot(x="frac_index",y="out_value",data=df_concat,hue="agg_index")
        plt.legend(prop={"size":30})
        plt.tick_params(axis="x",labelsize=30,rotation=45)
        plt.tick_params(axis="y",labelsize=30,rotation=45)
        plt.title(title,size=30)
        
        
        ### For Grass
        out_value_modi_element_grass_non_agg = cut_value_frac_range(out_value_modi_emis_non_agg,frac_index=1,frac_name="Grass",color="red")
        out_value_modi_element_grass_agg = cut_value_frac_range(out_value_modi_emis,frac_index=1,frac_name="Grass",color="red")

        
        
        out_value_modi_element_grass_non_agg["agg_index"] = "None Aggregated"
        out_value_modi_element_grass_agg["agg_index"] = "Aggregated"
        
        df_concat = pd.concat([out_value_modi_element_grass_non_agg,out_value_modi_element_grass_agg])
        
        fig, ax = plt.subplots()
        title="Grass End Member Temperature vs Fraction of Grass"
        sns.pointplot(x="frac_index",y="out_value",data=df_concat,hue="agg_index")
        plt.legend(prop={"size":30})
        plt.tick_params(axis="x",labelsize=30,rotation=45)
        plt.tick_params(axis="y",labelsize=30,rotation=45)
        plt.title(title,size=30)
        
        
    '''
        
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    
    
    fraction_range = np.arange(0,1.1,0.05)
    
    
    # Look up over the fraction_range to get the value of end member temperature at that fraction limit
    out_value_modi["frac_index"]=np.nan
    for i in range(len(fraction_range)):
        try:
            if fraction_range[i] == 0:
                out_value_modi.loc[(out_value_modi["value_fraction"] >= fraction_range[i])
            &(out_value_modi["value_fraction"] <= fraction_range[i+1]),"frac_index"]= fraction_range[i+1]
            
            
            else:
                out_value_modi.loc[(out_value_modi["value_fraction"] > fraction_range[i])
            &(out_value_modi["value_fraction"] <= fraction_range[i+1]),"frac_index"]= fraction_range[i+1]
        
        except:
            pass
        
    out_value_modi_element = out_value_modi.loc[out_value_modi["class"]==frac_index]
    
    
    fig, ax = plt.subplots()
    sns.pointplot(x="frac_index",y="out_value",data=out_value_modi_element,color=color)
    
    plt.xlabel("Fraction of "+str(frac_name),size=30)
    plt.ylabel("End member Temperature of "+str(frac_name),size=30)
    


    plt.tick_params(axis="x",labelsize=30,rotation=45)
    plt.tick_params(axis="y",labelsize=30,rotation=45)
    
    
    return out_value_modi_element        
    


def extract_out_temp(out_value_modi,kernel,emissivity=False):
    '''
    This function is to extract the output temperature for each grid cell based on 
    the out_value of temperature for land cover class and their fraction.
    
    Also, the function is also applied to plot pcolormesh by convert pandas DataFrame to
    pivot_table
    
    Input:
        out_value_modi: pandas Dataframe as a result of def groupby_outvalue
        emissivity: index to indicate if using Emissivity factor, so the final results should be elimiated by emissivity
        
    Output:
        out_value_modi_final: pandas Dataframe contains output values of temperature
        for each land cover class:
            'nrow': index of row
            'ncol': index of col
            'ori_temp': temperature of each pixel extracted from ECOSTRESS LST
            'out_temp': temperature of each pixel as results of Linear Algebra al
        
    Example:
        out_value_modi_final_list = []
        for i in range(len(kernel_list)):
            out_value_modi_final = extract_out_temp(out_value_modi_list[i],kernel=kernel_list[i])
            out_value_modi_final_list.append(out_value_modi_final)
    '''
    import matplotlib.pyplot as plt
    import pandas as pd
    
    
    # Insert new column contains final output temperature by multipy "out_value" * "fraction"
    out_value_modi["out_temp"]=out_value_modi["out_value"]*out_value_modi["fraction"]
    
    
    # Group the current dataframe by nrow and ncol columns
    out_value_modi_grp = out_value_modi.groupby(("nrow","ncol"))
    
    # Aggregate by sum of "out_temp" column and mean of "indep_value" column
    out_value_modi_grp_agg = out_value_modi_grp.agg({"out_temp":"sum",
                                                     "indep_value":"mean"}).reset_index()
                    
                    
    # Insert new column "resi_temp" by substrating "indep_value"-"out_temp" columns
    out_value_modi_grp_agg["resi_temp"] = out_value_modi_grp_agg["indep_value"] - out_value_modi_grp_agg["out_temp"]

#    for nrow in range(row):
#        for ncol in range(col):
#            try:
#                out_value_modi_final = out_value_modi_final.append({'nrow':nrow,
#                                                                'ncol':ncol,
#                                                                'ori_temp':out_value_modi.groupby(['nrow','ncol']).get_group((nrow,ncol))['indep_value'].mean(),
#                                                                'out_temp':sum(out_value_modi.groupby(['nrow','ncol']).
#                                                                get_group((nrow,ncol))['out_value']*out_value_modi.groupby(['nrow','ncol']).
#                                                                get_group((nrow,ncol))['fraction'])},
#                                                               ignore_index=True)
#                print(nrow,ncol)
#            except:
#                pass
#    
#    # Inserting new subtrated column by ori_temp and out_temp
#    out_value_modi_final.insert(len(out_value_modi_final.columns),
#                                column='resi_temp',
#                                value=out_value_modi_final['ori_temp']-out_value_modi_final['out_temp'])       
#                                                     
    # Convert pandas Dataframe to pandas pivot_table and plot using pcolormesh 
    # Generating color scale for plotting colormesh

    import cmocean  
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
    
    tick_max = out_value_modi_grp_agg[['indep_value','out_temp']].max().max()
    tick_min = out_value_modi_grp_agg[['indep_value','out_temp']].min().min()
    levels = MaxNLocator(nbins=(tick_max - tick_min)/0.2).tick_values(tick_min, tick_max)
    cmap = plt.get_cmap(cmocean.cm.thermal)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    # Pivot table of original temperature map (the one extracted by ECOSTRESS LST)
    df_pivot_ori = out_value_modi_grp_agg.pivot_table(index='nrow',
                                                    columns='ncol',
                                                    values='indep_value')
    

    # Pivot table of extracted temperature map (by Linear Algebra)
    df_pivot_result = out_value_modi_grp_agg.pivot_table(index='nrow',
                                                       columns='ncol',
                                                       values='out_temp')
    
    # Pivot table of residual temperature by subtrating original and 
    # output temperature
    df_pivot_resi = out_value_modi_grp_agg.pivot_table(index='nrow',
                                                     columns='ncol',
                                                     values='resi_temp')
    
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    plot_ori_temp = ax1.pcolormesh(df_pivot_ori,norm=norm,cmap=cmap)
    ax1.set_title('Original ECOSTRESS LST',size=30)
    ax1.set_aspect(aspect='equal') # Set aspect of colormesh so has equal square pixel

    plot_out_temp = ax2.pcolormesh(df_pivot_result,norm=norm, cmap=cmap)
    ax2.set_title('Output Temperature with Kernel = '+str(kernel),size=30)
    ax2.set_aspect(aspect='equal')
    
    # Adding colorbar
    fig.colorbar(plot_ori_temp,ax=ax1, shrink=.7)
    fig.colorbar(plot_out_temp,ax=ax2, shrink=.7)
    
    # Plotting residuals map 
    fig2, ax3 = plt.subplots()
    plot_resi_temp = ax3.pcolormesh(df_pivot_resi, cmap=cm.twilight_shifted)
    ax3.set_title('Residual Temperature Map'+str(kernel),size=30)
    ax3.set_aspect(aspect='equal')
    fig2.colorbar(plot_resi_temp, ax=ax3)    
                    
    return out_value_modi_grp_agg           


def critical_grid(out_value_modi_emis):
    
    
    from scipy import ndimage
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    
    
    out_value_modi_emis_canopy = out_value_modi_emis.groupby('class').get_group(0)
    
    df_pivot = out_value_modi_emis_canopy.pivot_table(index='nrow',
                                                      columns='ncol',
                                                     values='out_value')
    
    size = [3,5,7,9,11,13,15,17,21,25,31]

    outVar_list = []
    

    for i in size:
        mask = np.ones((i,i))
        mask[int(np.round(i/2)),int(np.round(i/2))] = 0

        outVar_list.append(ndimage.generic_filter(df_pivot.to_numpy(), 
                                                  np.nanstd, 
                                                  size=i,
                                                  mode='constant',
                                                  cval=np.nan,
                                                  footprint=mask
                                                  ))
        
    sns.violinplot(data=outVar_list)
    
    
    fig, ax = plt.subplots()
    sns.pointplot(data=outVar_list,ax=ax)
    ax.set_xticklabels(size)
    ax.set_xlabel('Moving Size',size=30)
    ax.set_ylabel('Standard Deviation',size=30)
    plt.tick_params(axis='x',labelsize=30)
    plt.tick_params(axis='y',labelsize=30)
    
    
    # Colormesh
    import cmocean  
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
    
    tick_max = np.nanmax(outVar_list)
    tick_min = np.nanmin(outVar_list)
    levels = MaxNLocator(nbins=(tick_max - tick_min)/0.2).tick_values(tick_min, tick_max)
    cmap = plt.get_cmap(cmocean.cm.thermal)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    #
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2)
    
    mesh1 = ax1.pcolormesh(outVar_list[0],norm=norm,cmap=cmap)
    fig.colorbar(mesh1)
    ax1.set_title('Moving window'+str(size[0]), size=30)
    ax1.set_aspect('equal')
    
    mesh2 = ax2.pcolormesh(outVar_list[2],norm=norm,cmap=cmap)
    ax2.set_title('Moving window'+str(size[2]), size=30)
    
    mesh3 = ax3.pcolormesh(outVar_list[4],norm=norm,cmap=cmap)
    ax3.set_title('Moving window'+str(size[4]), size=30)
    
    mesh4 = ax4.pcolormesh(outVar_list[-1],norm=norm,cmap=cmap)
    ax4.set_title('Moving window'+str(size[-1]), size=30)
    
    axes = ((ax1, ax2), (ax3, ax4))
    
    for ax in axes:
        for a in ax:
            a.set_aspect('equal')
    
    fig.colorbar(mesh1)

def std_convoluted(df_pivot, N):
    '''
    Example:
        var_convol = std_convoluted(df_pivot,N=3)
    '''
    
    import numpy as np
    import scipy
    
    im = df_pivot.to_numpy()
    im2 = (df_pivot.to_numpy())**2
    ones = np.ones(im.shape)

    kernel = np.ones((2*N+1, 2*N+1))
    
    s = scipy.signal.convolve2d(im, kernel, mode="same",fillvalue=np.nan,boundary='fill')
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    var_convol = np.nansqrt((s2 - s**2 / ns) / ns)
    
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(var_convol)
    ax.set_aspect('equal')
    fig.colorbar(mesh)
    
    return var_convol


def optimize_lsq(A,b,bounds,method, lsq_solver=None):
    """
    This function is to optimze the least-square solution for over-determinated 
    matrix
    m: number of equations
    n: number of unknown variables
    
    
    
    b = A*x
    b: independent matrix: array-like with shape of (m,) e.g. [303,305,307]
    A: coefficient matrix: array-like with shape of (m,n) 
        e.g. [[0.,1,0,],[0,1,0],[0,1,0]]
        
    bounds: bounds values for output results: tupple (min,max) e.g. (303,306)
    
    x: output 
    
    
    Example:
        out_x, res = optimize_lsq(A,b,bounds=(303,306), method="bvls", lsq_solver="exact")
    """
    
    from scipy.optimize import lsq_linear
    from scipy import stats
    import seaborn as sns


    if lsq_solver is not None:
        lsq_solver=lsq_solver
    
    res= lsq_linear(A,b,bounds=bounds, method="bvls", lsq_solver=lsq_solver)
    
    # Create new output varibale by multiplying out_put * A
    
    out_x = res.x*A
    for item in range(len(out_x)):
        out_x[item] = sum(out_x[item])
        
    # Aggregate by mean of output list
    out_x = out_x.mean(axis=1)
    
    # Plotting to compare output and input values


    g = sns.jointplot(x=out_x,y=b).annotate(stats.pearsonr)  
    
    
    print(res)
    return out_x, res
    
    
    
        
def moving_average(data, windows, nskip=None, levels=None):
    '''This function is based on the 
    
    This is done by convolving the image with a normalized box filter. 
    It simply takes the average of all the pixels under kernel area 
    and replaces the central element with this average. 
    This is done by the function cv2.blur() or cv2.boxFilter(). 
    We should specify the width and height of kernel. 
    A 3x3 normalized box filter would look like this:
                1  1  1
      1/9 *     1  1  1
                1  1  1       
    
    If you don’t want to use a normalized box filter, 
    use cv2.boxFilter() and pass the argument normalize=False to the function.
    
    Variables:
        data: formatted as numpy 2-D array 
        windows: kerner size for moving average function
        nskip: space between grid, for defining xtick 
        
        '''
    import cv2 ##import openCV library for image processing
    import numpy as np
    import cmocean
    from numpy.polynomial.polynomial import polyfit
    
    from scipy import stats
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    
    from progress.bar import Bar

    bar = Bar('Processing', max=20)
    for i in range(20):
        # Do some work
        bar.next()
    bar.finish()
    
    if nskip is None:
        nskip = len(data)/100
            
    
    
    ##Convoling the original image using moving average method or box filter
    kernel_size = (windows,windows) ##define kernel size 
    results_blur = cv2.blur(data, kernel_size)
    '''
    in case there is any nan or 0 values, create a mask 
    to ignore the calculation
    '''
    mask = (data>0)

    
    if np.any(~mask):
        scaling_vals= cv2.blur(mask.astype(np.float64), kernel_size)
        results_blur[mask] /= scaling_vals[mask]
        results_blur[~mask] = 0
        scaling_vals = None
        mask = None
        
    '''
    Plot the results using pcolormesh
    
    '''
    
    if levels is None:
        mask_data = np.ma.masked_equal(data, 0.0, copy=False)
        mask_results = np.ma.masked_equal(results_blur, 0.0, copy=False)
        tick_min = np.min(np.array([np.min(np.min(mask_data)),np.min(np.min(mask_results))]))
        tick_max = np.max(np.array([np.max(np.max(mask_data)), np.max(np.max(mask_results))]))
        
        levels = MaxNLocator(nbins=(tick_max - tick_min)/0.2).tick_values(tick_min, tick_max)
        cmap = plt.get_cmap(cmocean.cm.thermal)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    print(mask_results)

    '''
    Plotting the results in colormesh and display the distribution of the dataset
    '''
    ###Before bluring data
    fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(40,30))
    orig_map = ax1.pcolormesh(data, cmap = cmocean.cm.thermal, norm=norm)
    fig1.colorbar(orig_map, ax=ax1)
    '''
    Histogram plot
    '''
    ax2.hist(data.flatten(), bins=50, range=(tick_min,tick_max), color="pink",
                         alpha = 0.5)
    ax2.set_xlim(tick_min, tick_max)
    
    
    ##After bluring data
    fig2, (ax3, ax4) = plt.subplots(ncols=2, figsize=(40,30))
    blur_map = ax3.pcolormesh(results_blur, cmap = cmocean.cm.thermal, norm=norm)
    fig2.colorbar(blur_map, ax = ax3)
    
    ax4.hist(results_blur.flatten(), bins=50, range=(tick_min, tick_max), color="blue",
             alpha=0.5, label="Moving average, kernel_size= {:.2f}".format(windows))
    
    ax4.hist(data.flatten(), bins=50, range=(tick_min,tick_max), color="pink",
             alpha=0.5, label="Original data")
    
    ax4.set_xlim(tick_min, tick_max)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               prop={'size': 25})
    
    
#    plt.tick_params(axis='y', labelsize = 50, rotation=25)
#    plt.tick_params(axis='x', labelsize = 50, rotation=25)
    
    fig1.savefig(r"C:\Users\tvo\Desktop\Work\Updates\11092019\figure\moving_average_1.png",dpi=150)
    fig2.savefig(r"C:\Users\tvo\Desktop\Work\Updates\11092019\figure\moving_average_2.png",dpi=150)
    
    

        
    plt.show()

    
    
    
    return results_blur

def plot_colormesh(data, levels=None):
    '''
    Plotting map with pcolormesh 
    data: formatted as a dataframe
    
    '''
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import cmocean
    import numpy 
    
    font = {'family':'DejaVu Sans',
        'style':'oblique',
        'size':25
        }
    
    
    
     
    ###Transforming dataframe to grid with index=nrow and columsn=ncol
    data_pivot_orig = data.pivot_table(index="nrow",columns="ncol",values="MEAN_LST")
    data_pivot_results = data.pivot_table(index="nrow",columns="ncol",values="modelled_LST")

    
    
    if levels is None:
        tick_min = np.min(np.array([np.min(np.min(data_pivot_results)),np.min(np.min(data_pivot_orig))]))
        tick_max = np.max(np.array([np.max(np.max(data_pivot_results)), np.max(np.max(data_pivot_orig))]))
        
        levels = MaxNLocator(nbins=(tick_max - tick_min)/0.2).tick_values(tick_min, tick_max)
        cmap = plt.get_cmap(cmocean.cm.thermal)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))
    ax1.set_title('Original LST', size=50, fontdict=font, weight='bold')
    ax2.set_title('Downscaled LST', size=50, fontdict=font, weight='bold')
    
    ##Using function pcolormesh 
    orig_map = ax1.pcolormesh(data_pivot_orig, cmap = cmocean.cm.thermal, norm=norm)
    
    #orig_map_cb = orig_map.ax.figure.colorbar(orig_map)

    results_map = ax2.pcolormesh(data_pivot_results, cmap = cmocean.cm.thermal, norm=norm)
    #results_map_cb = results_map.ax.figure.colorbar(results_map)
    
    cb1 = fig.colorbar(results_map, ax=ax1)
    cb1.set_label('LST (K)', size=25, weight='bold', fontdict=font)
    cb2 = fig.colorbar(orig_map, ax=ax2)
    cb2.set_label('LST (K)', size=25, weight='bold', fontdict=font)
    
    
    """
    xticks and yticks
    """
    nticks = 10
    min_lats = np.min(data.lats.round())
    max_lats = np.max(data.lats.round())
    min_lons = np.min(data.lons.round())
    max_lons = np.max(data.lons.round())
    min_nrow = np.min(data.nrow)
    max_nrow = np.max(data.nrow)
    min_ncol = np.min(data.ncol)
    max_ncol = np.max(data.ncol)
    
    xticks = np.arange(min_ncol, max_ncol + max_ncol/nticks, (max_ncol - min_ncol)/nticks)
    xtickslabels = np.arange(min_lons, max_lons - max_lons/nticks, (max_lons-min_lons)/nticks)
    
    yticks = np.arange(min_nrow,max_nrow + max_nrow/nticks, max_nrow/nticks)
    ytickslabels = np.arange(min_lats, max_lats + max_lats/nticks, (max_lats-min_lats)/nticks)
    
    axes = (ax1, ax2)
    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtickslabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytickslabels)
        ax.set_xlabel('Longtitude (meters)', weight='bold',
                      fontdict=font)
        ax.set_ylabel('Latitude (meters)', weight='bold',
                      fontdict=font)
        
        ax.tick_params(axis='x', rotation=45, size=50)
        ax.tick_params(axis='y', rotation=45, size=50)
        ax.grid(linestyle='--')
        plt.tick_params(axis='y', labelsize = 50, rotation=25)
        plt.tick_params(axis='x', labelsize = 50, rotation=25)
        plt.show()

        




















    