# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:20:17 2020

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
def reduce_mem_usage(df, verbose=True):
    '''
    Function to reduce memory usage of the dataframe,
    by reducing type of data
    
    Example:
    out_value_modi = reduce_mem_usage(out_value_modi)
    '''
    import numpy as np
    
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def dbf_to_df(path, bad_value):
    
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
        
        # Define path containing interested files : whole NCY
        path = 'C:/Users/tvo/Documents/urban_project/Test_NYC/'
        

        
        # Read .vat.dbf files and convert to pandas Dataframe
        df_landcover_concat = dbf_to_df(path,bad_value=4500)
        
        
        # Save temporary file to a folder 
        path_save_pickle = '//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/'

        df_landcover_concat.to_pickle(path_save_pickle+'df_landcover_concat')


        
    
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
        
        
        
        
            
    return df_landcover_concat

def dbf_to_df_date(path_sub,cloud_masked=False):
    '''
    Function to convert LST and Emis for different dates:
        Example:
        
        date_code = '2019-07-28T040102UTC/'
        
        # Define path containing LST and Emis files for interested date:
        path_sub = 'C:/Users/tvo/Documents/urban_project/Test_NYC/'+date_code
        
        # Save temporary file to a folder 
        path_save_pickle_sub = '//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/'+date_code
        
        # Create a new directory if not exists yet
        import os
        try:
            os.mkdir(path_save_pickle_sub)
        except:
            pass
            
            
        # Run function to red dbf files:
        df_lst, df_emiswb = dbf_to_df_date(path_sub,cloud_masked=True)
        
        df_lst.to_pickle(path_save_pickle_sub+'df_lst')
        df_emiswb.to_pickle(path_save_pickle_sub+'df_emiswb')
        
    
    '''
    import pandas as pd
    from dbfread import DBF
    from pandas import DataFrame
    import os
    import numpy as np

    
        # Define empty list contains list of fiels in directory
    listdir_sub = []
    
    # Only search for files with extension .vat.dbf and append to listdir
    for file in os.listdir(path_sub):
        if file.endswith('.tif.vat.dbf'):
            listdir_sub.append(file)
            
    # Read .vat.dbf ofr LST and Emiswb files:
    dbf_dataset_sub = []
    for file in listdir_sub:
        dbf = DBF(path_sub + file)
        dbf_dataset_sub.append(dbf)

    if cloud_masked is False:
        # Creating a separate dataFrame for LST
        for item in dbf_dataset_sub:
            if 'LST' in str(item):         
                df_lst = DataFrame(item)
                
        # Creating a separate dataframe for Emis2
        for item in dbf_dataset_sub:
            if "emiswb" in str(item):
                df_emiswb = DataFrame(item)
                
    elif cloud_masked is True:
        # Creating a separate dataFrame for LST
        for item in dbf_dataset_sub:
            if 'LST_cloudmasked' in str(item):         
                df_lst = DataFrame(item)
                
        # Creating a separate dataframe for Emis2
        for item in dbf_dataset_sub:
            if "emiswb_cloudmasked" in str(item):
                df_emiswb = DataFrame(item)
        
    
    # Replacing non value e.g. 0 by -9999 and later ignore for calculation:
    # Using column "Count_1" to idenity -9999 values
    try:
        df_lst.loc[df_lst["COUNT_1"] == 0,"COUNT_1"] = -9999
    
        df_emiswb.loc[df_emiswb["COUNT_1"] == 0, "COUNT_1"]= -9999
    except:
        pass
    
    
    return df_lst, df_emiswb


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
        # Path to the values of whole NYC : 
        path_save_pickle = '//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/'
        path_save_pickle_sub = '//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/'+date_code
        
        
        
        df_landcover_concat_nyc = pd.read_pickle(path_save_pickle+'df_landcover_concat')
        df_emiswb_nyc = pd.read_pickle(path_save_pickle_sub+'df_emiswb')
        
        
        df_emis_repre = emis_pure_pixel(df_landcover_concat_nyc, df_emiswb_nyc)
        
        df_emis_repre.to_pickle(path_save_pickle_sub+'df_emis_repre')


        
        # Read pickle
        df_emis_repre = pd.read_pickle(path_save_pickle_sub+'df_emis_repre')
        
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
        for i in range(8):
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
    
def cloud_mask(file):
    '''
    Function to decode cloud mask image and assign cloudy pixels nan values
    
    '''
    import gdal
    import numpy as np
    
    
    dataset = gdal.Open(file)
    
    data = dataset.ReadAsArray()
    
    # Maske out nan values
    data = np.ma.masked_where(data == -999, data)
    
    # decode the cloud mask 8-bit
    qa_decode = np.vectorize(np.binary_repr)(data,width=8)
    
    
    
    # Create a new image wiith cloud decoded, bit 
    cloud = np.empty_like(data, dtype=np.int16)
    for row in range(len(qa_decode)):
        for col in range(len(qa_decode[row])):
            
            if qa_decode[row][col] != np.nan:
                cloud[row][col] = int(qa_decode[row][col][-1])
            else:
                pass
    
    
    
    return cloud_mask   
 
def cal_linear_alg_lstsq(df_landcover, df_lst, df_emiswb, df_emis_repre, cloud_mask, row, col, kernel_list,  _type, radiance=False,bounds=None, 
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
        
        # Whole NYC (70 m)
        row = 884
        col = 786

        
        # Testing with different kernel sizes
        # Radiance function
        out_value_list, end_time = cal_linear_alg_lstsq(df_landcover_concat, df_lst, df_emiswb, df_emis_repre,  
            row, col, kernel_list=[10], _type="radiance", 
            bounds=(290**4,310**4), moving_pixel = 4,  radiance=True)
        
        
        # Emissivity function
        out_value_list, end_time = cal_linear_alg_lstsq(df_landcover_concat, df_lst, df_emiswb, df_emis_repre,
            row, col, kernel_list=[25], _type="Emis", 
            bounds=(290**4,310**4), moving_pixel = 5,  radiance=False)
        
 

    
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy.linalg as la
    import pandas as pd
    from scipy import linalg
    from scipy.optimize import lsq_linear
    import time
    
    # Starting time
    start_time = time.time()

    

    
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
    from_dict function: 03.28.2020
    '''
    
    # New version: from_dict function: 03.28.2020
    # Create an empty pandas Dataframe with columns = 'value','nrow','ncol'
    # with the purpose of indexing each pixel with old row and column indexes
    
    coeff_matrix_df_dict = {}
    indep_matrix_df_dict = {}
    
    j = 0
    for nrow in range(row):
        for ncol in range(col):        
            # Ingnoring NoData value
            #if fraction_map[:,nrow,ncol].mean() == -9999:
            if fraction_map[:,nrow,ncol].mean() == -9999 or indepent_matrix[nrow,ncol].mean() == 0 or emis_matrix[nrow,ncol].mean() == 0:
                print(indepent_matrix[nrow,ncol].mean())
                pass
            else:
                print(indepent_matrix[nrow,ncol].mean())
                for i in range(8):
                    coeff_matrix_df_dict[j] = {'index':indices_map[:,nrow,ncol][i],
                                                                'class':i,
                                                              'value_fraction':fraction_map[:,nrow,ncol][i], # value of fraction fj
                                                              'nrow':nrow,
                                                              'ncol':ncol,
                                                              'indep_value':indepent_matrix[nrow,ncol],
                                                              'value_emis':list(df_emis_repre["Emis"].values)[i],
                                                              'value_emis_sum':emis_matrix[nrow,ncol],
                                                              'out_value':np.nan,
                                                              'cloud_mask':cloud_mask[nrow,ncol]}
                    
                    indep_matrix_df_dict[j] = {'index':indices_map[:,nrow,ncol][i],
                                                              'class':i,
                                                              'value_lst':indepent_matrix[nrow,ncol],
                                                              'value_emis_sum':emis_matrix[nrow,ncol],
                                                              'nrow':nrow,
                                                              'ncol':ncol}
                    
                    j = j + 1
            print(nrow,ncol)
                   
    coeff_matrix_df = pd.DataFrame.from_dict(coeff_matrix_df_dict,'index')
    indep_matrix_df = pd.DataFrame.from_dict(indep_matrix_df_dict,'index')
    
    print('Length of coeff_nmtrix_df : '+str(len(coeff_matrix_df)))

    # New version: from_dict function
    
    
        
    coeff_df = []
    indep_df = []
    out_value = []
    out_value_list = []

    # Testing with jumping kernel windows, e.g. it is not neccessary to moving every 1 pixel but instead of moving every 
    # 4 pixels. Doing so would speed up much time calculation especially when we consider the whole NYC domain.   
    
    #coeff_df_matrix_list = []
    #indep_df_matrix_list = []
    
    for kernel in kernel_list: # Looping over the kernel list for testing on different kernel size
        nrow = -moving_pixel # Starting from nrow index -movingpixe, so it will make up with moving 4 pixels
        count = 0

        while nrow < row:
            nrow = nrow + moving_pixel

            
            ncol = -moving_pixel # Starting from nrow index -movingpixel
            while ncol < col:
                ncol = ncol + moving_pixel
                
                
                # Applying linear algebra function for each kernel window:
                # Can consider parallel from this step for each kernel
                    
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
                if len(coeff_df) < 9*8: # As we consider 8 elements of land cover class 
                    pass
                else:
                    #coeff_df_matrix_list.append(coeff_df)
                    #indep_df_matrix_list.append(indep_df)
                    
                    
                    # Test .apply method: 03.28.2020
                    if radiance is True:
                        # Coefficient matrix values 
                        coeff_df_element = coeff_df.groupby('index')['value_fraction'].apply(lambda x: x.values).to_numpy()
                    
                        # Independent values
                        indep_df_element = np.array(list(map(lambda x:pow(x,4),
                                                     indep_df.groupby('index')["value_lst"].apply(lambda x: x.values[0]).to_numpy())))
                    
                    else:
                        coeff_df_grp = coeff_df.groupby('index')
                        # LST^4:
                        lst4 = np.array(list(map(lambda x:pow(x,4),
                                                     indep_df.groupby('index')["value_lst"].apply(lambda x: x.values[0]).to_numpy())))
                        # emissivity:
                        emis_sum = indep_df.groupby('index')["value_emis_sum"].apply(lambda x: x.values[0]).to_numpy()
                            
                        # Independent values: Element-wise multiplication 
                        indep_df_element = [a * b for a, b in zip(lst4,emis_sum)]

                        # Coefficiene matrix values: fraction i * emis i 
                        coeff_df_element = list(coeff_df_grp["value_fraction"].apply(lambda x: x.values) * coeff_df_grp["value_emis"].apply(lambda x: x.values))
                        
                    # Applying Least-square solutions with bounds or without bounds for Li
                    # Linera Algebra equations:

                
                    # Applying optimze function: Testing with Scipy package 
                    if bounds is not None:
                        res = lsq_linear(coeff_df_element,np.array(indep_df_element).reshape(len(indep_df_element),)
                                        , bounds=bounds)
                    else:
                        res = lsq_linear(coeff_df_element,np.array(indep_df_element).reshape(len(indep_df_element),))
                    
                    
                    
                        
                    # 0.01196 seconds for 72 rows (3x3 kernel)
                    
                    
                    
                   
                    # New Version: 03.28.2020
                    # Adding values of x to column 'out_value':
                    if radiance is True:
                        # Solution: x = 4sqrt(x)

                        coeff_df['out_value'] = np.array([res.x**(1/4)]*int(len(coeff_df)/8)).flatten()
                
                    else:


                        coeff_df['out_value'] = np.array([res.x**(1/4)]*int(len(coeff_df)/8)).flatten()
       
 
                    
                 
                    # Adding count column contains the order of kernel window
                    coeff_df["count"] = count + 1
                    
                    # Adding type colum such as radiance or temperature
                    coeff_df["type"] = _type
 
                    
                    # Append new dataframe to the existing dataframe
                    out_value.append(coeff_df)
                    
                    
                count = count + 1
            print(nrow,ncol,kernel)
        out_value_list.append(pd.concat(out_value))
            
    
    end_time = time.time() - start_time
    print('Processing time'+str(end_time))
    
    # Save data to pickle
#    path_output = '\\uahdata\rhome\py_code\aes509\data\staten_island_70m'
#    out_value_list[0].to_pickle(path_output+'out_value')
    
    
    
    # Test new version: 03.28.2020 (Staten Island 490 m )
    # Time: 8.951 seconds 
    # Test old version:
    # Time: 31.64 seconds
    
    
    # Test new version: 03.28.2020 (Staten Island 70 m )
    # Time: 395.76682 seconds 
    # Test old version:
    # Time:  seconds
    return out_value_list, end_time

def read_out_value_list(path, kernel_list, moving_pixel):
    '''
    Function to read multiple files of out_value_list for each case 
    
    Example:
        
                
    # Define list of kernel for testing 
    kernel_list = [25,30,50,70,90]
    moving_pixel = [10,15,25,25,45]
    
    
    # Linux:
    path = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/'
    
    
    out_value_list = read_out_value_list(path)
    
    # Window:
    path = '//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/'
    out_value_list = read_out_value_list(path)
        
    
    '''
    import pandas as pd
    import numpy as np
    import os
    import multiprocessing as mp
    
    
    
    filenames = []
    foldernames = []
    folder = os.listdir(path)
    
    #Looping over kenel_list and moving_pixel list to 
    # find the folder has the interested kernel and moving values 
    # And then append all files in the folder into a new list called filenames 
    for i in range(len(kernel_list)):     
        for item in folder:
            if 'kernel' in item and 'moving' in item:
                split = item.split('kernel_')[1].split('_moving_')
                if split[0] == str(kernel_list[i]) and split[1] == str(moving_pixel[i]):
                    foldernames.append(item)
                    filenames.append(os.listdir(path+'/'+item+'/')) # Append all filenames in that item folder
                else:
                    pass
            
    
            
    # Define variable out_value_list_all
    out_value_list = []  
    print(len(filenames))        
    for i in range(len(filenames)):
        df = []
        
        print('Reading for '+str(kernel_list[i])+' and Moving step of '+str(moving_pixel[i]) )
        
        for f in filenames[i]:
    
            df.append(reduce_mem_usage(pd.read_csv(path+foldernames[i]+'/'+f)))
        
        out_value_list.append(pd.concat(df))
                
               
      
            
    
    
    return out_value_list

def groupby_outvalue(out_value,row,col,kernel,path_fig):
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
        
    out_value_modi = out_value_modi_list[0]
    del out_value_modi_list[0]
    
    
     
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
    import os
    import numpy as np
    sns.set(style="whitegrid")
    
    
    out_value_grp = out_value.groupby(['nrow','ncol','class']) # Groupying by nrow, ncol 
    
    # Create a new dataframe to hold new groupyed and aggreated by mean values
    out_value_modi = out_value_grp.mean().reset_index()
    
    # Convert to np.float64
    out_value_modi['out_value'] = out_value_modi['out_value'].astype(dtype=np.float64)
    
    # Plotting violionplot to compare the distribution of end member temperature 
    fig, ax = plt.subplots(figsize=(30,15))
    g = sns.violinplot(x="class",y="out_value",data=out_value_modi,width=5)
    g.set_xticklabels(['Tree Canopy','Grass/Shrubs','Bare Soil',
                    'Water', 'Building','Road','Other Impervious',
                    'Railways'],rotation=25)
     

    plt.tick_params(axis='y',labelsize=25)
    plt.tick_params(axis="x", labelsize=25)          
    plt.xlabel('Land Cover Class',size=30,weight='bold') 
    plt.ylabel('Temperature (K)',size=30,weight='bold')     
    plt.ylim(280,312)
    plt.yticks(np.arange(280,312,2))
    
    try:
        os.mkdir(path_fig+'distribution/')
        
    except:
        pass
    
    
    fig.savefig(path_fig + 'distribution/all_member_distro')
    


    return out_value_modi




def canopy_temp(out_value_modi,tick_min,tick_max,tick_interval,bin_range,frac_index,frac_name,kernel,moving,path_fig,date,canopy_temp=False,only_end=False):
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
    
    path_fig = '/nas/rhome/tvo/py_code/aes509/data/whole_nyc_70m/fig/'
    
    out_value_modi_member_list = []
    for i in range(len(kernel_list)): 
        
        out_value_modi_member = canopy_temp(out_value_modi_list[i],
        tick_min=290,tick_max=304,tick_interval=0.1,frac_index=0,
        frac_name="Canopy",kernel=kernel_list[i],moving=moving_pixel[i],path_fig)
        
        out_value_modi_mem_list.append(out_value_modi_member)
    
    # Mapping fraction of land cover
    out_value_modi_emis_member = canopy_temp(out_value_modi,tick_min=0,tick_max=1,tick_interval=0.1,frac_index=0,frac_name="Canopy",
    canopy_temp=True)
    '''
    import cmocean  
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    # Define levels of ticks and tick intervals
    tick_min=tick_min
    tick_max=tick_max
    levels = MaxNLocator(nbins=(tick_max - tick_min)/tick_interval).tick_values(tick_min, tick_max)
    cmap = plt.get_cmap(cmocean.cm.thermal)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    
    # Get value only for canopy (fraction i = 0)
    out_value_modi_emis_member = out_value_modi.groupby('class').get_group(frac_index).groupby(('nrow','ncol')).agg({"out_value":"mean",
                                                            "value_fraction":"mean","index":"mean"}).reset_index()
    
    # Remove pixel with fraction = 0
    out_value_modi_emis_member_non_nan = out_value_modi_emis_member
    out_value_modi_emis_member_non_nan.loc[out_value_modi_emis_member_non_nan['value_fraction'] < bin_range,'out_value'] = np.nan
    
    # Original LST
    out_value_ori = out_value_modi.groupby(('nrow','ncol')).agg({'indep_value':'mean'}).reset_index()
    
    
    # Converting type of column from float16 to float64, as pandas pivot does 
    # not support float16 for reshaping the matrix 
    
    try:
        if canopy_temp is not False:
            df_pivot_result = out_value_modi_emis_member.pivot_table(index='nrow',
                                                       columns='ncol',
                                                       values='value_fraction')
        else: 
            df_pivot_result = out_value_modi_emis_member_non_nan.pivot_table(index='nrow',
                                                       columns='ncol',
                                                       values='out_value')
    
    except:
        out_value_modi_emis_member_non_nan['out_value'] = out_value_modi_emis_member_non_nan['out_value'].astype(np.float64)
        out_value_modi_emis_member['value_fraction'] = out_value_modi_emis_member['value_fraction'].astype(np.float64)
        
        out_value_ori['indep_value'] = out_value_ori['indep_value'].astype(np.float64)
        
        if canopy_temp is not False:
            df_pivot_result = out_value_modi_emis_member.pivot_table(index='nrow',
                                                       columns='ncol',
                                                       values='value_fraction')
        else: 
            df_pivot_result = out_value_modi_emis_member_non_nan.pivot_table(index='nrow',
                                                       columns='ncol',
                                                       values='out_value')


    # out_value_ori_non_nan.loc[(out_value_ori_non_nan['class'] == frac_index)&(out_value_ori_non_nan['value_fraction']==0),'out_value'] = np.nan

    # Df_pivot orignal LST
    df_pivot_ori_LST = out_value_ori.pivot_table(index='nrow',
                                                       columns='ncol',
                                                       values='indep_value')
    
    

    '''
    Plot 1 or 2 plot at the same time
    '''
    if only_end is not False:
        # Axes 2:
        fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(20,15))
        # Axes 1:
        plot_out_temp = ax1.pcolormesh(df_pivot_result,norm=norm, cmap=cmap)
        ax1.set_aspect(aspect='equal')
        plot_out_ori = ax2.pcolormesh(df_pivot_ori_LST,norm=norm, cmap=cmap)
        ax2.set_aspect(aspect='equal')
        
        # Modify colorbar 
        cbar_1 = fig.colorbar(plot_out_temp,ax=ax1, shrink=.3)
        cbar_2 = fig.colorbar(plot_out_ori, ax=ax2, shrink=.3)
        
        # access to cbar tick labels:
        cbar_1.ax.tick_params(labelsize=20) 
        cbar_2.ax.tick_params(labelsize=20) 
        
        if canopy_temp is not False:
            plt.title("Fraction of "+str(frac_name)+" Land Cover Class")
            cbar_1.set_label('Fraction of Land cover',size=30)
            fig.savefig(path_fig+frac_name+'_fraction_kernel_'+str(kernel)+'_moving_'+str(moving))
            
        elif canopy_temp is False:
            ax1.set_title(r"End member Temperature of "+str(frac_name)+" Land Cover Class"
                 "\n"
                 r"Kernel Window "+str(kernel) +' , Moving Step '+str(moving)+' '
                 "\n"
                 r"Date: "+str(date),size=20)
        
            ax2.set_title(r"Original ECOSTRESS LST"
                      "\n"
                      r"Date: "+str(date), size=20)
        
        
            cbar_1.set_label('LST (K)',size=30)
            cbar_2.set_label('LST (K)',size=30)
            fig.savefig(path_fig+frac_name+'_temp_kernel_'+str(kernel)+'_moving_'+str(moving))
        
    else:
        fig, ax1 = plt.subplots(figsize=(20,15))
        # Axes 1:
        plot_out_temp = ax1.pcolormesh(df_pivot_result,norm=norm, cmap=cmap)
        ax1.set_aspect(aspect='equal')
        
        # Modify colorbar 
        cbar_1 = fig.colorbar(plot_out_temp,ax=ax1, shrink=.3)
        
        
        # access to cbar tick labels:
        cbar_1.ax.tick_params(labelsize=20) 
        
        if canopy_temp is not False:
            plt.title("Fraction of "+str(frac_name)+" Land Cover Class")
            cbar_1.set_label('Fraction of Land cover',size=30)
            fig.savefig(path_fig+frac_name+'_fraction_kernel_'+str(kernel)+'_moving_'+str(moving))
            
        elif canopy_temp is False:
            ax1.set_title(r"End member Temperature of "+str(frac_name)+" Land Cover Class"
                 "\n"
                 r"Kernel Window "+str(kernel) +' , Moving Step '+str(moving)+' '
                 "\n"
                 r"Date: "+str(date),size=20)
            cbar_1.set_label('LST (K)',size=30)
            fig.savefig(path_fig+frac_name+'_temp_kernel_'+str(kernel)+'_moving_'+str(moving))
        
        
    
    

    
    
#    if canopy_temp is not False:
#        plt.title("Fraction of "+str(frac_name)+" Land Cover Class")
#        cbar_1.set_label('Fraction of Land cover',size=30)
#    else:
#        ax1.set_title(r"End member Temperature of "+str(frac_name)+" Land Cover Class"
#                 "\n"
#                 r"Kernel Window "+str(kernel) +' , Moving Step '+str(moving)+' '
#                 "\n"
#                 r"Date: "+str(date),size=20)
#        
#        ax2.set_title(r"Original ECOSTRESS LST"
#                      "\n"
#                      r"Date: "+str(date), size=20)
#        
#        
#        cbar_1.set_label('LST (K)',size=30)
#        cbar_2.set_label('LST (K)',size=30)
#    
#    plt.show(block=True)
#
#    if canopy_temp is not False:
#        fig.savefig(path_fig+frac_name+'_fraction_kernel_'+str(kernel)+'_moving_'+str(moving))
#    else:
#        fig.savefig(path_fig+frac_name+'_temp_kernel_'+str(kernel)+'_moving_'+str(moving))
#        
#    
    
    return out_value_modi_emis_member

def residual_map(df_1, df_2 , tick_min, tick_max, tick_interval,title_1,title_2,date,path_fig):
    '''
    Function to plot residuals map: difference btw two dataframe
    Input:
        df_1: dataframe 1
        df_2: dataframe 2
        
    Example:
        residual_map(out_value_modi_member_list[0], out_value_modi_member_list[-1], tick_min=290,
        tick_max = 304, tick_interval=0.1, title_1 = 'Kernel '+str(kernel_moving[0]),
        title_2 = 'Kernel '+str(kernel_moving[-1]), date= date_code[-1], path_fig=path_fig))
    '''
    import cmocean  
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    
    # Define levels of ticks and tick intervals
    tick_min=tick_min
    tick_max=tick_max
    levels = MaxNLocator(nbins=(tick_max - tick_min)/tick_interval).tick_values(tick_min, tick_max)
    cmap = plt.get_cmap(cmocean.cm.thermal)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    try:
        # Insert a new column from df_2 to df_1
        df_1.insert(len(df_1.columns),column='out_value_1',value=df_2['out_value'])
    
        # Insert a new column to df_1 which is the difference btw 2 columns 
        df_1.insert(len(df_1.columns),column='diff',value=df_1['out_value']-df_1['out_value_1'])

    except:
        pass
        
    # Create pivot table 
    df_pivot_result = df_1.pivot_table(index='nrow',
                                                       columns='ncol',
                                                       values='diff')

    
      




    
    fig, ax = plt.subplots()
    plot_out_temp = ax.pcolormesh(df_pivot_result,norm=norm, cmap=cmap)
    ax.set_aspect(aspect='equal')
    
    # Modify colorbar 
    cbar = fig.colorbar(plot_out_temp,ax=ax, shrink=.7)
    # access to cbar tick labels:
    cbar.ax.tick_params(labelsize=20) 
    
    

    plt.title(r"Residual Temperature of "+str(title_1)+''
              "\n"
              r" And "+str(title_2)+'')
    
    cbar.set_label('LST (K)',size=30)
    
    plt.show(block=True)
    
    try:
        os.mkdir(path_fig+'residual/')
        
    except:
        pass
    
    fig.savefig(path_fig+'residual/'+'residual_'+str(title_1)+'_and_'+str(title_2))
    
    
    
    
    
    return None


def test_sensitivity(out_value_modi_member_list,kernel_list,moving_pixel,class_name,path_fig,bounds_min,bounds_max):
    '''
    This function is for testing sensitivity of canopy temperature for 
    differenet kernel size
    
    Example:
    kernel_list = [10,15,20,25]
    test_sensitivity(out_value_modi_member_list,kernel_list=kernel_list,moving_pixel=moving_pixel,class_code=0,path_fig=path_fig)

    
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os
    
    
    
    kernel_moving = [[kernel,moving] for kernel,moving in zip(kernel_list,moving_pixel)]
    # Plotting canopy temperature for each kernel size, recommended to create a boxplot
    fig, ax = plt.subplots(figsize=(30,15))
    sns.violinplot(data=out_value_modi_member_list['out_value'],ax=ax)
    ax.set_xticklabels(kernel_moving)
    
    
    
    ax.set_xlabel('[Kernel Window, Moving pixel]',size=30,weight='bold')
    ax.set_ylabel(class_name + ' Temperature (K)', size=30, weight='bold')
    plt.tick_params(axis='y',labelsize=30)
    plt.tick_params(axis='x',labelsize=30)
    
    ax.set_ylim(bounds_min,bounds_max)
    
    plt.show(block=True)
    
    try:
        os.mkdir(path_fig+'sensitivity/')
        
    except:
        pass
    
    fig.savefig(path_fig+'sensitivity/'+class_name+str(kernel_moving))
    

        
        
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
        
    return None
def cut_value_frac_range(out_value_modi,frac_name,color,fraction_range,kernel,moving,path_fig):
        # Define a range for fraction 
    '''
    Example:

out_value_modi_member_cut_list = []  
for i in range(len(kernel_list)):     
    out_value_modi_cut = cut_value_frac_range(out_value_modi_member_list[i],frac_name="Canopy",color="green",
        fraction_range = np.arange(0,1.1,0.05),kernel=kernel_list[i],moving=moving_pixel[i],path_fig=path_fig)
    
    out_value_modi_member_cut_list.append(out_value_modi_cut)
    '''
        
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import os
    
    if fraction_range is None:
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
        
    # out_value_modi_element = out_value_modi.loc[out_value_modi["class"]==frac_index]
    
    
    fig, ax = plt.subplots(figsize=(30,15))
    sns.pointplot(x="frac_index",y="out_value",data=out_value_modi,color=color)
    
    plt.xlabel("Fraction of "+str(frac_name),size=30)
    plt.ylabel("End member Temperature of "+str(frac_name),size=30)
    
    
    plt.tick_params(axis="x",labelsize=30,rotation=45)
    plt.tick_params(axis="y",labelsize=30,rotation=45)
    
    try:
        os.mkdir(path_fig+'fraction/')
    except:
        pass
    
    
    fig.savefig(path_fig+'fraction/'+'canopy_temp_fraction_'+str(kernel)+'_moving_'+str(moving))
    
    return out_value_modi       

def extract_out_temp(out_value_modi,kernel,path_fig,tick_max, tick_min,emissivity=False,tick=False):
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
    import numpy as np
    
    
    # Convert to np.float64
    out_value_modi = out_value_modi.astype(dtype=np.float64)
    
    # Insert new column contains final output temperature by multipy "out_value" * "fraction"
    out_value_modi["out_temp"]=out_value_modi["out_value"]*out_value_modi["value_fraction"]
    
    
    # Group the current dataframe by nrow and ncol columns
    out_value_modi_grp = out_value_modi.groupby(("nrow","ncol"))
    
    # Aggregate by sum of "out_temp" column and mean of "indep_value" column
    out_value_modi_grp_agg = out_value_modi_grp.agg({"out_temp":"sum",
                                                     "indep_value":"mean"}).reset_index()
                    
                    
    # Insert new column "resi_temp" by substrating "indep_value"-"out_temp" columns
    out_value_modi_grp_agg["resi_temp"] = out_value_modi_grp_agg["indep_value"] - out_value_modi_grp_agg["out_temp"]

                                                         
    # Convert pandas Dataframe to pandas pivot_table and plot using pcolormesh 
    # Generating color scale for plotting colormesh

    import cmocean  
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
        
    if tick is False: 
        tick_max = out_value_modi_grp_agg[['indep_value','out_temp']].max().max()
        tick_min = out_value_modi_grp_agg[['indep_value','out_temp']].min().min()
        levels = MaxNLocator(nbins=(tick_max - tick_min)/0.2).tick_values(tick_min, tick_max)
        
    elif tick is True:
        tick_max = tick_max
        tick_min = tick_min
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
    ax1.set_title('Original ECOSTRESS LST')
    ax1.set_aspect(aspect='equal') # Set aspect of colormesh so has equal square pixel

    plot_out_temp = ax2.pcolormesh(df_pivot_result,norm=norm, cmap=cmap)
    ax2.set_title('Output Temperature with Kernel = '+str(kernel))
    ax2.set_aspect(aspect='equal')
    
    # Adding colorbar
    fig.colorbar(plot_ori_temp,ax=ax1, shrink=.7)
    fig.colorbar(plot_out_temp,ax=ax2, shrink=.7)
    
    fig.savefig(path_fig + 'output_LST_kernel_'+str(kernel))
    
    # Plotting residuals map 
    fig2, ax3 = plt.subplots()
    plot_resi_temp = ax3.pcolormesh(df_pivot_resi, cmap=cm.twilight_shifted)
    ax3.set_title('Residual Temperature Map'+str(kernel))
    ax3.set_aspect(aspect='equal')
    fig2.colorbar(plot_resi_temp, ax=ax3)   
    
    fig2.savefig(path_fig + 'resi_LST_kernel_'+str(kernel))
    
                    
    return out_value_modi_grp_agg   

