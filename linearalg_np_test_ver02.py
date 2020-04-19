# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:48:47 2020

@author: tvo


Testing with Parallel comptuting 
This version is more to use only for paralleling purpose, such as running on matrix server
It is not recommened to apply for other purpose, such as aggregating algorithm
"""


def cal_linear_alg_lstsq(df_landcover, df_lst, df_emiswb, df_emis_repre, row, col, kernel_list,moving_pixel=4):
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

        # Emissivity function
        coeff_df_matrix_list, indep_df_matrix_list = cal_linear_alg_lstsq(df_landcover_concat, df_lst, df_emiswb, df_emis_repre,  
            row, col, kernel_list=[25], moving_pixel = 5)
        
 

    
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
            if fraction_map[:,nrow,ncol].mean() == -9999:
                pass
            else:
                
                for i in range(8):
                    coeff_matrix_df_dict[j] = {'index':indices_map[:,nrow,ncol][i],
                    'class':i,
                                                              'value_fraction':fraction_map[:,nrow,ncol][i], # value of fraction fj
                                                              'nrow':nrow,
                                                              'ncol':ncol,
                                                              'indep_value':indepent_matrix[nrow,ncol],
                                                              'value_emis':list(df_emis_repre["Emis"].values)[i],
                                                              'value_emis_sum':emis_matrix[nrow,ncol],
                                                              'out_value':np.nan}
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
    # New version: from_dict function
    
    
        
    coeff_df = []
    indep_df = []

    # Testing with jumping kernel windows, e.g. it is not neccessary to moving every 1 pixel but instead of moving every 
    # 4 pixels. Doing so would speed up much time calculation especially when we consider the whole NYC domain.   
    
    coeff_df_matrix_list = []
    indep_df_matrix_list = []
    
    start_time = time.time()

    for kernel in kernel_list: # Looping over the kernel list for testing on different kernel size
        nrow = -moving_pixel # Starting from nrow index -movingpixe, so it will make up with moving 4 pixels
        
        count = 0 # Set counter of kernel

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
                    # Insert column of kernel index 
                    coeff_df.insert(len(coeff_df.columns),column='count',value=count)
                    
                    # Append current coeff_df and indep_df 
                    coeff_df_matrix_list.append(coeff_df)
                    indep_df_matrix_list.append(indep_df)
            
                    count = count + 1
    
    print(time.time() - start_time)
    return coeff_df_matrix_list, indep_df_matrix_list



def least_sq(coeff_df, indep_df, radiance, bounds, _type):
    '''
    Function to apply Least-squared solutions for Linear Algebra equation.
    The func is to apply for each kernel 
    
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
    
    

   
    # New Version: 03.28.2020
    # Adding values of x to column 'out_value':
    if radiance is True:
        # Solution: x = 4sqrt(x)

        coeff_df['out_value'] = np.array([res.x**(1/4)]*int(len(coeff_df)/8)).flatten()

    else:


        coeff_df['out_value'] = np.array([res.x**(1/4)]*int(len(coeff_df)/8)).flatten()
   
 
    

    
    # Adding type colum such as radiance or temperature
    coeff_df.insert(len(coeff_df.columns),column='type',value=_type)

            
    
    end_time = time.time() - start_time
    print('Processing time'+str(end_time))
    
    
    return coeff_df

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
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def collect_result(result):
    global results
    
    results.append(result)



if __name__ == "__main__":
    '''
    To apply this script, be sure we have:
        df_landcover_concat
        df_emiswb
        df_lst
        df_emis_repre
        
        
    '''
        
    
    from multiprocessing import Pool
    import multiprocessing as mp
    import pandas as pd
    import time
    
    # Windows: 
    path = '//uahdata/rhome/py_code/aes509/data/whole_nyc_70m/'
    
    # Step 0: Load required files
    df_landcover_concat = pd.read_pickle(path+'df_landcover_concat')
    df_emiswb = pd.read_pickle(path+'df_emiswb')
    df_lst = pd.read_pickle(path+'df_lst')
    df_emis_repre = pd.read_pickle('//uahdata/rhome/py_code/aes509/data/nyc_representative_value/df_emis_repre')

    # Step 1: Splitting the data by kernel
    # Staten Island
#    row = 303
#    col = 243
    
    # Whole NYC
    row = 884
    col = 786
 

    # Emissivity function
    coeff_df_matrix_list, indep_df_matrix_list = cal_linear_alg_lstsq(df_landcover_concat, df_lst, df_emiswb, df_emis_repre,  
        row, col, kernel_list=[50], moving_pixel = 15)
    
    

    kernel_list=[50]
    moving_pixel = 15
    
    data = [[coeff_df_matrix, indep_df_matrix] for coeff_df_matrix, indep_df_matrix in zip(coeff_df_matrix_list,indep_df_matrix_list)]

    # Export data:
    print(pd.concat(coeff_df_matrix_list).info)
    print('Reduced Memory Usage: '+str(reduce_mem_usage(pd.concat(coeff_df_matrix_list)).info()))
    reduce_mem_usage(pd.concat(coeff_df_matrix_list)).to_pickle(path+'coeff_df_matrix_list_kernel_'+str(kernel_list[0])+'_moving_'+str(moving_pixel))
    reduce_mem_usage(pd.concat(indep_df_matrix_list)).to_pickle(path+'indep_df_matrix_list_kernel_'+str(kernel_list[0])+'_moving_'+str(moving_pixel))

        
    # Step2: Applying parallelizing
    pool = mp.Pool(mp.cpu_count())
    
    results = []
    out_value_list = []
    start_time = time.time()
    print('Starting to parallelizing......')
    # Use loop to parallelize
    for coeff_df, indep_df in data:
        pool.apply_async(least_sq, 
                         args=(coeff_df, indep_df, False, (290**4,320**4), 'Emis'), 
                         callback=collect_result)
        
    
    # Close Pool and let all processes complete
    pool.close()
    pool.join()# postpones the execution of next line of code until all processes in the queue are done.

    # Sort results 
    out_value_list = pd.concat(results)
    
    print('Finishing parallelizing in '+str(time.time() - start_time)+' seconds')
#    



                    