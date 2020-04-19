# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:51:35 2020

@author: tvo


THIS VERSION IS MORE SUITABLE FOR CALCULATING EACH KERNEL 
WITHOUT AGGREGATING OVERLAPPING PIXELS

"""


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
        
        #
        import time
        start_time = time.time() 
        out_value_list = cal_linear_alg_lstsq(df_landcover_concat, df_lst, df_emiswb, df_emis_repre,
    row, col, kernel_list=[3], _type="Emis", 
    bounds=(290**4,310**4), moving_pixel = 2,  radiance=False)
        print(time.time() - start_time)
        
        49.32850956916809

    
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
    

    
    # Old version:
    # Create an empty pandas Dataframe with columns = 'value','nrow','ncol'
    # with the purpose of indexing each pixel with old row and column indexes
    coeff_matrix_df = pd.DataFrame(data=[], columns=['index','value_fraction','nrow','ncol','class','indep_value','value_emis','value_emis_sum','out_value'])
    indep_matrix_df = pd.DataFrame(data=[], columns=['index','value_lst','value_emis_sum','nrow','ncol'])
    
    
    
    # Looping over the whole domain to assign new coeff_matrix and independt value dataframes
    for nrow in range(row):
        for ncol in range(col):        
            # Ingnoring NoData value
            if fraction_map[:,nrow,ncol].mean() == -9999:
                pass
            else:
                for j in range(8):
                
                
                    coeff_matrix_df = coeff_matrix_df.append({'index':indices_map[:,nrow,ncol][0],
                                                          'value_fraction':fraction_map[:,nrow,ncol][j], # value of fraction fj
                                                          'nrow':nrow,
                                                          'ncol':ncol,
                                                          'class':j,
                                                          'indep_value':indepent_matrix[nrow,ncol],
                                                          'value_emis':list(df_emis_repre["Emis"].values)[j],
                                                          'value_emis_sum':emis_matrix[nrow,ncol]},ignore_index=True) # value of emiss ei
        
                    indep_matrix_df = indep_matrix_df.append({'index':indices_map[:,nrow,ncol][0],
                                                          'value_lst':indepent_matrix[nrow,ncol],
                                                          'value_emis_sum':emis_matrix[nrow,ncol],
                                                          'nrow':nrow,
                                                          'ncol':ncol},ignore_index=True)
            print(nrow,ncol)


    coeff_df = []
    indep_df = []
    # out_value = []
    out_value = pd.DataFrame(data=[], 
                             columns=['index','class','value','nrow','ncol','indep_value','out_value',"type"])
    out_value_list = []

    # Testing with jumping kernel windows, e.g. it is not neccessary to moving every 1 pixel but instead of moving every 
    # 4 pixels. Doing so would speed up much time calculation especially when we consider the whole NYC domain.   
    
    coeff_df_matrix_list = []
    indep_df_matrix_list = []
    
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
                    coeff_df_matrix_list.append(coeff_df)
                    indep_df_matrix_list.append(indep_df)
                    
                    # Older version:
                    ### Processing each kernel window
                    ### Consider parallel comptutation? As each kernel is indepent 
                    
                    index_list = list(coeff_df.groupby("index").groups.keys())
                    
                    indep_df_list  = []
                    coeff_df_list = []
                    for item in index_list:
                        
                        indep_df_loc = indep_df.loc[indep_df["index"]==item]
                        coeff_df_loc = coeff_df.loc[coeff_df["index"]==item]
                    
                    
                        # Applying with radience formula instead of direct LST function: 
                        # LST^4 = sum(fraction*Temp^4) + residuals
                        if radiance is True: 
                        # Independent values
                            indep_df_element = list(map(lambda x:pow(x,4),
                                                     np.array(indep_df_loc["value_lst"])))
                        
                            # Coefficient matrix values 
                            coeff_df_element = np.array(coeff_df_loc["value_fraction"])
                    
                        # Applying with Emis formula instead of direct LST function:emis_sum * LST^4 = sum(e*fraction*Temp^4) + residuals
                        else:

                            # LST^4:
                            lst4 = list(map(lambda x:pow(x,4),
                                            np.array(indep_df_loc["value_lst"])))
                            # emissivity:
                            emis_sum = indep_df_loc["value_emis_sum"].tolist()
                            
                            # Element-wise multiplication 
                            indep_df_element = [a * b for a, b in zip(lst4,emis_sum)][0]

                            # fraction i * emis i 
                            coeff_df_element = list(coeff_df_loc["value_fraction"] * coeff_df_loc["value_emis"])
                            
                        indep_df_list.append(indep_df_element)
                        coeff_df_list.append(coeff_df_element)
                            
                        


                    
                    # Applying function:
                    x, sum_res, rank, s = la.lstsq(coeff_df_list,
                                       indep_df_list)
                
                    # Applying optimze function: Testing with Scipy package 
                    if bounds is not None:
                        res = lsq_linear(coeff_df_list,
                                        np.array(indep_df_list).reshape(len(indep_df_list),)
                                        , bounds=bounds)
                    else:
                        res = lsq_linear(coeff_df_list,
                                     np.array(indep_df_list).reshape(len(indep_df_list),))
                    
                                
                    ### End processing each kernel. At this step, we should be able to extract 
                    # End member temperature for each kernel
                    # Time: 0.0239 seconds for 72 rows (3x3 kernel)
                    

                    
                    # Old version: 
                    for j in range(8):
                        if radiance is True:
                            # Solution: x = 4sqrt(x)
                            coeff_df.loc[coeff_df["class"]==j,"out_value"] = res.x[j]**(1/4)

                
                        else:

                            coeff_df.loc[coeff_df["class"]==j,"out_value"] = res.x[j]**(1/4)

                     # Old version
                     
                     
                    
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
                #print(count)
            print(nrow,ncol,kernel)
        out_value_list.append(out_value)
            
    
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
    
    out_value_grp = out_value.groupby(['nrow','ncol','class']) # Groupying by nrow, ncol 
    
    # Create a new dataframe to hold new groupyed and aggreated by mean values
    out_value_modi = out_value_grp.mean().reset_index()
    

    
    
    
    
                
    return out_value_modi


def cut_value_frac_range(out_value_modi,frac_index,frac_name,color,fraction_range):
        # Define a range for fraction 
    '''
    Example:
        cut_value_frac_range(out_value_modi,frac_index=0,frac_name="Canopy",color="green",
        fraction_range = np.arange(0,1.1,0,0005)))
    '''
        
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    
    if fraction_range is None:
        fraction_range = np.arange(0,1.1,0.005)
    
    
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
    
    
    return None       




def duplicate_kernel(out_value_modi_emis_non_agg,frac_index,tick_max,tick_min):
    '''
    Example:
        duplicate_kernel(out_value_modi_emis_non_agg,frac_index=0,tick_max=300,tick_min=297)
    '''
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import cmocean  
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import Normalize
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    
    
    
    # Define levels of ticks 
    levels = MaxNLocator(nbins=(tick_max - tick_min)/0.2).tick_values(tick_min, tick_max)
    cmap = plt.get_cmap(cmocean.cm.thermal)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    
    # Group the dataframe by kernel index 
    out_value_modi_emis_non_agg_grp = out_value_modi_emis_non_agg.groupby("count")
    
    index_kernel = list(out_value_modi_emis_non_agg_grp.groups.keys())
    
    fig, ax = plt.subplots()
    
    ax.set_xlim(out_value_modi_emis_non_agg.ncol.min(),
                out_value_modi_emis_non_agg.ncol.max())
    
    ax.set_ylim(out_value_modi_emis_non_agg.nrow.min(),
                out_value_modi_emis_non_agg.nrow.max())
    
    # Create a list of random colors 
    colors=list(matplotlib.colors.ColorConverter.colors.keys())

    for i in range(len(index_kernel)):
        
        # Creating pivot table of the colormesh
        df_pivot = out_value_modi_emis_non_agg_grp.get_group(index_kernel[i]).groupby("class").get_group(frac_index).pivot_table(index="nrow",
                                                            columns="ncol",
                                                            values="out_value")
        
        
        # Correct xticks and yticks
        xmin = df_pivot.index.min()
        xmax = df_pivot.index.max() + 1
        dx = 1
        ymin = df_pivot.columns.min()
        ymax = df_pivot.columns.max() + 1
        dy = 1
        
        # Corrected xticks and yticks
        x_corrected, y_corrected = np.meshgrid(np.arange(xmin,xmax+dx,dx)-
                                               dx/2.,np.arange(ymin,ymax+dy,dy)-dy/2.)
        
        print(x_corrected,y_corrected)
        

        
        ax.pcolormesh(x_corrected,
                      y_corrected,
                      df_pivot.values,
                      norm=norm,
                      cmap=cmap,
                      edgecolors=colors[i],

                    
                      alpha=0.6)
        ax.set_aspect(aspect="equal")
        
        
#        plt.axis([x_corrected.min(),x_corrected.max(),
#                  y_corrected.min(),y_corrected.max()])
        plt.xticks(np.arange(xmin,xmax,dx))
        plt.yticks(np.arange(ymin,ymax,dy))
        
        
        print(df_pivot)
    #plt.grid("True")
        

    import matplotlib.pyplot as plt
    import matplotlib
    kernel_compare = []
    
    
    row_index = list(out_value_modi_emis_non_agg.groupby("nrow").groups.keys())[20:23]
    fig, ax = plt.subplots()
    
    for j in range(len(row_index)):
        
        index = list(out_value_modi_emis_non_agg.groupby("nrow").get_group(row_index[j]).groupby("class").get_group(0).
                     groupby("count").groups.keys())
        
    
        print(index)
        colors=list(matplotlib.colors.ColorConverter.colors.keys())
        markers = ["o","*",'+']
        sizes = [400,200,400]
        alpha = [0.8,0.8,1]
        
        x_value = []
        y_value = []
        for i in range(len(index)):
            index_col = list(out_value_modi_emis_non_agg.groupby("nrow").get_group(row_index[j]).groupby("class").get_group(0).
                     groupby("count").get_group(index[i]).groupby("ncol").groups.keys())
            print(index_col)
            
            x_value.append(index_col[1])
            y_value.append(out_value_modi_emis_non_agg.groupby("nrow").get_group(row_index[j]).groupby("class").get_group(0).
                     groupby("count").get_group(index[i])["out_value"].mean())
            
        plt.scatter(x_value,
                 y_value,
                     marker=markers[j],
                     s=sizes[j],
                     alpha=alpha[j],
                     color=colors[j*100],
                     label="Nrow = "+str(row_index[j]))
        
        kernel_compare.append(y_value)
               
        
    plt.legend(prop={"size":30})
    plt.xlabel("Ncol",size=30)
    plt.ylabel("Canopy End member Temperature",size=30)
    plt.tick_params(axis="x",labelsize=30)
    plt.tick_params(axis="y",labelsize=30)
    
    
    lineStart = min(min(kernel_compare))
    lineEnd = max(max(kernel_compare)) 
    
    fig, ax1 = plt.subplots()
    ax1.scatter(kernel_compare[0],kernel_compare[-1][:len(kernel_compare[0])])
    ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
    ax1.set_xlabel("Row "+str(row_index[0]),size=30)
    ax1.set_ylabel("Row " + str(row_index[-1]),size=30)
    plt.tick_params(axis="x",labelsize=30)
    plt.tick_params(axis="y",labelsize=30)
    
    
    
    
    
    
    
    
    
    
    



















 
    