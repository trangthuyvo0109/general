# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:12:28 2019

@author: tvo
"""

'''
This fuction is applied to convert the original dataset achieved from geospatial
analysis to dictionary 
Original dataset

gridcode    Class       FID_AOI_grid        fraction


Destinated dataset

FIO_AOI_grid    Tree    Grass   ....    Water (Ocean)   fraction


######Be careful: the FID_Aoi_grid need to be dissolved by sum of fraction of land cover before
running this code. Otherwise it will return None values due to data duplication. 


'''

def convert_data(file):
    import pandas as pd
    
    ##Read original dataset using pandas dataframe
    data = pd.read_csv(file)
    

    
    ##Get unique values of column Class
    #column_order = data.Class.unique().tolist()
    column_order = ['Tree Canopy', 'Grass\\Shrubs', 'Bare Soil','Water'
                    ,'Buildings', 'Roads', 'Other Impervious', 'Railroads',
                    'Water (Ocean)']
    grid_FID = data.FID_AOI_3_
    mean_lst = data._LSTmean
    nrow = data.nrow
    ncol = data.ncol
    lats = data.lats
    lons = data.lons
    

    """
    Convert the original datafram with an aggration of sum fraction of each land cover class
    """
    

    
    #stacked.groupby('Category').agg({'Price':['count','sum','mean','std']})
    
    data = data.groupby(["gridcode","Class","FID_AOI_3_"]).agg(
            {"SUM_fracti":"sum"})
    
    data = data.reset_index().sort_values(by="FID_AOI_3_")

    
    data_pivot = data.pivot(index="FID_AOI_3_", columns = "Class", values = "SUM_fracti").reset_index()
    
    data_pivot = data_pivot.fillna(0)
    ##Fill nan values by 0.0 
    
    column_order.insert(0,"FID_AOI_3_")
    ##Insert new column FID to current column order list
 
    
    ##reset index for column of the pivot table by column order list
    data_pivot = data_pivot.reindex(column_order, axis=1)

    
    """
    New dataframe containes the rest of the variables 
    """

    
    ##Reset indexes of the rest of variables to the grid
    ext_data = pd.DataFrame(index = grid_FID)
    mean_lst.index = grid_FID
    nrow.index = grid_FID
    ncol.index = grid_FID
    lats.index = grid_FID
    lons.index = grid_FID
    ext_data.insert(len(ext_data.columns), column="MEAN_LST", value=mean_lst)
    ext_data.insert(len(ext_data.columns), column="nrow", value=nrow)
    ext_data.insert(len(ext_data.columns), column="ncol", value=ncol)
    ext_data.insert(len(ext_data.columns), column="lats", value=lats)
    ext_data.insert(len(ext_data.columns), column="lons", value=lons)
    
    
    ext_data = ext_data.reindex().groupby("FID_AOI_3_").agg("mean")
    print(ext_data)

    

    

    '''
    Merge two dataframes
    '''
    data_pivot = data_pivot.merge(ext_data, left_index = True, right_index = True)
    print(data_pivot)
    
    

    
    

    return data_pivot





if __name__ == '__main__':
    
    #file = 'C:/Users/Trang Vo/Downloads/AOI_3_broklyn_disolv_LST_test.txt'
    file = 'C:/Users/tvo/Desktop/Work/Workspace_1/AOI_3_1031/AOI_3_broklyn_disolv_test_LST.csv'
    
    
    
    data = convert_data(file)
    ##Convert dataframe to csv
    data_to_csv = data.to_csv("C:/Users/tvo/Desktop/Work/Workspace_1/AOI_3_1031/AOI_3_broklyn_disolv_test_LST_converted.csv")
    
    
    
    
    
    
    pass