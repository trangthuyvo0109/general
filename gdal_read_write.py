# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:06:02 2020

@author: tvo
"""

import os
import gdal





def cloud_mask(file, outFile):
    '''
    Function to decode cloud mask image and assign cloudy pixels nan values
    
    
    Example:
        file = 'C:/Users/tvo/Documents/urban_project/ECO_large_04092020_cloudmask/ECO2CLD.001_SDS_CloudMask_doy2019264063150_aid0001.tif'
        outFile = file.split('/ECO2')[0] + '/decoced/'+file.split('cloudmask/')[-1].split('.tif')[0]+'_decoded.tif'
        
        cloud_mask(file,outFile)
    
    '''
    import gdal
    import numpy as np
    
    
    dataset = gdal.Open(file)
    
    data = dataset.ReadAsArray()
    
    # Get row,col
    [cols, rows] = data.shape
    data_min = data.min()
    data_max = data.max()
    data_mean = int(data.mean())
    

    
    
    # Maske out nan values
    data = np.ma.masked_where(data == -999, data)
    
    # decode the cloud mask 8-bit
    qa_decode = np.vectorize(np.binary_repr)(data,width=8)
    
    
    
    # Create a new image wiith cloud decoded, bit 
    cloud = np.empty_like(data, dtype=np.int16)
    for row in range(len(qa_decode)):
        for col in range(len(qa_decode[row])):
            
            if qa_decode[row][col] != np.nan:
                cloud[row][col] = int(qa_decode[row][col][-2])
            else:
                pass
    
    
    
    # Driver
    driver = gdal.GetDriverByName('GTiff')
    
    # Output
    outdata = driver.Create(outFile, rows, cols, 1,  gdal.GDT_UInt16)
    
    # Set geotransform same as input
    outdata.SetGeoTransform(dataset.GetGeoTransform())
    
    # Set projection same as input
    outdata.SetProjection(dataset.GetProjection())
    
    # Write image
    outdata.GetRasterBand(1).WriteArray(cloud)
    
    # Set NoData
    outdata.GetRasterBand(1).SetNoDataValue(-999)
    
    
    # Save to disk
    outdata.FlushCache()
    
    outdata = None
    dataset = None
    
    
    
    
    
    import matplotlib.pyplot as plt
    plt.imshow(cloud)
    
    
    
    
    return cloud_mask   