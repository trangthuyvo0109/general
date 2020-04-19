# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:26:23 2019

@author: tvo
"""

'''
***Requirements:
    python > 3
    gdal, numpy, matplotlib, cartopy library
    
**The script has two main fuctions:
    read_hdf(file): read .hdf file and 
                    returns in values as an 2-dimensional array
                    metadata of the dataset
                    number of bands of the dataset

This script is written to read the .hdf file using gdal library

- gdal can be easily installed using: 
    conda install gdal

- library used to plot the dataset:
    matploblib
    cartopy (conda install cartopy or pip install cartopy)
    
    
- HDF files contain variables, and each variable has attributes that describe the variable.

- There are also "global attributes" that describes the overall file

- Here we are testing for reading the HLS dataset: Harmonized Landsat-8 and Sentinel-1 dataset


- The structure of the dataset is a 3-dimensional dataset, which is categorized by several bands and matrix of the values
of corresponding band. 

###To read the metadata of the dataset:
    dataset= gdal.Open(file)
    meta_dataset = dataset.GetMetadata()
 

###Get subdatasets of the dataset
    subset = dataset.GetSubDatasets()


    
'''

import numpy as np
import gdal
import matplotlib.pyplot as plt




def read_hdf(file):
    '''
    Fuction read_hdf to read the hdf file
    Varibale file is the link to the file need to be read
    file = "path/to/file/HLS.L30.T16SCC.2019001.v1.4.hdf"
    '''
    import gdal
    import numpy.ma
    
    #Open the file
    dataset = gdal.Open(file)
    
    ##Read metadata of the dataset"
    meta_dataset = dataset.GetMetadata()
    
    ##Get subdatasets:
    sub_dataset = dataset.GetSubDatasets()
    
    band_dataset = []
    data_array = []

    ##Read the dataset as an array
    for i in range(len(sub_dataset)):
        
        ##Extracting the title of each band dataset
        band = sub_dataset[i][0]
        
        ##Read values of each band set and return as an 2-dimensioonal array
        data = gdal.Open(band)
        data_arr = data.ReadAsArray()
        
        ##Append the title of each band into a list 
        band_dataset.append(band)
        
        

        ## Scale the data based on the Scale factor and masked dataset based on fill_value
        # For Landsat-8, band 9 and band 10 (Thermal Infrared)
        #Mask: if any pixel smaller than 0 (fill_value = -10000), defined as a mask
        if file.split("/HLS.")[-1][:3] == "L30" and (i == 8 or i == 9):
            data_arr = np.ma.masked_where(data_arr < -9999, data_arr)
            data_arr = data_arr * -0.01  #Except band 10 & 11 of Landsat-8, scalefactor= -0.01
        
        #Keep original value of QA layer, do not scale or mask it
        elif i == len(sub_dataset) - 1:                   
            data_arr = data_arr
            
        #The rest of band scale with factor 0.0001 and mask with fill_value=-1000
        else:
            data_arr = np.ma.masked_where(data_arr < -999, data_arr)
            data_arr = data_arr * 0.0001 
                
        
        #Conver fill_value to nan: 
        if i != len(sub_dataset) - 1:
            data_fill = np.ma.filled(data_arr, fill_value=0)
            data_fill[data_fill==0] = ["nan"]
        
        else:
            #(except for QA layer)
            data_fill = data_arr
            
        ##Append the values of each band into a list 
        data_array.append(data_fill)
    
        
        
    
    return data_array, band_dataset, meta_dataset

def qa_code(data, metadata, file):
    """
    Function to decode the QA layer and create separate cloud layer and add new bands to current dataset
        return values: 0 or 1 (0: no, 1:yes)
        circus: band 8 (1)
        cloud: band 9 (1)
        adjacent cloud: band 10 (1)
        cloud shadow: band 11 (1)
        
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy.ma
    from skimage.exposure import equalize_hist
    
    
    qa_layer = data[-1] ##Get the QA layer product from the dataset
    qa_decode = np.vectorize(np.binary_repr)(qa_layer,width=8) ##Decode the qa layer which is stored as unsigned 8-bit integer
    
    ##Create empty layers for each of the cloud layer
    circus = np.empty_like(qa_layer, dtype=np.int16)
    cloud = np.empty_like(qa_layer, dtype=np.int16)
    adj_could = np.empty_like(qa_layer, dtype=np.int16)
    sha_cloud = np.empty_like(qa_layer, dtype=np.int16)
    
    
    
    ##Loop over the empty cloud layers and add the corresponding bit values to each layers
    for row in range(len(qa_decode)):
        for col in range(len(qa_decode)):
            circus[row][col] = int(qa_decode[row][col][-1])
            cloud[row][col] = int(qa_decode[row][col][-2])
            adj_could[row][col] = int(qa_decode[row][col][-3])
            sha_cloud[row][col] = int(qa_decode[row][col][-4])
    

    
    ##The adjection cloud layer is not availabel in S30 and S10 product
    if file.split("/HLS.")[-1][:1] == 'S':
        cloud_set = np.dstack((circus, cloud, sha_cloud))
        band = ["circus","cloud","cloud shadow"]
        print("There is no adjection cloud layer in this product")      
        
    elif file.split("/HLS.")[-1][:1] == 'L':
        ##Stack existing cloud layers to a 3-D arrays with order of : circus, cloud, adj_cloud, sha_cloud
        cloud_set = np.dstack((circus, cloud, adj_could, sha_cloud))  
        band = ["circus","cloud","adjection cloud","cloud shadow"]
        print("There is adjection cloud layer in this product")    
        
        
    #Plotting cloud layers
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)
    axes = (ax1, ax2, ax3, ax4)
    for i in range(cloud_set.shape[-1]):
        map_cl = axes[i].imshow(cloud_set[:,:,i],cmap="gray")
        axes[i].set_title(band[i], size=50, weight="bold")
    
    fig.colorbar(map_cl)

        
    
    
    '''
    in case there is any cloudy pixels, create a mask 
    to ignore the calculation
    '''
    # Aggregate cloud_set value using max numpy, which means whenever it is cloudy pixel (no matter which clouds,
    # it dhould be classified as cloudy pixels)
    cloud_set = cloud_set.max(axis=2)
    
    #if any pixel larger than 0 (means cloudy pixels), defined as a mask
    cloud_set = np.ma.masked_where(cloud_set > 0, cloud_set)

    
    # Replace 0 (non-cloud) to 1 for later multiplication with current dataset
    cloud_set[cloud_set==0.0] = 1.0
    
    #Replace 1 (cloud) to nan (ignore calculation) for later multiplication with current dataset
    cloud_set_fill = np.ma.filled(cloud_set, fill_value=-9999).astype(np.float64)
    cloud_set_fill[cloud_set_fill==-9999] = "nan"
    
    
    # Multiply the cloud mask with the current dataset to return the value with a mask of cloud
    data[:-1] = data[:-1] * cloud_set_fill
    
    
    # Plot new dataset with a cloud mask, testing with band 1
    fig, ax = plt.subplots()
    map_cloud = ax.imshow(data[0])
    fig.colorbar(map_cloud)
    ax.set_title("Band 1 with cloud mask", size=50, weight="bold")
    
    plt.tick_params(axis="x", labelsize=35)
    plt.tick_params(axis="y", labelsize=35)  
    plt.show()          
        
        
    
    
    return data
    
    

def plot_hdf(data, metadata, bands=None, RGB=None):
    '''
    Documentation
    function plot_hdf to plot the array values have readed from .hdf file as an map 
    Variables:
        data: the dataset 
        bands: the index of bands need to be use. e.g. Band01: 0, Band02: 1,..... 
        
        
    Optional parameters:
        RGB: color composite. The input values must be as an array [band_red, band_green, band_blue]
        RGB=[0,2,5] ==> Red: Band 01; Green: Band 03; Blue: Band 06
        
    Scale and equalized function is applied to better represent the dataset
    Scale: 0.0001
    Fill Value: -1000
    '''
    import matplotlib.pyplot as plt
    from skimage.exposure import equalize_hist
    
    ##Set title depends on type of the product
    if file.split("/HLS.")[-1][:3] == "L30":
        title = str(metadata["LANDSAT_PRODUCT_ID"])
    else:
        title = str(metadata["PRODUCT_URI"])
    
    # Define mask for equalizing histogram, mask where valid value(not nan)
    mask_invalid = np.ma.masked_invalid(data)
    mask_invalid_invert = np.invert(mask_invalid.mask)
    
    
    
    if bands is not None:
        
        fig, ax = plt.subplots()
        data_hist = equalize_hist(data[bands], mask=mask_invalid_invert[bands])
        map_data = ax.imshow(data_hist)
        fig.colorbar(map_data)
        ax.set_title(title + "\n Band: " + str(bands+1)
                     ,size=25, weight="bold")
    
    
    
    if RGB is not None:
        rgb = {'r': data[RGB[0]-1], 'g':data[RGB[1]-1], 'b': data[RGB[2]-1]}
        rgb_mask = {'r': mask_invalid_invert[RGB[0]-1], 'g':mask_invalid_invert[RGB[1]-1], 'b': mask_invalid_invert[RGB[2]-1]}
        stack = lambda arr: np.dstack(list(arr.values()))
        data = stack(rgb)
        mask = stack(rgb_mask)
        
        ##Create a mask layer over the RGB dataset for equal histogram 
              
        fig, ax = plt.subplots()
        data_hist = equalize_hist(data, mask=mask)
        map_data = ax.imshow(data_hist)
        ax.set_title(title + "\n RGB: " + str(RGB)
                     ,size=25, weight="bold")
        fig.colorbar(map_data)


    plt.tick_params(axis="x", labelsize=35)
    plt.tick_params(axis="y", labelsize=35)
    
    return None


if __name__ == '__main__':
    #file = 'C:/Users/tvo/Downloads/HLS.L30.T16SCC.2019001.v1.4.hdf'
    #file = 'C:/Users/tvo/Documents/HLS.S30.T16SFD.2018040.v1.4.hdf'
    
    ##Read .hdf file
    data, bands, metadata = read_hdf(file)
    
    ##Set the boundary of the map based on the values in metadata
    bound_lons = [float(metadata["ULX"]) , 
                  float(metadata["ULX"]) + float(metadata["NCOLS"]) * float(metadata["SPATIAL_RESOLUTION"])]
    
    bound_lats = [float(metadata['ULY']) , 
                  float(metadata["ULY"]) - float(metadata["NROWS"]) * float(metadata["SPATIAL_RESOLUTION"])]
    
    ##Select the band that user would like to create a
    rgb = {'r': data[1], 'g':data[4], 'b': data[3]}
    
    #Plot the data Band01
    plot_hdf(data, metadata, bands=0)
#    
    ##Plot the data RGB = 1,2,3
    plot_hdf(data, metadata, RGB=[4,3,2])
#    

#    
    
    ##Create cloud band 
    data_cloud= qa_code(data, metadata, file)
    


    
    pass

