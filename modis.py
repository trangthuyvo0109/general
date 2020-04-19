# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:02:31 2019

@author: Trang Vo
"""

import numpy as np
import matplotlib.pyplot as plt

def read_modis1b(file, nointerp=False):
    """
    Documentation
    This function is to read modis dataset
    input file: file
    nointerp: default value is False. If user need to interpolate the lats/lons,
    to match the shape of the band, the define lats/lons values to interpolate
    
    """
    from netCDF4 import Dataset
    import numpy as np
    
    # Open the file
    nc = Dataset(file)
    #Define a variable to hold bands
    bands = dict()

    # Read in bands 1-2
    ev250 = nc.variables['EV_250_Aggr1km_RefSB']
    
    
    # Extract the data and apply scale/offset:
    for idx, _b in enumerate(np.arange(1, 3)):
        data = ev250[idx, :, :]
        bands[_b] = data*ev250.reflectance_scales[idx] + ev250.reflectance_offsets[idx]
    
    # Read in bands 3-7
    ev500 = nc.variables['EV_500_Aggr1km_RefSB']
    
    # Extract the data and apply scale/offset:
    for idx, _b in enumerate(np.arange(3, 8)):
        data = ev500[idx, :, :]
        bands[_b] = data*ev500.reflectance_scales[idx] + ev500.reflectance_offsets[idx]
    
    # Read in lats/lons
    lats = nc.variables['Latitude'][:]
    lons = nc.variables['Longitude'][:]
    
    if not nointerp:
        # Go interpolate the lats/lons to match the shape of the bands
        from scipy.ndimage import zoom
        factor = np.array(ev500[0, :, :].shape)/np.array(lats.shape)
        lats = zoom(lats, factor)
        lons = zoom(lons, factor)
    
    # Close the file
    nc.close()
    
    
    return bands, lats, lons


def scale_rgb(rgb, min_input=0.0, max_input=1.1):
    """
    Documentation goes here
    This function is applied to scale the image based on the input of RGB composite
    The input value should be a dict with keys r,g,b corresponding to Red, Green, Blue 
    
    Note to self: rgb is a dict with keys r,g,b (Red, Green, Blue)
    min_input, max_input: defined value range of scaling function.
    Default value: 
        min_input = 0.0
        max_input = 1.1
    """
    from skimage.exposure import equalize_hist
    
    _rgb = rgb.copy()
    
    # Build the mask:
    mask = rgb['r'].mask | rgb['g'].mask | rgb['b'].mask
    mask = ~mask # Flip the mask so True means keep the data
    
    for key, val in _rgb.items():  
        #Scale the data
        # The values in the scaling are based on info here:
        # http://gis-lab.info/docs/modis_true_color.pdf
        # Note: this will effectively make the masked array not-masked.
        data = np.interp(val, (min_input, max_input), (0, 255))
        
        #Apply the mask
        data = data*mask
        
        #Histogram-equalize the data
        data = equalize_hist(data)
        
        # The previous function takes our 8 bit numbers and returns
        # a float array in the range 0-1. Make sure we have 8 bit numbers
        # for the image.
        data = np.uint8(data*255)
        
        _rgb[key] = data
    
    
    return _rgb


def plot_modis1b_true_color(r, g, b, lats, lons, lat_map=None, lon_map=None,
                            binx=15000, biny=15000, noscale=False):
    '''
    This function is applied to plot image with true color composite
    Besides, the function also allows user to resample the image 
    
    noscale: if user would like to plot scaled image >> noscale=False
    if not: noscale=True
    '''
    # Make note that r,g,b should be masked arrays
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import pyresample
    
    # Pack the channels into a dict to simplify the following code
    rgb = {'r': r, 'g': g, 'b': b}
    
    if noscale is False:
        # Scale the data
        rgb = scale_rgb(rgb)
    else:
        # Keep original value 
        rgb = rgb
    
    
    # Pick a projection for the map
    map_proj = ccrs.Mercator()
    
    ll_proj = ccrs.PlateCarree()
    # Now, we need to resample the data to a regular grid.
    
    # Get the extent of the data, in the map coordinate system
    if lon_map is None:
        lon_map = lons.min(), lons.max()
    
    if lat_map is None:
        lat_map = lats.min(), lats.max()
        
    # Convert this to map coordinates
    ll_mapped = map_proj.transform_points(ll_proj,
                                          np.array(lon_map),
                                          np.array(lat_map))
        
    # Use this to the extent (bounds) of the map in projection coordinates
    x_map = ll_mapped[:, 0]
    y_map = ll_mapped[:, 1]
    
    map_extent = (*x_map, *y_map)
    
    # Define the target grid
    grid = pyresample.area_config.create_area_def('null', map_proj.proj4_params,
                                                  units='meters',
                                                  resolution=(binx, biny),
                                                  area_extent=map_extent)
    
    
    # Get the swath "grid" (aka source "grid")
    swath = pyresample.SwathDefinition(lons, lats)
    # Resample each band
    # To do so, need a radius of influence:
    radius_influence = 5000 # Author note: Possibly make this a parameter
    
    rgb_grid = dict()
    for key, val in rgb.items():
        data = pyresample.kd_tree.resample_nearest(swath, val, grid,
                                                   radius_of_influence=radius_influence)
        rgb_grid[key] = data
    
    # For an image, we need a mxnx3 array:
    rgb_grid = np.dstack(list(rgb_grid.values()))
    
    # Finally, make the map
    fig, ax = plt.subplots(subplot_kw={'projection': map_proj})
    
    # Set the extent, using the variables associated with the projection
    ax.set_extent(map_extent, map_proj)
    
    # Set the image, using the variables associated with the gridded data
    img = ax.imshow(rgb_grid, transform=grid.to_cartopy_crs(),
                    extent=grid.to_cartopy_crs().bounds, origin='upper')
    
    # Add a little map bling
    coast = ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=':')
    
    
    
    
    return img, coast, gl

if __name__ == '__main__':

    #file = '/path/to/file/MOD021KM.A2005240.1700.005.2010159164217.hdf'
    file = '//uahdata/rhome/py_code/aes509/data/MOD021KM.A2005240.1700.061.2017185042936.hdf'
    
    #Read function
    bands, lats, lons = read_modis1b(file) # slight change in variable names
    rgb = dict(r=bands[1], g=bands[4], b=bands[3])
    
    #Scale function
    rgb_scl = scale_rgb(rgb, min_input=0.0, max_input=0.5)
    # stack = lambda arr: np.dstack(list(arr.values()))
    
    # fig, ax = plt.subplots()
    # ax.imshow(stack(rgb_scl))
    
    #Plot function
    img, coastlines, grid = plot_modis1b_true_color(rgb_scl['r'], rgb_scl['g'],
                                                    rgb_scl['b'], lats, lons
                                                    , noscale=True)
    

    pass
    
































