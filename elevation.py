# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:52:11 2019

@author: tvo
"""
'''
This module contains functions for reading in GridFloat data from the National Elevation
Dataset (NED) and plotting the data.
Routine Listing
-------
read_ned_gridfloat : Read in GridFloat files
read_header : Read the header file associated with GridFloat files
read_flt : Read the *.flt file that contains elevation data
plot_elevation : Create a contour plot of elevation data
Notes
-------
This module is really intended for GridFloat files downloaded from the NED archive.
You can get these file here <https://viewer.nationalmap.gov/basic/ > .
The GridFloat format specification is here <https://www.loc.gov/preservation/digital/
formats/fdd/fdd000422.shtml>
'''

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace



def plot_elevation(lons, lats, elevation, cmap="Reds_r", levels = None, inset_bounds = None):
    """
    Create a contour plot, usually for elevation data.
    Given longitudes, latitudes, and elevation data, create a filled contour of the data. 
    The data should be regularly gridded. A colorbar for the plot
    is also created. Optionally, you can make an inset of a region of interest.
    
    Parameters
    ----------
    lons : array
    An N-element array of longitudes of the data
    
    lats : array
    An M-element array of latitudes of the data
    
    elevation : array
    An NxM-element array of the elevation of the data. The first
    dimension corresponds to the lons , and the second dimension
    corresponds to the lats .
    
    cmap : str, optional
    A named colormap name from Matplotlib. Default is to use Reds_r.
    
    levels : array, optional
    The levels to be contoured. By default, 10 levels spanning the
    min/max of the elevation will be used.
    
    inset : boolean, optional
    If True, an inset plot is also created.
    
    inset_bounds: array, optional
    The inset_bounds is a 4 element array-like value (e.g., an array, list, or tuple) 
    in the order of [lon_min, lon_max, lat_min, lat_max]. By default, None which means
    no inset_bounds has been set thus no inset will be created.
        
    
    Returns
    -------
    SimpleNamespace
        Returned value has several keys, some of which are optionally included.
            :contour: Matplotlib QuadContourSet
            :colorbar: Matplotlib Colorbar class
            :inset_bounds: Matplotlib QuadContourSet (if inset_bounds is requested)
            :polygon: Matplotlib Polygon (if inset_bounds is requested)
            The polygon of the inset.:lines:
            A list of Matplotlib CollectionPatch-es (if inset_bounds is requested)
            
    Examples
    -------
    >>> from scipy.stats import multivariate_normal
    >>> import numpy as np
    # Generate some lats and lons
    >>> lons = np.arange(100)
    >>> lats = np.arange(100)
    # Make these 2D "coordinate" arrays
    >>> lons2d, lats2d = np.meshgrid(lons, lats)
    # Fill in coordinates to find the 2D Gaussian PDF
    >>> coord = np.empty(lons2d.shape + (2,))
    >>> coord[:, :, 0] = lons2d
    >>> coord[:, :, 1] = lats2d
    >>> mean = [1, -2] # coordinates of Gaussian peak
    >>> cov = [[5.0, 0], [0.0, 20.0]] # covariance array - width of peak
    # Generate the elevation data
    >>> rv = multivariate_normal(mean, cov)
    >>> data = rv.pdf(coord)
    >>> data = data/np.max(data)*400 # scale the data
    >>> elv_plot = plot_elevation(lons, lats, data)
    # Pick some new levels
    >>> elv_plot2 = plot_elevation(lons, lats, data, levels = np.arange(0, 300, 50))
            
            
    
    """
    
    import matplotlib.colors as mplcol  
    from matplotlib.patches import Polygon, ConnectionPatch
    import numpy as np
   
    
    all_plots = SimpleNamespace() # all plots are added to this to be returned
    
    ##Pick levels for contour:
    if levels is None:
        levels = np.linspace(np.nanmin(elevation), np.nanmax(elevation), 10)
    
    
    # Build the color map, based on a Matplotlib colormap.
    # Get two extra colors for extending colors to the min _and_ max
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, levels.size+2))
    
    # The new color map for the levels
    cmap_mod = mplcol.ListedColormap(colors[1:-1, :])
    
    # Set the min/max (under/over) for the extended colorbar
    cmap_mod.set_over(colors[-1, :])
    cmap_mod.set_under(colors[0, :])
    
    # Pack these up into a dictionary for the contour plot,
    # and add extending contours to min and max
    contourProps = {'levels': levels, 'cmap': cmap_mod, 'extend': 'both'}
    
    ##Plot the elevation contour
    ##**contourProps: keyword extension
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    c_big = ax.contourf(lons, lats, elevation, **contourProps)
    all_plots.contour = c_big
    
    ##Make the colorbar
    cb = c_big.ax.figure.colorbar(c_big)
    all_plots.colorbar = cb
    

    
    ##Make an inset
    if inset_bounds is not None:        
        print("You have requested an inset: [lon_min, lon_max]=",inset_bounds[:2]," [lat_min, lat_max]=",inset_bounds[-2:])
        # Set the corners of the polygon
        #lat_poly = [34.65, 34.85]
        lat_poly = inset_bounds[-2:]
        #lon_poly = [-86.65, -86.45]
        lon_poly = inset_bounds[:2]
        # Define the vertices for the polygon
        lat_vert = [lat_poly[0], lat_poly[1], lat_poly[1], lat_poly[0]]
        lon_vert = [lon_poly[0], lon_poly[0], lon_poly[1], lon_poly[1]]
        lon_lat = np.column_stack((lon_vert, lat_vert))
        # Draw the polygon
        poly = Polygon(lon_lat)
        _null = c_big.ax.add_patch(poly)
        poly.set_facecolor('none')
        poly.set_edgecolor('black')
        all_plots.polygon = poly # add the polygon
        pass
  
        # Make the inset contour
        newax = c_big.ax.figure.add_axes([0, 0, 0.5, 0.25])
        c_inset = newax.contourf(lons, lats, elevation, **contourProps)
        all_plots.inset = c_inset # add the inset
        # Get the corners of the "big" contour
        ll, ul, lr, ur = c_big.ax.get_position().corners()
        # Set the width and height of the inset, and position it
#        inset_width = (inset_bounds[1] - inset_bounds[0])/3.0
#        inset_height = (inset_bounds[3] - inset_bounds[2])/3.0
        inset_width = 0.2
        inset_height = 0.15
        new_pos = [lr[0]-inset_width, lr[1], inset_width, inset_height]
        c_inset.ax.set_position(new_pos)
        # Next, "zoom" for the inset:
        _new_lim = c_inset.ax.set_xlim(lon_poly)
        _new_lim = c_inset.ax.set_ylim(lat_poly)
        # Modify ticks of the inset
        c_inset.ax.tick_params(labelbottom=False, labelleft=False)
        c_inset.ax.tick_params(bottom = False, left = False)
        
        # Finally, add the lines.
        # Get the lower left/upper right of the polygon and inset
        ll_poly = [lon_poly[0], lat_poly[0]]
        ur_poly = [lon_poly[1], lat_poly[1]]
        ll_inset, _, _, ur_inset = c_inset.ax.get_position().corners()
        # Now, add the lines
        patch_props = {'coordsA': 'data', 'axesA': c_big.ax,
        'coordsB': 'figure fraction', 'axesB': c_inset.ax}
        line1 = ConnectionPatch(xyA=ll_poly, xyB=ll_inset, **patch_props)
        _null = c_big.ax.add_patch(line1)
        line2 = ConnectionPatch(xyA=ur_poly, xyB=ur_inset, **patch_props)
        _null = c_big.ax.add_patch(line2)
        all_plots.lines = [line1, line2]

    
    return all_plots   
    
    pass



def read_header(file):
    """
    Documentation goes here
    This fuction read the header of file. 
    The extension of the header is usually .hdr
    """
    # We'll use a dictionary to hold header info:
    hdr = dict()
    with open(file, 'r') as f:
        for line in f:
            entity = line.split()
            print(entity)
            # Most of the values are numbers, so cast them.
            # The one that isn't, keep it as a string
            if 'byteorder' in entity[0]:
                val = entity[1]
            else:
                val = float(entity[1])
                hdr[entity[0]] = val
                pass
    return hdr


def read_flt(file):
    """
    Documentation goes here
    This function read the flt file and return the data
    into a 2-dimensional array. 
    """
    # Read the flt file - it contains 32 bit floats:
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data


##create module contains function
def read_ned_gridfloat(file):
    """
    Documentation goes here
    
    only works if gridfloat header file has x/yllcorner
    """
    import os.path
    import numpy as np
    from types import SimpleNamespace
#    import pdb; pdb.set_trace()
    
    #Find the filenames
    basename, ext = os.path.splitext(file)
    hdr_file = basename + '.hdr'
    flt_file = basename + '.flt'
    print(hdr_file)
    print(flt_file)
    
    
    
    ##Read in header
    hdr = read_header(hdr_file)
    ##Read in flt file
    data = read_flt(flt_file)
    
    # Change bad values to NaNs
    data[np.isclose(data, hdr['NODATA'])] = np.nan
    
    # We want a 2D array:
    data.shape = (int(hdr['nrows']), int(hdr['ncols']))
    
    # Rotate the data so it's oriented correctly
    data = np.flip(data, axis=0)
    
    # Find the lats and lons the data corresponds to. This
    # can be derived from some of the header attributes:
    lons = np.arange(hdr['ncols']) * hdr['cellsize'] + hdr['xllcorner']
    lats = np.arange(hdr['nrows']) * hdr['cellsize'] + hdr['yllcorner']
    
    # Pack everything up:
    return SimpleNamespace(lats = lats, lons = lons, elevation = data)
    
    

if __name__ == '__main__':
    import numpy as np
    file = '//uahdata/rhome/data/USGS_NED_1_n31w088_GridFloat-20191028T210717Z-001/USGS_NED_1_n31w088_GridFloat/usgs_ned_1_n31w088_gridfloat.flt'
    #file = r'//uahdata/rhome/data/USGS_NED_1_n31w088_GridFloat-20191028T210717Z-001/USGS_NED_1_n31w088_GridFloat/usgs_ned_1_n31w088_gridfloat.flt'
    data = read_ned_gridfloat(file)
    
    elv_plot = plot_elevation(data.lons, data.lats, data.elevation)
    elv_plot.contour.ax.figure.savefig("contour_test")
    elv_plot.contour.ax.figure.savefig(r"C:\Users\tvo\Desktop\Materials\ATS_509\ATS509_tvo\homework2_chap7\figure\contour_test")
    
    
    elv_plot_inset = plot_elevation(data.lons, data.lats, data.elevation, inset_bounds = [-87.8, -87.2, 30.4, 30.8])
    elv_plot_inset.contour.ax.figure.savefig("contour_test_inset")
    elv_plot_inset.contour.ax.figure.savefig(r"C:\Users\tvo\Desktop\Materials\ATS_509\ATS509_tvo\homework2_chap7\figure\contour_test_inset")
    
    
    file_2 = '//uahdata/rhome/data/USGS_NED_1_n38w108_GridFloat/usgs_ned_1_n38w108_gridfloat.flt'
    data_2 = read_ned_gridfloat(file_2)
    elv_plot_2 = plot_elevation(data_2.lons, data_2.lats, data_2.elevation)
    elv_plot_2.contour.ax.figure.savefig("contour_test_2")
    elv_plot_2.contour.ax.figure.savefig(r"C:\Users\tvo\Desktop\Materials\ATS_509\ATS509_tvo\homework2_chap7\figure\contour_test_2")
    
    
    offset= 0.4
    elv_plot_inset_2 = plot_elevation(data_2.lons, data_2.lats, data_2.elevation, 
                                      inset_bounds = [np.min(data_2.lons) + offset, np.max(data_2.lons) - offset,
                                                      np.min(data_2.lats) + offset, np.max(data_2.lats) - offset])
    elv_plot_inset_2.contour.ax.figure.savefig("contour_test_inset_2")
    elv_plot_inset_2.contour.ax.figure.savefig(r"C:\Users\tvo\Desktop\Materials\ATS_509\ATS509_tvo\homework2_chap7\figure\contour_test_inset_2")
    

    elv_plot_color_2 = plot_elevation(data_2.lons, data_2.lats, data_2.elevation, cmap = "terrain")
    elv_plot_color_2.contour.ax.figure.savefig("contour_test_colormap_2")
    elv_plot_color_2.contour.ax.figure.savefig(
            r"C:\Users\tvo\Desktop\Materials\ATS_509\ATS509_tvo\homework2_chap7\figure\contour_test_colormap_2")
    
    
    elv_plot_level_2 = plot_elevation(data_2.lons, data_2.lats, data_2.elevation, cmap = "terrain",
                                      levels = np.linspace(np.nanmin(data_2.elevation), np.nanmax(data_2.elevation), 100))
    elv_plot_level_2.contour.ax.figure.savefig("contour_test_level_2")
    elv_plot_level_2.contour.ax.figure.savefig(
            r"C:\Users\tvo\Desktop\Materials\ATS_509\ATS509_tvo\homework2_chap7\figure\contour_test_level_2")
    
    
    elv_plot_combine_2 = plot_elevation(data_2.lons, data_2.lats, data_2.elevation, cmap = "terrain",
                                      levels = np.linspace(np.nanmin(data_2.elevation), np.nanmax(data_2.elevation), 100),
                                      inset_bounds = [np.min(data_2.lons) + offset, np.max(data_2.lons) - offset,
                                                      np.min(data_2.lats) + offset, np.max(data_2.lats) - offset])
    elv_plot_combine_2.contour.ax.figure.savefig("contour_test_combine_2")
    elv_plot_combine_2.contour.ax.figure.savefig(
            r"C:\Users\tvo\Desktop\Materials\ATS_509\ATS509_tvo\homework2_chap7\figure\contour_test_combine_2")
    
    
    plt.close("all")
    
    
    
    
    
    
    
    pass





























