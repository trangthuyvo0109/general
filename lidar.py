# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:08:37 2019

@author: tvo
"""
import gdal
import rasterio as rio
import rasterio.mask

##Rasterio
from rasterio.plot import show
from rasterio.enums import Resampling
from rasterio.windows import Window

##Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors

##Cartopy 
import cartopy.crs as ccrs


import fiona
import skimage.transform as sk
import numpy as np
import rasterio.shutil



def raster_resample(file_orig,file):
    '''
    Open Raster and create a window to read the dataset (in case the dataset is too large)
    file_orig: directory to file before resampling
    file: directory to file after resampling 
    '''
    
    '''
    Reading non-resampled dataset with a zoom
    '''
    with rasterio.open(file_orig) as dataset:    
        window = Window.from_slices((int(dataset.height/2),int(dataset.height/2+4000)),
                                (int(dataset.width/2),int(dataset.width/2+4000)))
        data_window = dataset.read(1, window=window)
        
        
    """
    Reading resampled dataset, also within a zoom window for comparision
    """    
    with rasterio.open(file) as dataset_res:
        window_res = Window.from_slices((int(dataset_res.height/2),int(dataset_res.height/2+200)),
                                        (int(dataset_res.width/2),int(dataset_res.width/2+200)))
        data_window_res = dataset_res.read(1, window=window_res)
        
        ##Resampled data without windows
        data_res = dataset_res.read(1, masked=True)
        
    '''
    Plotting the non-resampled and resampled dataset
    '''
    ##Defining colormap 
    cmap = ListedColormap(["green",
                           "darkgreen",
                           "brown",
                           "blue",
                           "pink",
                           "gray",
                           "yellow",
                           "red",
                           "darkblue"
                           ])
    
    ##Add a legend for labels
    legend_labels = {"green":"Tree Canopy",
                     "darkgreen":"Grass/Shrub",
                     "brown":"Bare Soil",
                     "blue":"Water",
                     "pink":"Buildings/Roofs",
                     "gray":"Roads",
                     "yellow":"Other Impervious",
                     "red":"Railways",
                     "darkblue":"Ocean"
            }
    ##Define a normalization from values -> colors 
    norm = colors.BoundaryNorm(np.arange(np.min(data_window),np.max(data_window)+1,1),np.max(data_window)-np.min(data_window)+1)
    
    patches = [Patch(color=color, label=label)
    for color, label in legend_labels.items()]
    
    ##Define projection
    proj = ccrs.epsg(32618)
    
    
    '''
    Plot to compare original and resampled dataset within a window
    '''
    fig, (ax1, ax2) = plt.subplots(ncols=2, subplot_kw=dict(projection=proj))  
    show(data_window, ax=ax1, norm=norm, cmap=cmap)
    show(data_window_res, ax=ax2, norm=norm, cmap=cmap)
    

    ##Generating colorbar 
    ax1.legend(handles=patches, prop={"size":15})
    ax2.legend(handles=patches, prop={"size":15})
    
    ##Adding gridlines
#    ax1.gridlines(draw_labels=True, linestyle=":")
#    ax2.gridlines(draw_labels=True, linestyle=":")
    
    ##Change title
    ax1.set_title("Original LandCover NYC with resolution of "+ str(dataset.res[0])+" foot", size=25, weight="bold")
    ax2.set_title("Resampled LandCover NYC with resolution of "+ str(round(dataset_res.res[0])) + " meter", size=25, weight="bold")
    
    '''
    Plot resampled data with full screen 
    '''
    dataset_res_extent = np.asarray(dataset_res.bounds)[[0,2,1,3]]
    fig, ax3 = plt.subplots()
    show(data_res, ax=ax3, norm=norm, cmap=cmap, extent=dataset_res_extent)  
    ax3.set_title("Land Cover Map of NYC")
    plt.tick_params(axis="x",size=30)
    plt.show()


    
    

    #fig.colorbar(map_window)
    return dataset, data_window, dataset_res, data_window_res, data_res
    
def shapefile_mask(rasterfile,shapefile):
    import fiona
    import rasterio
    import rasterio.mask

    ###Read Shapefile
    with fiona.open(shapefile) as shapefile:
        shapes=[feature["geometry"] for feature in shapefile]
    
    ###Read Raster and clip by mask shapes
    with rasterio.open(rasterfile) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    
    ##Update and write to a new raster
    out_meta.update({"driver": "GTiff",
                     "height":out_image.shape[1],
                     "width":out_image.shape[2],
                     "transform": out_transform
            })
    out_file = "D:/Land_Cover_2017_6-inch/Land_Cover/NYC_2017_LiDAR_LandCover_10_proj_wgs84_Brokolyn.tif"
    with rasterio.open(out_file,
                       "w", **out_meta) as dest:
        dest.write(out_image)
    
    '''
    Color map, norm, extent
    '''
    ##Defining colormap 
    cmap = ListedColormap(["green",
                           "darkgreen",
                           "brown",
                           "blue",
                           "pink",
                           "gray",
                           "yellow",
                           "red",
                           "white"
                           ])
    
    ##Add a legend for labels
    legend_labels = {"green":"Tree Canopy",
                     "darkgreen":"Grass/Shrub",
                     "brown":"Bare Soil",
                     "blue":"Water",
                     "pink":"Buildings/Roofs",
                     "gray":"Roads",
                     "yellow":"Other Impervious",
                     "red":"Railways",
                     "white":"Ocean"
            }
    ##Define a normalization from values -> colors 
    norm = colors.BoundaryNorm(np.arange(np.min(out_image),np.max(out_image)+1,1),
                               np.max(out_image)-np.min(out_image)+1)
    
    ##Extent
    out_extent = np.asarray(src.bounds)[[0,2,1,3]]
    
    ##Patches
    patches = [Patch(color=color, label=label) for color, label in legend_labels.items()]

    '''
    Color map, norm, extent
    '''


    ##Plot masked raster
    
    fig, ax = plt.subplots()
    
    show(out_image, ax=ax, norm=norm, cmap=cmap, extent=out_extent) ##show masked image
    ax.legend(handles=patches, prop={"size":15}) ##Generating legend
    print(src.meta["crs"])
    ax.set_xticks(np.arange(out_extent[0],out_extent[1],5000))
    ax.set_xlabel("Longtitude (m)", size = 30, weight="bold")
    ax.set_yticks(np.arange(out_extent[2],out_extent[3],5000))
    ax.set_ylabel("Latitude (m)", size=30, weight="bold")
    ax.set_title("Land Cover Map Brokolyn, NYC, resolution: " + str(round(src.res[0])) + " meter \n" + 
             "Coordinate System: " + str(src.meta["crs"]), size=30, weight="bold")
    
    plt.grid(linestyle=":",color="gray")
    plt.tick_params(axis="x", labelsize=20, rotation=90)
    plt.tick_params(axis="y", labelsize=20)
    

    
    plt.show()
    
    return out_image, out_transform, out_meta, out_file
    
    
def reproject_shp(file, epsg):
    '''
    Update: 11.21.2019
    This code only works on linux system, there is a bug issue with crs between fiona and whatever
    ...
    
    '''
    import geopandas as gpd
    import earthpy as et
    from fiona.crs import from_epsg
    #from pyproj import CRS
    
    ##Read shape file
    shape = gpd.read_file(file)
    
    ##copy to a new file
    data_proj = shape.copy()
    
    
    #data_proj["geometry"] = data_proj["geometry"].to_crs({'init': 'epsg=32618')
    #crs = CRS('epsg:32618')
    #data_proj = data_proj.to_crs(cc)
    ##Reproject the geometries by replacing the values with projected ones
    data_proj["geometry"] = data_proj["geometry"].to_crs(epsg)
    
    # Determine the CRS of the GeoDataFrame
    data_proj.crs = from_epsg(epsg)
    
    print(data_proj.crs)

    ##Save the outfile
    #out_file = r'C:/Users/tvo/Desktop/Work/Workspace_1/AOI_4_1115/AOI_3_brokolyn_wgs84.shp'
    ##Save the disk
    data_proj.to_file("AOI_3_brokolyn_wgs84.shp")
    
    
    
    #fig, (ax1, ax2) = plt.subplots(ncols=2)
    ##Plot the original data
    ##shape.plot(ax=ax1)
    
    
    #Plot the projected data
    ##data_proj.plot(ax=ax2)
    
    
    return shape, data_proj
    


def fishnet(outputGridfbn, rasterfile):
    import os, sys
    import ogr
    from osgeo import osr
    from math import ceil
    import rasterio


   
    src = rasterio.open(rasterfile)
        
    # convert sys.argv to float\
    xmin = float(src.bounds[0])
    xmax = float(src.bounds[2])
    ymin = float(src.bounds[1])
    ymax = float(src.bounds[3])
    print(xmin, xmax, ymin, ymax)
    
    #grid heights, widths
    gridWidth = round(float(src.res[0]*100))
    gridHeigh = round(float(src.res[1]*100))
    
    #get rows, cols
    #rows = ceil(float(src.shape[0])/100)
    rows = ceil((ymax-ymin)/gridHeigh)
    print(rows)
    #cols = ceil(float(src.shape[1])/100)
    cols = ceil((xmax-xmin)/gridWidth)
    print(cols)
    
    #start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax - gridHeigh
    
    # Define coordinates
    dest_srs = osr.SpatialReference()
    dest_srs.ImportFromEPSG(32618)
    
    #create output file
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outputGridfbn):
        os.remove(outputGridfbn)
    outDataSource = outDriver.CreateDataSource(outputGridfbn)
    outLayer = outDataSource.CreateLayer(outputGridfbn, dest_srs, geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    

    #Create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1
        
        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0
        
        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            
            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            outFeature = None
            
            # new envelope for next poly
            ringYtop = ringYtop - gridHeigh
            ringYbottom = ringYbottom - gridHeigh
            
        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth
    
    # Save and close DataSources
    outDataSource = None
    
    #fig, ax = plt.subplots()
    #outDataSource.plot(ax=ax)

    
    return outputGridfbn
        
def fishnet_geometry(fishnet, boundary):
    """
    This function is used to do geometry calculation between fishnet and boundary layer: 
        intersection
        symetric difference
    ...
    """
    import geopandas as gpd
    
    #Define fishnet as poly1
    poly1 = gpd.read_file(fishnet)
    
    #Define bouundary layer as poly2
    poly2 = gpd.read_file(boundary)
    
    #Intersection between fishnet and boundary
    poly_intersect = gpd.overlay(poly1, poly2, how="intersection")
    
    # Buffer from boundary layer with unit of 50 meter 
    poly2_buffer = poly2.copy() # Copy a new layer from boundary layer as a buffer layer
    poly2_buffer["geometry"] = poly2_buffer["geometry"].buffer(50) # Create buffer with distance 50 meter
    
    # Symetric difference between buffer and old boundary layer 
    poly2_buffer_sym = gpd.overlay(poly2, poly2_buffer, how="symmetric_difference")
    
    # Intersect between interseteced fishnet and poly2_buffer_sym
    poly2_buffer_sym_inter = gpd.overlay(poly_intersect, poly2_buffer_sym, how="intersection")
    
    
    
    
    # Plot
    fig, ax = plt.subplots()
    #poly_intersect.plot(ax=ax, color="white", edgecolor="black", alpha=0.5)
    
    
    
    
    return poly_intersect, poly2_buffer, poly2_buffer_sym, poly2_buffer_sym_inter
        
def plot_raster_vector(rasterfile, shapefile):
    from descartes import PolygonPatch
    from fiona import collection
    import geopandas as gpd
    import cartopy.crs as ccrs
    
    # Open raster
    with rasterio.open(rasterfile) as src:
        out_image = src.read(1, masked=True)
       
        
    # Open shapefile
    shapes = gpd.read_file(shapefile)
    

        
    # Plot raster and shapefile
    '''
    Color map, norm, extent
    '''
    ##Defining colormap 
    cmap = ListedColormap(["green",
                           "darkgreen",
                           "brown",
                           "blue",
                           "pink",
                           "gray",
                           "yellow",
                           "red",
                           "white"
                           ])
    
    ##Add a legend for labels
    legend_labels = {"green":"Tree Canopy",
                     "darkgreen":"Grass/Shrub",
                     "brown":"Bare Soil",
                     "blue":"Water",
                     "pink":"Buildings/Roofs",
                     "gray":"Roads",
                     "yellow":"Other Impervious",
                     "red":"Railways",
                     "white":"Ocean"
            }
    ##Define a normalization from values -> colors 
    norm = colors.BoundaryNorm(np.arange(np.min(out_image),np.max(out_image)+1,1),
                               np.max(out_image)-np.min(out_image)+1)
    
    ##Extent
    out_extent_all = np.asarray(src.bounds)[[0,2,1,3]]
    
    out_extent = np.array([(src.bounds[0] + src.bounds[2])/2, (src.bounds[0] + src.bounds[2])/2 + 200,
                           (src.bounds[1] + src.bounds[3])/2, (src.bounds[1] + src.bounds[3])/2 + 200])
    #out_extent = np.array([584250,584250 + 300,4499750,4499750 + 300]) ##extent of zoom window
    

    
    ##Patches
    patches = [Patch(color=color, label=label) for color, label in legend_labels.items()]

    '''
    Color map, norm, extent
    '''
    proj=ccrs.epsg(src.crs.to_epsg()) ##define projection based on the projection of the raster layer 
    
    
    #plot figure with defined projection 
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj)) 
    
    
    #Plot raster
    ax.imshow(out_image, origin="upper", extent=out_extent_all, norm=norm,
              cmap=cmap)
    ax.set_extent(out_extent, proj)

    # Plot shapefile
    shapes.plot(ax=ax, color="white", edgecolor="gray", alpha=0.3)
    
    
    ax.set_xticks(np.arange(out_extent[0],out_extent[1] + 1,
                            (out_extent[1] - out_extent[0])/10))
    ax.set_yticks(np.arange(out_extent[2],out_extent[3] + 1,
                            (out_extent[3] - out_extent[2])/10))
    
    ax.set_title("Fishnet with resolution of 30 meter", size=50, 
                 weight="bold")
    
    plt.tick_params(axis="x", labelsize=25, rotation=90)
    plt.tick_params(axis="y", labelsize=25)
    
    #plt.grid(linestyle=":")
        
        
                    
def fraction_of_landcover(lidarfile):
    import rasterio
    
    with rasterio.open(lidarfile) as src:
        landcover = src.read(1, masked=True)
        
        
    
        
        
    
    
    return None              
    
    


if __name__ == '__main__':
    lidarfile = 'D:/Land_Cover_2017_6-inch/Land_Cover/NYC_2017_LiDAR_LandCover_10_proj_wgs84.tif'
    lidar_file_orig = 'D:/Land_Cover_2017_6-inch/Land_Cover/NYC_2017_LiDAR_LandCover.img'
    shapefile = 'C:/Users/tvo/Desktop/Work/Workspace_1/AOI_4_1115/AOI_3_brokolyn.shp'
    outputGridfbn = 'C:/Users/tvo/Desktop/Work/Workspace_1/AOI_4_1115/landcover_fishnet_1.shp'
    

    
#    dataset, data_window, dataset_res, data_window_res, data_res = raster_resample(lidar_file_orig, lidarfile)
    #shape_ori, data_proj, out_shapefile = reproject_shp(shapefile)
    #out_image, out_transform, out_meta, out_file = shapefile_mask(lidarfile, out_file)
    #outputGridfbn = fishnet("landcover_fishnet_1_2.shp", out_file)
    #plot_raster_vector(out_file, outputGridfbn)
    


 

    pass

  

