# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:29:25 2019

@author: tvo
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


##List of projections 
projs = [ccrs.Mercator(), ccrs.LambertCylindrical(),
ccrs.Robinson(), ccrs.InterruptedGoodeHomolosine(),
ccrs.Orthographic(), ccrs.Mollweide()]

##Setting projection of the basemap
proj = ccrs.Mercator()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj))


##And land feature from basemap 
land_feat = cfeature.LAND
land = ax.add_feature(land_feat, facecolor="burlywood")

##Set boundary of the map
bound_lons = [-130, -66.5]
bound_lats = [0, 60]

##Set map projection in lats/lons units 
ll_proj = ccrs.Geodetic()

##Set extent from boundary 
_ = ax.set_extent(bound_lons+bound_lats, ll_proj)


##Adding coastline feature for the basemap
_feat = cfeature.COASTLINE
coast = ax.add_feature(_feat)

##Gridline:
gl = ax.gridlines(draw_labels=True, linestyle=":")

##Define space between ticks of grid lines:
import matplotlib.ticker as mticker

del_lon = 15
del_lat = 15

gl.xlocator = mticker.MultipleLocator(del_lon)
gl.ylocator = mticker.MultipleLocator(del_lat)

#Add Ocean feature
ocean_feat = cfeature.OCEAN
ocean = ax.add_feature(ocean_feat, facecolor="steelblue")

##Add Country Border
_feat = cfeature.BORDERS
country = ax.add_feature(_feat, edgecolor="gray")

##Read csv without comma (tab instead of),define list of header cols
cols = ["day",'hour','lats','lons','pressure','winds','class']
file ='C:/Users/tvo/Downloads/katrina_track.txt'
track = pd.read_csv(file, delim_whitespace=True, names=cols)


## Plot points with coordinates and defined projection by transform=ll_proj
katrina_line = ax.plot(track.lons, track.lats, marker="d", transform=ll_proj)

#Read shapefile in cartopy
import cartopy.io.shapereader as shprdr
file = 'C:/Users/tvo/Downloads/tl_2010_01_county10/tl_2010_01_county10.shp'
rdr = shprdr.Reader(file)

recs =  list(rdr.records()) ##list of recors of the shape file
recs[0].attributes ##attributes of the first value of the shapefile
recs[0].geometry ##visualze the geometry of the first value of the shapefile


import pandas as pd
recs = [r.atributes for r in recs.recors()]
recs = pd.DataFrame(recs)

import numpy as np
fig, ax = plt.subplots(subplot_kw=dict(projection=proj))
ll_proj = ccrs.PlateCarree()
ax.set_extent([-89,-84.5,30,35.5], crs=ll_proj)
gl = ax.gridlines(draw_labels=True, linestyle=":")


ua_color = np.array([158, 27, 50])/255
uab_color = np.array([30, 107, 82])/255
uah_color = np.array([0, 119, 200])/255

for rec in rdr.records():
    cnty_name = rec.attributes["NAME10"].upper()
    if cnty_name =="MADISON":
        color = uah_color
    elif cnty_name=="JEFFERSON":
        color = uab_color
    elif cnty_name=="TUSCALOOSA":
        color = ua_color
    else:
        color = "none"
    _ = ax.add_geometries(rec.geometry, ll_proj, edgecolor="black",facecolor=color)
    




















