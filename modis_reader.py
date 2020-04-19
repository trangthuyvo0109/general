# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:24:41 2019

@author: tvo
"""

import netCDF4
file = 'C:/Users/tvo/Downloads/MOD021KM.A2005240.1700.061.2017185042936.hdf'
from netCDF4 import Dataset
nc = Dataset(file)