# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:14:04 2019

@author: tvo
"""

'''
This module contains functions for crowdsourcing weather data from Netatmo WeatherMap.

Routine Listing
-------
- netatmo_request : 
    Send request to the Netatmo WeatherMap via credentials and 
    returns in a list of stations inside therequested area 
    
- crowd_data : 
    Crowd the weather data from each stations which have been requested
    export the crowded in .csv format for storing the data

- modify_raw_data:
    Modify the crowded data by adding station_id, latitude and longitude 
    
    
- generate_HeatMap : 
    generate HeatMap from current observations

- generate_HeatMapTime:
    generate animated HeatMap by Time from current observations
    

-------
This module is required to create an account in Netatmo WeatherMap: 
    https://dev.netatmo.com/
    
After creating an account, user should find their credentials information via:
    https://dev.netatmo.com/myaccount/
    
'''



import patatmo
import pandas
import time
import os 
import datetime
from datetime import datetime   
import pandas as pd   




def netatmo_request(credentials, region=None):
    '''
    This function requests to get public data of the requested region 
    
    Parameters
    ----------
    * region: dictionary, optional
        The area of region is requested is a 4 elements dictionary
        in the order of {"lat_ne": latitude of the North-East corner
                     "lat_sw": latitude of the South-West corner
                     "lon_ne": longitude of the North-East corner
                     "lon_sw": longitude of the South-West corner
                     }
        
        By default, the value of New York city will be used.    
            region = {"lat_ne": 40.916730,
                      "lat_sw": 40.488842,
                      "lon_ne": -73.707089,
                      "lon_sw": -74.267392
                      }
    
    * credentials: dictionary
        credentials information of the user, which is created via Netatmo,
        which can be found: https://dev.netatmo.com/myaccount/,
        in the order of {"password":"your_password",
                         "username":"your_username",
                         "client_id":"your_client_id",
                         "client_secret":"your_client_secret"
                         }
        Note: This parameter is not optional, but for testing purpose, you can use my credentials information
                    credentials = {"password":"Vothuytrang_0109",
                                   "username":"trangthuyvo.hcmus@gmail.com",
                                   "client_id":    "5d97a908e0c2b13e7c77d804",
                                   "client_secret":"MuUSg23zdGL7ywTDRCLVIDl4gtyo50UvYGOOBcLy"}
  
    Returns
    -------
    client: patatmo.api.client.NetatmoClient
        contains credentials information of the user 
        this information will be repeatly used when we would like to request anything from Netatmo
    
    publicdata: patatmo.api.responsetypes.GetpublicdataResponse
        contains observations inside the region
        the observations could be either from activated or deactivated stations 
        
    device_id: list
        list of the device_id (which is named by mac address of the station)
        
    module_id: list
        list of the module_id of corresponsing device_id. Here we only consider module for outdoor observation. 
            
    Examples
    -------
    # Your region
    >>> region = {
    "lat_ne": 40.916730,
    "lat_sw": 40.488842,
    "lon_ne": -73.707089,
    "lon_sw": -74.267392,
    }
    # Your credentials
    >>> credentials = {
    "password":"Vothuytrang_0109",
    "username":"trangthuyvo.hcmus@gmail.com",
    "client_id":    "5d97a908e0c2b13e7c77d804",
    "client_secret":"MuUSg23zdGL7ywTDRCLVIDl4gtyo50UvYGOOBcLy"
    }
    # Send request
    >>> client, publicdata, device_id, module_id = netatmo_request(region=region, credentials=credentials) 
    
    '''  
    
    if region is None:
        # lat/lon outline of New York City
        region = {
            "lat_ne": 40.916730,
            "lat_sw": 40.488842,
            "lon_ne": -73.707089,
            "lon_sw": -74.267392,
            }
    
    
    
    # configure the authentication
    authentication = patatmo.api.authentication.Authentication(
        credentials=credentials,
        tmpfile = "temp_auth.json")
    
    
    # providing a path to a tmpfile is optionally.
    # If you do so, the tokens are stored there for later reuse,
    # e.g. next time you invoke this script.
    # This saves time because no new tokens have to be requested.
    # New tokens are then only requested if the old ones expire.
    
    # create an api client
    client = patatmo.api.client.NetatmoClient(authentication)
    
    
    # iusse the API request for get public data
    publicdata = client.Getpublicdata(region = region)
    
    
    
    # conver the response to a pandas.DataFrame
    publicdata.dataframe().to_csv("data_region.csv")
    

    # return list of device_id of first 50 stations
    device_id = publicdata.dataframe().id
    
    
    
    # return list of module_id
    module_id = []
    for i in range(len(publicdata.response["body"])):
        # only consider the module for temperature and humidity
        md_id = list(publicdata.response['body'][i]['measures'].keys())[0]
        module_id.append(md_id)
    


        
    return client, publicdata, device_id, module_id




def crowd_data(client, device_id, module_id, start_id=-1, directory=None, scale=None):
    '''
    This function is applied to crowd the data based on client, device_id, module_id 
    have been requested using function netatmo_request()
    ** Note: Due to the usage limit for Netatmo server, the progress for crowding data
    ** could take very long, for instance, 512000 measurements per hour
    ** For testing purpose, it is suggested to define a small region
    
    Parameters
    ----------
    client: patatmo.api.client.NetatmoClient
        contains credentials information of the user 
        this information will be repeatly used when we would like to request anything from Netatmo

    device_id: list
        list of the device_id (which is named by mac address of the station)
        
    module_id: list
        list of the module_id of corresponsing device_id.
        
    start_id: integer
        Just in case the progress has to be suspended, we can specific the order of the station 
        which has been stopped and continue crowding from this station until the end
        without looping over from the beginning
        By default, start_id equals to -1 which means start looping from the beginning 
        
    directory: string
        directory to hold the downloaded data.
        By default, it will be extracted to current_working_fold/crowdsourd/
        
    scale: string, optional
        resolution of the data that user would like to download  e.g. “1day”, “1week”, “1month”
        By default, maximum resolution of the data will be return in: "max"
        
        

  
    Returns
    -------
    raw_data_list: list
        list of multiple pandas DataFrame. Each DataFrame represents the raw data that have been crowded for each station

            
    Examples
    -------
    # Your region (testing with a small region)
    >>> region = {
    "lat_ne": 40.91,
    "lat_sw": 40.89,
    "lon_ne": -73.70,
    "lon_sw": -73.90,
    }
    # Your credentials
    >>> credentials = {
    "password":"Vothuytrang_0109",
    "username":"trangthuyvo.hcmus@gmail.com",
    "client_id":    "5d97a908e0c2b13e7c77d804",
    "client_secret":"MuUSg23zdGL7ywTDRCLVIDl4gtyo50UvYGOOBcLy"
    }
    # Send request
    >>> client, publicdata, device_id, module_id = netatmo_request(region=region, credentials=credentials) 
    # Check the number of available stations
    >>> len(device_id)
    Out[93]: 3
    # Start downloading 
    >>> raw_data_list = crowd_data(client, device_id, module_id, start_id=-1, scale="1day")
    
    '''
    
    # Define directory of the raw data
    if directory is None:
        # Making a new directory
        directory = "/crowdsourd_raw/" # for linux
        # directory = "\crowdsourd_raw\\" # for windows
        
        # Make directory if it does not exist. If yes, do not make
        if not os.path.exists(os.getcwd() + directory):
            os.mkdir(os.getcwd()+directory)
    
    
    # Define the resolution of the data
    if scale is None:
        scale = "max"
    
    
    """
    Attempt to archive first 1024 measurements from each station and append to the list
    """
    raw_data_list = []
    for i in range(len(device_id)):
        try:
            print("Downloading first measurements of the station "+str(i))
            # Get the first measurement of each station
            measure = client.Getmeasure(device_id[i], module_id[i], scale=scale).dataframe()
                       
            # Append the current measurement to a new list
            raw_data_list.append(measure)
            print(len(raw_data_list))
            
        except:
            # Catch the Server error e.g. Service Unavailable or Usage Reach Limit 
            # Let the code stop for 20 seconds and then continue the loop
            print("There is something with the server e.g. Usage Reach Limit. Please wait for 20 seconds"
            +" to reset the code")
            time.sleep(20)
            
        
            continue
    
    
    """
    Attempt to loop over each station to get the whole dataset until current time
    """
    for i in range(len(raw_data_list)):
        print("Downloading measurements of the station " + str(i))
        print("Please wait patiently, it could take a while .... ")
        
        # Just in case the progress has to be suspended, we can specific the order of the station 
        # which has been stopped and continue crowding from this station until the end
        # without looping over from the beginning
        if i <= start_id :
            pass
        
        else:
            # Get the current time
            timenow = time.mktime(datetime.now().timetuple())
            
            # Staring looping to download the data for each station
            while True:
                try:
                    print('Current time: ',datetime.now())
        
                    # Get the date begin by extracting from the last element of the dataframe
                    date_begin = time.mktime(raw_data_list[i].index[-1].timetuple())
                    print(i)
                    
                    # if date_begin is larger than current time (means earlier than current time)
                    if date_begin > timenow:
                        # stop the while loop
                        break
                    
                    # Get measure
                    measure = client.Getmeasure(device_id[i], module_id[i], date_begin=date_begin,
                                                scale=scale, 
                                                date_end=timenow)
                    
                    # Combine old measure and new measure
                    df = [raw_data_list[i], measure.dataframe()]
                    
                    # Concate them together
                    df_concat = pd.concat(df)
                    raw_data_list[i] = df_concat
                    print("Crowded time "+str(raw_data_list[i].index[-1]))
                    
                    
                    # Export data to .csv                     
                    path = os.getcwd() + directory #directory to hold the downloaded data, for linux 
                    
                    # Specific filename as the mac address for each station.
                    # As mac address contains semi-colon ":" which is not allowed to stored in Windows system
                    # Thus, semi-colon will be replaced by underscore "_" 
                    filename = device_id[i].replace(":","_")
                    raw_data_list[i].to_csv(path+filename)
                    
                
                # Catch the Server error e.g. Service Unavailable or Usage Reach Limit 
                # Let the code stop for 300 seconds and then continue the loop
                except:
                    print('Error catched time: ', datetime.now()) # Print the time at which results in an error
                    print('Wait for 300 seconds') # Wait for 300 seconds .... 
                    
                    # Lets the program stop for 300 seconds                    
                    time.sleep(300)
                    
                    # And then continue the while loop
                    continue
                                
    return raw_data_list


def modify_raw_data(publicdata, raw_dir=None, modi_dir=None):  
    '''
    This function is applied to modify the raw crowded data by adding:
        - device_id: mac address of the station
        - latitude: latitude of the station
        - longitude: longitude of the station
        
    Besides, the function is also allowed to extract the time value e.g. year, month, day 
        from the current timestamp and add to current DataFrame
        
    ** The reason of seperating this function with the crowd_data function is due to
    ** the progress of crowding data could take quite long and easily to be interrupted at some point
    
    
    Parameters
    ----------
    directory: string, optional
        directory of the raw crowded data 
        By default, it will be designated to current_working_folder/crowdsourd_raw/
        
    publicdata: patatmo.api.responsetypes.GetpublicdataResponse
        contains observations inside the region
        the observations could be either from activated or deactivated stations
        
        
    modi_dir: string, optional
        directory of the modifed crowded data 
        By default, it will be designated to current_working_folder/crowdsourd_modi/

    Returns
    -------
    data_list: list
        list of multiple pandas DataFrame. Each DataFrame represents the modified data that have been crowded for each station

            
    Examples
    -------

    
    '''  
    import pandas as pd
    import os
    
    # Default raw directory:
    if raw_dir is None:
        raw_dir = "/crowdsourd_raw/" # for linux
        # raw_dir = "\crowdsourd_raw\\" # for Windows
        
        
    # Default modified directory:
    if modi_dir is None:
        modi_dir = "/crowdsourd_modi/" # for linux
        # modi_dir = "\crowdsourd_modi\\" # for Windows
        
        # Make directory if it does not exist. If yes, do not make
        if not os.path.exists(os.getcwd() + modi_dir):
            os.mkdir(os.getcwd()+modi_dir)
        
        
    # Check length of filename whether it equals to 17 and append to a list of filename
    path = os.getcwd() + raw_dir
    listdir = []
    for filename in os.listdir(path):
       if len(filename) == 17:
           listdir.append(filename)
    
    # Read the input data as pandas Dataframe and put all together in a list
    measure_df_list = []
    
    for i in range(len(listdir)):
        
        # As we attempt to catch the stations information based on the metadata
        # Currently the data is still downloading
        # Which returns in error if the data is not available
        # Thus, we will try to ignore the unavailable data and 
        # Just do the calculation based on existing data
        try:
        
            # Read all files in the current directory 
            measure_df = pd.read_csv(path + listdir[i])
            
            # Define station ID from the file name
            station_id = listdir[i].replace("_",":")[:17] # Took only first 17 characters of the filename
            print(station_id)
            
            # Extract latitude and longitude from the file contains metadata of all stations
            coors = publicdata.dataframe()
            latitude = coors.loc[coors.id == listdir[i].replace("_",":")[:17]].latitude.values
            longitude = coors.loc[coors.id == listdir[i].replace("_",":")[:17]].longitude.values
            print(latitude)
            print(longitude)
            
            # Define a list of new columns will be added to the current DataFrame
            columns = ["station_id", "latitude", "longitude"]
            values = [station_id, str(latitude[0]), str(longitude[0])]
            print(values)
            for j in range(len(columns)):
                measure_df.insert(len(measure_df.columns),
                              column = columns[j],
                              value = values[j])
            
            # Aggregate the Dataframe by values of year, month and day and insert data columns to the current dataframe
            columns_dt = ["year", "month", "day"]
            values_dt = [pd.to_datetime(measure_df.time).dt.year,
                         pd.to_datetime(measure_df.time).dt.month,
                         pd.to_datetime(measure_df.time).dt.day]
            for k in range(len(columns_dt)):
                measure_df.insert(len(measure_df.columns),
                                  column = columns_dt[k],
                                  value = values_dt[k])
                
            # Export modified data to modified directory
            measure_df.to_csv(os.getcwd() + modi_dir + listdir[i])
            
            # Apend each DataFrame to a List 
            measure_df_list.append(measure_df)
        
        except Exception as e:
            print(e)
            # Ignore the error as mentioned above
            continue
        
    # Concat the list of Dataframe to one Dataframe
    measure_df_list_concat = pd.concat(measure_df_list)
    print(measure_df_list_concat)
    
    # Export the .csv 
    measure_df_list_concat.to_csv(os.getcwd() + "\data_combine.csv")
    
    return measure_df_list, measure_df_list_concat


# df = pd.read_csv('C:/Users/tvo/Desktop/test_heatmap.csv')

def generate_HeatMap(data, location=None, zoom_start=None):   

    '''
    This function generates Heat Map based on current observations
    
    Parameters
    ----------
    data: pandas DataFrame 
    The DataFrame of current observation in the order of [latitude, longitude, value]
    
    location: array [latitude, longitude], optional
    Location of center of the basemap that will be zoomed in when generating the basemap 
    By default, the location of New York City [40.693943, -73.985880] will be used
    
    zoom_start: integer, optional
    Level of zoom to the basemap. By default, value of 12 will be used
  
    Returns
    -------
    basemap
    The basemap in generated based on current observation. The basemap will be exported as an .html 
    file and can be visualized simply by open this .html file 
            
    Examples
    -------
    >>> generate_HeatMap(location=[40.693943, -73.985880], zoom_start=12, data=df)  
    '''        
    
    import folium
    from folium.plugins import HeatMap
    
    # Default values for location and zoom levels 
    if location is None:
        location = [40.693943, -73.985880]
        
    if zoom_start is None:
        zoom_start = 12
        
        
        
    # Generate the basemap by location and zoom level 
    basemap = folium.Map(location=location, control_scale=True, zoom_start=zoom_start)
    
    try:
        # Generate HeatMap and add to the current BaseMap 
        max_value = float(data[data.columns[-1]].max()) # Extracting the maximum value of the variable
        HeatMap(list(zip(data.latitude.values, data.longitude.values, data[data.columns[-1]].values)),
                            radius=17, 
                            max_val=max_value,
                            gradient={.4: 'white', .65: 'gray', 1: 'red'},
                            max_zoom=10).add_to(basemap)
        
       
    except Exception as e:
        # Catching Error whether the input data is a pandas DataFrame
        if type(data) is not pandas.core.frame.DataFrame:
            print("Please check the type of input data. It has to be a pandas DataFrame in the order of"+ 
                  "[latitude, longitude, value]")
        else:
            print(e)
        
    # Export basemap to .html file for visualization 
    basemap.save("C:/Users/tvo/Desktop/heatmap.html")
    
    return basemap


def generate_HeatMapTime(data, location=None, zoom_start=None):
    '''
    This function generates animated Heat Map by Time based on current observations
    
    Parameters
    ----------
    data: array of arrays or so called 2-D array.
    First dimension is related to the largest Time span (e.g. months of year if the user want to create a monthly animated HeatMap).
    Second dimension is related to the second-largest Time (e.g. days of month if the user want to create a montly animated HeatMap)
    The data grouped by Time is a 3 element array-like value (e.g., an array, list, or tuple) in the order of [latitude, longitude, value]
    
    
    
    location: array [latitude, longitude], optional
    Location of center of the basemap that will be zoomed in when generating the basemap 
    By default, the location of New York City [40.693943, -73.985880] will be used
    
    zoom_start: integer, optional
    Level of zoom to the basemap. By default, value of 12 will be used
  
    Returns
    -------
    basemap
    The animated basemap in generated based on current observation. 
    The basemap will be exported as an .html 
    file and can be visualized simply by open this .html file 
            
    Examples
    -------
    >>> 
    generate_HeatMapTime(location=[40.693943, -73.985880], zoom_start=12, data=df)  
    '''  
    import folium
    from folium.plugins import HeatMapWithTime
    
    # Default values for location and zoom levels 
    if location is None:
        location = [40.693943, -73.985880]
        
    if zoom_start is None:
        zoom_start = 12
        
        
        
    # Generate the basemap by location and zoom level 
    basemap = folium.Map(location=location, control_scale=True, zoom_start=zoom_start)    
    
    try:
        # Generate an animated HeatMap by Time 
        HeatMapWithTime(data, 
                        radius = 17,
                        min_opacity=0.5, 
                        max_opacity=0.8, 
                        use_local_extrema=True,
                        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
                        ).add_to(basemap)
    except Exception as e:
        print(e)
    
    
    basemap.save("C:/Users/tvo/Desktop/test_heatmaptime.html")
    return basemap

#
#
#df_concat.groupby(df_concat["month"]).values()
#
#
#
#for day in df_concat.day.sort_values().unique():
#    df_concat_day_list.append(df_concat.loc[df_concat.day == day, ["latitude", "longitude", "Temp"]].groupby(["latitude","longitude"]).sum().reset_index().values.tolist())
#    print(df_concat_day_list)

#
#df_concat_year_2015_month_list = []
#for month in df_concat_year_2015.month.sort_values().unique():
#    df_concat_year_2015_month_list.append(df_concat_year_2015.loc[df_concat_year_2015.month == month, ["latitude","longitude","Temperature"]].groupby(["latitude","longitude"]).mean().reset_index().values.tolist())
#    print(df_concat_year_2015_month_list)
#
#
