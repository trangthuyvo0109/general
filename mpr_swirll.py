# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:46:21 2020

@author: tvo
"""

def read_cdf(path,dimension):
    '''
    Function to read .nc files, which are defined as netCDF4 format
    
    Example:
        path = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/MPR/'
        
        read_cdf(path,dimension=1)
    '''
    
    from netCDF4 import Dataset
    import os
    import pandas as pd
    import numpy as np
    
    # Collecting files in a path 
    filenames = []
    for file in os.listdir(path):
        if '.nc' in file:
            filenames.append(path+file)
        
    # Read one example file:
    dataset_test = Dataset(filenames[1]) 

    # All keys:
    keys = dataset_test.variables.keys()
    
    # Extracting two dimensional variables
    two_dimension_variable = []
    for key in list(dataset_test.variables.keys()):
        if len(list(dataset_test[key].shape)) == 2:
            two_dimension_variable.append(dataset_test[key].name)
        else:
            pass
        
    # One dimensional variables
    one_dimension_varible = list(keys)[-9:]
        
    if dimension == 2:    
        # Read and covert 2-d variableas to separate dataframe:
        for file in filenames:
            # Split date from filename:
            date = file.split('UAH_MIPS_MPR_')[-1].split('.nc')[0]
            dataset = Dataset(file)
            
            len_first_dimen = dataset[list(dataset.variables.keys())[0]].shape[0]
            len_second_dimen = dataset['height'].shape[0]
            # Look over each file, 2d variable
            
            keys_list = list(keys)[:5]
            keys_list= keys_list + two_dimension_variable
            df_dict = {i: np.nan for i in keys_list}
                
            # Look over each variable and append data
            for key in list(keys_list):
                
                # Append value for latitude, longitude and altitude with len() = 1
                if len(dataset[key].shape) == 1 and dataset[key].shape[0] == 1:
                    df_dict[key] = np.array([dataset[key][:]]*(len_first_dimen*len_second_dimen)).flatten()
                    print(df_dict[key])
                    print('Case1',key)
                    
                 
                # Append value for epochTime, time, and item e.g.   with len() = len_first_dimen
                if len(dataset[key].shape) == 1 and dataset[key].shape[0] == len_first_dimen:
                    print('Case2',key)
                    df_dict[key] = np.array([dataset[key][:]]*len_second_dimen).flatten()
                    
                # Append value for heigh with len = len_second_dimen
                if len(dataset[key].shape) == 1 and dataset[key].shape[0] == len_second_dimen:
                    print('Case3',key)
                    df_dict[key] = np.array([dataset[key][:]]*len_first_dimen).flatten()
                
                # Append value for temperauter  with 2d shape dimension
                if len(dataset[key].shape) == 2:
                    print('Case4',key)
                    df_dict[key] = np.array(dataset[key][:,:].flatten())
                
            print(df_dict)
            
            # Convert dict to df
            df = pd.DataFrame.from_dict(df_dict) 
            
            # Convert epochTime to UTC time 
            df['epochTime'] = pd.to_datetime(df['epochTime'],unit='s')
            
            
            
            try:
                os.mkdir(path+'CSV/')
                
                
            except:
                pass
            
            try:
                os.mkdir(path+'CSV/'+str(date)+'/')
            except:
                pass
            
            
            df.to_pickle(path+'CSV/'+str(date)+'/'+'2d_'+str(date))
                
    if dimension == 1:
        # Read and covert 2-d variableas to separate dataframe:
        for file in filenames:
            # Split date from filename:
            date = file.split('UAH_MIPS_MPR_')[-1].split('.nc')[0]
            dataset = Dataset(file)
            
            len_first_dimen = dataset[list(dataset.variables.keys())[0]].shape[0]
            
            # Look over each file, 2d variable
            keys_list = list(keys)[:5]
            keys_list.pop(2)
            keys_list= keys_list + one_dimension_varible
            df_dict = {i: np.nan for i in keys_list}
            
            # Look over each variable and append data
            for key in list(keys_list):
                
                # Append value for latitude, longitude and altitude with len() = 1
                if len(dataset[key].shape) == 1 and dataset[key].shape[0] == 1:
                    df_dict[key] = np.array([dataset[key][:]]*(len_first_dimen)).flatten()
                    print(df_dict[key])
                    print('Case1',key)
                    
                 
                # Append value for epochTime, time, and item e.g.   with len() = len_first_dimen
                if len(dataset[key].shape) == 1 and dataset[key].shape[0] == len_first_dimen:
                    print('Case2',key)
                    df_dict[key] = np.array([dataset[key][:]]).flatten()
                    
                    
                
                # Append value for temperauter  with 2d shape dimension
                if len(dataset[key].shape) == 2:
                    print('Case4',key)
                    df_dict[key] = np.array(dataset[key][:]).flatten()
                
            print(df_dict)
            
            # Convert dict to df
            df = pd.DataFrame.from_dict(df_dict) 
            
            # Convert epochTime to UTC time 
            df['epochTime'] = pd.to_datetime(df['epochTime'],unit='s')
            
            
            
            try:
                os.mkdir(path+'CSV/')
                
            except:
                
                pass
            
            try: 
                os.mkdir(path+'CSV/'+str(date)+'/')
            except:
                pass
            

            
            df.to_pickle(path+'CSV/'+str(date)+'/'+'1d_'+str(date))
            
            
    return df
                    
            
        
        
def plot_countouf(df,variable,date,path_fig):
     '''
     Function to plot countourf map based on varaibale
     Only for plotting 2d data
     
     Example:
         import pandas as pd
         import os

         folder = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/MPR/CSV/'
         variable = 'temperature'
         for file in os.listdir(folder):
             date = file
             for item in os.listdir(folder+file):
                 print(item)
                 if '2d' in item:
                     df = pd.read_pickle(folder+file+'/'+item)
                     path_fig = folder+file+'/fig'
                     try:
                         os.mkdir(path_fig)
                     except:
                         pass
                     
                     plot_countouf(df,variable,date,path_fig = path_fig)
             
     '''
     
     import matplotlib.pyplot as plt
     import pandas as pd
     import numpy as np
     
     
     
     # Create df_pivot 
#     df_pivot = df.pivot_table(index='height',
#                               columns='epochTime',                  
#                               values=variable)
     
     
     df_grp = df.groupby('height')
     df_grp_list = []
     for item in list(df_grp.groups.keys()):
         df_grp_list.append(df_grp.get_group(item)[variable].tolist())
     
     # Plot countourf map
     levels = np.arange(df[variable].min(),df[variable].max()+10, 10)
     
     # Plotting
     fig, ax = plt.subplots()
     plot_coutourf = plt.contourf(df_grp_list,df['height'],levels=levels)
     ax.set_title('Vertical Cross-section of '+str(variable)+' at '+str(date))
     ax.set_xticklabels(labels=df['epochTime'])
     fig.colorbar(plot_coutourf)
     
     plt.tick_params(axis='x',rotation=25)
     
     fig.savefig(path_fig+'/vertical_profile_'+str(variable)+'_'+str(date))
     


def plot_1d(df,variable,interval,path_fig,title):
    '''
    Function to plot time series of the data
    
    Input:
    
    Example:

    folder = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/MPR/CSV/'
    # Linux:
    import os
    import pandas as pd
    from mpr_swirll import *
     
    folder = '/nas/rhome/tvo/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/MPR/CSV/'
    variable = 'integratedWaterVapor'
    title = 'Integrated Water Vapor'
    
    
    for file in os.listdir(folder):
        date = file
        for item in os.listdir(folder+file):
            print(item)
            if '1d' in item:
                path_fig = folder + file + '/fig/'  
                df = pd.read_pickle(folder+file+'/'+item)
                plot_1d(df,variable,interval='10T',path_fig = path_fig,title=title)
        
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import matplotlib.dates as mdates
    
    
    # Set time index
    df = df.set_index('epochTime')
    
    df_resampled  = df.resample(interval).mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(15,7),tight_layout=True)
    g = sns.pointplot(x='epochTime',y=variable,data=df_resampled)
    # plt.xticks(np.arange(0,len(df_resampled),np.round(len(df_resampled)/5)))
    
    # Set xtick label date format
    # for nn, ax in enumerate(ax):
    # assign locator and formatter for the xaxis ticks.
    # ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    unique_dates = sorted(list(df_resampled['epochTime'].drop_duplicates()))
    date_ticks = range(0, len(unique_dates), int(np.round(len(unique_dates)/7)))

    g.set_xticks(date_ticks);
    g.set_xticklabels([unique_dates[i].strftime('%m-%d %H:%M:%S') for i in date_ticks], rotation='vertical');
    g.set_xlabel('DateTime (UTC)',size=20,weight='bold');
    g.set_ylabel(title,size=20,weight='bold')
    date_start = df_resampled['epochTime'].tolist()[0]
    date_end = df_resampled['epochTime'].tolist()[-1]
    g.set_title(title+' from '+date_start.strftime(' %m/%d/%y')+ ' to '+date_end.strftime(' %m/%d/%y'),
                size=25,weight='bold')
    
    plt.tick_params(axis='x',labelsize=20)
    plt.tick_params(axis='y',labelsize=20)
    
    try:
        os.mkdir(path_fig)
        
    except:
        pass
    
    fig.savefig(path_fig+'1d_'+variable)
    
    
    
    return None

    
def gps_data(file,interval=False):
    '''
    Function to read GPS data, format .txt
    
    Example:
        file_gps = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/Courtland-20200407T175217Z-001/Courtland/GPS/ctd20088_GPS.txt'
        
        df_gps = gps_data(file_gps)
    
    
    
    '''
    import pandas as pd
    import datetime
    from datetime import datetime as dt
    
    
    
    df = pd.DataFrame(columns=['Julian Date','epochTime','Time','integratedWaterVapor_GPS','source'])
    
    df_csv = pd.read_csv(file,header=None)
    
    df_csv = df_csv.iloc[:,:3].to_numpy()
    
    for item in df_csv:
        df = df.append({'Julian Date':item[0],
                        'epochTime':dt.combine(dt.strptime(str(item[0]), '%y%j').date(),
                        dt.strptime(str(item[1]), '%H:%M').time()),
                        'Time':item[1],
                        'integratedWaterVapor_GPS':item[2],
                        'source':'GPS data'},ignore_index=True)
        
    
    df = df.set_index('epochTime')     
     
    if interval is True:
        df_resampled = df.resample(interval).mean().reset_index()
        
    else:
        df_resampled = df
    
    df_resampled['source'] = 'GPS data'
    
    
    return df_resampled
    


    
    
def plot_scatter(df_1,df_2,path_fig,off_set,color):
    '''
    Function to plot regression between MPR and GPS data
    
    Input:
        df_1: gps_data
        df_2: MPR data with selected time span similar to GPS data
        
        
    Example:
        import datetime
        import pandas as pd
        date = '2020-03-28 '
        if date == '2020-03-28 ':
            file_gps = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/Courtland-20200407T175217Z-001/Courtland/GPS/ctd20088_GPS.txt'
        else:
            file_gps = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/Courtland-20200407T175217Z-001/Courtland/GPS/ctd20089_GPS.txt'

            

        df_1 = gps_data(file_gps)
        time_end = df_1.index[-1] +datetime.timedelta(seconds=60*5)
        df_2 = pd.read_pickle('//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/MPR/CSV/20200327_215630/1d_20200327_215630')

        
        df_2_resampled = df_2.loc[(df_2.epochTime >= str(df_1.index[0]))&(df_2.epochTime <= str(time_end))]
        
        # Resample MPR data 5 min (same interval with GPS data)
        df_2_resampled = df_2_resampled.loc[df_2_resampled['rainTag'] == 0] # Removing data with rain
        df_2_resampled = df_2_resampled.set_index('epochTime').resample('5T').mean().reset_index()
        df_2_resampled = df_2_resampled.set_index('epochTime')
        df_2_resampled = df_2_resampled[df_2_resampled['integratedWaterVapor'].notnull()]
        

        # Manual remove outlier , such as 2 std from mean
        mean = df_2_resampled['integratedWaterVapor'].mean()
        std = df_2_resampled['integratedWaterVapor'].std()
        
        df_2_resampled = df_2_resampled.loc[df_2_resampled['integratedWaterVapor']<(mean+2*std)]
        df_1 = df_1.loc[df_1.index.isin(df_2_resampled.index.tolist())]
        
        
        # Plot scatter
        path_fig = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/results/soundings/'
        plot_scatter(df_1,df_2_resampled,path_fig,off_set=0.25,color='blue')
        
        
        # Plot compare different time
        
        time1 = date + '11:00:00'
        time2 = date + '23:00:00'
        time3 = date + '00:00:00'
        time4 = date + '12:00:00'
        df_1_sub1 = df_1.loc[(df_1.index > time1)&(df_1.index < time2)]
        df_1_sub2 = df_1.loc[(df_1.index > time3)&(df_1.index < time4)]
        df_2_sub1 = df_2_resampled.loc[(df_2_resampled.index > time1)&(df_2_resampled.index < time2)]
        df_2_sub2 = df_2_resampled.loc[(df_2_resampled.index > time3)&(df_2_resampled.index < time4)]
        plot_scatter(df_1_sub1,df_2_sub1,path_fig=path_fig+'split_'+str(time1).replace(' ','').replace(':',''),
        off_set=0.25,color='blue')
        plot_scatter(df_1_sub2,df_2_sub2,path_fig=path_fig+'split_'+str(time3).replace(' ','').replace(':',''),
        off_set=0.25,color='green')
        
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
    
    
    df_concat = pd.concat([df_1,df_2],axis=1)
    
    
    # Plot time series 
    fig, ax = plt.subplots(figsize=(30,15))
    ax.plot(df_concat['integratedWaterVapor_GPS'],'o',color=color,label='GPS data')
    ax.plot(df_concat['integratedWaterVapor'],color='red',linewidth=2.,label='SWIRLL MPR data')

        

    
    plt.legend(prop={'size':30})
    plt.tick_params(axis='x',labelsize=30)
    plt.tick_params(axis='y',labelsize=30)
    plt.xlabel('Date Time (in UTC)',size=30,weight='bold')
    plt.ylabel('IWV (cm)',size=30,weight='bold')

    plt.title('Date: '+str(df_1.index[0].date()),size=30,weight='bold')
    fig.savefig(path_fig+'time_series_'+str(df_1.index[0].date()).replace(' ','').replace(':','_'))
    
    # Plot scatter     
    fig, ax1 = plt.subplots(figsize=(15,15))
    ax1.scatter(df_concat['integratedWaterVapor_GPS'],df_concat['integratedWaterVapor'],color=color)

    
        
    # Plot 1:1 line
    min_value = df_concat[['integratedWaterVapor_GPS','integratedWaterVapor']].min().min() - off_set
    max_value = df_concat[['integratedWaterVapor_GPS','integratedWaterVapor']].max().max() + off_set
    ax1.plot([min_value,max_value],[min_value,max_value],color='black')
    plt.ylim(min_value,max_value)
    plt.xlim(min_value,max_value)
    
    plt.xlabel('GPS IWV (cm)',size=30,weight='bold')
    plt.ylabel('MPR IWV (cm)',size=30,weight='bold')
    
    
    plt.tick_params(axis='x',labelsize=25)
    plt.tick_params(axis='y',labelsize=25)
    
    plt.title('Date: '+str(df_1.index[0].date()),size=30,weight='bold')
    
    # Cal regression coeff
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=df_concat['integratedWaterVapor_GPS'], 
                                                                   y=df_concat['integratedWaterVapor'])
    

    
    # Add an annotation e.g. remember to change
    xy = (min_value + (max_value  - min_value)/10, max_value - (max_value  - min_value)/4) # Coordinate of the equation (x,y)
    rota = 0 # Angle for rotating the equation 
    plt.annotate(r"y=${0:.4f}$x+${1:.4f}$".format(slope,intercept) + "\n" + 
                 r"$R^{2}:$"+r"${0:.4f}$".format(r_value**2) + "\n" +
                 r"Number of samples: ${0:.0f}$".format(len(df_concat)) + "\n" +
                 r"SD: ${0:.4f}$".format(std_err) + "\n" +
                 r"p_value: $< 0.01$",
                 xy=xy,rotation=rota,size=30)
    
    fig.savefig(path_fig+'scatter_'+str(df_1.index[0].date()).replace(' ','').replace(':','_'))
    
      
    
    
def ballon_data(file, date,time_column,interval=False,type_df=False):
    '''
    Function to modify Surface data archiving from NOAA for Courland AL station
    
    Example:
        file_balloon = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/Courtland-20200407T175217Z-001/Courtland/soundings/upperair.UAH_Sonde.202003282100.Courtland_AL_calculated.txt'
        date = file_balloon.split('upperair.UAH_Sonde.')[1].split('.Courtland')[0][:8]
        time_column = 'time (sec)'
        data_ballon_CL = ballon_data(file_balloon, date,time_column,type_df=True)
        
        
        # HV
        file_ballon = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/MPR/soundings/upperair.UAH_Sonde.202003282100.Huntsville_AL.txt'
        date = file_balloon.split('upperair.UAH_Sonde.')[1].split('.Huntsville')[0][:8]
        time_column = 'UTC (HH:MM:SS)'
        data_ballon_HV = ballon_data(file_balloon, date,time_column,type_df=True)
        
    '''
    import pandas as pd
    import datetime
    
    # Read .csv file
    try:
        data_ballon = pd.read_csv(file)
        
    except:
        data_ballon = pd.read_csv(file,skiprows=2)
        data_ballon = data_ballon[:-1]
    
    if time_column is not 'UTC (HH:MM:SS)':
        data_ballon = data_ballon.rename(columns={time_column:'UTC (HH:MM:SS)'})
        time_column = 'UTC (HH:MM:SS)'
        
    else:
        pass

    

    
    
    
    if type_df is True:
        UTC_datetime = []
        for item in data_ballon[time_column].values.tolist():
            
            try:    
                utc_datetime = datetime.datetime.combine(datetime.datetime.strptime(date,"%Y%m%d"),
                                                 datetime.datetime.strptime(str(item).replace(' ',''),"%H:%M:%S").time())
                
            except:
                utc_datetime = datetime.datetime.combine(datetime.datetime.strptime(date,"%Y%m%d"),
                                                 datetime.datetime.strptime(str(item[0]).replace(' ',''),"%H:%M:%S").time())
                
        
            UTC_datetime.append(utc_datetime)
        
    else:
        UTC_datetime = []
        time_column = ['UTC (HH:MM:SS)']
        
        for item in data_ballon[time_column].tolist():
        
            utc_datetime = datetime.datetime.combine(datetime.datetime.strptime(date,"%Y%m%d"),
                                                 datetime.datetime.strptime(str(item),"%H:%M:%S").time())
        
            UTC_datetime.append(utc_datetime)
        
        

    data_ballon[time_column] = UTC_datetime
    

    
    # Reset index to datetime index
    data_ballon = data_ballon.set_index(time_column)
    
    # Resample
    if interval is False:
        data_ballon = data_ballon
    else:
        data_ballon = data_ballon.resample(interval).mean()
    

    
    return data_ballon


    
def grp_sounding_pressure(data_ballon):
    '''
    Example:
        data_ballon_grp_HV = grp_sounding_pressure(data_ballon_HV)
    '''
    
    import numpy as np
    bin_pressure = np.arange(100,1100,10)
    
    
    for i in range(len(bin_pressure)):
        try:
            data_ballon.loc[(data_ballon['pressure(mb)']>=bin_pressure[i])&
                           (data_ballon['pressure(mb)']<bin_pressure[i+1]),'bin_index'] = bin_pressure[i + 1]
            
        except:
            pass
        
    data_ballon = data_ballon.groupby('bin_index').mean().reset_index()   
    
    return data_ballon    
    
    
    
def sounding_IWV(data_ballon):
    '''
    Function to calculate IWV values based on data_ballon
    using integration approach
    
    Example:
        data_ballon_IWV, total_IWV = sounding_IWV(data_ballon)
        
        import os
        # Sounding at CL
        folder_CL = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/Courtland-20200407T175217Z-001/Courtland/soundings/'
        data_ballon_IWV_CL_list = []
        total_IWV_CL_list = []
        data_ballon_IWV_grp_CL_list = []
        
        for item in os.listdir(folder_CL):
            print(item)
            date = item.split('upperair.UAH_Sonde.')[1].split('.Courtland')[0][:8]
            time_column = 'time (sec)'
            data_ballon = ballon_data(folder_CL + item, date,time_column,type_df=True)
            data_ballon_IWV, total_IWV = sounding_IWV(data_ballon)
            data_ballon_IWV_grp_CL = grp_sounding_pressure(data_ballon_IWV)
            
            data_ballon_IWV_CL_list.append(data_ballon_IWV)
            total_IWV_CL_list.append(total_IWV)
            data_ballon_IWV_grp_CL_list.append(data_ballon_IWV_grp_CL)
            

    
    
        # Sounding at Huntsville
        folder_HV = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/MPR/soundings/'
        data_ballon_IWV_HV_list = []
        total_IWV_HV_list = []
        data_ballon_IWV_grp_HV_list = []
        
        for item in os.listdir(folder_HV):
            print(item)
            date = item.split('upperair.UAH_Sonde.')[1].split('.Huntsville')[0][:8]
            time_column = 'UTC (HH:MM:SS)'
            data_ballon = ballon_data(folder_HV + item, date,time_column,type_df=True)
            data_ballon_IWV, total_IWV = sounding_IWV(data_ballon)
            data_ballon_IWV_grp_HV = grp_sounding_pressure(data_ballon_IWV)
            
            data_ballon_IWV_HV_list.append(data_ballon_IWV)
            total_IWV_HV_list.append(total_IWV)
            data_ballon_IWV_grp_HV_list.append(data_ballon_IWV_grp_HV)
            

        
        # Plot (raw data)
        import numpy as np
        column_index = 'rv (kg/kg)'
        for j in range(len(data_ballon_IWV_CL_list)):
            df_CL = data_ballon_IWV_CL_list[j]
            df_HV = data_ballon_IWV_HV_list[j]
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(30,15))
            
            plt.title('Integrated Water Vapor measurements starting from '+str(data_ballon_IWV_CL_list[j].index[0])
            +' until '+str(data_ballon_IWV_CL_list[j].index[-1]),size=30)
                
#            plt.plot(df_CL['IWV (mm)'],df_CL['height (m MSL)'],color='blue',
#            label='Soundings at Courtland Airport with Total IWV = '+ str(np.round(total_IWV_CL_list[j],3)) + ' mm')
#            
#            plt.plot(df_HV['IWV (mm)'],df_HV['height (m AGL)'],color='red',
#            label='Soundings at SWIRLL with Total IWV = '+ str(np.round(total_IWV_HV_list[j],3)) + ' mm')
            
            plt.plot(df_CL[column_index],df_CL['height (m MSL)'],color='blue',
            label='Total IWV at CDL = '+ str(np.round(total_IWV_CL_list[j],3)) + ' mm')
            
            plt.plot(df_HV[column_index],df_HV['height (m AGL)'],color='red',
            label='Total IWV at SWIRLL = '+ str(np.round(total_IWV_HV_list[j],3)) + ' mm')            
            
            plt.legend(prop={'size':30})
            plt.xlabel(r'Mixing ratio $r_{v}$ (kg/kg)',size=30)
            plt.ylabel('Height (m AGL) or Height (m MSL)',size=30)
            plt.tick_params(axis='x',labelsize=25)
            plt.tick_params(axis='y',labelsize=25)
            
            path_fig = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/results/soundings/'
            fig.savefig(path_fig+str(data_ballon_IWV_CL_list[j].index[0]).replace(' ','_').replace(':','_'))
            #fig.savefig(path_fig+str(j))
            
        # Plot (averaging by bin)
        for j in range(len(data_ballon_IWV_grp_HV_list)):
            df_CL = data_ballon_IWV_grp_CL_list[j]
            df_HV = data_ballon_IWV_grp_HV_list[j]
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(30,15))
            
            plt.title('Integrated Water Vapor measurements '+'\n'+
            str(data_ballon_IWV_CL_list[j].index[0])
            +' - '+str(data_ballon_IWV_CL_list[j].index[-1]),size=35,weight='bold')
                
#            plt.plot(df_CL['IWV (mm)'],df_CL['height (m MSL)'],color='blue',linewidth=2.,
#            label='Soundings at Courtland Airport with Total IWV = '+ str(np.round(total_IWV_CL_list[j],3)) + ' mm')
#            
#            plt.plot(df_HV['IWV (mm)'],df_HV['height (m AGL)'],color='red',linewidth=2.,
#            label='Soundings at SWIRLL with Total IWV = '+ str(np.round(total_IWV_HV_list[j],3)) + ' mm')
#            
            plt.plot(df_CL[column_index],df_CL['height (m MSL)'],color='blue',linewidth=2.,
            label='Total IWV at CDL = '+ str(np.round(total_IWV_CL_list[j],3)) + ' mm')
            
            plt.plot(df_HV[column_index],df_HV['height (m AGL)'],color='red',linewidth=2.,
            label='Total IWV at SWIRLL = '+ str(np.round(total_IWV_HV_list[j],3)) + ' mm')
             
            
            plt.legend(prop={'size':35})
            plt.xlabel(r'Mixing ratio $r_{v} (kg/kg)$',size=35,weight='bold')
            plt.ylabel('Height (m AGL)',size=35,weight='bold')
            plt.tick_params(axis='x',labelsize=30)
            plt.tick_params(axis='y',labelsize=30)
            #plt.xlim(-0.5,4.5)
            plt.ylim(0,16000)
            
            path_fig = '//uahdata/rhome/Materials/ATS_671/final_project/ATS671_Data-20200402T223007Z-001/ATS671_Data/results/soundings/'
            fig.savefig(path_fig+'moving_average_'+str(data_ballon_IWV_CL_list[j].index[0]).replace(' ','_').replace(':','_'))
            #fig.savefig(path_fig+str(j))
        
    
    
    
    '''
    import numpy as np
    
    # Calculate es(T)
    data_ballon['es(T) (mb)'] = 6.112 * np.exp(17.67 * data_ballon['temp (deg C)']/(
            data_ballon['temp (deg C)'] + 243.5)) 
    
    # Calculate e
    data_ballon['e(T) (mb)'] = data_ballon['RH (%)']/100*data_ballon['es(T) (mb)']
    
    # Calculate rv
    data_ballon['rv (kg/kg)'] = 0.622 * data_ballon['e(T) (mb)'] / data_ballon['pressure(mb)']
    
    # Calculate delta_p
    data_ballon['delta_p (mb)'] = np.append(np.abs(np.diff(data_ballon['pressure(mb)'])),np.nan)
    
    # Calculate rv_mean
    mean_value_list = []
    for i in range(len(data_ballon['rv (kg/kg)']) - 1):
        print(i)
        mean_value = (data_ballon['rv (kg/kg)'].iloc[i + 1] + data_ballon['rv (kg/kg)'].iloc[i])/2
        mean_value_list.append(mean_value)
        
    data_ballon['rv_mean (kg/kg)'] = np.append(mean_value_list,np.nan)


    # Calculate IWV (mm)
    data_ballon['IWV (mm)'] = 1/(1000 * 9.81) * data_ballon['delta_p (mb)'] * data_ballon['rv_mean (kg/kg)'] * 100 * 1000
    
    # Calculate total IWV (mm)
    
    total_IWV = np.sum(data_ballon['IWV (mm)'])
    
    data_ballon_IWV = data_ballon
    
    print('Total Integrated Water Vapor (mm): ',total_IWV)

    
    return data_ballon_IWV, total_IWV
    
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
