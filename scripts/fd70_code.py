# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:52:15 2023

@author: matth
"""
import pandas as pd
import os
import glob
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr
import act

filename = '../FD70/data'



def cleanup_string(text):
    return text.replace("\x01FD", "").replace("\x02", "").replace('\x00', '').strip()




def parse_surface_meteorology_section(lines):
    parsed_values = [parse_surface_met_line(line) for line in lines]

    # Merge the dictionaries into a dataframe
    df = pd.DataFrame(parsed_values)
    df['temperature'] = pd.to_numeric(df.temperature, errors='coerce')
    assert len(df.time) > 0
    df['time'] = pd.to_datetime(df.time)
    df = df.set_index('time')

    # Convert the dataframe to datarray and ensure variable types
    ds = df.to_xarray()
    ds['time'] = pd.to_datetime(ds.time)
    ds['present_wx0'] = ds.present_wx0.astype(int)
    ds['present_wx1'] = ds.present_wx1.astype(int)
    ds['present_wx2'] = ds.present_wx2.astype(int)
    ds['precip_intensity'] = ds.precip_intensity.astype(float)
    ds['temperature'] = ds.temperature.astype(float)
    ds['dewpoint'] = ds.dewpoint.astype(float)
    ds['relative_humidity'] = ds.relative_humidity.astype(float)
    ds['precip_accumulation'] = ds.precip_accumulation.astype(float)
    ds['snowfall_accumulation'] = ds.snowfall_accumulation.astype(float)
    ds['mor_1_minute'] = ds.mor_1_minute.astype(float)
    ds['mor_10_minute'] = ds.mor_10_minute.astype(float)

    return ds


def parse_surface_met_line(text):
    
    # Drop empty lists
    parsed = text.split(' ')
    parsed = ' '.join(parsed).split()
    
    try:
        time = parsed[0]

        # Test to make sure time is longer than 10 characters
        assert len(time) > 10
        
        status = parsed[1]
        mor_1_minute = parsed[2]
        mor_10_minute = parsed[3]
        
        current_precip0 = parsed[4]
        current_precip1 = parsed[5]
        current_precip2 = parsed[6]
        
        present_wx0 = parsed[7]
        present_wx1 = parsed[8]
        present_wx2 = parsed[9]
        
        precip_intensity = parsed[10]
        precip_accumulation = parsed[11]
        snowfall_accumulation = parsed[12]
        
        temperature = parsed[13]
        dewpoint = parsed[14]
        relative_humidity = parsed[15]
        
        if '\\' in temperature:
            temperature = ''
            
        output = {'time':time,
                  'status': status,
                  'mor_1_minute': mor_1_minute,
                  'mor_10_minute': mor_10_minute,
                  'current_precip0': current_precip0,
                  'current_precip1': current_precip1,
                  'current_precip2': current_precip2,
                  'present_wx0': present_wx0,
                  'present_wx1': present_wx1,
                  'present_wx2': present_wx2,
                  'precip_intensity': precip_intensity,
                  'precip_accumulation': precip_accumulation,
                  'snowfall_accumulation':snowfall_accumulation,
                  'temperature': temperature,
                  'dewpoint': dewpoint,
                  'relative_humidity': relative_humidity
                 }
            
            
        return output
        

    except:
        pass
    
        output = ['time',
                  'status',
                  'mor_1_minatusute',
                  'mor_10_minute',
                  'current_precip0',
                  'current_precip1',
                  'current_precip2',
                  'present_wx0',
                  'present_wx1',
                  'present_wx2',
                  'precip_intensity',
                  'precip_accumulation',
                  'snowfall_accumulation',
                  'temperature',
                  'dewpoint',
                  'relative_humidity',]
    
        return pd.DataFrame(columns=output)
        
    

mean_diameters = [0.12, 0.14, 0.17, 0.20, 0.23, 0.27, 0.32,
                  0.38, 0.45, 0.53, 0.62, 0.73, 0.86, 1.02,
                  1.14, 1.23, 1.32, 1.41, 1.52, 1.63, 1.75,
                  1.88, 2.02, 2.16, 2.32, 2.49, 2.68, 2.87,
                  3.08, 3.31, 3.55, 3.82, 4.10, 4.40, 4.72,
                  5.07, 5.44, 5.84, 6.27, 6.73, 7]

def parse_dsd_line(line):
    ref_dsd=' '.join(line.split(' ')).split()
    
    reflectivity = np.array(ref_dsd[0], float)
    dsd_array = np.array(ref_dsd[1:], float)
    
    dsd = xr.DataArray(dsd_array, coords={'mean_diameter':mean_diameters})
    ref = xr.DataArray(reflectivity)
    
    ds = xr.Dataset({'drop_size_distribution': dsd,
                     'reflectivity': ref}).expand_dims('time')
    return ds


def parse_drop_size_distribution_section(lines):
    return xr.concat([parse_dsd_line(line) for line in lines], dim='time')


# Creating metadata attributes and renaming variables to common names
attrs_dict = {'status': {'standard_name': 'Overall Alert, 0=ok, I=info, W=Warning, A=Alarm'},
              'mor_1_minute': {'standard_name': 'MOR, 1-minute Average',
                               'units':'meters'},
              'mor_10_minute': {'standard_name': 'MOR, 10-minute Average',
                          'units': 'meters'},
              'current_precip0': {'standard_name': 'Precipitation Type, NWS CODE'},
              'present_wx0': {'standard_name': 'Present Weather Instant (WMO table 4680)'},
              'present_wx1': {'standard_name': 'Present Weather 15 min (WMO table 4680)'},
              'present_wx2': {'standard_name': 'Present Weather 1 hr (WMO table 4680)'},
              'precip_intesnity': {'standard_name': 'Precipitation Intensity',
                        'units': 'mm/hr'},
              'precip_accumulation': {'standard_name': 'Precipitation Accumulation',
                         'units': 'mm'},
              'snowfall_accumulation': {'standard_name': 'Snow Accumulation',
                          'units': 'mm'},
              'temperature': {'standard_name': 'Ambient Temperature',
                          'units': 'degC'},
              'dewpoint': {'standard_name': 'Dew Point Temperature',
                          'units': 'degC'},
              'relative_humidity': {'standard_name': 'Relative Humidity',
                          'units': 'Percent (%)'},
              'reflectivity':{'standard_name':'Reflectivity (dBZ)',
                              'units':'dBZ'}}


file_out = open('output.log', 'w')
for file in files:
    if file.endswith('.txt'):
        print(file)
        file_in = open(file)

        for line in file_in:
            input_line = bytearray(line, 'utf-8')
            #input_line=input_line.replace("\x1", b'')
            input_line=input_line.replace(b"\x1b[J", b'')       #remove \x1b[J
            input_line=input_line.replace(b"\x1b[20D", b'')     #remove \x1b[20D
            input_line=input_line.replace(b"\x1b[H", b'')       #remove \x1b[H
            input_line=input_line.replace(b"\x1b[0m", b'')      #remove \x1b[0m
            input_line=input_line.replace(b"\x1b[0;0m", b'')    #remove \x1b[0;0m
            input_line=input_line.replace(b"\x1b[1;32m", b'')   #remove \x1b[1;32m
            input_line=input_line.replace(b"\x1b[1;34m", b'')   #remove \x1b[1;34m
            input_line=input_line.replace(b"\x1b[1;35m", b'')   #remove \x1b[1;35m
            input_line=input_line.replace(b"\x1b[1;36m", b'')   #remove \x1b[1;36m
            input_line=input_line.replace(b"\x1b[1m", b'')      #remove \x1b[1m
            input_line=input_line.replace(b"\x07", b'')         #remove \x07 (BEL)
            input_line=input_line.replace(b"\x03", b'')         #remove \x07 (BEL)
        
            p = input_line.find(b"\x08")
            while p>0:                          #apply backspace and remove 'BS'
                del input_line[p]
                del input_line[p-1]
                p = input_line.find(b"\x08")
        
            file_out.write(input_line.decode())

    else:
        print('.log file, skipping')

file_in.close
file_out.close

file1 = open('output.log', 'r')
lines = file1.readlines()
cleaned_strings = [cleanup_string(line) for line in lines[:]]


mean_diameters = [0.12, 0.14, 0.17, 0.20, 0.23, 0.27, 0.32,
                  0.38, 0.45, 0.53, 0.62, 0.73, 0.86, 1.02,
                  1.14, 1.23, 1.32, 1.41, 1.52, 1.63, 1.75,
                  1.88, 2.02, 2.16, 2.32, 2.49, 2.68, 2.87,
                  3.08, 3.31, 3.55, 3.82, 4.10, 4.40, 4.72,
                  5.07, 5.44, 5.84, 6.27, 6.73, 7]

ds = parse_surface_meteorology_section(cleaned_strings[::12])
dsd_ds = parse_drop_size_distribution_section(cleaned_strings[6::12])
ds = ds.isel(time=range(len(dsd_ds.time)))
dsd_ds['time'] = ds.time
ds = xr.merge([ds, dsd_ds])

ds=ds.drop_duplicates(dim='time').sortby('time')
# Decode the WMO 4680 codes
act.utils.decode_present_weather(ds,variable='present_wx0')
act.utils.decode_present_weather(ds,variable='present_wx1')
act.utils.decode_present_weather(ds,variable='present_wx2')

# Lopping through to rename variables
for variable in attrs_dict.keys():
    if variable in list(ds.variables):
        ds[variable].attrs = attrs_dict[variable]


for i in np.unique(ds.time.dt.strftime('%Y%m%d')):
    daily_data = ds.sel(time=i)
    daily_data.to_netcdf(i+'_fd70.nc')
    print(i)


# for i in np.unique(ds.time.dt.strftime('%Y%m%d')):
#    daily_data=ds.sel(time=i)
#    print (daily_data)
