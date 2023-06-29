# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:18:35 2022

@author: matth
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, date, timezone
import datetime as dt
import matplotlib.pyplot as plt
import pytz
from astral import LocationInfo
from astral.sun import sun
from metpy.calc import wet_bulb_temperature
from metpy.units import units
from metpy.calc import lcl
from windrose import WindroseAxes

from thermofeel import (thermofeel, calculate_cos_solar_zenith_angle,
                        calculate_saturation_vapour_pressure_multiphase)
from pysolar import solar


# Tower Lat and Lon for Global Wet Bulb Calculation
lat = 41.70121
lon = -87.99495

# Creating metadata attributes and renaming variables to common names
attrs_dict_tower = {
    'JDA': {'standard_name': 'Ordinal day of the year'},
    'T_LST': {'standard_name': 'Central Standard time at end of period'},
    'TaC_60m': {'standard_name': 'Average 60 m temperature',
                'units': 'degC'},
    'spd_60m': {'standard_name': 'Average 60 m wind speed',
                'units': 'm/s'},
    'spdv60m': {'standard_name': 'Vector-averaged 60 m wind speed',
                'units': 'm/s'},
    'dirV60m': {'standard_name': 'Vector-averaged 60 m wind direction',
                'units': 'Degees 0-360'},
    'sdir60m': {'standard_name': 'Standard deviation of 60 m wind direction',
                'units': 'Degrees 0-360'},
    'e_10m': {'standard_name': 'Average 10 m vapor pressure',
              'units': 'kPa'},
    'rh_10m': {'standard_name': 'Average 10 m relative humidity',
               'units': 'Percent (%)'},
    'Tdp_10m': {'standard_name': 'Average 10 m dew point temperature',
                'units': 'degC'},
    'TaC_10m': {'standard_name': 'Average 10 m temperature',
                'units': 'degC'},
    'spd_10m': {'standard_name': 'Average 10 m wind speed',
                'units': 'm/s'},
    'spdV10m': {'standard_name': 'Vector-averaged 10 m wind speed',
                'units': 'm/s'},
    'dirV10m': {'standard_name': 'Average 10 m Vector-average 10m wind direction',
                'units': 'Degrees 0-360'},
    'sdir10m': {'standard_name': 'Standard deviation of 10 m wind direction',
                'units': 'Degress 0-360'},
    'baroKPa': {'standard_name': 'Average station barometric pressure',
                'units': 'kPa'},
    'radW/m2': {'standard_name': 'Average global irradiation',
                'units': 'W/m^2'},
    'netW/m2': {'standard_name': 'Average net radiation',
                'units': 'W/m^2'},
    'Ta_diff': {'standard_name': 'Average temperature different per meter',
                'units': 'degC/m'},
    'asp_60m': {'standard_name': '60 m aspirator flow monitor',
                'units': 'Percent (%) of time flow above minimum'},
    'asp_10m': {'standard_name': '10 m aspirator flow monitor',
                'units': 'Percent (%) of time flow above minimum'},
    'battVDC': {'standard_name': 'Battery voltage monitor',
                'units': 'V'},
    'precpmm': {'standard_name': 'Average precipitation',
                'units': 'mm'}}

variable_mapping_tower = {'TaC_60m': '60m_temperature',
                          'spd_60m': '60m_windspeed',
                          'spdv60m': 'vector_avg_60m_windspd',
                          'dirV60m': 'vector_avg_60m_winddir',
                          'sdir60m': 'stdDev_60m_winddir',
                          'e_10m': '10m_vapor_pres',
                          'rh_10m': '10m_relhumidity',
                          'Tdp_10m': '10m_dewpoint',
                          'TaC_10m': '10m_temperature',
                          'spd_10m': '10m_windspeed',
                          'spdV10m': 'vector_avg_10m_windspd',
                          'dirV10m': 'vector_avg_10m_winddir',
                          'sdir10m': 'stdDev_10m_winddir',
                          'baroKPa': 'absolute_pressure',
                          'radW/m2': 'global_irradiation',
                          'netW/m2': 'net_radiation',
                          'Ta_diff': 'temperature_diff',
                          'asp_60m': '60m_asp',
                          'asp_10m': '10m_asp',
                          'precpmm': 'precip'}


# To pull today's date
today = date.today()
date_format = today.strftime("%Y%m%d")

# Splice the date into the format for year month day.
year = date_format[0:4]
month = date_format[4:6]
day = date_format[6:8]

# This section reads in data from the ATMOS webpage for 48 hrs or running total.
# Defining the pre-determined name of columns from the tower's website.
cols = ['JDA', 'T_LST', 'TaC_60m', 'spd_60m', 'spdv60m', 'dirV60m', 'sdir60m',
        'e_10m', 'rh_10m', 'Tdp_10m', 'TaC_10m', 'spd_10m', 'spdV10m',
        'dirV10m', 'sdir10m', 'baroKPa', 'radW/m2', 'netW/m2', 'Ta_diff',
        'asp_60m', 'asp_10m', 'battVDC', 'precpmm', 'T_LST2', 'JDA2']

# Link to latest 48 hr tower data.
#url = 'https://www.atmos.anl.gov/ANLMET/anltower.48'

# Link to the running total.
# url='https://www.atmos.anl.gov/ANLMET/anltower.not_qc'

# Reads the tower data into pandas from the URL or text. This will also
# skips the extra header information on the web page and removes the last line
# which restates the column names.
df = pd.read_table('../Tower/YTD_Tower.txt', sep='\s+',  skiprows=[0, 1],
                   header=None, names=cols, na_values=-99999.00)


# Changing the data types from Object to float for most of the columns.
df[['TaC_60m', 'spd_60m', 'spdv60m', 'dirV60m', 'sdir60m', 'e_10m', 'rh_10m',
    'Tdp_10m', 'TaC_10m', 'spd_10m', 'spdV10m', 'dirV10m', 'sdir10m',
    'baroKPa', 'radW/m2', 'netW/m2', 'Ta_diff', 'asp_60m', 'asp_10m',
    'battVDC', 'precpmm']] = df[['TaC_60m', 'spd_60m', 'spdv60m', 'dirV60m',
                                'sdir60m', 'e_10m', 'rh_10m', 'Tdp_10m',
                                 'TaC_10m', 'spd_10m', 'spdV10m', 'dirV10m',
                                 'sdir10m', 'baroKPa', 'radW/m2', 'netW/m2',
                                 'Ta_diff', 'asp_60m', 'asp_10m', 'battVDC',
                                 'precpmm']].apply(pd.to_numeric,
                                                   errors='coerce')
# Changing the indexing and removing the last line as it repeats column names
df = df.reset_index(drop=True)
df = df.head(-1)

# Creates the date from Julian day and converts time to UTC from Central time.
df['time'] = ""
for i in range(len(df.JDA)):
    doy = df.JDA[i]
    merged_date = datetime.strptime(
        year + "-" + df.JDA[i], "%Y-%j").strftime("%Y-%m-%d")
    local = pytz.timezone("US/Central")
    naive = datetime.strptime(
        merged_date + ' ' + df.T_LST[i], "%Y-%m-%d %H:%M")
    utc_dt = naive.astimezone(timezone.utc)
    df['time'][i] = datetime.strptime(str(utc_dt)[:19], "%Y-%m-%d %H:%M:%S")


# Converts the date into a datetime64 type.
df['time'] = df['time'].astype('datetime64')

# Changes indexing column to date and time in UTC
df.set_index('time', inplace=True)

# Deletes the last 2 columns that contains date and time again
df = df.iloc[:, :-2]

# Converts it to an xarray dataset
ds = df.to_xarray()

# Lopping through to add attributes to the variables.
for variable in attrs_dict_tower.keys():
    if variable in list(ds.variables):
        ds[variable].attrs = attrs_dict_tower[variable]

# Rename the variables
ds = ds.rename(variable_mapping_tower)

# Calculating the Pascal Stability Index
stability = []
for i in range(len(ds.global_irradiation)):
    # Daytime calculation
    if ds.global_irradiation[i] >= 700.0:
        if ds['10m_windspeed'][i] >= 5.0:
            classification = 'C'

        elif ds['10m_windspeed'][i] >= 3.0 <= 5.0:
            classification = 'B'

        else:
            classification = 'A'

    elif ds.global_irradiation[i] >= 350.0 <= 750.0:
        if ds['10m_windspeed'][i] >= 6.0:
            classification = 'D'

        elif ds['10m_windspeed'][i] >= 5.0 <= 6.0:
            classification = 'C'

        elif ds['10m_windspeed'][i] >= 2.0 <= 5.0:
            classification = 'B'

        else:
            classification = 'A'

    elif ds.global_irradiation[i] >= 50.0 <= 350.0:
        if ds['10m_windspeed'][i] >= 5.0:
            classification = 'D'

        elif ds['10m_windspeed'][i] >= 2.0 <= 5.0:
            classification = 'C'

        else:
            classification = 'B'

    elif ds.global_irradiation[i] >= 0.25 <= 50.0:
        classification = 'D'

    # New criteria for day time
    # if ds['10m_windspeed'][i] <= 2.0:
    #    if ds.global_irradiation[i] < 350.0:
    #        classification = 'B'

    #    elif ds.global_irradiation[i] >=350.0 <= 750.0:
    #        classification = 'A-B'

    #    elif ds.global_irradiation[i] <= 750.0:
    #        classification = 'A'

    # elif ds['10m_windspeed'][i] >= 2.0 <=3.0:
    #    if ds.global_irradiation[i] < 350.0:
    #        classification = 'C'

    #    elif ds.global_irradiation[i] >=350.0 <= 750.0:
    #        classification = 'B'
    #
    #    elif ds.global_irradiation[i] <= 750.0:
    #        classification = 'A'

    # elif ds['10m_windspeed'][i] >= 3.0 <=5.0:
    #    if ds.global_irradiation[i] < 350.0:
    #        classification = 'C'

    #    elif ds.global_irradiation[i] >=350.0 <= 750.0:
    #        classification = 'B-C'

    #    elif ds.global_irradiation[i] <= 750.0:
    #        classification = 'B'

    # elif ds['10m_windspeed'][i] >= 5.0 <=6.0:
    #    if ds.global_irradiation[i] < 350.0:
    #        classification = 'C'

    #    elif ds.global_irradiation[i] >=350.0 <= 750.0:
    #        classification = 'B-C'

    #    elif ds.global_irradiation[i] <= 750.0:
    #        classification = 'B'

    # else:
    #    if ds.global_irradiation[i] < 350.0:
    #        classification = 'D'

    #    elif ds.global_irradiation[i] >=350.0 <= 750.0:
    #        classification = 'D'

    #    elif ds.global_irradiation[i] <= 750.0:
    #        classification = 'C'

    # Nighttime Calculation
    if ds.global_irradiation[i] <= 0.25:
        if ds.stdDev_10m_winddir[i] >= 22.5:
            if ds['10m_windspeed'][i] >= 3.6:
                classification = 'D'

            elif ds['10m_windspeed'][i] >= 2.9 <= 3.6:
                classification = 'E'

            else:
                classification = 'F'

        elif ds.stdDev_10m_winddir[i] >= 17.5 <= 22.5:
            if ds['10m_windspeed'][i] >= 3.0:
                classification = 'D'

            elif ds['10m_windspeed'][i] >= 2.4 <= 3.0:
                classification = 'E'

            else:
                classification = 'F'

        elif ds.stdDev_10m_winddir[i] >= 12.5 <= 17.5:
            if ds['10m_windspeed'][i] >= 2.4:
                classification = 'D'

            else:
                classification = 'E'

        elif ds.stdDev_10m_winddir[i] >= 7.5 <= 12.5:
            classification = 'D'

        elif ds.stdDev_10m_winddir[i] >= 3.8 <= 7.5:
            if ds['10m_windspeed'][i] >= 5.0:
                classification = 'D'

            else:
                classification = 'E'

        elif ds.stdDev_10m_winddir[i] >= 2.1 <= 3.8:
            if ds['10m_windspeed'][i] >= 5.0:
                classification = 'D'

            elif ds['10m_windspeed'][i] >= 3.0 <= 5.0:
                classification = 'E'

            else:
                classification = 'F'

        else:
            classification = 'G'

    stability.append(classification)
stability = np.array(stability)
ds['stability_class'] = ('time', stability)


# Wet-Bulb Calculation using MetPy:
# ds = ds.assign(
#    wet_bulb_temp=wet_bulb_temperature((ds.absolute_pressure*10) *
#                  units.hPa, ds['10m_temperature'] * units.degC,
#                  ds['10m_dewpoint']*units.degC))
#ds['wet_bulb_temp'].attrs['units'] = 'degC'


# Global Wet Bulb Simple
# https://www.weather.gov/media/tsa/pdf/WBGTpaper2.pdf

global_wet_bulb_simple = thermofeel.calculate_wbgts(
    (ds['10m_temperature'].data+273.15))
ds['global_wet_bulb_simple'] = ('time', global_wet_bulb_simple)
ds['global_wet_bulb_simple'].attrs['units'] = 'degC'

# Loops through and extracts individual days to save the data.
#for i in np.unique(ds.time.dt.strftime('%Y%m%d')):
#    daily_data = ds.sel(time=i)
#    daily_data.to_netcdf(i+'_60mtower.nc')
#    print(i)


# Close the dataset.
xr.Dataset.close(ds)
