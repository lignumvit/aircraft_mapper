import netCDF4
import pathlib as path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# vars_to_read is the default set of variables that get read when calling read_nc below when
# variables are not specified (i.e. when called like "utils.read_nc(netcdf_obj)"). 
vars_to_read = ['Time','GGALT','LATC','LONC', # 4-D Position
                'UIC','VIC','WIC',            # winds
                'ATX','PSFC','EWX',           # other state params
               ]

def sfm_to_datetime(sfm, tunits: str) -> list[datetime]:
    """
    sfm_to_datetime converts an iterable of seconds from midnight with units of tunits to a list

    :param sfm: An iterable/array of times with units of seconds.
    :param tunits: A string defining the units of sfm. Expected to be in the form of "seconds from YYYY-MM-DD HH:mm:SS +HHMM"

    :return: Returns a list of Python datetimes.
    """

    deltas = np.array([timedelta(seconds=s) for s in sfm])
    tunits_split = tunits.split(' ')
    t0_iso_str = tunits_split[2]+"T"+tunits_split[3]+tunits_split[4]
    t0_dt = datetime.fromisoformat(t0_iso_str)
    
    dts = [t0_dt + delta for delta in deltas]
    return dts

def open_flight_nc(file_path: str) -> netCDF4._netCDF4.Dataset:
    """
    open_flight_nc simply checks to see if the file at the provided path string exists and opens it.

    :param file_path: A path string to a flight data file, e.g. "./test/test_flight.nc"

    :return: Returns netCDF4._netCDF4.Dataset object.
    """

    fp_path = path.Path(file_path)
    if not fp_path.is_file():
        raise FileNotFoundError('testing excptions')

    return netCDF4.Dataset(file_path)

def read_flight_nc(nc: netCDF4._netCDF4.Dataset, read_vars: list[str] = vars_to_read) -> pd.DataFrame:
    """
    read_flight_nc reads a set of variables into memory.

    NOTE: a high-rate, usually 25 Hz, flight data file is assumed.

    :param nc: netCDF4._netCDF4.Dataset object opened by open_flight_nc.
    :param read_vars: An optional list of strings of variable names to be read into memory. A default
                      list, vars_to_read, is specified above. Passing in a similar list will read in those variables
                      instead.

    :return: Returns either a dictionary of numpy time series arrays or a single pandas data frame.
    """

    data = [] # an empty list to accumulate Dataframes of each variable to be read in

    hz = 25
    sub_seconds = np.arange(0,25,1)/25.

    for var in read_vars:
        if var == "Time":
            # time is provided every second, so need to calculate 25 Hz times efficiently
            tunits = getattr(nc[var],'units')
            time = nc[var][:]

            time_25hz = np.zeros((len(time),hz)) # 2-D
            for i,inc in enumerate(sub_seconds):
                time_25hz[:,i] = time + inc
            output = np.ravel(time_25hz) # ravel to 1-D
            data.append(pd.DataFrame({var: output}))
            dt_list = sfm_to_datetime(output, tunits)
            data.append(pd.DataFrame({'datetime': dt_list}))
        else:
            ndims = len(np.shape(nc[var][:]))
            if ndims == 2:
                # 2-D, 25 Hz variables can just be raveled into 1-D time series
                output = np.ravel(nc[var][:])
                data.append(pd.DataFrame({var: output}))
            elif ndims == 1:
                # 1-D variables in 25 Hz data files exist (e.g. GGALT is sampled at 20 Hz, but by default,
                # this is filtered to 1Hz instead of fudged to 25 Hz). Do interpolation to 25 Hz so all time series
                # have same length.
                output_1d = nc[var][:]
                output_2d = np.zeros((len(output_1d),hz))*float("NaN")
                for i in range(len(output_1d)-1):
                    output_2d[i,:] = output_1d[i] + sub_seconds*(output_1d[i+1]-output_1d[i]) # divide by 1s omitted
                output = np.ravel(output_2d)
                data.append(pd.DataFrame({var: output}))
            else:
                raise RuntimeError(f"Variable {var} is {ndims}-dimensional. Only 1-D or 2-D variables are handled.")
              

    # concatenate the list of dataframes into a single dataframe
    data = pd.concat(data, axis=1, ignore_index=False)

    return data
