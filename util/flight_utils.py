import netCDF4
import pathlib as path
import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Iterable
from util import genpro_utils, dap_utils

# vars_to_read is the default set of variables that get read when calling read_nc below when
# variables are not specified (i.e. when called like "utils.read_nc(netcdf_obj)"). 
vars_to_read = ['Time','GGALT','LATC','LONC', # 4-D Position
                'UIC','VIC','WIC',            # winds
                'ATX','PSFC','EWX',           # other state params
               ]

def sfm_to_datetime(sfm: Iterable[float], tunits: str) -> list[datetime]:
    """
    sfm_to_datetime converts an iterable of seconds from midnight with units of tunits to a list

    :param sfm: An iterable/array of times with units of seconds from UTC midnight
    :param tunits: A string defining the units of sfm. Expected to be in the form of "seconds from YYYY-MM-DD 00:00:00 +0000"

    :return: Returns a list of Python datetimes.
    """

    deltas = np.array([timedelta(seconds=float(s)) for s in sfm])
    tunits_split = tunits.split(' ')
    #t0_iso_str = tunits_split[2]+"T"+tunits_split[3]+tunits_split[4]
    t0_iso_str = tunits_split[2][0:10]+"T00:00:00+0000"
    t0_dt = datetime.fromisoformat(t0_iso_str)
    
    dts = [t0_dt + delta for delta in deltas]
    return dts

def find_flight_fnames(dir_path: str) -> list[str]:
    """
    find_flight_fnames just searches a directory for all *.nc and *.cdf files (older campaigns,
    e.g. SHEBA and ACE-Asia, use .cdf) and returns a list of them.

    :param dir_path: a path to the directory containing flight netcdf files

    :return: Returns a list of flight netcdf files.
    """
    flight_fnames = sorted([fname for fname in os.listdir(dir_path) if fnmatch(fname, "*.nc") or fnmatch(fname, "*.cdf")])
    return flight_fnames

def find_flight_paths(location) -> list[str]:
    """
    find_flight_paths lists the flight netcdf files that make up one field campaign.

    :param location: a directory of flight netcdf files, a glob pattern matching flight files
                     (e.g. ".../HRT/NetCDF/RF*", to leave out preflight or ferry files), or a
                     list of either, for campaigns whose flights are split over several
                     directories (several aircraft, or LRT and HRT holding different flights).
                     A single file can also be given. A glob pattern also matches GENPRO
                     files (see util.genpro_utils), which have no file name extension, and
                     KOFSE's DAP position listings (see util.dap_utils).

    :return: Returns a list of file paths, directory by directory in the order given.
             A missing or unreadable directory raises OSError.
    """
    locations = [location] if isinstance(location, (str, os.PathLike)) else list(location)
    paths = []
    for loc in map(str, locations):
        if any(c in loc for c in '*?['):
            paths.extend(sorted(p for p in glob.glob(loc)
                                if fnmatch(p, "*.nc") or fnmatch(p, "*.cdf")
                                or (os.path.isfile(p) and (genpro_utils.genpro_format(p)
                                                           or dap_utils.is_dap(p)))))
        elif os.path.isfile(loc):
            paths.append(loc)
        else:
            paths.extend(os.path.join(loc, fname) for fname in find_flight_fnames(loc))
    return paths

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

def is_longitude(var: netCDF4._netCDF4.Variable) -> bool:
    """
    is_longitude guesses whether a flight variable holds longitude (GLON, GGLON, LONC, XLONC, ...),
    from its name or long_name.
    """
    long_name = getattr(var, 'long_name', '')
    return 'LON' in var.name.upper() or 'longitude' in str(long_name).lower()

def interp_1hz_to_25hz(var: netCDF4._netCDF4.Variable, sub_seconds: np.ndarray) -> np.ndarray:
    """
    interp_1hz_to_25hz linearly interpolates a 1-D, 1 Hz variable onto the 25 Hz time base of a
    high-rate file, returning a 1-D array of length len(var) * len(sub_seconds). The final
    second, which has no following sample to interpolate toward, is left as NaN.

    Missing (masked) samples become NaN, so a data gap stays a gap. Interpolating straight
    through the masked values instead produced false ramps and (0, 0) positions.

    Longitude is interpolated the short way around: a step from 179.9 to -179.9 is treated as
    +0.2 deg, not -359.8 deg. Interpolating straight across the dateline otherwise sweeps
    through every longitude within one second, which draws a line across the whole map.

    :param var: A 1-D netCDF variable sampled once per second.
    :param sub_seconds: Fractions of a second at which to interpolate, e.g. np.arange(25)/25.

    :return: Returns a 1-D numpy array of floats.
    """
    values = np.ma.filled(np.ma.asarray(var[:], dtype=float), np.nan)
    steps = np.diff(values)
    if is_longitude(var):
        steps = (steps + 180.0) % 360.0 - 180.0  # shortest way around
    output_2d = np.full((len(values), len(sub_seconds)), np.nan)
    output_2d[:-1, :] = values[:-1, None] + sub_seconds[None, :] * steps[:, None]  # divide by 1s omitted
    if is_longitude(var):
        output_2d = (output_2d + 180.0) % 360.0 - 180.0  # back into [-180, 180)
    return np.ravel(output_2d)

def read_flight_nc_25hz(nc: netCDF4._netCDF4.Dataset, read_vars: list[str] = vars_to_read) -> pd.DataFrame:
    """
    read_flight_nc reads a set of variables into memory.

    NOTE: a high-rate, usually 25 Hz, flight data file is assumed.

    :param nc: netCDF4._netCDF4.Dataset object opened by open_flight_nc.
    :param read_vars: An optional list of strings of variable names to be read into memory. A default
                      list, vars_to_read, is specified above. Passing in a similar list will read in those variables
                      instead.

    :return: Returns a pandas data frame.
    """

    data = [] # an empty list to accumulate Dataframes of each variable to be read in

    hz = 25
    sub_seconds = np.arange(0,25,1)/25.

    for var in read_vars:
        try:
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
                    output = interp_1hz_to_25hz(nc[var], sub_seconds)
                    data.append(pd.DataFrame({var: output}))
                else:
                    raise RuntimeError(f"Variable {var} is {ndims}-dimensional. Only 1-D or 2-D variables are handled.")
        except Exception as e:
            #print(f"Issue reading {var}: {e}")
            pass
              

    # concatenate the list of dataframes into a single dataframe and return it. A file
    # holding none of read_vars gives an empty DataFrame rather than a concat error.
    if not data:
        return pd.DataFrame()
    return pd.concat(data, axis=1, ignore_index=False)

def read_flight_nc_1hz(nc: netCDF4._netCDF4.Dataset, read_vars: list[str] = vars_to_read) -> pd.DataFrame:
    """
    read_flight_nc reads a set of variables into memory.

    NOTE: a low-rate, 1 Hz, flight data file is assumed

    :param nc: netCDF4._netCDF4.Dataset object opened by open_flight_nc.
    :param read_vars: An optional list of strings of variable names to be read into memory. A default
                      list, vars_to_read, is specified above. Passing in a similar list will read in those variables
                      instead.

    :return: Returns a pandas data frame.
    """

    data = [] # an empty list to accumulate Dataframes of each variable to be read in
    for var in read_vars:
        try:
            if var == "Time":
                # time is provided every second, so need to calculate 25 Hz times efficiently
                tunits = getattr(nc[var],'units')
                time = nc[var][:]
                data.append(pd.DataFrame({var: time}))
                dt_list = sfm_to_datetime(time, tunits)
                data.append(pd.DataFrame({'datetime': dt_list}))
            else:
                output = nc[var][:]
                if np.ndim(output) == 2:
                    # Some older files (1970s-80s GENPRO conversions) store even position
                    # at several samples per second, as (Time, spsN). Keep each second's
                    # first sample so every variable has one value per Time.
                    output = output[:, 0]
                data.append(pd.DataFrame({var: output}))
        except Exception as e:
            #print(f"Issue reading {var}: {e}")
            pass


    # concatenate the list of dataframes into a single dataframe and return it. A file
    # holding none of read_vars gives an empty DataFrame rather than a concat error.
    if not data:
        return pd.DataFrame()
    return pd.concat(data, axis=1, ignore_index=False)

def read_flight_nc(nc: netCDF4._netCDF4.Dataset, read_vars: list[str] = vars_to_read) -> pd.DataFrame:
    """
    read_flight_nc simply figures out if the flight netcdf object is 1 hz or 25 hz and calls the appropriate reader.

    :param nc: A netcdf object for a flight netcdf file.
    :param read_vars: A list of variable names to be read in the netcdf object. Optional. Default is "vars_to_read" specified
                      above.

    :return: Returns Pandas DataFrame
    """
    dim_names = list(nc.dimensions.keys())
    if 'sps25' in dim_names:
        df = read_flight_nc_25hz(nc, read_vars)
    else:
        df = read_flight_nc_1hz(nc, read_vars)
    return df

def thinning_stride(nc: netCDF4._netCDF4.Dataset, df: pd.DataFrame, period_s: float) -> int:
    """
    thinning_stride returns the row stride that keeps one row of df every period_s seconds.

    RAF files have one Time entry per second, so rows per second is len(df) divided by the
    length of the Time dimension: 1 for low-rate files, 25 for 25 Hz files (whose 1 Hz
    variables are interpolated up to 25 Hz on reading), and so on for any other rate.

    :param nc: The netCDF4 Dataset df was read from.
    :param df: The DataFrame read from nc.
    :param period_s: Seconds between kept samples. 0 keeps every row.

    :return: Returns the stride, an int >= 1.
    """
    if period_s <= 0 or len(df) == 0 or 'Time' not in nc.dimensions:
        return 1
    rows_per_second = len(df) / len(nc.dimensions['Time'])
    return max(1, round(period_s * rows_per_second))

def flight_time_span(nc: netCDF4._netCDF4.Dataset, file_path: str = None):
    """
    flight_time_span returns the first and last time of a flight file.

    Times come from the Time variable (seconds since the date in its units), or base_time +
    time_offset in older files, ignoring garbage times. RAF's archive file names also give
    them, e.g. RF01.19850605.132000_155450.PNI.nc (an end before the start means the flight
    crossed midnight). Either can be wrong -- a few files hold scrambled times, and a few
    names placeholder ones (e.g. ..._010101) -- and an error almost always lengthens the
    span, so when both are available the shorter span is returned.

    :return: Returns (start, end) as datetimes, or (None, None) if they can't be read.
    """
    spans = []
    m = re.search(r'\.(\d{8})\.(\d{6})_(\d{6})', os.path.basename(file_path or ''))
    if m:
        try:
            start = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
            end = datetime.strptime(m.group(1) + m.group(3), '%Y%m%d%H%M%S')
            spans.append((start, end + timedelta(days=1) if end < start else end))
        except ValueError:
            pass
    try:
        time = np.ma.filled(np.ma.asarray(nc['Time'][:], dtype=float), np.nan)
        base = netCDF4.num2date(0, nc['Time'].units, only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
        base = datetime(*base.timetuple()[:6])   # plain, timezone-naive UTC
    except Exception:
        try:
            # older conventions: Unix-epoch base_time plus seconds of time_offset
            time = np.ma.filled(np.ma.asarray(nc['time_offset'][:], dtype=float), np.nan)
            base = datetime(1970, 1, 1) + timedelta(seconds=float(nc['base_time'][...]))
        except Exception:
            time = base = None
    if base is not None:
        # genpro_utils.time_span keeps only times that run like a clock, which drops fill
        # values and corrupt records
        span = genpro_utils.time_span(base, time)
        if span[0] is not None:
            spans.append(span)
    if not spans:
        return None, None
    return min(spans, key=lambda span: span[1] - span[0])

def flight_type(file_path: str) -> str:
    """
    flight_type reads a flight's type from RAF's file naming: 'research' (RF01, ...rf01.nc),
    'ferry' (FF), 'test' (TF), or 'unknown'. The file name is tried first, then the name of
    the directory holding it (GENPRO files sit in directories named after the flight).
    """
    kinds = {'R': 'research', 'F': 'ferry', 'T': 'test'}
    parent, fname = os.path.split(file_path)
    for name in (fname, os.path.basename(parent)):
        m = re.search(r'(?i)(r|f|t)f\d\d', name)
        if m:
            return kinds[m.group(1).upper()]
    return 'unknown'

def read_all_flights(data_dir: str, 
                     field_campaigns: list[str], 
                     read_vars: list[str] = vars_to_read,
                     period_s: float = 0) -> dict[str,dict[str,pd.DataFrame]]:
    """
    read_all_flights reads the requested variables from every flight of every requested
    field campaign.

    :param data_dir: Path to the directory holding one subdirectory per field campaign.
    :param field_campaigns: A list of field campaign names, each expected to have a "lrt"
                            subdirectory of flight netcdf files under data_dir.
    :param read_vars: A list of variable names to be read in the netcdf object. Optional. Default is "vars_to_read" specified
                      above.
    :param period_s: Keep one sample every period_s seconds, whatever the file's sampling
                     rate: period_s = 20 keeps every 20th row of a 1 Hz file and every 500th
                     row of a 25 Hz file. 0 (the default) keeps every sample. Thinning here
                     rather than at plot time keeps the data out of memory, and out of the
                     figures bokeh embeds in the notebook.

    :return: Returns a dictionary of dictionaries of Pandas DataFrames, keyed by field
             campaign and then by flight file name.
    """
    campaign_dirs = {campaign: data_dir + "/" + campaign + "/lrt" for campaign in field_campaigns}
    return read_all_flights_from_dirs(campaign_dirs, read_vars, period_s)

def read_all_flights_from_dirs(campaign_dirs: dict[str,str],
                               read_vars: list[str] = vars_to_read,
                               period_s: float = 0,
                               skip_missing: bool = False,
                               campaign_vars: dict[str,list[str]] = None) -> dict[str,dict[str,pd.DataFrame]]:
    """
    read_all_flights_from_dirs is read_all_flights for when each campaign's flight files live
    in their own, arbitrarily laid-out directory (e.g. the EOL archive on glade, where the
    path differs by aircraft and by data version).

    :param campaign_dirs: A dictionary mapping field campaign name (used as the legend label)
                          to the directory holding that campaign's flight netcdf files, or
                          to anything else find_flight_paths accepts (a glob pattern, or a
                          list of directories/patterns). Campaigns are read in the
                          dictionary's order.
    :param read_vars: A list of variable names to be read in the netcdf object.
    :param period_s: See read_all_flights.
    :param skip_missing: If True, a campaign whose directory is missing, unreadable, or holds
                         no *.nc files is reported and skipped instead of raising.
    :param campaign_vars: Optional per-campaign override of read_vars, keyed by campaign name,
                          for campaigns whose usual position variables are unreliable (e.g.
                          {'FIB': ['LAT','LON']} to read the inertial position).

    :return: Returns a dictionary of dictionaries of Pandas DataFrames, keyed by field
             campaign and then by flight file name. Each DataFrame's attrs hold 'path' (the
             file it was read from), 'start' and 'end' (datetimes of its first and last
             sample, or None) and 'flight_type' (see flight_type).
    """
    if period_s < 0:
        raise ValueError(f"period_s must be >= 0, got {period_s}")

    all_campaign_nc = {} # a dictionary of dictionaries with keys of field campaigns
    for campaign, campaign_dir in campaign_dirs.items():
        print(campaign)
        campaign_read_vars = (campaign_vars or {}).get(campaign, read_vars)
        flight_dict = {} # a dictionary of Pandas DataFrames with keys of file names
        try:
            flight_paths = find_flight_paths(campaign_dir)
        except OSError as e:
            if not skip_missing:
                raise
            print(f"  skipping {campaign}: cannot read {e.filename or campaign_dir} ({e.strerror})")
            continue
        if not flight_paths and skip_missing:
            print(f"  skipping {campaign}: no flight files in {campaign_dir}")
            continue
        for flight_path in flight_paths:
            if dap_utils.is_dap(flight_path):
                # text position listings (KOFSE), one or more flights each
                segments = dap_utils.read_dap_flights(flight_path, campaign_read_vars, period_s)
            elif not (fnmatch(flight_path, "*.nc") or fnmatch(flight_path, "*.cdf")):
                # GENPRO files hold one or more flight segments each
                segments = genpro_utils.read_genpro_flights(flight_path, campaign_read_vars,
                                                            period_s)
            else:
                flight_nc = open_flight_nc(flight_path)
                df = read_flight_nc(flight_nc, campaign_read_vars)
                stride = thinning_stride(flight_nc, df, period_s)
                start, end = flight_time_span(flight_nc, flight_path)
                flight_nc.close()
                if stride > 1:
                    # reset the index so the thinned frame looks like a natively lower-rate
                    # read rather than one with gaps in its index
                    df = df.iloc[::stride].reset_index(drop=True)
                df.attrs['start'], df.attrs['end'] = start, end
                segments = {os.path.basename(flight_path): df}
            for fname, df in segments.items():
                # Record where the flight came from, for reports (util.campaign_report)
                df.attrs['path'] = flight_path
                df.attrs['flight_type'] = flight_type(flight_path)
                # Key by file name. Flights of a campaign spread over several directories
                # (e.g. one per aircraft) can share a file name, so fall back to the full path.
                flight_dict[flight_path + fname[len(os.path.basename(flight_path)):]
                            if fname in flight_dict else fname] = df
        all_campaign_nc[campaign] = flight_dict
    return all_campaign_nc


class flight_obj:
    """
    flight_obj's are classes that hold flight data (i.e. variables indicated by read_vars) from a provided file path string.
    The __init__ takes a file path string and a list of vars to read (vars_to_read by default).
    The __init__ assigns:
    self.file_path: str; the file path passed in
    self.read_vars_attempted: list[str]; the originally passed in list of vars to read
    self.nc: netCDF4._netCDF4.Dataset; the opened netcdf object
    self.df: pd.DataFrame; a dataframe holding the read in data
    self.rate: str; a string indicating the rate of the data read in
    self.read_vars: list[str]; list of the vars that were successfully read in
    """
    def __init__(self, file_path: str, read_vars: list[str] = vars_to_read):
        # assign input vars
        self.file_path = path.Path(file_path)
        self.read_vars_attempted = read_vars

        # open netcdf file if the file exists, assign to self.nc
        if self.file_path.is_file():
            self.nc = netCDF4.Dataset(self.file_path)
        else:
            raise FileNotFoundError(f"File {self.file_path} did not exist!")

        # read in the variables, assign DataFrame to self.df,
        #                               rate to self.rate,
        #                               vars read in to self.read_vars
        dim_names = list(self.nc.dimensions.keys())
        if 'sps25' in dim_names:
            self.df = read_flight_nc_25hz(self.nc, self.read_vars_attempted)
            self.rate = "25Hz"
        else:
            self.df = read_flight_nc_1hz(self.nc, self.read_vars_attempted)
            self.rate = "1Hz"
        self.read_vars = list(self.df.keys())

