import netCDF4
import pathlib as path

read_vars = ['ADIFR', 'BDIFR', 'ADIFRTEMP', 'BDIFRTEMP', # adifr, bdifr
             'QCF', 'QCFR', 'QCR', 'QC_A', 'QC_A2', 'QCFRC',      # raw, dynamic pressures (q)
             'GGLAT','GGLON','GGNSAT','GGQUAL','GGSPD','GGTRK', # GPS variables
             'PSFD', 'PSFRD', 'PSXC', # static pressures
             'VEW', 'VNS', 'VSPD', 'GGVEW', 'GGVNS', 'GGVSPD', 'VEWC', 'VNSC', # aircraft velocities, raw and blended
             'UI', 'UIC', 'VI', 'VIC', 'WI', 'WIC', # winds, both uncorrected and GPS-corrected
             'PALTF', 'PALT', 'GGALT', 'ALT', # altitudes
             'TASF', 'TASFR', 'TASR', 'TAS_A', 'TAS_A2', 'MACHX', # speeds
             'PITCH', 'ROLL', 'THDG', # attitude
             'AKRD', 'SSLIP', # flow angles
             'RHUM', 'RICE', 'ATX', 'BNORMA', 'BLATA', 'BLONGA', 'WDC',
            ]

def open_nc(file_path: str) -> netCDF4._netCDF4.Dataset:

    fp_path = path.Path(file_path)
    if not fp_path.is_file():
        raise FileNotFoundError('testing excptions')

    return netCDF4.Dataset(file_path)

def read_nc(nc: netCDF4._netCDF4.Dataset):
    # sometimes the netcdf4 api produces an issue with big-endian buffer on little-endian compiler
    byte_swap = False
    
    # create empty placeholders for asc, histo_asc and units
    data = {}
    units = {}
    
    # use the netcdf4 api to get the netcdf data into a dataframe
#    try:
        
    # loop over keys in netCDF file and organize
    #for i in nc.variables.keys():
    for i in read_vars:
        try:
            output = nc[i][:]
            data[i] = pd.DataFrame(output)
            units_var = nc.variables[i].getncattr('units')
            units[i] = pd.Series(units_var)
            data[i].columns = pd.MultiIndex.from_tuples(zip(data[i].columns, units[i]))

        except Exception as e:
            print(e)

    # add times
    i = 'Time'
    output = nc[i][:]
    data[i] = pd.DataFrame(output)
    units_var = nc.variables[i].getncattr('units')
    units[i] = pd.Series(units_var)
    data[i].columns = pd.MultiIndex.from_tuples(zip(data[i].columns, units[i]))

    # concatenate the dataframe
    data = pd.concat(data, axis=1, ignore_index=False)
    # clean up the dataframe by dropping some of the multi-index rows
    data.columns = data.columns.droplevel(1)
    data.columns = data.columns.droplevel(1)

    # add a datetime-type time as well
    data['datetime'] = [timedelta(seconds=int(time)) for time in data['Time']]

    return data
