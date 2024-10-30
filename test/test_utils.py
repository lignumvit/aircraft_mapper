import sys
sys.path.append('..')
import util.utils as utils
import pytest
import numpy as np
from datetime import datetime

def test_open_nc():
    # test bad file path, check that FileNotFoundError results
    file_path = "./test_fligt_25hz.nc"
    with pytest.raises(FileNotFoundError):
        utils.open_flight_nc(file_path)

    # now test where file exists
    file_path = "./test_flight_25hz.nc"
    assert utils.open_flight_nc(file_path)
    
def test_read_nc():
    vars_to_read = ['Time','GGALT','LATC','LONC','UIC','VIC','WIC']

    # open up the netcdf files
    file_path_1hz = "./test_flight_1hz.nc"
    file_path_25hz = "./test_flight_25hz.nc"
    nc_1hz = utils.open_flight_nc(file_path_1hz)
    nc_25hz = utils.open_flight_nc(file_path_25hz)

    # test individual read functions for 1 hz and 25 hz files
    #    only tests that no errors were thrown, so not the best test
    df_1hz = utils.read_flight_nc_1hz(nc_1hz, vars_to_read)
    df_25hz = utils.read_flight_nc_25hz(nc_25hz, vars_to_read)

    # test the wrapper function that detects what sampling rate is contained and calls the appropriate read function
    df = utils.read_flight_nc(nc_1hz, vars_to_read)
    df = utils.read_flight_nc(nc_25hz, vars_to_read)

def test_sfm_to_datetime():
    # a test array of sfm (seconds-from-midnight) with units of tunits
    sfm = np.array([0,60,3600.5])
    tunits = 'seconds since 2018-01-15 00:00:00 +0000'

    # call the function
    dt_list = utils.sfm_to_datetime(sfm,tunits)

    # define the expected result
    expected_list = [datetime.fromisoformat('2018-01-15T00:00:00+00:00'),
                     datetime.fromisoformat('2018-01-15T00:01:00+00:00'),
                     datetime.fromisoformat('2018-01-15T01:00:00.5+00:00')]
    # compare
    for (test, expected) in zip(dt_list, expected_list):
        assert test == expected

if __name__ == "__main__":
    test_read_nc()
