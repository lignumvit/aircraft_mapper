import sys
sys.path.append('..')
import util.utils as utils
import pytest
import numpy as np
from datetime import datetime

def test_open_nc():
    # test bad file path, check that FileNotFoundError results
    file_path = "./test_fligt.nc"
    with pytest.raises(FileNotFoundError):
        utils.open_flight_nc(file_path)

    # now test where file exists
    file_path = "./test_flight.nc"
    assert utils.open_flight_nc(file_path)
    
def test_read_nc():
    file_path = "./test_flight.nc"
    nc = utils.open_flight_nc(file_path)

    vars_to_read = ['Time','GGALT','LATC','LONC','UIC','VIC','WIC']
    df = utils.read_flight_nc(nc, vars_to_read) # only tests that no errors were thrown, so not a great test

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
