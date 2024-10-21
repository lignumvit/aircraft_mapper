import sys
sys.path.append('..')
import util.utils as utils
import pytest

def test_open_nc():
    # test bad file path, check that FileNotFoundError results
    file_path = "./test_fligt.nc"
    with pytest.raises(FileNotFoundError):
        utils.open_nc(file_path)

    # now test where file exists
    file_path = "./test_flight.nc"
    assert utils.open_nc(file_path)
    
def test_read_nc():
    file_path = "./test_flight.nc"
    nc = utils.open_nc(file_path)

