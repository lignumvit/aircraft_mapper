# import local utils
import sys
sys.path.append('..')
import util.model_utils as utils
# import regular modules
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cftime
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable



cam_fname = './test_ll.nc'

def test_calc_pressure():
    ds = xr.open_dataset(cam_fname)

    # read in variables
    times = ds.time.values

    # read in variables to extract pressures
    hyam = ds.hyam.values
    hybm = ds.hybm.values
    p0 = ds.P0.values
    ps = ds.PS.sel({'time': times[0]}).values

    hyai = ds.hyai.values
    hybi = ds.hybi.values

    pm = utils.calc_pressure_cesm(ps, p0, hyam, hybm)
    pi = utils.calc_pressure_cesm(ps, p0, hyai, hybi)

    # check dimension sizes
    assert pm.shape[0] == 1
    assert pm.shape[1] == len(hyam)
    assert pm.shape[2] == len(ds.lat.values)
    assert pm.shape[3] == len(ds.lon.values)

    assert pi.shape[1] == len(hyai)

    # check that there are no NaNs in the pressures
    assert np.all(np.isfinite(pm))
    assert np.all(np.isfinite(pi))

    # check that pressure decreased everywhere
    nlevi = pi.shape[1]
    dp = pi[:,1:nlevi,:,:] - pi[:,0:nlevi-1,:,:]
    assert np.all(dp < 0)

if __name__ == "__main__":
    test_calc_pressure()
