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

def test_calc_phii():
    ds = xr.open_dataset(cam_fname)

    g = 9.8067
    phis = ds.PHIS.values/g
    z = ds.Z3.values[:,::-1,:,:]

    zi_interp = utils.calc_phii_midpoint(z,phis)

    nz = zi_interp.shape[1]
    dz = zi_interp[:,1:,:,:]-zi_interp[:,0:nz-1,:,:]
    assert np.all(dz > 0)

def test_calc_phii_hydro():
    ds = xr.open_dataset(cam_fname)

    # compute zi (z at interfaces) via interpolation, extrapolation
    g = 9.8067
    zs = ds.PHIS.values[0,:,:]/g
    z = ds.Z3.values[:,::-1,:,:]
    zi_interp = utils.calc_phii_midpoint(z,zs)

    # compute zi (z at interfaces) via hydrostatic equation
    # read in p, both at grid-cell centers and at vertical interfaces
    ps = ds.PS.values
    p = utils.calc_pressure_cesm(ps, ds.P0.values, ds.hyam.values, ds.hybm.values)
    pi = utils.calc_pressure_cesm(ps, ds.P0.values, ds.hyai.values, ds.hybi.values)
    # read in the rest of the thermodynamic state variables
    tk = ds.T.values[:,::-1,:,:]
    q = ds.Q.values[:,::-1,:,:]
    # compute the density of moist air given specific humidity
    rhom = utils.calc_rhom(p,tk,q)
    # now, calculate heights of vertical iterfaces via hydrostatic integration
    zi_hydro = utils.calc_phii_hydro(zs,rhom,pi)

    for zi in range(zi_hydro.shape[1]):
        # prints out a table of index, zi_interp, zi_hydro, and relative difference
        print(f"{zi:2d} {zi_interp[0,zi,100,100]:8.2f} {zi_hydro[0,zi,100,100]:8.2f} " \
              f"{(zi_interp[0,zi,100,100]-zi_hydro[0,zi,100,100])/zi_hydro[0,zi,100,100]*100:5.2f}")

    # See how close the two calcs of interface heights are. Want to calculate relative differences, so need to
    # only consider points where zi_interp == zi_hydro == zs != 0
    zi_interp = zi_interp.flatten()
    zi_hydro = zi_hydro.flatten()
    non_zero_mask = zi_interp != 0
    ratio = (zi_interp[non_zero_mask]-zi_hydro[non_zero_mask])/zi_hydro[non_zero_mask]*100

    print(f"{np.min(ratio)} {np.max(ratio)} {np.mean(ratio)} {np.std(ratio)}")

    assert np.mean(ratio) < 0.69 # interp method had a relative bias of 0.681 %
    assert np.std(ratio) < 1.02  # standard deviation between methods of 1.013 %
    assert np.all(ratio < 6.4)   # the worst single difference between methods was 6.372 %

if __name__ == "__main__":
    test_calc_phii_hydro()
