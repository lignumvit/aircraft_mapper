import netCDF4
import pathlib as path
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Iterable

g = 9.8067
Rd = 287.1
Rv = 461.5
ep = Rd/Rv

def calc_pressure_cesm(ps: np.ndarray, p0: float, a: np.array, b: np.array) -> np.ndarray:
    """
    calc_pressure_cesm computes pressure on the model levels given surface pressure, reference pressure, and the hybrid
                       level coefficients a and b. See overview at 
                       https://www2.cesm.ucar.edu/models/atm-cam/docs/usersguide/node25.html
                       Simply, p = a*p0 + b*ps.

    :param ps: A 2-D (nlat by nlon) or 3-D (ntime by nlat by nlon) numpy array of surface pressure
    :param p0: The scalar reference pressure
    :param a: An array of a hybrid level coefficients 
    :param b: An array of b hybrid level coefficients 

    :return: Returns  4-D array of grid-cell-center pressure.
    """

    # if ps is 2-D, make it 3-D
    if len(ps.shape) == 2: ps = ps[None,:,:]

    # figure out what the output dimensions should be
    ntimes = ps.shape[0]
    nlevs = len(a)
    nlats = ps.shape[1]
    nlons = ps.shape[2]

    # output p array
    p = np.zeros((ntimes,nlevs,nlats,nlons))*float("NaN")

    # iterate through times, levels and calculate pressures
    for ti in range(ntimes):
        for i in range(nlevs):
            p[ti,i,:,:] = a[i]*p0 + b[i]*ps[ti,:,:]

    p = p[:,::-1,:,:] # flip level index so that pressures are decreasing with level, height

    return p

def calc_phii_midpoint(z: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """
    calc_phii_midpoint estimates the vertical interfaces of grid cells via midpoint

    :param z: A 4-D (ntime by nlev by nlat by nlon) numpy 
              array grid-cell-center geopotential height in meters
    :param zs: a 2-D array of the geopotential height of the surface in meters

    :return: Returns a 4-D numpy array of geopotential height at the interfaces.
    """

    nt = z.shape[0]
    nz = z.shape[1]
    nlat = z.shape[2]
    nlon = z.shape[3]

    zi = np.zeros((nt,nz+1,nlat,nlon)) # if 3 levels were provided, want to return a 4-level interface height
    zi[:,0,:,:] = zs # set the surface, 0 index
    # for the, say, 3 level z's provided, compute the 2 midpoints
    zi[:,1:nz,:,:] = 0.5*(z[:,1:nz,:,:]+z[:,0:nz-1,:,:])
    # now, estimate the model top height: ztop = z[-1] + (z[-1] - 0.5*(z[-1] + z[-2]))
    zi[:,nz,:,:] = z[:,-1,:,:] + z[:,-1,:,:] - zi[:,nz-1,:,:]
    return zi

def calc_t_virt(tk: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    calc_t_virt calculates the virtual temperature given temperature and specific humidity

    :param tk: a N-D array of temperature in Kelvins
    :param q: a N-D array of specific humidity (rhov/rho), non-dimensional

    :return: Returns an N-D numpy array of density of moist air
    """
    factor = 1./ep - 1
    tv = tk*(1 + q * factor)
    return tv

def calc_rhom(p: np.ndarray, tk: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    calc_rhom calculates the density of moist air given pressure, temperature, and specific humidity

    :param p: a N-D array of pressure in Pa
    :param tk: a N-D array of temperature in Kelvins
    :param q: a N-D array of specific humidity (rhov/rho), non-dimensional

    :return: Returns an N-D numpy array of density of moist air
    """
    rhom = p/Rd/calc_t_virt(tk,q)
    return rhom

def calc_phii_hydro(zs: np.ndarray, rhom: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    calc_phii_hydro estimates the vertical interfaces of grid cells via integration on the hydrostatic
                    equation.

    :param zs: a 2-D array of the geopotential height of the surface in meters
    :param rhom: a 4-D array of density of the air (e.g. moist air)
    :param pi: a 4-D array of pressure at the vertical interfaces grid-cell interfaces

    :return: Returns a 4-D numpy array of geopotential height at the interfaces.
    """

    nt = rhom.shape[0]
    nz = rhom.shape[1] # number of model levels, not interfaces
    nlat = rhom.shape[2]
    nlon = rhom.shape[3]

    zi = np.zeros((nt,nz+1,nlat,nlon)) # if 3 levels were provided, want to return a 4-level interface height
    zi[:,0,:,:] = zs # set the surface, 0 index
    for i in range(nz):
        dp = pi[:,i+1,:,:] - pi[:,i,:,:]
        dz = -dp/rhom[:,i,:,:]/g
        zi[:,i+1,:,:] = zi[:,i,:,:] + dz
    return zi


