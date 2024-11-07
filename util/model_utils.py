import netCDF4
import pathlib as path
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Iterable

def calc_pressure_cesm(ps: np.ndarray, p0: float, a: np.array, b: np.array) -> list[datetime]:
    """
    calc_pressure_cesm computes pressure on the model levels given surface pressure, reference pressure, and the hybrid
                       level coefficients a and b. See overview at 
                       https://www2.cesm.ucar.edu/models/atm-cam/docs/usersguide/node25.html
                       Simply, p = a*p0 + b*ps.

    :param ps: A 2-D (nlat by nlon) or 3-D (ntime by nlat by nlon) numpy array of surface pressure
    :param p0: The scalar reference pressure
    :param a: An array of a hybrid level coefficients 
    :param b: An array of b hybrid level coefficients 


    :return: Returns a list of Python datetimes.
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
