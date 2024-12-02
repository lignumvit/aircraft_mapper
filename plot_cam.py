import cartopy.crs as ccrs
import cftime
import matplotlib.pyplot as plt
import xarray as xr

cam_fname = './test/test_ll.nc'

ds = xr.open_dataset(cam_fname)

# read in variables
times = ds.time.values
lats = ds.lat.values
lons = ds.lon.values

# read in variables to extract pressures
hyam = ds.hyam.values
hybm = ds.hybm.values
p0 = ds.P0.values
ps_t0 = ds.PS.sel({'time': times[0]})

hyai = ds.hyai.values
hybi = ds.hybi.values

# plot surface pressure map
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
img = plt.pcolormesh(ds.lon, ds.lat, ps_t0/100., transform=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
plt.colorbar(img, orientation="vertical", fraction=0.0242, pad=0.01)
plt.savefig('./plots/test_ll.png')
