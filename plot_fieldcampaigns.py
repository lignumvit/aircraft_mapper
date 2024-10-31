import util.utils
import util.plot_utils
import math
import numpy as np
import netCDF4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from metpy.plots import SkewT
from metpy.units import pandas_dataframe_to_unit_arrays, units
from datetime import datetime, timedelta
from IPython.display import display
from bokeh.io import push_notebook, show, output_notebook, export_png
from bokeh.layouts import row
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, save
from bokeh.models import Title, CustomJS, Select, TextInput, Button, LinearAxis, Range1d
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models.tickers import DatetimeTicker
from bokeh.palettes import Category10
import warnings
import itertools 
import holoviews as hv 
from holoviews import dim, opts
hv.extension('bokeh', 'matplotlib')
#warnings.filterwarnings('ignore')
#output_notebook()
#get_ipython().run_line_magic('matplotlib', 'inline')


# Open all netcdf files and read them into data frames for all field campaigns
field_campaigns = ['CAESAR','ACCLIP','TI3GER','MethaneAir21','SPICULE','OTREC', \
                   'ECLIPSE2019','WE-CAN','SOCRATES','ECLIPSE','ARISTO2016', \
                   'ORCAS','ICEBRIDGE-2015','ARISTO2015','CSET','WINTER','NOREASTER','FRAPPE','DEEPWAVE','CONTRAST']
data_dir = '/Users/ckruse/flight_data/'

dir_path = data_dir + '/' + field_campaigns[0] + '/lrt'
fnames = util.utils.find_flight_fnames(dir_path)

vars_to_read = ['GGLAT','GGLON']
all_campaign_dfs = util.utils.read_all_flights(data_dir, field_campaigns, vars_to_read)

plot = util.plot_utils.plot_campaigns(all_campaign_dfs, show=False)
plot.add_layout(plot.legend[0],'right')

export_png(plot,filename='./plots/RAF_flights_2014_2024.png',width=1000,height=600, scale_factor=2)
