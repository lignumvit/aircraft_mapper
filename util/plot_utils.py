import pandas as pd
import numpy as np
import itertools
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import Title
from bokeh.palettes import Category10

def plot_track(df: pd.DataFrame, mask: pd.Series = None, title: str =''):
    if mask is None:
        mask = np.ones(len(df))
    # get latitude and longitude from dataframe
    latitude = df["LATC"].to_numpy().squeeze()
    longitude = df["LONC"].to_numpy().squeeze()
    
    # update to mercator projection
    k = 6378137
    longitude = longitude * (k * np.pi/180.0)
    latitude = np.log(np.tan((90 + latitude) * np.pi/360.0)) * k
    
    # create the plot layout and add axis labels
    try:
        plot = figure(width=600, height=600, title=title, x_axis_type="mercator", y_axis_type="mercator") 
        plot.add_layout(Title(text="Longitude [Degrees]", align="center"), "below")
        plot.add_layout(Title(text="Latitude [Degrees]", align="center"), "left")
        
        # add the flight track in yellow and add Esri World Imagery as the basemap
        # try:
        if sum(mask) != len(latitude):
            # condition = np.logical_and(data[flight]["Time"].to_numpy() > start,
            #                            data[flight]["Time"].to_numpy() < end)
            lat = df["GGLAT"][mask].to_numpy().squeeze()
            lon = df["GGLON"][mask].to_numpy().squeeze()
            lon = lon * (k * np.pi/180.0)
            lat = np.log(np.tan((90 + lat) * np.pi/360.0)) * k
            plot.multi_line([longitude,lon],[latitude,lat], color=["yellow","red"])
        else:
            plot.line(longitude,latitude, color="yellow")
    
        plot.add_tile("Esri World Imagery", retina=True)
        show(plot)
    except Exception as e:
        print(e)

def plot_campaigns(all_campaign_dfs: dict[str,dict[str,pd.DataFrame]], show_plot: bool = True):

    campaign = list(all_campaign_dfs.keys())[0]

    plot = figure(width=1000, height=600, x_axis_type="mercator", y_axis_type="mercator") 
    plot.add_layout(Title(text="Longitude [Degrees]", align="center"), "below")
    plot.add_layout(Title(text="Latitude [Degrees]", align="center"), "left")


    colors = itertools.cycle(Category10[8])
    for campaign in all_campaign_dfs.keys():
        color=next(colors)
        for flight, df in all_campaign_dfs[campaign].items():
            # If GGLAT was read in, use it. Otherwise, default to GPS-corrected IRU lat/lon.
            # NOTE: GPS-corrected IRU lat/lon has problems when flights cross the dateline.
            if "GGLAT" in list(df.keys()):
                latitude = df["GGLAT"].to_numpy().squeeze()
                longitude = df["GGLON"].to_numpy().squeeze()
            elif "LATC" in list(df.keys()):
                latitude = df["LATC"].to_numpy().squeeze()
                longitude = df["LONC"].to_numpy().squeeze()
            # To prevent wrap around, detect when a flight crossed the dateline and unwrap the longitudes.
            #    That is, when lats go -178, -179, 180, 179, 178, ..., and most flights on east of the dateline
            #    (negative) make all longitudes east of the dateline (negative)
            if np.any(longitude > 179) and np.any(longitude < -179):
                num_pos = np.sum(longitude > 0)
                num_neg = np.sum(longitude < 0)
                if num_pos >= num_neg:
                    longitude = np.where(longitude < 0, longitude + 360, longitude)
                else:
                    longitude = np.where(longitude > 0, longitude - 360, longitude)
            #  ARISTO2017 only had high-rate data, with a handful of lat/lons of 0,0
            if campaign == "ARISTO2017":
                mask = np.argwhere(latitude < 20)
                latitude = np.delete(latitude,mask)
                longitude = np.delete(longitude,mask)
            # shift flights so that the wrap point is where we haven't flown (e.g. longitude of +75)
            #wrap_longitude = 75.0
            #longitude = np.where(longitude > wrap_longitude, longitude - 360, longitude)
            # now, convert lat/lon into plot coordinates
            k = 6378137
            longitude = longitude * (k * np.pi/180.0)
            latitude = np.log(np.tan((90 + latitude) * np.pi/360.0)) * k
            # make that plot
            plot.line(longitude,latitude, color=color, legend_label=campaign)

    plot.legend.click_policy = 'hide'
    plot.legend.label_text_font_size = '10px'
    plot.legend.ncols = 2
    plot.add_tile("CartoDB Positron", retina=True)
    plot.add_layout(plot.legend[0],'right')
    #plot.add_tile("Esri World Imagery", retina=True) # uncomment to add satellite imagery

    if show_plot:
        show(plot)
    else:
        return plot
