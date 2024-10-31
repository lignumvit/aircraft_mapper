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

def plot_campaigns(all_campaign_dfs: dict[str,dict[str,pd.DataFrame]]):

    campaign = list(all_campaign_dfs.keys())[0]

    plot = figure(width=900, height=600, x_axis_type="mercator", y_axis_type="mercator") 
    plot.add_layout(Title(text="Longitude [Degrees]", align="center"), "below")
    plot.add_layout(Title(text="Latitude [Degrees]", align="center"), "left")


    colors = itertools.cycle(Category10[8])
    for campaign in all_campaign_dfs.keys():
        color=next(colors)
        for flight, df in all_campaign_dfs[campaign].items():
            #latitude = df["LATC"].to_numpy().squeeze()
            #longitude = df["LONC"].to_numpy().squeeze()
            latitude = df["GGLAT"].to_numpy().squeeze()
            longitude = df["GGLON"].to_numpy().squeeze()
            if np.any(longitude > 175) and np.any(longitude < 175):
                num_pos = np.sum(longitude > 0)
                num_neg = np.sum(longitude < 0)
                if num_pos >= num_neg:
                    longitude = np.where(longitude < 0, longitude + 360, longitude)
                else:
                    longitude = np.where(longitude > 0, longitude - 360, longitude)
            k = 6378137
            longitude = longitude * (k * np.pi/180.0)
            latitude = np.log(np.tan((90 + latitude) * np.pi/360.0)) * k
            
            plot.line(longitude,latitude, color=color, legend_label=campaign)

    #plot.add_tile("Esri World Imagery", retina=True)
    #plot.legend.location = 'right'
    plot.legend.click_policy = 'hide'
    plot.add_tile("CartoDB Positron", retina=True)

    show(plot)
