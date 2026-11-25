import pandas as pd
import numpy as np
import itertools
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import (Title, DataRange1d, Range1d, CustomJSTickFormatter,
                          ColumnDataSource, LabelSet)
from bokeh.palettes import Category10, Category20

_BASEMAP_TAG = 'basemap'
PROJECTIONS = ('equirectangular', 'robinson')

def _projector(projection: str = 'equirectangular', wrap_longitude: float = 75.0):
    """
    Build the lon/lat -> plot coordinate mapping for a projection.

    Both supported projections put the map's left/right seam at wrap_longitude, so a seam
    at +75 deg (where nothing has flown) centers the figure on the Pacific and keeps
    dateline-crossing flights continuous.

    :param projection: 'equirectangular' (lon/lat degrees straight through) or 'robinson'
                       (pseudocylindrical, via pyproj; plot coordinates are metres).
    :param wrap_longitude: longitude of the map seam, in degrees.

    :return: Returns (to_xy, x_bounds, y_bounds, seam_jump), where to_xy maps lon/lat
             arrays to plot coordinates, the bounds tuples are the full extent of the
             projection, and seam_jump is the plot-x distance beyond which a step in a
             track means it crossed the seam rather than actually flew there.
    """
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}")

    if projection == 'equirectangular':
        def to_xy(lon, lat):
            lon = np.asarray(lon, dtype=float)
            return np.where(lon > wrap_longitude, lon - 360.0, lon), np.asarray(lat, dtype=float)
        x_bounds = (wrap_longitude - 360.0, wrap_longitude)
        y_bounds = (-90.0, 90.0)
    else:
        import pyproj
        lon_0 = wrap_longitude - 180.0
        # Project longitude *relative* to the central meridian, with lon_0=0. Handing proj
        # an absolute longitude lets it re-normalise into [-180, 180], which collapses the
        # two seam edges onto each other -- the left edge of the map would be drawn at the
        # right. Keeping the shift ourselves means -180 and +180 stay distinct edges.
        crs = pyproj.CRS.from_proj4("+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs")
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        def to_xy(lon, lat):
            rel = np.asarray(lon, dtype=float) - lon_0
            rel = np.where(rel > 180.0, rel - 360.0, rel)
            rel = np.where(rel < -180.0, rel + 360.0, rel)
            x, y = transformer.transform(rel, np.asarray(lat, dtype=float))
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

        # Robinson's outline is widest at the equator and tallest at the poles.
        xe, _ = to_xy(np.array([lon_0 - 180.0, lon_0 + 180.0]), np.zeros(2))
        _, ye = to_xy(np.zeros(2) + lon_0, np.array([-90.0, 90.0]))
        x_bounds = (float(xe.min()), float(xe.max()))
        y_bounds = (float(ye.min()), float(ye.max()))

    seam_jump = 0.5 * (x_bounds[1] - x_bounds[0])
    return to_xy, x_bounds, y_bounds, seam_jump

def _densify(coords: np.ndarray, max_step: float = 2.0) -> np.ndarray:
    """
    Insert intermediate lon/lat points so long straight segments bend correctly once
    projected. A country border that runs straight along a meridian in lon/lat is a curve
    in Robinson, and without this it would be drawn as a chord across the curve.

    :param coords: (N, 2) array of lon/lat degrees.
    :param max_step: maximum spacing between successive points, in degrees.

    :return: Returns the densified (M, 2) array.
    """
    out = []
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        n = max(int(max(abs(x1 - x0), abs(y1 - y0)) / max_step), 1)
        out.append(np.column_stack([np.linspace(x0, x1, n, endpoint=False),
                                    np.linspace(y0, y1, n, endpoint=False)]))
    out.append(coords[-1:])
    return np.vstack(out)

def _split_at_seam(x: np.ndarray, y: np.ndarray, seam_jump: float):
    """
    Break a track where it crosses the map seam, so no false line is drawn straight across
    the figure. Bokeh's line glyph treats NaN as a gap.

    :param x: projected x coordinates.
    :param y: projected y coordinates.
    :param seam_jump: x distance above which a step is taken to be a seam crossing.

    :return: Returns the (x, y) arrays with NaNs inserted at any seam crossing.
    """
    if x.size < 2:
        return x, y
    breaks = np.where(np.abs(np.diff(x)) > seam_jump)[0]
    if breaks.size == 0:
        return x, y
    return np.insert(x, breaks + 1, np.nan), np.insert(y, breaks + 1, np.nan)

def _natural_earth_polygons(resolution: str, category: str, name: str) -> list:
    """
    Read a Natural Earth dataset and flatten it to a list of shapely Polygons.

    :param resolution: Natural Earth resolution, '110m', '50m' or '10m'.
    :param category: Natural Earth category, e.g. 'physical' or 'cultural'.
    :param name: Natural Earth dataset name, e.g. 'land' or 'admin_0_countries'.

    :return: Returns a list of shapely Polygon objects.
    """
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import MultiPolygon

    fname = shpreader.natural_earth(resolution=resolution, category=category, name=name)
    polygons = []
    for geom in shpreader.Reader(fname).geometries():
        polygons += list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    return polygons

def _clip_at_seam(polygon, wrap_longitude: float) -> list:
    """
    Split a lon/lat polygon at the map seam.

    Countries that straddle the seam (Russia, Antarctica, Kazakhstan at +75 deg) would
    otherwise have their rings torn across the whole figure, painting bands of land where
    there is none. Clipping into the piece west of the seam and the piece east of it --
    the latter shifted a world west -- lands everything inside (wrap-360, wrap].

    :param polygon: a shapely Polygon in lon/lat degrees.
    :param wrap_longitude: longitude of the map seam, in degrees.

    :return: Returns a list of shapely Polygons, already shifted into the map's range.
    """
    from shapely.affinity import translate
    from shapely.geometry import box

    pieces = []
    for lo, hi, shift in ((-180.0, wrap_longitude, 0.0), (wrap_longitude, 180.0, -360.0)):
        if lo >= hi:
            continue
        clipped = polygon.intersection(box(lo, -90.0, hi, 90.0))
        if clipped.is_empty:
            continue
        parts = clipped.geoms if hasattr(clipped, 'geoms') else [clipped]
        for part in parts:
            if part.geom_type == 'Polygon' and not part.is_empty:
                pieces.append(translate(part, xoff=shift) if shift else part)
    return pieces

def _basemap_rings(resolution: str, category: str, name: str, projection: str,
                   wrap_longitude: float) -> list:
    """
    Read a Natural Earth dataset and return its rings as projected (x, y) arrays, grouped
    per polygon so holes stay attached to their exterior.

    :return: Returns a list of lists of (x, y) tuples -- one inner list per polygon, with
             the exterior ring first and any holes after it.
    """
    to_xy, _, _, _ = _projector(projection, wrap_longitude)
    out = []
    for polygon in _natural_earth_polygons(resolution, category, name):
        for piece in _clip_at_seam(polygon, wrap_longitude):
            rings = []
            for ring in [piece.exterior, *piece.interiors]:
                coords = np.asarray(ring.coords)
                if projection != 'equirectangular':
                    coords = _densify(coords)
                rings.append(to_xy(coords[:, 0], coords[:, 1]))
            out.append(rings)
    return out

def _exclude_basemap_from_autorange(plot):
    """
    Keep the basemap out of the figure's auto-ranging.

    The basemap covers the whole globe, so if it took part in auto-ranging the figure would
    zoom out to the full world no matter where the flights actually are. Range on the data
    glyphs only. Basemap glyphs are identified by their tag, so this works whatever order
    things get added in.
    """
    data_renderers = [r for r in plot.renderers
                      if hasattr(r, 'data_source') and _BASEMAP_TAG not in r.tags]
    if data_renderers:
        for rng in (plot.x_range, plot.y_range):
            if isinstance(rng, DataRange1d):
                rng.renderers = data_renderers

def add_land_fill(plot, projection: str = 'equirectangular', wrap_longitude: float = 75.0,
                  resolution: str = '110m', color: str = '#efefef', alpha: float = 1.0):
    """
    Fill the Natural Earth land polygons. Together with the figure's background fill (the
    ocean) this stands in for raster basemap tiles, which bokeh can only serve in Web
    Mercator and so cannot follow either of these projections.

    :param plot: the bokeh figure to draw on.
    :param projection: one of PROJECTIONS; must match what the tracks were drawn with.
    :param wrap_longitude: longitude of the map seam, in degrees.
    :param resolution: Natural Earth resolution, '110m' (default, global figures),
                       '50m', or '10m'. Finer resolutions are slower and only pay off
                       when zoomed in.
    :param color: land fill color.
    :param alpha: land fill opacity.

    :return: Returns the bokeh GlyphRenderer holding the land.
    """
    rings_per_polygon = _basemap_rings(resolution, 'physical', 'land',
                                       projection, wrap_longitude)
    # multi_polygons nests three deep: a list of multi-polygons, each a list of polygon
    # parts, each a list of rings -- exterior first, then any holes.
    xs = [[[x for x, _ in rings]] for rings in rings_per_polygon]
    ys = [[[y for _, y in rings]] for rings in rings_per_polygon]

    land = plot.multi_polygons(xs=xs, ys=ys, fill_color=color, fill_alpha=alpha,
                               line_color=None, level='underlay', tags=[_BASEMAP_TAG])
    _exclude_basemap_from_autorange(plot)
    return land

def add_country_outlines(plot, projection: str = 'equirectangular',
                         wrap_longitude: float = 75.0, resolution: str = '110m',
                         color: str = '#666666', line_width: float = 0.5,
                         alpha: float = 0.7):
    """
    Draw Natural Earth country outlines (coastlines + international borders) on top of the
    land fill. These are vectors, so they stay sharp at any export resolution.

    :param plot: the bokeh figure to draw on.
    :param projection: one of PROJECTIONS; must match what the tracks were drawn with.
    :param wrap_longitude: longitude of the map seam, in degrees.
    :param resolution: Natural Earth resolution, '110m' (default), '50m' or '10m'.
    :param color: outline color.
    :param line_width: outline width in pixels.
    :param alpha: outline opacity.

    :return: Returns the bokeh GlyphRenderer holding the outlines.
    """
    rings_per_polygon = _basemap_rings(resolution, 'cultural', 'admin_0_countries',
                                       projection, wrap_longitude)
    xs = [x for rings in rings_per_polygon for x, _ in rings]
    ys = [y for rings in rings_per_polygon for _, y in rings]

    # level='underlay' keeps the outlines above the land fill but below the flight tracks,
    # no matter what order things get added in.
    outlines = plot.multi_line(xs=xs, ys=ys, color=color, line_width=line_width,
                               alpha=alpha, level='underlay', tags=[_BASEMAP_TAG])
    _exclude_basemap_from_autorange(plot)
    return outlines

def add_graticule(plot, projection: str = 'robinson', wrap_longitude: float = 75.0,
                  lon_step: float = 30.0, lat_step: float = 30.0,
                  color: str = '#999999', line_width: float = 0.5, alpha: float = 0.6,
                  label_meridians: bool = True):
    """
    Draw meridians and parallels. Robinson's meridians are curved and its parallels are
    unevenly spaced, so the figure's own x/y axes carry no useful longitude/latitude
    information and a drawn graticule has to stand in for them.

    :param plot: the bokeh figure to draw on.
    :param projection: one of PROJECTIONS.
    :param wrap_longitude: longitude of the map seam, in degrees.
    :param lon_step: meridian spacing in degrees.
    :param lat_step: parallel spacing in degrees.
    :param color: graticule line color.
    :param line_width: graticule line width in pixels.
    :param alpha: graticule opacity.
    :param label_meridians: label meridians along the bottom and parallels down the left.

    :return: Returns the bokeh GlyphRenderer holding the graticule.
    """
    to_xy, _, _, _ = _projector(projection, wrap_longitude)
    west = wrap_longitude - 360.0

    xs, ys = [], []
    lats = np.arange(-90.0, 90.0 + 1e-9, 1.0)
    for lon in np.arange(west, wrap_longitude + 1e-9, lon_step):
        x, y = to_xy(np.full_like(lats, lon), lats)
        xs.append(x); ys.append(y)
    lons = np.arange(west, wrap_longitude + 1e-9, 1.0)
    for lat in np.arange(-90.0 + lat_step, 90.0 - 1e-9, lat_step):
        x, y = to_xy(lons, np.full_like(lons, lat))
        xs.append(x); ys.append(y)

    grid = plot.multi_line(xs=xs, ys=ys, color=color, line_width=line_width, alpha=alpha,
                           level='underlay', tags=[_BASEMAP_TAG])

    if label_meridians:
        def fold(lon):
            return ((lon + 180.0) % 360.0) - 180.0
        # Labels sit just *inside* the frame. Outside it they get clipped away, and on a
        # pseudocylindrical the widest parallel touches the frame edge exactly, so the
        # equator label is the first to disappear.
        # Meridians along the bottom, every other one so they do not collide where
        # Robinson's outline narrows; parallels down the left edge.
        mer = np.arange(west, wrap_longitude + 1e-9, 2 * lon_step)
        mx, my = to_xy(mer, np.full_like(mer, -90.0))
        par = np.arange(-60.0, 60.0 + 1e-9, lat_step)
        px, py = to_xy(np.full_like(par, west), par)
        source = ColumnDataSource(dict(
            x=np.concatenate([mx, px]), y=np.concatenate([my, py]),
            text=[f"{fold(v):.0f}" for v in mer] + [f"{v:.0f}" for v in par],
            x_off=[0] * len(mer) + [5] * len(par),
            y_off=[9] * len(mer) + [0] * len(par),
            align=['center'] * len(mer) + ['left'] * len(par)))
        plot.add_layout(LabelSet(x='x', y='y', text='text', source=source,
                                 x_offset='x_off', y_offset='y_off', text_align='align',
                                 text_font_size='9px', text_color=color,
                                 text_baseline='middle'))

    _exclude_basemap_from_autorange(plot)
    return grid

def add_basemap(plot, projection: str = 'equirectangular', wrap_longitude: float = 75.0,
                resolution: str = '110m', ocean_color: str = '#cfd1d6',
                land_color: str = '#efefef', outline_color: str = '#666666',
                outline_width: float = 0.5, outline_alpha: float = 0.7,
                graticule: bool = None):
    """
    Draw a vector basemap -- ocean background, filled land, country outlines, and for
    projections whose axes are not lon/lat, a graticule.

    :param plot: the bokeh figure to draw on.
    :param projection: one of PROJECTIONS; must match what the tracks were drawn with.
    :param wrap_longitude: longitude of the map seam, in degrees.
    :param resolution: Natural Earth resolution passed to the land and outline layers.
    :param ocean_color: figure background fill, i.e. the ocean.
    :param land_color: land fill color.
    :param outline_color: country outline color.
    :param outline_width: country outline width in pixels.
    :param outline_alpha: country outline opacity.
    :param graticule: draw meridians/parallels. Defaults to on for every projection whose
                      axes are not already lon/lat degrees.

    :return: Returns the (land, outlines, graticule) GlyphRenderers; graticule is None if
             it was not drawn.
    """
    if graticule is None:
        graticule = projection != 'equirectangular'

    plot.background_fill_color = ocean_color
    land = add_land_fill(plot, projection=projection, wrap_longitude=wrap_longitude,
                         resolution=resolution, color=land_color)
    outlines = add_country_outlines(plot, projection=projection,
                                    wrap_longitude=wrap_longitude, resolution=resolution,
                                    color=outline_color, line_width=outline_width,
                                    alpha=outline_alpha)
    grid = add_graticule(plot, projection=projection,
                         wrap_longitude=wrap_longitude) if graticule else None
    return land, outlines, grid

# Position variable pairs, best first. The names changed over RAF's history:
#   GGLAT/GGLON  GPS (2000s on)
#   LATC/LONC    GPS-corrected inertial position (1990s on)
#   XLATC/XLONC  GPS-corrected inertial position (some 1990s projects)
#   GLAT/GLON    GPS (from ~1990; patchy in the early years)
#   LAT/LON      inertial position (1990s), or the only position in 1970s-early 80s data
#   ALAT/ALON    inertial position (1980s)
#   CLAT/CLON    LORAN-C position (1980s)
#   VLAT/VLON    VLF/Omega navigation position (1970s GENPRO files; VLONG there)
POSITION_VARS = (('GGLAT', 'GGLON'), ('LATC', 'LONC'), ('XLATC', 'XLONC'), ('GLAT', 'GLON'),
                 ('LAT', 'LON'), ('ALAT', 'ALON'), ('CLAT', 'CLON'), ('VLAT', 'VLON'))

def _drop_jumps(lat: np.ndarray, lon: np.ndarray, max_step: float = 0.5,
                max_offset: float = 1.0, min_fragment: float = 0.05):
    """
    Remove position glitches by track continuity. Older inertial/LORAN/early-GPS positions
    hold isolated spikes and short runs of garbage (e.g. longitude 0 for a few minutes, or
    a position drifting near (0, 0) after an INS reset) that draw long straight lines.

    The track is cut wherever two consecutive valid samples are further apart than
    max_offset + max_step * (number of rows between them) degrees (great-circle-ish:
    longitude is scaled by cos(latitude)). Rows are evenly spaced in time, so this is a
    speed limit that allows for data gaps; with the default 0.5 deg per row it is far
    above any aircraft's speed even at 60 s per row. Fragments holding less than
    min_fragment of the flight's valid samples are taken to be glitches and set to NaN,
    and where two kept fragments meet, the jump between them is left as a gap.

    :return: Returns (lat, lon) copies with glitches set to NaN.
    """
    lat, lon = lat.copy(), lon.copy()
    valid = np.flatnonzero(np.isfinite(lat) & np.isfinite(lon))
    if valid.size < 3:
        return lat, lon
    vlat, vlon = lat[valid], lon[valid]
    dlon = (np.diff(vlon) + 180.0) % 360.0 - 180.0
    dist = np.hypot(np.diff(vlat), dlon * np.cos(np.deg2rad(0.5 * (vlat[1:] + vlat[:-1]))))
    allowed = max_offset + max_step * np.diff(valid)
    cuts = np.flatnonzero(dist > allowed) + 1
    if cuts.size == 0:
        return lat, lon
    for start, end in zip(np.r_[0, cuts], np.r_[cuts, valid.size]):
        if end - start < min_fragment * valid.size:
            lat[valid[start:end]] = np.nan
            lon[valid[start:end]] = np.nan
        elif end < valid.size:
            # keep this fragment, but don't draw the jump to whatever follows it
            lat[valid[end - 1]] = np.nan
            lon[valid[end - 1]] = np.nan
    return lat, lon

def flight_position(df: pd.DataFrame, min_valid: float = 0.5):
    """
    Pick a flight's latitude/longitude from whichever position variables were read.

    Pairs are tried in POSITION_VARS order, and the first one with at least min_valid of
    its samples valid is used. Early GPS (GLAT/GLON, ~1990-93) in particular can be
    missing for most of a flight, in which case the inertial position is used instead. If
    no pair reaches min_valid, the pair with the most valid samples is used.

    Invalid samples become NaN, which bokeh draws as a gap: NaNs, latitudes beyond +/-90,
    longitudes beyond +/-360 (fill values in older files), positions within 10 deg of
    (0, 0), and latitude exactly equal to longitude (garbage in some 1980s files). Older
    files record (0, 0) during GPS/INS dropouts and after the data system stopped (e.g.
    SHEBA RF04's last 6 h), and an inertial position reset to (0, 0) drifts from there
    for hours (e.g. SMILE, TTT); left in, these draw lines to the Gulf of Guinea. No NCAR
    aircraft campaign has flown within 10 deg of (0, 0): every sample there through 2026
    is one of these artifacts. Isolated jumps are then removed by _drop_jumps.

    :param df: one flight's DataFrame, as read by util.flight_utils.
    :param min_valid: fraction of valid samples a pair needs to be accepted outright.

    :return: Returns (latitude, longitude) numpy arrays, or None if no position variables
             were read.
    """
    best = None
    for lat_name, lon_name in POSITION_VARS:
        if lat_name not in df.columns or lon_name not in df.columns:
            continue
        lat = np.ravel(df[lat_name].to_numpy(dtype=float))
        lon = np.ravel(df[lon_name].to_numpy(dtype=float))
        with np.errstate(invalid='ignore'):
            valid = (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90.0)
                     & (np.abs(lon) <= 360.0)
                     & ~((np.abs(lat) < 10) & (np.abs((lon + 180.0) % 360.0 - 180.0) < 10))
                     & (lat != lon))
        lat = np.where(valid, lat, np.nan)
        lon = np.where(valid, lon, np.nan)
        fraction = valid.mean() if valid.size else 0.0
        if fraction >= min_valid:
            return _drop_jumps(lat, lon)
        if best is None or fraction > best[2]:
            best = (lat, lon, fraction)
    return None if best is None else _drop_jumps(*best[:2])

def _campaign_track(campaign: str, flights: dict[str, pd.DataFrame], to_xy, seam_jump: float):
    """
    Project every flight of a campaign and join them into one NaN-separated track, so the
    whole campaign is a single bokeh glyph. One glyph per flight looks the same but makes
    the saved notebook several times larger (every glyph carries its own model objects).

    :return: Returns projected (x, y) arrays, or None if no flight had a position.
    """
    xs, ys = [], []
    for flight, df in flights.items():
        position = flight_position(df)
        if position is None:
            print(f"  skipping {campaign}/{flight}: no position variables were read")
            continue
        latitude, longitude = position
        #  ARISTO2017 only had high-rate data, with a handful of bad lat/lons
        if campaign == "ARISTO2017":
            bad = latitude < 20
            latitude = np.where(bad, np.nan, latitude)
            longitude = np.where(bad, np.nan, longitude)
        # project, putting the map's left/right seam at wrap_longitude. A seam at
        # +75 deg has no flights on it, which centers the figure on the Pacific and
        # keeps dateline-crossing flights continuous. Anything that does cross the
        # seam is broken there rather than drawn straight across the figure.
        x, y = to_xy(longitude, latitude)
        x, y = _split_at_seam(x, y, seam_jump)
        xs += [x, [np.nan]]
        ys += [y, [np.nan]]
    if not xs:
        return None
    # float32 is ~1 m precision even in Robinson's metres, and halves the data bokeh embeds
    return (np.concatenate(xs[:-1]).astype(np.float32),
            np.concatenate(ys[:-1]).astype(np.float32))

def _map_figure(projection: str, wrap_longitude: float, width: int, height: int):
    """
    Make an empty map figure for plot_campaigns and friends.

    :return: Returns (plot, to_xy, seam_jump).
    """
    # Mercator stretched the poles so badly that this polar-heavy dataset (lat -82 to +87)
    # came out taller than it was wide. Both projections here roughly halve that aspect
    # ratio: equirectangular plots lon/lat degrees directly, robinson is pseudocylindrical
    # (plot coordinates are metres) and trades straight meridians for a rounder globe.
    to_xy, x_bounds, y_bounds, seam_jump = _projector(projection, wrap_longitude)

    # Neither projection's aspect ratio is locked to the figure's: bokeh maps each axis
    # range onto the frame independently, so shrinking height squashes the map vertically
    # rather than cropping it. Handy for fitting a wide figure, at the cost of geographic
    # fidelity -- shapes stop being true to the projection.
    plot = figure(width=width, height=height)

    if projection == 'equirectangular':
        plot.add_layout(Title(text="Longitude [Degrees]", align="center"), "below")
        plot.add_layout(Title(text="Latitude [Degrees]", align="center"), "left")
        # Tracks run from wrap_longitude - 360 to wrap_longitude, so tick values run below
        # -180. Fold the labels back into [-180, 180] so e.g. 110 E reads as 110, not -250.
        plot.xaxis.formatter = CustomJSTickFormatter(code="""
            return (((tick + 180.0) % 360.0 + 360.0) % 360.0 - 180.0).toFixed(0);
        """)
        # Stop panning/zooming at one world width, so the map can't scroll into a second
        # copy of the continents. Bounds are the seam meridian either side of the map.
        plot.x_range.bounds = x_bounds
        plot.y_range.bounds = y_bounds
    else:
        # Plot coordinates are projected metres, so the axes carry no readable lon/lat.
        # add_basemap draws a graticule instead.
        plot.xaxis.visible = False
        plot.yaxis.visible = False
        plot.xgrid.grid_line_color = None
        plot.ygrid.grid_line_color = None
        plot.outline_line_color = None
        # Frame the whole globe rather than auto-ranging onto the tracks: cropping a
        # pseudocylindrical cuts through its curved outline, and it would push the
        # graticule labels at the poles and seam off the figure.
        plot.x_range = Range1d(*x_bounds, bounds=x_bounds)
        plot.y_range = Range1d(*y_bounds, bounds=y_bounds)
    return plot, to_xy, seam_jump

def plot_campaign_groups(all_campaign_dfs: dict[str,dict[str,pd.DataFrame]],
                         campaign_groups: dict[str,str], show_plot: bool = True,
                         wrap_longitude: float = 50.0, projection: str = 'equirectangular',
                         width: int = 1000, height: int = 450, palette: list[str] = None,
                         line_width: float = 1, title: str = None):
    """
    Plot every campaign's flight tracks colored by group (e.g. decade or aircraft), with
    one legend entry per group rather than per campaign. Useful when there are too many
    campaigns for a readable legend.

    :param all_campaign_dfs: campaign -> flight -> DataFrame, as read by util.flight_utils.
    :param campaign_groups: campaign -> group label. Groups are colored, layered (later
                            groups on top) and listed in the legend in order of first
                            appearance in this dictionary. Campaigns missing from it are
                            not plotted.
    :param palette: colors for the groups, in order. Defaults to Category10.
    :param title: optional figure title.
    """
    plot, to_xy, seam_jump = _map_figure(projection, wrap_longitude, width, height)
    if title:
        plot.title.text = title

    groups = list(dict.fromkeys(campaign_groups.values()))
    colors = palette or Category10[10]
    for group, color in zip(groups, itertools.cycle(colors)):
        xs, ys = [], []
        for campaign, flights in all_campaign_dfs.items():
            if campaign_groups.get(campaign) != group:
                continue
            track = _campaign_track(campaign, flights, to_xy, seam_jump)
            if track is not None:
                xs += [track[0], [np.nan]]
                ys += [track[1], [np.nan]]
        if xs:
            plot.line(np.concatenate(xs).astype(np.float32),
                      np.concatenate(ys).astype(np.float32), color=color,
                      line_width=line_width, legend_label=group)

    plot.legend.click_policy = 'hide'
    plot.legend.label_text_font_size = '10px'
    add_basemap(plot, projection=projection, wrap_longitude=wrap_longitude)
    plot.add_layout(plot.legend[0], 'right')

    if show_plot:
        show(plot)
    else:
        return plot

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

def plot_campaigns(all_campaign_dfs: dict[str,dict[str,pd.DataFrame]], show_plot: bool = True,
                   wrap_longitude: float = 50.0,
                   projection: str = 'equirectangular',
                   width: int = 1000, height: int = 450,
                   palette: list[str] = None, legend_ncols: int = 2,
                   legend_font_size: str = '7px', title: str = None):
    """
    Plot every campaign's flight tracks on one map, one color and legend entry per campaign.

    :param palette: colors cycled through campaign by campaign. Defaults to Category10[8].
    :param legend_ncols: number of legend columns.
    :param legend_font_size: legend label font size, e.g. '7px'.
    :param title: optional figure title.
    """
    plot, to_xy, seam_jump = _map_figure(projection, wrap_longitude, width, height)
    if title:
        plot.title.text = title

    colors = itertools.cycle(palette or Category10[8])
    for campaign, flights in all_campaign_dfs.items():
        color = next(colors)
        track = _campaign_track(campaign, flights, to_xy, seam_jump)
        if track is not None:
            plot.line(*track, color=color, legend_label=campaign)

    plot.legend.click_policy = 'hide'
    plot.legend.label_text_font_size = legend_font_size
    plot.legend.glyph_height = 8
    plot.legend.label_height = 8
    plot.legend.ncols = legend_ncols
    add_basemap(plot, projection=projection, wrap_longitude=wrap_longitude)
    plot.add_layout(plot.legend[0],'right')

    if show_plot:
        show(plot)
    else:
        return plot

def plot_campaigns_custom(all_campaign_dfs: dict[str,dict[str,pd.DataFrame]], show_plot: bool = True,
                          wrap_longitude: float = 75.0,
                          projection: str = 'equirectangular',
                          width: int = 1000, height: int = 550):

    plot, to_xy, seam_jump = _map_figure(projection, wrap_longitude, width, height)

    #colors = itertools.cycle(Category10[10])
    colors = itertools.cycle(Category20[20])
    for campaign, flights in all_campaign_dfs.items():
        if campaign in ['SOCRATES','RICO','VOCALS','HIPPO-1','HIPPO-2','HIPPO-3','HIPPO-4','CAESAR','ORCAS','CSET']:
            color=next(colors)
            line_width=1.5
        else:
            color='gray'
            line_width=1
        track = _campaign_track(campaign, flights, to_xy, seam_jump)
        if track is not None:
            plot.line(*track, color=color, legend_label=campaign, line_width=line_width)

    plot.legend.click_policy = 'hide'
    plot.legend.label_text_font_size = '10px'
    plot.legend.ncols = 2
    add_basemap(plot, projection=projection, wrap_longitude=wrap_longitude)
    plot.add_layout(plot.legend[0],'right')

    if show_plot:
        show(plot)
    else:
        return plot
