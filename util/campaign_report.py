"""
A plain-text report of field campaigns -- dates, aircraft, flights, flight hours and the
area flown -- built from the flight data read by util.flight_utils.read_all_flights_from_dirs.

Everything is derived from the flight files themselves:
  * Flights: a flight's data can be split over several files (RF01A, RF01B, ...), or one
    flight can sit in both a high- and a low-rate directory, so per aircraft the files are
    put in time order and merged wherever one starts within MERGE_GAP of the previous
    one's end. Each merged run counts as one flight, typed research/ferry/test from RAF's
    file names (RFnn/FFnn/TFnn).
  * Flight hours: the summed duration of the merged flights.
  * Area: the share of each flight's samples over each US state, country or named ocean/sea
    (Natural Earth boundaries; research flights only where flights are typed, so ferries
    don't dominate). Only the larger shares are listed.
"""
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import shapely
import cartopy.io.shapereader as shapereader

from util.plot_utils import flight_position

MERGE_GAP = timedelta(minutes=30)
MAX_FLIGHT = timedelta(hours=24)   # longer "flights" come from bad times; hours not counted

# Aircraft by the platform directory name EOL's archive uses
AIRCRAFT_DIRS = {
    'electra_n595kr': 'Electra N595KR', 'electra_n308d': 'Electra N308D',
    'queenaira80_n304d': 'Queen Air N304D', 'queenair_n304d': 'Queen Air N304D',
    'queenairb80_n306d': 'Queen Air N306D', 'queenair_n306d': 'Queen Air N306D',
    'sabreliner_n307d': 'Sabreliner N307D', 'kingair_n312d': 'King Air N312D',
    'c130_n130ar': 'C-130 N130AR', 'gv_n677f': 'GV N677F',
}


def flight_aircraft(path: str, default: str) -> str:
    """The aircraft a flight file belongs to, from the '.../aircraft/<platform>/...' part of
    its path, or default if the path doesn't say."""
    m = re.search(r'/aircraft/([^/]+)/', path or '')
    return AIRCRAFT_DIRS.get(m.group(1).lower(), default) if m else default


# ---------------------------------------------------------------- flights

def merge_flights(flights: dict[str, pd.DataFrame], default_aircraft: str) -> list[dict]:
    """
    Merge a campaign's flight files into flights (see the module docstring).

    :return: Returns a list of dicts with 'aircraft', 'start', 'end' (None if the files
             had no readable time), 'type' and 'keys' (the flight files merged).
    """
    timed, untimed = [], []
    for key, df in flights.items():
        seg = dict(aircraft=flight_aircraft(df.attrs.get('path'), default_aircraft),
                   start=df.attrs.get('start'), end=df.attrs.get('end'),
                   type=df.attrs.get('flight_type', 'unknown'), keys=[key])
        (timed if seg['start'] is not None else untimed).append(seg)
    merged = []
    for seg in sorted(timed, key=lambda s: (s['aircraft'], s['start'])):
        last = merged[-1] if merged else None
        if (last is not None and last['aircraft'] == seg['aircraft']
                and seg['start'] <= last['end'] + MERGE_GAP):
            last['end'] = max(last['end'], seg['end'])
            last['keys'] += seg['keys']
            if last['type'] == 'unknown':
                last['type'] = seg['type']
        else:
            merged.append(dict(seg, keys=list(seg['keys'])))
    return merged + untimed


# ---------------------------------------------------------------- area

class Regions:
    """Point-in-polygon lookup of US states, countries and named seas (Natural Earth)."""

    def __init__(self):
        labels, geoms = [], []

        def add(resolution, category, name, label):
            for rec in shapereader.Reader(shapereader.natural_earth(resolution, category, name)).records():
                text = label(rec.attributes)
                if text and rec.geometry is not None:
                    labels.append(text)
                    geoms.append(rec.geometry)

        # US states first: a point in a state is reported as the state, not as the USA
        add('50m', 'cultural', 'admin_1_states_provinces_lakes',
            lambda a: f"{a['name']}, USA" if a['admin'] == 'United States of America' else None)
        add('50m', 'cultural', 'admin_0_countries',
            lambda a: None if a['ADMIN'] == 'United States of America' else a['NAME'])
        add('50m', 'physical', 'geography_marine_polys',
            lambda a: _sea_name(a['name']) if a['featurecla'] not in ('river', 'reef') else None)
        self.labels = np.array(labels, dtype=object)
        self.geoms = geoms
        self.tree = shapely.STRtree(geoms)
        # coarse label: a US state's country
        self.coarse = np.array([('USA' if l.endswith(', USA') else l) for l in self.labels], dtype=object)

    def lookup(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Index into self.labels of each point's region; points in no polygon (e.g. just
        off a coast) take the nearest region within 3 deg, else -1."""
        points = shapely.points(lon, lat)
        none = len(self.geoms)
        idx = np.full(len(points), none)
        p, g = self.tree.query(points, predicate='intersects')
        np.minimum.at(idx, p, g)   # first match wins: states before countries before seas
        idx[idx == none] = -1
        missing = np.flatnonzero(idx < 0)
        if missing.size:
            p, g = self.tree.query_nearest(points[missing], max_distance=3.0, all_matches=False)
            idx[missing[p]] = g
        return idx


def _sea_name(name: str) -> str:
    """Natural Earth spells a few oceans in capitals ('SOUTHERN OCEAN')."""
    return name.title() if name.isupper() else name


def describe_area(lon: np.ndarray, lat: np.ndarray, regions: Regions, max_items: int = 5,
                  min_share: float = 0.05) -> str:
    """
    Describe where a campaign flew as its largest regions and their shares of flight time,
    e.g. 'Colorado, USA 71%; Wyoming, USA 18%'. US states are listed if a few of them cover
    most of the flying; otherwise states are combined into 'USA'.
    """
    if lon.size == 0:
        return 'no position data'
    # one lookup per 0.1 deg cell, weighted by the samples in it
    cells, counts = np.unique(np.round(np.c_[lon, lat], 1), axis=0, return_counts=True)
    idx = regions.lookup(cells[:, 0], cells[:, 1])
    total = counts.sum()
    for names in (regions.labels, regions.coarse):
        share = pd.Series(counts, index=[names[i] if i >= 0 else 'unidentified' for i in idx])
        share = share.groupby(level=0).sum().sort_values(ascending=False) / total
        top = share[share >= min_share][:max_items]
        if top.sum() >= 0.6:
            break
    items = [f"{name} {100 * s:.0f}%" for name, s in top.items()]
    rest = share.drop(top.index)
    if rest.sum() >= 0.005:
        items.append(f"{len(rest)} other region{'s' if len(rest) > 1 else ''} {100 * rest.sum():.0f}%")
    return '; '.join(items)


def _extent(lon: np.ndarray, lat: np.ndarray) -> str:
    """Latitude/longitude range, taking the shorter way round in longitude."""
    if lon.size == 0:
        return ''
    lon = (lon + 180.0) % 360.0 - 180.0
    west, east = lon.min(), lon.max()
    shifted = lon % 360.0                       # 0..360: continuous across the dateline
    if shifted.max() - shifted.min() < east - west:
        west, east = shifted.min(), shifted.max()
        west, east = (west + 180) % 360 - 180, (east + 180) % 360 - 180

    def fmt(v, pos, neg):
        return f"{abs(v):.1f}{pos if v >= 0 else neg}"
    return (f"lat {fmt(lat.min(), 'N', 'S')} to {fmt(lat.max(), 'N', 'S')}, "
            f"lon {fmt(west, 'E', 'W')} to {fmt(east, 'E', 'W')}")


# ---------------------------------------------------------------- report

def campaign_summaries(all_campaign_dfs: dict[str, dict[str, pd.DataFrame]],
                       campaigns: list[tuple], regions: Regions = None) -> list[dict]:
    """
    Summarize each campaign of all_campaign_dfs.

    :param all_campaign_dfs: campaign -> flight -> DataFrame, from read_all_flights_from_dirs.
    :param campaigns: (name, first flight 'YYYY-MM-DD', aircraft, location) tuples, as in
                      plot_fieldcampaigns_archive; gives the order, the fallback date and
                      the aircraft of flights whose path doesn't name one.
    """
    regions = regions or Regions()
    summaries = []
    for name, first_flight, aircraft, location in campaigns:
        flights = all_campaign_dfs.get(name)
        if not flights:
            continue
        merged = merge_flights(flights, aircraft)
        starts = [f['start'] for f in merged if f['start'] is not None]
        durations = [f['end'] - f['start'] for f in merged if f['start'] is not None]
        good = [d for d in durations if d <= MAX_FLIGHT]
        types = pd.Series([f['type'] for f in merged]).value_counts()

        # positions, from research flights where flights are typed
        research = {k for f in merged if f['type'] == 'research' for k in f['keys']}
        lons, lats = [], []
        for key, df in flights.items():
            if research and key not in research:
                continue
            pos = flight_position(df)
            if pos is not None:
                ok = np.isfinite(pos[0]) & np.isfinite(pos[1])
                lats.append(pos[0][ok])
                lons.append(pos[1][ok])
        lon = np.concatenate(lons) if lons else np.array([])
        lat = np.concatenate(lats) if lats else np.array([])

        paths = [df.attrs.get('path', '') for df in flights.values()]
        dirs = list(dict.fromkeys(os.path.dirname(p) for p in paths))
        # data formats: netCDF, KOFSE's DAP text listings, else GENPRO (no extension)
        formats = list(dict.fromkeys('netCDF' if p.endswith(('.nc', '.cdf')) else
                                     'text listings' if p.endswith('.txt') else 'GENPRO'
                                     for p in paths))
        flight_aircraft_names = list(dict.fromkeys(f['aircraft'] for f in merged))
        summaries.append(dict(
            name=name, decade=first_flight[:3] + '0s',
            first=min(starts) if starts else datetime.fromisoformat(first_flight),
            last=max(f['end'] for f in merged if f['end'] is not None) if starts else None,
            aircraft=', '.join(flight_aircraft_names) or aircraft,
            per_aircraft={a: sum(f['aircraft'] == a for f in merged) for a in flight_aircraft_names},
            n_flights=len(merged), types=types.to_dict(),
            hours=sum(good, timedelta()).total_seconds() / 3600, n_bad_times=len(durations) - len(good),
            n_untimed=len(merged) - len(starts),
            area=describe_area(lon, lat, regions), extent=_extent(lon, lat),
            dirs=dirs, formats=formats, n_files=len(flights)))
    return summaries


def _flights_text(s: dict) -> str:
    parts = [f"{s['types'][t]} {t}" for t in ('research', 'ferry', 'test') if t in s['types']]
    if 'unknown' in s['types']:
        parts.append(f"{s['types']['unknown']} not typed")
    text = f"{s['n_flights']} ({', '.join(parts)})"
    if len(s['per_aircraft']) > 1:
        text += '; ' + ', '.join(f"{a}: {n}" for a, n in s['per_aircraft'].items())
    return text


def write_campaign_report(all_campaign_dfs: dict[str, dict[str, pd.DataFrame]],
                          campaigns: list[tuple], path: str, title: str = None,
                          notes: str = '') -> list[dict]:
    """
    Write a plain-text report of every campaign: a summary by decade and by aircraft, an
    index with one line per campaign, then each campaign's details.

    :param path: the text file to write.
    :param title: first line of the report.
    :param notes: extra text for the report's introduction.

    :return: Returns the per-campaign summaries (see campaign_summaries).
    """
    summaries = campaign_summaries(all_campaign_dfs, campaigns)
    W = 100
    out = []
    add = out.append
    title = title or 'NCAR Research Aviation Facility field campaigns'
    add(title)
    add('=' * len(title))
    add(f"Written {datetime.now():%Y-%m-%d} from the flight-level data files on glade.")
    if notes:
        add(notes.rstrip())
    add('')
    add("How the numbers are made (see util/campaign_report.py):")
    add("  Flights  data files merged per aircraft where one starts within 30 min of the last")
    add("           one's end (a flight can be split over files). Typed from RAF's file names:")
    add("           RFnn research, FFnn ferry, TFnn test; 'not typed' where the name doesn't say.")
    add("           Counts are of flights with data on glade; a few flights' data are lost.")
    add("  Hours    summed duration of those flights.")
    add("  Area     share of flight time over each US state, country or named sea (research")
    add("           flights only, where typed), listing regions with >=5% of the time.")
    add("           Boundaries and sea names are Natural Earth's (its Indian Ocean, for example,")
    add("           reaches south of Australia to 60S, where the Southern Ocean begins).")
    add('')

    total_h = sum(s['hours'] for s in summaries)
    total_f = sum(s['n_flights'] for s in summaries)
    add(f"{len(summaries)} campaigns, {total_f} flights, {total_h:,.0f} flight hours, "
        f"{summaries[0]['first']:%Y}-{max(s['first'] for s in summaries):%Y}")
    add('')

    def table(groups, heading):
        """groups: (label, [(summary, flights, hours), ...]) pairs"""
        add(f"  {heading:<18} {'Campaigns':>9} {'Flights':>8} {'Hours':>8}   Years")
        for label, rows in groups:
            years = [s['first'].year for s, _, _ in rows]
            add(f"  {label:<18} {len(rows):>9} {sum(f for _, f, _ in rows):>8} "
                f"{sum(h for _, _, h in rows):>8,.0f}   {min(years)}-{max(years)}")
        add('')

    decades = list(dict.fromkeys(s['decade'] for s in summaries))
    add('By decade')
    table([(d, [(s, s['n_flights'], s['hours']) for s in summaries if s['decade'] == d])
           for d in decades], 'Decade')
    # a multi-aircraft campaign's hours are split between its aircraft by flight count
    aircraft = list(dict.fromkeys(a for a in AIRCRAFT_DIRS.values()
                                  if any(a in s['per_aircraft'] for s in summaries)))
    add('By aircraft (hours of multi-aircraft campaigns split by flight count)')
    table([(a, [(s, s['per_aircraft'][a], s['hours'] * s['per_aircraft'][a] / s['n_flights'])
                for s in summaries if a in s['per_aircraft']]) for a in aircraft], 'Aircraft')

    add('INDEX')
    add('-' * W)
    add(f"{'Year':<5} {'Campaign':<27} {'Aircraft':<34} {'Flts':>4} {'Hours':>6}  Main area")
    for s in summaries:
        main = s['area'].split(';')[0]
        add(f"{s['first']:%Y}  {s['name'][:27]:<27} {s['aircraft'][:34]:<34} {s['n_flights']:>4} "
            f"{s['hours']:>6.0f}  {main}")
    add('')

    for decade in decades:
        ss = [s for s in summaries if s['decade'] == decade]
        add('')
        add('=' * W)
        add(f"{decade} ({len(ss)} campaigns)")
        add('=' * W)
        for s in ss:
            add('')
            add(s['name'])
            dates = f"{s['first']:%Y-%m-%d}"
            if s['last'] is not None and s['last'].date() != s['first'].date():
                dates += f" to {s['last']:%Y-%m-%d}"
            add(f"  Dates      {dates}")
            add(f"  Aircraft   {s['aircraft']}")
            add(f"  Flights    {_flights_text(s)}")
            hours = f"{s['hours']:.1f}"
            if s['n_untimed'] or s['n_bad_times']:
                hours += f" (not counting {s['n_untimed'] + s['n_bad_times']} flight(s) without good times)"
            add(f"  Hours      {hours}")
            add(f"  Area       {s['area']}")
            if s['extent']:
                add(f"  Extent     {s['extent']}")
            label = 'Data'
            for d in s['dirs']:
                add(f"  {label:<10} {d}")
                label = ''
            add(f"  {'':<10} ({s['n_files']} files/segments: {' and '.join(s['formats'])})")
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    return summaries
