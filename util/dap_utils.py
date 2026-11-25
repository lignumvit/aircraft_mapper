"""
Reader for the text position listings of KOFSE (Kuwait Oil-Fire Smoke Experiment, 1991)
in /glade/campaign/eol/data/kofse/aircraft/DAP. KOFSE's data were processed with DAP
rather than RAF's usual software, and the archive keeps only the raw tapes, so these
listings are the only processed Electra positions on glade. A listing looks like

    FLIGHT NUMBER 12
    DATE:  6/1/91

    time          ALATC         ALONC
     UCT
    ------------  ------------  ------------
        7:38:06   2.62848e+01   5.05206e+01
        ...

one row every 10 s. The position columns are named differently in some listings (ALATC/
ALONC, ILATC/ILONC, ILATL/ILONL); they are read as LAT/LON. The header date is wrong in a few files (KFN02 and KFN03 repeat
KFN12's header; KFC17's year is cut to "19"). The same listings are also filed in
files_by_date/<DDMM>/<name>.OUT, so where that copy exists, its folder gives the day and
month, and the year is the one most of the directory's listings give.
"""
import collections
import glob
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from util.genpro_utils import time_span

SPLIT_GAP = 30 * 60   # s; a longer gap in a listing separates two flights


def is_dap(path: str) -> bool:
    """True if path looks like a DAP position listing."""
    try:
        with open(path, 'r', errors='replace') as f:
            head = f.read(600)
    except OSError:
        return False
    return 'FLIGHT NUMBER' in head and 'DATE:' in head and re.search(r'\btime\b', head) is not None


def _header_date(text: str):
    """(month, day, 2-digit year) from a listing's 'DATE: m/d/yy' line, or None."""
    m = re.search(r'DATE:\s*(\d{1,2})/(\d{1,2})/(\d{2})', text)
    return tuple(int(x) for x in m.groups()) if m else None


def flight_date(path: str, text: str):
    """The flight's date (see the module docstring), or None."""
    header = _header_date(text)
    folder = os.path.dirname(path)
    years = collections.Counter()
    for other in glob.glob(os.path.join(folder, '*.txt')):
        with open(other, 'r', errors='replace') as f:
            d = _header_date(f.read(300))
        if d:
            years[d[2]] += 1
    yy = years.most_common(1)[0][0] if years else (header[2] if header else None)
    if yy is None:
        return None
    year = 1900 + yy if yy >= 50 else 2000 + yy
    name = os.path.splitext(os.path.basename(path))[0]
    copies = glob.glob(os.path.join(folder, 'files_by_date', '[0-9]' * 4, name + '.OUT'))
    if copies:
        ddmm = os.path.basename(os.path.dirname(copies[0]))
        return datetime(year, int(ddmm[2:]), int(ddmm[:2]))
    return datetime(year, header[0], header[1]) if header else None


def read_dap_flights(path: str, read_vars: list[str], period_s: float = 0) -> dict[str, pd.DataFrame]:
    """
    Read a DAP position listing into one DataFrame per flight, in the form
    util.flight_utils.read_all_flights_from_dirs gives netCDF flights: the columns of
    read_vars present in the listing (its position as LAT/LON), one row per listed time (10 s), or per period_s
    seconds. A listing is split into flights at gaps of more than 30 minutes (KFN1516
    holds flights 15 and 16). Each frame's attrs hold 'start' and 'end' datetimes.

    :return: Returns {label: DataFrame}; the label is the file name, with '#2', ... added
             for later flights in the same listing.
    """
    with open(path, 'r', errors='replace') as f:
        text = f.read()
    m = re.search(r'^\s*time\s+(.*)$', text, re.M)
    if not m:
        raise ValueError(f"{path} is not a DAP position listing")
    columns = m.group(1).split()
    # the position columns, whatever the listing calls them, are LAT and LON
    columns = ['LAT' if 'LAT' in c.upper() else 'LON' if 'LON' in c.upper() else c for c in columns]
    rows = re.findall(r'^\s*(\d+):(\d\d):(\d\d)((?:\s+\S+){%d})\s*$' % len(columns), text, re.M)
    seconds = np.array([int(h) * 3600 + int(mi) * 60 + int(s) for h, mi, s, _ in rows], float)
    values = np.array([v.split() for *_, v in rows], float).reshape(len(rows), len(columns))
    # times past midnight UTC restart at 0:00:00
    seconds += 86400 * np.cumsum(np.r_[0, np.diff(seconds) < -12 * 3600])
    date = flight_date(path, text)
    dt = np.median(np.diff(seconds)) if len(seconds) > 1 else 10.0
    stride = max(1, round(period_s / dt)) if period_s > 0 else 1

    flights = {}
    breaks = np.flatnonzero(np.diff(seconds) > SPLIT_GAP) + 1
    for i, (a, b) in enumerate(zip(np.r_[0, breaks], np.r_[breaks, len(seconds)])):
        df = pd.DataFrame({c: values[a:b, j] for j, c in enumerate(columns) if c in read_vars})
        df = df.iloc[::stride].reset_index(drop=True)
        df.attrs['start'], df.attrs['end'] = time_span(date, seconds[a:b])
        label = os.path.basename(path) + (f"#{i + 1}" if i else '')
        flights[label] = df
    return flights
