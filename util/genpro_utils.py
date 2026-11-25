"""
Readers for RAF's GENPRO flight-data files, the processed output RAF produced before
netCDF (1970s-early 1990s). RAF converted most GENPRO data to netCDF; these readers are for
the few campaigns on glade whose data exist only as GENPRO (e.g. STRATEX, 1976), or whose
netCDF conversion is bad (OPUS-II, 1983).

Two generations of the format are handled. Both are stored on glade as Cray COS-blocked
files (8-byte block/record control words around the data):

GENPRO-I (1970s-early 80s, written on NCAR's CDC 7600)
    Each record is a stream of 60-bit CDC words. The first record of each flight is a text
    header in 6-bit CDC display code, which describes the data:
        "... THE 33 CYCLES OF 45 SAMP/CYC = 1485 WERE THEN SCALED INTO 20 BIT INTEGERS AND
         PACKED 3 SAMPLES/WORD INTO 495 60 BIT PACKED WORDS ..."
    followed by one line per parameter:
        "48)   1     RAW INS LATITUDE    DEG     ALAT     DEGS = (N/ 1000.0) -  200.0"
    i.e. index, samples per cycle (a cycle is 1 s), title, name, units and the scaling
    value = N/P - AD of each 20-bit integer N. Each data record then holds that many
    cycles, every cycle holding each parameter's samples in header order. Some files
    were run through a character translation to EBCDIC (one 6-bit character -> one byte),
    which is undone here.

GENPRO-II (1980s, written on NCAR's Cray)
    The header is ASCII, 80-character logical lines, from "BEGINHD" to "ENDHD", giving
    LOGBIT (bits per logical record, one record per second), DATLOG (logical records
    per physical record), and for each variable its BITS, FSTBIT (1-based first bit in
    the logical record), TERM and FACTOR: value = N/FACTOR - TERM.

Only the first sample of each second is kept, which is all a map needs.
"""
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# CDC 6-bit display code, as the ASCII characters RAF's EBCDIC translation used
DISPLAY_CODE = ':ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/()$= ,.#[]%"_!&\'?<>@\\^;'
_EBCDIC_TO_DC = np.zeros(256, np.uint8)
for _i, _c in enumerate(DISPLAY_CODE):
    _EBCDIC_TO_DC[_c.encode('cp500')[0]] = _i

# Variable names differ from the netCDF-era ones; map them onto the names
# util.plot_utils.POSITION_VARS knows.
ALIASES = {'ALONG': 'ALON', 'VLONG': 'VLON'}

_MONTHS = {m: i + 1 for i, m in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])}


def cos_records(path: str, max_bytes: int = None) -> list:
    """
    Split a Cray COS-blocked file into its records.

    Every 4096-byte block starts with a block control word, and every record ends with a
    record control word; each control word's low 9 bits count the 8-byte data words up to
    the next control word. End-of-file control words are returned as None (a tape image
    can hold several files).

    :param path: the file to read.
    :param max_bytes: read only this many bytes from the start (enough to sniff a header).

    :return: Returns a list of bytes objects (and None for each end-of-file).
    """
    with open(path, 'rb') as f:
        d = f.read() if max_bytes is None else f.read(max_bytes)
    recs, cur, pos, n = [], bytearray(), 0, len(d)
    while pos + 8 <= n:
        cw = int.from_bytes(d[pos:pos + 8], 'big')
        kind, fwi = cw >> 60, cw & 0x1ff
        if kind == 0x8:              # end of record; drop its unused trailing bits
            ubc = (cw >> 54) & 0x3f
            if ubc:
                cur = cur[:len(cur) - ubc // 8]
            recs.append(bytes(cur))
            cur = bytearray()
        elif kind == 0xE:            # end of file
            recs.append(None)
        elif kind == 0xF:            # end of data
            break
        elif kind != 0x0:            # not a block control word: not COS blocked
            raise ValueError(f"{path} is not a COS-blocked file")
        pos += 8
        cur += d[pos:pos + 8 * fwi]
        pos += 8 * fwi
    return recs


def _bits_to_ints(bits: np.ndarray, nbit: int) -> np.ndarray:
    """Read rows of nbit bits (most significant first) as unsigned integers."""
    return bits.reshape(-1, nbit).astype(np.int64).dot(1 << np.arange(nbit - 1, -1, -1, dtype=np.int64))


def _parse_date(text: str):
    """First date like '05JUN76', '16JULY76', '5 MAY 79' or '02 JUL 85' in text, or None."""
    m = re.search(r'(?<!\d)(\d{1,2})\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*(\d{2})(?!\d)', text)
    if not m:
        return None
    day, month, year = int(m.group(1)), _MONTHS[m.group(2)], 1900 + int(m.group(3))
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


# ---------------------------------------------------------------- GENPRO-I

_G1_PARAM = re.compile(r'(\d+)\)\s*(\d+)\s+(.{20,52}?)\s(\S+)\s+[^=]{0,12}= \(N/\s*([\d.]+)\)\s*-\s*([\d.]+)')
_G1_LAYOUT = re.compile(r'THE\s*(\d+) CYCLES OF\s*(\d+) SAMP/CYC = *(\d+).*?INTO\s*(\d+) 60 BIT')


def _record_text(rec: bytes, fmt: str) -> str:
    """Decode a record as CDC display code ('cdc') or EBCDIC ('ebcdic') text."""
    if fmt == 'ebcdic':
        return rec.decode('cp500', 'replace')
    bits = np.unpackbits(np.frombuffer(rec, np.uint8))
    codes = _bits_to_ints(bits[:len(bits) // 6 * 6], 6)
    return ''.join(np.array(list(DISPLAY_CODE))[codes])


def _genpro1_header(rec: bytes):
    """Parse a GENPRO-I header record, or return None if rec is not one."""
    for fmt in ('cdc', 'ebcdic'):
        # the layout description comes within the first ~300 characters
        if 'DESCRIPTION OF RECORD' in _record_text(rec[:400], fmt):
            text = _record_text(rec, fmt)
            break
    else:
        return None
    m = _G1_LAYOUT.search(text)
    if not m:
        return None
    ncyc, spc, _, nword = map(int, m.groups())
    params, offset = {}, 0
    for p in _G1_PARAM.finditer(text):
        name = p.group(4)
        params[name] = dict(offset=offset, P=float(p.group(5)), AD=float(p.group(6)))
        offset += int(p.group(2))
    if offset != spc:   # every sample of a cycle must be accounted for
        return None
    return dict(fmt=fmt, title=text[:60].strip(), date=_parse_date(text[:120]),
                ncyc=ncyc, spc=spc, nword=nword, params=params)


def _genpro1_values(rec: bytes, h: dict) -> np.ndarray:
    """Unpack one GENPRO-I data record into an (ncyc, spc) array of 20-bit integers."""
    if h['fmt'] == 'ebcdic':
        codes = _EBCDIC_TO_DC[np.frombuffer(rec[:h['nword'] * 10], np.uint8)]
        bits = np.unpackbits(codes[:, None], axis=1)[:, 2:].ravel()
    else:
        bits = np.unpackbits(np.frombuffer(rec, np.uint8))
    n = _bits_to_ints(bits[:h['nword'] * 60], 20)
    return n[:h['ncyc'] * h['spc']].reshape(h['ncyc'], h['spc'])


def read_genpro1(path: str, read_vars: list[str]) -> list[dict]:
    """
    Read a GENPRO-I file. A file can hold several flights, each starting with its own
    header record.

    :return: Returns a list of segments, one per header: dicts with 'title', 'date'
             (datetime of the flight's day, or None), 'time' (seconds after that midnight,
             one per second) and 'data' (name -> array, for those of read_vars present).
    """
    segments, h, rows, data_len = [], None, [], 0

    def close():
        if h is not None and rows:
            n = np.concatenate(rows)
            data = {}
            for name, p in h['params'].items():
                out = ALIASES.get(name, name)
                if out in read_vars:
                    data[out] = n[:, p['offset']] / p['P'] - p['AD']
            t = h['params'].get('TIME') or h['params'].get('TPTIME')
            time = n[:, t['offset']] / t['P'] - t['AD'] if t else np.arange(len(n), dtype=float)
            segments.append(dict(title=h['title'], date=h['date'], time=time, data=data))

    for rec in cos_records(path):
        if rec is None:
            continue
        # data records all have one length, headers another; only parse the odd ones out
        header = _genpro1_header(rec) if h is None or abs(len(rec) - data_len) > 8 else None
        if header is not None:
            close()
            h, rows = header, []
            data_len = (h['nword'] * 10 if h['fmt'] == 'ebcdic' else
                        (h['nword'] * 60 + 63) // 64 * 8)
            continue
        if h is None:
            continue
        if len(rec) * 8 >= (h['nword'] * 80 if h['fmt'] == 'ebcdic' else h['nword'] * 60):
            rows.append(_genpro1_values(rec, h))
    close()
    return segments


# ---------------------------------------------------------------- GENPRO-II

def _genpro2_header(text: str) -> dict:
    """Parse the ASCII header of a GENPRO-II file (80-character logical lines)."""
    info, order, variables = {}, None, {}
    for line in (text[i:i + 80] for i in range(0, len(text), 80)):
        m = re.match(r'\s*/?\s*(\w+)\s*=\s*(.*)', line)
        if not m:
            continue
        key, val = m.groups()
        if key in ('LOGBIT', 'DATLOG', 'DATSIZ'):
            info[key] = int(val.split()[0])
        elif key == 'PROJECT':
            info['PROJECT'] = val.split('/')[0].strip().strip('"').strip()
        elif key == 'BEGSNP':
            info['BEGSNP'] = [float(x) for x in re.findall(r'[-\d.E+]+', val.split('/')[0])]
        elif key == 'ORDVAR':
            order = [s.strip() for s in val.split(',')]
        elif key == 'LETVAR' and '%FOR,' in val and order:
            body, name = val.rsplit('%FOR,', 1)
            v = variables.setdefault(name.split()[0], {})
            parts = [p.strip() for p in re.findall(r'"[^"]*"|[^,]+', body) if p.strip()]
            for k, p in zip(order, parts):
                if not p.startswith('"'):
                    try:
                        v[k] = float(p)
                    except ValueError:
                        pass
    info['vars'] = variables
    return info


def read_genpro2(path: str, read_vars: list[str]) -> list[dict]:
    """
    Read a GENPRO-II file (one flight, or part of one).

    :return: Returns a one-element list of the same segment dicts read_genpro1 returns.
    """
    recs = [r for r in cos_records(path) if r is not None]
    text = b''.join(r for r in recs[:200] if len(r) == 800).decode('latin1')
    if 'ENDHD' not in text:
        raise ValueError(f"{path}: no GENPRO-II header")
    info = _genpro2_header(text[:text.find('ENDHD')])
    nbits = info['DATLOG'] * info['LOGBIT']
    data_recs = [r for r in recs if len(r) != 800 and len(r) * 8 >= nbits]
    if not data_recs:
        raise ValueError(f"{path}: header does not match its data records")
    bits = np.stack([np.unpackbits(np.frombuffer(r, np.uint8))[:nbits] for r in data_recs])
    bits = bits.reshape(-1, info['LOGBIT'])

    def var(name):
        v = info['vars'][name]
        nbit, first = int(v['BITS']), int(v['FSTBIT']) - 1
        raw = _bits_to_ints(np.ascontiguousarray(bits[:, first:first + nbit]), nbit).astype(float)
        raw[raw == 0] = np.nan       # zero-filled records
        return raw / v['FACTOR'] - v['TERM']

    data = {}
    for name in info['vars']:
        out = ALIASES.get(name, name)
        if out in read_vars:
            data[out] = var(name)
    # PTIME (processor time) or TPTIME (tape time) is seconds after midnight; pick the
    # one that agrees with the header's start time (the other can be a tape counter)
    begin = info.get('BEGSNP', [0, 0, 0])
    begin_s = begin[0] * 3600 + begin[1] * 60 + begin[2]
    time, best = np.arange(len(bits), dtype=float) + begin_s, np.inf
    for name in ('PTIME', 'TPTIME', 'TIME'):
        if name in info['vars']:
            t = var(name)
            first = t[np.isfinite(t)][:1]
            if first.size and abs(first[0] - begin_s) < best:
                time, best = t, abs(first[0] - begin_s)
    project = info.get('PROJECT', '')
    return [dict(title=project, date=_parse_date(project), time=time, data=data)]


# ---------------------------------------------------------------- common

def genpro_format(path: str):
    """
    Return 'genpro1' or 'genpro2' if path is a GENPRO file these readers handle, else None.
    Only the start of the file is read.
    """
    try:
        recs = [r for r in cos_records(path, max_bytes=65536) if r]
    except (OSError, ValueError):
        return None
    if not recs:
        return None
    if b'BEGINHD' in recs[0][:40]:
        return 'genpro2'
    if _genpro1_header(recs[0]) is not None:
        return 'genpro1'
    return None


def time_span(date, time: np.ndarray):
    """
    First and last datetime of a flight from its seconds-after-midnight, ignoring garbage
    times (zero-filled or corrupt records): a time counts only if a neighbouring sample is
    0-120 s from it, as in a running clock.

    :return: Returns (start, end) datetimes, or (None, None).
    """
    t = np.asarray(time, dtype=float)
    if date is None or t.size < 2:
        return None, None
    with np.errstate(invalid='ignore'):
        step = np.diff(t)
        ok_step = (step > 0) & (step <= 120)
        ok = np.zeros(t.size, bool)
        ok[:-1] |= ok_step
        ok[1:] |= ok_step
        ok &= (t >= 0) & (t < 2 * 86400)
    if not ok.any():
        return None, None
    good = t[ok]
    return date + timedelta(seconds=float(good.min())), date + timedelta(seconds=float(good.max()))


def read_genpro_flights(path: str, read_vars: list[str], period_s: float = 0) -> dict[str, pd.DataFrame]:
    """
    Read a GENPRO file into one DataFrame per flight segment, in the form
    util.flight_utils.read_all_flights_from_dirs gives netCDF flights: the columns of
    read_vars present in the file (ALONG/VLONG renamed ALON/VLON), one row per second, or
    per period_s seconds. Each frame's attrs hold 'start' and 'end' (datetimes, or None).

    :return: Returns {label: DataFrame}; the label is the file name, with '#2', '#3', ...
             added for later flights in a multi-flight file.
    """
    fmt = genpro_format(path)
    if fmt is None:
        raise ValueError(f"{path} is not a GENPRO file")
    segments = (read_genpro1 if fmt == 'genpro1' else read_genpro2)(path, read_vars)
    stride = max(1, round(period_s)) if period_s > 0 else 1   # one row per second
    flights = {}
    for i, seg in enumerate(segments):
        df = pd.DataFrame(seg['data']).iloc[::stride].reset_index(drop=True)
        df.attrs['start'], df.attrs['end'] = time_span(seg['date'], seg['time'])
        df.attrs['title'] = seg['title']
        label = os.path.basename(path) + (f"#{i + 1}" if i else '')
        flights[label] = df
    return flights
