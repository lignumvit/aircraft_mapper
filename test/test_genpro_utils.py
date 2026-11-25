import sys
sys.path.append('..')
import numpy as np
import pytest
from datetime import datetime
import util.genpro_utils as gp


def cos_file(path, records):
    """Write records (bytes) as a single-block Cray COS-blocked file."""
    words = []
    padded = [r + b'\0' * (-len(r) % 8) for r in records]
    nwords = [len(r) // 8 for r in padded]
    # block control word, then each record followed by its record control word; each
    # control word's low 9 bits count the data words up to the next control word
    words.append((0 << 60) | nwords[0])
    for i, r in enumerate(padded):
        words.extend(int.from_bytes(r[j:j + 8], 'big') for j in range(0, len(r), 8))
        ubc = 8 * (len(r) - len(records[i]))
        nxt = nwords[i + 1] if i + 1 < len(records) else 0
        words.append((0x8 << 60) | (ubc << 54) | nxt)
    words.append(0xE << 60)   # end of file
    words.append(0xF << 60)   # end of data
    assert len(words) < 512
    with open(path, 'wb') as f:
        f.write(b''.join(w.to_bytes(8, 'big') for w in words))


def pack_bits(values, nbit):
    bits = np.array([[(v >> (nbit - 1 - k)) & 1 for k in range(nbit)] for v in values], np.uint8)
    return np.packbits(bits.ravel()).tobytes()


def test_cos_records(tmp_path):
    path = tmp_path / 'cos'
    cos_file(path, [b'abcdefgh' * 3, b'xyz'])
    assert gp.cos_records(str(path)) == [b'abcdefgh' * 3, b'xyz', None]


def test_parse_date():
    assert gp._parse_date('NCAR ELECTRA STRATEX-1 05JUN76') == datetime(1976, 6, 5)
    assert gp._parse_date('584 POWER PLANT PLUME 16JULY76') == datetime(1976, 7, 16)
    assert gp._parse_date('674-01 OPUS-II (LRT) 13 APR 83') == datetime(1983, 4, 13)
    assert gp._parse_date('no date here') is None


def test_time_span_ignores_garbage():
    day = datetime(1983, 4, 13)
    t = np.array([0, 36000, 36001, 36002, 900000, 36004, 36005, 0])
    start, end = gp.time_span(day, t)
    assert start == datetime(1983, 4, 13, 10, 0, 0)
    assert end == datetime(1983, 4, 13, 10, 0, 5)


def test_genpro2(tmp_path):
    # header: 10 lines of 80 characters per 800-byte record
    lines = [' BEGINHD', '/PROJECT= "674-01 OPUS-II (LRT) 13 APR 83"',
             '/BEGSNP = (  9.0000 , 40.000 , 35.000 )',
             ' LOGBIT = 96', ' DATLOG = 4',
             ' ORDVAR = UNITS, SAMPLE, RATE, BITS, FSTBIT, SKIP']
    for i, name in enumerate(['PTIME', 'ALAT', 'ALON']):
        lines.append(f' LETVAR = " DEG", 1, 1, 32, {1 + 32 * i}, 0, %FOR, {name}')
    lines.append(' ORDVAR = CONKEY, SCLKEY, TERM, FACTOR')
    for name, term in [('PTIME', 0.0), ('ALAT', 100.0), ('ALON', 200.0)]:
        lines.append(f' LETVAR = 1, 2, {term}, 1000.0, %FOR, {name}')
    lines.append(' ENDHD')
    text = ''.join(l.ljust(80) for l in lines)
    text += ' ' * (-len(text) % 800)
    header = [text[i:i + 800].encode('ascii') for i in range(0, len(text), 800)]
    # 8 seconds of data, 4 per physical record
    t = 34835 + np.arange(8)
    lat, lon = 34.9 + 0.001 * np.arange(8), -120.4 - 0.001 * np.arange(8)
    raw = np.c_[t * 1000, (lat + 100) * 1000, (lon + 200) * 1000].round().astype(int)
    data = [pack_bits(raw[i:i + 4].ravel(), 32) for i in (0, 4)]
    path = tmp_path / 'G51665'
    cos_file(path, header + data)

    assert gp.genpro_format(str(path)) == 'genpro2'
    flights = gp.read_genpro_flights(str(path), ['ALAT', 'ALON'])
    df = flights['G51665']
    np.testing.assert_allclose(df['ALAT'], lat, atol=1e-9)
    np.testing.assert_allclose(df['ALON'], lon, atol=1e-9)
    assert df.attrs['start'] == datetime(1983, 4, 13, 9, 40, 35)
    assert df.attrs['end'] == datetime(1983, 4, 13, 9, 40, 42)


def display_code(text):
    codes = [gp.DISPLAY_CODE.index(c) for c in text]
    codes += [gp.DISPLAY_CODE.index(' ')] * (-len(codes) % 10)   # whole 60-bit words
    return pack_bits(codes, 6)


@pytest.mark.parametrize('ebcdic', [False, True])
def test_genpro1(tmp_path, ebcdic):
    text = ('NCAR ELECTRA STRATEX-1 05JUN76     FIRST TIME ON THIS FILE 18 01  0  '
            'DESCRIPTION OF RECORD --   3 PARAMETERS WERE SAVED AT THEIR RESPECTIVE RATES. '
            'THE  2 CYCLES OF   3 SAMP/CYC =    6 WERE THEN SCALED INTO 20 BIT INTEGERS AND '
            'PACKED 3 SAMPLES/WORD INTO   2 60 BIT PACKED WORDS  '
            ' 1)   1     PROCESSOR TIME (SECONDS AFTER MDNT)     TIME     SECONDS = (N/    1.0) -    0.0 '
            ' 2)   1     RAW INS LATITUDE (DEG)                  ALAT     (DEG)   = (N/ 1000.0) -  200.0 '
            ' 3)   1     RAW INS LONGITUDE (DEG)                 ALONG    (DEG)   = (N/ 1000.0) -  200.0 ')
    t = 64860 + np.arange(4)
    lat, lon = 37.432 + 0.001 * np.arange(4), -122.061 - 0.001 * np.arange(4)
    raw = np.c_[t, (lat + 200) * 1000, (lon + 200) * 1000].round().astype(int)
    if ebcdic:
        # a 6-bit character translated to an EBCDIC byte, 10 per 60-bit word
        def record(values):
            codes = np.unpackbits(np.frombuffer(pack_bits(values, 20), np.uint8))
            codes = codes.reshape(-1, 6).dot(1 << np.arange(5, -1, -1))
            return ''.join(gp.DISPLAY_CODE[c] for c in codes).encode('cp500')
        header = text.encode('cp500')
    else:
        record = lambda values: pack_bits(values, 20)
        header = display_code(text)
    path = tmp_path / 'G00070C'
    cos_file(path, [header, record(raw[:2].ravel()), record(raw[2:].ravel())])

    assert gp.genpro_format(str(path)) == 'genpro1'
    df = gp.read_genpro_flights(str(path), ['ALAT', 'ALON'])['G00070C']
    np.testing.assert_allclose(df['ALAT'], lat, atol=1e-9)
    np.testing.assert_allclose(df['ALON'], lon, atol=1e-9)   # ALONG is read as ALON
    assert df.attrs['start'] == datetime(1976, 6, 5, 18, 1, 0)
    assert df.attrs['end'] == datetime(1976, 6, 5, 18, 1, 3)


def test_not_genpro(tmp_path):
    path = tmp_path / 'md5sum.txt'
    path.write_text('d41d8cd98f00b204e9800998ecf8427e  G00070C\n')
    assert gp.genpro_format(str(path)) is None
