import sys
sys.path.append('..')
import numpy as np
from datetime import datetime
import util.dap_utils as dap


def write_listing(path, header_date, column_names, rows):
    lines = ['', '', 'FLIGHT NUMBER 12   ', f'DATE:  {header_date}', '',
             'time          ' + ''.join(f'{c:<14}' for c in column_names),
             ' UCT', '------------  ------------  ------------']
    lines += [f'{t:>11}   {a:>11}   {b:>11}' for t, a, b in rows]
    path.write_text('\n'.join(lines) + '\n')


def test_read_listing_splits_flights_and_fixes_date(tmp_path):
    rows = [('7:45:54', '0.00000e+00', '0.00000e+00'),
            ('7:46:04', '2.62848e+01', '5.05206e+01'),
            ('7:46:14', '-NaN', 'NaN'),
            ('7:46:24', '2.62850e+01', '5.05113e+01'),
            ('9:00:04', '2.70000e+01', '5.00000e+01'),   # 74 min gap: a second flight
            ('9:00:14', '2.70010e+01', '5.00010e+01')]
    path = tmp_path / 'KFN02.txt'
    write_listing(path, '6/1/91', ['ILATC', 'ILONC'], rows)
    # the files_by_date copy says this listing is from 20 May, not the header's 1 June
    (tmp_path / 'files_by_date' / '2005').mkdir(parents=True)
    (tmp_path / 'files_by_date' / '2005' / 'KFN02.OUT').write_text(path.read_text())

    assert dap.is_dap(str(path))
    flights = dap.read_dap_flights(str(path), ['LAT', 'LON'])
    assert list(flights) == ['KFN02.txt', 'KFN02.txt#2']
    first, second = flights.values()
    np.testing.assert_allclose(first['LAT'], [0, 26.2848, np.nan, 26.285])
    np.testing.assert_allclose(second['LON'], [50.0, 50.001])
    assert first.attrs['start'] == datetime(1991, 5, 20, 7, 45, 54)
    assert first.attrs['end'] == datetime(1991, 5, 20, 7, 46, 24)
    assert second.attrs['start'] == datetime(1991, 5, 20, 9, 0, 4)


def test_not_dap(tmp_path):
    path = tmp_path / 'notes.txt'
    path.write_text('just some notes\n')
    assert not dap.is_dap(str(path))
