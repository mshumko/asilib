from datetime import datetime, timedelta

import numpy as np
import pytest

from asilib.mission.example_satellite import Example_Satellite


def test_example_satellite_default_ephemeris():
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = start + timedelta(minutes=10)

    sat = Example_Satellite(time_range=(start, end), cadence_s=30)
    times, lla = sat.ephemeris()

    assert lla.shape == (len(times), 3)
    assert times[0] == start
    assert np.all(np.isfinite(lla))
    assert 0 <= sat.raan_deg < 360

    assert np.all(np.abs(lla[:, 0]) <= 90)
    assert np.all((lla[:, 1] >= -180) & (lla[:, 1] <= 180))
    # Skyfield returns geodetic altitude, so a circular orbit in ECI does not
    # yield perfectly constant altitude against the WGS84 ellipsoid.
    assert np.nanmean(lla[:, 2]) == pytest.approx(500.0, abs=30.0)
    assert np.nanmin(lla[:, 2]) > 430.0
    assert np.nanmax(lla[:, 2]) < 570.0


def test_example_satellite_eccentric_orbit_altitude_varies():
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = start + timedelta(hours=1)

    sat = Example_Satellite(time_range=(start, end), cadence_s=60, eccentricity=0.01)
    _, lla = sat.ephemeris()

    assert np.nanmax(lla[:, 2]) > np.nanmin(lla[:, 2])


def test_example_satellite_ephemeris_df_columns():
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = start + timedelta(minutes=2)

    sat = Example_Satellite(time_range=(start, end), cadence_s=30)
    df = sat.ephemeris_df()

    assert list(df.columns) == ['lat', 'lon', 'alt']
    assert len(df) > 0


def test_example_satellite_invalid_ltan_raises():
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = start + timedelta(minutes=2)

    with pytest.raises(ValueError, match='ltan_hours'):
        Example_Satellite(time_range=(start, end), ltan_hours=24.0)
