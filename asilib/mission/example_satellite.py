"""
Example satellite ephemeris model using Skyfield + SGP4 propagation.
"""


from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    import skyfield.api
    import sgp4.api
except ImportError:
    raise ImportError(
        "Skyfield and SGP4 are required for the Example_Satellite class. "
        "Please install them with `pip install skyfield sgp4` or "
		"`uv add skyfield sgp4`."
    )


EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 398600.4418
SGP4_EPOCH_OFFSET_DAYS = 2433281.5


@lru_cache(maxsize=1)
def _load_solar_ephemeris():
	planets = skyfield.api.load('de421.bsp')
	return planets['earth'], planets['sun']


@dataclass
class Example_Satellite:
	"""
	Generate a simple (lat, lon, alt) ephemeris from Keplerian elements.

	Notes
	-----
	The orbit is initialized from classical elements and propagated with SGP4.
	Skyfield is then used to convert propagated positions to geodetic
	latitude, longitude, and altitude.
	"""

	time_range: tuple[datetime | str | np.datetime64 | pd.Timestamp, datetime | str | np.datetime64 | pd.Timestamp]
	cadence_s: float = 5
	altitude_km: float = 500.0
	eccentricity: float = 0.0
	inclination_deg: float = 97.4
	ltan_hours: float = 22.5
	arg_perigee_deg: float = 0.0
	mean_anomaly_deg: float = 0.0
	semi_major_axis_km: float | None = None

	def __post_init__(self) -> None:
		start = pd.Timestamp(self.time_range[0])
		end = pd.Timestamp(self.time_range[1])
		if end <= start:
			raise ValueError('time_range must be increasing (end > start).')
		if self.cadence_s <= 0:
			raise ValueError('cadence_s must be > 0.')
		if not (0 <= self.eccentricity < 1):
			raise ValueError('eccentricity must satisfy 0 <= e < 1.')
		if not (0 <= self.ltan_hours < 24):
			raise ValueError('ltan_hours must satisfy 0 <= ltan_hours < 24.')

		self.start_time = start
		self.end_time = end

		if self.semi_major_axis_km is None:
			self.semi_major_axis_km = EARTH_RADIUS_KM + self.altitude_km
		elif self.semi_major_axis_km <= EARTH_RADIUS_KM:
			raise ValueError('semi_major_axis_km must exceed Earth radius.')

		self.ts = skyfield.api.load.timescale()
		self.raan_deg = self._resolve_raan_deg()
		self._satellite = self._build_satellite()

	def ephemeris(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		Return timestamps and satellite LLA coordinates.

		Returns
		-------
		times: np.ndarray
			Array of ``datetime.datetime`` timestamps.
		lla: np.ndarray
			Array of shape ``(n_times, 3)`` with columns
			``(latitude_deg, longitude_deg, altitude_km)``.
		"""
		times = self._make_time_array()
		times_index = pd.to_datetime(times)
		year = times_index.year.to_numpy()
		month = times_index.month.to_numpy()
		day = times_index.day.to_numpy()
		hour = times_index.hour.to_numpy()
		minute = times_index.minute.to_numpy()
		second = (
			times_index.second.to_numpy(dtype=float)
			+ times_index.microsecond.to_numpy(dtype=float) / 1e6
			+ times_index.nanosecond.to_numpy(dtype=float) / 1e9
		)
		sf_time = self.ts.utc(
			year,
			month,
			day,
			hour,
			minute,
			second,
		)

		geocentric = self._satellite.at(sf_time)
		lat_angle, lon_angle = skyfield.api.wgs84.latlon_of(geocentric)
		lat = lat_angle.degrees
		lon = ((lon_angle.degrees + 180) % 360) - 180
		alt = skyfield.api.wgs84.height_of(geocentric).km
		lla = np.column_stack((lat, lon, alt))
		return pd.to_datetime(times).to_pydatetime(), lla

	def ephemeris_df(self) -> pd.DataFrame:
		"""Return the ephemeris as a DataFrame indexed by UTC timestamps."""
		times, lla = self.ephemeris()
		return pd.DataFrame(index=pd.to_datetime(times), data={'lat': lla[:, 0], 'lon': lla[:, 1], 'alt': lla[:, 2]})

	def _make_time_array(self) -> np.ndarray:
		cadence_ns = int(np.round(self.cadence_s * 1e9))
		step = np.timedelta64(cadence_ns, 'ns')
		start = self.start_time.to_datetime64()
		end = self.end_time.to_datetime64()
		return np.arange(start, end + step, step)

	def _build_satellite(self):
		epoch_dt = self.start_time.to_pydatetime()
		jd, fr = sgp4.api.jday(
			epoch_dt.year,
			epoch_dt.month,
			epoch_dt.day,
			epoch_dt.hour,
			epoch_dt.minute,
			epoch_dt.second + epoch_dt.microsecond / 1e6,
		)
		epoch_days = jd + fr - SGP4_EPOCH_OFFSET_DAYS

		mean_motion_rad_s = np.sqrt(EARTH_MU_KM3_S2 / self.semi_major_axis_km**3)
		mean_motion_rad_min = mean_motion_rad_s * 60.0

		satrec = sgp4.api.Satrec()
		satrec.sgp4init(
			sgp4.api.WGS84,
			'i',
			99999,
			epoch_days,
			0.0,  # bstar
			0.0,  # ndot
			0.0,  # nddot
			self.eccentricity,
			np.deg2rad(self.arg_perigee_deg),
			np.deg2rad(self.inclination_deg),
			np.deg2rad(self.mean_anomaly_deg),
			mean_motion_rad_min,
			np.deg2rad(self.raan_deg),
		)
		return skyfield.api.EarthSatellite.from_satrec(satrec, self.ts)

	def _resolve_raan_deg(self) -> float:
		earth, sun = _load_solar_ephemeris()
		t0 = self.ts.utc(
			self.start_time.year,
			self.start_time.month,
			self.start_time.day,
			self.start_time.hour,
			self.start_time.minute,
			self.start_time.second + self.start_time.microsecond / 1e6,
		)
		sun_ra_rad = earth.at(t0).observe(sun).apparent().radec()[0].radians
		raan_rad = np.mod(sun_ra_rad + np.deg2rad((self.ltan_hours - 12.0) * 15.0), 2.0 * np.pi)
		return float(np.rad2deg(raan_rad))