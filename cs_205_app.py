import dash
from dash.dependencies import Output, Input
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request
import os
import math

server = Flask(__name__)
app = dash.Dash(
	__name__,
	server=server,
	external_stylesheets=[dbc.themes.BOOTSTRAP],
	meta_tags=[
		{"name": "viewport", "content": "width=device-width, initial-scale=1"}
	]
)

MAX_DATA_POINTS = 1000
UPDATE_FREQ_MS = 100

# iPhone data
iphone_time_accel = deque(maxlen=MAX_DATA_POINTS)
iphone_accel_x = deque(maxlen=MAX_DATA_POINTS)
iphone_accel_y = deque(maxlen=MAX_DATA_POINTS)
iphone_accel_z = deque(maxlen=MAX_DATA_POINTS)
iphone_time_gyro = deque(maxlen=MAX_DATA_POINTS)
iphone_gyro_x = deque(maxlen=MAX_DATA_POINTS)
iphone_gyro_y = deque(maxlen=MAX_DATA_POINTS)
iphone_gyro_z = deque(maxlen=MAX_DATA_POINTS)

# Apple Watch data
watch_time_accel = deque(maxlen=MAX_DATA_POINTS)
watch_accel_x = deque(maxlen=MAX_DATA_POINTS)
watch_accel_y = deque(maxlen=MAX_DATA_POINTS)
watch_accel_z = deque(maxlen=MAX_DATA_POINTS)
watch_time_gyro = deque(maxlen=MAX_DATA_POINTS)
watch_gyro_x = deque(maxlen=MAX_DATA_POINTS)
watch_gyro_y = deque(maxlen=MAX_DATA_POINTS)
watch_gyro_z = deque(maxlen=MAX_DATA_POINTS)

# Optional env configuration for device filtering and units
WATCH_DEVICE_ID = os.environ.get("WATCH_DEVICE_ID")
ACCEL_UNITS = (os.environ.get("ACCEL_UNITS") or "m_s2").lower()  # accepted: m_s2, g
GYRO_UNITS = (os.environ.get("GYRO_UNITS") or "rad_s").lower()   # accepted: rad_s, deg_s


def _to_datetime(value):
	"""Convert various timestamp magnitudes (s, ms, us, ns) to datetime."""
	if value is None:
		return None
	try:
		v = float(value)
		# Heuristic based on magnitude
		if v > 1e17:  # nanoseconds
			seconds = v / 1e9
		elif v > 1e14:  # microseconds
			seconds = v / 1e6
		elif v > 1e11:  # milliseconds
			seconds = v / 1e3
		else:  # seconds
			seconds = v
		return datetime.fromtimestamp(seconds)
	except Exception:
		return None


def _extract_xyz(values):
	"""Return (x, y, z) from dict with keys or from list/tuple."""
	if values is None:
		return None
	if isinstance(values, dict):
		return values.get("x"), values.get("y"), values.get("z")
	if isinstance(values, (list, tuple)) and len(values) >= 3:
		return values[0], values[1], values[2]
	return None


def _extract_wrist_motion_accel(values):
	"""Extract accelerometer data from Apple Watch wrist motion values."""
	if values is None or not isinstance(values, dict):
		return None
	return (
		values.get("accelerationX"),
		values.get("accelerationY"),
		values.get("accelerationZ")
	)


def _extract_wrist_motion_gyro(values):
	"""Extract gyroscope data from Apple Watch wrist motion values."""
	if values is None or not isinstance(values, dict):
		return None
	return (
		values.get("rotationRateX"),
		values.get("rotationRateY"),
		values.get("rotationRateZ")
	)


def _convert_accel(v):
	if v is None:
		return None
	if ACCEL_UNITS in ("g", "gee", "grav"):
		return float(v) * 9.80665
	return float(v)


def _convert_gyro(v):
	if v is None:
		return None
	if GYRO_UNITS in ("deg_s", "deg/s", "degrees_per_second", "degrees_s"):
		return float(v) * (math.pi / 180.0)
	return float(v)

app.layout = dbc.Container(
	[
		# Header/Navbar
		dbc.Navbar(
			dbc.Container(
				[
				dbc.NavbarBrand(
					"FormAI - Exercise Form Analysis",
					className="fs-4 fw-bold"
				),
					dbc.NavbarToggler(id="navbar-toggler"),
				],
				fluid=True
			),
			color="primary",
			dark=True,
			className="mb-4 shadow"
		),
		
		# Main content
		dbc.Row(
			[
				dbc.Col(
					[
						dbc.Card(
							[
							dbc.CardHeader(
								"iPhone Accelerometer",
								className="bg-primary text-white fw-bold"
							),
								dbc.CardBody(
									[
										dcc.Graph(
											id="iphone_accel_graph",
											config={"displayModeBar": False},
											style={"height": "400px"}
										)
									]
								)
							],
							className="mb-4 shadow-sm"
						)
					],
					md=6,
					className="mb-3"
				),
				dbc.Col(
					[
						dbc.Card(
							[
							dbc.CardHeader(
								"iPhone Gyroscope",
								className="bg-info text-white fw-bold"
							),
								dbc.CardBody(
									[
										dcc.Graph(
											id="iphone_gyro_graph",
											config={"displayModeBar": False},
											style={"height": "400px"}
										)
									]
								)
							],
							className="mb-4 shadow-sm"
						)
					],
					md=6,
					className="mb-3"
				),
				dbc.Col(
					[
						dbc.Card(
							[
							dbc.CardHeader(
								"Apple Watch Accelerometer",
								className="bg-success text-white fw-bold"
							),
								dbc.CardBody(
									[
										dcc.Graph(
											id="watch_accel_graph",
											config={"displayModeBar": False},
											style={"height": "400px"}
										)
									]
								)
							],
							className="mb-4 shadow-sm"
						)
					],
					md=6,
					className="mb-3"
				),
				dbc.Col(
					[
						dbc.Card(
							[
							dbc.CardHeader(
								"Apple Watch Gyroscope",
								className="bg-warning text-dark fw-bold"
							),
								dbc.CardBody(
									[
										dcc.Graph(
											id="watch_gyro_graph",
											config={"displayModeBar": False},
											style={"height": "400px"}
										)
									]
								)
							],
							className="mb-4 shadow-sm"
						)
					],
					md=6,
					className="mb-3"
				),
			],
			className="g-3"
		),
		
		# Hidden interval component for auto-updates
		dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
	],
	fluid=True,
	className="py-4"
)


@app.callback(
	[
		Output("iphone_accel_graph", "figure"),
		Output("iphone_gyro_graph", "figure"),
		Output("watch_accel_graph", "figure"),
		Output("watch_gyro_graph", "figure")
	],
	Input("counter", "n_intervals"),
)
def update_graph(_counter):
	# iPhone accelerometer graph
	iphone_accel_data = [
		go.Scatter(
			x=list(iphone_time_accel),
			y=list(d),
			name=name,
			line=dict(width=2, color=color),
			mode='lines'
		)
		for d, name, color in zip(
			[iphone_accel_x, iphone_accel_y, iphone_accel_z],
			["Ax", "Ay", "Az"],
			["#FF6B6B", "#4ECDC4", "#45B7D1"]
		)
	]
	iphone_accel_graph = {
		"data": iphone_accel_data,
		"layout": go.Layout(
			{
				"title": {"text": "iPhone Accelerometer", "font": {"size": 18}},
				"xaxis": {"type": "date", "gridcolor": "#e0e0e0"},
				"yaxis": {"title": "Acceleration (m/s²)", "gridcolor": "#e0e0e0"},
				"plot_bgcolor": "white",
				"paper_bgcolor": "white",
				"font": {"family": "Arial, sans-serif"},
				"hovermode": "x unified",
				"legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
			}
		),
	}
	if len(iphone_time_accel) > 0:
		iphone_accel_graph["layout"]["xaxis"]["range"] = [min(iphone_time_accel), max(iphone_time_accel)]
		_all_accel = list(iphone_accel_x) + list(iphone_accel_y) + list(iphone_accel_z)
		_ymin = min(_all_accel)
		_ymax = max(_all_accel)
		_pad = (_ymax - _ymin) * 0.1 if _ymax != _ymin else 0.5
		iphone_accel_graph["layout"]["yaxis"]["range"] = [_ymin - _pad, _ymax + _pad]

	# iPhone gyroscope graph
	iphone_gyro_data = [
		go.Scatter(
			x=list(iphone_time_gyro),
			y=list(d),
			name=name,
			line=dict(width=2, color=color),
			mode='lines'
		)
		for d, name, color in zip(
			[iphone_gyro_x, iphone_gyro_y, iphone_gyro_z],
			["Gx", "Gy", "Gz"],
			["#FF6B6B", "#4ECDC4", "#45B7D1"]
		)
	]
	iphone_gyro_graph = {
		"data": iphone_gyro_data,
		"layout": go.Layout(
			{
				"title": {"text": "iPhone Gyroscope", "font": {"size": 18}},
				"xaxis": {"type": "date", "gridcolor": "#e0e0e0"},
				"yaxis": {"title": "Angular Velocity (rad/s)", "gridcolor": "#e0e0e0"},
				"plot_bgcolor": "white",
				"paper_bgcolor": "white",
				"font": {"family": "Arial, sans-serif"},
				"hovermode": "x unified",
				"legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
			}
		),
	}
	if len(iphone_time_gyro) > 0:
		iphone_gyro_graph["layout"]["xaxis"]["range"] = [min(iphone_time_gyro), max(iphone_time_gyro)]
		_all_gyro = list(iphone_gyro_x) + list(iphone_gyro_y) + list(iphone_gyro_z)
		_gmin = min(_all_gyro)
		_gmax = max(_all_gyro)
		_gpad = (_gmax - _gmin) * 0.1 if _gmax != _gmin else 0.5
		iphone_gyro_graph["layout"]["yaxis"]["range"] = [_gmin - _gpad, _gmax + _gpad]

	# Apple Watch accelerometer graph
	watch_accel_data = [
		go.Scatter(
			x=list(watch_time_accel),
			y=list(d),
			name=name,
			line=dict(width=2, color=color),
			mode='lines'
		)
		for d, name, color in zip(
			[watch_accel_x, watch_accel_y, watch_accel_z],
			["Ax", "Ay", "Az"],
			["#28A745", "#20C997", "#17A2B8"]
		)
	]
	watch_accel_graph = {
		"data": watch_accel_data,
		"layout": go.Layout(
			{
				"title": {"text": "Apple Watch Accelerometer", "font": {"size": 18}},
				"xaxis": {"type": "date", "gridcolor": "#e0e0e0"},
				"yaxis": {"title": "Acceleration (m/s²)", "gridcolor": "#e0e0e0"},
				"plot_bgcolor": "white",
				"paper_bgcolor": "white",
				"font": {"family": "Arial, sans-serif"},
				"hovermode": "x unified",
				"legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
			}
		),
	}
	if len(watch_time_accel) > 0:
		watch_accel_graph["layout"]["xaxis"]["range"] = [min(watch_time_accel), max(watch_time_accel)]
		_all_accel = list(watch_accel_x) + list(watch_accel_y) + list(watch_accel_z)
		_ymin = min(_all_accel)
		_ymax = max(_all_accel)
		_pad = (_ymax - _ymin) * 0.1 if _ymax != _ymin else 0.5
		watch_accel_graph["layout"]["yaxis"]["range"] = [_ymin - _pad, _ymax + _pad]

	# Apple Watch gyroscope graph
	watch_gyro_data = [
		go.Scatter(
			x=list(watch_time_gyro),
			y=list(d),
			name=name,
			line=dict(width=2, color=color),
			mode='lines'
		)
		for d, name, color in zip(
			[watch_gyro_x, watch_gyro_y, watch_gyro_z],
			["Gx", "Gy", "Gz"],
			["#FFC107", "#FD7E14", "#DC3545"]
		)
	]
	watch_gyro_graph = {
		"data": watch_gyro_data,
		"layout": go.Layout(
			{
				"title": {"text": "Apple Watch Gyroscope", "font": {"size": 18}},
				"xaxis": {"type": "date", "gridcolor": "#e0e0e0"},
				"yaxis": {"title": "Angular Velocity (rad/s)", "gridcolor": "#e0e0e0"},
				"plot_bgcolor": "white",
				"paper_bgcolor": "white",
				"font": {"family": "Arial, sans-serif"},
				"hovermode": "x unified",
				"legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
			}
		),
	}
	if len(watch_time_gyro) > 0:
		watch_gyro_graph["layout"]["xaxis"]["range"] = [min(watch_time_gyro), max(watch_time_gyro)]
		_all_gyro = list(watch_gyro_x) + list(watch_gyro_y) + list(watch_gyro_z)
		_gmin = min(_all_gyro)
		_gmax = max(_all_gyro)
		_gpad = (_gmax - _gmin) * 0.1 if _gmax != _gmin else 0.5
		watch_gyro_graph["layout"]["yaxis"]["range"] = [_gmin - _gpad, _gmax + _gpad]

	return iphone_accel_graph, iphone_gyro_graph, watch_accel_graph, watch_gyro_graph


@server.route("/data", methods=["POST"])
def data():  # listens to the data streamed from the sensor logger
	# Parse JSON safely
	payload_obj = request.get_json(silent=True)
	if payload_obj is None:
		try:
			payload_obj = json.loads(request.data)
		except Exception:
			return "bad request", 400

	# Optional device filter - check top-level deviceId
	if WATCH_DEVICE_ID:
		device_id = payload_obj.get("deviceId") or payload_obj.get("device_id")
		if device_id is not None and str(device_id) != str(WATCH_DEVICE_ID):
			return "success"  # Skip processing if device doesn't match

	for d in payload_obj.get('payload', []):
		name_raw = d.get("name") or d.get("sensor") or ""
		name = str(name_raw).lower()
		ts_raw = d.get("time") or d.get("timestamp") or d.get("ts")
		ts = _to_datetime(ts_raw)
		values_obj = d.get("values") or d.get("value") or d.get("data")
		
		if ts is None:
			continue

		# Handle Apple Watch wrist motion data
		if name in ("wrist motion", "wristmotion"):
			accel_xyz = _extract_wrist_motion_accel(values_obj)
			gyro_xyz = _extract_wrist_motion_gyro(values_obj)
			
			if accel_xyz is not None:
				ax, ay, az = accel_xyz
				if ax is not None and ay is not None and az is not None:
					if len(watch_time_accel) == 0 or ts > watch_time_accel[-1]:
						watch_time_accel.append(ts)
						watch_accel_x.append(_convert_accel(ax))
						watch_accel_y.append(_convert_accel(ay))
						watch_accel_z.append(_convert_accel(az))
			
			if gyro_xyz is not None:
				gx, gy, gz = gyro_xyz
				if gx is not None and gy is not None and gz is not None:
					if len(watch_time_gyro) == 0 or ts > watch_time_gyro[-1]:
						watch_time_gyro.append(ts)
						watch_gyro_x.append(_convert_gyro(gx))
						watch_gyro_y.append(_convert_gyro(gy))
						watch_gyro_z.append(_convert_gyro(gz))
		
		# Handle iPhone accelerometer and gyroscope data
		else:
			xyz = _extract_xyz(values_obj)
			if xyz is None:
				continue
			
			x, y, z = xyz

			if name in ("accelerometer", "accel"):
				if len(iphone_time_accel) == 0 or ts > iphone_time_accel[-1]:
					iphone_time_accel.append(ts)
					iphone_accel_x.append(_convert_accel(x))
					iphone_accel_y.append(_convert_accel(y))
					iphone_accel_z.append(_convert_accel(z))
			elif name in ("gyroscope", "gyro"):
				if len(iphone_time_gyro) == 0 or ts > iphone_time_gyro[-1]:
					iphone_time_gyro.append(ts)
					iphone_gyro_x.append(_convert_gyro(x))
					iphone_gyro_y.append(_convert_gyro(y))
					iphone_gyro_z.append(_convert_gyro(z))

	return "success"


if __name__ == "__main__":
	app.run(port=8000, host="0.0.0.0")