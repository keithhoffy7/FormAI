import dash
from dash.dependencies import Output, Input, State
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request
import os
import math
import pandas as pd
import numpy as np
import joblib
import csv
import glob

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

# Recording state for classification
is_recording = False
recording_data = {
	"phone_accel": {"time": [], "x": [], "y": [], "z": []},
	"phone_gyro": {"time": [], "x": [], "y": [], "z": []},
	"watch_accel": {"time": [], "x": [], "y": [], "z": []},
	"watch_gyro": {"time": [], "x": [], "y": [], "z": []}
}

# Model loading
MODEL_DIR = "models"
model = None
scaler = None
model_loaded = False


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


def _to_seconds(value):
	"""Convert datetime or timestamp to seconds since epoch."""
	if value is None:
		return None
	if isinstance(value, datetime):
		return value.timestamp()
	try:
		v = float(value)
		if v > 1e17:  # nanoseconds
			return v / 1e9
		elif v > 1e14:  # microseconds
			return v / 1e6
		elif v > 1e11:  # milliseconds
			return v / 1e3
		else:  # seconds
			return v
	except Exception:
		return None


def load_latest_model():
	"""Load the most recent trained model and scaler."""
	global model, scaler, model_loaded
	
	if not os.path.exists(MODEL_DIR):
		return False
	
	# Find latest model file
	model_files = glob.glob(os.path.join(MODEL_DIR, "form_classifier_*.pkl"))
	if not model_files:
		return False
	
	latest_model = max(model_files, key=os.path.getctime)
	timestamp = os.path.basename(latest_model).replace("form_classifier_", "").replace(".pkl", "")
	scaler_file = os.path.join(MODEL_DIR, f"scaler_{timestamp}.pkl")
	feature_names_file = os.path.join(MODEL_DIR, f"feature_names_{timestamp}.pkl")
	
	try:
		model = joblib.load(latest_model)
		if os.path.exists(scaler_file):
			scaler = joblib.load(scaler_file)
		else:
			scaler = None
		
		# Store feature names if available
		if os.path.exists(feature_names_file):
			global expected_feature_names
			expected_feature_names = joblib.load(feature_names_file)
		else:
			expected_feature_names = None
		
		model_loaded = True
		return True
	except Exception as e:
		print(f"Error loading model: {e}")
		return False

# Global variable for expected feature names
expected_feature_names = None


def extract_features_from_df(df):
	"""Extract features from a DataFrame (same as train_model.py)."""
	features = {}
	
	sensor_cols = {
		'phone_accel': ['phone_accel_x', 'phone_accel_y', 'phone_accel_z'],
		'phone_gyro': ['phone_gyro_x', 'phone_gyro_y', 'phone_gyro_z'],
		'watch_accel': ['watch_accel_x', 'watch_accel_y', 'watch_accel_z'],
		'watch_gyro': ['watch_gyro_x', 'watch_gyro_y', 'watch_gyro_z']
	}
	
	for sensor_name, cols in sensor_cols.items():
		available_cols = [c for c in cols if c in df.columns]
		
		if not available_cols:
			for stat in ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75']:
				features[f'{sensor_name}_{stat}'] = 0.0
			continue
		
		sensor_data = df[available_cols].values.flatten()
		sensor_data = sensor_data[~np.isnan(sensor_data)]
		
		if len(sensor_data) == 0:
			for stat in ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75']:
				features[f'{sensor_name}_{stat}'] = 0.0
			continue
		
		features[f'{sensor_name}_mean'] = np.mean(sensor_data)
		features[f'{sensor_name}_std'] = np.std(sensor_data)
		features[f'{sensor_name}_min'] = np.min(sensor_data)
		features[f'{sensor_name}_max'] = np.max(sensor_data)
		features[f'{sensor_name}_range'] = np.max(sensor_data) - np.min(sensor_data)
		features[f'{sensor_name}_median'] = np.median(sensor_data)
		features[f'{sensor_name}_q25'] = np.percentile(sensor_data, 25)
		features[f'{sensor_name}_q75'] = np.percentile(sensor_data, 75)
		
		for col in available_cols:
			col_data = df[col].dropna().values
			if len(col_data) > 0:
				features[f'{col}_mean'] = np.mean(col_data)
				features[f'{col}_std'] = np.std(col_data)
				features[f'{col}_max_abs'] = np.max(np.abs(col_data))
	
	if 'timestamp' in df.columns and len(df) > 1:
		features['duration'] = df['timestamp'].max() - df['timestamp'].min()
	else:
		features['duration'] = 0.0
	
	features['num_points'] = len(df)
	
	if 'phone_accel_x' in df.columns and 'watch_accel_x' in df.columns:
		phone_accel = df[['phone_accel_x', 'phone_accel_y', 'phone_accel_z']].values.flatten()
		watch_accel = df[['watch_accel_x', 'watch_accel_y', 'watch_accel_z']].values.flatten()
		phone_accel = phone_accel[~np.isnan(phone_accel)]
		watch_accel = watch_accel[~np.isnan(watch_accel)]
		
		if len(phone_accel) > 0 and len(watch_accel) > 0:
			min_len = min(len(phone_accel), len(watch_accel))
			if min_len > 1:
				corr = np.corrcoef(phone_accel[:min_len], watch_accel[:min_len])[0, 1]
				features['phone_watch_accel_corr'] = corr if not np.isnan(corr) else 0.0
			else:
				features['phone_watch_accel_corr'] = 0.0
		else:
			features['phone_watch_accel_corr'] = 0.0
	else:
		features['phone_watch_accel_corr'] = 0.0
	
	return features


def save_recorded_data_to_csv():
	"""Save recorded data to a temporary CSV file."""
	global recording_data
	
	# Find overlapping time period
	phone_times = set()
	watch_times = set()
	
	for sensor_type in ["phone_accel", "phone_gyro"]:
		phone_times.update(recording_data[sensor_type]["time"])
	
	for sensor_type in ["watch_accel", "watch_gyro"]:
		watch_times.update(recording_data[sensor_type]["time"])
	
	if not phone_times and not watch_times:
		return None
	
	all_times = phone_times.union(watch_times) if phone_times and watch_times else (phone_times or watch_times)
	
	time_to_seconds = {}
	for ts in all_times:
		time_to_seconds[ts] = _to_seconds(ts)
	
	sorted_times = sorted(all_times, key=lambda t: time_to_seconds[t])
	
	# Create lookup dictionaries
	phone_accel_lookup = {ts: (x, y, z) for ts, x, y, z in 
	                     zip(recording_data["phone_accel"]["time"],
	                         recording_data["phone_accel"]["x"],
	                         recording_data["phone_accel"]["y"],
	                         recording_data["phone_accel"]["z"])}
	phone_gyro_lookup = {ts: (x, y, z) for ts, x, y, z in 
	                    zip(recording_data["phone_gyro"]["time"],
	                        recording_data["phone_gyro"]["x"],
	                        recording_data["phone_gyro"]["y"],
	                        recording_data["phone_gyro"]["z"])}
	watch_accel_lookup = {ts: (x, y, z) for ts, x, y, z in 
	                     zip(recording_data["watch_accel"]["time"],
	                         recording_data["watch_accel"]["x"],
	                         recording_data["watch_accel"]["y"],
	                         recording_data["watch_accel"]["z"])}
	watch_gyro_lookup = {ts: (x, y, z) for ts, x, y, z in 
	                    zip(recording_data["watch_gyro"]["time"],
	                        recording_data["watch_gyro"]["x"],
	                        recording_data["watch_gyro"]["y"],
	                        recording_data["watch_gyro"]["z"])}
	
	# Create temporary CSV
	temp_file = f"temp_classification_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
	
	with open(temp_file, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([
			"timestamp",
			"phone_accel_x", "phone_accel_y", "phone_accel_z",
			"phone_gyro_x", "phone_gyro_y", "phone_gyro_z",
			"watch_accel_x", "watch_accel_y", "watch_accel_z",
			"watch_gyro_x", "watch_gyro_y", "watch_gyro_z"
		])
		
		for ts in sorted_times:
			ts_seconds = time_to_seconds[ts]
			phone_accel = phone_accel_lookup.get(ts, (None, None, None))
			phone_gyro = phone_gyro_lookup.get(ts, (None, None, None))
			watch_accel = watch_accel_lookup.get(ts, (None, None, None))
			watch_gyro = watch_gyro_lookup.get(ts, (None, None, None))
			
			writer.writerow([
				ts_seconds,
				phone_accel[0], phone_accel[1], phone_accel[2],
				phone_gyro[0], phone_gyro[1], phone_gyro[2],
				watch_accel[0], watch_accel[1], watch_accel[2],
				watch_gyro[0], watch_gyro[1], watch_gyro[2]
			])
	
	return temp_file


def classify_lift():
	"""Classify the recorded lift using the trained model."""
	global model, scaler, model_loaded
	
	if not model_loaded:
		if not load_latest_model():
			return None, "Model not found. Please train a model first."
	
	# Save recorded data to CSV
	csv_file = save_recorded_data_to_csv()
	if csv_file is None:
		return None, "No data recorded."
	
	try:
		# Load CSV and extract features
		df = pd.read_csv(csv_file)
		features = extract_features_from_df(df)
		feature_df = pd.DataFrame([features])
		
		# Ensure feature order matches what the model expects
		global expected_feature_names
		
		# Use saved feature names if available
		if expected_feature_names:
			expected_features = expected_feature_names
		elif scaler and hasattr(scaler, 'feature_names_in_'):
			expected_features = list(scaler.feature_names_in_)
		else:
			expected_features = None
		
		if expected_features:
			# Reorder and fill missing features
			for feat in expected_features:
				if feat not in feature_df.columns:
					feature_df[feat] = 0.0
			feature_df = feature_df[expected_features]
		
		if scaler:
			feature_df_scaled = scaler.transform(feature_df)
		else:
			feature_df_scaled = feature_df.values
		
		# Predict
		prediction = model.predict(feature_df_scaled)[0]
		probabilities = model.predict_proba(feature_df_scaled)[0]
		
		# Clean up temp file
		if os.path.exists(csv_file):
			os.remove(csv_file)
		
		confidence = max(probabilities) * 100
		return prediction, f"Confidence: {confidence:.1f}%"
		
	except Exception as e:
		# Clean up temp file
		if os.path.exists(csv_file):
			os.remove(csv_file)
		return None, f"Classification error: {str(e)}"

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
		
		# Classification Control Panel
		dbc.Row(
			[
				dbc.Col(
					[
						dbc.Card(
							[
								dbc.CardHeader(
									"Form Classification",
									className="bg-dark text-white fw-bold"
								),
								dbc.CardBody(
									[
										dbc.Row(
											[
												dbc.Col(
													[
														dbc.Button(
															"🔴 Start Recording",
															id="start-classify-btn",
															color="success",
															className="me-2 mb-2",
															size="lg",
															disabled=False
														),
														dbc.Button(
															"⏹️ Stop & Classify",
															id="stop-classify-btn",
															color="danger",
															className="mb-2",
															size="lg",
															disabled=True
														)
													],
													md=6
												),
												dbc.Col(
													[
														html.Div(id="classification-status", className="mb-2"),
														html.Div(id="classification-result")
													],
													md=6
												)
											]
										)
									]
								)
							],
							className="mb-4 shadow-sm"
						)
					],
					width=12
				)
			]
		),
		
		# Hidden interval component for auto-updates
		dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
		# Hidden store for recording state
		dcc.Store(id="recording-state-store", data={"is_recording": False}),
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


@app.callback(
	[
		Output("start-classify-btn", "disabled"),
		Output("stop-classify-btn", "disabled"),
		Output("classification-status", "children"),
		Output("recording-state-store", "data")
	],
	[
		Input("start-classify-btn", "n_clicks"),
		Input("stop-classify-btn", "n_clicks")
	],
	[State("recording-state-store", "data")],
	prevent_initial_call=True
)
def handle_recording_controls(start_clicks, stop_clicks, recording_state):
	global is_recording, recording_data
	
	ctx = dash.callback_context
	if not ctx.triggered:
		return False, True, html.Div("Ready to record", className="text-muted"), {"is_recording": False}
	
	trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
	current_state = recording_state.get("is_recording", False) if recording_state else False
	
	if trigger_id == "start-classify-btn":
		# Start recording
		is_recording = True
		recording_data = {
			"phone_accel": {"time": [], "x": [], "y": [], "z": []},
			"phone_gyro": {"time": [], "x": [], "y": [], "z": []},
			"watch_accel": {"time": [], "x": [], "y": [], "z": []},
			"watch_gyro": {"time": [], "x": [], "y": [], "z": []}
		}
		status = dbc.Alert("🔴 Recording... Do your lift now!", color="danger", className="mb-0")
		return True, False, status, {"is_recording": True}
	
	elif trigger_id == "stop-classify-btn":
		# Stop recording and classify
		is_recording = False
		status = dbc.Alert("⏳ Classifying...", color="info", className="mb-0")
		return False, True, status, {"is_recording": False}
	
	return not current_state, current_state, html.Div("Ready", className="text-muted"), {"is_recording": current_state}


@app.callback(
	Output("classification-result", "children"),
	Input("stop-classify-btn", "n_clicks"),
	prevent_initial_call=True
)
def classify_on_stop(stop_clicks):
	"""Classify the lift when stop button is clicked."""
	if stop_clicks is None or stop_clicks == 0:
		return ""
	
	# Small delay to ensure recording has stopped
	import time
	time.sleep(0.1)
	
	prediction, message = classify_lift()
	
	if prediction is None:
		return dbc.Alert(f"❌ {message}", color="danger", className="mb-0")
	
	# Format result
	if prediction == "good":
		color = "success"
		icon = "✅"
		title = "Good Form!"
	else:
		color = "warning"
		icon = "⚠️"
		title = "Needs Improvement"
	
	return dbc.Alert(
		[
			html.H5(f"{icon} {title}", className="mb-2"),
			html.P(f"Prediction: {prediction.upper()}", className="mb-1"),
			html.P(message, className="mb-0 small")
		],
		color=color,
		className="mb-0"
	)


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
					
					# Record data if recording for classification
					if is_recording:
						recording_data["watch_accel"]["time"].append(ts)
						recording_data["watch_accel"]["x"].append(_convert_accel(ax))
						recording_data["watch_accel"]["y"].append(_convert_accel(ay))
						recording_data["watch_accel"]["z"].append(_convert_accel(az))
			
			if gyro_xyz is not None:
				gx, gy, gz = gyro_xyz
				if gx is not None and gy is not None and gz is not None:
					if len(watch_time_gyro) == 0 or ts > watch_time_gyro[-1]:
						watch_time_gyro.append(ts)
						watch_gyro_x.append(_convert_gyro(gx))
						watch_gyro_y.append(_convert_gyro(gy))
						watch_gyro_z.append(_convert_gyro(gz))
					
					# Record data if recording for classification
					if is_recording:
						recording_data["watch_gyro"]["time"].append(ts)
						recording_data["watch_gyro"]["x"].append(_convert_gyro(gx))
						recording_data["watch_gyro"]["y"].append(_convert_gyro(gy))
						recording_data["watch_gyro"]["z"].append(_convert_gyro(gz))
		
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
				
				# Record data if recording for classification
				if is_recording:
					recording_data["phone_accel"]["time"].append(ts)
					recording_data["phone_accel"]["x"].append(_convert_accel(x))
					recording_data["phone_accel"]["y"].append(_convert_accel(y))
					recording_data["phone_accel"]["z"].append(_convert_accel(z))
			elif name in ("gyroscope", "gyro"):
				if len(iphone_time_gyro) == 0 or ts > iphone_time_gyro[-1]:
					iphone_time_gyro.append(ts)
					iphone_gyro_x.append(_convert_gyro(x))
					iphone_gyro_y.append(_convert_gyro(y))
					iphone_gyro_z.append(_convert_gyro(z))
				
				# Record data if recording for classification
				if is_recording:
					recording_data["phone_gyro"]["time"].append(ts)
					recording_data["phone_gyro"]["x"].append(_convert_gyro(x))
					recording_data["phone_gyro"]["y"].append(_convert_gyro(y))
					recording_data["phone_gyro"]["z"].append(_convert_gyro(z))

	return "success"


# Try to load model on startup
if __name__ == "__main__":
	# Try to load the latest model
	if load_latest_model():
		print("✅ Model loaded successfully")
	else:
		print("⚠️  No model found. Train a model first using train_model.py")
	
	app.run(port=8000, host="0.0.0.0")