#!/usr/bin/env python3
"""
Train a classifier to distinguish between good and bad form for gym lifts.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime


def load_all_samples(data_dir="data"):
    """Load all CSV samples and return as DataFrame with labels."""
    samples = []
    labels = []
    file_paths = []
    
    # Walk through data directory
    for root, dirs, files in os.walk(data_dir):
        # Extract lift_type and form_label from path
        path_parts = Path(root).parts
        if len(path_parts) >= 3 and path_parts[0] == data_dir:
            lift_type = path_parts[1]
            form_label = path_parts[2]
            
            # Load CSV files
            for file in files:
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    try:
                        df = pd.read_csv(filepath)
                        if len(df) > 0:  # Only add non-empty files
                            samples.append(df)
                            labels.append(form_label)
                            file_paths.append(filepath)
                    except Exception as e:
                        print(f"⚠️  Error loading {filepath}: {e}")
    
    return samples, labels, file_paths


def extract_features(df):
    """Extract features from a single sample DataFrame.
    
    Features include:
    - Statistical features (mean, std, min, max, etc.) for each sensor
    - Time-domain features (range, peak-to-peak, etc.)
    - Correlation between sensors
    """
    features = {}
    
    # Sensor columns
    sensor_cols = {
        'phone_accel': ['phone_accel_x', 'phone_accel_y', 'phone_accel_z'],
        'phone_gyro': ['phone_gyro_x', 'phone_gyro_y', 'phone_gyro_z'],
        'watch_accel': ['watch_accel_x', 'watch_accel_y', 'watch_accel_z'],
        'watch_gyro': ['watch_gyro_x', 'watch_gyro_y', 'watch_gyro_z']
    }
    
    # For each sensor type
    for sensor_name, cols in sensor_cols.items():
        # Check which columns exist in the dataframe
        available_cols = [c for c in cols if c in df.columns]
        
        if not available_cols:
            # If sensor not available, add zeros
            for stat in ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75']:
                features[f'{sensor_name}_{stat}'] = 0.0
            continue
        
        # Combine all axes for this sensor
        sensor_data = df[available_cols].values.flatten()
        sensor_data = sensor_data[~np.isnan(sensor_data)]  # Remove NaN
        
        if len(sensor_data) == 0:
            for stat in ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75']:
                features[f'{sensor_name}_{stat}'] = 0.0
            continue
        
        # Statistical features
        features[f'{sensor_name}_mean'] = np.mean(sensor_data)
        features[f'{sensor_name}_std'] = np.std(sensor_data)
        features[f'{sensor_name}_min'] = np.min(sensor_data)
        features[f'{sensor_name}_max'] = np.max(sensor_data)
        features[f'{sensor_name}_range'] = np.max(sensor_data) - np.min(sensor_data)
        features[f'{sensor_name}_median'] = np.median(sensor_data)
        features[f'{sensor_name}_q25'] = np.percentile(sensor_data, 25)
        features[f'{sensor_name}_q75'] = np.percentile(sensor_data, 75)
        
        # Per-axis features if available
        for col in available_cols:
            col_data = df[col].dropna().values
            if len(col_data) > 0:
                features[f'{col}_mean'] = np.mean(col_data)
                features[f'{col}_std'] = np.std(col_data)
                features[f'{col}_max_abs'] = np.max(np.abs(col_data))
    
    # Duration feature
    if 'timestamp' in df.columns and len(df) > 1:
        duration = df['timestamp'].max() - df['timestamp'].min()
        features['duration'] = duration
    else:
        features['duration'] = 0.0
    
    # Number of data points
    features['num_points'] = len(df)
    
    # Correlation features (if multiple sensors available)
    if 'phone_accel_x' in df.columns and 'watch_accel_x' in df.columns:
        phone_accel = df[['phone_accel_x', 'phone_accel_y', 'phone_accel_z']].values.flatten()
        watch_accel = df[['watch_accel_x', 'watch_accel_y', 'watch_accel_z']].values.flatten()
        phone_accel = phone_accel[~np.isnan(phone_accel)]
        watch_accel = watch_accel[~np.isnan(watch_accel)]
        
        if len(phone_accel) > 0 and len(watch_accel) > 0:
            # Interpolate to same length for correlation
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


def prepare_training_data(data_dir="data"):
    """Load all samples and prepare feature matrix and labels."""
    print("📂 Loading samples from:", data_dir)
    samples, labels, file_paths = load_all_samples(data_dir)
    
    if len(samples) == 0:
        print("❌ No samples found!")
        return None, None, None
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Count by label (keep all classes separate for personalized feedback)
    from collections import Counter
    label_counts = Counter(labels)
    print(f"   Label distribution: {dict(label_counts)}")
    
    # Extract features from each sample
    print("\n🔧 Extracting features...")
    feature_list = []
    valid_indices = []
    
    for i, (sample, label, filepath) in enumerate(zip(samples, labels, file_paths)):
        try:
            features = extract_features(sample)
            feature_list.append(features)
            valid_indices.append(i)
        except Exception as e:
            print(f"⚠️  Error extracting features from {filepath}: {e}")
    
    if len(feature_list) == 0:
        print("❌ No valid features extracted!")
        return None, None, None
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_list)
    valid_labels = [labels[i] for i in valid_indices]
    
    print(f"✅ Extracted {len(feature_df.columns)} features from {len(feature_df)} samples")
    
    return feature_df, valid_labels, file_paths


def train_classifier(X, y, test_size=0.2, random_state=42):
    """Train a Random Forest classifier."""
    print("\n🎯 Training classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced'  # Handle class imbalance
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n📊 Evaluation Results:")
    print("=" * 60)
    
    # Training accuracy
    train_pred = rf.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {train_acc:.3f}")
    
    # Test accuracy
    test_pred = rf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    
    # Feature importance
    print("\n🔝 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return rf, scaler, test_acc, feature_importance


def save_model(model, scaler, feature_importance, accuracy, feature_names, model_dir="models"):
    """Save the trained model and metadata."""
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(model_dir, f"form_classifier_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, f"scaler_{timestamp}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to: {scaler_path}")
    
    # Save feature names for proper alignment during inference
    feature_names_path = os.path.join(model_dir, f"feature_names_{timestamp}.pkl")
    joblib.dump(feature_names, feature_names_path)
    print(f"💾 Feature names saved to: {feature_names_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'accuracy': float(accuracy),
        'model_type': 'RandomForestClassifier',
        'feature_names': feature_names,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved to: {metadata_path}")
    
    return model_path, scaler_path, metadata_path


def main():
    """Main training function."""
    print("=" * 60)
    print("Gym Lift Form Classifier Training")
    print("=" * 60)
    
    # Prepare data
    X, y, file_paths = prepare_training_data()
    
    if X is None:
        return
    
    # Check if we have enough samples
    from collections import Counter
    label_counts = Counter(y)
    print(f"\n📊 Sample distribution: {dict(label_counts)}")
    
    if len(label_counts) < 2:
        print("❌ Need at least 2 different labels to train a classifier!")
        return
    
    min_samples = min(label_counts.values())
    if min_samples < 5:
        print(f"⚠️  Warning: One class has only {min_samples} samples. More data recommended.")
    
    # Train model
    model, scaler, accuracy, feature_importance = train_classifier(X, y)
    
    # Save model
    model_path, scaler_path, metadata_path = save_model(model, scaler, feature_importance, accuracy, list(X.columns))
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Collect more data if accuracy is low")
    print(f"  2. Review feature importance to understand what the model uses")
    print(f"  3. Test the model on new samples")


if __name__ == "__main__":
    main()

