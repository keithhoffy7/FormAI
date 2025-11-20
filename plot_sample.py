#!/usr/bin/env python3
"""
Plot CSV sample data to visualize sensor readings.

Usage:
    python plot_sample.py <csv_file>
    python plot_sample.py  # Will prompt for file
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path


def plot_csv_data(csv_file):
    """Plot accelerometer and gyroscope data from CSV file."""
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return False
    
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    # Check required columns
    required_cols = ['timestamp']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ CSV missing required columns. Found: {df.columns.tolist()}")
        return False
    
    # Calculate relative time (seconds from start) for the overall timeline
    if len(df) > 0:
        start_time = df['timestamp'].min()
        df['time_seconds'] = df['timestamp'] - start_time
    else:
        print("❌ CSV file is empty")
        return False
    
    # Create a common time grid by interpolating to align phone and watch data
    # This ensures both sensors are plotted on the same time axis
    from scipy import interpolate
    import numpy as np
    
    # Find the overall time range
    all_times = df['time_seconds'].values
    time_min = all_times.min()
    time_max = all_times.max()
    
    # Create a common time grid (use the higher sampling rate)
    # Estimate sampling rate from non-null data
    phone_times = df[df['phone_accel_x'].notna()]['time_seconds'].values if 'phone_accel_x' in df.columns else []
    watch_times = df[df['watch_accel_x'].notna()]['time_seconds'].values if 'watch_accel_x' in df.columns else []
    
    if len(phone_times) > 1:
        phone_rate = 1.0 / np.mean(np.diff(phone_times))
    else:
        phone_rate = 100  # default
    
    if len(watch_times) > 1:
        watch_rate = 1.0 / np.mean(np.diff(watch_times))
    else:
        watch_rate = 100  # default
    
    # Use the higher sampling rate for the common grid
    common_rate = max(phone_rate, watch_rate, 50)  # at least 50 Hz
    common_times = np.linspace(time_min, time_max, int((time_max - time_min) * common_rate))
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'Sensor Data: {os.path.basename(csv_file)}', fontsize=16, fontweight='bold')
    
    # Helper function to interpolate data to common time grid
    def interpolate_to_common_grid(data_col):
        """Interpolate sensor data to the common time grid."""
        # Get valid data points
        mask = data_col.notna()
        if mask.sum() < 2:  # Need at least 2 points to interpolate
            return common_times, np.full_like(common_times, np.nan)
        
        valid_times = df.loc[mask, 'time_seconds'].values
        valid_data = data_col[mask].values
        
        # Interpolate to common time grid
        try:
            f = interpolate.interp1d(valid_times, valid_data, kind='linear', 
                                   bounds_error=False, fill_value=np.nan)
            interpolated = f(common_times)
            return common_times, interpolated
        except:
            # Fallback: return original data if interpolation fails
            return valid_times, valid_data
    
    # Plot 1: Phone Accelerometer
    ax1 = axes[0]
    phone_accel_cols = [col for col in df.columns if 'phone_accel' in col]
    if phone_accel_cols:
        for col in phone_accel_cols:
            axis_name = col.split('_')[-1].upper()
            time_vals, data_vals = interpolate_to_common_grid(df[col])
            # Remove NaN values for plotting
            valid_mask = ~np.isnan(data_vals)
            if valid_mask.sum() > 0:
                ax1.plot(time_vals[valid_mask], data_vals[valid_mask], 
                        label=f'Phone Accel {axis_name}', linewidth=1.5)
    ax1.set_ylabel('Acceleration (m/s²)', fontsize=11)
    ax1.set_title('Phone Accelerometer', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Phone Gyroscope
    ax2 = axes[1]
    phone_gyro_cols = [col for col in df.columns if 'phone_gyro' in col]
    if phone_gyro_cols:
        for col in phone_gyro_cols:
            axis_name = col.split('_')[-1].upper()
            time_vals, data_vals = interpolate_to_common_grid(df[col])
            valid_mask = ~np.isnan(data_vals)
            if valid_mask.sum() > 0:
                ax2.plot(time_vals[valid_mask], data_vals[valid_mask], 
                        label=f'Phone Gyro {axis_name}', linewidth=1.5)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
    ax2.set_title('Phone Gyroscope', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Plot 3: Watch Accelerometer
    ax3 = axes[2]
    watch_accel_cols = [col for col in df.columns if 'watch_accel' in col]
    if watch_accel_cols:
        for col in watch_accel_cols:
            axis_name = col.split('_')[-1].upper()
            time_vals, data_vals = interpolate_to_common_grid(df[col])
            valid_mask = ~np.isnan(data_vals)
            if valid_mask.sum() > 0:
                ax3.plot(time_vals[valid_mask], data_vals[valid_mask], 
                        label=f'Watch Accel {axis_name}', linewidth=1.5)
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=11)
    ax3.set_title('Apple Watch Accelerometer', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # Plot 4: Watch Gyroscope
    ax4 = axes[3]
    watch_gyro_cols = [col for col in df.columns if 'watch_gyro' in col]
    if watch_gyro_cols:
        for col in watch_gyro_cols:
            axis_name = col.split('_')[-1].upper()
            time_vals, data_vals = interpolate_to_common_grid(df[col])
            valid_mask = ~np.isnan(data_vals)
            if valid_mask.sum() > 0:
                ax4.plot(time_vals[valid_mask], data_vals[valid_mask], 
                        label=f'Watch Gyro {axis_name}', linewidth=1.5)
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
    ax4.set_title('Apple Watch Gyroscope', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    # Set same x-axis range for all plots for better comparison
    if len(common_times) > 0:
        x_min = common_times.min()
        x_max = common_times.max()
        for ax in axes:
            ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print(f"Data Summary: {os.path.basename(csv_file)}")
    print("=" * 60)
    print(f"Total data points: {len(df)}")
    print(f"Duration: {df['time_seconds'].iloc[-1]:.2f} seconds")
    print(f"Sampling rate: ~{len(df) / df['time_seconds'].iloc[-1]:.1f} Hz")
    print("\nData availability:")
    
    sensor_types = [
        ('Phone Accelerometer', phone_accel_cols),
        ('Phone Gyroscope', phone_gyro_cols),
        ('Watch Accelerometer', watch_accel_cols),
        ('Watch Gyroscope', watch_gyro_cols)
    ]
    
    for sensor_name, cols in sensor_types:
        if cols:
            non_null_count = sum([df[col].notna().sum() for col in cols])
            total_expected = len(df) * len(cols)
            percentage = (non_null_count / total_expected * 100) if total_expected > 0 else 0
            print(f"  {sensor_name}: {non_null_count}/{total_expected} values ({percentage:.1f}%)")
        else:
            print(f"  {sensor_name}: No data")
    
    print("=" * 60 + "\n")
    
    # Show plot
    plt.show()
    
    return True


def find_csv_files(directory="data"):
    """Find all CSV files in the data directory."""
    csv_files = []
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    csv_files.append(full_path)
    return sorted(csv_files)


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # File provided as argument
        csv_file = sys.argv[1]
        plot_csv_data(csv_file)
    else:
        # Interactive mode - let user choose file
        print("=" * 60)
        print("CSV Data Plotter")
        print("=" * 60)
        
        csv_files = find_csv_files()
        
        if not csv_files:
            print("❌ No CSV files found in 'data' directory")
            print("\nUsage: python plot_sample.py <path_to_csv_file>")
            return
        
        print(f"\nFound {len(csv_files)} CSV file(s):\n")
        for i, file in enumerate(csv_files, 1):
            rel_path = os.path.relpath(file)
            print(f"  {i}. {rel_path}")
        
        print(f"\n  {len(csv_files) + 1}. Enter custom path")
        print(f"  0. Exit")
        
        try:
            choice = input("\nSelect file number: ").strip()
            
            if choice == "0":
                print("Goodbye!")
                return
            elif choice == str(len(csv_files) + 1):
                custom_path = input("Enter CSV file path: ").strip()
                plot_csv_data(custom_path)
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    plot_csv_data(csv_files[idx])
                else:
                    print("❌ Invalid selection")
        except (ValueError, KeyboardInterrupt):
            print("\nGoodbye!")


if __name__ == "__main__":
    main()

