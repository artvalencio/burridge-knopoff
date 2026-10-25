"""
Export of the simulation results to CSV files (long format: one row per
time and block, or per time and sensor).

Note: the rows are built as one Python dict each before pandas sees them. For
the default run (10,000 samples x 10,000 blocks) the block file has 1e8 rows,
which needs tens of GB of RAM to build and several GB on disk.
"""

import numpy as np
import pandas as pd
import os
import config

class DataExporter:
    """
    Writes the simulation results to CSV files in config.CSV_DIR.
    """

    def __init__(self):
        """Create the output directories (config.CSV_DIR and config.ANIMATION_DIR) if needed."""
        self.output_dir = config.OUTPUT_DIR
        self.csv_dir = config.CSV_DIR
        self.animation_dir = config.ANIMATION_DIR
        
        # Create directories if they don't exist
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.animation_dir, exist_ok=True)
        
    def export_block_positions(self, times, positions, filename='block_positions_time.csv'):
        """
        Export block positions over time to CSV
        
        Args:
            times: list of time values (years)
            positions: list of position arrays, each of shape (n_blocks, 3)
            filename: output CSV filename
        """
        print(f"\nExporting block positions to {filename}...")
        
        # One dict per (time, block) row: simple but memory-hungry (see module docstring)
        data_rows = []

        for t_idx, (t, pos) in enumerate(zip(times, positions)):
            for block_id in range(pos.shape[0]):
                data_rows.append({
                    'time': t,
                    'block_id': block_id,
                    'x': pos[block_id, 0],
                    'y': pos[block_id, 1],
                    'z': pos[block_id, 2]
                })
        
        df = pd.DataFrame(data_rows)
        output_path = os.path.join(self.csv_dir, filename)
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} rows to {output_path}")
        
    def export_sensor_potentials(self, times, potentials, sensor_positions, filename='sensor_potential_time.csv'):
        """
        Export sensor gravitational potentials over time to CSV
        
        Args:
            times: list of time values (years)
            potentials: list of potential arrays, each of shape (n_sensors,)
            sensor_positions: array of shape (n_sensors, 3) with sensor positions
            filename: output CSV filename
        """
        print(f"\nExporting sensor potentials to {filename}...")
        
        # One row per (time, sensor); the sensor coordinates are repeated on every row
        data_rows = []

        for t_idx, (t, pot) in enumerate(zip(times, potentials)):
            for sensor_id in range(pot.shape[0]):
                data_rows.append({
                    'time': t,
                    'sensor_id': sensor_id,
                    'x_sensor': sensor_positions[sensor_id, 0],
                    'y_sensor': sensor_positions[sensor_id, 1],
                    'z_sensor': sensor_positions[sensor_id, 2],
                    'potential': pot[sensor_id]
                })
        
        df = pd.DataFrame(data_rows)
        output_path = os.path.join(self.csv_dir, filename)
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} rows to {output_path}")
        
    def export_earthquake_events(self, events, filename='earthquake_events.csv'):
        """
        Export detected earthquake events to CSV
        
        Args:
            events: list of event dictionaries
            filename: output CSV filename
        """
        if not events:
            print("\nNo earthquake events to export.")
            return
        
        print(f"\nExporting earthquake events to {filename}...")
        
        df = pd.DataFrame(events)
        
        # Epicentre coordinates in metres. KNOWN ISSUE: epicenter_i is the
        # index along y in the block storage order, so x and y are swapped.
        df['epicenter_x'] = df['epicenter_i'] * config.BLOCK_SPACING
        df['epicenter_y'] = df['epicenter_j'] * config.BLOCK_SPACING
        
        output_path = os.path.join(self.csv_dir, filename)
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} earthquake events to {output_path}")
        
    def export_all_data(self, times, positions, potentials, events, sensor_positions):
        """
        Export all simulation data to CSV files
        
        Args:
            times: list of time values
            positions: list of position arrays
            potentials: list of potential arrays
            events: list of earthquake events
            sensor_positions: array of sensor positions
        """
        print("\n" + "="*60)
        print("EXPORTING SIMULATION DATA")
        print("="*60)
        
        self.export_block_positions(times, positions)
        self.export_sensor_potentials(times, potentials, sensor_positions)
        self.export_earthquake_events(events)
        
        print("\n" + "="*60)
        print("DATA EXPORT COMPLETE")
        print("="*60)
        print(f"All files saved to: {self.csv_dir}")
