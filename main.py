"""
Main script to run Burridge-Knopoff earthquake simulation
with gravity sensor array
"""

import sys
import numpy as np
from src.burridge_knopoff import BurridgeKnopoffSystem
from src.gravity_sensors import GravitySensorArray
from src.solver import EarthquakeSimulator
from src.data_export import DataExporter
from src.visualization import Visualizer
import config

def main():
    """Main execution function"""
    
    print("="*70)
    print("BURRIDGE-KNOPOFF EARTHQUAKE SIMULATION")
    print("WITH GRAVITY SENSOR ARRAY")
    print("="*70)
    print(f"\nSimulation Parameters:")
    print(f"  Grid: {config.N_BLOCKS_X} x {config.N_BLOCKS_Y} blocks")
    print(f"  Block spacing: {config.BLOCK_SPACING/1000:.2f} km")
    print(f"  Total area: {config.N_BLOCKS_X * config.BLOCK_SPACING/1000:.1f} x {config.N_BLOCKS_Y * config.BLOCK_SPACING/1000:.1f} km")
    print(f"\n  Sensors: {config.N_SENSORS_X} x {config.N_SENSORS_Y} array")
    print(f"  Sensor height: {config.SENSOR_HEIGHT} km above fault")
    print(f"\n  Plate velocity: {config.PLATE_VELOCITY * 100:.1f} cm/year")
    print(f"  Simulation time: {config.SIMULATION_TIME} years")
    print(f"  Sampling interval: {config.DT_SAMPLE} years")
    print("="*70)
    
    # Initialize system components
    print("\n[1/5] Initializing Burridge-Knopoff system...")
    bk_system = BurridgeKnopoffSystem()
    print("     ✓ Block system initialized")
    
    print("\n[2/5] Initializing gravity sensor array...")
    gravity_sensors = GravitySensorArray()
    print("     ✓ Sensor array initialized")
    
    print("\n[3/5] Initializing simulator...")
    simulator = EarthquakeSimulator(bk_system, gravity_sensors)
    print("     ✓ Simulator ready")
    
    # Run simulation
    print("\n[4/5] Running simulation...")
    print("     This may take several minutes for a 100x100 grid...")
    try:
        times, positions, velocities, potentials, events = simulator.run_simulation()
        print("     ✓ Simulation completed successfully")
    except Exception as e:
        print(f"\n✗ ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Export data
    print("\n[5/5] Exporting results...")
    exporter = DataExporter()
    try:
        exporter.export_all_data(times, positions, potentials, events, 
                                gravity_sensors.sensor_positions)
        print("     ✓ Data export completed")
    except Exception as e:
        print(f"\n✗ ERROR during data export: {e}")
        import traceback
        traceback.print_exc()
    
    # Create visualizations
    print("\n[6/6] Creating visualizations...")
    visualizer = Visualizer(gravity_sensors)
    try:
        visualizer.create_all_visualizations(times, potentials, events)
        print("     ✓ Visualizations completed")
    except Exception as e:
        print(f"\n✗ ERROR during visualization: {e}")
        print("     Note: Visualization requires ffmpeg for animations")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    print(f"Simulation time: {times[-1]:.2f} years")
    print(f"Time samples: {len(times)}")
    print(f"Earthquake events detected: {len(events)}")
    
    if events:
        magnitudes = [e['magnitude'] for e in events]
        print(f"\nEvent statistics:")
        print(f"  Magnitude range: {min(magnitudes):.2f} - {max(magnitudes):.2f}")
        print(f"  Mean magnitude: {np.mean(magnitudes):.2f}")
        print(f"  Largest event: M{max(magnitudes):.2f} at t={events[np.argmax(magnitudes)]['time']:.2f} years")
    
    print(f"\nOutput files:")
    print(f"  CSV data: {config.CSV_DIR}/")
    print(f"  Visualizations: {config.ANIMATION_DIR}/")
    print("="*70)
    print("\n✓ ALL TASKS COMPLETED SUCCESSFULLY\n")

if __name__ == "__main__":
    main()
