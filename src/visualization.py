"""
Visualization module for creating animations and plots
of gravitational potential changes and earthquake events
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import os
import config

class Visualizer:
    """
    Creates visualizations and animations of simulation results
    """
    
    def __init__(self, gravity_sensors):
        """
        Initialize visualizer
        
        Args:
            gravity_sensors: GravitySensorArray instance
        """
        self.gravity_sensors = gravity_sensors
        self.animation_dir = config.ANIMATION_DIR
        
        # Ensure animation directory exists
        os.makedirs(self.animation_dir, exist_ok=True)
        
    def create_potential_animation(self, times, potentials, events, 
                                   output_filename='gravity_potential_animation.mp4'):
        """
        Create animation of gravitational potential changes over time
        
        Args:
            times: list of time values (years)
            potentials: list of potential arrays (n_sensors,)
            events: list of earthquake event dictionaries
            output_filename: name of output animation file
        """
        print(f"\nCreating gravitational potential animation...")
        
        # Get sensor grid coordinates
        x_grid, y_grid = self.gravity_sensors.get_sensor_grid_coordinates()
        
        # Convert potentials to 2D grid for visualization
        potential_grids = []
        for pot in potentials:
            pot_grid = pot.reshape(config.N_SENSORS_Y, config.N_SENSORS_X)
            potential_grids.append(pot_grid)
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine color scale limits
        all_potentials = np.concatenate([p.flatten() for p in potential_grids])
        vmin = np.percentile(all_potentials, 1)
        vmax = np.percentile(all_potentials, 99)
        
        # Create colormap with diverging colors centered at zero
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        # Initial plot
        im = ax.pcolormesh(x_grid/1000, y_grid/1000, potential_grids[0],
                          cmap='RdBu_r', norm=norm, shading='auto')
        
        cbar = plt.colorbar(im, ax=ax, label='Gravitational Potential Change (J/kg)')
        ax.set_xlabel('X Position (km)')
        ax.set_ylabel('Y Position (km)')
        ax.set_title(f'Gravitational Potential Change - Time: {times[0]:.3f} years')
        ax.set_aspect('equal')
        
        # Add event markers
        event_markers = []
        
        def update(frame):
            """Update function for animation"""
            im.set_array(potential_grids[frame].ravel())
            ax.set_title(f'Gravitational Potential Change - Time: {times[frame]:.3f} years')
            
            # Clear previous event markers
            for marker in event_markers:
                marker.remove()
            event_markers.clear()
            
            # Add markers for recent events
            current_time = times[frame]
            for event in events:
                if abs(event['time'] - current_time) < 1.0:  # Within 1 year
                    x_epi = event['epicenter_i'] * config.BLOCK_SPACING / 1000
                    y_epi = event['epicenter_j'] * config.BLOCK_SPACING / 1000
                    marker = ax.plot(x_epi, y_epi, 'y*', markersize=15, 
                                   markeredgecolor='black', markeredgewidth=1)[0]
                    event_markers.append(marker)
            
            return [im] + event_markers
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(times),
                                      interval=50, blit=True, repeat=True)
        
        # Save animation
        output_path = os.path.join(self.animation_dir, output_filename)
        anim.save(output_path, writer='ffmpeg', fps=20, dpi=100)
        
        plt.close(fig)
        print(f"Animation saved to {output_path}")
        
    def plot_event_comparison(self, times, potentials, events, 
                             event_idx=0, time_window=5.0,
                             output_filename='event_comparison.png'):
        """
        Create before/after comparison plots for a specific earthquake event
        
        Args:
            times: list of time values (years)
            potentials: list of potential arrays
            events: list of earthquake events
            event_idx: index of event to visualize
            time_window: time window (years) before/after event
            output_filename: name of output file
        """
        if not events or event_idx >= len(events):
            print("No events available for comparison plot.")
            return
        
        event = events[event_idx]
        event_time = event['time']
        
        print(f"\nCreating before/after comparison for event at t={event_time:.3f} years...")
        
        # Find time indices
        times_array = np.array(times)
        before_idx = np.argmin(np.abs(times_array - (event_time - time_window)))
        after_idx = np.argmin(np.abs(times_array - (event_time + time_window)))
        event_time_idx = np.argmin(np.abs(times_array - event_time))
        
        # Get potentials
        pot_before = potentials[before_idx].reshape(config.N_SENSORS_Y, config.N_SENSORS_X)
        pot_event = potentials[event_time_idx].reshape(config.N_SENSORS_Y, config.N_SENSORS_X)
        pot_after = potentials[after_idx].reshape(config.N_SENSORS_Y, config.N_SENSORS_X)
        
        # Get sensor grid
        x_grid, y_grid = self.gravity_sensors.get_sensor_grid_coordinates()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Determine common color scale
        vmin = min(pot_before.min(), pot_event.min(), pot_after.min())
        vmax = max(pot_before.max(), pot_event.max(), pot_after.max())
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        # Before plot
        im0 = axes[0].pcolormesh(x_grid/1000, y_grid/1000, pot_before,
                                cmap='RdBu_r', norm=norm, shading='auto')
        axes[0].set_xlabel('X Position (km)')
        axes[0].set_ylabel('Y Position (km)')
        axes[0].set_title(f'Before Event\n(t = {times[before_idx]:.3f} years)')
        axes[0].set_aspect('equal')
        
        # During event plot
        im1 = axes[1].pcolormesh(x_grid/1000, y_grid/1000, pot_event,
                                cmap='RdBu_r', norm=norm, shading='auto')
        axes[1].set_xlabel('X Position (km)')
        axes[1].set_ylabel('Y Position (km)')
        axes[1].set_title(f'During Event\n(t = {times[event_time_idx]:.3f} years)')
        axes[1].set_aspect('equal')
        
        # Mark epicenter
        x_epi = event['epicenter_i'] * config.BLOCK_SPACING / 1000
        y_epi = event['epicenter_j'] * config.BLOCK_SPACING / 1000
        axes[1].plot(x_epi, y_epi, 'y*', markersize=20, 
                    markeredgecolor='black', markeredgewidth=2, label='Epicenter')
        axes[1].legend()
        
        # After plot
        im2 = axes[2].pcolormesh(x_grid/1000, y_grid/1000, pot_after,
                                cmap='RdBu_r', norm=norm, shading='auto')
        axes[2].set_xlabel('X Position (km)')
        axes[2].set_ylabel('Y Position (km)')
        axes[2].set_title(f'After Event\n(t = {times[after_idx]:.3f} years)')
        axes[2].set_aspect('equal')
        
        # Add colorbar
        fig.colorbar(im1, ax=axes, label='Gravitational Potential Change (J/kg)')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.animation_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Comparison plot saved to {output_path}")
        
    def plot_magnitude_frequency(self, events, output_filename='magnitude_frequency.png'):
        """
        Plot magnitude-frequency distribution (Gutenberg-Richter law)
        
        Args:
            events: list of earthquake events
            output_filename: name of output file
        """
        if not events:
            print("No events available for magnitude-frequency plot.")
            return
        
        print("\nCreating magnitude-frequency plot...")
        
        magnitudes = [e['magnitude'] for e in events]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(magnitudes, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Frequency')
        ax.set_title('Earthquake Magnitude-Frequency Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.animation_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Magnitude-frequency plot saved to {output_path}")
        
    def create_all_visualizations(self, times, potentials, events):
        """
        Create all standard visualizations
        
        Args:
            times: list of time values
            potentials: list of potential arrays
            events: list of earthquake events
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Create animation
        self.create_potential_animation(times, potentials, events)
        
        # Create event comparison plots for first few events
        num_events_to_plot = min(3, len(events))
        for i in range(num_events_to_plot):
            self.plot_event_comparison(times, potentials, events, 
                                      event_idx=i,
                                      output_filename=f'event_{i+1}_comparison.png')
        
        # Create magnitude-frequency plot
        self.plot_magnitude_frequency(events)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"All visualizations saved to: {self.animation_dir}")
