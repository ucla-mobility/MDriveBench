"""
Trajectory Logger for SafeBench Scenic Runner
Logs all agent state data (position, velocity, acceleration, orientation) at each step
for later conversion to CoLMDriver format. This includes ego, dynamic NPCs, and
static scene props spawned by Scenic.

Outputs:
  - CSV: Human-readable trajectory data with all agent states
  - NPZ: Numpy-compressed binary format for efficient storage and loading
  - Manifest JSON: Metadata about agents and episode parameters
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ActorState:
    """State of a single actor at a timestep"""
    timestamp: float
    actor_id: str
    actor_type: str  # 'ego', 'npc', etc.
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    velocity: float
    ax: float
    ay: float
    az: float
    roll: float
    pitch: float
    yaw: float
    vehicle_model: str = ""  # exact CARLA blueprint e.g. vehicle.kawasaki.ninja


class TrajectoryLogger:
    """
    Logs trajectory data from SafeBench simulation for all agents at each timestep.
    
    Supports two output formats:
    1. CSV: Easy to inspect, human-readable
    2. NPZ: Efficient storage, quick loading for CoLMDriver conversion
    """
    
    def __init__(self, output_dir: Path, scenario_name: str, log_interval: int = 1, town: str = None):
        """
        Initialize trajectory logger.
        
        Args:
            output_dir: Directory to save trajectory logs
            scenario_name: Name of the scenario being logged
            log_interval: Log every N simulation steps (1 = every step, 10 = every 10 steps)
            town: CARLA town name (e.g., 'Town01', 'Town05') for scenario replay
        """
        self.output_dir = Path(output_dir)
        self.scenario_name = scenario_name
        self.log_interval = log_interval
        self.town = town
        self.step_counter = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file for human-readable trajectory data
        self.csv_path = self.output_dir / f"{scenario_name}_trajectory.csv"
        self.csv_file = None
        self.csv_writer = None
        self._init_csv()
        
        # In-memory storage for NPZ output
        self.trajectories: Dict[str, List[ActorState]] = {}
        self.episode_metadata: Dict[str, Any] = {
            'scenario_name': scenario_name,
            'log_interval': log_interval,
            'total_steps': 0,
            'agents': {},
            'town': town,
        }
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        self.csv_file = open(self.csv_path, 'w', newline='')
        fieldnames = [
            'step', 'timestamp', 'actor_id', 'actor_type', 'vehicle_model',
            'x', 'y', 'z',
            'vx', 'vy', 'vz', 'velocity',
            'ax', 'ay', 'az',
            'roll', 'pitch', 'yaw'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
    
    def log_step(self, running_status: Dict[str, Any], game_time: float):
        """
        Log a single simulation step from the running_status dict.
        
        Args:
            running_status: Dict with ego and actor states from scenario.get_running_status()
            game_time: Current simulation time
        """
        self.step_counter += 1
        
        # Only log at specified intervals
        if self.step_counter % self.log_interval != 0:
            return
        
        # Debug: print keys on first step to see what's available
        if self.step_counter == 1:
            static_keys = [k for k in running_status.keys() if 'static' in k.lower()]
            adv_keys = [k for k in running_status.keys() if 'adv_agent' in k.lower()]
            print(f"[TrajectoryLogger] First step - adv_keys: {adv_keys}, static_keys: {static_keys}")
        
        # Extract ego vehicle data
        if 'ego_x' in running_status:
            ego_state = ActorState(
                timestamp=game_time,
                actor_id='ego_0',
                actor_type='ego',
                x=running_status['ego_x'],
                y=running_status['ego_y'],
                z=running_status.get('ego_z', 0.0),
                vx=running_status.get('ego_velocity_x', 0.0),
                vy=running_status.get('ego_velocity_y', 0.0),
                vz=running_status.get('ego_velocity_z', 0.0),
                velocity=running_status.get('ego_velocity', 0.0),
                ax=running_status.get('ego_acceleration_x', 0.0),
                ay=running_status.get('ego_acceleration_y', 0.0),
                az=running_status.get('ego_acceleration_z', 0.0),
                roll=running_status.get('ego_roll', 0.0),
                pitch=running_status.get('ego_pitch', 0.0),
                yaw=running_status.get('ego_yaw', 0.0),
            )
            self._log_actor_state(ego_state, self.step_counter)
        
        # Extract NPC/adversarial actor data
        # Format: adv_agent_0, adv_agent_1, etc.
        npc_index = 0
        static_count = 0
        for key, value in running_status.items():
            if key.startswith('adv_agent_'):
                actor_type = value.get('actor_type', 'npc') if isinstance(value, dict) else 'npc'
                vehicle_model = value.get('vehicle_model', '') if isinstance(value, dict) else ''
                actor_state = ActorState(
                    timestamp=game_time,
                    actor_id=key,
                    actor_type=actor_type,
                    x=value.get('x', 0.0),
                    y=value.get('y', 0.0),
                    z=value.get('z', 0.0),
                    vx=value.get('velocity_x', 0.0),
                    vy=value.get('velocity_y', 0.0),
                    vz=value.get('velocity_z', 0.0),
                    velocity=value.get('velocity', 0.0),
                    ax=value.get('acceleration_x', 0.0),
                    ay=value.get('acceleration_y', 0.0),
                    az=value.get('acceleration_z', 0.0),
                    roll=value.get('roll', 0.0),
                    pitch=value.get('pitch', 0.0),
                    yaw=value.get('yaw', 0.0),
                    vehicle_model=vehicle_model,
                )
                self._log_actor_state(actor_state, self.step_counter)
                npc_index += 1
            elif key.startswith('static_prop_'):
                actor_type = value.get('actor_type', 'static') if isinstance(value, dict) else 'static'
                vehicle_model = value.get('vehicle_model', '') if isinstance(value, dict) else ''
                actor_state = ActorState(
                    timestamp=game_time,
                    actor_id=key,
                    actor_type=actor_type,
                    x=value.get('x', 0.0),
                    y=value.get('y', 0.0),
                    z=value.get('z', 0.0),
                    vx=value.get('velocity_x', 0.0),
                    vy=value.get('velocity_y', 0.0),
                    vz=value.get('velocity_z', 0.0),
                    velocity=value.get('velocity', 0.0),
                    ax=value.get('acceleration_x', 0.0),
                    ay=value.get('acceleration_y', 0.0),
                    az=value.get('acceleration_z', 0.0),
                    roll=value.get('roll', 0.0),
                    pitch=value.get('pitch', 0.0),
                    yaw=value.get('yaw', 0.0),
                    vehicle_model=vehicle_model,
                )
                self._log_actor_state(actor_state, self.step_counter)
    
    def _log_actor_state(self, state: ActorState, step: int):
        """Write actor state to CSV and store in memory"""
        # Write to CSV
        row = {
            'step': step,
            **asdict(state)
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        # Store in memory for NPZ export
        if state.actor_id not in self.trajectories:
            self.trajectories[state.actor_id] = []
        self.trajectories[state.actor_id].append(state)
        
        # Track agent metadata
        if state.actor_id not in self.episode_metadata['agents']:
            self.episode_metadata['agents'][state.actor_id] = {
                'type': state.actor_type,
                'vehicle_model': state.vehicle_model,
                'first_step': step,
            }
        self.episode_metadata['agents'][state.actor_id]['last_step'] = step
    
    def save_npz(self):
        """Save all trajectories to NPZ format for efficient loading"""
        npz_path = self.output_dir / f"{self.scenario_name}_trajectory.npz"
        
        # Convert trajectories to structured arrays for each actor
        actor_arrays = {}
        for actor_id, states in self.trajectories.items():
            if not states:
                continue
            
            # Create structured array with all fields
            dtype = np.dtype([
                ('timestamp', float),
                ('x', float),
                ('y', float),
                ('z', float),
                ('vx', float),
                ('vy', float),
                ('vz', float),
                ('velocity', float),
                ('ax', float),
                ('ay', float),
                ('az', float),
                ('roll', float),
                ('pitch', float),
                ('yaw', float),
            ])
            
            data = np.array(
                [(s.timestamp, s.x, s.y, s.z, s.vx, s.vy, s.vz, s.velocity,
                  s.ax, s.ay, s.az, s.roll, s.pitch, s.yaw) for s in states],
                dtype=dtype
            )
            actor_arrays[actor_id] = data
        
        # Update metadata
        self.episode_metadata['total_steps'] = self.step_counter
        self.episode_metadata['logged_steps'] = sum(len(states) for states in self.trajectories.values()) // len(self.trajectories) if self.trajectories else 0
        
        # Save all arrays
        np.savez_compressed(npz_path, metadata=self.episode_metadata, **actor_arrays)
        print(f"Saved trajectory NPZ to {npz_path}")
        return npz_path
    
    def save_manifest(self):
        """Save episode metadata as JSON"""
        manifest_path = self.output_dir / f"{self.scenario_name}_manifest.json"
        self.episode_metadata['total_steps'] = self.step_counter
        
        with open(manifest_path, 'w') as f:
            json.dump(self.episode_metadata, f, indent=2)
        
        print(f"Saved manifest to {manifest_path}")
        return manifest_path
    
    def close(self):
        """Close CSV file and save final outputs"""
        if self.csv_file:
            self.csv_file.close()
        
        print(f"Saved trajectory CSV to {self.csv_path}")
        self.save_npz()
        self.save_manifest()


class TrajectoryReader:
    """
    Read previously logged trajectory data in NPZ format.
    Used for converting SafeBench trajectories to CoLMDriver format.
    """
    
    def __init__(self, npz_path: Path):
        """Load NPZ trajectory file"""
        self.npz_path = Path(npz_path)
        self.data = np.load(npz_path, allow_pickle=True)
        self.metadata = dict(self.data['metadata'].item())
        self.actors = {key: self.data[key] for key in self.data.files if key != 'metadata'}
    
    def get_actor_trajectory(self, actor_id: str) -> Optional[np.ndarray]:
        """Get trajectory for a specific actor"""
        return self.actors.get(actor_id)
    
    def get_all_actors(self) -> Dict[str, np.ndarray]:
        """Get all actor trajectories"""
        return self.actors
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get episode metadata"""
        return self.metadata
    
    def get_csv_path(self) -> Optional[Path]:
        """Get path to corresponding CSV file"""
        csv_path = self.npz_path.parent / self.npz_path.name.replace('_trajectory.npz', '_trajectory.csv')
        if csv_path.exists():
            return csv_path
        return None
