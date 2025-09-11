#!/usr/bin/env python3
"""TRELLIS-compatible camera system for interactive pose manipulation.

This module implements a comprehensive camera system that matches the TRELLIS
coordinate system and provides interactive controls for camera pose manipulation.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraPose:
    """TRELLIS-compatible camera pose representation."""
    yaw: float      # degrees (0-360)
    pitch: float    # degrees (-90 to 90)
    radius: float   # distance from origin (positive)
    fov: float      # field of view in degrees (10-120)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'yaw': self.yaw,
            'pitch': self.pitch,
            'radius': self.radius,
            'fov': self.fov
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'CameraPose':
        """Create from dictionary format."""
        return cls(
            yaw=data['yaw'],
            pitch=data['pitch'],
            radius=data['radius'],
            fov=data['fov']
        )
    
    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert spherical coordinates to cartesian position.
        
        Returns:
            (x, y, z) position in 3D space
        """
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        x = self.radius * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.radius * math.sin(pitch_rad)
        z = self.radius * math.cos(pitch_rad) * math.cos(yaw_rad)
        
        return (x, y, z)
    
    @classmethod
    def from_cartesian(cls, x: float, y: float, z: float, fov: float = 40.0) -> 'CameraPose':
        """Create camera pose from cartesian coordinates.
        
        Args:
            x, y, z: 3D position
            fov: Field of view in degrees
            
        Returns:
            CameraPose object
        """
        radius = math.sqrt(x*x + y*y + z*z)
        
        if radius == 0:
            return cls(0.0, 0.0, 1.0, fov)
        
        pitch = math.degrees(math.asin(y / radius))
        yaw = math.degrees(math.atan2(x, z))
        
        # Normalize yaw to 0-360
        if yaw < 0:
            yaw += 360
        
        return cls(yaw, pitch, radius, fov)
    
    def validate(self) -> bool:
        """Validate camera pose parameters.
        
        Returns:
            True if valid, False otherwise
        """
        return (
            0 <= self.yaw <= 360 and
            -90 <= self.pitch <= 90 and
            self.radius > 0 and
            10 <= self.fov <= 120
        )
    
    def normalize(self) -> 'CameraPose':
        """Normalize parameters to valid ranges."""
        return CameraPose(
            yaw=self.yaw % 360,
            pitch=max(-90, min(90, self.pitch)),
            radius=max(0.1, self.radius),
            fov=max(10, min(120, self.fov))
        )


class TRELLISCameraSystem:
    """TRELLIS-compatible camera system with interactive controls."""
    
    def __init__(self, default_radius: float = 2.5, default_fov: float = 40.0):
        """Initialize TRELLIS camera system.
        
        Args:
            default_radius: Default camera distance
            default_fov: Default field of view in degrees
        """
        self.default_radius = default_radius
        self.default_fov = default_fov
        self.current_pose = CameraPose(0.0, 0.0, default_radius, default_fov)
        
        # Camera constraints
        self.min_radius = 0.5
        self.max_radius = 10.0
        self.min_fov = 10.0
        self.max_fov = 120.0
        self.min_pitch = -89.0  # Avoid singularities
        self.max_pitch = 89.0
        
        # Pose history for undo/redo
        self.pose_history: List[CameraPose] = [self.current_pose]
        self.history_index = 0
        self.max_history = 50
        
        logger.info(f"TRELLIS camera system initialized")
    
    def set_pose(self, yaw: float, pitch: float, radius: float, fov: Optional[float] = None) -> None:
        """Set camera pose using spherical coordinates.
        
        Args:
            yaw: Yaw angle in degrees (0-360)
            pitch: Pitch angle in degrees (-90 to 90)
            radius: Distance from origin (positive)
            fov: Field of view in degrees (optional)
        """
        if fov is None:
            fov = self.current_pose.fov
        
        new_pose = CameraPose(yaw, pitch, radius, fov).normalize()
        
        if new_pose.validate():
            self._add_to_history(new_pose)
            self.current_pose = new_pose
            logger.debug(f"Camera pose set: {new_pose}")
        else:
            logger.warning(f"Invalid camera pose rejected: {new_pose}")
    
    def set_cartesian(self, x: float, y: float, z: float, fov: Optional[float] = None) -> None:
        """Set camera pose using cartesian coordinates.
        
        Args:
            x, y, z: 3D position
            fov: Field of view in degrees (optional)
        """
        if fov is None:
            fov = self.current_pose.fov
        
        pose = CameraPose.from_cartesian(x, y, z, fov)
        self.set_pose(pose.yaw, pose.pitch, pose.radius, pose.fov)
    
    def get_pose(self) -> CameraPose:
        """Get current camera pose."""
        return self.current_pose
    
    def get_cartesian(self) -> Tuple[float, float, float]:
        """Get current camera position in cartesian coordinates."""
        return self.current_pose.to_cartesian()
    
    def orbit(self, delta_yaw: float, delta_pitch: float) -> None:
        """Orbit camera around center point.
        
        Args:
            delta_yaw: Change in yaw angle (degrees)
            delta_pitch: Change in pitch angle (degrees)
        """
        new_yaw = (self.current_pose.yaw + delta_yaw) % 360
        new_pitch = max(self.min_pitch, 
                       min(self.max_pitch, self.current_pose.pitch + delta_pitch))
        
        self.set_pose(new_yaw, new_pitch, self.current_pose.radius, self.current_pose.fov)
    
    def zoom(self, delta_radius: float) -> None:
        """Zoom camera in/out by changing radius.
        
        Args:
            delta_radius: Change in radius (negative = zoom in)
        """
        new_radius = max(self.min_radius,
                        min(self.max_radius, self.current_pose.radius + delta_radius))
        
        self.set_pose(self.current_pose.yaw, self.current_pose.pitch, new_radius, self.current_pose.fov)
    
    def set_fov(self, fov: float) -> None:
        """Set field of view.
        
        Args:
            fov: Field of view in degrees
        """
        new_fov = max(self.min_fov, min(self.max_fov, fov))
        self.set_pose(self.current_pose.yaw, self.current_pose.pitch, self.current_pose.radius, new_fov)
    
    def reset_to_default(self) -> None:
        """Reset camera to default pose."""
        self.set_pose(0.0, 0.0, self.default_radius, self.default_fov)
    
    def look_at_from_position(self, position: Tuple[float, float, float], 
                             target: Tuple[float, float, float] = (0, 0, 0)) -> None:
        """Set camera to look at target from specific position.
        
        Args:
            position: Camera position (x, y, z)
            target: Target position to look at
        """
        # Calculate relative position
        rel_x = position[0] - target[0]
        rel_y = position[1] - target[1]
        rel_z = position[2] - target[2]
        
        self.set_cartesian(rel_x, rel_y, rel_z)
    
    def generate_orbit_poses(self, num_poses: int = 8, 
                           pitch_range: Tuple[float, float] = (-30, 30),
                           radius: Optional[float] = None) -> List[CameraPose]:
        """Generate evenly distributed orbit poses around model.
        
        Args:
            num_poses: Number of poses to generate
            pitch_range: Range of pitch angles (min, max)
            radius: Fixed radius (use current if None)
            
        Returns:
            List of camera poses
        """
        if radius is None:
            radius = self.current_pose.radius
        
        poses = []
        yaw_step = 360.0 / num_poses
        
        for i in range(num_poses):
            yaw = i * yaw_step
            # Vary pitch slightly for more interesting views
            pitch = np.random.uniform(pitch_range[0], pitch_range[1])
            
            pose = CameraPose(yaw, pitch, radius, self.current_pose.fov)
            poses.append(pose)
        
        logger.info(f"Generated {len(poses)} orbit poses")
        return poses
    
    def generate_hemisphere_poses(self, num_poses: int = 12,
                                 min_elevation: float = 10.0,
                                 radius: Optional[float] = None) -> List[CameraPose]:
        """Generate poses distributed over upper hemisphere.
        
        Args:
            num_poses: Number of poses to generate
            min_elevation: Minimum elevation angle (degrees)
            radius: Fixed radius (use current if None)
            
        Returns:
            List of camera poses
        """
        if radius is None:
            radius = self.current_pose.radius
        
        poses = []
        
        # Use fibonacci spiral for even distribution
        golden_ratio = (1 + 5**0.5) / 2
        
        for i in range(num_poses):
            # Fibonacci spiral points
            y = 1 - (i / float(num_poses - 1)) * (1 - math.sin(math.radians(min_elevation)))
            radius_at_y = math.sqrt(1 - y * y)
            theta = 2 * math.pi * i / golden_ratio
            
            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y
            
            # Convert to spherical
            pose = CameraPose.from_cartesian(x * radius, y * radius, z * radius, self.current_pose.fov)
            poses.append(pose)
        
        logger.info(f"Generated {len(poses)} hemisphere poses")
        return poses
    
    def _add_to_history(self, pose: CameraPose) -> None:
        """Add pose to history for undo/redo."""
        # Remove any history after current index
        self.pose_history = self.pose_history[:self.history_index + 1]
        
        # Add new pose
        self.pose_history.append(pose)
        self.history_index += 1
        
        # Limit history size
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
            self.history_index -= 1
    
    def undo(self) -> bool:
        """Undo last camera pose change.
        
        Returns:
            True if undo was successful
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.current_pose = self.pose_history[self.history_index]
            logger.debug(f"Undo: {self.current_pose}")
            return True
        return False
    
    def redo(self) -> bool:
        """Redo camera pose change.
        
        Returns:
            True if redo was successful
        """
        if self.history_index < len(self.pose_history) - 1:
            self.history_index += 1
            self.current_pose = self.pose_history[self.history_index]
            logger.debug(f"Redo: {self.current_pose}")
            return True
        return False
    
    def save_poses(self, poses: List[CameraPose], filepath: Union[str, Path]) -> None:
        """Save camera poses to JSON file.
        
        Args:
            poses: List of camera poses
            filepath: Output file path
        """
        filepath = Path(filepath)
        
        data = {
            'version': '1.0',
            'coordinate_system': 'TRELLIS',
            'poses': [pose.to_dict() for pose in poses]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(poses)} poses to {filepath}")
    
    def load_poses(self, filepath: Union[str, Path]) -> List[CameraPose]:
        """Load camera poses from JSON file.
        
        Args:
            filepath: Input file path
            
        Returns:
            List of camera poses
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        poses = [CameraPose.from_dict(pose_data) for pose_data in data['poses']]
        
        logger.info(f"Loaded {len(poses)} poses from {filepath}")
        return poses
    
    def get_transform_matrix(self) -> np.ndarray:
        """Get camera transform matrix (4x4) for the current pose.
        
        Returns:
            4x4 transformation matrix
        """
        position = self.get_cartesian()
        
        # Camera looks at origin
        forward = np.array([0.0, 0.0, 0.0]) - np.array(position)
        forward = forward / np.linalg.norm(forward)
        
        # World up vector
        up = np.array([0.0, 1.0, 0.0])
        
        # Right vector
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        # Recalculate up vector
        up = np.cross(forward, right)
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[0:3, 0] = right
        transform[0:3, 1] = up
        transform[0:3, 2] = forward
        transform[0:3, 3] = position
        
        return transform
    
    def interpolate_poses(self, start_pose: CameraPose, end_pose: CameraPose, 
                         num_steps: int = 10) -> List[CameraPose]:
        """Interpolate between two camera poses.
        
        Args:
            start_pose: Starting camera pose
            end_pose: Ending camera pose
            num_steps: Number of interpolation steps
            
        Returns:
            List of interpolated poses
        """
        poses = []
        
        for i in range(num_steps):
            t = i / (num_steps - 1)  # 0 to 1
            
            # Interpolate each parameter
            yaw = self._interpolate_angle(start_pose.yaw, end_pose.yaw, t)
            pitch = start_pose.pitch + t * (end_pose.pitch - start_pose.pitch)
            radius = start_pose.radius + t * (end_pose.radius - start_pose.radius)
            fov = start_pose.fov + t * (end_pose.fov - start_pose.fov)
            
            pose = CameraPose(yaw, pitch, radius, fov)
            poses.append(pose)
        
        return poses
    
    def _interpolate_angle(self, start: float, end: float, t: float) -> float:
        """Interpolate between two angles, handling 360° wraparound."""
        # Normalize angles to 0-360
        start = start % 360
        end = end % 360
        
        # Find shortest path
        diff = end - start
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        result = start + t * diff
        return result % 360


# Convenience functions

def create_trellis_camera(radius: float = 2.5, fov: float = 40.0) -> TRELLISCameraSystem:
    """Create a TRELLIS camera system with default settings.
    
    Args:
        radius: Default camera distance
        fov: Default field of view
        
    Returns:
        TRELLISCameraSystem instance
    """
    return TRELLISCameraSystem(radius, fov)


def generate_standard_views(camera: TRELLISCameraSystem) -> List[CameraPose]:
    """Generate standard viewing angles.
    
    Args:
        camera: TRELLIS camera system
        
    Returns:
        List of standard camera poses
    """
    radius = camera.current_pose.radius
    fov = camera.current_pose.fov
    
    return [
        CameraPose(0.0, 0.0, radius, fov),      # Front
        CameraPose(90.0, 0.0, radius, fov),     # Right
        CameraPose(180.0, 0.0, radius, fov),    # Back
        CameraPose(270.0, 0.0, radius, fov),    # Left
        CameraPose(45.0, 45.0, radius, fov),    # Top-front-right
        CameraPose(45.0, -30.0, radius, fov),   # Bottom-front-right
        CameraPose(135.0, 30.0, radius, fov),   # Top-back-right
        CameraPose(225.0, 0.0, radius, fov),    # Back-left
    ]