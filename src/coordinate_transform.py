"""Coordinate transformation utilities for aligning point clouds with camera space."""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CoordinateTransform:
    """Handles coordinate transformations for point clouds and cameras."""
    
    def __init__(self, scale: float = 1.0, offset: np.ndarray = None):
        """Initialize the coordinate transformer.
        
        Args:
            scale: Scale factor to apply
            offset: 3D offset vector to apply
        """
        self.scale = scale
        self.offset = offset if offset is not None else np.zeros(3, dtype=np.float32)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply scale and offset transformation to points.
        
        The transformation is: transformed = points * scale + offset
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Transformed points
        """
        if points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3, got shape {points.shape}")
        
        # Apply transformation: scale first, then offset
        transformed = points * self.scale + self.offset
        
        logger.debug(f"Transformed {len(points)} points with scale={self.scale}, offset={self.offset}")
        
        return transformed
    
    def transform_bounds(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform bounding box coordinates.
        
        Args:
            min_bounds: Minimum bounds (3D)
            max_bounds: Maximum bounds (3D)
            
        Returns:
            Tuple of (transformed_min, transformed_max)
        """
        # Transform both corners
        transformed_min = min_bounds * self.scale + self.offset
        transformed_max = max_bounds * self.scale + self.offset
        
        # Ensure min/max relationship is preserved
        actual_min = np.minimum(transformed_min, transformed_max)
        actual_max = np.maximum(transformed_min, transformed_max)
        
        return actual_min, actual_max
    
    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply inverse transformation to points.
        
        The inverse transformation is: original = (transformed - offset) / scale
        
        Args:
            points: Nx3 array of transformed points
            
        Returns:
            Original points
        """
        if self.scale == 0:
            raise ValueError("Cannot invert transform with scale=0")
        
        original = (points - self.offset) / self.scale
        
        return original
    
    def transform_camera_pose(self, pose_matrix: np.ndarray) -> np.ndarray:
        """Transform a camera pose matrix to align with the transformed space.
        
        Args:
            pose_matrix: 4x4 transformation matrix
            
        Returns:
            Transformed 4x4 matrix
        """
        if pose_matrix.shape != (4, 4):
            raise ValueError(f"Pose matrix must be 4x4, got shape {pose_matrix.shape}")
        
        # Create transformation matrix for scale and offset
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] *= self.scale  # Scale the rotation part
        transform[:3, 3] = pose_matrix[:3, 3] * self.scale + self.offset  # Transform translation
        
        # The rotation part remains the same, only translation is transformed
        transformed = pose_matrix.copy()
        transformed[:3, 3] = transform[:3, 3]
        
        return transformed
    
    def fit_to_bounds(self, points: np.ndarray, target_min: np.ndarray, target_max: np.ndarray) -> 'CoordinateTransform':
        """Fit points to target bounds by computing appropriate scale and offset.
        
        Args:
            points: Nx3 array of points
            target_min: Target minimum bounds (3D)
            target_max: Target maximum bounds (3D)
            
        Returns:
            New CoordinateTransform instance with computed parameters
        """
        if len(points) == 0:
            return CoordinateTransform()
        
        # Get current bounds
        current_min = np.min(points, axis=0)
        current_max = np.max(points, axis=0)
        
        # Compute extent
        current_extent = current_max - current_min
        target_extent = target_max - target_min
        
        # Avoid division by zero
        current_extent = np.where(current_extent > 0, current_extent, 1.0)
        
        # Compute scale (use minimum to fit within bounds)
        scales = target_extent / current_extent
        scale = np.min(scales)
        
        # Compute offset to center in target bounds
        current_center = (current_min + current_max) / 2
        target_center = (target_min + target_max) / 2
        offset = target_center - current_center * scale
        
        logger.info(f"Computed transform to fit bounds: scale={scale}, offset={offset}")
        
        return CoordinateTransform(scale=scale, offset=offset)
    
    def compose(self, other: 'CoordinateTransform') -> 'CoordinateTransform':
        """Compose this transform with another.
        
        The result applies 'other' first, then 'self'.
        
        Args:
            other: Another CoordinateTransform
            
        Returns:
            New composed transform
        """
        # Composition: T1(T2(x)) = T2(x) * scale1 + offset1
        #                        = (x * scale2 + offset2) * scale1 + offset1
        #                        = x * (scale2 * scale1) + (offset2 * scale1 + offset1)
        
        new_scale = other.scale * self.scale
        new_offset = other.offset * self.scale + self.offset
        
        return CoordinateTransform(scale=new_scale, offset=new_offset)
    
    def __str__(self) -> str:
        """String representation."""
        return f"CoordinateTransform(scale={self.scale}, offset={self.offset})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()


class AlignedCoordinateSystem:
    """Manages alignment between point cloud and camera coordinate systems."""
    
    def __init__(self, transform_data: dict):
        """Initialize from transform data.
        
        Args:
            transform_data: Dictionary with 'scale', 'offset', and 'aabb' fields
        """
        self.scale = float(transform_data.get('scale', 1.0))
        self.offset = np.array(transform_data.get('offset', [0, 0, 0]), dtype=np.float32)
        
        aabb = transform_data.get('aabb', [[-1, -1, -1], [1, 1, 1]])
        self.aabb_min = np.array(aabb[0], dtype=np.float32)
        self.aabb_max = np.array(aabb[1], dtype=np.float32)
        
        self.transform = CoordinateTransform(scale=self.scale, offset=self.offset)
    
    def align_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Align point cloud to camera coordinate system.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Aligned points
        """
        return self.transform.transform_points(points)
    
    def align_camera_poses(self, pose_matrices: list) -> list:
        """Align camera poses with the point cloud space.
        
        Args:
            pose_matrices: List of 4x4 camera pose matrices
            
        Returns:
            List of aligned pose matrices
        """
        aligned_poses = []
        for pose in pose_matrices:
            aligned = self.transform.transform_camera_pose(pose)
            aligned_poses.append(aligned)
        
        return aligned_poses
    
    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the scene bounds after transformation.
        
        Returns:
            Tuple of (min_bounds, max_bounds)
        """
        return self.transform.transform_bounds(self.aabb_min, self.aabb_max)
    
    def validate_alignment(self, points: np.ndarray, camera_positions: list) -> bool:
        """Validate that points and cameras are properly aligned.
        
        Args:
            points: Point cloud points
            camera_positions: List of camera positions
            
        Returns:
            True if alignment looks valid
        """
        if len(points) == 0 or len(camera_positions) == 0:
            logger.warning("Empty points or cameras for validation")
            return False
        
        # Transform points
        aligned_points = self.align_point_cloud(points)
        
        # Check if points are within expected bounds
        scene_min, scene_max = self.get_scene_bounds()
        points_min = np.min(aligned_points, axis=0)
        points_max = np.max(aligned_points, axis=0)
        
        # Allow some margin
        margin = 0.1 * (scene_max - scene_min)
        
        if np.any(points_min < scene_min - margin) or np.any(points_max > scene_max + margin):
            logger.warning("Points exceed expected bounds significantly")
            return False
        
        # Check if cameras are at reasonable distances from points
        points_center = np.mean(aligned_points, axis=0)
        
        for cam_pos in camera_positions:
            distance = np.linalg.norm(cam_pos - points_center)
            if distance > 10 * np.linalg.norm(scene_max - scene_min):
                logger.warning(f"Camera too far from scene: distance={distance}")
                return False
        
        return True