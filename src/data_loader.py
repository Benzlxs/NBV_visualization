"""Data loader utilities for ABO dataset."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TransformsLoader:
    """Loader for transforms.json file containing camera poses and metadata."""
    
    def __init__(self, json_path: Path):
        """Initialize the transforms loader.
        
        Args:
            json_path: Path to transforms.json file
        """
        self.json_path = Path(json_path)
        self.data: Optional[Dict[str, Any]] = None
    
    def load(self) -> Dict[str, Any]:
        """Load and parse the transforms.json file.
        
        Returns:
            Dictionary containing the loaded data
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"Transforms file not found: {self.json_path}")
        
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded transforms from {self.json_path}")
        return self.data
    
    def get_camera_angle_x(self) -> Optional[float]:
        """Extract camera_angle_x from the first frame.
        
        Returns:
            Camera angle in radians, or None if not found
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        frames = self.data.get('frames', [])
        if frames and 'camera_angle_x' in frames[0]:
            return float(frames[0]['camera_angle_x'])
        
        logger.warning("camera_angle_x not found in data")
        return None
    
    def get_frames(self) -> List[Dict[str, Any]]:
        """Get all camera frames.
        
        Returns:
            List of frame dictionaries
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        return self.data.get('frames', [])
    
    def get_scale(self) -> float:
        """Get the scale factor.
        
        Returns:
            Scale factor, defaults to 1.0 if not found
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        return float(self.data.get('scale', 1.0))
    
    def get_offset(self) -> np.ndarray:
        """Get the offset vector.
        
        Returns:
            3D offset vector, defaults to [0, 0, 0] if not found
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        offset = self.data.get('offset', [0.0, 0.0, 0.0])
        return np.array(offset, dtype=np.float32)
    
    def get_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the axis-aligned bounding box.
        
        Returns:
            Tuple of (min_bounds, max_bounds) arrays
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        aabb = self.data.get('aabb', [[-1, -1, -1], [1, 1, 1]])
        min_bounds = np.array(aabb[0], dtype=np.float32)
        max_bounds = np.array(aabb[1], dtype=np.float32)
        
        return min_bounds, max_bounds
    
    def validate_data(self) -> bool:
        """Validate that the loaded data has all required fields.
        
        Returns:
            True if data is valid, False otherwise
        """
        if self.data is None:
            return False
        
        # Check for required fields
        required_fields = ['frames']
        for field in required_fields:
            if field not in self.data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check frames structure
        frames = self.data.get('frames', [])
        if not frames:
            logger.error("No frames found in data")
            return False
        
        # Check first frame has required fields
        first_frame = frames[0]
        frame_fields = ['file_path', 'transform_matrix']
        for field in frame_fields:
            if field not in first_frame:
                logger.error(f"Missing field in frame: {field}")
                return False
        
        # Check transform matrix structure
        transform = first_frame.get('transform_matrix', [])
        if len(transform) != 4 or any(len(row) != 4 for row in transform):
            logger.error("Invalid transform matrix structure")
            return False
        
        return True


class CameraPoseParser:
    """Parser for camera poses from transforms data."""
    
    def __init__(self, transforms_loader: TransformsLoader):
        """Initialize the camera pose parser.
        
        Args:
            transforms_loader: Loaded TransformsLoader instance
        """
        self.loader = transforms_loader
    
    def get_camera_poses(self) -> List[np.ndarray]:
        """Extract all camera poses as 4x4 numpy matrices.
        
        Returns:
            List of 4x4 transformation matrices
        """
        frames = self.loader.get_frames()
        poses = []
        
        for i, frame in enumerate(frames):
            try:
                matrix = self._parse_transform_matrix(frame)
                poses.append(matrix)
            except Exception as e:
                logger.warning(f"Failed to parse pose for frame {i}: {e}")
        
        return poses
    
    def _parse_transform_matrix(self, frame: Dict[str, Any]) -> np.ndarray:
        """Parse a single transform matrix from a frame.
        
        Args:
            frame: Frame dictionary containing transform_matrix
            
        Returns:
            4x4 numpy array
            
        Raises:
            ValueError: If matrix is invalid
        """
        if 'transform_matrix' not in frame:
            raise ValueError("No transform_matrix in frame")
        
        matrix_data = frame['transform_matrix']
        matrix = np.array(matrix_data, dtype=np.float32)
        
        # Validate dimensions
        if matrix.shape != (4, 4):
            raise ValueError(f"Invalid matrix shape: {matrix.shape}")
        
        # Validate it's a valid transformation matrix (bottom row should be [0,0,0,1])
        expected_bottom = np.array([0, 0, 0, 1])
        if not np.allclose(matrix[3], expected_bottom, atol=1e-6):
            logger.warning(f"Non-standard transformation matrix bottom row: {matrix[3]}")
        
        return matrix
    
    def get_camera_positions(self) -> List[np.ndarray]:
        """Extract camera positions (translations) from all poses.
        
        Returns:
            List of 3D position vectors
        """
        poses = self.get_camera_poses()
        positions = [pose[:3, 3] for pose in poses]
        return positions
    
    def get_camera_rotations(self) -> List[np.ndarray]:
        """Extract camera rotation matrices from all poses.
        
        Returns:
            List of 3x3 rotation matrices
        """
        poses = self.get_camera_poses()
        rotations = [pose[:3, :3] for pose in poses]
        return rotations
    
    def get_camera_info(self, frame_index: int) -> Dict[str, Any]:
        """Get comprehensive information for a specific camera.
        
        Args:
            frame_index: Index of the frame/camera
            
        Returns:
            Dictionary with camera information
        """
        frames = self.loader.get_frames()
        if frame_index < 0 or frame_index >= len(frames):
            raise ValueError(f"Invalid frame index: {frame_index}")
        
        frame = frames[frame_index]
        pose = self._parse_transform_matrix(frame)
        
        info = {
            'index': frame_index,
            'file_path': frame.get('file_path', ''),
            'pose_matrix': pose,
            'position': pose[:3, 3],
            'rotation_matrix': pose[:3, :3],
        }
        
        # Add camera angle if available
        if frame_index == 0:  # camera_angle_x is typically only in first frame
            camera_angle = self.loader.get_camera_angle_x()
            if camera_angle is not None:
                info['camera_angle_x'] = camera_angle
                info['fov_x'] = camera_angle  # FOV in radians
        
        return info
    
    def validate_all_poses(self) -> Tuple[bool, List[int]]:
        """Validate all camera poses in the dataset.
        
        Returns:
            Tuple of (all_valid, list_of_invalid_indices)
        """
        frames = self.loader.get_frames()
        invalid_indices = []
        
        for i, frame in enumerate(frames):
            try:
                self._parse_transform_matrix(frame)
            except Exception as e:
                logger.error(f"Invalid pose at index {i}: {e}")
                invalid_indices.append(i)
        
        all_valid = len(invalid_indices) == 0
        return all_valid, invalid_indices