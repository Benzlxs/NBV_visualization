"""Point cloud loader utilities for PLY files."""

import open3d as o3d
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PointCloudLoader:
    """Loader for PLY point cloud files."""
    
    def __init__(self, ply_path: Path):
        """Initialize the point cloud loader.
        
        Args:
            ply_path: Path to PLY file
        """
        self.ply_path = Path(ply_path)
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self._points: Optional[np.ndarray] = None
        self._colors: Optional[np.ndarray] = None
    
    def load(self) -> o3d.geometry.PointCloud:
        """Load the PLY file using Open3D.
        
        Returns:
            Open3D PointCloud object
            
        Raises:
            FileNotFoundError: If the PLY file doesn't exist
            RuntimeError: If loading fails
        """
        if not self.ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {self.ply_path}")
        
        try:
            self.pcd = o3d.io.read_point_cloud(str(self.ply_path))
            
            # Cache points and colors
            self._points = np.asarray(self.pcd.points, dtype=np.float32)
            
            if self.pcd.has_colors():
                self._colors = np.asarray(self.pcd.colors, dtype=np.float32)
            else:
                self._colors = None
            
            logger.info(f"Loaded point cloud from {self.ply_path}")
            logger.info(f"Points: {len(self._points)}, Has colors: {self.has_colors()}")
            
            return self.pcd
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PLY file: {e}")
    
    def get_points(self) -> np.ndarray:
        """Get point coordinates as numpy array.
        
        Returns:
            Nx3 array of point coordinates
        """
        if self._points is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
        
        return self._points.copy()
    
    def get_colors(self) -> Optional[np.ndarray]:
        """Get point colors as numpy array.
        
        Returns:
            Nx3 array of RGB colors in range [0, 1], or None if no colors
        """
        if self._points is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
        
        return self._colors.copy() if self._colors is not None else None
    
    def get_colors_uint8(self) -> Optional[np.ndarray]:
        """Get point colors as uint8 array.
        
        Returns:
            Nx3 array of RGB colors in range [0, 255], or None if no colors
        """
        colors = self.get_colors()
        if colors is not None:
            return (colors * 255).astype(np.uint8)
        return None
    
    def has_colors(self) -> bool:
        """Check if the point cloud has color information.
        
        Returns:
            True if colors are available
        """
        if self.pcd is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
        
        return self.pcd.has_colors()
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the point cloud.
        
        Returns:
            Tuple of (min_bounds, max_bounds) arrays
        """
        if self._points is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
        
        if len(self._points) == 0:
            return np.zeros(3), np.zeros(3)
        
        min_bounds = np.min(self._points, axis=0)
        max_bounds = np.max(self._points, axis=0)
        
        return min_bounds, max_bounds
    
    def get_center(self) -> np.ndarray:
        """Get the center of the point cloud.
        
        Returns:
            3D center point
        """
        min_bounds, max_bounds = self.get_bounds()
        return (min_bounds + max_bounds) / 2
    
    def get_extent(self) -> np.ndarray:
        """Get the extent (size) of the point cloud.
        
        Returns:
            3D extent vector
        """
        min_bounds, max_bounds = self.get_bounds()
        return max_bounds - min_bounds
    
    def get_point_count(self) -> int:
        """Get the number of points.
        
        Returns:
            Number of points
        """
        if self._points is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
        
        return len(self._points)
    
    def downsample(self, voxel_size: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Downsample the point cloud using voxel grid filter.
        
        Args:
            voxel_size: Size of voxels for downsampling
            
        Returns:
            Tuple of (downsampled_points, downsampled_colors)
        """
        if self.pcd is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
        
        downsampled = self.pcd.voxel_down_sample(voxel_size)
        
        points = np.asarray(downsampled.points, dtype=np.float32)
        colors = None
        if downsampled.has_colors():
            colors = np.asarray(downsampled.colors, dtype=np.float32)
        
        logger.info(f"Downsampled from {self.get_point_count()} to {len(points)} points")
        
        return points, colors
    
    def validate(self) -> bool:
        """Validate the loaded point cloud.
        
        Returns:
            True if valid, False otherwise
        """
        if self.pcd is None or self._points is None:
            logger.error("Point cloud not loaded")
            return False
        
        if len(self._points) == 0:
            logger.error("Point cloud has no points")
            return False
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(self._points)):
            logger.error("Point cloud contains NaN or infinite values")
            return False
        
        # Check color validity if present
        if self._colors is not None:
            if self._colors.shape != self._points.shape:
                logger.error("Color array shape doesn't match points")
                return False
            
            if np.any((self._colors < 0) | (self._colors > 1)):
                logger.warning("Color values outside [0, 1] range")
        
        return True
    
    def get_statistics(self) -> dict:
        """Get statistics about the point cloud.
        
        Returns:
            Dictionary with statistics
        """
        if self._points is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
        
        min_bounds, max_bounds = self.get_bounds()
        
        stats = {
            'num_points': self.get_point_count(),
            'has_colors': self.has_colors(),
            'min_bounds': min_bounds.tolist(),
            'max_bounds': max_bounds.tolist(),
            'center': self.get_center().tolist(),
            'extent': self.get_extent().tolist(),
        }
        
        # Add color statistics if available
        if self._colors is not None:
            stats['color_min'] = np.min(self._colors, axis=0).tolist()
            stats['color_max'] = np.max(self._colors, axis=0).tolist()
            stats['color_mean'] = np.mean(self._colors, axis=0).tolist()
        
        return stats