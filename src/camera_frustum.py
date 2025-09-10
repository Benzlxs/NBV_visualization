"""Camera frustum visualization utilities for viser.

This module provides functionality to create and display camera frustums in 3D space,
showing the field of view and orientation of cameras in the scene.
"""

import numpy as np
import viser
import viser.transforms as vtf
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CameraFrustumRenderer:
    """Renders camera frustums in viser to visualize camera positions and orientations."""
    
    def __init__(self, viser_server: viser.ViserServer, apply_coordinate_fix: bool = True):
        """Initialize the camera frustum renderer.
        
        Args:
            viser_server: The viser server instance to add frustums to
            apply_coordinate_fix: Whether to apply coordinate system conversion (180° Y-axis rotation)
                                 Set to True for datasets where cameras face outward from scene
        """
        self.server = viser_server
        self.frustum_handles: List[Any] = []  # Store handles for all frustums
        self.frame_handles: List[Any] = []    # Store handles for coordinate frames
        self.apply_coordinate_fix = apply_coordinate_fix
        
    def add_camera_frustum(
        self,
        pose_matrix: np.ndarray,
        fov_x: float,
        aspect_ratio: float = 16/9,
        scale: float = 0.15,
        color: Tuple[int, int, int] = (255, 0, 0),
        name: Optional[str] = None,
        show_frame: bool = True,
        frame_axes_length: float = 0.05,
        image: Optional[Any] = None
    ) -> Any:
        """Add a single camera frustum to the scene.
        
        Args:
            pose_matrix: 4x4 transformation matrix representing camera pose
            fov_x: Horizontal field of view in radians
            aspect_ratio: Aspect ratio (width/height) of the camera
            scale: Scale factor for frustum size (distance from camera center to image plane)
            color: RGB color tuple for the frustum (0-255 per channel)
            name: Optional name for the frustum (auto-generated if None)
            show_frame: Whether to show coordinate frame axes at camera position
            frame_axes_length: Length of coordinate frame axes
            image: Optional PIL Image to display on the frustum
            
        Returns:
            Handle to the created frustum object
        """
        # Generate unique name if not provided
        if name is None:
            name = f"/camera_frustum_{len(self.frustum_handles)}"
        
        try:
            # Convert pose matrix to SE3 for viser
            # The pose matrix represents the camera-to-world transformation
            if self.apply_coordinate_fix:
                # Apply coordinate system conversion for datasets where cameras face outward
                corrected_pose = self._convert_camera_coordinate_system(pose_matrix)
                se3 = vtf.SE3.from_matrix(corrected_pose)
                logger.debug(f"Applied coordinate system fix for camera '{name}'")
            else:
                # Use pose matrix directly (for ABO dataset and similar)
                se3 = vtf.SE3.from_matrix(pose_matrix)
            
            # Extract rotation (as quaternion) and translation
            rotation_wxyz = se3.rotation().wxyz
            position = se3.translation()
            
            # Log camera information for debugging
            logger.debug(f"Adding frustum '{name}' at position {position}")
            
            # Prepare frustum parameters with proper data types
            frustum_params = {
                'name': str(name),
                'fov': float(fov_x),  # Field of view in radians
                'aspect': float(aspect_ratio),
                'scale': float(scale),  # Size of the frustum visualization
                'wxyz': rotation_wxyz.astype(np.float64),  # Rotation as quaternion (ensure float64)
                'position': position.astype(np.float64),  # 3D position (ensure float64)
                'color': tuple(int(c) for c in color)  # RGB color as tuple of ints
            }
            
            # Always add image parameter (can be None)
            # Ensure image is in correct format for viser
            if image is not None:
                # Convert PIL Image to numpy array if needed
                if hasattr(image, 'mode'):
                    # PIL Image - convert to RGB numpy array
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image_array = np.array(image, dtype=np.uint8)
                else:
                    # Already numpy array
                    image_array = image
                
                # Apply image transformation if coordinate fix is enabled
                if self.apply_coordinate_fix:
                    # When we apply 180° Y-axis rotation to camera, the image needs to be:
                    # 1. Rotated 180° (flip both horizontally and vertically)
                    # This ensures the image orientation matches the corrected camera orientation
                    image_array = self._transform_image_for_coordinate_fix(image_array)
                    logger.debug(f"Applied image transformation for coordinate fix: {name}")
                
                frustum_params['image'] = image_array
            else:
                frustum_params['image'] = image  # None
            
            # Create the camera frustum
            frustum_handle = self.server.scene.add_camera_frustum(**frustum_params)
            
            # Store the handle for later reference
            self.frustum_handles.append(frustum_handle)
            
            # Optionally add coordinate frame at camera position
            if show_frame:
                frame_name = f"{name}_frame"
                frame_handle = self.server.scene.add_frame(
                    name=frame_name,
                    wxyz=rotation_wxyz,
                    position=position,
                    axes_length=frame_axes_length
                )
                self.frame_handles.append(frame_handle)
            
            return frustum_handle
            
        except Exception as e:
            # Provide detailed error information for dtype issues
            error_msg = f"Failed to add camera frustum '{name}': {e}"
            
            # Add debugging information for dtype errors
            if "dtype" in str(e).lower():
                logger.error(f"{error_msg}")
                logger.error(f"Frustum parameters types:")
                for key, value in frustum_params.items():
                    if value is not None:
                        logger.error(f"  {key}: {type(value)} = {value}")
                    else:
                        logger.error(f"  {key}: None")
            else:
                logger.error(error_msg)
            
            raise
    
    def add_camera_frustums_batch(
        self,
        pose_matrices: List[np.ndarray],
        fov_x: float,
        aspect_ratio: float = 16/9,
        scale: float = 0.15,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        base_name: str = "/camera",
        show_frames: bool = True,
        frame_axes_length: float = 0.05,
        images: Optional[List[Any]] = None
    ) -> List[Any]:
        """Add multiple camera frustums to the scene in a batch.
        
        Args:
            pose_matrices: List of 4x4 transformation matrices
            fov_x: Horizontal field of view in radians (same for all cameras)
            aspect_ratio: Aspect ratio (width/height) of the cameras
            scale: Scale factor for frustum size
            colors: Optional list of RGB colors, one per camera. If None, uses gradient coloring
            base_name: Base name for frustums (will append index)
            show_frames: Whether to show coordinate frames
            frame_axes_length: Length of coordinate frame axes
            images: Optional list of PIL Images, one per camera
            
        Returns:
            List of handles to created frustum objects
        """
        num_cameras = len(pose_matrices)
        logger.info(f"Adding {num_cameras} camera frustums to scene")
        
        # Generate colors if not provided
        if colors is None:
            colors = self._generate_gradient_colors(num_cameras)
        elif len(colors) != num_cameras:
            logger.warning(f"Color count ({len(colors)}) doesn't match camera count ({num_cameras})")
            colors = self._generate_gradient_colors(num_cameras)
        
        # Add each frustum
        handles = []
        for i, pose_matrix in enumerate(pose_matrices):
            name = f"{base_name}_{i:03d}"  # e.g., /camera_000, /camera_001, etc.
            
            try:
                # Get image for this camera if provided
                camera_image = None
                if images is not None and i < len(images):
                    camera_image = images[i]
                
                handle = self.add_camera_frustum(
                    pose_matrix=pose_matrix,
                    fov_x=fov_x,
                    aspect_ratio=aspect_ratio,
                    scale=scale,
                    color=colors[i],
                    name=name,
                    show_frame=show_frames,
                    frame_axes_length=frame_axes_length,
                    image=camera_image  # Add image parameter
                )
                handles.append(handle)
                
            except Exception as e:
                logger.error(f"Failed to add frustum {i}: {e}")
                # Continue with other frustums even if one fails
                handles.append(None)
        
        # Log statistics
        successful = sum(1 for h in handles if h is not None)
        logger.info(f"Successfully added {successful}/{num_cameras} camera frustums")
        
        return handles
    
    def _generate_gradient_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate a gradient of colors for visualizing multiple cameras.
        
        Creates a color gradient from red to blue through the color wheel,
        which helps distinguish between different camera positions.
        
        Args:
            num_colors: Number of colors to generate
            
        Returns:
            List of RGB color tuples (0-255 per channel)
        """
        colors = []
        
        # Special case for single camera
        if num_colors == 1:
            return [(255, 0, 0)]  # Red
        
        # Generate colors using HSV color space for better distribution
        for i in range(num_colors):
            # Hue varies from 0 (red) to 240 (blue) degrees
            hue = (i / (num_colors - 1)) * 240
            
            # Convert HSV to RGB (simplified for hue variation only)
            # Using full saturation and value
            h_prime = hue / 60
            c = 1.0  # Chroma (saturation * value)
            x = c * (1 - abs((h_prime % 2) - 1))
            
            if h_prime < 1:
                r, g, b = c, x, 0
            elif h_prime < 2:
                r, g, b = x, c, 0
            elif h_prime < 3:
                r, g, b = 0, c, x
            elif h_prime < 4:
                r, g, b = 0, x, c
            else:
                r, g, b = x, 0, c
            
            # Convert to 0-255 range
            colors.append((
                int(r * 255),
                int(g * 255),
                int(b * 255)
            ))
        
        return colors
    
    def _convert_camera_coordinate_system(self, pose_matrix: np.ndarray) -> np.ndarray:
        """Convert camera pose from dataset coordinate system to viser coordinate system.
        
        This fixes the common issue where dataset cameras face outward instead of inward.
        The ABO dataset likely uses a coordinate system where cameras point away from the
        scene center. We apply a 180-degree rotation around the Y axis to flip the camera
        direction so cameras point toward the scene center.
        
        Args:
            pose_matrix: Original 4x4 camera pose matrix
            
        Returns:
            Corrected 4x4 camera pose matrix with proper orientation
        """
        # Create 180-degree rotation around Y axis to flip camera direction
        # This matrix flips X and Z axes while keeping Y unchanged
        flip_matrix = np.array([
            [-1,  0,  0,  0],  # Flip X axis
            [ 0,  1,  0,  0],  # Keep Y axis unchanged
            [ 0,  0, -1,  0],  # Flip Z axis
            [ 0,  0,  0,  1]   # Homogeneous coordinate
        ], dtype=np.float32)
        
        # Apply the coordinate system flip transformation
        # This makes cameras point toward the scene center instead of away from it
        corrected_pose = pose_matrix @ flip_matrix
        
        logger.debug(f"Applied coordinate system conversion to camera pose")
        return corrected_pose
    
    def _transform_image_for_coordinate_fix(self, image_array: np.ndarray) -> np.ndarray:
        """Transform image to match coordinate system fix.
        
        When we apply a 180° Y-axis rotation to the camera frustum to fix orientation,
        the image also needs to be transformed to maintain the correct visual correspondence.
        
        Args:
            image_array: Input image as numpy array (H, W, 3)
            
        Returns:
            Transformed image array with correct orientation
        """
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            logger.warning(f"Unexpected image shape for transformation: {image_array.shape}")
            return image_array
        
        # For a 180° Y-axis rotation of the camera frustum, the image needs to be rotated 180°
        # This is equivalent to flipping both horizontally and vertically
        # 
        # The reasoning:
        # - 180° Y-axis rotation flips the camera's X and Z axes
        # - This means the image's horizontal (X) axis is flipped
        # - And the image's depth perception (Z) is flipped, which affects vertical orientation
        # - Result: 180° image rotation (flip both axes)
        
        transformed_image = np.rot90(image_array, k=2)  # Rotate 180° (k=2 means 2*90°)
        
        logger.debug(f"Transformed image from {image_array.shape} to {transformed_image.shape}")
        return transformed_image
    
    def update_frustum_visibility(self, visible: bool):
        """Update visibility of all frustums.
        
        Args:
            visible: Whether frustums should be visible
        """
        for handle in self.frustum_handles:
            if handle is not None:
                handle.visible = visible
        
        # Also update frame visibility
        for handle in self.frame_handles:
            if handle is not None:
                handle.visible = visible
    
    def update_frustum_scale(self, scale: float):
        """Update the scale of all frustums.
        
        Note: This requires recreating frustums as viser doesn't support
        dynamic scale updates for camera frustums.
        
        Args:
            scale: New scale factor
        """
        logger.warning("Dynamic frustum scale update not implemented. "
                      "Frustums need to be recreated with new scale.")
    
    def get_frustum_info(self) -> Dict[str, Any]:
        """Get information about rendered frustums.
        
        Returns:
            Dictionary containing frustum statistics
        """
        return {
            'num_frustums': len(self.frustum_handles),
            'num_frames': len(self.frame_handles),
            'num_valid_frustums': sum(1 for h in self.frustum_handles if h is not None),
            'num_valid_frames': sum(1 for h in self.frame_handles if h is not None)
        }
    
    def clear_all(self):
        """Remove all frustums and frames from the scene."""
        # Remove frustums
        for handle in self.frustum_handles:
            if handle is not None:
                try:
                    handle.remove()
                except Exception as e:
                    logger.warning(f"Failed to remove frustum: {e}")
        
        # Remove frames
        for handle in self.frame_handles:
            if handle is not None:
                try:
                    handle.remove()
                except Exception as e:
                    logger.warning(f"Failed to remove frame: {e}")
        
        # Clear handle lists
        self.frustum_handles.clear()
        self.frame_handles.clear()
        
        logger.info("Cleared all camera frustums and frames")


def compute_aspect_ratio_from_image_shape(width: int, height: int) -> float:
    """Compute aspect ratio from image dimensions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Aspect ratio (width/height)
    """
    if height == 0:
        logger.warning("Image height is 0, returning default aspect ratio")
        return 16/9
    
    return width / height


def fov_to_focal_length(fov: float, sensor_size: float) -> float:
    """Convert field of view to focal length.
    
    Args:
        fov: Field of view in radians
        sensor_size: Sensor size (width or height) in arbitrary units
        
    Returns:
        Focal length in same units as sensor_size
    """
    return sensor_size / (2 * np.tan(fov / 2))


def focal_length_to_fov(focal_length: float, sensor_size: float) -> float:
    """Convert focal length to field of view.
    
    Args:
        focal_length: Focal length in arbitrary units
        sensor_size: Sensor size (width or height) in same units
        
    Returns:
        Field of view in radians
    """
    return 2 * np.arctan(sensor_size / (2 * focal_length))