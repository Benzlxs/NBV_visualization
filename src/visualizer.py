"""Main visualization module for ABO dataset."""

import numpy as np
import viser
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .viser_server import BasicViserServer
from .data_loader import TransformsLoader, CameraPoseParser
from .point_cloud_loader import PointCloudLoader
from .coordinate_transform import AlignedCoordinateSystem
from .camera_frustum import CameraFrustumRenderer
from .image_loader import ImageLoader

logger = logging.getLogger(__name__)


class ABOVisualizer:
    """Main visualizer for ABO dataset with point clouds and cameras."""
    
    def __init__(self, data_path: Path, host: str = "0.0.0.0", port: int = 8080):
        """Initialize the visualizer.
        
        Args:
            data_path: Path to dataset directory containing transforms.json and mesh.ply
            host: Host address for viser server
            port: Port number for viser server
        """
        self.data_path = Path(data_path)
        self.server_wrapper = BasicViserServer(host=host, port=port)
        self.server: Optional[viser.ViserServer] = None
        
        # Data components
        self.transforms_loader: Optional[TransformsLoader] = None
        self.camera_parser: Optional[CameraPoseParser] = None
        self.point_cloud_loader: Optional[PointCloudLoader] = None
        self.coordinate_system: Optional[AlignedCoordinateSystem] = None
        self.image_loader: Optional[ImageLoader] = None
        
        # Visualization components
        self.point_cloud_handle = None
        self.camera_frustum_renderer: Optional[CameraFrustumRenderer] = None
        
    def load_data(self):
        """Load all data from the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load transforms
        transforms_path = self.data_path / "transforms.json"
        self.transforms_loader = TransformsLoader(transforms_path)
        transforms_data = self.transforms_loader.load()
        
        if not self.transforms_loader.validate_data():
            raise ValueError("Invalid transforms data")
        
        # Initialize camera parser
        self.camera_parser = CameraPoseParser(self.transforms_loader)
        
        # Initialize coordinate system
        self.coordinate_system = AlignedCoordinateSystem(transforms_data)
        
        # Load point cloud
        ply_path = self.data_path / "mesh.ply"
        self.point_cloud_loader = PointCloudLoader(ply_path)
        self.point_cloud_loader.load()
        
        if not self.point_cloud_loader.validate():
            raise ValueError("Invalid point cloud data")
        
        # Initialize image loader 
        self.image_loader = ImageLoader(self.data_path)
        
        logger.info("Data loading complete")
        
    def add_point_cloud(self, downsample_voxel_size: Optional[float] = None,
                       point_size: float = 0.01):
        """Add point cloud to the visualization.
        
        Args:
            downsample_voxel_size: Optional voxel size for downsampling
            point_size: Size of points in visualization
        """
        if self.server is None:
            raise RuntimeError("Server not started. Call start() first.")
        
        if self.point_cloud_loader is None or self.coordinate_system is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        # Get points and colors
        if downsample_voxel_size is not None and downsample_voxel_size > 0:
            points, colors = self.point_cloud_loader.downsample(downsample_voxel_size)
            logger.info(f"Using downsampled point cloud with {len(points)} points")
        else:
            points = self.point_cloud_loader.get_points()
            colors = self.point_cloud_loader.get_colors()
            logger.info(f"Using full point cloud with {len(points)} points")
        
        # Apply coordinate transformation
        aligned_points = self.coordinate_system.align_point_cloud(points)
        
        # Convert colors to uint8 if available, otherwise use default gray
        if colors is not None:
            colors_uint8 = (colors * 255).astype(np.uint8)
        else:
            # Use default gray color for all points
            colors_uint8 = np.full((len(aligned_points), 3), 128, dtype=np.uint8)
        
        # Add to viser scene
        self.point_cloud_handle = self.server.scene.add_point_cloud(
            name="/point_cloud",
            points=aligned_points,
            colors=colors_uint8,
            point_size=point_size,
            point_shape="circle"
        )
        
        logger.info("Point cloud added to scene")
        
    def add_world_frame(self, axes_length: float = 0.5):
        """Add world coordinate frame.
        
        Args:
            axes_length: Length of coordinate axes
        """
        if self.server is None:
            raise RuntimeError("Server not started. Call start() first.")
        
        self.server.scene.add_frame(
            name="/world",
            axes_length=axes_length
        )
    
    def load_camera_images(self,
                          max_images: Optional[int] = None,
                          max_workers: int = 4,
                          show_progress: bool = True) -> Dict[str, Any]:
        """Load camera images from disk.
        
        Args:
            max_images: Maximum number of images to load (None for all)
            max_workers: Number of parallel threads for loading
            show_progress: Whether to show loading progress
            
        Returns:
            Dictionary with loaded images
        """
        if self.image_loader is None or self.transforms_loader is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        # Get list of image filenames from transforms data
        frames = self.transforms_loader.get_frames()
        filenames = [frame['file_path'] for frame in frames]
        
        # Limit number of images if requested
        if max_images is not None and max_images > 0:
            filenames = filenames[:max_images]
            logger.info(f"Loading {len(filenames)}/{len(frames)} images")
        else:
            logger.info(f"Loading all {len(filenames)} images")
        
        # Load images in batch
        images = self.image_loader.load_images_batch(
            filenames=filenames,
            max_workers=max_workers,
            show_progress=show_progress
        )
        
        # Get loading statistics
        stats = self.image_loader.get_statistics()
        logger.info(f"Image loading complete: {stats}")
        
        return images
    
    def add_camera_frustums(self, 
                          frustum_scale: float = 0.15,
                          show_frames: bool = True,
                          frame_axes_length: float = 0.05,
                          use_gradient_colors: bool = True,
                          load_images: bool = False,
                          max_images: Optional[int] = None):
        """Add camera frustums to the visualization.
        
        Args:
            frustum_scale: Scale factor for frustum size (distance to image plane)
            show_frames: Whether to show coordinate frames at camera positions
            frame_axes_length: Length of coordinate frame axes
            use_gradient_colors: Whether to use gradient coloring for cameras
            load_images: Whether to load and display camera images on frustums
            max_images: Maximum number of images to load (None for all)
        """
        if self.server is None:
            raise RuntimeError("Server not started. Call start() first.")
        
        if self.camera_parser is None or self.coordinate_system is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        # Initialize frustum renderer if not already done
        if self.camera_frustum_renderer is None:
            self.camera_frustum_renderer = CameraFrustumRenderer(self.server)
        
        # Get camera poses and apply coordinate transformation
        camera_poses = self.camera_parser.get_camera_poses()
        aligned_poses = self.coordinate_system.align_camera_poses(camera_poses)
        
        # Get camera FOV (field of view)
        # FOV is typically only stored in the first frame
        fov_x = self.transforms_loader.get_camera_angle_x()
        if fov_x is None:
            # Use default FOV if not specified (~60 degrees)
            fov_x = np.radians(60)
            logger.warning("No camera_angle_x found, using default FOV of 60 degrees")
        
        # Load images if requested
        camera_images = None
        if load_images:
            logger.info("Loading camera images for frustum display")
            loaded_images = self.load_camera_images(
                max_images=max_images,
                max_workers=4,
                show_progress=True
            )
            
            # Convert image dictionary to list ordered by frame index
            frames = self.transforms_loader.get_frames()
            camera_images = []
            for frame in frames[:max_images if max_images else len(frames)]:
                filename = frame['file_path']
                image = loaded_images.get(filename)
                camera_images.append(image)  # Can be None if not loaded
            
            logger.info(f"Prepared {len([img for img in camera_images if img is not None])} images for frustum display")
        
        # Get aspect ratio from first loaded image if available
        aspect_ratio = 16 / 9  # Default
        if camera_images and camera_images[0] is not None:
            first_image = camera_images[0]
            aspect_ratio = first_image.width / first_image.height
            logger.info(f"Using aspect ratio {aspect_ratio:.2f} from loaded images")
        
        # Generate colors for cameras if using gradient
        colors = None  # Will use default gradient if None
        
        # Add all camera frustums with images
        logger.info(f"Adding {len(aligned_poses)} camera frustums with images={load_images}")
        handles = self.camera_frustum_renderer.add_camera_frustums_batch(
            pose_matrices=aligned_poses,
            fov_x=fov_x,
            aspect_ratio=aspect_ratio,
            scale=frustum_scale,
            colors=colors,  # None uses gradient coloring
            base_name="/camera",
            show_frames=show_frames,
            frame_axes_length=frame_axes_length,
            images=camera_images  # Pass loaded images
        )
        
        # Log statistics
        info = self.camera_frustum_renderer.get_frustum_info()
        logger.info(f"Camera frustum statistics: {info}")
        
    def start(self):
        """Start the visualization server."""
        self.server = self.server_wrapper.start()
        return self.server
    
    def visualize_point_cloud_only(self, downsample_voxel_size: Optional[float] = None):
        """Simple visualization showing only the point cloud.
        
        Args:
            downsample_voxel_size: Optional voxel size for downsampling
        """
        # Load data
        self.load_data()
        
        # Start server
        self.start()
        
        # Add world frame
        self.add_world_frame()
        
        # Add point cloud
        self.add_point_cloud(downsample_voxel_size=downsample_voxel_size)
        
        # Get statistics
        stats = self.point_cloud_loader.get_statistics()
        logger.info(f"Point cloud statistics: {stats}")
        
        # Set initial camera position based on scene bounds
        scene_min, scene_max = self.coordinate_system.get_scene_bounds()
        scene_center = (scene_min + scene_max) / 2
        scene_size = np.linalg.norm(scene_max - scene_min)
        
        # Position camera to view the scene
        camera_distance = scene_size * 2
        self.server.scene.set_up_direction("+z")
        
        logger.info(f"Scene bounds: min={scene_min}, max={scene_max}")
        logger.info(f"Scene center: {scene_center}, size: {scene_size}")
    
    def visualize_with_cameras(self, 
                              downsample_voxel_size: Optional[float] = None,
                              frustum_scale: float = 0.15,
                              show_camera_frames: bool = False):
        """Visualization showing both point cloud and camera frustums.
        
        Args:
            downsample_voxel_size: Optional voxel size for point cloud downsampling
            frustum_scale: Scale factor for camera frustum size
            show_camera_frames: Whether to show coordinate frames at cameras
        """
        # Load data
        self.load_data()
        
        # Start server  
        self.start()
        
        # Add world frame
        self.add_world_frame()
        
        # Add point cloud
        self.add_point_cloud(downsample_voxel_size=downsample_voxel_size)
        
        # Add camera frustums without images
        self.add_camera_frustums(
            frustum_scale=frustum_scale,
            show_frames=show_camera_frames,
            frame_axes_length=0.03,  # Smaller frames for cameras
            load_images=False  # Don't load images in basic camera visualization
        )
        
        # Get and log statistics
        pc_stats = self.point_cloud_loader.get_statistics()
        logger.info(f"Point cloud statistics: {pc_stats}")
        
        camera_info = self.camera_frustum_renderer.get_frustum_info()
        logger.info(f"Camera statistics: {camera_info}")
        
        # Set initial view
        scene_min, scene_max = self.coordinate_system.get_scene_bounds()
        scene_center = (scene_min + scene_max) / 2
        scene_size = np.linalg.norm(scene_max - scene_min)
        
        self.server.scene.set_up_direction("+z")
        
        logger.info(f"Scene bounds: min={scene_min}, max={scene_max}")
        logger.info(f"Scene center: {scene_center}, size: {scene_size}")
        
    def visualize_with_images(self, 
                             downsample_voxel_size: Optional[float] = None,
                             frustum_scale: float = 0.15,
                             show_camera_frames: bool = False,
                             max_images: Optional[int] = None):
        """Complete visualization with point cloud, camera frustums, and images.
        
        Args:
            downsample_voxel_size: Optional voxel size for point cloud downsampling
            frustum_scale: Scale factor for camera frustum size  
            show_camera_frames: Whether to show coordinate frames at cameras
            max_images: Maximum number of images to load (None for all)
        """
        # Load data
        self.load_data()
        
        # Start server
        self.start()
        
        # Add world frame
        self.add_world_frame()
        
        # Add point cloud
        self.add_point_cloud(downsample_voxel_size=downsample_voxel_size)
        
        # Add camera frustums with images
        self.add_camera_frustums(
            frustum_scale=frustum_scale,
            show_frames=show_camera_frames,
            frame_axes_length=0.03,
            load_images=True,  # Enable image loading
            max_images=max_images
        )
        
        # Get and log statistics
        pc_stats = self.point_cloud_loader.get_statistics()
        logger.info(f"Point cloud statistics: {pc_stats}")
        
        camera_info = self.camera_frustum_renderer.get_frustum_info()
        logger.info(f"Camera statistics: {camera_info}")
        
        # Get image loading statistics
        if self.image_loader:
            img_stats = self.image_loader.get_statistics()
            logger.info(f"Image loading statistics: {img_stats}")
        
        # Set initial view
        scene_min, scene_max = self.coordinate_system.get_scene_bounds()
        scene_center = (scene_min + scene_max) / 2
        scene_size = np.linalg.norm(scene_max - scene_min)
        
        self.server.scene.set_up_direction("+z")
        
        logger.info(f"Scene bounds: min={scene_min}, max={scene_max}")
        logger.info(f"Scene center: {scene_center}, size: {scene_size}")
        
    def keep_alive(self):
        """Keep the server running."""
        self.server_wrapper.keep_alive()
    
    def stop(self):
        """Stop the server."""
        self.server_wrapper.stop()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        info = {
            "data_path": str(self.data_path),
            "server_running": self.server is not None
        }
        
        if self.transforms_loader is not None:
            frames = self.transforms_loader.get_frames()
            info["num_cameras"] = len(frames)
            info["camera_angle_x"] = self.transforms_loader.get_camera_angle_x()
            info["scale"] = self.transforms_loader.get_scale()
            info["offset"] = self.transforms_loader.get_offset().tolist()
        
        if self.point_cloud_loader is not None:
            info["point_cloud"] = self.point_cloud_loader.get_statistics()
        
        if self.coordinate_system is not None:
            scene_min, scene_max = self.coordinate_system.get_scene_bounds()
            info["scene_bounds"] = {
                "min": scene_min.tolist(),
                "max": scene_max.tolist()
            }
        
        return info


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ABO dataset")
    parser.add_argument("data_path", type=str, help="Path to dataset directory")
    parser.add_argument("--downsample", type=float, default=None,
                       help="Voxel size for downsampling (optional)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host address")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port number")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create visualizer
    visualizer = ABOVisualizer(
        data_path=args.data_path,
        host=args.host,
        port=args.port
    )
    
    try:
        # Visualize point cloud
        visualizer.visualize_point_cloud_only(downsample_voxel_size=args.downsample)
        
        # Print info
        info = visualizer.get_info()
        print("\nVisualization Info:")
        print(f"- Cameras: {info.get('num_cameras', 0)}")
        print(f"- Points: {info.get('point_cloud', {}).get('num_points', 0)}")
        print(f"- Scene bounds: {info.get('scene_bounds', {})}")
        
        print("\nPress Ctrl+C to stop the server")
        visualizer.keep_alive()
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        visualizer.stop()


if __name__ == "__main__":
    main()