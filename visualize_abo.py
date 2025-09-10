#!/usr/bin/env python3
"""Main script to visualize ABO dataset."""

import sys
from pathlib import Path
from src.visualizer import ABOVisualizer
import logging


def main():
    """Test the visualizer with the actual ABO dataset."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
    
    # ABO dataset path
    abo_data_path = Path("dataset/ABO/renders/fffc46603b43bd6b282701f6877ffe74e4cee40ed2b92d68872bcf873efca7dc")
    
    if not abo_data_path.exists():
        print(f"ABO dataset not found at {abo_data_path}")
        print("Please ensure the dataset is available.")
        return 1
    
    print(f"Loading ABO dataset from {abo_data_path}")
    
    # Create visualizer
    visualizer = ABOVisualizer(
        data_path=abo_data_path,
        host="0.0.0.0",
        port=8080
    )
    
    try:
        # Choose visualization mode
        import argparse
        parser = argparse.ArgumentParser(description="Visualize ABO dataset")
        parser.add_argument("--mode", choices=["point_cloud", "cameras", "both"], 
                          default="both", help="Visualization mode")
        parser.add_argument("--downsample", type=float, default=0.01,
                          help="Voxel size for point cloud downsampling")
        parser.add_argument("--frustum-scale", type=float, default=0.15,
                          help="Scale for camera frustums")
        parser.add_argument("--show-camera-frames", action="store_true",
                          help="Show coordinate frames at camera positions")
        parser.add_argument("--with-images", action="store_true",
                          help="Load and display camera images on frustums")
        parser.add_argument("--max-images", type=int, default=None,
                          help="Maximum number of images to load (for performance)")
        
        # Parse args from remaining argv (skip script name)
        import sys as _sys
        args = parser.parse_args(_sys.argv[1:] if len(_sys.argv) > 1 else [])
        
        # Start visualization based on mode
        if args.mode == "point_cloud":
            print("Starting point cloud only visualization...")
            visualizer.visualize_point_cloud_only(downsample_voxel_size=args.downsample)
        elif args.mode == "cameras":
            if args.with_images:
                print("Starting camera visualization with images...")
                visualizer.visualize_with_images(
                    downsample_voxel_size=args.downsample,
                    frustum_scale=args.frustum_scale,
                    show_camera_frames=args.show_camera_frames,
                    max_images=args.max_images
                )
            else:
                print("Starting camera visualization (with point cloud)...")
                visualizer.visualize_with_cameras(
                    downsample_voxel_size=args.downsample,
                    frustum_scale=args.frustum_scale,
                    show_camera_frames=args.show_camera_frames
                )
        else:  # both
            if args.with_images:
                print("Starting full visualization with point cloud, cameras, and images...")
                visualizer.visualize_with_images(
                    downsample_voxel_size=args.downsample,
                    frustum_scale=args.frustum_scale,
                    show_camera_frames=args.show_camera_frames,
                    max_images=args.max_images
                )
            else:
                print("Starting full visualization with point cloud and cameras...")
                visualizer.visualize_with_cameras(
                    downsample_voxel_size=args.downsample,
                    frustum_scale=args.frustum_scale,
                    show_camera_frames=args.show_camera_frames
                )
        
        # Print info
        info = visualizer.get_info()
        print("\n=== Visualization Info ===")
        print(f"Data path: {info['data_path']}")
        print(f"Number of cameras: {info['num_cameras']}")
        print(f"Camera FOV (angle_x): {info.get('camera_angle_x', 'N/A')} radians")
        print(f"Scale factor: {info['scale']}")
        print(f"Coordinate offset: {info['offset']}")
        
        # Print image loading info if applicable
        if args.with_images and hasattr(visualizer, 'image_loader') and visualizer.image_loader:
            img_stats = visualizer.image_loader.get_statistics()
            print(f"\nImage Loading:")
            print(f"  - Loaded: {img_stats.get('loaded_images', 0)}/{img_stats.get('total_images', 0)} images")
            if 'image_dimensions' in img_stats:
                dims = img_stats['image_dimensions']
                print(f"  - Image size: {dims.get('common_size', 'N/A')} (most common)")
        elif args.with_images:
            print(f"\nImage Loading: Requested but no images were loaded")
        
        pc_info = info.get('point_cloud', {})
        print(f"\nPoint cloud:")
        print(f"  - Points: {pc_info.get('num_points', 0)}")
        print(f"  - Has colors: {pc_info.get('has_colors', False)}")
        print(f"  - Bounds: min={pc_info.get('min_bounds', [])}, max={pc_info.get('max_bounds', [])}")
        
        scene_bounds = info.get('scene_bounds', {})
        print(f"\nScene bounds after transformation:")
        print(f"  - Min: {scene_bounds.get('min', [])}")
        print(f"  - Max: {scene_bounds.get('max', [])}")
        
        print(f"\n=== Access the visualization at http://localhost:8080 ===")
        if args.with_images:
            print("📷 Images are loaded and displayed on camera frustums")
            print("💡 Tip: Navigate around to see different camera viewpoints with images")
        else:
            print("📷 To see images on camera frustums, use --with-images flag")
        print("Press Ctrl+C to stop the server")
        
        visualizer.keep_alive()
        
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        visualizer.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())