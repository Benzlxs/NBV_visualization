#!/usr/bin/env python3
"""Image capture and export system for rendered GLB models.

This module implements Step 7: Image Capture & Export with:
- Save rendered images in multiple formats
- Configurable resolution and quality
- Batch capture at multiple poses  
- Integration with existing file system
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil

from .blender_pipeline import (
    BlenderRenderingPipeline, RenderConfig, RenderQuality, RenderJob, RenderResult
)
from .trellis_camera import CameraPose, TRELLISCameraSystem, generate_standard_views
from .glb_loader import GLBLoader

logger = logging.getLogger(__name__)


@dataclass
class CaptureSession:
    """Information about an image capture session."""
    session_id: str
    model_path: Path
    output_dir: Path
    config: RenderConfig
    poses: List[CameraPose]
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[RenderResult] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.metadata is None:
            self.metadata = {}


class ImageCaptureExporter:
    """High-level image capture and export system."""
    
    def __init__(self, 
                 output_base_dir: Union[str, Path] = "renders",
                 blender_executable: Optional[str] = None):
        """Initialize image capture system.
        
        Args:
            output_base_dir: Base directory for all captures
            blender_executable: Path to Blender executable
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize rendering pipeline
        self.pipeline = BlenderRenderingPipeline(blender_executable)
        
        # Session tracking
        self.current_session: Optional[CaptureSession] = None
        self.completed_sessions: List[CaptureSession] = []
        
        logger.info(f"Image capture system initialized, output dir: {self.output_base_dir}")
    
    def create_capture_config(self,
                            format: str = "PNG",
                            resolution: Tuple[int, int] = (512, 512),
                            quality: RenderQuality = RenderQuality.STANDARD,
                            **kwargs) -> RenderConfig:
        """Create capture configuration.
        
        Args:
            format: Output format (PNG, JPEG, EXR, TIFF)
            resolution: Image resolution
            quality: Render quality preset
            **kwargs: Additional configuration options
            
        Returns:
            RenderConfig for capturing
        """
        config = self.pipeline.create_render_config(
            resolution=resolution,
            quality=quality,
            file_format=format,
            use_gpu=False,  # Force CPU for stability
            **kwargs
        )
        
        return config
    
    def capture_single_image(self,
                           model_path: Union[str, Path],
                           pose: CameraPose,
                           output_name: Optional[str] = None,
                           config: Optional[RenderConfig] = None) -> RenderResult:
        """Capture a single image at specified pose.
        
        Args:
            model_path: Path to GLB model
            pose: Camera pose for capture
            output_name: Optional custom output filename
            config: Render configuration
            
        Returns:
            RenderResult with capture outcome
        """
        if config is None:
            config = self.create_capture_config()
        
        # Generate output filename if not provided
        if output_name is None:
            timestamp = int(time.time())
            format_ext = config.file_format.lower()
            output_name = f"capture_{pose.yaw:03.0f}_{pose.pitch:03.0f}_{timestamp}.{format_ext}"
        
        output_path = self.output_base_dir / output_name
        
        logger.info(f"Capturing single image: {output_name}")
        result = self.pipeline.render_single(model_path, pose, output_path, config)
        
        if result.success:
            logger.info(f"Single image captured: {output_path}")
        else:
            logger.error(f"Single image capture failed: {result.error_message}")
        
        return result
    
    def capture_pose_sequence(self,
                            model_path: Union[str, Path],
                            poses: List[CameraPose],
                            session_name: Optional[str] = None,
                            config: Optional[RenderConfig] = None) -> CaptureSession:
        """Capture a sequence of images at multiple poses.
        
        Args:
            model_path: Path to GLB model
            poses: List of camera poses
            session_name: Optional session name for organization
            config: Render configuration
            
        Returns:
            CaptureSession with results
        """
        if config is None:
            config = self.create_capture_config()
        
        # Generate session info
        timestamp = datetime.now()
        if session_name is None:
            session_name = f"capture_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        session_id = f"{session_name}_{int(time.time())}"
        session_dir = self.output_base_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session
        session = CaptureSession(
            session_id=session_id,
            model_path=Path(model_path),
            output_dir=session_dir,
            config=config,
            poses=poses,
            start_time=timestamp,
            metadata={
                "model_name": Path(model_path).name,
                "pose_count": len(poses),
                "config": asdict(config)
            }
        )
        
        self.current_session = session
        
        logger.info(f"Starting capture session: {session_name} ({len(poses)} poses)")
        
        # Batch render all poses
        results = self.pipeline.render_batch(
            model_path=model_path,
            poses=poses,
            output_dir=session_dir,
            config=config,
            filename_pattern=f"{session_name}_{{yaw:03.0f}}_{{pitch:03.0f}}_{{index:04d}}.{config.file_format.lower()}"
        )
        
        # Update session with results
        session.results = results
        session.end_time = datetime.now()
        
        # Generate session report
        self._generate_session_report(session)
        
        # Mark session as completed
        self.completed_sessions.append(session)
        self.current_session = None
        
        successful_renders = sum(1 for r in results if r.success)
        total_time = (session.end_time - session.start_time).total_seconds()
        
        logger.info(f"Capture session completed: {successful_renders}/{len(poses)} successful ({total_time:.1f}s)")
        
        return session
    
    def capture_standard_views(self,
                             model_path: Union[str, Path],
                             session_name: Optional[str] = None,
                             config: Optional[RenderConfig] = None) -> CaptureSession:
        """Capture standard camera views (front, side, back, etc.).
        
        Args:
            model_path: Path to GLB model
            session_name: Optional session name
            config: Render configuration
            
        Returns:
            CaptureSession with standard views
        """
        # Generate standard views
        camera_system = TRELLISCameraSystem()
        standard_poses = generate_standard_views(camera_system)
        
        if session_name is None:
            session_name = f"standard_views_{Path(model_path).stem}"
        
        return self.capture_pose_sequence(model_path, standard_poses, session_name, config)
    
    def capture_orbit_sequence(self,
                             model_path: Union[str, Path],
                             orbit_steps: int = 12,
                             elevation: float = 30.0,
                             radius: float = 2.5,
                             session_name: Optional[str] = None,
                             config: Optional[RenderConfig] = None) -> CaptureSession:
        """Capture orbital sequence around model.
        
        Args:
            model_path: Path to GLB model
            orbit_steps: Number of steps around orbit
            elevation: Camera elevation angle
            radius: Distance from model
            session_name: Optional session name
            config: Render configuration
            
        Returns:
            CaptureSession with orbit sequence
        """
        # Generate orbit poses
        poses = []
        for i in range(orbit_steps):
            yaw = (360.0 * i) / orbit_steps
            pose = CameraPose(yaw, elevation, radius, 40.0)
            poses.append(pose)
        
        if session_name is None:
            session_name = f"orbit_{Path(model_path).stem}_{orbit_steps}steps"
        
        return self.capture_pose_sequence(model_path, poses, session_name, config)
    
    def export_session_data(self, session: CaptureSession, include_images: bool = False) -> Path:
        """Export session data and optionally images to archive.
        
        Args:
            session: CaptureSession to export
            include_images: Whether to include rendered images
            
        Returns:
            Path to exported archive
        """
        export_dir = self.output_base_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Create export package
        export_name = f"{session.session_id}_export"
        export_path = export_dir / export_name
        export_path.mkdir(exist_ok=True)
        
        # Export session metadata
        metadata_file = export_path / "session_metadata.json"
        session_data = {
            "session_id": session.session_id,
            "model_path": str(session.model_path),
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "config": {
                **asdict(session.config),
                "quality": session.config.quality.value  # Convert enum to string
            },
            "metadata": session.metadata,
            "poses": [asdict(pose) for pose in session.poses],
            "results": [
                {
                    "success": r.success,
                    "output_path": str(r.output_path) if r.output_path else None,
                    "render_time": r.render_time,
                    "error_message": r.error_message
                }
                for r in session.results
            ]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Export camera poses in TRELLIS format
        poses_file = export_path / "camera_poses.json"
        camera_system = TRELLISCameraSystem()
        camera_system.save_poses(session.poses, poses_file)
        
        # Copy images if requested
        if include_images:
            images_dir = export_path / "images"
            images_dir.mkdir(exist_ok=True)
            
            for result in session.results:
                if result.success and result.output_path and result.output_path.exists():
                    target_path = images_dir / result.output_path.name
                    shutil.copy2(result.output_path, target_path)
        
        # Create archive
        archive_path = export_dir / f"{export_name}.zip"
        shutil.make_archive(str(archive_path.with_suffix('')), 'zip', export_path)
        
        # Clean up temporary export directory
        shutil.rmtree(export_path, ignore_errors=True)
        
        logger.info(f"Session exported to: {archive_path}")
        return archive_path
    
    def _generate_session_report(self, session: CaptureSession):
        """Generate session report file.
        
        Args:
            session: CaptureSession to report on
        """
        report_file = session.output_dir / "session_report.txt"
        
        successful_renders = sum(1 for r in session.results if r.success)
        failed_renders = len(session.results) - successful_renders
        total_time = (session.end_time - session.start_time).total_seconds() if session.end_time else 0
        avg_time = total_time / len(session.results) if session.results else 0
        
        report_content = f"""
RENDER SESSION REPORT
====================

Session ID: {session.session_id}
Model: {session.model_path.name}
Start Time: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else 'N/A'}
Duration: {total_time:.1f} seconds

CONFIGURATION
============
Resolution: {session.config.resolution[0]}x{session.config.resolution[1]}
Quality: {session.config.quality.value}
Engine: {session.config.engine}
Format: {session.config.file_format}
Samples: {session.config.get_samples()}

RESULTS
=======
Total Poses: {len(session.poses)}
Successful Renders: {successful_renders}
Failed Renders: {failed_renders}
Success Rate: {(successful_renders/len(session.poses)*100):.1f}%
Average Render Time: {avg_time:.1f}s

POSE DETAILS
===========
"""
        
        for i, (pose, result) in enumerate(zip(session.poses, session.results)):
            status = "SUCCESS" if result.success else "FAILED"
            render_time = f"{result.render_time:.1f}s" if result.success else "N/A"
            
            report_content += f"  {i+1:2d}. yaw={pose.yaw:6.1f}° pitch={pose.pitch:6.1f}° radius={pose.radius:.1f} - {status} ({render_time})\\n"
        
        if failed_renders > 0:
            report_content += "\\nERROR DETAILS\\n============\\n"
            for i, result in enumerate(session.results):
                if not result.success:
                    pose = session.poses[i]
                    report_content += f"  Pose {i+1} (yaw={pose.yaw:.1f}°, pitch={pose.pitch:.1f}°): {result.error_message}\\n"
        
        report_file.write_text(report_content)
        logger.info(f"Session report generated: {report_file}")
    
    def get_session_summary(self, session: CaptureSession) -> Dict[str, Any]:
        """Get summary statistics for a session.
        
        Args:
            session: CaptureSession to summarize
            
        Returns:
            Dictionary with session statistics
        """
        if not session.results:
            return {"status": "no_results"}
        
        successful_renders = sum(1 for r in session.results if r.success)
        total_time = (session.end_time - session.start_time).total_seconds() if session.end_time else 0
        
        return {
            "session_id": session.session_id,
            "model_name": session.model_path.name,
            "total_poses": len(session.poses),
            "successful_renders": successful_renders,
            "failed_renders": len(session.results) - successful_renders,
            "success_rate": successful_renders / len(session.results),
            "total_time": total_time,
            "average_render_time": total_time / len(session.results) if session.results else 0,
            "output_dir": str(session.output_dir)
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.pipeline.cleanup()
        logger.info("Image capture system cleaned up")


# Convenience functions for quick captures
def quick_capture_single(model_path: Union[str, Path],
                        pose: CameraPose,
                        output_path: Union[str, Path],
                        format: str = "PNG",
                        resolution: Tuple[int, int] = (512, 512),
                        quality: RenderQuality = RenderQuality.STANDARD) -> bool:
    """Quick single image capture.
    
    Args:
        model_path: Path to GLB model
        pose: Camera pose
        output_path: Output image path
        format: Image format
        resolution: Image resolution  
        quality: Render quality
        
    Returns:
        True if successful
    """
    exporter = ImageCaptureExporter()
    config = exporter.create_capture_config(format, resolution, quality)
    
    try:
        result = exporter.pipeline.render_single(model_path, pose, output_path, config)
        return result.success
    finally:
        exporter.cleanup()


def quick_capture_standard_views(model_path: Union[str, Path],
                                output_dir: Union[str, Path],
                                format: str = "PNG",
                                resolution: Tuple[int, int] = (512, 512),
                                quality: RenderQuality = RenderQuality.STANDARD) -> int:
    """Quick capture of standard views.
    
    Args:
        model_path: Path to GLB model
        output_dir: Output directory
        format: Image format
        resolution: Image resolution
        quality: Render quality
        
    Returns:
        Number of successful captures
    """
    exporter = ImageCaptureExporter(output_dir)
    config = exporter.create_capture_config(format, resolution, quality)
    
    try:
        session = exporter.capture_standard_views(model_path, config=config)
        return sum(1 for r in session.results if r.success)
    finally:
        exporter.cleanup()