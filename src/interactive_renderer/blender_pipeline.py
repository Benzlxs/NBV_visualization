#!/usr/bin/env python3
"""High-quality Blender rendering pipeline for interactive camera controls.

This module implements Step 6: Blender Rendering Pipeline with:
- Configurable render quality and resolution
- Material and lighting optimization  
- Batch rendering capabilities
- Integration with TRELLIS camera system
"""

import logging
import subprocess
import tempfile
import json
import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from .blender_obj_renderer import BlenderOBJRenderer
from .trellis_camera import CameraPose, TRELLISCameraSystem
from .glb_loader import GLBLoader

logger = logging.getLogger(__name__)


class RenderQuality(Enum):
    """Predefined render quality settings."""
    PREVIEW = "preview"      # Fast preview (64 samples, EEVEE)
    STANDARD = "standard"    # Balanced quality (128 samples, CYCLES)
    HIGH = "high"           # High quality (256 samples, CYCLES)
    PRODUCTION = "production" # Maximum quality (512 samples, CYCLES)


@dataclass
class RenderConfig:
    """Configuration for Blender rendering."""
    # Resolution
    resolution: Tuple[int, int] = (512, 512)
    
    # Quality settings
    quality: RenderQuality = RenderQuality.STANDARD
    samples: Optional[int] = None  # Override quality default
    engine: str = "CYCLES"  # CYCLES or BLENDER_EEVEE
    
    # Denoising
    use_denoising: bool = True
    denoiser: str = "OPTIX"  # OPTIX, OIDN, or NLM
    
    # Lighting
    use_hdri: bool = True
    hdri_strength: float = 1.0
    sun_strength: float = 3.0
    sun_angle: float = 45.0
    
    # Materials
    material_preview: bool = False  # Use material preview mode
    
    # Performance
    use_gpu: bool = False  # Default to CPU for compatibility
    tile_size: int = 256
    
    # Output
    file_format: str = "PNG"  # PNG, JPEG, EXR, TIFF
    color_depth: str = "8"    # 8, 16, 32
    compression: int = 15     # 0-100 for JPEG, 0-15 for PNG
    
    def get_samples(self) -> int:
        """Get render samples based on quality setting."""
        if self.samples is not None:
            return self.samples
            
        quality_samples = {
            RenderQuality.PREVIEW: 32,
            RenderQuality.STANDARD: 128,
            RenderQuality.HIGH: 256,
            RenderQuality.PRODUCTION: 512
        }
        return quality_samples.get(self.quality, 128)
    
    def get_engine(self) -> str:
        """Get render engine based on quality setting."""
        if self.quality == RenderQuality.PREVIEW:
            return "BLENDER_EEVEE"
        return self.engine


@dataclass
class RenderJob:
    """Individual render job configuration."""
    model_path: Path
    pose: CameraPose
    output_path: Path
    config: RenderConfig
    job_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RenderResult:
    """Result of a render operation."""
    job: RenderJob
    success: bool
    output_path: Optional[Path] = None
    render_time: float = 0.0
    error_message: Optional[str] = None
    blender_output: Optional[str] = None


class BlenderRenderingPipeline:
    """High-quality Blender rendering pipeline with batch capabilities."""
    
    def __init__(self, blender_executable: Optional[str] = None):
        """Initialize Blender rendering pipeline.
        
        Args:
            blender_executable: Path to Blender executable
        """
        self.renderer = BlenderOBJRenderer(blender_executable)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="render_pipeline_"))
        
        # Render queue and status
        self.render_queue: List[RenderJob] = []
        self.completed_renders: List[RenderResult] = []
        self.current_job: Optional[RenderJob] = None
        self.is_rendering = False
        
        # Callbacks
        self.progress_callback: Optional[Callable[[RenderJob, float], None]] = None
        self.completion_callback: Optional[Callable[[RenderResult], None]] = None
        
        logger.info("Blender rendering pipeline initialized")
    
    def create_render_config(self, 
                           quality: RenderQuality = RenderQuality.STANDARD,
                           resolution: Tuple[int, int] = (512, 512),
                           **kwargs) -> RenderConfig:
        """Create render configuration with sensible defaults.
        
        Args:
            quality: Render quality preset
            resolution: Output resolution
            **kwargs: Additional config parameters
            
        Returns:
            RenderConfig instance
        """
        config = RenderConfig(quality=quality, resolution=resolution)
        
        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        return config
    
    def render_single(self, 
                     model_path: Union[str, Path],
                     pose: CameraPose,
                     output_path: Union[str, Path],
                     config: Optional[RenderConfig] = None) -> RenderResult:
        """Render a single image with specified pose.
        
        Args:
            model_path: Path to GLB model
            pose: Camera pose for rendering
            output_path: Output image path
            config: Render configuration
            
        Returns:
            RenderResult with outcome
        """
        if config is None:
            config = self.create_render_config()
        
        job = RenderJob(
            model_path=Path(model_path),
            pose=pose,
            output_path=Path(output_path),
            config=config,
            job_id=f"single_{int(time.time())}"
        )
        
        return self._execute_render_job(job)
    
    def render_batch(self,
                    model_path: Union[str, Path],
                    poses: List[CameraPose],
                    output_dir: Union[str, Path],
                    config: Optional[RenderConfig] = None,
                    filename_pattern: str = "render_{yaw:03.0f}_{pitch:03.0f}_{radius:.1f}.png") -> List[RenderResult]:
        """Render multiple images with different poses.
        
        Args:
            model_path: Path to GLB model
            poses: List of camera poses
            output_dir: Output directory
            config: Render configuration
            filename_pattern: Output filename pattern
            
        Returns:
            List of RenderResult instances
        """
        if config is None:
            config = self.create_render_config()
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, pose in enumerate(poses):
            # Generate filename from pattern
            filename = filename_pattern.format(
                yaw=pose.yaw,
                pitch=pose.pitch,
                radius=pose.radius,
                fov=pose.fov,
                index=i
            )
            
            output_path = output_dir / filename
            
            job = RenderJob(
                model_path=Path(model_path),
                pose=pose,
                output_path=output_path,
                config=config,
                job_id=f"batch_{i:04d}",
                metadata={"batch_index": i, "total_poses": len(poses)}
            )
            
            logger.info(f"Rendering batch {i+1}/{len(poses)}: {filename}")
            result = self._execute_render_job(job)
            results.append(result)
            
            # Report progress
            if self.progress_callback:
                self.progress_callback(job, (i + 1) / len(poses))
        
        return results
    
    def queue_render(self, job: RenderJob):
        """Add render job to queue."""
        self.render_queue.append(job)
        logger.info(f"Queued render job: {job.job_id}")
    
    def process_queue_async(self):
        """Process render queue in background thread."""
        if self.is_rendering:
            logger.warning("Already rendering, cannot start new queue processing")
            return
        
        thread = threading.Thread(target=self._process_queue, daemon=True)
        thread.start()
    
    def _process_queue(self):
        """Process all jobs in render queue."""
        self.is_rendering = True
        
        try:
            while self.render_queue:
                job = self.render_queue.pop(0)
                self.current_job = job
                
                logger.info(f"Processing render job: {job.job_id}")
                result = self._execute_render_job(job)
                
                self.completed_renders.append(result)
                
                if self.completion_callback:
                    self.completion_callback(result)
                    
        except Exception as e:
            logger.error(f"Queue processing failed: {e}")
        finally:
            self.is_rendering = False
            self.current_job = None
    
    def _execute_render_job(self, job: RenderJob) -> RenderResult:
        """Execute a single render job.
        
        Args:
            job: Render job to execute
            
        Returns:
            RenderResult with outcome
        """
        start_time = time.time()
        
        try:
            # Create Blender script for rendering
            script_path = self._create_render_script(job)
            
            # Execute Blender render
            result = self._run_blender_render(script_path, job)
            
            render_time = time.time() - start_time
            
            if result.returncode == 0 and job.output_path.exists():
                logger.info(f"Render successful: {job.output_path} ({render_time:.1f}s)")
                return RenderResult(
                    job=job,
                    success=True,
                    output_path=job.output_path,
                    render_time=render_time,
                    blender_output=result.stdout
                )
            else:
                logger.error(f"Render failed: {result.stderr}")
                return RenderResult(
                    job=job,
                    success=False,
                    render_time=render_time,
                    error_message=result.stderr,
                    blender_output=result.stdout
                )
                
        except Exception as e:
            render_time = time.time() - start_time
            logger.error(f"Render job failed: {e}")
            return RenderResult(
                job=job,
                success=False,
                render_time=render_time,
                error_message=str(e)
            )
    
    def _create_render_script(self, job: RenderJob) -> Path:
        """Create Blender Python script for rendering.
        
        Args:
            job: Render job configuration
            
        Returns:
            Path to generated script
        """
        # Convert GLB to OBJ first
        loader = GLBLoader()
        model = loader.load_model(job.model_path)
        
        obj_path = self.temp_dir / f"{job.job_id}_model.obj"
        model.mesh.export(obj_path, file_type='obj', include_normals=True, include_texture=True)
        
        # Generate camera position
        x, y, z = job.pose.to_cartesian()
        
        script_content = f'''
import bpy
import mathutils
import math
import os
import sys

# Force headless operation - no display required
print("=== STARTING HEADLESS BLENDER RENDER ===")

# Configure headless rendering
try:
    # Disable GPU rendering completely for stability  
    if hasattr(bpy.context.scene, 'cycles'):
        bpy.context.scene.cycles.device = 'CPU'
    
    # Set preferences for headless mode
    if hasattr(bpy.context, 'preferences'):
        prefs = bpy.context.preferences
        if hasattr(prefs, 'system'):
            prefs.system.use_gpu_subdivision = False
        if hasattr(prefs, 'view'):
            prefs.view.show_splash = False
            
except Exception as e:
    print(f"Warning: Could not configure headless preferences: {{e}}")

print("=== BLENDER RENDER PIPELINE ===")
print(f"Job ID: {job.job_id}")
print(f"Model: {job.model_path.name}")
print(f"Pose: yaw={job.pose.yaw}, pitch={job.pose.pitch}, radius={job.pose.radius}, fov={job.pose.fov}")
print(f"Output: {job.output_path}")
print(f"Blender version: {{bpy.app.version}}")

try:
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # Set render engine
    bpy.context.scene.render.engine = '{job.config.get_engine()}'
    
    # Configure render settings
    scene = bpy.context.scene
    scene.render.resolution_x = {job.config.resolution[0]}
    scene.render.resolution_y = {job.config.resolution[1]}
    scene.render.filepath = r"{str(job.output_path)}"
    scene.render.image_settings.file_format = '{job.config.file_format}'
    scene.render.image_settings.color_depth = '{job.config.color_depth}'
    
    if '{job.config.file_format}' == 'PNG':
        scene.render.image_settings.compression = {job.config.compression}
    elif '{job.config.file_format}' == 'JPEG':
        scene.render.image_settings.quality = {job.config.compression}
    
    # Configure engine-specific settings
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = {job.config.get_samples()}
        scene.cycles.use_denoising = {str(job.config.use_denoising)}
        
        # Force CPU rendering for headless stability
        print("Forcing CPU rendering for headless operation")
        scene.cycles.device = 'CPU'
        
        scene.cycles.tile_size = {job.config.tile_size}
        
    elif scene.render.engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = {job.config.get_samples()}
        scene.eevee.use_ssr = True
        scene.eevee.use_ssr_refraction = True
        scene.eevee.use_bloom = True
    
    # Import OBJ model
    obj_path = r"{str(obj_path)}"
    bpy.ops.import_scene.obj(filepath=obj_path)
    imported = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    
    if imported:
        # Center and scale model
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3
        
        for obj in imported:
            for vertex in obj.data.vertices:
                world_vertex = obj.matrix_world @ vertex.co
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_vertex[i])
                    max_coords[i] = max(max_coords[i], world_vertex[i])
        
        center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
        size = [max_coords[i] - min_coords[i] for i in range(3)]
        max_size = max(size)
        
        for obj in imported:
            obj.location.x -= center[0]
            obj.location.y -= center[1]
            obj.location.z -= center[2]
            
            if max_size > 0:
                scale_factor = 2.0 / max_size
                obj.scale = (scale_factor, scale_factor, scale_factor)
        
        # Create camera
        cam_data = bpy.data.cameras.new("RenderCamera")
        cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        
        # Set FOV
        cam_data.lens_unit = 'FOV'
        cam_data.angle = math.radians({job.pose.fov})
        
        # Position camera
        cam_obj.location = ({x}, {y}, {z})
        
        # Point at origin
        direction = mathutils.Vector((0, 0, 0)) - cam_obj.location
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        
        bpy.context.scene.camera = cam_obj
        
        # Setup lighting
        if {str(job.config.use_hdri)}:
            # Use HDRI environment lighting
            world = bpy.context.scene.world
            world.use_nodes = True
            nodes = world.node_tree.nodes
            links = world.node_tree.links
            
            # Clear existing nodes
            for node in nodes:
                nodes.remove(node)
            
            # Add Environment Texture node
            env_node = nodes.new(type='ShaderNodeTexEnvironment')
            background_node = nodes.new(type='ShaderNodeBackground')
            output_node = nodes.new(type='ShaderNodeOutputWorld')
            
            # Connect nodes
            links.new(env_node.outputs['Color'], background_node.inputs['Color'])
            links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
            
            # Set strength
            background_node.inputs['Strength'].default_value = {job.config.hdri_strength}
        
        # Add sun light
        sun_data = bpy.data.lights.new("Sun", type='SUN')
        sun_data.energy = {job.config.sun_strength}
        sun_obj = bpy.data.objects.new("Sun", sun_data)
        bpy.context.collection.objects.link(sun_obj)
        
        # Position sun at angle
        sun_angle_rad = math.radians({job.config.sun_angle})
        sun_obj.location = (2 * math.cos(sun_angle_rad), 2 * math.sin(sun_angle_rad), 2)
        sun_obj.rotation_euler = (sun_angle_rad, 0, 0)
        
        # Render
        print("Starting render...")
        bpy.ops.render.render(write_still=True)
        print("✅ Render complete!")
        
        # Verify output file
        if os.path.exists(r"{str(job.output_path)}"):
            print(f"✅ Output saved: {job.output_path}")
        else:
            print("❌ Output file not found")
    
    else:
        print("❌ No objects imported")

except Exception as e:
    print(f"❌ Error: {{type(e).__name__}}: {{e}}")
    import traceback
    traceback.print_exc()

print("=== RENDER COMPLETE ===")
'''
        
        script_path = self.temp_dir / f"{job.job_id}_render.py"
        script_path.write_text(script_content)
        
        return script_path
    
    def _run_blender_render(self, script_path: Path, job: RenderJob) -> subprocess.CompletedProcess:
        """Execute Blender render with script.
        
        Args:
            script_path: Path to Blender Python script
            job: Render job configuration
            
        Returns:
            subprocess.CompletedProcess result
        """
        # Use Blender in true headless mode with minimal dependencies
        cmd = [
            self.renderer.blender_executable, 
            '--background',  # No GUI
            '--factory-startup',  # Clean startup
            '--python', str(script_path)
        ]
        
        env = os.environ.copy()
        # Complete headless setup
        env.pop('DISPLAY', None)  # Remove any display
        env['BLENDER_SYSTEM_SCRIPTS'] = ''  # Minimal system
        env['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering
        env['MESA_GL_VERSION_OVERRIDE'] = '3.3'  # Ensure compatibility
        
        timeout = 300  # 5 minutes max per render
        
        if job.config.quality == RenderQuality.PRODUCTION:
            timeout = 600  # 10 minutes for production renders
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        return result
    
    def get_render_progress(self) -> Dict[str, Any]:
        """Get current rendering progress.
        
        Returns:
            Progress information
        """
        return {
            "is_rendering": self.is_rendering,
            "current_job": asdict(self.current_job) if self.current_job else None,
            "queue_length": len(self.render_queue),
            "completed_count": len(self.completed_renders)
        }
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")


# Convenience functions for easy use
def quick_render(model_path: Union[str, Path],
                pose: CameraPose,
                output_path: Union[str, Path],
                quality: RenderQuality = RenderQuality.STANDARD) -> RenderResult:
    """Quick single render with default settings.
    
    Args:
        model_path: Path to GLB model
        pose: Camera pose
        output_path: Output image path
        quality: Render quality preset
        
    Returns:
        RenderResult
    """
    pipeline = BlenderRenderingPipeline()
    config = pipeline.create_render_config(quality=quality)
    
    try:
        return pipeline.render_single(model_path, pose, output_path, config)
    finally:
        pipeline.cleanup()


def batch_render_views(model_path: Union[str, Path],
                      poses: List[CameraPose],
                      output_dir: Union[str, Path],
                      quality: RenderQuality = RenderQuality.STANDARD) -> List[RenderResult]:
    """Batch render multiple views with default settings.
    
    Args:
        model_path: Path to GLB model
        poses: List of camera poses
        output_dir: Output directory
        quality: Render quality preset
        
    Returns:
        List of RenderResult instances
    """
    pipeline = BlenderRenderingPipeline()
    config = pipeline.create_render_config(quality=quality)
    
    try:
        return pipeline.render_batch(model_path, poses, output_dir, config)
    finally:
        pipeline.cleanup()