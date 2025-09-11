#!/usr/bin/env python3
"""Completely headless Blender renderer that works reliably without display.

This module provides a rock-solid headless Blender rendering solution that:
- Never requires display or GUI components
- Uses the most minimal Blender setup possible
- Has extensive fallback and error recovery
- Generates reliable, high-quality renders
"""

import logging
import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

from .trellis_camera import CameraPose
from .glb_loader import GLBLoader

logger = logging.getLogger(__name__)


@dataclass
class HeadlessRenderResult:
    """Result of headless rendering operation."""
    success: bool
    output_path: Optional[Path] = None
    render_time: float = 0.0
    error_message: Optional[str] = None
    blender_output: Optional[str] = None


class HeadlessBlenderRenderer:
    """Ultra-reliable headless Blender renderer with no display dependencies."""
    
    def __init__(self, blender_executable: Optional[str] = None):
        """Initialize headless renderer.
        
        Args:
            blender_executable: Path to Blender executable
        """
        self.blender_executable = blender_executable or self._find_blender()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="headless_blender_"))
        
        # Validate Blender
        self._validate_blender()
        
        logger.info("Headless Blender renderer initialized")
    
    def _find_blender(self) -> str:
        """Find Blender executable."""
        candidates = ['blender', '/usr/bin/blender', '/usr/local/bin/blender']
        
        for candidate in candidates:
            try:
                result = subprocess.run([candidate, '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return candidate
            except:
                continue
        
        raise RuntimeError("Blender not found in PATH")
    
    def _validate_blender(self):
        """Validate Blender installation."""
        try:
            result = subprocess.run([self.blender_executable, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"Blender validation failed: {result.stderr}")
            logger.info(f"Blender validated: {result.stdout.split()[1] if result.stdout else 'unknown version'}")
        except Exception as e:
            raise RuntimeError(f"Blender validation error: {e}")
    
    def render_glb_model(self, 
                        glb_path: Union[str, Path],
                        pose: CameraPose,
                        output_path: Union[str, Path],
                        resolution: Tuple[int, int] = (512, 512),
                        samples: int = 64,
                        engine: str = "CYCLES") -> HeadlessRenderResult:
        """Render GLB model with specified camera pose.
        
        Args:
            glb_path: Path to GLB model
            pose: Camera pose
            output_path: Output image path
            resolution: Image resolution
            samples: Render samples
            engine: Render engine (CYCLES/BLENDER_EEVEE)
            
        Returns:
            HeadlessRenderResult
        """
        start_time = time.time()
        
        try:
            glb_path = Path(glb_path)
            output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert GLB to OBJ (proven working approach)
            obj_path = self._convert_glb_to_obj(glb_path)
            
            # Create minimal Blender script
            script_path = self._create_minimal_render_script(
                obj_path, pose, output_path, resolution, samples, engine
            )
            
            # Execute headless render
            result = self._execute_headless_render(script_path)
            
            render_time = time.time() - start_time
            
            # Check results
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Headless render successful: {output_path} ({render_time:.1f}s)")
                return HeadlessRenderResult(
                    success=True,
                    output_path=output_path,
                    render_time=render_time,
                    blender_output=result.stdout
                )
            else:
                error_msg = result.stderr or "Unknown render error"
                logger.error(f"Headless render failed: {error_msg}")
                return HeadlessRenderResult(
                    success=False,
                    render_time=render_time,
                    error_message=error_msg,
                    blender_output=result.stdout
                )
                
        except Exception as e:
            render_time = time.time() - start_time
            logger.error(f"Headless render exception: {e}")
            return HeadlessRenderResult(
                success=False,
                render_time=render_time,
                error_message=str(e)
            )
    
    def _convert_glb_to_obj(self, glb_path: Path) -> Path:
        """Convert GLB to OBJ for reliable import.
        
        Args:
            glb_path: Path to GLB file
            
        Returns:
            Path to converted OBJ file
        """
        loader = GLBLoader()
        model = loader.load_model(glb_path)
        
        obj_path = self.temp_dir / f"model_{int(time.time())}.obj"
        
        # Export with materials and textures
        model.mesh.export(
            obj_path, 
            file_type='obj', 
            include_normals=True,
            include_texture=True
        )
        
        return obj_path
    
    def _create_minimal_render_script(self,
                                    obj_path: Path,
                                    pose: CameraPose,
                                    output_path: Path,
                                    resolution: Tuple[int, int],
                                    samples: int,
                                    engine: str) -> Path:
        """Create minimal Blender render script.
        
        Args:
            obj_path: Path to OBJ model
            pose: Camera pose
            output_path: Output image path
            resolution: Image resolution
            samples: Render samples
            engine: Render engine
            
        Returns:
            Path to render script
        """
        x, y, z = pose.to_cartesian()
        
        # Ultra-minimal Blender script with maximum compatibility
        script_content = f'''
import bpy
import mathutils
import math
import sys
import os

print("=== HEADLESS BLENDER RENDER ===")
print("Blender version:", bpy.app.version)
print("Python version:", sys.version)

try:
    # Configure for absolute headless operation
    bpy.context.preferences.view.show_splash = False
    
    # Clear default scene completely
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # Clear materials and textures
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Set render engine first
    scene = bpy.context.scene
    scene.render.engine = '{engine}'
    
    # Configure render settings
    scene.render.resolution_x = {resolution[0]}
    scene.render.resolution_y = {resolution[1]}
    scene.render.resolution_percentage = 100
    scene.render.filepath = r"{str(output_path)}"
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    
    # Engine-specific settings with maximum stability
    if scene.render.engine == 'CYCLES':
        # Force CPU rendering for maximum stability
        scene.cycles.device = 'CPU'
        scene.cycles.samples = {samples}
        
        # Denoising with version compatibility
        try:
            scene.cycles.use_denoising = True
            # Try different denoiser options based on Blender version
            if hasattr(scene.cycles, 'denoiser'):
                # Blender 3.0+ API
                available_denoisers = ['OPTIX', 'OIDN', 'NLM']
                for denoiser in available_denoisers:
                    try:
                        scene.cycles.denoiser = denoiser
                        break
                    except:
                        continue
            else:
                # Legacy API
                scene.cycles.use_denoising = True
        except Exception as e:
            print(f"Denoising setup failed, continuing without: {{e}}")
            scene.cycles.use_denoising = False
        
        # Disable GPU-related features
        scene.cycles.feature_set = 'SUPPORTED'
        
        # Tile size with version compatibility
        if hasattr(scene.cycles, 'tile_size'):
            scene.cycles.tile_size = 64
        
    elif scene.render.engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = {samples}
        scene.eevee.use_ssr = False  # Disable for stability
        scene.eevee.use_ssr_refraction = False
        scene.eevee.use_bloom = False
        scene.eevee.use_volumetric_lights = False
    
    # Import OBJ model with error handling
    print(f"Importing model: {obj_path}")
    try:
        bpy.ops.import_scene.obj(
            filepath=r"{str(obj_path)}",
            axis_forward='-Z',
            axis_up='Y'
        )
        print("Model import successful")
    except Exception as e:
        print(f"Model import failed: {{e}}")
        raise
    
    # Get imported objects
    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    
    if not imported_objects:
        raise Exception("No mesh objects imported")
    
    print(f"Imported {{len(imported_objects)}} mesh objects")
    
    # Center and scale model
    all_vertices = []
    for obj in imported_objects:
        for vertex in obj.data.vertices:
            world_vertex = obj.matrix_world @ vertex.co
            all_vertices.append(world_vertex)
    
    if all_vertices:
        # Calculate bounding box
        min_coords = [min(v[i] for v in all_vertices) for i in range(3)]
        max_coords = [max(v[i] for v in all_vertices) for i in range(3)]
        center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
        size = [max_coords[i] - min_coords[i] for i in range(3)]
        max_size = max(size) if size else 1.0
        
        # Apply centering and scaling
        for obj in imported_objects:
            obj.location = (obj.location.x - center[0], 
                          obj.location.y - center[1], 
                          obj.location.z - center[2])
            
            if max_size > 0:
                scale_factor = 2.0 / max_size
                obj.scale = (scale_factor, scale_factor, scale_factor)
        
        print(f"Model centered and scaled (max_size={{max_size:.3f}})")
    
    # Create camera
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    
    # Set camera parameters
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians({pose.fov})
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0
    
    # Position camera
    cam_obj.location = ({x}, {y}, {z})
    
    # Point camera at origin
    direction = mathutils.Vector((0, 0, 0)) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Set as active camera
    scene.camera = cam_obj
    
    print(f"Camera positioned at ({{cam_obj.location.x:.2f}}, {{cam_obj.location.y:.2f}}, {{cam_obj.location.z:.2f}})")
    
    # Add simple lighting
    light_data = bpy.data.lights.new("Sun", type='SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (2, 2, 2)
    light_obj.rotation_euler = (0.785398, 0, 0.785398)  # 45 degrees
    
    # Add fill light
    fill_data = bpy.data.lights.new("Fill", type='SUN')
    fill_data.energy = 1.0
    fill_obj = bpy.data.objects.new("Fill", fill_data)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.location = (-2, -2, 1)
    fill_obj.rotation_euler = (-0.785398, 0, -0.785398)
    
    print("Lighting setup complete")
    
    # Final render
    print("Starting render...")
    bpy.ops.render.render(write_still=True)
    print("Render complete!")
    
    # Verify output
    if os.path.exists(r"{str(output_path)}"):
        file_size = os.path.getsize(r"{str(output_path)}")
        print(f"Output file created: {{file_size}} bytes")
    else:
        raise Exception("Output file not created")

except Exception as e:
    print(f"RENDER FAILED: {{type(e).__name__}}: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== HEADLESS RENDER COMPLETE ===")
'''
        
        script_path = self.temp_dir / f"render_{int(time.time())}.py"
        script_path.write_text(script_content)
        
        return script_path
    
    def _execute_headless_render(self, script_path: Path) -> subprocess.CompletedProcess:
        """Execute Blender render with maximum headless compatibility.
        
        Args:
            script_path: Path to render script
            
        Returns:
            subprocess.CompletedProcess result
        """
        # Ultimate headless command with all compatibility flags
        cmd = [
            self.blender_executable,
            '--background',           # No GUI
            '--factory-startup',      # Clean startup
            '--enable-autoexec',      # Enable scripts
            '--python', str(script_path)
        ]
        
        # Ultimate headless environment
        env = os.environ.copy()
        
        # Remove all display-related variables
        display_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 'DESKTOP_SESSION']
        for var in display_vars:
            env.pop(var, None)
        
        # Force software rendering
        env.update({
            'LIBGL_ALWAYS_SOFTWARE': '1',
            'GALLIUM_DRIVER': 'llvmpipe',
            'MESA_GL_VERSION_OVERRIDE': '3.3',
            'MESA_GLSL_VERSION_OVERRIDE': '330',
            'BLENDER_SYSTEM_SCRIPTS': '',
            'BLENDER_SYSTEM_DATAFILES': '',
            'BLENDER_USER_SCRIPTS': '',
            'BLENDER_USER_DATAFILES': ''
        })
        
        logger.info(f"Executing headless render: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes max
                env=env,
                cwd=self.temp_dir  # Run from temp directory
            )
            
            return result
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Render timeout (5 minutes)")
        except Exception as e:
            raise RuntimeError(f"Render execution failed: {e}")
    
    def batch_render(self,
                    glb_path: Union[str, Path],
                    poses: List[CameraPose],
                    output_dir: Union[str, Path],
                    resolution: Tuple[int, int] = (512, 512),
                    samples: int = 64,
                    engine: str = "CYCLES",
                    filename_pattern: str = "render_{yaw:03.0f}_{pitch:03.0f}_{index:03d}.png") -> List[HeadlessRenderResult]:
        """Batch render multiple poses.
        
        Args:
            glb_path: Path to GLB model
            poses: List of camera poses
            output_dir: Output directory
            resolution: Image resolution
            samples: Render samples
            engine: Render engine
            filename_pattern: Output filename pattern
            
        Returns:
            List of HeadlessRenderResult instances
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        logger.info(f"Starting batch render: {len(poses)} poses")
        
        for i, pose in enumerate(poses):
            filename = filename_pattern.format(
                yaw=pose.yaw,
                pitch=pose.pitch,
                radius=pose.radius,
                fov=pose.fov,
                index=i
            )
            
            output_path = output_dir / filename
            
            logger.info(f"Rendering {i+1}/{len(poses)}: {filename}")
            
            result = self.render_glb_model(
                glb_path, pose, output_path, resolution, samples, engine
            )
            
            results.append(result)
            
            if result.success:
                logger.info(f"  ✅ Success: {result.render_time:.1f}s")
            else:
                logger.error(f"  ❌ Failed: {result.error_message}")
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch render complete: {successful}/{len(poses)} successful")
        
        return results
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Headless renderer cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


# Convenience functions
def quick_headless_render(glb_path: Union[str, Path],
                         pose: CameraPose,
                         output_path: Union[str, Path],
                         resolution: Tuple[int, int] = (512, 512),
                         samples: int = 64) -> bool:
    """Quick headless render with default settings.
    
    Args:
        glb_path: Path to GLB model
        pose: Camera pose
        output_path: Output image path
        resolution: Image resolution
        samples: Render samples
        
    Returns:
        True if successful
    """
    renderer = HeadlessBlenderRenderer()
    
    try:
        result = renderer.render_glb_model(glb_path, pose, output_path, resolution, samples)
        return result.success
    finally:
        renderer.cleanup()


def quick_batch_render(glb_path: Union[str, Path],
                      poses: List[CameraPose],
                      output_dir: Union[str, Path],
                      resolution: Tuple[int, int] = (512, 512),
                      samples: int = 64) -> int:
    """Quick batch headless render.
    
    Args:
        glb_path: Path to GLB model
        poses: List of camera poses
        output_dir: Output directory
        resolution: Image resolution
        samples: Render samples
        
    Returns:
        Number of successful renders
    """
    renderer = HeadlessBlenderRenderer()
    
    try:
        results = renderer.batch_render(glb_path, poses, output_dir, resolution, samples)
        return sum(1 for r in results if r.success)
    finally:
        renderer.cleanup()