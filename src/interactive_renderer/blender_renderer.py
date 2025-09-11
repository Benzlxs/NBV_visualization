"""Blender-based renderer for GLB models following TRELLIS approach.

This module provides high-quality rendering of GLB models using Blender's Python API,
implementing TRELLIS-compatible camera positioning and rendering pipeline.
"""

import logging
import subprocess
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import math

logger = logging.getLogger(__name__)


class BlenderRenderer:
    """High-quality GLB renderer using Blender Python API."""
    
    def __init__(self, blender_executable: Optional[str] = None):
        """Initialize Blender renderer.
        
        Args:
            blender_executable: Path to Blender executable (auto-detect if None)
        """
        self.blender_executable = blender_executable or self._find_blender()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="blender_renderer_"))
        self.current_model_path: Optional[Path] = None
        self.render_settings = {
            'resolution': (512, 512),
            'samples': 64,
            'use_denoising': True,
            'engine': 'CYCLES',  # or 'EEVEE' for faster rendering
        }
        
        # Validate Blender installation
        self._validate_blender()
    
    def _find_blender(self) -> str:
        """Find Blender executable automatically."""
        possible_paths = [
            'blender',  # In PATH
            '/usr/bin/blender',
            '/usr/local/bin/blender',
            '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '--version'], 
                                      capture_output=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"Found Blender at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        raise RuntimeError("Blender executable not found. Please install Blender or specify path.")
    
    def _validate_blender(self) -> None:
        """Validate Blender installation and version."""
        try:
            result = subprocess.run([self.blender_executable, '--version'], 
                                  capture_output=True, timeout=10, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Blender validation failed: {result.stderr}")
            
            version_info = result.stdout
            logger.info(f"Blender validation successful: {version_info.split()[1]}")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender version check timed out")
        except Exception as e:
            raise RuntimeError(f"Blender validation error: {e}")
    
    def load_glb_model(self, glb_path: Union[str, Path]) -> bool:
        """Load a GLB model for rendering.
        
        Args:
            glb_path: Path to GLB file
            
        Returns:
            True if successful, False otherwise
        """
        glb_path = Path(glb_path)
        
        if not glb_path.exists():
            logger.error(f"GLB file not found: {glb_path}")
            return False
        
        if not glb_path.suffix.lower() == '.glb':
            logger.error(f"File is not a GLB: {glb_path}")
            return False
        
        self.current_model_path = glb_path
        logger.info(f"GLB model loaded: {glb_path.name}")
        return True
    
    def set_render_settings(self, **settings) -> None:
        """Update render settings.
        
        Args:
            **settings: Render settings to update
        """
        self.render_settings.update(settings)
        logger.debug(f"Render settings updated: {settings}")
    
    def render_image(
        self,
        yaw: float,
        pitch: float,
        radius: float,
        fov: float = 40.0,
        output_path: Optional[Union[str, Path]] = None,
        resolution: Optional[Tuple[int, int]] = None
    ) -> Path:
        """Render image from specified camera pose.
        
        Args:
            yaw: Camera yaw angle in degrees
            pitch: Camera pitch angle in degrees  
            radius: Camera distance from origin
            fov: Field of view in degrees
            output_path: Path to save rendered image (auto-generate if None)
            resolution: Render resolution (use default if None)
            
        Returns:
            Path to rendered image
            
        Raises:
            RuntimeError: If rendering fails
        """
        if self.current_model_path is None:
            raise RuntimeError("No GLB model loaded. Call load_glb_model() first.")
        
        # Set output path
        if output_path is None:
            output_path = self.temp_dir / f"render_{yaw:.1f}_{pitch:.1f}_{radius:.1f}.png"
        else:
            output_path = Path(output_path)
        
        # Set resolution
        if resolution is None:
            resolution = self.render_settings['resolution']
        
        # Create Blender script for rendering
        script_path = self._create_render_script(
            glb_path=self.current_model_path,
            yaw=yaw,
            pitch=pitch,
            radius=radius,
            fov=fov,
            output_path=output_path,
            resolution=resolution
        )
        
        # Execute Blender rendering
        success = self._execute_blender_script(script_path)
        
        if not success or not output_path.exists():
            raise RuntimeError(f"Blender rendering failed for pose yaw={yaw}, pitch={pitch}, radius={radius}")
        
        logger.info(f"Rendered image: {output_path}")
        return output_path
    
    def _create_render_script(
        self,
        glb_path: Path,
        yaw: float,
        pitch: float,
        radius: float,
        fov: float,
        output_path: Path,
        resolution: Tuple[int, int]
    ) -> Path:
        """Create Blender Python script for rendering.
        
        Args:
            glb_path: Path to GLB model
            yaw: Camera yaw in degrees
            pitch: Camera pitch in degrees
            radius: Camera distance
            fov: Field of view in degrees
            output_path: Output image path
            resolution: Render resolution
            
        Returns:
            Path to created script
        """
        script_content = f'''
import bpy
import bmesh
import mathutils
import math
import sys
import os

# Try to ensure numpy is available for GLB import
try:
    import numpy
    print("Numpy already available in Blender")
except ImportError:
    print("Numpy not found, attempting to install...")
    try:
        # Try to use pip to install numpy
        import subprocess as sp
        sp.check_call([sys.executable, "-m", "pip", "install", "--user", "numpy"])
        import numpy
        print("Numpy installed successfully")
    except Exception as e:
        print(f"Warning: Could not install numpy: {{e}}")
        print("Attempting to use system numpy path...")
        # Add conda environment's site-packages to path
        conda_site_packages = "/home/li325/miniconda3/envs/vggt/lib/python3.10/site-packages"
        if os.path.exists(conda_site_packages):
            sys.path.insert(0, conda_site_packages)
            try:
                import numpy
                print(f"Using numpy from: {{numpy.__file__}}")
            except ImportError:
                print("ERROR: Still cannot import numpy")

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Set render engine (handle different Blender versions)
engine = '{self.render_settings["engine"]}'
if engine == 'EEVEE':
    # Blender 3.0 uses BLENDER_EEVEE
    try:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    except:
        bpy.context.scene.render.engine = 'EEVEE'
elif engine == 'CYCLES':
    bpy.context.scene.render.engine = 'CYCLES'
else:
    bpy.context.scene.render.engine = engine

# Configure Cycles settings if using Cycles
if '{self.render_settings["engine"]}' == 'CYCLES':
    bpy.context.scene.cycles.samples = {self.render_settings["samples"]}
    bpy.context.scene.cycles.use_denoising = {self.render_settings["use_denoising"]}

# Import GLB model
bpy.ops.import_scene.gltf(filepath=r"{str(glb_path)}")

# Get imported objects and center them
imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

if not imported_objects:
    print("ERROR: No mesh objects imported from GLB")
    sys.exit(1)

# Calculate bounding box of all imported objects
min_coords = [float('inf')] * 3
max_coords = [float('-inf')] * 3

for obj in imported_objects:
    for vertex in obj.data.vertices:
        world_vertex = obj.matrix_world @ vertex.co
        for i in range(3):
            min_coords[i] = min(min_coords[i], world_vertex[i])
            max_coords[i] = max(max_coords[i], world_vertex[i])

# Calculate center and size
center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
size = [max_coords[i] - min_coords[i] for i in range(3)]
max_size = max(size)

print(f"Model center: {{center}}")
print(f"Model size: {{size}}, max: {{max_size}}")

# Move all objects to center them at origin
for obj in imported_objects:
    obj.location.x -= center[0]
    obj.location.y -= center[1] 
    obj.location.z -= center[2]

# Scale model to fit in unit sphere (optional, following GLB loader normalization)
if max_size > 0:
    scale_factor = 2.0 / max_size
    for obj in imported_objects:
        obj.scale = (scale_factor, scale_factor, scale_factor)

# Set up camera with TRELLIS-compatible positioning
camera = bpy.data.cameras.new(name="Camera")
camera_obj = bpy.data.objects.new("Camera", camera)
bpy.context.collection.objects.link(camera_obj)

# Set camera FOV
camera.lens_unit = 'FOV'
camera.angle = math.radians({fov})

# Position camera using spherical coordinates (TRELLIS format)
yaw_rad = math.radians({yaw})
pitch_rad = math.radians({pitch})
radius = {radius}

# Convert spherical to cartesian coordinates
# TRELLIS uses right-handed coordinate system with Y-up
x = radius * math.cos(pitch_rad) * math.sin(yaw_rad)
y = radius * math.sin(pitch_rad)
z = radius * math.cos(pitch_rad) * math.cos(yaw_rad)

camera_obj.location = (x, y, z)

# Make camera look at origin
direction = mathutils.Vector((0, 0, 0)) - camera_obj.location
camera_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

print(f"Camera position: {{camera_obj.location}}")
print(f"Camera rotation: {{camera_obj.rotation_euler}}")

# Set up lighting
# Add environment lighting
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # White background
bg.inputs[1].default_value = 1.0  # Strength

# Add key light
light_data = bpy.data.lights.new(name="KeyLight", type='SUN')
light_data.energy = 3.0
light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.location = (2, 2, 2)

# Set active camera
bpy.context.scene.camera = camera_obj

# Configure render settings
scene = bpy.context.scene
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}
scene.render.filepath = r"{str(output_path)}"
scene.render.image_settings.file_format = 'PNG'

# Render
print(f"Rendering to: {{scene.render.filepath}}")
bpy.ops.render.render(write_still=True)

print("Rendering complete")
'''
        
        # Save script to temporary file
        script_path = self.temp_dir / "render_script.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def _execute_blender_script(self, script_path: Path, timeout: int = 120) -> bool:
        """Execute Blender script.
        
        Args:
            script_path: Path to Blender Python script
            timeout: Timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use TRELLIS-style command flags
            cmd = [
                self.blender_executable,
                '-b',  # Background mode (TRELLIS style)
                '-P', str(script_path)  # Python script (TRELLIS style)
            ]
            
            # Set up environment for headless rendering (working approach)
            env = os.environ.copy()
            # Use vggt environment Python (Python 3.10 matches Blender)
            env['BLENDER_SYSTEM_PYTHON'] = '/home/li325/miniconda3/envs/vggt/bin/python'
            # Headless mode - set empty display
            env['DISPLAY'] = ''
            
            logger.debug(f"Executing Blender command: {' '.join(cmd)}")
            logger.debug(f"Using BLENDER_SYSTEM_PYTHON: {env.get('BLENDER_SYSTEM_PYTHON')}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                text=True,
                env=env
            )
            
            if result.returncode == 0:
                logger.debug("Blender script executed successfully")
                logger.debug(f"Blender stdout: {result.stdout}")
                return True
            else:
                logger.error(f"Blender script failed with return code {result.returncode}")
                logger.error(f"Blender stderr: {result.stderr}")
                logger.error(f"Blender stdout: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Blender script timed out after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Error executing Blender script: {e}")
            return False
    
    def render_multiple_poses(
        self,
        poses: List[Dict[str, float]],
        output_dir: Optional[Union[str, Path]] = None,
        resolution: Optional[Tuple[int, int]] = None
    ) -> List[Path]:
        """Render images from multiple camera poses.
        
        Args:
            poses: List of pose dictionaries with keys: yaw, pitch, radius, fov
            output_dir: Directory to save images (use temp if None)
            resolution: Render resolution
            
        Returns:
            List of paths to rendered images
        """
        if output_dir is None:
            output_dir = self.temp_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        rendered_images = []
        
        for i, pose in enumerate(poses):
            try:
                output_path = output_dir / f"pose_{i:03d}_{pose['yaw']:.1f}_{pose['pitch']:.1f}.png"
                
                image_path = self.render_image(
                    yaw=pose['yaw'],
                    pitch=pose['pitch'],
                    radius=pose['radius'],
                    fov=pose.get('fov', 40.0),
                    output_path=output_path,
                    resolution=resolution
                )
                
                rendered_images.append(image_path)
                logger.info(f"Rendered pose {i+1}/{len(poses)}: {image_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to render pose {i}: {e}")
                # Continue with next pose
        
        logger.info(f"Rendered {len(rendered_images)}/{len(poses)} poses")
        return rendered_images
    
    def generate_trellis_poses(
        self,
        num_poses: int = 8,
        radius: float = 2.0,
        fov: float = 40.0,
        pitch_range: Tuple[float, float] = (-30, 30),
        yaw_range: Tuple[float, float] = (0, 360)
    ) -> List[Dict[str, float]]:
        """Generate TRELLIS-compatible camera poses.
        
        Args:
            num_poses: Number of poses to generate
            radius: Camera distance from origin
            fov: Field of view in degrees
            pitch_range: Range of pitch angles (min, max) in degrees
            yaw_range: Range of yaw angles (min, max) in degrees
            
        Returns:
            List of pose dictionaries
        """
        poses = []
        
        # Generate evenly distributed poses
        yaw_step = (yaw_range[1] - yaw_range[0]) / num_poses
        
        for i in range(num_poses):
            # Evenly distribute yaw angles
            yaw = yaw_range[0] + i * yaw_step
            
            # Vary pitch slightly for more interesting views
            pitch = np.random.uniform(pitch_range[0], pitch_range[1])
            
            pose = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'radius': float(radius),
                'fov': float(fov)
            }
            poses.append(pose)
        
        logger.info(f"Generated {len(poses)} TRELLIS-compatible poses")
        return poses
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class TRELLISCameraController:
    """TRELLIS-compatible camera pose controller."""
    
    def __init__(self):
        """Initialize camera controller."""
        self.yaw = 0.0      # degrees
        self.pitch = 0.0    # degrees  
        self.radius = 2.0   # distance units
        self.fov = 40.0     # degrees
        
        # Constraints
        self.min_radius = 0.5
        self.max_radius = 10.0
        self.min_pitch = -90.0
        self.max_pitch = 90.0
    
    def set_spherical_pose(self, yaw: float, pitch: float, radius: float) -> None:
        """Set camera pose using spherical coordinates.
        
        Args:
            yaw: Yaw angle in degrees (0-360)
            pitch: Pitch angle in degrees (-90 to 90)
            radius: Distance from origin (positive)
        """
        self.yaw = self._normalize_yaw(yaw)
        self.pitch = self._clamp_pitch(pitch)
        self.radius = self._clamp_radius(radius)
    
    def set_fov(self, fov_degrees: float) -> None:
        """Set field of view.
        
        Args:
            fov_degrees: Field of view in degrees (10-120)
        """
        self.fov = max(10.0, min(120.0, fov_degrees))
    
    def get_pose_dict(self) -> Dict[str, float]:
        """Get current pose as TRELLIS-format dictionary.
        
        Returns:
            Dictionary with keys: yaw, pitch, radius, fov
        """
        return {
            'yaw': self.yaw,
            'pitch': self.pitch,
            'radius': self.radius,
            'fov': self.fov
        }
    
    def get_cartesian_position(self) -> Tuple[float, float, float]:
        """Convert spherical pose to cartesian coordinates.
        
        Returns:
            (x, y, z) position in 3D space
        """
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        x = self.radius * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.radius * math.sin(pitch_rad)
        z = self.radius * math.cos(pitch_rad) * math.cos(yaw_rad)
        
        return (x, y, z)
    
    def _normalize_yaw(self, yaw: float) -> float:
        """Normalize yaw to 0-360 range."""
        return yaw % 360.0
    
    def _clamp_pitch(self, pitch: float) -> float:
        """Clamp pitch to valid range."""
        return max(self.min_pitch, min(self.max_pitch, pitch))
    
    def _clamp_radius(self, radius: float) -> float:
        """Clamp radius to valid range."""
        return max(self.min_radius, min(self.max_radius, radius))
    
    def validate_pose_constraints(self) -> bool:
        """Validate current pose against constraints.
        
        Returns:
            True if pose is valid, False otherwise
        """
        return (
            self.min_radius <= self.radius <= self.max_radius and
            self.min_pitch <= self.pitch <= self.max_pitch and
            10.0 <= self.fov <= 120.0
        )


# Convenience functions

def render_abo_model(
    model_id: str,
    yaw: float,
    pitch: float,
    radius: float = 2.0,
    fov: float = 40.0,
    output_path: Optional[Union[str, Path]] = None,
    abo_dataset_path: Optional[Path] = None
) -> Path:
    """Convenience function to render an ABO model.
    
    Args:
        model_id: ABO model identifier
        yaw: Camera yaw angle in degrees
        pitch: Camera pitch angle in degrees
        radius: Camera distance from origin
        fov: Field of view in degrees
        output_path: Output image path
        abo_dataset_path: Path to ABO dataset
        
    Returns:
        Path to rendered image
    """
    from .glb_loader import GLBLoader
    
    # Load GLB model
    loader = GLBLoader()
    model = loader.load_abo_model(model_id, abo_dataset_path)
    
    # Render with Blender
    renderer = BlenderRenderer()
    renderer.load_glb_model(model.file_path)
    
    return renderer.render_image(yaw, pitch, radius, fov, output_path)


def validate_blender_installation() -> Dict[str, Any]:
    """Validate Blender installation and capabilities.
    
    Returns:
        Dictionary with validation results
    """
    try:
        renderer = BlenderRenderer()
        return {
            'blender_found': True,
            'blender_path': renderer.blender_executable,
            'version_check': True,
            'can_render': True,
            'error': None
        }
    except Exception as e:
        return {
            'blender_found': False,
            'blender_path': None,
            'version_check': False,
            'can_render': False,
            'error': str(e)
        }