"""Alternative Blender renderer using OBJ format to bypass GLB/numpy issues.

This module converts GLB models to OBJ format for Blender rendering,
avoiding numpy version compatibility issues with Blender's GLB importer.
"""

import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import math

from .blender_renderer import BlenderRenderer, TRELLISCameraController
from .glb_loader import GLBLoader

logger = logging.getLogger(__name__)


class BlenderOBJRenderer(BlenderRenderer):
    """Blender renderer that converts GLB to OBJ for compatibility."""
    
    def __init__(self, blender_executable: Optional[str] = None):
        """Initialize Blender OBJ renderer."""
        super().__init__(blender_executable)
        self.glb_loader = GLBLoader()
        self.converted_obj_path: Optional[Path] = None
        self.converted_mtl_path: Optional[Path] = None
    
    def load_glb_model(self, glb_path: Union[str, Path]) -> bool:
        """Load a GLB model by converting it to OBJ format.
        
        Args:
            glb_path: Path to GLB file
            
        Returns:
            True if successful, False otherwise
        """
        glb_path = Path(glb_path)
        
        if not glb_path.exists():
            logger.error(f"GLB file not found: {glb_path}")
            return False
        
        try:
            # Load GLB with our loader
            logger.info(f"Loading GLB model: {glb_path}")
            model = self.glb_loader.load_model(glb_path)
            
            if model.mesh is None:
                logger.error("GLB model has no geometry")
                return False
            
            # Convert to OBJ format
            obj_path = self.temp_dir / f"{glb_path.stem}.obj"
            mtl_path = self.temp_dir / f"{glb_path.stem}.mtl"
            
            logger.info(f"Converting GLB to OBJ: {obj_path}")
            
            # Export using trimesh
            model.mesh.export(
                obj_path,
                file_type='obj',
                include_normals=True,
                include_texture=True,
                include_color=True
            )
            
            if obj_path.exists():
                self.current_model_path = glb_path  # Keep original for reference
                self.converted_obj_path = obj_path
                self.converted_mtl_path = mtl_path if mtl_path.exists() else None
                
                logger.info(f"Successfully converted GLB to OBJ")
                logger.info(f"OBJ size: {obj_path.stat().st_size / 1024:.1f} KB")
                
                return True
            else:
                logger.error("OBJ conversion failed - file not created")
                return False
                
        except Exception as e:
            logger.error(f"Failed to convert GLB to OBJ: {e}")
            return False
    
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
        """Create Blender Python script for rendering OBJ models.
        
        Args:
            glb_path: Path to original GLB model (ignored, uses converted OBJ)
            yaw: Camera yaw in degrees
            pitch: Camera pitch in degrees
            radius: Camera distance
            fov: Field of view in degrees
            output_path: Output image path
            resolution: Render resolution
            
        Returns:
            Path to created script
        """
        if self.converted_obj_path is None:
            raise RuntimeError("No OBJ model loaded. Call load_glb_model() first.")
        
        # Use the converted OBJ path
        obj_path = self.converted_obj_path
        mtl_path = self.converted_mtl_path
        
        script_content = f'''
import bpy
import bmesh
import mathutils
import math
import sys
import os

print("=" * 50)
print("BLENDER OBJ RENDER SCRIPT")
print("=" * 50)

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Set render engine (robust approach like TRELLIS)
engine = '{self.render_settings["engine"]}'
print(f"Setting render engine: {{engine}}")

# Handle Blender 3.0+ engine names robustly
if engine == 'EEVEE':
    try:
        # Try Blender 3.0+ naming first
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        print("Using BLENDER_EEVEE")
    except TypeError:
        try:
            # Fallback to legacy naming
            bpy.context.scene.render.engine = 'EEVEE' 
            print("Using EEVEE")
        except TypeError:
            # Final fallback to CYCLES
            bpy.context.scene.render.engine = 'CYCLES'
            print("Fallback to CYCLES")
elif engine == 'CYCLES':
    bpy.context.scene.render.engine = 'CYCLES'
    print("Using CYCLES")
else:
    # Try direct assignment, fallback to CYCLES
    try:
        bpy.context.scene.render.engine = engine
        print(f"Using {{engine}}")
    except TypeError:
        bpy.context.scene.render.engine = 'CYCLES'
        print("Fallback to CYCLES")

# Configure Cycles settings if using Cycles
if engine == 'CYCLES':
    bpy.context.scene.cycles.samples = {self.render_settings["samples"]}
    bpy.context.scene.cycles.use_denoising = {self.render_settings["use_denoising"]}

# Import OBJ model
obj_path = r"{str(obj_path)}"
print(f"Importing OBJ: {{obj_path}}")

try:
    bpy.ops.import_scene.obj(filepath=obj_path)
    print("OBJ import successful")
except Exception as e:
    print(f"OBJ import failed: {{e}}")
    # Try alternative import
    bpy.ops.wm.obj_import(filepath=obj_path)

# Get imported objects
imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

if not imported_objects:
    print("ERROR: No mesh objects imported from OBJ")
    sys.exit(1)

print(f"Imported {{len(imported_objects)}} mesh objects")

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

# Scale model to fit in unit sphere (optional)
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

# Add fill light
fill_light_data = bpy.data.lights.new(name="FillLight", type='AREA')
fill_light_data.energy = 1.0
fill_light_data.size = 5.0
fill_light_obj = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
bpy.context.collection.objects.link(fill_light_obj)
fill_light_obj.location = (-2, -1, 1)

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
print("=" * 50)
'''
        
        # Save script to temporary file
        script_path = self.temp_dir / "obj_render_script.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def cleanup(self) -> None:
        """Clean up temporary files including converted OBJ."""
        # Clean up converted files
        if self.converted_obj_path and self.converted_obj_path.exists():
            try:
                self.converted_obj_path.unlink()
            except:
                pass
                
        if self.converted_mtl_path and self.converted_mtl_path.exists():
            try:
                self.converted_mtl_path.unlink()
            except:
                pass
        
        # Call parent cleanup
        super().cleanup()


# Convenience function
def render_abo_model_via_obj(
    model_id: str,
    yaw: float,
    pitch: float,
    radius: float = 2.0,
    fov: float = 40.0,
    output_path: Optional[Union[str, Path]] = None,
    abo_dataset_path: Optional[Path] = None
) -> Path:
    """Render ABO model using OBJ conversion.
    
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
    
    # Render with OBJ conversion
    renderer = BlenderOBJRenderer()
    renderer.load_glb_model(model.file_path)
    
    return renderer.render_image(yaw, pitch, radius, fov, output_path)