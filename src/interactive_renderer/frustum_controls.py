#!/usr/bin/env python3
"""Interactive 3D Camera Frustum Controls - Direct 3D manipulation interface.

This module provides a viser-based interactive interface for manipulating
camera poses through direct 3D frustum visualization and drag-and-drop control.
"""

import viser
import numpy as np
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging

from .trellis_camera import TRELLISCameraSystem, CameraPose
from .glb_loader import GLBLoader
from .blender_pipeline import BlenderRenderingPipeline, RenderQuality

logger = logging.getLogger(__name__)


class CameraFrustum:
    """3D Camera frustum representation with interactive handles."""
    
    def __init__(self, pose: CameraPose, near: float = 0.1, far: float = 1.5):
        """Initialize camera frustum.
        
        Args:
            pose: Camera pose (TRELLIS format)
            near: Near plane distance  
            far: Far plane distance (fixed size, independent of radius)
        """
        self.pose = pose
        self.near = near
        self.far = far  # Fixed frustum depth, not dependent on camera distance
        
        # Frustum geometry
        self.vertices = None
        self.lines = None
        self._compute_frustum_geometry()
    
    def _compute_frustum_geometry(self):
        """Compute frustum vertices and line connections."""
        # Camera position
        cam_pos = np.array(self.pose.to_cartesian())
        
        # Camera forward direction (looking at origin)
        forward = np.array([0.0, 0.0, 0.0]) - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Camera up and right vectors
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # FOV calculations
        fov_rad = np.radians(self.pose.fov)
        aspect_ratio = 1.0  # Square aspect ratio
        
        # Near plane dimensions
        near_height = 2 * self.near * np.tan(fov_rad / 2)
        near_width = near_height * aspect_ratio
        
        # Far plane dimensions
        far_height = 2 * self.far * np.tan(fov_rad / 2)
        far_width = far_height * aspect_ratio
        
        # Near plane corners
        near_center = cam_pos + forward * self.near
        near_tl = near_center + up * (near_height/2) - right * (near_width/2)
        near_tr = near_center + up * (near_height/2) + right * (near_width/2)
        near_bl = near_center - up * (near_height/2) - right * (near_width/2)
        near_br = near_center - up * (near_height/2) + right * (near_width/2)
        
        # Far plane corners
        far_center = cam_pos + forward * self.far
        far_tl = far_center + up * (far_height/2) - right * (far_width/2)
        far_tr = far_center + up * (far_height/2) + right * (far_width/2)
        far_bl = far_center - up * (far_height/2) - right * (far_width/2)
        far_br = far_center - up * (far_height/2) + right * (far_width/2)
        
        # Store vertices
        self.vertices = {
            'camera': cam_pos,
            'near_center': near_center,
            'far_center': far_center,
            'near_tl': near_tl, 'near_tr': near_tr,
            'near_bl': near_bl, 'near_br': near_br,
            'far_tl': far_tl, 'far_tr': far_tr,
            'far_bl': far_bl, 'far_br': far_br
        }
        
        # Define line connections
        self.lines = [
            # Near plane rectangle
            ('near_tl', 'near_tr'), ('near_tr', 'near_br'),
            ('near_br', 'near_bl'), ('near_bl', 'near_tl'),
            # Far plane rectangle
            ('far_tl', 'far_tr'), ('far_tr', 'far_br'),
            ('far_br', 'far_bl'), ('far_bl', 'far_tl'),
            # Connecting lines (frustum edges)
            ('near_tl', 'far_tl'), ('near_tr', 'far_tr'),
            ('near_bl', 'far_bl'), ('near_br', 'far_br'),
            # Camera to near plane center (viewing direction)
            ('camera', 'near_center')
        ]
    
    def get_frustum_lines(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get frustum as list of line segments.
        
        Returns:
            List of (start_point, end_point) tuples
        """
        lines = []
        for start_key, end_key in self.lines:
            start_pos = self.vertices[start_key]
            end_pos = self.vertices[end_key]
            lines.append((start_pos, end_pos))
        return lines
    
    def update_pose(self, new_pose: CameraPose):
        """Update frustum with new camera pose."""
        self.pose = new_pose
        # Keep far plane fixed - don't change with radius
        self._compute_frustum_geometry()


class InteractiveFrustumControls:
    """Interactive 3D camera frustum controls with direct manipulation."""
    
    def __init__(self, server_port: int = 8080):
        """Initialize interactive frustum controls.
        
        Args:
            server_port: Port for viser server
        """
        # Initialize viser server
        self.server = viser.ViserServer(port=server_port)
        
        # Initialize TRELLIS camera system
        self.camera_system = TRELLISCameraSystem(default_radius=2.5, default_fov=40.0)
        
        # Camera frustum visualization
        self.frustum = CameraFrustum(self.camera_system.get_pose())
        
        # Model and rendering state
        self.current_model_path: Optional[Path] = None
        self.model_mesh = None
        self.render_callback: Optional[Callable] = None
        
        # Blender rendering pipeline
        self.blender_pipeline: Optional[BlenderRenderingPipeline] = None
        self.current_rendered_image: Optional[Path] = None
        self.enable_auto_render = True
        
        # Interaction state
        self._dragging = False
        self._drag_start_pos = None
        self._drag_handle = None
        self._last_render_time = 0
        self._render_cooldown = 0.5
        
        # Setup UI and 3D scene
        self._setup_ui()
        self._setup_3d_scene()
        
        # Initialize Blender rendering pipeline  
        self._setup_blender_pipeline()
        
        logger.info(f"Interactive frustum controls started on port {server_port}")
    
    def _setup_ui(self):
        """Set up viser UI components."""
        
        # Model loading section
        with self.server.gui.add_folder("📁 Model Loading", expand_by_default=True):
            self.model_path_input = self.server.gui.add_text(
                "Model Path",
                initial_value="dataset/ABO/raw/3dmodels/original/0/B01LR5RSG0.glb"
            )
            
            self.load_model_button = self.server.gui.add_button("Load GLB Model")
            self.load_model_button.on_click(self._load_model)
            
            self.model_status = self.server.gui.add_text(
                "Status",
                initial_value="No model loaded",
                disabled=True
            )
        
        # Camera information display
        with self.server.gui.add_folder("🎥 Camera Information", expand_by_default=True):
            self.pose_display = self.server.gui.add_text(
                "Current Pose",
                initial_value=self._format_pose_display(),
                disabled=True
            )
            
            self.position_display = self.server.gui.add_text(
                "3D Position",
                initial_value=self._format_position_display(),
                disabled=True
            )
        
        # 3D Manipulation instructions
        with self.server.gui.add_folder("🎮 3D Controls", expand_by_default=True):
            self.server.gui.add_markdown(
                """
**Direct 3D Frustum Manipulation:**
- 🖱️ **Drag Camera**: Click and drag the camera position (red sphere)
- 📐 **Drag Frustum**: Click and drag frustum corners to adjust FOV
- 🎯 **Drag Target**: Click and drag the target point (where camera looks)
- 🔄 **Orbit**: Right-click drag to orbit around the model
                """
            )
            
            # Quick control buttons
            self.reset_button = self.server.gui.add_button("Reset Camera")
            self.reset_button.on_click(self._reset_camera)
            
            self.standard_views_button = self.server.gui.add_button("Cycle Standard Views")
            self.standard_views_button.on_click(self._cycle_standard_views)
        
        # Blender rendering controls
        with self.server.gui.add_folder("🎨 Blender Rendering", expand_by_default=True):
            
            self.auto_render_checkbox = self.server.gui.add_checkbox(
                "Auto Render on Camera Move", 
                initial_value=True
            )
            self.auto_render_checkbox.on_update(self._on_auto_render_change)
            
            self.render_quality_dropdown = self.server.gui.add_dropdown(
                "Render Quality",
                options=["PREVIEW", "STANDARD", "HIGH"],
                initial_value="PREVIEW"
            )
            
            self.render_resolution_dropdown = self.server.gui.add_dropdown(
                "Resolution", 
                options=["256x256", "512x512", "1024x1024"],
                initial_value="512x512"
            )
            
            self.render_button = self.server.gui.add_button("🎨 Render Current View")
            self.render_button.on_click(self._manual_render)
            
            self.render_status = self.server.gui.add_text(
                "Render Status",
                initial_value="Ready for Blender rendering",
                disabled=True
            )
            
            self.last_render_info = self.server.gui.add_text(
                "Last Render",
                initial_value="No renders yet",
                disabled=True
            )
        
        # Export controls  
        with self.server.gui.add_folder("💾 Export & Save"):
            self.export_pose_button = self.server.gui.add_button("Export Current Pose")
            self.export_pose_button.on_click(self._export_current_pose)
        
        # Visual settings
        with self.server.gui.add_folder("👁️ Visualization"):
            self.show_frustum_checkbox = self.server.gui.add_checkbox(
                "Show Camera Frustum",
                initial_value=True
            )
            self.show_frustum_checkbox.on_update(self._on_visual_change)
            
            self.show_target_checkbox = self.server.gui.add_checkbox(
                "Show Target Point",
                initial_value=True
            )
            self.show_target_checkbox.on_update(self._on_visual_change)
            
            self.frustum_opacity_slider = self.server.gui.add_slider(
                "Frustum Opacity",
                min=0.1, max=1.0, step=0.1,
                initial_value=0.3
            )
            self.frustum_opacity_slider.on_update(self._on_visual_change)
    
    def _setup_3d_scene(self):
        """Set up interactive 3D scene with frustum visualization."""
        # Add coordinate axes for reference
        self._add_coordinate_axes()
        
        # Add camera frustum visualization
        self._update_frustum_visualization()
        
        # Set up interaction handlers
        self._setup_interaction_handlers()
    
    def _setup_blender_pipeline(self):
        """Initialize Blender rendering pipeline."""
        try:
            self.blender_pipeline = BlenderRenderingPipeline()
            logger.info("✅ Blender rendering pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Blender pipeline: {e}")
            self.blender_pipeline = None
    
    def _add_coordinate_axes(self):
        """Add coordinate axes to the scene for reference."""
        axis_length = 1.0
        
        # X axis (red)
        self.server.scene.add_spline_catmull_rom(
            name="axis_x",
            positions=np.array([[0, 0, 0], [axis_length, 0, 0]]),
            color=(1.0, 0.0, 0.0),
            line_width=3.0
        )
        
        # Y axis (green) 
        self.server.scene.add_spline_catmull_rom(
            name="axis_y",
            positions=np.array([[0, 0, 0], [0, axis_length, 0]]),
            color=(0.0, 1.0, 0.0),
            line_width=3.0
        )
        
        # Z axis (blue)
        self.server.scene.add_spline_catmull_rom(
            name="axis_z",
            positions=np.array([[0, 0, 0], [0, 0, axis_length]]),
            color=(0.0, 0.0, 1.0),
            line_width=3.0
        )
    
    def _update_frustum_visualization(self):
        """Update 3D frustum visualization in the scene."""
        if not self.show_frustum_checkbox.value:
            self._clear_frustum_visualization()
            return
        
        # Update frustum geometry
        self.frustum.update_pose(self.camera_system.get_pose())
        
        # Camera position with translation control
        cam_pos = self.frustum.vertices['camera']
        
        # Create translation-only transform control (no scaling/rotation)
        self._camera_translate_handle = self.server.scene.add_transform_controls(
            name="camera_translate",
            position=cam_pos,
            scale=0.5,
            disable_axes=False,
            disable_sliders=True,
            disable_rotations=True,  # Disable rotation handles
        )
        
        # Create orientation control at target point
        target_pos = np.array([0.0, 0.0, 0.0])
        self._camera_look_handle = self.server.scene.add_transform_controls(
            name="camera_look_target", 
            position=target_pos,
            scale=0.3,
            disable_axes=False,
            disable_sliders=True,
            disable_rotations=True,
        )
        
        # Visual indicators
        self.server.scene.add_icosphere(
            name="camera_position",
            radius=0.06,
            color=(1.0, 0.2, 0.2),
            position=cam_pos
        )
        
        # Set up interaction after creating handles
        self._setup_camera_interaction()
        
        # Target point (draggable green sphere)
        if self.show_target_checkbox.value:
            target_pos = np.array([0.0, 0.0, 0.0])  # Model center
            self.server.scene.add_icosphere(
                name="camera_target",
                radius=0.05,
                color=(0.2, 1.0, 0.2),
                position=target_pos
            )
        
        # Frustum wireframe
        frustum_lines = self.frustum.get_frustum_lines()
        
        for i, (start_pos, end_pos) in enumerate(frustum_lines):
            line_name = f"frustum_line_{i}"
            
            # Different colors for different parts
            if i < 4:  # Near plane
                color = (0.8, 0.8, 1.0)
                width = 2.0
            elif i < 8:  # Far plane
                color = (0.6, 0.6, 1.0)
                width = 2.0
            elif i < 12:  # Connecting edges
                color = (0.4, 0.4, 1.0)
                width = 1.5
            else:  # Viewing direction
                color = (1.0, 1.0, 0.4)
                width = 3.0
            
            self.server.scene.add_spline_catmull_rom(
                name=line_name,
                positions=np.array([start_pos, end_pos]),
                color=color,
                line_width=width
            )
        
        # Frustum fill planes (optional)
        self._add_frustum_planes()
        
        # Add rendered image display if available
        self._add_rendered_image_display()
    
    def _add_frustum_planes(self):
        """Add semi-transparent planes to show frustum volume."""
        vertices = self.frustum.vertices
        opacity = self.frustum_opacity_slider.value * 0.3  # Lower opacity for planes
        
        # Near plane
        near_vertices = np.array([
            vertices['near_tl'], vertices['near_tr'],
            vertices['near_br'], vertices['near_bl']
        ])
        near_faces = np.array([[0, 1, 2], [0, 2, 3]])
        
        self.server.scene.add_mesh_simple(
            name="frustum_near_plane",
            vertices=near_vertices,
            faces=near_faces,
            color=(0.8, 0.8, 1.0),
            opacity=opacity
        )
        
        # Far plane
        far_vertices = np.array([
            vertices['far_tl'], vertices['far_tr'],
            vertices['far_br'], vertices['far_bl']
        ])
        far_faces = np.array([[0, 1, 2], [0, 2, 3]])
        
        self.server.scene.add_mesh_simple(
            name="frustum_far_plane",
            vertices=far_vertices,
            faces=far_faces,
            color=(0.6, 0.6, 1.0),
            opacity=opacity
        )
    
    def _add_rendered_image_display(self):
        """Add rendered image display at the frustum position."""
        if not self.current_rendered_image or not self.current_rendered_image.exists():
            return
        
        try:
            # Position the image plane at the far plane of the frustum
            vertices = self.frustum.vertices
            
            # Create a plane at the far plane position
            far_center = vertices['far_center']
            far_tl = vertices['far_tl']
            far_tr = vertices['far_tr'] 
            far_bl = vertices['far_bl']
            far_br = vertices['far_br']
            
            # Create image plane vertices
            image_vertices = np.array([far_tl, far_tr, far_br, far_bl])
            image_faces = np.array([[0, 1, 2], [0, 2, 3]])
            
            # Add image plane to scene
            self.server.scene.add_mesh_simple(
                name="rendered_image_plane",
                vertices=image_vertices,
                faces=image_faces,
                color=(1.0, 1.0, 1.0),
                opacity=0.9
            )
            
            logger.debug(f"Added rendered image plane at far frustum")
            
        except Exception as e:
            logger.error(f"Failed to add rendered image display: {e}")
    
    def _clear_frustum_visualization(self):
        """Clear frustum visualization from scene."""
        # Remove all frustum-related objects
        frustum_objects = [
            "camera_position", "camera_target", "camera_translate", "camera_look_target",
            "frustum_near_plane", "frustum_far_plane", "rendered_image_plane"
        ]
        
        for obj_name in frustum_objects:
            try:
                self.server.scene.remove(obj_name)
            except:
                pass
        
        # Clear camera handle references
        if hasattr(self, '_camera_translate_handle'):
            delattr(self, '_camera_translate_handle')
        if hasattr(self, '_camera_look_handle'):
            delattr(self, '_camera_look_handle')
        
        # Remove frustum lines
        for i in range(20):  # Remove up to 20 lines
            try:
                self.server.scene.remove(f"frustum_line_{i}")
            except:
                pass
    
    def _setup_interaction_handlers(self):
        """Set up interaction handlers for 3D manipulation."""
        # This will be set up after the camera handle is created
        pass
    
    def _setup_camera_interaction(self):
        """Set up camera transform interaction after handle creation."""
        # Handle camera position changes (translation)
        if hasattr(self, '_camera_translate_handle'):
            @self._camera_translate_handle.on_update
            def _(_):
                # Get new camera position
                new_pos = self._camera_translate_handle.position
                
                # Update camera system - calculate new spherical coordinates
                # but maintain the look-at relationship
                target_pos = self._camera_look_handle.position if hasattr(self, '_camera_look_handle') else np.array([0.0, 0.0, 0.0])
                
                # Calculate relative position (camera relative to target)
                rel_pos = new_pos - target_pos
                
                # Convert to spherical coordinates relative to target
                self.camera_system.set_cartesian(rel_pos[0], rel_pos[1], rel_pos[2])
                
                # Update visual elements
                self._update_displays()
                
                # Trigger auto-render if enabled
                if self.enable_auto_render and self.auto_render_checkbox.value:
                    self._throttled_blender_render()
                
                logger.info(f"Camera moved to: ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})")
        
        # Handle look-at target changes (what camera points towards)
        if hasattr(self, '_camera_look_handle'):
            @self._camera_look_handle.on_update
            def _(_):
                # Get new target position
                target_pos = self._camera_look_handle.position
                camera_pos = self._camera_translate_handle.position if hasattr(self, '_camera_translate_handle') else np.array([0.0, 0.0, 2.5])
                
                # Calculate new relative position
                rel_pos = camera_pos - target_pos
                
                # Update camera system to look at new target
                self.camera_system.set_cartesian(rel_pos[0], rel_pos[1], rel_pos[2])
                
                # Update visual elements  
                self._update_displays()
                
                # Trigger auto-render if enabled
                if self.enable_auto_render and self.auto_render_checkbox.value:
                    self._throttled_blender_render()
                
                logger.info(f"Camera target moved to: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
    
    def _load_model(self, _):
        """Load GLB model."""
        try:
            model_path = Path(self.model_path_input.value)
            
            if not model_path.exists():
                self.model_status.value = f"❌ File not found: {model_path}"
                return
            
            # Load GLB model
            loader = GLBLoader()
            model = loader.load_model(model_path)
            
            self.current_model_path = model_path
            self.model_mesh = model.mesh
            
            # Add model to viser scene
            self._update_model_display()
            
            self.model_status.value = f"✅ Loaded: {model_path.name} ({len(model.mesh.vertices)} vertices)"
            
            # Reset camera to good default view
            self._reset_camera()
            
            logger.info(f"Loaded model: {model_path}")
            
        except Exception as e:
            self.model_status.value = f"❌ Error: {str(e)}"
            logger.error(f"Failed to load model: {e}")
    
    def _update_model_display(self):
        """Update model display in viser scene."""
        if self.model_mesh is not None:
            # Add mesh to viser scene
            vertices = np.array(self.model_mesh.vertices)
            faces = np.array(self.model_mesh.faces)
            
            # Center and scale mesh
            center = vertices.mean(axis=0)
            vertices = vertices - center
            scale = 2.0 / np.max(np.abs(vertices))
            vertices = vertices * scale
            
            # Add to scene
            self.server.scene.add_mesh_simple(
                name="model",
                vertices=vertices,
                faces=faces,
                color=(0.7, 0.8, 0.9),
                opacity=0.8
            )
    
    def _format_pose_display(self) -> str:
        """Format pose information for display."""
        pose = self.camera_system.get_pose()
        return f"yaw={pose.yaw:.1f}°, pitch={pose.pitch:.1f}°, radius={pose.radius:.2f}, fov={pose.fov:.1f}°"
    
    def _format_position_display(self) -> str:
        """Format 3D position information for display."""
        x, y, z = self.camera_system.get_cartesian()
        return f"position=({x:.2f}, {y:.2f}, {z:.2f})"
    
    def _update_displays(self):
        """Update all information displays."""
        self.pose_display.value = self._format_pose_display()
        self.position_display.value = self._format_position_display()
        self._update_frustum_visualization()
    
    def _reset_camera(self, _=None):
        """Reset camera to default pose."""
        self.camera_system.reset_to_default()
        self._update_displays()
        
        if self.render_callback:
            self._throttled_render()
    
    def _cycle_standard_views(self, _):
        """Cycle through standard camera views."""
        # Implement cycling through standard views
        standard_poses = [
            CameraPose(0.0, 0.0, 2.5, 40.0),      # Front
            CameraPose(90.0, 0.0, 2.5, 40.0),     # Right
            CameraPose(180.0, 0.0, 2.5, 40.0),    # Back
            CameraPose(270.0, 0.0, 2.5, 40.0),    # Left
            CameraPose(45.0, 30.0, 2.5, 40.0),    # Top-right
        ]
        
        # Simple cycling (can be made more sophisticated)
        if not hasattr(self, '_view_index'):
            self._view_index = 0
        
        self.camera_system.current_pose = standard_poses[self._view_index]
        self._view_index = (self._view_index + 1) % len(standard_poses)
        
        self._update_displays()
        
        if self.render_callback:
            self._throttled_render()
    
    def _export_current_pose(self, _):
        """Export current camera pose to JSON."""
        try:
            output_path = Path("current_camera_pose.json")
            current_pose = self.camera_system.get_pose()
            self.camera_system.save_poses([current_pose], output_path)
            
            self.render_status.value = f"✅ Exported: {output_path}"
            logger.info(f"Exported current pose to {output_path}")
            
        except Exception as e:
            self.render_status.value = f"❌ Export failed: {str(e)}"
            logger.error(f"Failed to export pose: {e}")
    
    def _manual_render(self, _):
        """Trigger manual Blender render."""
        if not self.current_model_path:
            self.render_status.value = "❌ No model loaded"
            return
            
        if not self.blender_pipeline:
            self.render_status.value = "❌ Blender pipeline not available"
            return
            
        self._trigger_blender_render()
    
    def _on_auto_render_change(self, _):
        """Handle auto render checkbox change."""
        self.enable_auto_render = self.auto_render_checkbox.value
        if self.enable_auto_render:
            logger.info("Auto-rendering enabled")
        else:
            logger.info("Auto-rendering disabled")
    
    def _throttled_render(self):
        """Throttled rendering to avoid too frequent updates."""
        current_time = time.time()
        if current_time - self._last_render_time > self._render_cooldown:
            self._trigger_render()
            self._last_render_time = current_time
    
    def _trigger_render(self):
        """Trigger rendering with current camera pose."""
        if self.render_callback and self.current_model_path:
            pose = self.camera_system.get_pose()
            
            # Update render status
            self.render_status.value = f"🔄 Rendering..."
            
            # Trigger render in background thread
            threading.Thread(
                target=self._background_render,
                args=(pose,),
                daemon=True
            ).start()
    
    def _background_render(self, pose: CameraPose):
        """Background rendering to avoid blocking UI."""
        try:
            if self.render_callback:
                result = self.render_callback(self.current_model_path, pose)
                
                if result:
                    self.render_status.value = f"✅ Rendered: {result.name}"
                else:
                    self.render_status.value = f"❌ Render failed"
                    
        except Exception as e:
            self.render_status.value = f"❌ Render error: {str(e)}"
            logger.error(f"Background render failed: {e}")
    
    def _throttled_blender_render(self):
        """Throttled Blender rendering to avoid too frequent updates."""
        current_time = time.time()
        if current_time - self._last_render_time > self._render_cooldown:
            self._trigger_blender_render()
            self._last_render_time = current_time
    
    def _trigger_blender_render(self):
        """Trigger Blender rendering with current camera pose."""
        if not self.blender_pipeline or not self.current_model_path:
            return
            
        pose = self.camera_system.get_pose()
        
        # Update render status
        self.render_status.value = f"🎨 Blender rendering..."
        
        # Trigger render in background thread
        threading.Thread(
            target=self._background_blender_render,
            args=(pose,),
            daemon=True
        ).start()
    
    def _background_blender_render(self, pose: CameraPose):
        """Background Blender rendering to avoid blocking UI."""
        try:
            if not self.blender_pipeline:
                return
                
            # Create render config based on UI settings
            quality_map = {
                "PREVIEW": RenderQuality.PREVIEW,
                "STANDARD": RenderQuality.STANDARD, 
                "HIGH": RenderQuality.HIGH
            }
            quality = quality_map.get(self.render_quality_dropdown.value, RenderQuality.PREVIEW)
            
            # Parse resolution
            res_str = self.render_resolution_dropdown.value
            width, height = map(int, res_str.split('x'))
            
            config = self.blender_pipeline.create_render_config(
                quality=quality,
                resolution=(width, height)
            )
            
            # Create output path
            timestamp = int(time.time() * 1000)
            output_path = Path(f"/tmp/blender_render_{pose.yaw:.0f}_{pose.pitch:.0f}_{timestamp}.png")
            
            logger.info(f"🎨 Starting Blender render: {quality.value} @ {width}x{height}")
            
            # Execute render
            result = self.blender_pipeline.render_single(
                self.current_model_path, 
                pose, 
                output_path, 
                config
            )
            
            if result.success and result.output_path:
                self.current_rendered_image = result.output_path
                self.render_status.value = f"✅ Rendered in {result.render_time:.1f}s"
                self.last_render_info.value = f"📸 {result.output_path.name} ({width}x{height})"
                
                # Update frustum visualization to show rendered image
                self._update_displays()
                
                logger.info(f"✅ Blender render complete: {result.output_path.name}")
                
            else:
                self.render_status.value = f"❌ Render failed"
                self.last_render_info.value = f"Failed: {result.error_message or 'Unknown error'}"
                logger.error(f"❌ Blender render failed: {result.error_message}")
                
        except Exception as e:
            self.render_status.value = f"❌ Render error: {str(e)}"
            self.last_render_info.value = f"Error: {str(e)}"
            logger.error(f"Background Blender render failed: {e}")
    
    def _on_visual_change(self, _):
        """Handle visual setting changes."""
        self._update_frustum_visualization()
    
    def set_render_callback(self, callback: Callable[[Path, CameraPose], Optional[Path]]):
        """Set callback function for rendering.
        
        Args:
            callback: Function that takes (model_path, pose) and returns output path
        """
        self.render_callback = callback
        logger.info("Render callback set")
    
    def get_current_pose(self) -> CameraPose:
        """Get current camera pose."""
        return self.camera_system.get_pose()
    
    def set_pose(self, pose: CameraPose):
        """Set camera pose programmatically."""
        self.camera_system.current_pose = pose
        self._update_displays()
    
    def move_camera_to_position(self, position: Tuple[float, float, float]):
        """Move camera to specific 3D position (looking at origin)."""
        x, y, z = position
        self.camera_system.set_cartesian(x, y, z)
        self._update_displays()
        
        if self.render_callback:
            self._throttled_render()
    
    def run(self):
        """Run the interactive interface."""
        try:
            logger.info("Interactive frustum controls running. Press Ctrl+C to stop.")
            logger.info("Use your browser to interact with the 3D camera frustum!")
            
            # Keep server running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Interactive frustum controls stopped")
        except Exception as e:
            logger.error(f"Error running interactive controls: {e}")
        finally:
            self.server.stop()


# Convenience function for quick setup
def launch_frustum_controls(model_path: Optional[str] = None,
                           port: int = 8080,
                           render_callback: Optional[Callable] = None) -> InteractiveFrustumControls:
    """Launch interactive 3D frustum controls.
    
    Args:
        model_path: Optional initial model path
        port: Server port
        render_callback: Optional render function
        
    Returns:
        InteractiveFrustumControls instance
    """
    controls = InteractiveFrustumControls(server_port=port)
    
    if render_callback:
        controls.set_render_callback(render_callback)
    
    if model_path:
        controls.model_path_input.value = model_path
        controls._load_model(None)
    
    return controls