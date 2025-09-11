#!/usr/bin/env python3
"""Interactive camera controls using viser for real-time pose manipulation.

This module provides a viser-based interactive interface for manipulating
TRELLIS camera poses with real-time preview and export capabilities.
"""

import viser
import numpy as np
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import logging

from .trellis_camera import TRELLISCameraSystem, CameraPose, generate_standard_views
from .glb_loader import GLBLoader

logger = logging.getLogger(__name__)


class InteractiveCameraControls:
    """Interactive camera controls with viser UI."""
    
    def __init__(self, server_port: int = 8080):
        """Initialize interactive camera controls.
        
        Args:
            server_port: Port for viser server
        """
        # Initialize viser server
        self.server = viser.ViserServer(port=server_port)
        
        # Initialize TRELLIS camera system
        self.camera_system = TRELLISCameraSystem(default_radius=2.5, default_fov=40.0)
        
        # Model and rendering state
        self.current_model_path: Optional[Path] = None
        self.model_mesh = None
        self.render_callback: Optional[Callable] = None
        
        # UI state
        self._pose_preview_enabled = True
        self._auto_render = True
        self._last_render_time = 0
        self._render_cooldown = 0.5  # seconds
        
        # Setup UI
        self._setup_ui()
        
        logger.info(f"Interactive camera controls started on port {server_port}")
    
    def _setup_ui(self):
        """Set up viser UI components."""
        
        # Model loading section
        with self.server.gui.add_folder("📁 Model Loading"):
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
        
        # Camera pose controls
        with self.server.gui.add_folder("🎥 Camera Pose Controls", expand_by_default=True):
            
            # Current pose display
            self.pose_display = self.server.gui.add_text(
                "Current Pose",
                initial_value="yaw=0.0°, pitch=0.0°, radius=2.5, fov=40.0°",
                disabled=True
            )
            
            # Spherical coordinate controls
            self.yaw_slider = self.server.gui.add_slider(
                "Yaw (°)", 
                min=0.0, max=360.0, step=1.0, 
                initial_value=0.0
            )
            
            self.pitch_slider = self.server.gui.add_slider(
                "Pitch (°)", 
                min=-89.0, max=89.0, step=1.0, 
                initial_value=0.0
            )
            
            self.radius_slider = self.server.gui.add_slider(
                "Radius", 
                min=0.5, max=10.0, step=0.1, 
                initial_value=2.5
            )
            
            self.fov_slider = self.server.gui.add_slider(
                "FOV (°)", 
                min=10.0, max=120.0, step=1.0, 
                initial_value=40.0
            )
            
            # Bind slider callbacks
            self.yaw_slider.on_update(self._on_pose_change)
            self.pitch_slider.on_update(self._on_pose_change)
            self.radius_slider.on_update(self._on_pose_change)
            self.fov_slider.on_update(self._on_pose_change)
        
        # Quick controls
        with self.server.gui.add_folder("⚡ Quick Controls"):
            
            # Standard views
            with self.server.gui.add_folder("Standard Views"):
                views_data = [
                    ("Front", 0.0, 0.0),
                    ("Right", 90.0, 0.0),
                    ("Back", 180.0, 0.0),
                    ("Left", 270.0, 0.0),
                    ("Top", 45.0, 45.0),
                    ("Bottom", 45.0, -30.0),
                ]
                
                for name, yaw, pitch in views_data:
                    button = self.server.gui.add_button(name)
                    button.on_click(lambda yaw=yaw, pitch=pitch: self._set_standard_view(yaw, pitch))
            
            # Orbit controls
            with self.server.gui.add_folder("Orbit Controls"):
                orbit_yaw_button = self.server.gui.add_button("Orbit +45° Yaw")
                orbit_yaw_button.on_click(lambda: self._orbit_camera(45.0, 0.0))
                
                orbit_pitch_up_button = self.server.gui.add_button("Orbit +15° Pitch")
                orbit_pitch_up_button.on_click(lambda: self._orbit_camera(0.0, 15.0))
                
                orbit_pitch_down_button = self.server.gui.add_button("Orbit -15° Pitch")
                orbit_pitch_down_button.on_click(lambda: self._orbit_camera(0.0, -15.0))
            
            # Zoom controls
            with self.server.gui.add_folder("Zoom Controls"):
                zoom_in_button = self.server.gui.add_button("Zoom In")
                zoom_in_button.on_click(lambda: self._zoom_camera(-0.5))
                
                zoom_out_button = self.server.gui.add_button("Zoom Out")
                zoom_out_button.on_click(lambda: self._zoom_camera(0.5))
        
        # Camera management
        with self.server.gui.add_folder("💾 Camera Management"):
            
            # Pose history
            self.undo_button = self.server.gui.add_button("Undo")
            self.undo_button.on_click(self._undo_pose)
            
            self.redo_button = self.server.gui.add_button("Redo")
            self.redo_button.on_click(self._redo_pose)
            
            self.reset_button = self.server.gui.add_button("Reset to Default")
            self.reset_button.on_click(self._reset_camera)
            
            # Pose export/import
            self.export_pose_button = self.server.gui.add_button("Export Current Pose")
            self.export_pose_button.on_click(self._export_current_pose)
            
            self.export_standard_button = self.server.gui.add_button("Export Standard Views")
            self.export_standard_button.on_click(self._export_standard_views)
        
        # Rendering controls
        with self.server.gui.add_folder("🖼️ Rendering Controls"):
            
            self.auto_render_checkbox = self.server.gui.add_checkbox(
                "Auto Render", 
                initial_value=True
            )
            self.auto_render_checkbox.on_update(self._on_auto_render_change)
            
            self.render_button = self.server.gui.add_button("Manual Render")
            self.render_button.on_click(self._manual_render)
            
            self.render_status = self.server.gui.add_text(
                "Render Status",
                initial_value="Ready",
                disabled=True
            )
        
        # Preview controls
        with self.server.gui.add_folder("👁️ Preview Options"):
            
            self.show_camera_checkbox = self.server.gui.add_checkbox(
                "Show Camera Position",
                initial_value=True
            )
            self.show_camera_checkbox.on_update(self._on_preview_change)
            
            self.show_frustum_checkbox = self.server.gui.add_checkbox(
                "Show Camera Frustum",
                initial_value=True
            )
            self.show_frustum_checkbox.on_update(self._on_preview_change)
    
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
            
            # Reset camera to default view
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
                color=(0.8, 0.8, 0.9)
            )
    
    def _on_pose_change(self, _):
        """Handle camera pose changes from sliders."""
        # Update camera system
        self.camera_system.set_pose(
            yaw=self.yaw_slider.value,
            pitch=self.pitch_slider.value,
            radius=self.radius_slider.value,
            fov=self.fov_slider.value
        )
        
        # Update pose display
        pose = self.camera_system.get_pose()
        self.pose_display.value = f"yaw={pose.yaw:.1f}°, pitch={pose.pitch:.1f}°, radius={pose.radius:.2f}, fov={pose.fov:.1f}°"
        
        # Update camera preview
        self._update_camera_preview()
        
        # Auto render if enabled
        if self._auto_render:
            self._throttled_render()
    
    def _set_standard_view(self, yaw: float, pitch: float):
        """Set camera to standard view."""
        self.camera_system.set_pose(yaw, pitch, self.radius_slider.value, self.fov_slider.value)
        self._sync_sliders_to_camera()
        self._update_camera_preview()
        
        if self._auto_render:
            self._throttled_render()
    
    def _orbit_camera(self, delta_yaw: float, delta_pitch: float):
        """Orbit camera by delta amounts."""
        self.camera_system.orbit(delta_yaw, delta_pitch)
        self._sync_sliders_to_camera()
        self._update_camera_preview()
        
        if self._auto_render:
            self._throttled_render()
    
    def _zoom_camera(self, delta_radius: float):
        """Zoom camera by delta amount."""
        self.camera_system.zoom(delta_radius)
        self._sync_sliders_to_camera()
        self._update_camera_preview()
        
        if self._auto_render:
            self._throttled_render()
    
    def _sync_sliders_to_camera(self):
        """Sync slider values to current camera pose."""
        pose = self.camera_system.get_pose()
        self.yaw_slider.value = pose.yaw
        self.pitch_slider.value = pose.pitch
        self.radius_slider.value = pose.radius
        self.fov_slider.value = pose.fov
        
        # Update pose display
        self.pose_display.value = f"yaw={pose.yaw:.1f}°, pitch={pose.pitch:.1f}°, radius={pose.radius:.2f}, fov={pose.fov:.1f}°"
    
    def _update_camera_preview(self):
        """Update camera preview visualization."""
        if not self.show_camera_checkbox.value:
            return
        
        pose = self.camera_system.get_pose()
        position = pose.to_cartesian()
        
        # Add camera position marker (with error handling for viser API)
        try:
            self.server.scene.add_icosphere(
                name="camera_position",
                radius=0.05,
                color=(1.0, 0.0, 0.0),
                position=position
            )
            
            # Add camera frustum if enabled
            if self.show_frustum_checkbox.value:
                self._add_camera_frustum(pose)
                
        except Exception as e:
            # Gracefully handle viser API differences
            logger.debug(f"Camera preview visualization not available: {e}")
            pass
    
    def _add_camera_frustum(self, pose: CameraPose):
        """Add camera frustum visualization."""
        position = np.array(pose.to_cartesian())
        
        # Calculate frustum corners at radius distance
        fov_rad = np.radians(pose.fov)
        aspect = 1.0  # Square aspect ratio
        
        # Near plane size
        near_height = 2 * 0.1 * np.tan(fov_rad / 2)  # Near at 0.1 units
        near_width = near_height * aspect
        
        # Far plane size (at object)
        far_height = 2 * pose.radius * np.tan(fov_rad / 2)
        far_width = far_height * aspect
        
        # Create frustum lines from camera to target
        origin = np.array([0, 0, 0])  # Looking at origin
        
        # Simple frustum representation - just show viewing direction
        direction = origin - position
        direction = direction / np.linalg.norm(direction)
        
        # End point on target
        end_point = position + direction * pose.radius
        
        # Add viewing ray
        self.server.scene.add_spline_catmull_rom(
            name="camera_frustum",
            positions=np.array([position, end_point]),
            color=(0.0, 1.0, 0.0),
            line_width=2.0
        )
    
    def _undo_pose(self, _):
        """Undo last pose change."""
        if self.camera_system.undo():
            self._sync_sliders_to_camera()
            self._update_camera_preview()
    
    def _redo_pose(self, _):
        """Redo pose change."""
        if self.camera_system.redo():
            self._sync_sliders_to_camera()
            self._update_camera_preview()
    
    def _reset_camera(self, _=None):
        """Reset camera to default pose."""
        self.camera_system.reset_to_default()
        self._sync_sliders_to_camera()
        self._update_camera_preview()
        
        if self._auto_render:
            self._throttled_render()
    
    def _export_current_pose(self, _):
        """Export current camera pose to JSON."""
        try:
            output_path = Path("current_pose.json")
            current_pose = self.camera_system.get_pose()
            self.camera_system.save_poses([current_pose], output_path)
            
            self.render_status.value = f"✅ Exported: {output_path}"
            logger.info(f"Exported current pose to {output_path}")
            
        except Exception as e:
            self.render_status.value = f"❌ Export failed: {str(e)}"
            logger.error(f"Failed to export pose: {e}")
    
    def _export_standard_views(self, _):
        """Export standard views to JSON."""
        try:
            output_path = Path("standard_views.json")
            standard_views = generate_standard_views(self.camera_system)
            self.camera_system.save_poses(standard_views, output_path)
            
            self.render_status.value = f"✅ Exported: {output_path} ({len(standard_views)} views)"
            logger.info(f"Exported {len(standard_views)} standard views to {output_path}")
            
        except Exception as e:
            self.render_status.value = f"❌ Export failed: {str(e)}"
            logger.error(f"Failed to export standard views: {e}")
    
    def _on_auto_render_change(self, _):
        """Handle auto render checkbox change."""
        self._auto_render = self.auto_render_checkbox.value
        
        if self._auto_render:
            self._throttled_render()
    
    def _manual_render(self, _):
        """Trigger manual render."""
        self._trigger_render()
    
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
    
    def _on_preview_change(self, _):
        """Handle preview option changes."""
        self._update_camera_preview()
    
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
        self._sync_sliders_to_camera()
        self._update_camera_preview()
    
    def run(self):
        """Run the interactive interface."""
        try:
            logger.info("Interactive camera controls running. Press Ctrl+C to stop.")
            
            # Keep server running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Interactive camera controls stopped")
        except Exception as e:
            logger.error(f"Error running interactive controls: {e}")
        finally:
            self.server.stop()


# Convenience function for quick setup
def launch_interactive_controls(model_path: Optional[str] = None, 
                              port: int = 8080,
                              render_callback: Optional[Callable] = None) -> InteractiveCameraControls:
    """Launch interactive camera controls.
    
    Args:
        model_path: Optional initial model path
        port: Server port
        render_callback: Optional render function
        
    Returns:
        InteractiveCameraControls instance
    """
    controls = InteractiveCameraControls(server_port=port)
    
    if render_callback:
        controls.set_render_callback(render_callback)
    
    if model_path:
        controls.model_path_input.value = model_path
        controls._load_model(None)
    
    return controls