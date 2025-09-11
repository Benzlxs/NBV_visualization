#!/usr/bin/env python3
"""Interactive Blender rendering tool with web interface.

This module combines Step 5 (Interactive Camera Controls) with Step 6 (Blender Rendering Pipeline)
to create a complete browser-based tool for interactive GLB model rendering.
"""

import viser
import numpy as np
import time
import threading
import logging
import base64
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
import tempfile
import shutil

from .blender_pipeline import BlenderRenderingPipeline, RenderConfig, RenderQuality, RenderResult
from .trellis_camera import TRELLISCameraSystem, CameraPose, generate_standard_views
from .glb_loader import GLBLoader
from .image_capture_export import ImageCaptureExporter

logger = logging.getLogger(__name__)


@dataclass
class RenderRequest:
    """Request for a render operation."""
    pose: CameraPose
    config: RenderConfig
    request_id: str
    timestamp: float


class InteractiveBlenderTool:
    """Complete interactive Blender rendering tool with web interface."""
    
    def __init__(self, server_port: int = 8080):
        """Initialize interactive Blender tool.
        
        Args:
            server_port: Port for viser server
        """
        # Initialize viser server
        self.server = viser.ViserServer(port=server_port)
        self.server_port = server_port
        
        # Initialize rendering systems
        self.pipeline = BlenderRenderingPipeline()
        self.camera_system = TRELLISCameraSystem(default_radius=2.5, default_fov=40.0)
        self.exporter = ImageCaptureExporter(output_base_dir="interactive_renders")
        
        # Model and state
        self.current_model_path: Optional[Path] = None
        self.model_mesh = None
        self.current_render_config = self._default_render_config()
        
        # Render queue and management
        self.render_queue: List[RenderRequest] = []
        self.active_renders: Dict[str, RenderResult] = {}
        self.completed_renders: Dict[str, RenderResult] = {}
        self.is_rendering = False
        self.render_worker_thread: Optional[threading.Thread] = None
        
        # UI state
        self.auto_render_enabled = False
        self.last_render_time = 0
        self.render_cooldown = 2.0  # seconds between auto-renders
        
        # Setup UI
        self._setup_ui()
        
        logger.info(f"Interactive Blender tool started on port {server_port}")
    
    def _default_render_config(self) -> RenderConfig:
        """Create default render configuration."""
        return self.pipeline.create_render_config(
            quality=RenderQuality.STANDARD,
            resolution=(512, 512),
            format="PNG",
            use_gpu=False  # Stable CPU rendering
        )
    
    def _setup_ui(self):
        """Set up complete viser UI."""
        
        # Header
        with self.server.gui.add_folder("🎨 Interactive Blender Renderer", expand_by_default=True):
            self.status_text = self.server.gui.add_text(
                "Status", 
                initial_value="🚀 Ready to render! Load a model to begin.",
                disabled=True
            )
        
        # Model loading section
        with self.server.gui.add_folder("📁 Model Loading", expand_by_default=True):
            self.model_path_input = self.server.gui.add_text(
                "GLB Model Path", 
                initial_value="dataset/ABO/raw/3dmodels/original/0/B01LR5RSG0.glb"
            )
            
            self.load_model_button = self.server.gui.add_button("🔄 Load GLB Model")
            self.load_model_button.on_click(self._load_model)
            
            self.model_info = self.server.gui.add_text(
                "Model Info",
                initial_value="No model loaded",
                disabled=True
            )
        
        # Camera controls section
        with self.server.gui.add_folder("🎥 Camera Controls", expand_by_default=True):
            
            # Current pose display
            self.pose_display = self.server.gui.add_text(
                "Current Pose",
                initial_value="yaw=0.0°, pitch=0.0°, radius=2.5, fov=40.0°",
                disabled=True
            )
            
            # Camera parameter sliders
            self.yaw_slider = self.server.gui.add_slider(
                "Yaw (°)", min=0.0, max=360.0, step=1.0, initial_value=0.0
            )
            self.pitch_slider = self.server.gui.add_slider(
                "Pitch (°)", min=-89.0, max=89.0, step=1.0, initial_value=0.0  
            )
            self.radius_slider = self.server.gui.add_slider(
                "Radius", min=0.5, max=10.0, step=0.1, initial_value=2.5
            )
            self.fov_slider = self.server.gui.add_slider(
                "FOV (°)", min=10.0, max=120.0, step=1.0, initial_value=40.0
            )
            
            # Bind slider callbacks
            self.yaw_slider.on_update(self._on_pose_change)
            self.pitch_slider.on_update(self._on_pose_change)
            self.radius_slider.on_update(self._on_pose_change)
            self.fov_slider.on_update(self._on_pose_change)
        
        # Render settings section
        with self.server.gui.add_folder("⚙️ Render Settings", expand_by_default=True):
            
            # Quality preset
            self.quality_dropdown = self.server.gui.add_dropdown(
                "Quality",
                options=["preview", "standard", "high", "production"],
                initial_value="standard"
            )
            self.quality_dropdown.on_update(self._on_settings_change)
            
            # Resolution
            self.resolution_dropdown = self.server.gui.add_dropdown(
                "Resolution",
                options=["256x256", "512x512", "1024x1024"],
                initial_value="512x512"
            )
            self.resolution_dropdown.on_update(self._on_settings_change)
            
            # Engine
            self.engine_dropdown = self.server.gui.add_dropdown(
                "Render Engine",
                options=["CYCLES", "BLENDER_EEVEE"],
                initial_value="CYCLES"
            )
            self.engine_dropdown.on_update(self._on_settings_change)
            
            # Samples override
            self.samples_number = self.server.gui.add_number(
                "Samples (override)", initial_value=128, min=16, max=512, step=16
            )
            self.samples_number.on_update(self._on_settings_change)
        
        # Render controls section
        with self.server.gui.add_folder("🖼️ Render Controls", expand_by_default=True):
            
            # Auto render toggle
            self.auto_render_checkbox = self.server.gui.add_checkbox(
                "Auto Render on Pose Change", initial_value=False
            )
            self.auto_render_checkbox.on_update(self._on_auto_render_change)
            
            # Manual render button
            self.render_button = self.server.gui.add_button("🎨 Render Now")
            self.render_button.on_click(self._manual_render)
            
            # Render status
            self.render_status = self.server.gui.add_text(
                "Render Status",
                initial_value="Ready",
                disabled=True
            )
            
            # Render queue info
            self.queue_status = self.server.gui.add_text(
                "Queue Status",
                initial_value="Queue: 0 pending, 0 completed",
                disabled=True
            )
        
        # Quick actions section
        with self.server.gui.add_folder("⚡ Quick Actions"):
            
            # Standard views
            with self.server.gui.add_folder("Standard Views"):
                standard_views = [
                    ("Front", 0.0, 0.0),
                    ("Right", 90.0, 0.0), 
                    ("Back", 180.0, 0.0),
                    ("Left", 270.0, 0.0),
                    ("Top", 45.0, 45.0),
                    ("Bottom", 45.0, -30.0)
                ]
                
                for name, yaw, pitch in standard_views:
                    button = self.server.gui.add_button(f"📷 {name}")
                    button.on_click(lambda yaw=yaw, pitch=pitch: self._set_standard_view(yaw, pitch))
            
            # Batch operations
            with self.server.gui.add_folder("Batch Operations"):
                self.render_all_views_button = self.server.gui.add_button("🎬 Render All Standard Views")
                self.render_all_views_button.on_click(self._render_all_standard_views)
                
                self.orbit_render_button = self.server.gui.add_button("🔄 Render 12-Step Orbit")
                self.orbit_render_button.on_click(self._render_orbit_sequence)
        
        # Results section
        with self.server.gui.add_folder("📊 Render Results"):
            
            # Current render display
            self.current_render_display = self.server.gui.add_text(
                "Latest Render",
                initial_value="No renders completed yet",
                disabled=True
            )
            
            # Clear results button
            self.clear_results_button = self.server.gui.add_button("🗑️ Clear Results")
            self.clear_results_button.on_click(self._clear_results)
            
            # Export session button
            self.export_session_button = self.server.gui.add_button("📦 Export Session")
            self.export_session_button.on_click(self._export_session)
        
        # Preview section
        with self.server.gui.add_folder("👁️ 3D Preview"):
            
            self.show_model_checkbox = self.server.gui.add_checkbox(
                "Show 3D Model", initial_value=True
            )
            self.show_model_checkbox.on_update(self._update_3d_preview)
            
            self.show_camera_checkbox = self.server.gui.add_checkbox(
                "Show Camera Position", initial_value=True
            )
            self.show_camera_checkbox.on_update(self._update_3d_preview)
        
        # Start render worker
        self._start_render_worker()
    
    def _load_model(self, _):
        """Load GLB model."""
        try:
            model_path = Path(self.model_path_input.value)
            
            if not model_path.exists():
                self.model_info.value = f"❌ File not found: {model_path}"
                self.status_text.value = "❌ Model loading failed"
                return
            
            self.status_text.value = "🔄 Loading model..."
            
            # Load GLB model
            loader = GLBLoader()
            model = loader.load_model(model_path)
            
            self.current_model_path = model_path
            self.model_mesh = model.mesh
            
            # Update model info
            vertices = len(model.mesh.vertices)
            faces = len(model.mesh.faces)
            self.model_info.value = f"✅ {model_path.name} ({vertices:,} vertices, {faces:,} faces)"
            
            # Update 3D preview
            self._update_3d_preview()
            
            # Reset camera
            self._reset_camera()
            
            self.status_text.value = f"✅ Model loaded: {model_path.name}"
            logger.info(f"Loaded model: {model_path}")
            
        except Exception as e:
            self.model_info.value = f"❌ Error: {str(e)}"
            self.status_text.value = "❌ Model loading failed"
            logger.error(f"Failed to load model: {e}")
    
    def _update_3d_preview(self, _=None):
        """Update 3D scene preview."""
        try:
            # Clear existing scene objects
            # Note: Viser doesn't have a clear method, so we'll add with consistent names
            
            # Show 3D model if enabled
            if self.show_model_checkbox.value and self.model_mesh is not None:
                vertices = np.array(self.model_mesh.vertices, dtype=np.float32)
                faces = np.array(self.model_mesh.faces, dtype=np.uint32)
                
                # Center and normalize
                center = vertices.mean(axis=0)
                vertices = vertices - center
                scale = 2.0 / np.max(np.abs(vertices))
                vertices = vertices * scale
                
                # Add mesh to scene
                self.server.scene.add_mesh_simple(
                    name="model_preview",
                    vertices=vertices,
                    faces=faces,
                    color=(0.8, 0.8, 0.9)
                )
            
            # Show camera position if enabled
            if self.show_camera_checkbox.value:
                pose = self.camera_system.get_pose()
                position = np.array(pose.to_cartesian(), dtype=np.float32)
                
                # Camera marker
                self.server.scene.add_icosphere(
                    name="camera_position",
                    radius=0.1,
                    color=(1.0, 0.0, 0.0),
                    position=position
                )
                
                # View direction line
                origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                self.server.scene.add_spline_catmull_rom(
                    name="view_direction",
                    positions=np.array([position, origin]),
                    color=(0.0, 1.0, 0.0),
                    line_width=3.0
                )
                
        except Exception as e:
            logger.debug(f"3D preview update failed: {e}")
    
    def _on_pose_change(self, _):
        """Handle camera pose changes."""
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
        
        # Update 3D preview
        self._update_3d_preview()
        
        # Auto render if enabled
        if self.auto_render_enabled and self.current_model_path:
            self._throttled_auto_render()
    
    def _on_settings_change(self, _):
        """Handle render settings changes."""
        try:
            # Parse quality
            quality_map = {
                "preview": RenderQuality.PREVIEW,
                "standard": RenderQuality.STANDARD,
                "high": RenderQuality.HIGH,
                "production": RenderQuality.PRODUCTION
            }
            quality = quality_map.get(self.quality_dropdown.value, RenderQuality.STANDARD)
            
            # Parse resolution
            res_str = self.resolution_dropdown.value
            width, height = map(int, res_str.split('x'))
            resolution = (width, height)
            
            # Update render config
            self.current_render_config = self.pipeline.create_render_config(
                quality=quality,
                resolution=resolution,
                engine=self.engine_dropdown.value,
                samples=int(self.samples_number.value),
                file_format="PNG"
            )
            
            logger.info(f"Render settings updated: {quality.value}, {resolution}, {self.engine_dropdown.value}")
            
        except Exception as e:
            logger.error(f"Failed to update render settings: {e}")
    
    def _on_auto_render_change(self, _):
        """Handle auto render toggle."""
        self.auto_render_enabled = self.auto_render_checkbox.value
        
        if self.auto_render_enabled:
            self.status_text.value = "🚀 Auto-render enabled - move camera to render!"
        else:
            self.status_text.value = "⏸️ Auto-render disabled - use manual render"
    
    def _throttled_auto_render(self):
        """Auto render with throttling."""
        current_time = time.time()
        if current_time - self.last_render_time > self.render_cooldown:
            self._queue_render()
            self.last_render_time = current_time
    
    def _manual_render(self, _):
        """Trigger manual render."""
        if not self.current_model_path:
            self.status_text.value = "❌ Load a model first!"
            return
            
        self._queue_render()
    
    def _queue_render(self):
        """Add current pose to render queue."""
        if not self.current_model_path:
            return
            
        pose = self.camera_system.get_pose()
        request_id = f"render_{int(time.time() * 1000)}"
        
        request = RenderRequest(
            pose=pose,
            config=self.current_render_config,
            request_id=request_id,
            timestamp=time.time()
        )
        
        self.render_queue.append(request)
        self._update_queue_status()
        
        logger.info(f"Queued render: {request_id}")
    
    def _set_standard_view(self, yaw: float, pitch: float):
        """Set camera to standard view."""
        self.camera_system.set_pose(yaw, pitch, self.radius_slider.value, self.fov_slider.value)
        self._sync_sliders_to_camera()
        self._update_3d_preview()
        
        if self.auto_render_enabled and self.current_model_path:
            self._queue_render()
    
    def _render_all_standard_views(self, _):
        """Render all standard views."""
        if not self.current_model_path:
            self.status_text.value = "❌ Load a model first!"
            return
            
        standard_poses = generate_standard_views(self.camera_system)
        
        for pose in standard_poses:
            request = RenderRequest(
                pose=pose,
                config=self.current_render_config,
                request_id=f"standard_{pose.yaw:.0f}_{pose.pitch:.0f}_{int(time.time() * 1000)}",
                timestamp=time.time()
            )
            self.render_queue.append(request)
        
        self._update_queue_status()
        self.status_text.value = f"🎬 Queued {len(standard_poses)} standard views for rendering"
    
    def _render_orbit_sequence(self, _):
        """Render orbital sequence."""
        if not self.current_model_path:
            self.status_text.value = "❌ Load a model first!"
            return
            
        # Generate 12-step orbit
        orbit_poses = []
        for i in range(12):
            yaw = (360.0 * i) / 12
            pose = CameraPose(yaw, 30.0, self.radius_slider.value, self.fov_slider.value)
            orbit_poses.append(pose)
        
        for i, pose in enumerate(orbit_poses):
            request = RenderRequest(
                pose=pose,
                config=self.current_render_config,
                request_id=f"orbit_{i:02d}_{int(time.time() * 1000)}",
                timestamp=time.time()
            )
            self.render_queue.append(request)
        
        self._update_queue_status()
        self.status_text.value = f"🔄 Queued 12-step orbit sequence for rendering"
    
    def _sync_sliders_to_camera(self):
        """Sync slider values to current camera pose."""
        pose = self.camera_system.get_pose()
        self.yaw_slider.value = pose.yaw
        self.pitch_slider.value = pose.pitch
        self.radius_slider.value = pose.radius
        self.fov_slider.value = pose.fov
        
        # Update pose display
        self.pose_display.value = f"yaw={pose.yaw:.1f}°, pitch={pose.pitch:.1f}°, radius={pose.radius:.2f}, fov={pose.fov:.1f}°"
    
    def _reset_camera(self):
        """Reset camera to default pose."""
        self.camera_system.reset_to_default()
        self._sync_sliders_to_camera()
        self._update_3d_preview()
    
    def _start_render_worker(self):
        """Start background render worker thread."""
        if self.render_worker_thread and self.render_worker_thread.is_alive():
            return
            
        self.render_worker_thread = threading.Thread(target=self._render_worker, daemon=True)
        self.render_worker_thread.start()
        logger.info("Render worker thread started")
    
    def _render_worker(self):
        """Background render worker."""
        while True:
            try:
                if self.render_queue and not self.is_rendering:
                    self.is_rendering = True
                    request = self.render_queue.pop(0)
                    
                    self._update_queue_status()
                    self.render_status.value = f"🎨 Rendering {request.request_id}..."
                    
                    # Execute render
                    output_path = self.exporter.output_base_dir / f"{request.request_id}.png"
                    result = self.pipeline.render_single(
                        self.current_model_path, 
                        request.pose, 
                        output_path, 
                        request.config
                    )
                    
                    # Store result
                    self.completed_renders[request.request_id] = result
                    
                    # Update UI
                    if result.success:
                        self.render_status.value = f"✅ Completed {request.request_id} ({result.render_time:.1f}s)"
                        self.current_render_display.value = f"✅ Latest: {request.request_id} - {result.output_path.name} ({result.render_time:.1f}s)"
                        self.status_text.value = f"✅ Render complete! {result.output_path.name}"
                    else:
                        self.render_status.value = f"❌ Failed {request.request_id}"
                        self.current_render_display.value = f"❌ Latest: {request.request_id} - FAILED"
                        self.status_text.value = f"❌ Render failed: {result.error_message}"
                    
                    self._update_queue_status()
                    self.is_rendering = False
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Render worker error: {e}")
                self.is_rendering = False
                time.sleep(1)
    
    def _update_queue_status(self):
        """Update render queue status display."""
        pending = len(self.render_queue)
        completed = len(self.completed_renders)
        self.queue_status.value = f"Queue: {pending} pending, {completed} completed"
    
    def _clear_results(self, _):
        """Clear render results."""
        self.completed_renders.clear()
        self.current_render_display.value = "No renders completed yet"
        self.status_text.value = "🗑️ Results cleared"
        self._update_queue_status()
    
    def _export_session(self, _):
        """Export current session data."""
        if not self.completed_renders:
            self.status_text.value = "❌ No renders to export"
            return
            
        try:
            # Create export summary
            export_dir = self.exporter.output_base_dir / "session_exports"
            export_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            export_file = export_dir / f"interactive_session_{timestamp}.json"
            
            session_data = {
                "timestamp": timestamp,
                "model_path": str(self.current_model_path) if self.current_model_path else None,
                "render_config": {
                    "quality": self.current_render_config.quality.value,
                    "resolution": self.current_render_config.resolution,
                    "engine": self.current_render_config.engine,
                    "samples": self.current_render_config.get_samples()
                },
                "completed_renders": {
                    req_id: {
                        "success": result.success,
                        "output_path": str(result.output_path) if result.output_path else None,
                        "render_time": result.render_time,
                        "error_message": result.error_message
                    }
                    for req_id, result in self.completed_renders.items()
                }
            }
            
            import json
            with open(export_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.status_text.value = f"📦 Session exported: {export_file.name}"
            logger.info(f"Session exported to {export_file}")
            
        except Exception as e:
            self.status_text.value = f"❌ Export failed: {str(e)}"
            logger.error(f"Session export failed: {e}")
    
    def run(self):
        """Run the interactive tool."""
        try:
            logger.info("🎨 Interactive Blender Tool running!")
            logger.info(f"🌐 Open your browser to: http://localhost:{self.server_port}")
            logger.info("Press Ctrl+C to stop")
            
            # Keep server running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Interactive Blender Tool stopped")
        except Exception as e:
            logger.error(f"Error running interactive tool: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.pipeline.cleanup()
            self.exporter.cleanup()
            self.server.stop()
            logger.info("Interactive tool cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Convenience function to launch the tool
def launch_interactive_blender_tool(port: int = 8080, 
                                   model_path: Optional[str] = None) -> InteractiveBlenderTool:
    """Launch the interactive Blender rendering tool.
    
    Args:
        port: Server port
        model_path: Optional initial model to load
        
    Returns:
        InteractiveBlenderTool instance
    """
    tool = InteractiveBlenderTool(server_port=port)
    
    if model_path:
        tool.model_path_input.value = model_path
        tool._load_model(None)
    
    return tool


if __name__ == "__main__":
    # Launch interactive tool
    logging.basicConfig(level=logging.INFO)
    
    print("🎨 Launching Interactive Blender Rendering Tool...")
    print("=" * 50)
    
    tool = launch_interactive_blender_tool(
        port=8080,
        model_path="dataset/ABO/raw/3dmodels/original/0/B01LR5RSG0.glb"
    )
    
    tool.run()