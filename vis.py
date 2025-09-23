import viser
from viser import transforms as vtf
import imageio
import json
import os
import numpy as np
from einops import repeat
import math
import trimesh

def pinhole_z_depth_to_xyz(depth, f, H=512, W=512):
    if not isinstance(depth, float):
        H, W = depth.shape
        z = depth
    else:
        z = np.ones((H, W), dtype=np.float32) * depth
    y, x = np.mgrid[:H, :W]
    x = x - W // 2
    y = H // 2 - y
    x = x / f * z
    y = - y / f * z
    return np.stack([x, y, z], -1)

fov = 0.6981317007977318
# fov = 1.6981317007977318
focal_length = (512/2) / math.tan(fov/2)

bg_img = np.linspace(250, 200, 128).astype(int)
bg_img = repeat(bg_img, 'n -> n 128 3')

class Viewer:
    def __init__(self, render_folder='renderings/renders/ab7d054fee3746c0b0c048838cedd91b'):
        self.server = viser.ViserServer()
        self.server.scene.set_background_image(bg_img)
        self.server.scene.set_up_direction('-y')
        self.render_folder = render_folder
        self._init_ui()
        self.draw_frame()

    def _init_ui(self):
        # Load transforms.json to get number of frames
        json_path = os.path.join(self.render_folder, 'transforms.json')
        with open(json_path, 'r') as f:
            self.transforms_info = json.load(f)
        self.num_frames = len(self.transforms_info['frames'])

        self.current_frame_id = 0

        self.slider = self.server.gui.add_slider(
            label='FrameID', min=0, max=self.num_frames-1,
            step=1, initial_value=0)
        @self.slider.on_update
        def _(_):
            try:
                self.current_root_node.remove()
            except:
                print('no current root node, but does not matter')
                pass
            self.current_frame_id = self.slider.value
            self.draw_frame()

        self.next_button = self.server.gui.add_button(label='  Next ->')
        @self.next_button.on_click
        def _(_):
            self.current_frame_id = (self.current_frame_id + 1) % self.num_frames
            self.slider.value = self.current_frame_id
            self.draw_frame()

        self.prev_button = self.server.gui.add_button(label='<- Prev  ')
        @self.prev_button.on_click
        def _(_):
            self.current_frame_id = (self.current_frame_id - 1) % self.num_frames
            self.slider.value = self.current_frame_id
            self.draw_frame()

    def draw_frame(self):
        coord = self.server.scene.add_frame('/', show_axes=False)
        self.current_root_node = coord

        # Load and display the mesh
        mesh_path = os.path.join(self.render_folder, 'mesh.ply')
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)

            # Apply coordinate transformation to match camera view
            # Blender uses Z-up, but viser uses Y-up by default
            # We need to rotate the mesh 90 degrees around X axis
            vertices = mesh.vertices.copy()
            # Transform from Blender (Z-up) to viser coordinate system
            transformed_vertices = np.zeros_like(vertices)
            transformed_vertices[:, 0] = vertices[:, 0]  # X stays the same
            transformed_vertices[:, 1] = -vertices[:, 2]  # Y = -Z (from Blender)
            transformed_vertices[:, 2] = vertices[:, 1]   # Z = Y (from Blender)

            # Display the mesh
            self.server.scene.add_mesh_simple(
                '/mesh',
                vertices=transformed_vertices,
                faces=mesh.faces,
                color=(200, 200, 200),  # Light gray color
                wireframe=False
            )

        # Display camera frustums with images (no depth visualization)
        for i, frame in enumerate(self.transforms_info['frames'][::10]):
            file_path = frame['file_path']
            idx = file_path.replace('.png', '')
            img_path = os.path.join(self.render_folder, file_path)

            # Load image
            img = imageio.imread(img_path)
            rgb_full = img[..., :3]

            pose = np.array(frame['transform_matrix'])

            ### < convert coordinate system >
            rpy = vtf.SO3.from_matrix(pose[:3,:3]).as_rpy_radians()
            r,p,y = rpy.roll, rpy.pitch, rpy.yaw
            rpy = [r-np.pi/2, -y, p]

            pos = pose[:3,3]
            x,y,z = pos
            xyz = [x, -z, y]

            rot = vtf.SO3.from_rpy_radians(*rpy)
            wxyz = rot.wxyz
            position = np.array(xyz)
            ### </convert coordinate system >

            # Only show camera frustum for current frame or all frames
            if i == self.current_frame_id or True:
                # Highlight current camera
                self.server.scene.add_camera_frustum(
                    f'/cam_{idx}',
                    fov=fov, aspect=1,
                    image=rgb_full,
                    wxyz=wxyz, position=position,
                    scale=0.15
                )


if __name__ == '__main__':
    viewer = Viewer(render_folder='renderings/renders/5f62800c792d47a5a11677afa4d1020c/' )#'renderings/renders/c871c724955a46c1a50d4cae3ef3f1ee/')
    input('Press Enter to quit')
