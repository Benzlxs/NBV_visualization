import viser 
from viser import transforms as vtf
import imageio 
import json 
import os
import numpy as np
from einops import repeat
import math

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
    def __init__(self, subset='renderings/renders'):
        self.server = viser.ViserServer()
        self.server.scene.set_background_image(bg_img)
        self.server.scene.set_up_direction('-y')
        self.subset = subset
        self.subset_list = sorted(os.listdir(self.subset))
        self.subset_length = len(self.subset_list)
        self._init_ui()
        self.draw_frame()

    def _init_ui(self):
        self.current_obj_id = 0

        self.slider = self.server.gui.add_slider(
            label='FramdID', min=0, max=self.subset_length-1,
            step=1, initial_value=0)
        @self.slider.on_update
        def _(_):
            try:
                self.current_root_node.remove()
            except:
                print('no current root node, but does not matter')
                pass
            self.current_obj_id = self.slider.value
            self.draw_frame()
        
        self.next_botton = self.server.gui.add_button(label='  Next ->')
        @self.next_botton.on_click
        def _(_):
            self.current_obj_id = (self.current_obj_id + 1) % self.subset_length
            self.slider.value = self.current_obj_id
            self.draw_frame()

        self.prev_botton = self.server.gui.add_button(label='<- Prev  ')
        @self.prev_botton.on_click
        def _(_):
            self.current_obj_id = (self.current_obj_id - 1) % self.subset_length
            self.slider.value = self.current_obj_id
            self.draw_frame()
    
    def draw_frame(self):
        json_path = os.path.join(self.subset, self.subset_list[self.current_obj_id], 'transforms.json')
        with open(json_path, 'r') as f:
            info = json.load(f)

        coord = self.server.scene.add_frame('/', show_axes=False)
        self.current_root_node = coord

        for frame in info['frames']:
            file_path = frame['file_path']
            idx = file_path.replace('.png', '')
            img_path = json_path.replace('transforms.json', file_path)
            depth_path = img_path.replace('.png', '_depth.png')
            depth = imageio.imread(depth_path)
            img = imageio.imread(img_path)
            rgb_full = img[..., :3]
            rgb = img[..., :3]
            # mask = img[..., -1] == 255
            mask = depth < 65534

            depth_range = 2 * 0.5 * np.sqrt(3)
            min_depth = 2 - 0.5 * np.sqrt(3)
            depth = depth / 65535.0 * depth_range + min_depth
            pose = np.array(frame['transform_matrix'])

            pcd = pinhole_z_depth_to_xyz(depth, focal_length)

            rgb = rgb[mask]
            pcd = pcd[mask]

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

            pose1 = np.eye(4)
            pose1[:3,:3] = rot.as_matrix()
            pose1[:3,3] = position
            ### </convert coordinate system >

            self.server.scene.add_camera_frustum(
                f'/cam_{idx}',
                fov=fov, aspect=1,
                image=rgb_full, 
                wxyz=wxyz, position=position)
            
            self.server.scene.add_point_cloud(
                f'/cam_{idx}/pcd', points=pcd, colors=rgb, 
                point_size=0.003, point_shape='circle')


if __name__ == '__main__':
    viewer = Viewer()
    input('Press Enter to quit')