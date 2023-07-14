import torch
import cv2
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    PointLights, Materials, TexturesUV, PerspectiveCameras,TexturesVertex
)
from pytorch3d.renderer.mesh.shader import SoftPhongShader as TexturedPhongShader

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': (197, 27, 125),  # L lower leg
        'light_pink': (233, 163, 201),  # L upper leg
        'light_green': (161, 215, 106),  # L lower arm
        'green': (77, 146, 33),  # L upper arm
        'red': (215, 48, 39),  # head
        'light_red': (252, 146, 114),  # head
        'light_orange': (252, 141, 89),  # chest
        'purple': (118, 42, 131),  # R lower leg
        'light_purple': (175, 141, 195),  # R upper
        'light_blue': (145, 191, 219),  # R lower arm
        'blue': (69, 117, 180),  # R upper arm
        'gray': (130, 130, 130),  #
        'white': (255, 255, 255),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    elif joints.shape[1] == 21:  # hand
        parents = np.array([
            -1,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            0,
            13,
            14,
            15,
            0,
            17,
            18,
            19,
        ])
        ecolors = {
            0: 'light_purple',
            1: 'light_green',
            2: 'light_green',
            3: 'light_green',
            4: 'light_green',
            5: 'pink',
            6: 'pink',
            7: 'pink',
            8: 'pink',
            9: 'light_blue',
            10: 'light_blue',
            11: 'light_blue',
            12: 'light_blue',
            13: 'light_red',
            14: 'light_red',
            15: 'light_red',
            16: 'light_red',
            17: 'purple',
            18: 'purple',
            19: 'purple',
            20: 'purple',
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image

def visualize_reconstruction(img, img_size, gt_kp, vertices, pred_kp, camera, renderer, mano_face):
    ###convert to openGL coordinate
    device=vertices.device
    vertices = vertices * torch.from_numpy(np.asarray([-1, -1, 1])).unsqueeze(0).float().to(device)
    focal_length = camera[:,[0,2]]
    camera_center = camera[:,[1,3]]
    gt_vis = gt_kp[:, 2].astype(bool)
    res = img.shape[1]
    camera_t = torch.from_numpy(np.array([0,0,0])).unsqueeze(0).float().to(device)
    renderer.camera_config( res, focal_length,camera_center, camera_t, has_lights=True)
    face_idx = torch.tensor(mano_face).to(device)
    face_idx = face_idx.repeat((vertices.shape[0], 1, 1)).contiguous()
    tex_vertex = torch.zeros_like(vertices) + torch.tensor([255 / 255, 192 / 255, 203 / 255]).float().to(device)
    images = renderer.render_hand_vertex(vertices, face_idx, tex_vertex)
    rend_img = images[0, :, :, :3].cpu().numpy()
    mask = images[0, :, :, 3]>0
    mask = mask.cpu().numpy()
    rend_img = img*(1-mask[:,:,None])+rend_img*mask[:,:,None]
    # Draw skeleton
    gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * img_size
    pred_joint = pred_kp * img_size
    img_with_gt = draw_skeleton( img, gt_joint, draw_edges=False, vis=gt_vis)
    skel_img = draw_skeleton(img_with_gt, pred_joint)

    combined = np.hstack([skel_img, rend_img])

    return combined

def visualize_reconstruction_ortho_proj(img, img_size, gt_kp, vertices, pred_kp, camera, renderer, mano_face,focal_length=1000):
    ###convert to openGL coordinate
    device = 'cuda'
    res = img.shape[1]
    gt_vis = gt_kp[:, 2].astype(bool)
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * np.absolute(camera[0]) +1e-9)])
    ###convert to openGL coordinate
    vertices = vertices * np.asarray([-1, -1, 1])
    vertices = torch.from_numpy(vertices).unsqueeze(0).float().to(device)
    camera_t = camera_t * np.asarray([-1, -1, 1])
    camera_t = torch.from_numpy(camera_t).unsqueeze(0).float().to(device)
    znear = 0.1
    dist = torch.abs(camera_t[:, 2] - torch.mean(vertices, axis=1)[:, 2])
    zfar = dist + 20
    camera_center = res / 2
    fov = 2 * np.arctan(camera_center / focal_length) * 180 / np.pi
    renderer.camera_config_fov(fov, res, znear, zfar, camera_t, has_lights=True)
    face_idx = torch.tensor(mano_face).to(device)
    face_idx = face_idx.repeat((vertices.shape[0], 1, 1)).contiguous()
    tex_vertex = torch.zeros_like(vertices) + torch.tensor([255 / 255, 192 / 255, 203 / 255]).float().to(device)
    images = renderer.render_hand_vertex(vertices, face_idx, tex_vertex)
    rend_img = images[0, :, :, :3].cpu().numpy()
    mask = images[0, :, :, 3]>0
    mask = mask.cpu().numpy()
    rend_img = img*(1-mask[:,:,None])+rend_img*mask[:,:,None]
    # Draw skeleton
    gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * img_size
    pred_joint = ((pred_kp[:, :2] + 1) * 0.5) * img_size
    img_with_gt = draw_skeleton( img, gt_joint, draw_edges=False, vis=gt_vis)
    skel_img = draw_skeleton(img_with_gt, pred_joint)

    combined = np.hstack([skel_img, rend_img])

    return combined

class Renderer:
    def __init__(self, device):

        self.cameras = None
        self.image_size = None
        self.renderer = None

        self.device = device

    def config_renderer(self,has_lights=False):
        if has_lights:
            lights = PointLights(device=self.device, location=[[0.0, 0.0, -2.0]])
        else:
            lights = PointLights(device=self.device, location=[[1.0, 1.0, -2.0]], ambient_color=[[1.0, 1.0, 1.0]],
                                 diffuse_color=[[0., 0., 0.]], specular_color=[[0, 0, 0]])
        ###Lights for 3D visualization

        material = Materials(device=self.device, ambient_color=[[1, 1, 1]], diffuse_color=[[1, 1, 1]],
                             specular_color=[[0, 0, 0]])

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0,
            faces_per_pixel=1,
            bin_size=0
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=TexturedPhongShader(
                device=self.device,
                lights=lights,
                materials=material,
                cameras=self.cameras)
        )

        return renderer

    def render_hand_vertex(self, vertices, faces_idx, tex_vertex):
        batch_size = vertices.shape[0]

        # move vertices to the center
        verts = vertices.clone()
        # verts = verts - torch.mean(verts,1).unsqueeze(1)
        tex = TexturesVertex(verts_features=tex_vertex)
        mesh = Meshes(verts=verts, faces=faces_idx, textures=tex)

        fragments = self.renderer.rasterizer(mesh)
        images = self.renderer.shader(fragments, mesh)
        output = torch.cat((images[:,:,:,:3],fragments.zbuf),axis=-1)

        return output

    def render_hand_vertex_occlusion_aware(self, vertices, faces_idx, tex_vertex):

        # move vertices to the center
        verts = vertices.clone()
        tex = TexturesVertex(verts_features=tex_vertex)
        mesh = Meshes(verts=verts, faces=faces_idx, textures=tex)
        ###occlusion_aware_rasterization
        stack_faces = faces_idx.view(-1, 3)
        fragments = self.renderer.rasterizer(mesh)
        pix_to_face = fragments.pix_to_face[:,:,:,0]
        occ_masks = torch.zeros((verts.shape[:2])).to(verts.device)
        for n_batch in range(verts.shape[0]):
            visible_tri_id = torch.unique(pix_to_face[n_batch])[1:]
            visible_vertices_id = stack_faces[visible_tri_id.tolist(),:]
            visible_vertices_id = torch.unique(visible_vertices_id)
            occ_masks[n_batch, visible_vertices_id.tolist()] = 1
        new_tex_vertex = tex_vertex * occ_masks[:,:,None]
        tex = TexturesVertex(verts_features=new_tex_vertex)
        mesh = Meshes(verts=verts, faces=faces_idx, textures=tex)

        images = self.renderer.shader(fragments, mesh)
        rend_img = images[:, :, :, :3].permute(0,3,1,2)
        mask = images[:, :, :, 3] > 0
        mask = mask[:,None,:,:]

        return rend_img, mask

    def camera_config_fov(self, fov, image_size, znear, zfar, camera_t, has_lights=False):
        self.cameras = FoVPerspectiveCameras(znear=znear, zfar=zfar, fov=fov, device=self.device, T=camera_t)
        self.image_size = image_size
        self.renderer = self.config_renderer(has_lights)

    def camera_config(self, image_size,focal_length,principal_point, camera_t,has_lights=False):
        self.cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, image_size=((image_size, image_size),),device=self.device, T=camera_t)
        self.image_size = image_size
        self.renderer = self.config_renderer(has_lights)