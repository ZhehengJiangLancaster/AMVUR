import torch
import  numpy as np
import cv2
from AMVUR.utils.image_ops import flip_kp

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return torch.stack((x,y,z),1).float()

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return torch.stack((x,y,z),1).float()

def rotate_3d( S, r, f):
    """Process gt 3D keypoints and apply all augmentation transforms."""
    # in-plane rotation
    rot_mat = np.eye(3)
    if not r == 0:
        rot_rad = -r * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
    S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
    # flip the x coordinates
    if f:
        S = flip_kp(S)
    S = S.astype('float32')
    return S

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def proj_func(xyz, K):
    '''
    xyz: N x num_points x 3
    K: N x 3 x 3
    '''
    uv = torch.bmm(K,xyz.permute(0,2,1))
    uv = uv.permute(0, 2, 1)
    out_uv = torch.zeros_like(uv[:,:,:2]).to(device=uv.device)
    out_uv = torch.addcdiv(out_uv, uv[:,:,:2], uv[:,:,2].unsqueeze(-1).repeat(1,1,2), value = 1)
    return out_uv

def proj_var(mu,log_var, K):
    '''
    xyz: N x num_points x 3
    K: N x 3 x 3
    '''
    K_expand = K[:,None,:,:].expand(-1,log_var.shape[1],-1,-1)
    log_var_K = K_expand * log_var[:,:,None,:].exp()
    log_var_new = torch.einsum('bjpq,bjqi->bjpi', [log_var_K, K_expand.permute(0, 1, 3, 2)])
    log_var_new = torch.log(torch.diagonal(log_var_new,0,-2,-1))
    exp_var_new = log_var_new.exp()
    log_var_new[:, :, 0] = torch.log(mu[:, :, 0].square() / mu[:, :, 2].square() * (
                exp_var_new[:, :, 0] / mu[:, :, 0].square() + exp_var_new[:, :, 2] / mu[:, :, 2].square()))
    log_var_new[:, :, 1] = torch.log(mu[:, :, 1].square() / mu[:, :, 2].square() * (
                exp_var_new[:, :, 1] / mu[:, :, 1].square() + exp_var_new[:, :, 2] / mu[:, :, 2].square()))
    uv_var = log_var_new[:,:,:2]
    return uv_var

def camparam2Ks(cam_param,norm_param=1):
    Ks_i = torch.zeros([cam_param.shape[0], 3, 3], device=cam_param.device)
    Ks_i[:, 0, 0] = cam_param[:, 0] * norm_param
    Ks_i[:, 0, 2] = cam_param[:, 1] * norm_param
    Ks_i[:, 1, 1] = cam_param[:, 2] * norm_param
    Ks_i[:, 1, 2] = cam_param[:, 3] * norm_param
    Ks_i[:, 2, 2] = 1
    return Ks_i


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def oriMesh2bbMesh(oriMesh, img2bb_affineTrans, bb_res, f, c):
    #output: vertices_bb is the coordinates of vertices in 3D bounding box
    #        vertices_img_bb_uv is the projected coordinates of vertices in 2D bounding box
    vertices_img = cam2pixel(oriMesh, f, c)
    vertices_xy1 = torch.cat((vertices_img[:, :2], torch.ones_like(vertices_img[:, :1])), 1)
    vertices_img_bb = vertices_img.contiguous()
    vertices_img_bb_uv = torch.mm(img2bb_affineTrans, vertices_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    vertices_img_bb[:, :2] = vertices_img_bb_uv
    vertices_bb = pixel2cam(vertices_img_bb, f, torch.tensor((bb_res / 2, bb_res / 2)))
    return vertices_bb, vertices_img_bb_uv


def batch_oriMesh2bbMesh(oriMesh_batch, img2bb_affineTrans_batch, bb_res, cam_param_batch):
    #output: vertices_bb is the coordinates of vertices in 3D bounding box
    #        vertices_img_bb_uv is the projected coordinates of vertices in 2D bounding box
    vertices_bb_list = []
    vertices_img_bb_uv_list = []
    for i in range(oriMesh_batch.shape[0]):
        vertices_img = cam2pixel(oriMesh_batch[i,:,:], cam_param_batch[i,0:2], cam_param_batch[i,2:])
        vertices_xy1 = torch.cat((vertices_img[:, :2], torch.ones_like(vertices_img[:, :1])), 1)
        vertices_img_bb = vertices_img.contiguous()
        vertices_img_bb_uv = torch.mm(img2bb_affineTrans_batch[i,:,:], vertices_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
        vertices_img_bb[:, :2] = vertices_img_bb_uv
        vertices_bb = pixel2cam(vertices_img_bb, cam_param_batch[i,0:2], torch.tensor((bb_res / 2, bb_res / 2)))
        vertices_img_bb_uv_list.append(vertices_img_bb_uv)
        vertices_bb_list.append(vertices_bb)
    return torch.stack(vertices_bb_list), torch.stack(vertices_img_bb_uv_list)

def oriJoints2bbJoints(oriJoints, img2bb_affineTrans, bb_res, f, c):
    #output: vertices_bb is the coordinates of vertices in 3D bounding box
    #        vertices_img_bb_uv is the projected coordinates of vertices in 2D bounding box
    joints_img = cam2pixel(oriJoints, f, c)
    joints_xy1 = torch.cat((joints_img[:, :2], torch.ones_like(joints_img[:, :1])), 1)
    joints_img_bb = joints_img.contiguous()
    joints_img_bb_uv = torch.mm(img2bb_affineTrans, joints_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    joints_img_bb[:, :2] = joints_img_bb_uv
    joints_bb = pixel2cam(joints_img_bb, f, torch.tensor((bb_res / 2, bb_res / 2)))
    return joints_bb, joints_img_bb_uv

def linear_trans_var(log_var, A):
    """

    :param log_var: size = [B,C,1,V]
    :param A: size = [B,C,J,V]
    :return:
    """
    # joint_regressor_torch_var = torch.einsum('bcik,bckj->bcij', [A.permute(0, 1, 3, 2),log_var.exp()])
    joint_regressor_torch_var = A*log_var.exp()
    pred_3d_joints_mano_from_mesh_var = torch.einsum('bcik,bckj->bcij', [joint_regressor_torch_var, A.permute(0,1,3,2)])
    new_log_var = torch.log(torch.diagonal(pred_3d_joints_mano_from_mesh_var,0,-2,-1)).permute(0,2,1)
    return new_log_var

def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    it ignores the depth(Z) value
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0].abs() * X_trans.view(shape[0], -1)).view(shape)
    return X_2d

def orthographic_projection_var(X_var, camera):
    X_2d_var = X_var[:, :, :2].clone()
    shape = X_2d_var.shape
    X_2d_var = (torch.log(camera[:, 0].square())[:,None]+X_2d_var.view(shape[0], -1)).view(shape)
    return X_2d_var