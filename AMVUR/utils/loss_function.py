import torch
import numpy as np
def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_root = gt_keypoints_3d[:, 0,:]
        gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
        pred_root = pred_keypoints_3d[:, 0,:]
        pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_mesh):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_mesh == 1]
    gt_vertices_with_shape = gt_vertices[has_mesh == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()

def pose_var_loss(var):
    loss_KLD = 0.5 * torch.mean(-1 -var + var.exp())
    return loss_KLD

def KLD_vertices_loss(mu,var, prior_mu,prior_var):
    loss_KLD = 0.5 * torch.mean(
        prior_var - var + ((prior_mu - mu).abs() + var.exp()) / prior_var.exp() - 1)
    return loss_KLD

def KLD_3d_joints_loss(mu,var, prior_mu,prior_var):
    prior_root = prior_mu[:, 0, :]
    prior_keypoints_3d = prior_mu - prior_root[:, None, :]
    mu_root = mu[:, 0, :]
    mu_keypoints_3d = mu - mu_root[:, None, :]
    loss_KLD = 0.5 * torch.mean(prior_var -var + ((prior_keypoints_3d-mu_keypoints_3d).pow(2) + var.exp())/prior_var.exp()-1)

    return loss_KLD

def KLD_2d_joints_loss(mu,var, prior):
    loss_KLD = 0.5 * torch.mean(-1 -var + (mu - prior).pow(2) + var.exp())
    return loss_KLD

def KLD_camera_loss(mu,var, Ks_mu, Ks_var=torch.tensor(1,dtype=float),norm_param=1):
    Ks_mu = Ks_mu.view(Ks_mu.shape[0],-1)
    Ks_mu_s = Ks_mu[:,[0,2,4,5]]/norm_param
    Ks_var = Ks_var.to(mu.device)
    loss_KLD = 0.5 * torch.mean(Ks_var -var + ((mu - Ks_mu_s).pow(2) + var.exp())/Ks_var.exp()-1)
    return loss_KLD

def photo_loss(imageA, imageB, mask, eps=1e-6):
    """
    l2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
    Parameters:
        imageA       --torch.tensor (B, 3, H, W), range (0, 1), RGB order
        imageB       --same as imageA
    """
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss

def tsa_pose_loss(tsaposes):
    #tilt-swing-azimuth pose prior loss
    '''
    tsaposes: (B,16,3)
    '''
    pi = np.pi
    '''
    max_nonloss = torch.tensor([[3.15,0.01,0.01],
                                [5*pi/180,10*pi/180,100*pi/180],#0
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#3
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#6
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#9
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [90*pi/180,pi/8,pi/8],#12
                                [5*pi/180,5*pi/180,pi/8],
                                [5*pi/180,5*pi/180,100*pi/180]]).float().to(tsaposes.device)
    min_nonloss = torch.tensor([[3.13,-0.01,-0.01],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#0
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#3
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#6
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#9
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [0,-pi/8,-pi/8],#12
                                [-5*pi/180,-5*pi/180,-pi/8],
                                [-5*pi/180,-5*pi/180,-10*pi/180]]).float().to(tsaposes.device)
    '''
    max_nonloss = torch.tensor([[3.15,0.01,0.01],
                                [5*pi/180,10*pi/180,100*pi/180],#0 INDEX
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#3 MIDDLE
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,20*pi/180,100*pi/180],#6 PINKY
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#9 RING
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [90*pi/180,3*pi/16,pi/8],#12 THUMB
                                [5*pi/180,5*pi/180,pi/8],
                                [5*pi/180,5*pi/180,100*pi/180]]).float().to(tsaposes.device)
    min_nonloss = torch.tensor([[3.13,-0.01,-0.01],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#0
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#3
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-20*pi/180,-10*pi/180,-10*pi/180],#6
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#9
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [0,-pi/8,-pi/8],#12
                                [-5*pi/180,-5*pi/180,-pi/8],
                                [-5*pi/180,-5*pi/180,-20*pi/180]]).float().to(tsaposes.device)
    median_nonloss = (max_nonloss+min_nonloss)/2
    #tsa_pose_errors = torch.where(tsaposes>max_nonloss.unsqueeze(0),tsaposes-median_nonloss.unsqueeze(0),torch.zeros_like(tsaposes)) + torch.where(tsaposes<min_nonloss.unsqueeze(0),-tsaposes+median_nonloss.unsqueeze(0),torch.zeros_like(tsaposes))
    tsa_pose_errors = torch.where(tsaposes>max_nonloss.unsqueeze(0),tsaposes-max_nonloss.unsqueeze(0),torch.zeros_like(tsaposes)) + torch.where(tsaposes<min_nonloss.unsqueeze(0),-tsaposes+min_nonloss.unsqueeze(0),torch.zeros_like(tsaposes))
    tsa_pose_loss = torch.mean(tsa_pose_errors.mul(torch.tensor([1,1,2]).float().to(tsa_pose_errors.device)))#.cpu()
    #import pdb; pdb.set_trace()
    return tsa_pose_loss