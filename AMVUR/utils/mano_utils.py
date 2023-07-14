import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pickle
import numpy as np
import os
# mano part from https://github.com/boukhayma/3dhand
#-------------------
# Mano in Pytorch
#-------------------
bases_num = 10 
#pose_num = 6
pose_num = 30
mesh_num = 778
keypoints_num = 16


def rodrigues(r):       
    theta = torch.sqrt(torch.sum(torch.pow(r, 2),1))  

    def S(n_):   
        ns = torch.split(n_, 1, 1)     
        Sn_ = torch.cat([torch.zeros_like(ns[0]),-ns[2],ns[1],ns[2],torch.zeros_like(ns[0]),-ns[0],-ns[1],ns[0],torch.zeros_like(ns[0])], 1)
        Sn_ = Sn_.view(-1, 3, 3)      
        return Sn_    

    n = r/(theta.view(-1, 1))   
    Sn = S(n) 

    #R = torch.eye(3).unsqueeze(0) + torch.sin(theta).view(-1, 1, 1)*Sn\
    #        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)
    
    I3 = Variable(torch.eye(3).unsqueeze(0).to(r.device))

    R = I3 + torch.sin(theta).view(-1, 1, 1)*Sn\
        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)

    Sr = S(r)
    theta2 = theta**2     
    R2 = I3 + (1.-theta2.view(-1,1,1)/6.)*Sr\
        + (.5-theta2.view(-1,1,1)/24.)*torch.matmul(Sr,Sr)
    
    idx = np.argwhere((theta<1e-30).data.cpu().numpy())

    if (idx.size):
        R[idx,:,:] = R2[idx,:,:]

    return R,Sn

def get_poseweights(poses, bsize):
    # pose: batch x 24 x 3                                                    
    pose_matrix, _ = rodrigues(poses[:,1:,:].contiguous().view(-1,3))
    #pose_matrix, _ = rodrigues(poses.view(-1,3))    
    pose_matrix = pose_matrix - Variable(torch.from_numpy(np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0),bsize*(keypoints_num-1),axis=0)).to(device=poses.device))
    pose_matrix = pose_matrix.view(bsize, -1)
    return pose_matrix


def rot_pose_beta_to_mesh(rots, poses, betas):
    # import pdb; pdb.set_trace()
    # dd = pickle.load(open('examples/data/MANO_RIGHT.pkl', 'rb'),encoding='latin1')
    # MANO_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data/MANO_RIGHT.pkl')
    MANO_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modeling/data/MANO_RIGHT.pkl')
    dd = pickle.load(open(MANO_file, 'rb'), encoding='latin1')
    kintree_table = dd['kintree_table']
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}

    devices = rots.device

    mesh_mu = Variable(
        torch.from_numpy(np.expand_dims(dd['v_template'], 0).astype(np.float32)).to(device=devices))  # zero mean
    mesh_pca = Variable(torch.from_numpy(np.expand_dims(dd['shapedirs'], 0).astype(np.float32)).to(device=devices))
    posedirs = Variable(torch.from_numpy(np.expand_dims(dd['posedirs'], 0).astype(np.float32)).to(device=devices))
    J_regressor = Variable(
        torch.from_numpy(np.expand_dims(dd['J_regressor'].todense(), 0).astype(np.float32)).to(device=devices))
    weights = Variable(torch.from_numpy(np.expand_dims(dd['weights'], 0).astype(np.float32)).to(device=devices))
    hands_components = Variable(
        torch.from_numpy(np.expand_dims(np.vstack(dd['hands_components'][:pose_num]), 0).astype(np.float32)).to(
            device=devices))
    hands_mean = Variable(torch.from_numpy(np.expand_dims(dd['hands_mean'], 0).astype(np.float32)).to(device=devices))
    root_rot = Variable(torch.FloatTensor([np.pi, 0., 0.]).unsqueeze(0).to(device=devices))

    mesh_face = Variable(torch.from_numpy(np.expand_dims(dd['f'], 0).astype(np.int16)).to(device=devices))

    # import pdb; pdb.set_trace()

    batch_size = rots.size(0)

    mesh_face = mesh_face.repeat(batch_size, 1, 1)
    # import pdb; pdb.set_trace()
    poses = (hands_mean + torch.matmul(poses.unsqueeze(1), hands_components).squeeze(1)).view(batch_size,
                                                                                              keypoints_num - 1, 3)
    # [b,15,3] [0:3]index [3:6]mid [6:9]pinky [9:12]ring [12:15]thumb

    # import pdb; pdb.set_trace()

    # for visualization
    # rots = torch.zeros_like(rots); rots[:,0]=np.pi/2

    # poses = torch.ones_like(poses)*1
    # poses = torch.cat((poses[:,:3].contiguous().view(batch_size,1,3),poses_),1)
    poses = torch.cat((root_rot.repeat(batch_size, 1).view(batch_size, 1, 3), poses), 1)  # [b,16,3]

    v_shaped = (torch.matmul(betas.unsqueeze(1),
                             mesh_pca.repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2).contiguous().view(batch_size,
                                                                                                        bases_num,
                                                                                                        -1)).squeeze(1)
                + mesh_mu.repeat(batch_size, 1, 1).view(batch_size, -1)).view(batch_size, mesh_num, 3)

    pose_weights = get_poseweights(poses, batch_size)  # [b,135]

    v_posed = v_shaped + torch.matmul(posedirs.repeat(batch_size, 1, 1, 1),
                                      (pose_weights.view(batch_size, 1, (keypoints_num - 1) * 9, 1)).repeat(1, mesh_num,
                                                                                                            1,
                                                                                                            1)).squeeze(
        3)

    J_posed = torch.matmul(v_shaped.permute(0, 2, 1), J_regressor.repeat(batch_size, 1, 1).permute(0, 2, 1))
    J_posed = J_posed.permute(0, 2, 1)
    J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]

    pose = poses.permute(1, 0, 2)
    pose_split = torch.split(pose, 1, 0)
    # import pdb; pdb.set_trace()

    angle_matrix = []
    for i in range(keypoints_num):
        out, tmp = rodrigues(pose_split[i].contiguous().view(-1, 3))
        angle_matrix.append(out)

    # with_zeros = lambda x: torch.cat((x,torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1)),1)

    with_zeros = lambda x: \
        torch.cat(
            (x, Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size, 1, 1).to(device=devices))), 1)

    pack = lambda x: torch.cat((Variable(torch.zeros(batch_size, 4, 3).to(device=devices)), x), 2)

    results = {}
    results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size, 3, 1)), 2))

    for i in range(1, kintree_table.shape[1]):
        tmp = with_zeros(torch.cat((angle_matrix[i],
                                    (J_posed_split[i] - J_posed_split[parent[i]]).view(batch_size, 3, 1)), 2))
        results[i] = torch.matmul(results[parent[i]], tmp)

    results_global = results

    results2 = []

    for i in range(len(results)):
        vec = (torch.cat((J_posed_split[i], Variable(torch.zeros(batch_size, 1).to(device=devices))), 1)).view(
            batch_size, 4, 1)
        results2.append((results[i] - pack(torch.matmul(results[i], vec))).unsqueeze(0))

    results = torch.cat(results2, 0)

    T = torch.matmul(results.permute(1, 2, 3, 0),
                     weights.repeat(batch_size, 1, 1).permute(0, 2, 1).unsqueeze(1).repeat(1, 4, 1, 1))
    Ts = torch.split(T, 1, 2)
    rest_shape_h = torch.cat((v_posed, Variable(torch.ones(batch_size, mesh_num, 1).to(device=devices))), 2)
    rest_shape_hs = torch.split(rest_shape_h, 1, 2)

    v = Ts[0].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1, mesh_num) \
        + Ts[1].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1, mesh_num) \
        + Ts[2].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1, mesh_num) \
        + Ts[3].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1, mesh_num)

    # v = v.permute(0,2,1)[:,:,:3]
    Rots = rodrigues(rots)[0]
    # import pdb; pdb.set_trace()
    Jtr = []

    for j_id in range(len(results_global)):
        Jtr.append(results_global[j_id][:, :3, 3:4])

    # Add finger tips from mesh to joint list
    '''
    Jtr.insert(4,v[:,:3,333].unsqueeze(2))
    Jtr.insert(8,v[:,:3,444].unsqueeze(2))
    Jtr.insert(12,v[:,:3,672].unsqueeze(2))
    Jtr.insert(16,v[:,:3,555].unsqueeze(2))
    Jtr.insert(20,v[:,:3,745].unsqueeze(2)) 
    '''
    # For FreiHand
    Jtr.insert(4, v[:, :3, 320].unsqueeze(2))
    Jtr.insert(8, v[:, :3, 443].unsqueeze(2))
    Jtr.insert(12, v[:, :3, 672].unsqueeze(2))
    Jtr.insert(16, v[:, :3, 555].unsqueeze(2))
    Jtr.insert(20, v[:, :3, 744].unsqueeze(2))

    Jtr = torch.cat(Jtr, 2)  # .permute(0,2,1)

    # v = torch.matmul(Rots,v[:,:3,:]).permute(0,2,1) #.contiguous().view(batch_size,-1)
    # Jtr = torch.matmul(Rots,Jtr).permute(0,2,1) #.contiguous().view(batch_size,-1)
    v = torch.matmul(Rots, v[:, :3, :]).permute(0, 2, 1)  # .contiguous().view(batch_size,-1)
    Jtr = torch.matmul(Rots, Jtr).permute(0, 2, 1)  # .contiguous().view(batch_size,-1)

    # return torch.cat((Jtr,v), 1)
    return torch.cat((Jtr, v), 1), mesh_face, poses


