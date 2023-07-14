"""
I have added mask images here
mask images are in ./datasets/FreiHAND_pub_v2

"""


import cv2
import math
import json
from PIL import Image
import os.path as op
import numpy as np
import pickle
import time
import random
from AMVUR.utils.projection_utils import proj_func, gen_trans_from_patch_cv
from AMVUR.utils.image_ops import img_from_base64, crop_bbshape, flip_img, flip_pose, flip_kp, transform, rot_aa
import torch
import torchvision.transforms as transforms
from AMVUR.modeling._mano import MANO
TRAIN_INDEX = 1000

class HandMeshTSVDataset(object):
    def __init__(self, args, data_dir, img_dir, depth_dir, meta_dir, txt_file, is_train=True, cv2_output=False, scale_factor=1):

        self.args = args
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.meta_dir = meta_dir
        self.txt_file = txt_file
        self.depth_dir = depth_dir
        self.is_train = is_train
        self.cv2_output = cv2_output
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.scale_factor = 0.25 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = 0.4
        self.rot_factor = 90 # Random rotation in the range [-rot_factor, rot_factor]
        self.input_res = 224
        ###mano joints order
        self.joints_definition = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1',
                                'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
        ###HO3D joints order
        # self.joints_definition = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4', 'Ring_4', 'Pinky_4')
        self.root_index = self.joints_definition.index('Wrist')
        self.file_list = None if txt_file is None else self.get_txt_file(txt_file)
        anno_dir = self.data_dir.split('/')
        self.anno_dir = op.join(*anno_dir[:-1], 'annotations',"HO3D_{}_data.json".format(anno_dir[-1]))
        with open(self.anno_dir, 'r') as f:
            self.anno = json.load(f)
        self.mano_model = MANO()

    def get_txt_file(self, txt_file):
        with open(txt_file,'r') as f:
            file_list = [line.rstrip() for line in f]
            return file_list


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise

        if self.args.multiscale_inference == False:
            rot = 0 # rotation
            sc = 1.0 # scaling
        elif self.args.multiscale_inference == True:
            rot = self.args.rot
            sc = self.args.sc

        # if self.is_train:
        #     sc = 1.0
        #     # Each channel is multiplied with a number
        #     # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        #     pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
	    #
        #     # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
        #     rot = min(2*self.rot_factor,
        #             max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
        #
        #     # The scale is multiplied with a number
        #     # in the area [1-scaleFactor,1+scaleFactor]
        #     sc = min(1+self.scale_factor,
        #             max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
        #     # but it is zero with probability 3/5
        #     if np.random.uniform() <= 0.6:
        #         rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img,bb_shape = crop_bbshape(rgb_img, center, scale,
                      [self.input_res, self.input_res], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img,bb_shape

    def j2d_processing(self, kp, center, scale, r, Ks, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2],t = transform(kp[i,0:2]+1, center, scale,
                                  [self.input_res, self.input_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.input_res - 1.
        # flip the x coordinates
        scale_inv = 1/scale
        scale_matrix = torch.tensor([[scale_inv, 0.0, 0.0], [0.0, scale_inv, 0.0], [0.0, 0.0, 1.0]])
        trans1 = center[0] * scale_inv - self.input_res // 2
        trans2 = center[1] * scale_inv - self.input_res // 2
        trans_matrix = torch.tensor([[1.0, 0.0, -trans1], [0.0, 1.0, -trans2], [0.0, 0.0, 1.0]])
        Ks_crop = torch.mm(trans_matrix, torch.mm(scale_matrix,Ks))
        # Ks_crop = torch.mm(torch.from_numpy(np.float32(t)), Ks)
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp, Ks_crop


    def j3d_processing(self, S, r, f):
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

    def pose_processing(self, pose, r, f):
        """Process MANO theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def get_meta(self, idx):
        file = self.file_list[idx].split('/')
        meta_path = op.join(self.data_dir, file[0], self.meta_dir, file[1]+'.pkl')
        with open(meta_path, 'rb') as fi:
            meta = pickle.load(fi)
        return meta

    def get_image(self, idx):
        file = self.file_list[idx].split('/')
        image_file = op.join(self.data_dir, file[0], self.img_dir, file[1]+'.png')
        cv2_im = cv2.imread(image_file)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        return cv2_im

    def get_hand_center(self, meta):
        cam_param = meta['cam_param']
        Ks = torch.Tensor([[cam_param['focal'][0],0,cam_param['princpt'][0]],[0,cam_param['focal'][1],cam_param['princpt'][1]],[0,0,1]]).float()

        if 'bbox' in meta.keys():  # for evaluation set
            uv_hand = torch.tensor([[meta['bbox'][0], meta['bbox'][1]],
                                 [meta['bbox'][0]+meta['bbox'][2], meta['bbox'][1]+meta['bbox'][3]]])
        else:
            j3d = torch.Tensor(meta['joints_coord_cam']).float()
            uv_hand = proj_func(j3d.unsqueeze(0), Ks.unsqueeze(0)).squeeze()

        crop_center = (torch.max(uv_hand, 0)[0] + torch.min(uv_hand, 0)[0]) / 2
        if self.is_train:
            noise = 5 * torch.randn([2])  # torch.normal(0, 20, size=(2))
            crop_center = noise + crop_center
        min_uv = torch.max(torch.min(uv_hand, 0)[0], torch.zeros(2)) - torch.tensor([10.0, 10.0])
        max_uv = torch.min(torch.max(uv_hand, 0)[0], torch.tensor([640.0, 480.0])) + torch.tensor([10.0, 10.0])
        scale_num = 4
        crop_size_best = scale_num * torch.max(max_uv - crop_center, crop_center - min_uv)  # 4*
        crop_size_best = torch.max(crop_size_best)
        crop_size_best = torch.min(torch.max(crop_size_best, torch.ones(1) * 50.0), torch.ones(1) * 640.0)
        scale = crop_size_best / self.input_res
        return crop_center, scale, uv_hand, Ks

    def get_img_info(self, idx):
        if self.hw_tsv is not None:
            line_no = self.get_line_no(idx)
            row = self.hw_tsv[line_no]
            try:
                # json string format with "height" and "width" being the keys
                return json.loads(row[1])[0]
            except ValueError:
                # list of strings representing height and width in order
                hw_str = row[1].split(' ')
                hw_dict = {"height": int(hw_str[0]), "width": int(hw_str[1])}
                return hw_dict

    def get_img_key(self, idx):
        line_no = self.get_line_no(idx)
        # based on the overhead of reading each row.
        if self.hw_tsv:
            return self.hw_tsv[line_no][0]
        elif self.label_tsv:
            return self.label_tsv[line_no][0]
        else:
            return self.img_tsv[line_no][0]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        start = time.time()
        img = self.get_image(idx)#[480,640,3]
        img_key = self.file_list[idx][-4:]
        meta = self.anno['annotations'][idx]
        if self.is_train:
            mano_param = meta['mano_param']
            handJoints3D = np.asarray(meta['joints_coord_cam'])
        else:
            mano_param = {
                'pose': np.zeros(48),
                'shape': np.zeros(10),
                'trans': np.zeros(3)
            }
            handJoints3D = np.zeros((21,3))
        cam_param = meta['cam_param']

        hand_center, scale, hand_uv, Ks = self.get_hand_center(meta)
        if hand_center[0] < 650 and hand_center[1] < 490 and self.is_train:

            has_2d_joints = 1
            has_3d_joints = 1
            joints_2d = np.ones((handJoints3D.shape[0], 3))
            joints_2d[:, 0:2] = np.asarray(hand_uv)
            joints_3d = np.zeros((handJoints3D.shape[0], 4))
            joints_3d[:,0:3] = handJoints3D.copy()

            # Get MANO parameters, if available
            has_mano = np.asarray(1)
            pose = np.asarray(mano_param['pose'])
            betas = np.asarray(mano_param['shape'])
        else:
            # make 2d_loss, 3d_loss and vertex loss become 0
            has_2d_joints = 0
            has_3d_joints = 0
            joints_2d = np.zeros((handJoints3D.shape[0], 3))
            joints_3d = np.zeros((handJoints3D.shape[0], 4))
            joints_3d[:, 0:3] = handJoints3D.copy()

            has_mano = np.asarray(0)
            pose = np.asarray(mano_param['pose'])
            betas = np.asarray(mano_param['shape'])
            if self.is_train:
                hand_center = torch.tensor([img.shape[0]/2, img.shape[1]/2])

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()
        # Process image
        img,bb_shape = self.rgb_processing(img, hand_center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        transfromed_img = self.normalize_img(img)

        # normalize 3d pose by aligning the wrist as the root (at origin)
        if self.is_train:
            root_coord = joints_3d[self.root_index,:-1].copy()
        else:
            root_coord = np.asarray(meta['root_joint_cam'])
        joints_3d_gt = joints_3d[:,:-1].copy()
        joints_3d[:,:-1] = joints_3d[:,:-1] - root_coord[None,:]
        # 3d pose augmentation
        joints_3d_transformed = self.j3d_processing(joints_3d.copy(), rot, flip)
        # 2d pose augmentation
        joints_2d_transformed, Ks_crop = self.j2d_processing(joints_2d.copy(), hand_center, sc*scale, rot, Ks, flip)
        j2d_proj_screen = joints_2d_transformed.copy()
        j2d_proj_screen[:,:-1] = proj_func(torch.tensor(joints_3d_gt, device=Ks_crop.device,dtype=torch.float).unsqueeze(0), Ks_crop.unsqueeze(0)).squeeze().numpy()/img.shape[-1]

        ###################################
        # Masking percantage
        mvm_percent = 0.0 # or 0.05
        ###################################

        mjm_mask = np.ones((21,1))
        if self.is_train:
            num_joints = 21
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_joints) # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_joints),replace=False,size=masked_num)
            mjm_mask[indices,:] = 0.0
        mjm_mask = torch.from_numpy(mjm_mask).float()

        mvm_mask = np.ones((195,1))
        if self.is_train:
            num_vertices = 195
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
            mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()

        meta_data = {}
        meta_data['ori_img'] = img
        meta_data['Ks_crop'] = Ks_crop
        meta_data['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        meta_data['betas'] = torch.from_numpy(betas).float()
        meta_data['root_coord'] = torch.from_numpy(root_coord).float().unsqueeze(0)
        meta_data['cam_param'] = torch.tensor([cam_param['focal'] + cam_param['princpt']]).float().squeeze()
        meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
        meta_data['has_3d_joints'] = has_3d_joints
        meta_data['has_mano'] = has_mano
        meta_data['mjm_mask'] = mjm_mask
        meta_data['mvm_mask'] = mvm_mask

        # Get 2D keypoints and apply augmentation transforms
        meta_data['has_2d_joints'] = has_2d_joints
        meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()
        meta_data['joints_2d_screen'] = j2d_proj_screen

        meta_data['scale'] = float(sc * scale)
        meta_data['center'] = np.asarray(hand_center).astype(np.float32)

        return img_key, transfromed_img, meta_data


class HO3DDataset(HandMeshTSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """
    def __init__(self, args, txt, is_train=True, cv2_output=False, scale_factor=1):
        data_dir = args.data_dir + '/' + txt[:-4]
        img_file = 'rgb'
        depth_file = 'depth'
        meta_file = 'meta'
        txt_file = args.data_dir + '/' + txt

        super(HO3DDataset, self).__init__(
            args, data_dir, img_file, depth_file, meta_file, txt_file, is_train, cv2_output=cv2_output, scale_factor=scale_factor)
