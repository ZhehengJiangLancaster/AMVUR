"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import json
import time
import datetime
import torch
from torchvision.utils import make_grid
import numpy as np
import cv2
from AMVUR.modeling._mano import MANO, Mesh
import AMVUR.modeling.data.config as cfg
from AMVUR.modeling.build_AMVUR import build_AMVUR
from AMVUR.datasets.build import make_HO3D_v2_data_loader
from experiments.train_eval_options import parse_args
from AMVUR.utils.logger import setup_logger
from AMVUR.utils.comm import synchronize, is_main_process, get_rank
from AMVUR.utils.miscellaneous import mkdir, set_seed
from AMVUR.utils.metric_logger import AverageMeter
from AMVUR.utils.renderer import Renderer,visualize_reconstruction
from AMVUR.utils.loss_function import keypoint_2d_loss,keypoint_3d_loss,vertices_loss,KLD_vertices_loss,KLD_3d_joints_loss,KLD_camera_loss,tsa_pose_loss,pose_var_loss,photo_loss,reconstruction_loss

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

class UnNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_batch):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        unnorm_tensor= tensor_batch.clone()
        for batch in range(unnorm_tensor.shape[0]):
            tensor = unnorm_tensor[batch,:,:,:]
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return unnorm_tensor

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run(args, train_dataloader, METRO_model, mano_model, renderer, mesh_sampler):

    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs

    optimizer = torch.optim.Adam(params=list(METRO_model.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)

    # define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    if args.distributed:
        METRO_model = torch.nn.parallel.DistributedDataParallel(
            METRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    start_training_time = time.time()
    end = time.time()
    METRO_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    log_loss_vertices = AverageMeter()
    log_loss_reconstruction = AverageMeter()
    log_loss_KLD = AverageMeter()
    log_loss_texture = AverageMeter()
    iteration_time = time.time()

    for iteration, (img_keys, images, annotations) in enumerate(train_dataloader):

        METRO_model.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        images = images.cuda()
        gt_2d_joints = annotations['joints_2d_screen'].cuda()
        gt_pose = annotations['pose'].cuda()
        gt_betas = annotations['betas'].cuda()
        Ks_crop = annotations['Ks_crop'].cuda()
        has_mesh = annotations['has_mano'].cuda()
        has_3d_joints = has_mesh
        has_2d_joints = has_mesh
        mjm_mask = annotations['mjm_mask'].cuda()
        mvm_mask = annotations['mvm_mask'].cuda()
        # generate mesh
        gt_vertices, gt_3d_joints = mano_model.layer(gt_pose, gt_betas)
        gt_vertices = gt_vertices/1000.0
        gt_3d_joints = gt_3d_joints/1000.0

        # normalize gt based on hand's wrist 
        gt_3d_root = gt_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
        gt_vertices = gt_vertices - gt_3d_root[:, None, :]
        gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
        gt_3d_joints_with_tag = torch.ones((batch_size,gt_3d_joints.shape[1],4)).cuda()
        gt_3d_joints_with_tag[:,:,:3] = gt_3d_joints

        # prepare masks for mask vertex/joint modeling
        mjm_mask_ = mjm_mask.expand(-1,-1,2051)
        mvm_mask_ = mvm_mask.expand(-1,-1,2051)
        meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)
        
        # Transformer
        pred_camera, AMVUR_results, MANO_results, rendered_image = METRO_model(images, mano_model, renderer, ortho_proj=False, meta_masks=meta_masks, is_train=True)
        pred_vertices_samp, pred_vertices_mu, pred_vertices_var = AMVUR_results[1]
        pred_3d_joints_samp, pred_3d_joints_mu, pred_3d_joints_var = AMVUR_results[0]
        ###Mano
        pred_vertices_mano,pred_vertices_mano_mu,pred_vertices_mano_var = MANO_results[1]
        pred_3d_joints_mano, pred_3d_joints_mano_mu, pred_3d_joints_mano_var = MANO_results[0]

        pred_2d_joints = AMVUR_results[2][0]
        pred_2d_joints_mano = MANO_results[2][0]

        rendered_hand = rendered_image[0]
        mask = rendered_image[1]

        loss_vertices = vertices_loss(criterion_vertices, pred_vertices_mano_mu, gt_vertices, has_mesh)
        loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_mano_mu, gt_3d_joints_with_tag, has_3d_joints)
        loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_mano, gt_2d_joints, has_2d_joints)

        loss_reconstruction = reconstruction_loss(criterion_2d_keypoints, mano_model, pred_camera[0], MANO_results[3],
                                                  pred_vertices_mano, pred_3d_joints_mano, pred_2d_joints_mano,
                                                  gt_2d_joints, has_2d_joints)
        loss_KLD = KLD_3d_joints_loss(pred_3d_joints_mu,pred_3d_joints_var,pred_3d_joints_mano_mu,pred_3d_joints_mano_var)+\
                   KLD_vertices_loss(pred_vertices_mu, pred_vertices_var, pred_vertices_mano_mu,pred_vertices_mano_var)+\
                   KLD_camera_loss(pred_camera[1],pred_camera[2],Ks_crop,norm_param=images.shape[-1]*2)+\
                   pose_var_loss(MANO_results[4])

        # compute texture loss
        loss_texture = photo_loss(images, rendered_hand, mask)
            
        # we empirically use hyperparameters to balance difference losses
        if epoch<=20:
            loss = loss_reconstruction + loss_KLD*0.1 +loss_texture*0
        else:
            loss = loss_reconstruction + loss_KLD * 0.1 + loss_texture * 0.0001

        # update logs
        log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
        log_loss_vertices.update(loss_vertices.item(), batch_size)
        log_losses.update(loss.item(), batch_size)
        log_loss_reconstruction.update(loss_reconstruction.item(), batch_size)
        log_loss_KLD.update(loss_KLD.item(), batch_size)
        log_loss_texture.update(loss_texture.item(), batch_size)

        # back prop
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'speed:{speed}','max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, speed = round(end-iteration_time,3),
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                + '  loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, reconstruction_loss: {:.4f}, KLD_loss:{:.4f}, texture_loss:{:.4f},compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_2djoints.avg, log_loss_3djoints.avg, log_loss_vertices.avg, log_loss_reconstruction.avg,log_loss_KLD.avg,log_loss_KLD.avg, batch_time.avg, data_time.avg,
                    optimizer.param_groups[0]['lr'])
            )
            if iteration % 500 == 1 or iteration == max_iter:
                unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                vis_rendered_hand = rendered_hand * mask + images * (~mask)
                vis_rendered_hand = unorm(vis_rendered_hand.detach())
                visual_imgs = visualize_mesh(   renderer,
                                                annotations['ori_img'].detach(),
                                                annotations['joints_2d'].detach(),
                                                pred_vertices_mano_mu.detach(),
                                                pred_camera[1].detach(),
                                                pred_2d_joints_mano.detach(),
                                                vis_rendered_hand,
                                                mano_face=mano_model.face)
                visual_imgs = visual_imgs.transpose(0,1)
                visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = np.asarray(visual_imgs)

                if is_main_process()==True:
                    stamp = str(epoch) + '_' + str(iteration)
                    temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

        iteration_time = time.time()

        if iteration % iters_per_epoch == 0:
            if epoch%5==1:
                checkpoint_dir = save_checkpoint(METRO_model, args, epoch, iteration)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    checkpoint_dir = save_checkpoint(METRO_model, args, epoch, iteration)

def run_eval_and_save(args, val_dataloader, METRO_model, mano_model, renderer, mesh_sampler):

    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    if args.distributed:
        METRO_model = torch.nn.parallel.DistributedDataParallel(
            METRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    METRO_model.eval()

    run_inference_hand_mesh(args, val_dataloader,
                            METRO_model,
                            criterion_keypoints,
                            criterion_vertices,
                            0,
                            mano_model, mesh_sampler,
                            renderer)
    checkpoint_dir = save_checkpoint(METRO_model, args, 0, 0)
    return

def run_inference_hand_mesh(args, val_loader, METRO_model, criterion, criterion_vertices, epoch, mano_model, mesh_sampler, renderer):
    # switch to evaluate mode
    METRO_model.eval()
    fname_output_save = []
    mesh_output_save = []
    joint_output_save = []
    with torch.no_grad():
        for i, (img_keys, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            # compute output
            images = images.cuda()

            # forward-pass
            pred_camera, AMVUR_results, MANO_results, rendered_images = METRO_model(images, mano_model, renderer, ortho_proj=False)
            rendered_hand = rendered_images[0]
            mask = rendered_images[1]
            # obtain 3d joints from full mesh
            gt_root_joint_cam = annotations['root_coord'].to(MANO_results[1][1].device)
            pred_3d_joints_from_mesh = mano_model.get_3d_joints(MANO_results[1][1])
            pred_3d_wrist = pred_3d_joints_from_mesh[:,cfg.J_NAME.index('Wrist'),:]
            pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_wrist[:, None, :]+gt_root_joint_cam
            pred_vertices_output = MANO_results[1][1] - pred_3d_wrist[:, None, :]+gt_root_joint_cam
            HO3D_joints_name = (
            'Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2',
            'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4',
            'Ring_4', 'Pinky_4')
            # convert to openGL coordinate system.
            pred_vertices_output *= torch.Tensor([1, -1, -1]).to(pred_vertices_output.device)
            pred_3d_joints_from_mesh *= torch.Tensor([1, -1, -1]).to(pred_vertices_output.device)

            for j in range(batch_size):
                fname_output_save.append(img_keys[j])
                pred_vertices_list = pred_vertices_output[j].tolist()
                mesh_output_save.append(pred_vertices_list)
                pred_3d_joints_from_mesh_n = transform_joint_to_other_db(pred_3d_joints_from_mesh[j].cpu(), mano_model.joints_name, HO3D_joints_name)
                pred_3d_joints_from_mesh_list = pred_3d_joints_from_mesh_n.tolist()
                joint_output_save.append(pred_3d_joints_from_mesh_list)

            if i%20==0:
                unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                vis_rendered_hand = rendered_hand * mask + images * (~mask)
                vis_rendered_hand = unorm(vis_rendered_hand.detach())
                visual_imgs = visualize_mesh(   renderer,
                                                annotations['ori_img'].detach(),
                                                annotations['joints_2d'].detach(),
                                                MANO_results[1][1].detach(),
                                                pred_camera[1].detach(),
                                                MANO_results[2][1].detach(),
                                                vis_rendered_hand,
                                                mano_face=mano_model.face)

                visual_imgs = visual_imgs.transpose(0,1)
                visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = np.asarray(visual_imgs)

                inference_setting = 'sc%02d_rot%s'%(int(args.sc*10),str(int(args.rot)))
                temp_fname = args.output_dir + 'HO3D_results_'+inference_setting+'_batch'+str(i)+'.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

    print('save results to pred.json')
    with open('pred.json', 'w') as f:
        json.dump([joint_output_save, mesh_output_save], f)

    resolved_submit_cmd = 'zip ' + args.output_dir + 'pred.zip  ' +  'pred.json'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    resolved_submit_cmd = 'rm pred.json'
    print(resolved_submit_cmd)
    os.system(resolved_submit_cmd)
    return 

def visualize_mesh( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d,
                    rendered_hand,
                    mano_face):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(21))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 10 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        texture_hand = rendered_hand[i].cpu().numpy().transpose(1, 2, 0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        vertices = pred_vertices[i].unsqueeze(0)
        cam = pred_camera[i].unsqueeze(0)*224*2
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer,mano_face)
        rend_img = np.concatenate((img,rend_img, texture_hand), axis=1)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        print("Init distributed training on local rank {}".format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        synchronize()
   
    mkdir(args.output_dir)
    logger = setup_logger("METRO", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mano model
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.cuda()
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(device=args.device)
    _AMVUR_network = build_AMVUR(args, logger)
    _AMVUR_network.to(args.device)
    logger.info("Training parameters %s", args)

    if args.run_eval_only==True:
        val_dataloader = make_HO3D_v2_data_loader(args, args.val_txt,
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run_eval_and_save(args, val_dataloader, _AMVUR_network, mano_model, renderer, mesh_sampler)

    else:
        train_dataloader = make_HO3D_v2_data_loader(args, args.train_txt,
                                            args.distributed, is_train=True, scale_factor=args.img_scale_factor)
        run(args, train_dataloader, _AMVUR_network, mano_model, renderer, mesh_sampler)

if __name__ == "__main__":
    args = parse_args()
    if args.config_json is not None:
        with open(args.config_json, "r") as f:
            json_dic = json.load(f)
            for parse_key, parse_value in json_dic.items():
                vars(args)[parse_key] = parse_value
    main(args)
