"""
add image features

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy as np
import torch
from torch import nn
from .bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler
from AMVUR.utils.mano_utils import rot_pose_beta_to_mesh
from .bert.modeling_bert import BertLayerNorm as LayerNormClass
import AMVUR.modeling.data.config as cfg
from AMVUR.modeling._mano import Mesh
from AMVUR.utils.feature_align import feature_align
from AMVUR.utils.projection_utils import orthographic_projection,proj_func,linear_trans_var,proj_var,camparam2Ks,orthographic_projection_var

class AMVUR_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(AMVUR_Encoder, self).__init__(config)
        self.config = config
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_dim = config.img_feature_dim

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)
        self.cross_attention = CrossAttention(config)

        self.apply(self.init_weights)


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None):

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids[:,21:])

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        img_embedding_output = self.img_embedding(img_feats)

        embeddings = position_embeddings + img_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        ###cross_attention
        embeddings_cross_v = self.cross_attention(embeddings[:,21:,:], embeddings[:,:21,:], attention_mask=attention_mask, head_mask=None)[0]
        embeddings_cross_v = embeddings_cross_v + embeddings[:, 21:, :]
        ###self_attention
        encoder_outputs = self.encoder(embeddings_cross_v,
                extended_attention_mask, head_mask=head_mask)
        sequence_output = torch.cat([embeddings[:,:21,:],encoder_outputs[0]],dim=1)

        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)

        return outputs

class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_q, hidden_states_k,  attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states_q)
        mixed_key_layer = self.key(hidden_states_k)
        mixed_value_layer = self.value(hidden_states_k)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class AMVUR(BertPreTrainedModel):

    def __init__(self, config):
        super(AMVUR, self).__init__(config)
        self.config = config
        self.bert = AMVUR_Encoder(config)
        self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
        if self.config.output_feature_dim==3:
            self.cls_head_var = nn.Linear(config.hidden_size, self.config.output_feature_dim)
            self.residual_var = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
        self.apply(self.init_weights)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,  config.img_feature_dim)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None):
        '''
        # self.bert has three outputs
        # predictions[0]: output tokens
        # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
        # predictions[2]: attentions, if enable "self.config.output_attentions"
        '''
        predictions = self.bert(img_feats=img_feats, input_ids=None, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        pred_score = self.cls_head(predictions[0])
        res_img_feats = self.residual(img_feats)
        pred_score = pred_score + res_img_feats
        if pred_score.shape[-1] == 3:
            pred_score_var = self.cls_head_var(predictions[0])
        if pred_score.shape[-1] == 3:
            if self.config.output_attentions and self.config.output_hidden_states:
                return pred_score, pred_score_var, predictions[1], predictions[-1]
            else:
                return pred_score, pred_score_var
        else:
            if self.config.output_attentions and self.config.output_hidden_states:
                return pred_score, predictions[1], predictions[-1]
            else:
                return pred_score

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

class AMVUR_Hand_Network(torch.nn.Module):

    def __init__(self, args, config, backbone, trans_encoder,trans_texture):
        super(AMVUR_Hand_Network, self).__init__()
        self.config = config
        self.backbone = backbone
        self.mesh_sampler = Mesh()
        self.trans_encoder = trans_encoder
        self.trans_texture = trans_texture
        self.upsampling = torch.nn.Linear(195, 778)
        self.upsampling_var = torch.nn.Linear(195, 778)
        self.orth_fc = nn.Sequential(*[nn.Linear(2048, 512), nn.Linear(512, 3)])
        self.orth_fc_var = nn.Sequential(*[nn.Linear(2048, 512), nn.Linear(512, 3)])
        self.pose_projection1_ = torch.nn.Linear(2048, 512)
        self.pose_projection2_ = torch.nn.Linear(512, 40*2)
        self.samp_3D_scale = nn.Sequential(*[nn.Linear((778+21)*3, 512), nn.Linear(512, 1)])
        self.samp_2D_scale = nn.Sequential(*[nn.Linear(21*2, 1)])
        # rot layers
        self.rot_reg = nn.Sequential(*[nn.Linear(2048, 512), nn.Linear(512, 3)])
        # scale layers
        self.scale_reg = nn.Sequential(*[nn.Linear(2048, 512), nn.Linear(512, 1)])
        # Trans layers
        self.trans_reg = nn.Sequential(*[nn.Linear(2048, 512), nn.Linear(512, 3)])
        # Ks layers
        self.ks_reg = nn.Sequential(*[nn.Linear(2048, 512), nn.Linear(512, 4)])
        self.ks_var_reg = nn.Sequential(*[nn.Linear(2048, 512), nn.Linear(512, 4)])
        self.samp_ks_scale = nn.Sequential(*[nn.Linear(4, 1)])


        self.init_weights()

    def init_weights(self):
        #import pdb; pdb.set_trace()
        #len_scale_reg = len(self.scale_reg)
        '''
        for m in self.scale_reg:
            #import pdb; pdb.set_trace()
            if hasattr(m, 'weight'):#remove ReLU
                normal_init(m, std=0.1,bias=0.95)
        '''
        normal_init(self.scale_reg[0],std=0.001)
        normal_init(self.scale_reg[1],std=0.001,bias=0.95)

        normal_init(self.trans_reg[0],std=0.001)
        normal_init(self.trans_reg[1],std=0.001)
        nn.init.constant_(self.trans_reg[1].bias[2],0.65)

        normal_init(self.ks_reg[0], std=0.001)
        normal_init(self.ks_reg[1], std=0.001)
        nn.init.constant_(self.ks_reg[1].bias[0], 0.45)
        nn.init.constant_(self.ks_reg[1].bias[1], 0.1)
        nn.init.constant_(self.ks_reg[1].bias[2], 0.45)


    def reparameterize(self, mu, log_var, s=0.001):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)# standard deviation
        eps = torch.randn_like(std)# `randn_like` as we need the same size
        sample = mu + (eps * std)*s# sampling as if coming from the input space
        return sample

    def inverse_interpolation(self, spatial_feat, vertices_2d_norm):
        outputs = []
        for feat_map in spatial_feat:
            outputs.append(feature_align(feat_map, vertices_2d_norm, feat_map.shape[2:4]))
        vertices_feature = torch.cat(outputs, dim=1)
        vertices_feature = vertices_feature.permute(0, 2, 1)
        return vertices_feature

    def occlusion_aware_texture_regression(self, images, spatial_feat, vertices, cam_param, renderer, mano_face,
                                           ortho_proj=True):
        device = vertices.device
        res = images.size(-1)
        if ortho_proj:
            vertices_2d = orthographic_projection(vertices.contiguous(), cam_param.contiguous())
            vertices_2d_norm = ((vertices_2d + 1) * 0.5) * images.shape[-1]
            focal_length = 1000
            camera_t = torch.stack(
                [cam_param[:, 1], cam_param[:, 2], 2 * focal_length / (res * torch.absolute(cam_param[:, 0]) + 1e-9)],
                -1)
            ###convert to openGL coordinate
            vertices = vertices * torch.tensor([-1, -1, 1], device=device)
            vertices = vertices.float().to(device)
            camera_t = camera_t * torch.tensor([-1, -1, 1], device=device)
            camera_t = camera_t.float().to(device)
            znear = 0.1
            dist = torch.abs(camera_t[:, 2] - torch.mean(vertices, axis=1)[:, 2])
            zfar = dist + 20
            camera_center = res / 2
            fov = 2 * np.arctan(camera_center / focal_length) * 180 / np.pi
            renderer.camera_config_fov(fov, res, znear, zfar, camera_t, has_lights=True)
        else:
            cam_param=cam_param[1]
            Ks_i = camparam2Ks(cam_param, images.shape[-1] * 2)
            vertices_2d_norm = proj_func(vertices, Ks_i) / res
            ###convert to openGL coordinate
            vertices = vertices * torch.from_numpy(np.asarray([-1, -1, 1])).unsqueeze(0).float().to(device)
            focal_length = cam_param[:, [0, 2]]*images.shape[-1] * 2
            camera_center = cam_param[:, [1, 3]]*images.shape[-1] * 2
            camera_t = torch.from_numpy(np.array([0, 0, 0])).unsqueeze(0).float().to(device)
            renderer.camera_config(res, focal_length, camera_center, camera_t, has_lights=True)
        vertices_feature = self.inverse_interpolation(spatial_feat, vertices_2d_norm)
        vertices_rgb = self.trans_texture(vertices_feature)[0]
        face_idx = torch.tensor(mano_face).to(device)
        face_idx = face_idx.repeat((vertices.shape[0], 1, 1)).contiguous()
        images, mask = renderer.render_hand_vertex_occlusion_aware(vertices, face_idx, vertices_rgb)
        return images, mask

    def forward(self, images, mesh_model, renderer, ortho_proj=False, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,48))
        template_pose = template_pose.cuda()
        template_betas = torch.zeros((1,10)).cuda()
        template_vertices, template_3d_joints = mesh_model.layer(template_pose, template_betas)
        template_vertices = template_vertices/1000.0
        template_3d_joints = template_3d_joints/1000.0
        template_vertices = self.mesh_sampler.downsample(template_vertices)

        # normalize
        template_root = template_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
        template_3d_joints = template_3d_joints - template_root[:, None, :]
        template_vertices = template_vertices - template_root[:, None, :]
        num_joints = template_3d_joints.shape[1]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # extract global image feature using a CNN backbone
        image_feat_ori, spatial_feat = self.backbone(images)

        rot = self.rot_reg(image_feat_ori)
        scale = self.scale_reg(image_feat_ori)
        trans = self.trans_reg(image_feat_ori)
        predicted_pose_shape = self.pose_projection1_(image_feat_ori)
        predicted_pose_shape = self.pose_projection2_(predicted_pose_shape)
        predicted_pose_shape_mu = predicted_pose_shape[:, :40]
        predicted_pose_shape_var = predicted_pose_shape[:, 40:]
        jv_mu, faces, tsa_poses = rot_pose_beta_to_mesh(rot, predicted_pose_shape_mu[:, :30],
                                                                             torch.zeros_like(predicted_pose_shape_mu[:, 30:]).cuda())
        jv_var, _, _ = rot_pose_beta_to_mesh(rot, predicted_pose_shape_var[:, :30],
                                                                             torch.zeros_like(predicted_pose_shape_var[:, 30:]).cuda())
        #import pdb; pdb.set_trace()
        jv_mu = trans.unsqueeze(1) +torch.abs(scale.unsqueeze(2)) * jv_mu[:, :, :]
        jv_var = torch.log(scale.unsqueeze(2).square()) + jv_var[:, :, :]
        samp_3D_scale = self.samp_3D_scale(jv_var.view(batch_size,-1))
        jv_samp = self.reparameterize(jv_mu,jv_var,samp_3D_scale[:,:,None])
        pred_vertices_mano_samp = jv_samp[:,21:]
        pred_vertices_mano_mu = jv_mu[:,21:]
        pred_vertices_mano_var = jv_var[:,21:]

        pred_3d_joints_mano_from_mesh_mu = mesh_model.get_3d_joints(pred_vertices_mano_mu)
        pred_3d_joints_mano_from_mesh_var = linear_trans_var(pred_vertices_mano_var.permute(0, 2, 1)[:, :, None, :],
                                                             mesh_model.joint_regressor_torch.expand(batch_size, 3, -1,
                                                                                                     -1))
        pred_3d_joints_mano_from_mesh_samp = self.reparameterize(pred_3d_joints_mano_from_mesh_mu, pred_3d_joints_mano_from_mesh_var,samp_3D_scale[:,:,None])

        # concatinate image feat and template mesh
        image_feat = image_feat_ori.view(batch_size, 1, -1).expand(-1, ref_vertices.shape[-2], -1)
        features = torch.cat([ref_vertices, image_feat], dim=2)

        if is_train==True:
            constant_tensor = torch.ones_like(features).cuda()*0.01
            features = features*meta_masks + constant_tensor*(1-meta_masks)

        # forward pass
        features_mu, features_var = self.trans_encoder(features)
        if ortho_proj:
            features_mu = features_mu[:, :, :]
        else:
            features_mu = trans.unsqueeze(1) + features_mu[:, :, :]

        features_var = features_var[:,:,:]

        # features = features + ref_vertices
        pred_vertices_mu = features_mu[:,num_joints:,:]
        pred_vertices_var = features_var[:,num_joints:,:]
        pred_vertices_mu = self.upsampling(pred_vertices_mu.transpose(1,2)).transpose(1,2)
        pred_vertices_var = self.upsampling_var(pred_vertices_var.transpose(1, 2)).transpose(1, 2)
        pred_vertices_samp = self.reparameterize(pred_vertices_mu,pred_vertices_var,samp_3D_scale[:,:,None])

        pred_3d_joints_from_mesh_mu = mesh_model.get_3d_joints(pred_vertices_mu)
        pred_3d_joints_from_mesh_var = linear_trans_var(pred_vertices_var.permute(0, 2, 1)[:, :, None, :],
                                                             mesh_model.joint_regressor_torch.expand(batch_size, 3, -1,
                                                                                                     -1))
        pred_3d_joints_from_mesh_samp = self.reparameterize(pred_3d_joints_from_mesh_mu, pred_3d_joints_from_mesh_var,
                                                            samp_3D_scale[:, :, None])

        if ortho_proj:
            ## orthographic_projection
            camera_param = self.orth_fc(image_feat_ori)
            camera_param_var = self.orth_fc_var(image_feat_ori)
            pred_2d_joints_mano_from_mesh_mu = orthographic_projection(pred_3d_joints_mano_from_mesh_mu.contiguous(),
                                                               camera_param.contiguous())
            pred_2d_joints_mano_from_mesh_var = orthographic_projection(pred_3d_joints_mano_from_mesh_var.contiguous(),
                                                               camera_param_var.contiguous())
            samp_2D_scale = self.samp_2D_scale(pred_2d_joints_mano_from_mesh_var.contiguous().view(batch_size,-1))
            pred_2d_joints_mano_from_mesh_samp = self.reparameterize(pred_2d_joints_mano_from_mesh_mu,
                                                                     pred_2d_joints_mano_from_mesh_var,
                                                                     samp_2D_scale[:, :, None])
            pred_2d_joints_from_mesh_mu = orthographic_projection(pred_3d_joints_from_mesh_mu.contiguous(),
                                                               camera_param.contiguous())
            pred_2d_joints_from_mesh_var = orthographic_projection_var(pred_3d_joints_from_mesh_var.contiguous(),
                                                               camera_param.contiguous())
            pred_2d_joints_from_mesh_samp = self.reparameterize(pred_2d_joints_from_mesh_mu, pred_2d_joints_from_mesh_var, samp_2D_scale[:,:,None])
        else:
            ## Ks
            camera_param_mu = self.ks_reg(image_feat_ori)
            camera_param_var = self.ks_var_reg(image_feat_ori)
            camera_param_sample_scale = self.samp_ks_scale(camera_param_var)
            camera_param_sample = self.reparameterize(camera_param_mu, camera_param_var, camera_param_sample_scale)
            camera_param_sample = torch.abs(camera_param_sample)
            camera_param_mu = torch.abs(camera_param_mu)
            Ks = camparam2Ks(camera_param_sample, images.shape[-1] * 2)
            camera_param = [camera_param_sample, camera_param_mu, camera_param_var]
            pred_2d_joints_mano_from_mesh_mu = proj_func(pred_3d_joints_mano_from_mesh_mu, Ks)/images.shape[-1]
            pred_2d_joints_mano_from_mesh_var = proj_var(pred_3d_joints_mano_from_mesh_mu, pred_3d_joints_mano_from_mesh_var, Ks)/images.shape[-1]
            samp_2D_scale = self.samp_2D_scale(pred_2d_joints_mano_from_mesh_var.contiguous().view(batch_size,-1))
            pred_2d_joints_mano_from_mesh_samp = self.reparameterize(pred_2d_joints_mano_from_mesh_mu, pred_2d_joints_mano_from_mesh_var, samp_2D_scale[:,:,None])
            pred_2d_joints_from_mesh_mu = proj_func(pred_3d_joints_from_mesh_mu, Ks)/images.shape[-1]
            pred_2d_joints_from_mesh_var = proj_var(pred_3d_joints_from_mesh_mu, pred_3d_joints_from_mesh_var, Ks)/images.shape[-1]
            pred_2d_joints_from_mesh_samp = self.reparameterize(pred_2d_joints_from_mesh_mu, pred_2d_joints_from_mesh_var, samp_2D_scale[:,:,None])

        # AMVUR results
        pred_3d_joints = [pred_3d_joints_from_mesh_samp, pred_3d_joints_from_mesh_mu, pred_3d_joints_from_mesh_var]
        pred_vertices = [pred_vertices_samp, pred_vertices_mu, pred_vertices_var]
        pred_2d_joints = [pred_2d_joints_from_mesh_samp, pred_2d_joints_from_mesh_mu, pred_2d_joints_from_mesh_var]
        AMVUR_results = [pred_3d_joints,pred_vertices,pred_2d_joints]
        # mano results
        pred_3d_joints_mano = [pred_3d_joints_mano_from_mesh_samp, pred_3d_joints_mano_from_mesh_mu, pred_3d_joints_mano_from_mesh_var]
        pred_vertices_mano = [pred_vertices_mano_samp, pred_vertices_mano_mu, pred_vertices_mano_var]
        pred_2d_joints_mano = [pred_2d_joints_mano_from_mesh_samp, pred_2d_joints_mano_from_mesh_mu, pred_2d_joints_mano_from_mesh_var]
        MANO_results = [pred_3d_joints_mano,pred_vertices_mano,pred_2d_joints_mano,tsa_poses,jv_var]

        ### texture regression
        rendered_image, mask = self.occlusion_aware_texture_regression(images, spatial_feat, pred_vertices_mano_mu, camera_param,
                                                                renderer, mesh_model.face, ortho_proj)


        return camera_param, AMVUR_results, MANO_results, [rendered_image, mask]
