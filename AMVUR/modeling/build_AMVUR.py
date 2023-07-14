import torch
from AMVUR.modeling.bert import BertConfig
from AMVUR.modeling.modeling_AMVUR import AMVUR_Hand_Network as AMVUR_Network
from AMVUR.modeling.modeling_AMVUR import AMVUR
from AMVUR.modeling.hrnet.config import config as hrnet_config
from AMVUR.modeling.hrnet.hrnet_cls_net import get_cls_net
from AMVUR.modeling.hrnet.config import update_config as hrnet_update_config
import torchvision.models as models
def build_AMVUR(args,logger):
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

    if args.run_eval_only == True and args.resume_checkpoint != None and args.resume_checkpoint != 'None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _AMVUR_network = torch.load(args.resume_checkpoint)

    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, AMVUR
            config = config_class.from_pretrained(args.config_name if args.config_name \
                                                      else args.model_name_or_path)

            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size * 4)

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            logger.info("Init model from scratch.")
            trans_encoder.append(model)
        texture_config = BertConfig
        texture_config = texture_config.from_pretrained(args.config_name if args.config_name \
                                                  else args.model_name_or_path)
        texture_config.output_attentions = False
        texture_config.max_position_embeddings = 1024
        texture_config.hidden_dropout_prob = 0.1
        texture_config.img_feature_dim = 448
        texture_config.output_feature_dim = 3
        texture_config.num_hidden_layers = 2
        texture_config.hidden_size = 216
        texture_config.num_hidden_layers = 4
        texture_config.intermediate_size = 216
        trans_texture = AMVUR(texture_config)

        # create backbone model
        if args.arch == 'hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch == 'hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Transformers total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        _AMVUR_network = AMVUR_Network(args, config, backbone, trans_encoder, trans_texture)

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            cpu_device = torch.device('cpu')
            state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
            _AMVUR_network.load_state_dict(state_dict, strict=False)
            del state_dict
        return _AMVUR_network