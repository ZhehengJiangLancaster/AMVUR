import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_txt", default='HOD3/train.txt', type=str, required=False,
                        help="txt file with all data for training.")
    parser.add_argument("--val_txt", default='HOD3/test.txt', type=str, required=False,
                        help="txt file with all data for validation.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int,
                        help="adjust image resolution.")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='AMVUR/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/HO3D/test/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument('-a', '--arch', default='resnet50',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float,
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=1.0, type=float)
    parser.add_argument("--joints_loss_weight", default=1.0, type=float)
    parser.add_argument("--vloss_w_full", default=0.5, type=float)
    parser.add_argument("--vloss_w_sub", default=0.5, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument("--num_hidden_layers", default=-1, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=-1, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False,
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str,
                        help="The Image Feature Dimension.")
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',)
    parser.add_argument("--multiscale_inference", default=False, action='store_true',)
    # if enable "multiscale_inference", dataloader will apply transformations to the test image based on
    # the rotation "rot" and scale "sc" parameters below
    parser.add_argument("--rot", default=0, type=float)
    parser.add_argument("--sc", default=1.0, type=float)
    parser.add_argument("--aml_eval", default=False, action='store_true',)

    parser.add_argument('--logging_steps', type=int, default=1,
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88,
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    parser.add_argument('--config_json', type=str, default=None)
    args = parser.parse_args()
    return args