import os
import os.path as osp
import argparse
import yaml
from utils import update_values
from utils import int_tuple, float_tuple, str_tuple, bool_flag

COCO_DIR = os.path.expanduser('data/coco')

parser = argparse.ArgumentParser()

# Optimization hyperparameters
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
# by default, it is disabled
parser.add_argument('--decay_lr_epochs', default=200, type=float)
parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--eval_epochs', default=1, type=int)
parser.add_argument('--eval_mode_after', default=40, type=int)
parser.add_argument('--disable_l1_loss_after', default=200, type=int)
parser.add_argument('--path_opts', type=str,
                    default='options/vg_baseline_small.yaml', help="Options.")
# Dataset options common to both VG and COCO
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=5000, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# Generator losses
parser.add_argument('--mask_loss_weight', type=float)
parser.add_argument('--l1_pixel_loss_weight', type=float)
parser.add_argument('--bbox_pred_loss_weight', type=float)

# Discriminator Losses
parser.add_argument('--d_loss_weight', type=float)
# multiplied by d_loss_weight
parser.add_argument('--d_obj_weight', type=float)
parser.add_argument('--ac_loss_weight', type=float)
# multiplied by d_loss_weight
parser.add_argument('--d_img_weight', type=float)

# Output options
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--visualize_every', type=int,
                    default=100, help="visualize to visdom.")
parser.add_argument('--timing', action="store_true")
parser.add_argument('--output_dir', type=str)
parser.add_argument('--log_suffix', type=str)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--resume', type=str)
parser.add_argument('--evaluate', action='store_true',
    help="Set to evaluate the model.")
parser.add_argument('--evaluate_train', action='store_true',
    help="Set to evaluate the training set.")

# For inference in run_model.py
parser.add_argument('--scene_graphs_json',
                    default='samples/scene_graph_crops/test.json')
parser.add_argument('--output_demo_dir', default='output/results')
parser.add_argument('--samples_path', default='samples')
parser.add_argument('--draw_scene_graphs', type=int, default=0)

# For saving crops
parser.add_argument('--sv_crops', default=0, type=int)

args = parser.parse_args()

options = {
    "data": {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "data_opts": {
            "num_train_samples": args.num_train_samples,
            "num_val_samples": args.num_val_samples,
        },
    },
    "optim": {
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "eval_epochs": args.eval_epochs,
        # generative model
        "mask_loss_weight": args.mask_loss_weight,
        "l1_pixel_loss_weight": args.l1_pixel_loss_weight,
        "bbox_pred_loss_weight": args.bbox_pred_loss_weight,
        # discriminator
        "d_loss_weight": args.d_loss_weight,
        "d_obj_weight": args.d_obj_weight,
        "ac_loss_weight": args.ac_loss_weight,
        "d_img_weight": args.d_img_weight,
    },
    "logs": {
        "output_dir": args.output_dir,
    },
}

with open(args.path_opts, "r") as f:
    options_yaml = yaml.load(f)
with open(options_yaml["data"]["data_opts_path"], "r") as f:
    data_opts = yaml.load(f)
    options_yaml["data"]["data_opts"] = data_opts

options = update_values(options, options_yaml)
if args.log_suffix:
    options["logs"]["name"] += "-" + args.log_suffix
