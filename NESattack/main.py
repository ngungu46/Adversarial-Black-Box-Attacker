import os
import json
import shutil
import argparse
from tensorflow.python.client import device_lib

from NESAttack import *
from POAttack import *
from utils import *

SIGMA = 0.001
EPS_DECAY = 0.001
EPS_0 = 0.5
N = 100
K = 1
E_ADV = 0.05
MAX_QUERIES = 20000
MAX_LR = 0.01
MIN_LR = 0.001
LR = 0.01
ATTACK_TYPE = 'PO'
ORIG_PATH = './imagenet_val/n01751748/ILSVRC2012_val_00000001.JPEG'
ADV_PATH = './imagenet_val/n02105855/ILSVRC2012_val_00000003.JPEG'

def main():

    print("hey")
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_image-dir', type=str, default=ORIG_PATH)
    parser.add_argument('--adv-image-dir', type=str, default=ADV_PATH)
    parser.add_argument('--adv-cls', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=N)
    parser.add_argument('--sigma', type=float, default=SIGMA)
    parser.add_argument('--start-epsilon', type=float, default=EPS_0)
    parser.add_argument('--target-epsilon', type=float, default=E_ADV)
    parser.add_argument('--num-classes', type=int, default=K)
    parser.add_argument('--max-queries', type=int, default=MAX_QUERIES)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--max-lr', type=float, default=MAX_LR)
    parser.add_argument('--min-lr', type=float, default=MIN_LR)
    parser.add_argument('--epsilon-decay', type=float, default=EPS_DECAY)
    parser.add_argument('--attack-type', type=str, default=ATTACK_TYPE)
    args = parser.parse_args()

    img_orig, _ = get_image(args.orig_image_dir)
    img_adv, _ = get_image(args.adv_image_dir)
    y_adv = args.adv_cls
    if args.attack_type == 'PO':
        attacker = PartialInfoAttack(
            args.target_epsilon,
            args.start_epsilon,
            args.sigma,
            args.batch_size,
            args.epsilon_decay,
            args.max_lr,
            args.min_lr,
            args.num_classes,
            args.max_queries
        )

        x_adv, y_res, out, count = attacker.attack(img_adv, y_adv, img_orig)
        if out:
            print(f"Successfully attack with {count} queries")
        else:
            print(f"Unsuccessfully attack under {args.max_queries} queries")

    else:
        attacker = NESAttack(
            args.target_epsilon,
            args.lr,
            args.batch_size,
            args.sigma,
            args.max_queries
        )

        res, _, prob, count, out = attacker.attack(img_orig, y_adv)
        if out:
            print(f"Successfully attack with {count} queries, prob class {prob}")
        else:
            print(f"Unsuccessfully attack under {args.max_queries} queries")

    
    display_images(res.reshape(1, 299, 299, 3))

if __name__ == "__main__":
    print("helo")
    main()