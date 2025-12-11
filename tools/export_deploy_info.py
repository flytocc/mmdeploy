import argparse
import os
import os.path as osp

import mmengine
from mmdeploy.backend.sdk.export_info import export2SDK
from mmdeploy.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to backends.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument(
        '--work-dir',
        default=os.getcwd(),
        help='the dir to save logs and models')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # create work_dir if not
    mmengine.mkdir_or_exist(osp.abspath(args.work_dir))

    export2SDK(
        deploy_cfg,
        model_cfg,
        args.work_dir,
        pth=checkpoint_path,
        device=args.device)


if __name__ == '__main__':
    main()
