import argparse
import os
from pathlib import Path

import torch

import swin_mae
from tooth_dataset import DentalDataset, build_transform, get_intraoral_images
from train import (
    apply_config_defaults,
    get_args_parser as get_train_args_parser,
    get_model_kwargs,
    load_config,
    save_visualization,
    select_categorized_visualization_paths,
)


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Visualize Swin-MAE reconstructions from a pretraining checkpoint.'
    )
    parser.add_argument(
        '--checkpoint',
        default='exp/swinmae_v1/checkpoint-400.pth',
        type=str,
        help='pretraining checkpoint path',
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        type=str,
        help='training YAML used to construct the model',
    )
    parser.add_argument(
        '--data_path',
        default=None,
        type=str,
        help='intraoral dataset path; defaults to DATA.DATA_PATH in the YAML',
    )
    parser.add_argument(
        '--num_images',
        default=None,
        type=int,
        help='images sampled per category; defaults to LOG.VIS_NUM_IMAGES in the YAML',
    )
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    return parser


def load_training_args(config_path):
    parser = get_train_args_parser()
    config = load_config(config_path)
    apply_config_defaults(parser, config)
    args = parser.parse_args([])
    args.config = config_path
    args.config_dict = config
    return args


def load_model(checkpoint_path, train_args, device):
    model = swin_mae.__dict__[train_args.model](**get_model_kwargs(train_args))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    result = model.load_state_dict(state_dict, strict=True)
    print(f'Loaded checkpoint: {checkpoint_path}')
    print(result)
    model.to(device)
    model.eval()
    return model, int(checkpoint.get('epoch', 0))


def main(args):
    train_args = load_training_args(args.config)
    train_args.data_path = args.data_path or train_args.data_path
    train_args.output_dir = os.path.join(os.path.dirname(args.checkpoint), 'inference')
    train_args.vis_num_images = (
        args.num_images if args.num_images is not None else train_args.vis_num_images
    )
    train_args.seed = args.seed if args.seed is not None else train_args.seed

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available. Use --device cpu if needed.')
    device = torch.device(args.device)

    image_paths = get_intraoral_images(train_args.data_path)
    dataset = DentalDataset(image_paths)
    paths_by_category = select_categorized_visualization_paths(
        dataset,
        train_args.vis_num_images,
        train_args.seed,
    )
    if not paths_by_category:
        raise ValueError(f'No visualization images found under {train_args.data_path}')

    print('Visualization samples:')
    for category in ('process', 'sextant', 'single_tooth'):
        print(f'  {category}: {len(paths_by_category.get(category, []))} images')

    model, checkpoint_epoch = load_model(args.checkpoint, train_args, device)
    transform = build_transform(is_train=False, args=train_args)
    Path(train_args.output_dir).mkdir(parents=True, exist_ok=True)

    # save_visualization adds one before formatting the epoch filename.
    visualization_epoch = max(checkpoint_epoch - 1, 0)
    torch.manual_seed(train_args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(train_args.seed)

    for category, category_paths in paths_by_category.items():
        save_visualization(
            model=model,
            image_paths=category_paths,
            transform=transform,
            device=device,
            epoch=visualization_epoch,
            args=train_args,
            category=category,
        )

    print(f'Visualizations saved to: {os.path.abspath(train_args.output_dir)}')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
