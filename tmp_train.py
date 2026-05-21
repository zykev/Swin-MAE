# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import PIL.Image

import timm

# assert timm.__version__ == "0.3.2"  # version check
# import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.tooth_datasets import build_dataset
from util.lr_decay import add_weight_decay

import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--vis_freq', default=100, type=int,
                        help='log reconstruction visualizations every N epochs, 0 for final epoch only')
    parser.add_argument('--vis_num_images', default=8, type=int,
                        help='number of fixed images used for reconstruction visualization')
    parser.add_argument('--wandb', action='store_true',
                        help='enable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='mae-pretrain', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default='intraoral', type=str,
                        help='Weights & Biases entity name')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Weights & Biases run name')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='load pretrained model weights from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int, dest='local_rank')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def build_visualization_transform(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)

    return transforms.Compose([
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def denormalize_batch(imgs):
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    return (imgs * std + mean).clamp(0, 1)


def select_visualization_paths(image_paths, num_images, seed):
    if len(image_paths) == 0:
        return []

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(image_paths),
        size=min(num_images, len(image_paths)),
        replace=False,
    )
    return [image_paths[int(i)] for i in indices]


def select_categorized_visualization_paths(dataset, num_images, seed):
    if not hasattr(dataset, 'image_paths') or len(dataset.image_paths) == 0:
        return {}

    categories = {
        'process': [],
        'sextant': [],
        'single_tooth': [],
    }
    for path in dataset.image_paths:
        parts = set(Path(path).parts)
        for category in categories:
            if category in parts:
                categories[category].append(path)
                break

    return {
        category: select_visualization_paths(paths, num_images, seed + i)
        for i, (category, paths) in enumerate(categories.items())
        if len(paths) > 0
    }


@torch.no_grad()
def log_reconstruction_visualizations(
        model, image_paths, transform, device, epoch, args, log_writer, tag,
        save_path=None, wandb_run=None):
    if log_writer is None and wandb_run is None:
        return
    if len(image_paths) == 0:
        return

    was_training = model.training
    model.eval()

    imgs = []
    for path in image_paths:
        img = PIL.Image.open(path).convert('RGB')
        imgs.append(transform(img))
    imgs = torch.stack(imgs, dim=0).to(device, non_blocking=True)

    _, pred, mask = model(imgs, mask_ratio=args.mask_ratio)

    if args.norm_pix_loss:
        target = model.patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred = pred * (var + 1.e-6).sqrt() + mean

    pred = model.unpatchify(pred)
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
    mask = model.unpatchify(mask)

    im_masked = imgs * (1 - mask)
    im_paste = imgs * (1 - mask) + pred * mask

    panels = []
    for i in range(imgs.shape[0]):
        panels.extend([
            denormalize_batch(imgs[i:i + 1]).squeeze(0),
            denormalize_batch(im_masked[i:i + 1]).squeeze(0),
            denormalize_batch(pred[i:i + 1]).squeeze(0),
            denormalize_batch(im_paste[i:i + 1]).squeeze(0),
        ])

    grid = vutils.make_grid(panels, nrow=4, padding=4)
    if log_writer is not None:
        log_writer.add_image(tag, grid, epoch)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vutils.save_image(grid, save_path)
    if wandb_run is not None:
        import wandb
        if save_path is not None:
            wandb_image = wandb.Image(save_path)
        else:
            wandb_image = wandb.Image(grid.detach().cpu().permute(1, 2, 0).numpy())
        wandb_run.log({tag: wandb_image}, step=(epoch + 1) * 1000)

    if was_training:
        model.train()


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.distributed:
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # # simple augmentation
    # transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # print(dataset_train)
    dataset_train = build_dataset(is_train=True, args=args)
    vis_paths_by_category = {}
    vis_transform = None
    if misc.is_main_process() and args.vis_num_images > 0:
        vis_paths_by_category = select_categorized_visualization_paths(
            dataset_train, args.vis_num_images, args.seed
        )
        vis_transform = build_visualization_transform(args)
        print("Visualization samples:")
        for category, paths in vis_paths_by_category.items():
            print(f"  {category}: {len(paths)} images")

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    wandb_run = None
    if misc.is_main_process() and args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb logging requested, but wandb is not installed.") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir if args.output_dir else None,
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.output_dir and misc.is_main_process():
        log_config = {
            'type': 'config',
            'job_dir': os.path.dirname(os.path.realpath(__file__)),
            'args': vars(args),
            'effective_batch_size': eff_batch_size,
        }
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_config) + "\n")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, load_optimizer=False)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            wandb_run=wandb_run,
            args=args
        )
        visualize_this_epoch = (
            (log_writer is not None or wandb_run is not None) and
            args.vis_num_images > 0 and
            ((args.vis_freq > 0 and epoch % args.vis_freq == 0) or epoch + 1 == args.epochs)
        )
        if visualize_this_epoch:
            for category, vis_paths in vis_paths_by_category.items():
                log_reconstruction_visualizations(
                    model_without_ddp, vis_paths, vis_transform,
                    device, epoch, args, log_writer,
                    tag=f'reconstruction/{category}',
                    save_path=os.path.join(
                        args.output_dir,
                        'reconstruction',
                        category,
                        f'epoch_{epoch:04d}.jpg',
                    ) if args.output_dir else None,
                    wandb_run=wandb_run,
                )
        save_this_epoch = (
            args.output_dir and
            ((args.vis_freq > 0 and epoch % args.vis_freq == 0) or epoch + 1 == args.epochs)
        )
        if save_this_epoch:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if misc.is_main_process():
            if wandb_run is not None:
                wandb_run.log(log_stats, step=(epoch + 1) * 1000)
            if args.output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
