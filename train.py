import argparse
import json
import numpy as np
import os
from pathlib import Path

import PIL.Image
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import swin_mae
from tooth_dataset import build_dataset, build_transform
from utils.engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_freq', default=400, type=int)
    parser.add_argument('--checkpoint_encoder', default='', type=str)
    parser.add_argument('--checkpoint_decoder', default='', type=str)
    parser.add_argument('--data_path', default='.datasets/intraoral', type=str)  # fill in the dataset path here
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # model parameters
    parser.add_argument('--model', default='swin_mae', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # optimizer parameters
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # other parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--vis_freq', default=100, type=int,
                        help='save reconstruction visualizations every N epochs, 0 for final epoch only')
    parser.add_argument('--vis_num_images', default=8, type=int,
                        help='number of fixed images used for reconstruction visualization')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
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
    parser.add_argument('--dist_backend', default=None, type=str,
                        help='distributed backend, defaults to nccl')

    return parser


def denormalize_batch(imgs):
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    return (imgs * std + mean).clamp(0, 1)


def select_visualization_paths(dataset, num_images, seed):
    if num_images <= 0 or not hasattr(dataset, 'image_paths') or len(dataset.image_paths) == 0:
        return []

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(dataset.image_paths),
        size=min(num_images, len(dataset.image_paths)),
        replace=False,
    )
    return [dataset.image_paths[int(i)] for i in indices]


@torch.no_grad()
def save_visualization(model, image_paths, transform, device, epoch, args, log_writer=None):
    if len(image_paths) == 0 or args.output_dir is None:
        return

    was_training = model.training
    model.eval()

    imgs = []
    for path in image_paths:
        img = PIL.Image.open(path).convert('RGB')
        imgs.append(transform(img))
    imgs = torch.stack(imgs, dim=0).to(device, non_blocking=True)

    _, pred, mask = model(imgs)

    if args.norm_pix_loss:
        target = model.patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred = pred * (var + 1.e-6).sqrt() + mean

    pred = model.unpatchify(pred)
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size ** 2 * 3)
    mask = model.unpatchify(mask)

    im_masked = imgs * (1 - mask)
    im_reconstruction = pred * mask
    im_paste = imgs * (1 - mask) + pred * mask

    panels = []
    for i in range(imgs.shape[0]):
        panels.extend([
            denormalize_batch(imgs[i:i + 1]).squeeze(0),
            denormalize_batch(im_masked[i:i + 1]).squeeze(0),
            denormalize_batch(im_reconstruction[i:i + 1]).squeeze(0),
            denormalize_batch(im_paste[i:i + 1]).squeeze(0),
        ])

    grid = vutils.make_grid(panels, nrow=4, padding=4)
    vis_dir = os.path.join(args.output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f'epoch_{epoch + 1:04d}.jpg')
    vutils.save_image(grid, save_path)

    if log_writer is not None:
        log_writer.add_image('reconstruction', grid, epoch + 1)

    if was_training:
        model.train()


def main(args):
    misc.init_distributed_mode(args)

    # Fixed random seeds
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up training equipment
    if args.distributed:
        device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    elif args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        args.device = 'cpu'
        device = torch.device(args.device)
    else:
        device = torch.device(args.device)
    cudnn.benchmark = True

    # Set dataset
    dataset_train = build_dataset(is_train=True, args=args)
    if len(dataset_train) == 0:
        raise ValueError(f"No training images found under data_path: {args.data_path}")
    vis_paths = select_visualization_paths(dataset_train, args.vis_num_images, args.seed)
    vis_transform = build_transform(is_train=False, args=args)
    if misc.is_main_process() and len(vis_paths) > 0:
        print(f"Visualization samples: {len(vis_paths)} images")

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    # Log output
    if args.log_dir is not None and misc.is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        if SummaryWriter is None:
            print("TensorBoard is not installed; training will continue without tensorboard logging.")
            log_writer = None
        else:
            log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Set model
    model = swin_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, mask_ratio=args.mask_ratio)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        ddp_kwargs = {}
        if device.type == 'cuda':
            ddp_kwargs['device_ids'] = [args.gpu]
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        model_without_ddp = model.module

    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=5e-2, betas=(0.9, 0.95))  # 原来是5E-2
    loss_scaler = NativeScaler()

    # Create model
    misc.load_model(args=args, model_without_ddp=model_without_ddp)

    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        visualize_this_epoch = (
            args.output_dir and
            misc.is_main_process() and
            args.vis_num_images > 0 and
            ((args.vis_freq > 0 and (epoch + 1) % args.vis_freq == 0) or epoch + 1 == args.epochs)
        )
        if visualize_this_epoch:
            save_visualization(
                model_without_ddp, vis_paths, vis_transform,
                device, epoch, args, log_writer=log_writer,
            )
        if args.output_dir and misc.is_main_process() and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch + 1)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    arg = get_args_parser()
    arg = arg.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
