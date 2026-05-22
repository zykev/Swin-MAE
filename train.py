import argparse
import datetime
import json
import numpy as np
import os
import time
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
try:
    import yaml
except ModuleNotFoundError:
    yaml = None

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import swin_mae
from tooth_dataset import build_dataset, build_transform
from utils.engine_pretrain import train_one_epoch


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    if value.lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # common parameters
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='yaml config path; command line args override yaml values')
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_freq', default=400, type=int)
    parser.add_argument('--checkpoint_encoder', default='', type=str)
    parser.add_argument('--checkpoint_decoder', default='', type=str)
    parser.add_argument('--data_path', default='.datasets/intraoral', type=str)  # fill in the dataset path here
    parser.add_argument('--val_data_path', default='', type=str,
                        help='optional validation dataset path; if empty, split data_path by val_ratio')
    parser.add_argument('--val_ratio', default=0.1, type=float,
                        help='validation ratio when val_data_path is empty')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # model parameters
    parser.add_argument('--model', default='swin_mae', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--in_chans', default=3, type=int)
    parser.add_argument('--decoder_embed_dim', default=None, type=int)
    parser.add_argument('--embed_dim', default=96, type=int)
    parser.add_argument('--depths', default=(2, 2, 2, 2), type=int, nargs='+')
    parser.add_argument('--num_heads', default=(3, 6, 12, 24), type=int, nargs='+')
    parser.add_argument('--window_size', default=7, type=int)
    parser.add_argument('--qkv_bias', default=True, type=str2bool)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--drop_path_rate', default=0.1, type=float)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--attn_drop_rate', default=0.0, type=float)
    parser.add_argument('--patch_norm', default=True, type=str2bool)
    parser.add_argument('--ape', default=False, type=str2bool,
                        help='accepted from Swin config for compatibility; current SwinMAE does not use APE')
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
    parser.add_argument('--wandb', action='store_true',
                        help='enable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='swin-mae-pretrain', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)
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


def load_config(config_path):
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required to read --config. Install it with `pip install pyyaml`.")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config or {}


def apply_config_defaults(parser, config):
    actions = {action.dest for action in parser._actions}
    key_map = {
        'pretrain_img_size': 'input_size',
    }
    defaults = {}
    for section_value in config.values():
        if not isinstance(section_value, dict):
            continue
        for key, value in section_value.items():
            dest = key_map.get(key.lower(), key.lower())
            if dest in actions:
                defaults[dest] = value
    for key, value in config.items():
        if not isinstance(value, dict):
            dest = key_map.get(key.lower(), key.lower())
            if dest in actions:
                defaults[dest] = value
    if defaults:
        parser.set_defaults(**defaults)


def parse_args():
    parser = get_args_parser()
    config_args, _ = parser.parse_known_args()
    config = load_config(config_args.config)
    apply_config_defaults(parser, config)
    args = parser.parse_args()
    args.config_dict = config
    return args


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


def get_model_kwargs(args):
    common_kwargs = dict(
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        decoder_embed_dim=args.decoder_embed_dim,
        qkv_bias=args.qkv_bias,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        patch_norm=args.patch_norm,
        norm_pix_loss=args.norm_pix_loss,
        mask_ratio=args.mask_ratio,
    )
    if args.model == 'swin_mae':
        common_kwargs.update(dict(
            depths=tuple(args.depths),
            embed_dim=args.embed_dim,
            num_heads=tuple(args.num_heads),
            window_size=args.window_size,
            drop_path_rate=args.drop_path_rate,
        ))
    return common_kwargs


@torch.no_grad()
def save_visualization(model, image_paths, transform, device, epoch, args, log_writer=None, wandb_run=None):
    if len(image_paths) == 0:
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
    save_path = None
    if args.output_dir is not None:
        vis_dir = os.path.join(args.output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, f'epoch_{epoch + 1:04d}.jpg')
        vutils.save_image(grid, save_path)

    if log_writer is not None:
        log_writer.add_image('reconstruction', grid, epoch + 1)
    if wandb_run is not None:
        import wandb
        if save_path is not None:
            image = wandb.Image(save_path)
        else:
            image = wandb.Image(grid.detach().cpu().permute(1, 2, 0).numpy())
        wandb_run.log({'reconstruction': image}, step=epoch + 1)

    if was_training:
        model.train()


@torch.no_grad()
def evaluate_loss(model, data_loader, device, args):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for samples, _ in data_loader:
        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            loss, _, _ = model(samples)
        batch_size = samples.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    if total_count == 0:
        return None
    if misc.is_dist_avail_and_initialized():
        loss_count = torch.tensor([total_loss, total_count], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(loss_count)
        total_loss = loss_count[0].item()
        total_count = loss_count[1].item()
    return total_loss / total_count


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
    dataset_val = build_dataset(is_train=False, args=args)
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
    if args.distributed:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=False,
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
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

    wandb_run = None
    if args.wandb and misc.is_main_process():
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

    # Set model
    model = swin_mae.__dict__[args.model](**get_model_kwargs(args))
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
    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps({
                'type': 'config',
                'args': {k: v for k, v in vars(args).items() if k != 'config_dict'},
                'config': args.config_dict,
            }) + "\n")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        val_loss = evaluate_loss(model, data_loader_val, device, args) if len(dataset_val) > 0 else None
        if misc.is_main_process() and log_writer is not None:
            log_writer.add_scalar('epoch/train_loss', train_stats['loss'], epoch + 1)
            if val_loss is not None:
                log_writer.add_scalar('epoch/val_loss', val_loss, epoch + 1)
        visualize_this_epoch = (
            args.output_dir and
            misc.is_main_process() and
            args.vis_num_images > 0 and
            ((args.vis_freq > 0 and (epoch + 1) % args.vis_freq == 0) or epoch + 1 == args.epochs)
        )
        if visualize_this_epoch:
            save_visualization(
                model_without_ddp, vis_paths, vis_transform,
                device, epoch, args, log_writer=log_writer, wandb_run=wandb_run,
            )
        if args.output_dir and misc.is_main_process() and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch + 1)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }
        if val_loss is not None:
            log_stats['val_loss'] = val_loss

        if args.output_dir and misc.is_main_process():
            if wandb_run is not None:
                wandb_run.log(log_stats, step=epoch + 1)
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    arg = parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
