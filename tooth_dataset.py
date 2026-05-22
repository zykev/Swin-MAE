import os
import glob
import PIL.Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 自定义 Dataset 映射，用于读取图片列表
class DentalDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = PIL.Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # MAE 训练通常忽略 label，这里返回 0 作为占位符
        return img, 0

def get_intraoral_images(data_path):
    """
    根据口腔数据结构爬取所有训练图片路径。
    支持:
    1. Collector / *_process / {process, sextant, single_tooth} / sample_id / ...
    2. split / *_process / {process, sextant, single_tooth} / sample_id / ...
       例如 .datasets/intraoral/test/s1_process/process/testid1/D.png
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data_path does not exist or is not a directory: {data_path}")

    all_images = set()
    process_folders = []

    for root, dirs, _ in os.walk(data_path):
        for dirname in dirs:
            if dirname.endswith('_process'):
                process_folders.append(os.path.join(root, dirname))

    for base_folder in process_folders:
        # 路径: .../*_process/process/sample_id/{D,F,L,R,U}.png
        process_root = os.path.join(base_folder, 'process')
        if os.path.isdir(process_root):
            all_images.update(glob.glob(os.path.join(process_root, "*", "*.png")))

        # 路径: .../*_process/sextant/sample_id/{F,L,R}/*.png
        sextant_root = os.path.join(base_folder, 'sextant')
        if os.path.isdir(sextant_root):
            all_images.update(glob.glob(os.path.join(sextant_root, "*", "[FLR]", "*.png")))

        # 路径: .../*_process/single_tooth/sample_id/{D,F,L,R,U}/*.png
        tooth_root = os.path.join(base_folder, 'single_tooth')
        if os.path.isdir(tooth_root):
            all_images.update(glob.glob(os.path.join(tooth_root, "*", "[DFLRU]", "*.png")))

    if not all_images:
        # 兜底支持普通图片目录，但排除 mask/标注目录，避免把标签图当作 MAE 训练图。
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for pattern in patterns:
            for path in glob.glob(os.path.join(data_path, "**", pattern), recursive=True):
                parts = {part.lower() for part in os.path.normpath(path).split(os.sep)}
                if any("mask" in part for part in parts):
                    continue
                all_images.add(path)

    return sorted(all_images)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # 获取所有符合结构的图片
    data_path = args.data_path if is_train or not getattr(args, 'val_data_path', '') else args.val_data_path
    print(f"Scanning intraoral images from {data_path}...")
    all_imgs = get_intraoral_images(data_path)

    if not is_train and getattr(args, 'val_data_path', ''):
        img_list = all_imgs
    else:
        val_ratio = getattr(args, 'val_ratio', 0.0)
        if val_ratio > 0:
            split_idx = int(len(all_imgs) * (1.0 - val_ratio))
            split_idx = min(max(split_idx, 1), len(all_imgs))
            if is_train:
                img_list = all_imgs[:split_idx]
            else:
                img_list = all_imgs[split_idx:]
        else:
            img_list = all_imgs

    dataset = DentalDataset(img_list, transform=transform)
    
    split_name = 'train' if is_train else 'val'
    print(f"{split_name} dataset built, images: {len(dataset)}")
    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation='bicubic',
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=mean,
        #     std=std,
        # )
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)])
        

        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size,
                scale=(0.5, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.03,
                )
            ], p=0.5),
            transforms.RandomRotation(
                degrees=10,
                interpolation=transforms.InterpolationMode.BICUBIC,
                fill=0,
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
