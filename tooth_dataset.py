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
    根据特定结构爬取所有图片路径
    结构: Collector / *_process / {process, sextant, single_tooth} / sample_id / ...
    """
    all_images = []
    
    # 1. 遍历收集者文件夹 (e.g., peiliu)
    for collector in os.listdir(data_path):
        collector_path = os.path.join(data_path, collector)
        if not os.path.isdir(collector_path):
            continue
            
        # 2. 遍历以 _process 结尾的日期文件夹
        for date_folder in os.listdir(collector_path):
            if not date_folder.endswith('_process'):
                continue
            
            base_folder = os.path.join(collector_path, date_folder)
            
            # --- 分支 A: process 文件夹 ---
            # 路径: .../process/sample_id/{D,F,L,R,U}.png
            process_root = os.path.join(base_folder, 'process')
            if os.path.exists(process_root):
                # 匹配所有 sample_id 下的 png
                all_images.extend(glob.glob(os.path.join(process_root, "*", "*.png")))

            # --- 分支 B: sextant 文件夹 ---
            # 路径: .../sextant/sample_id/{F,L,R}/*.png
            sextant_root = os.path.join(base_folder, 'sextant')
            if os.path.exists(sextant_root):
                # 匹配 sample_id 下 F, L, R 文件夹内的所有 png
                all_images.extend(glob.glob(os.path.join(sextant_root, "*", "[FLR]", "*.png")))

            # --- 分支 C: single_tooth 文件夹 ---
            # 路径: .../single_tooth/sample_id/{D,F,L,R,U}/*.png
            tooth_root = os.path.join(base_folder, 'single_tooth')
            if os.path.exists(tooth_root):
                # 匹配 sample_id 下 D, F, L, R, U 文件夹内的所有 png
                all_images.extend(glob.glob(os.path.join(tooth_root, "*", "[DFLRU]", "*.png")))

    return sorted(all_images)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # 获取所有符合结构的图片
    print(f"正在从 {args.data_path} 检索口腔影像数据...")
    all_imgs = get_intraoral_images(args.data_path)
    
    # # 简单的训练/验证集划分 (例如 90% 训练, 10% 验证)
    # # 为了保证实验可重复，这里进行了排序并固定划分
    # split_idx = int(len(all_imgs) * 0.9)
    # if is_train:
    #     img_list = all_imgs[:split_idx]
    # else:
    #     img_list = all_imgs[split_idx:]

    dataset = DentalDataset(all_imgs, transform=transform)
    
    print(f"{'训练' if is_train else '验证'}集构建完成，包含图片数量: {len(dataset)}")
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
