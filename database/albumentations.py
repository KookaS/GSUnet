import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

crop_height = 400
crop_width = 400
# randomCrop, verticalFlip, RandomRotate90, Traspose, HorizontalFlip
train_transform = A.Compose(
    [
        # A.SmallestMaxSize(max_size=160),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=crop_height, width=crop_width),
        # A.augmentations.transforms.Blur (blur_limit=7, p=0.5),
        # A.augmentations.transforms.Downscale (scale_min=0.25, scale_max=0.25,p=0.5),
        # A.augmentations.transforms.Sharpen (alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        # A.augmentations.transforms.ToGray(p=0.5),
        # A.augmentations.transforms.VerticalFlip(p=0.5),
        # A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        # A.augmentations.transforms.Transpose(p=0.5),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        # A.augmentations.transforms.HorizontalFlip(p=0.5),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)

test_transform = A.Compose(
    [
        A.RandomCrop(height=crop_height, width=crop_width),
        ToTensorV2()
    ]
)
