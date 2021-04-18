import albumentations as A
import albumentations.pytorch
import config

# TRANSFORMS FOR CLASSIFICATION ENGINE

train_transform_cls = A.Compose([
    A.SmallestMaxSize(config.presize),
    A.RandomCrop(config.presize, config.presize),
    A.Rotate(limit=30),
    A.CenterCrop(config.crop, config.crop),
    A.Normalize(),
    A.HorizontalFlip(),
    A.Cutout(),
    albumentations.pytorch.ToTensorV2()])

valid_transform_cls = A.Compose([
    A.SmallestMaxSize(config.presize),
    A.CenterCrop(config.crop, config.crop),
    A.Normalize(),
    albumentations.pytorch.ToTensorV2()])

test_transform_cls = A.Compose([
    A.SmallestMaxSize(config.presize),
    A.CenterCrop(config.crop, config.crop),
    A.Normalize(),
    albumentations.pytorch.ToTensorV2()])

tta_transform_cls = A.Compose([
    A.SmallestMaxSize(config.presize),
    A.Normalize(),
    albumentations.pytorch.ToTensorV2()
])

# TRANSFORMS FOR LOCALIZATION ENGINE

train_transform_loc = A.Compose([
    A.RandomSizedBBoxSafeCrop(config.crop, config.crop),
    A.Rotate(limit=20),
    A.Normalize(),
    A.HorizontalFlip(),
    A.Cutout(),
    albumentations.pytorch.ToTensorV2()],
    bbox_params=A.BboxParams(format='albumentations', min_area=256, min_visibility=0.1))

valid_transform_loc = A.Compose([
    A.RandomSizedBBoxSafeCrop(config.crop, config.crop),
    A.Normalize(),
    albumentations.pytorch.ToTensorV2()],
    bbox_params=A.BboxParams(format='albumentations', min_area=256, min_visibility=0.1))

test_transform_loc = A.Compose([
    A.RandomSizedBBoxSafeCrop(config.crop, config.crop),
    A.Normalize(),
    albumentations.pytorch.ToTensorV2()],
    bbox_params=A.BboxParams(format='albumentations', min_area=256, min_visibility=0.1))
