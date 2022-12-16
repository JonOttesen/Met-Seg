import torch
import monai
import torch.nn as nn

import numpy as np

from t_seg.dataset.containers import DatasetContainer
from t_seg.dataset.loaders import VolumeLoader

from t_seg.models import MultiLoss
from t_seg.models.losses import (
    FocalTverskyLoss,
    WeightedBCEWithLogitsLoss,
    )

from t_seg.metrics import (
    MultiMetric,
    Accuracy,
    DiceCoefficient
    )

from t_seg.trainer import Trainer

from t_seg.preprocessing import (
    VolumeInputDropout,
    )

from t_seg.models.HRNet3D.hrnet import HighResolutionNet
from t_seg.models.HRNet3D.config import hrnet_w18, hrnet_w32, hrnet_w48

from monai.transforms import (
    CropForegroundd,
    RandRotated,
    RandFlipd,
    RandRotate90d,
    RandAdjustContrastd,
    RandHistogramShiftd,
    NormalizeIntensityd,
    RandStdShiftIntensityd,
    RandScaleIntensityd,
    RandSpatialCropd,
    SpatialPadd,
    RandZoomd,
)


# Not good practice, but who cares
import warnings
warnings.filterwarnings("ignore")


# """
train = DatasetContainer.from_json('./dataset/train_nii.json')
train.add_shapes()
valid = DatasetContainer.from_json('./dataset/valid_nii.json')
valid.add_shapes()
# From non-prconverted datasets
# """
voxel_sizes = (0.9375, 0.9375, 1)
# OG size (128, 128, 128)
# UNet recommended size (112, 160, 128)
# My calculated size (122.42152471, 148.28522711, 115.5245374)

size = (128, 128, 128)

transforms = monai.transforms.Compose([
    CropForegroundd(keys=["image", "mask"], source_key="image", margin=3, return_coords=False, mode='constant', constant_values=0),

    RandRotated(range_x=0.4, range_y=0.4, range_z=0.4, padding_mode='zeros', mode=('bilinear', 'nearest'), keys=['image', 'mask']),
    RandZoomd(keys=["image", "mask"], prob=0.25, min_zoom=0.85, max_zoom=1.25, mode=('trilinear', 'nearest')),

    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=1),
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=2),
    RandRotate90d(keys=["image", "mask"], prob=0.25, max_k=3, spatial_axes=(0, 1)),

    RandAdjustContrastd(keys=["image"], gamma=(0.5, 2), prob=0.1),
    RandHistogramShiftd(keys=["image"], num_control_points=(5, 15), prob=0.1),

    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandStdShiftIntensityd(keys=["image"], factors=0.1, prob=0.1, nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.1),

    VolumeInputDropout(keys=["image"], prob=0.25, num_sequences=4),

    SpatialPadd(keys=["image", "mask"], spatial_size=size, mode='constant', constant_values=0),
    RandSpatialCropd(keys=["image", "mask"], roi_size=size, random_center=True, random_size=False),
    ])

valid_transforms = monai.transforms.Compose([
    CropForegroundd(keys=["image", "mask"], source_key="image", margin=2, k_divisible=16, return_coords=False, mode='constant', constant_values=0),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

    # SpatialPadd(keys=["image", "mask"], spatial_size=size, mode='constant', constant_values=0),
    # RandSpatialCropd(keys=["image", "mask"], roi_size=size, random_center=True, random_size=False),
    ])


order_dict = {0: ('bravo', False), 1: ('t1', True), 2: ('t1', False), 3: ('flair', False)}

train_loader = VolumeLoader(
    datasetcontainer=train,
    transforms=transforms,
    sequence_order = order_dict,
    voxel_sizes=voxel_sizes
    )


valid_loader = VolumeLoader(
    datasetcontainer=valid,
    transforms=valid_transforms,
    sequence_order = order_dict,
    voxel_sizes=voxel_sizes,
    )

loss = [(1, monai.losses.TverskyLoss(alpha=0.7, beta=0.3, sigmoid=True, batch=False)), (1, WeightedBCEWithLogitsLoss(weight=10.))]
# loss = [(1, FocalTverskyLoss(gamma=4./3, alpha=0.7, beta=0.3, one_hot_encode=False, batch=False, smooth=1e-5)), (1, WeightedBCEWithLogitsLoss(weight=10.))]


loss = MultiLoss(losses=loss)

# loss = torch.nn.BCEWithLogitsLoss()
# loss = monai.losses.TverskyLoss(include_background=True, to_onehot_y=False, sigmoid=True, alpha=0.3, beta=0.7)


config = {
    "name": "hrnet3d_tversky",
    "epochs": 800,
    "iterative": False,
    "images_pr_iteration": 1,
    "val_images_pr_iteration": 1,
    "batch_size": 2,
    "learning_rate": 5e-3,
    "optimizer": "AdamW",
    "lr_scheduler": "CosineAnnealingLR",
    "save_dir": "/itf-fi-ml/home/jonakri/Segmentation/HRNet3D_diff_size_focal",
    "save_period": 100,
    "mixup": False,
    "size": size,
    "smooth": 1e-5,
}


metrics = {
    'CrossEntropy': torch.nn.BCEWithLogitsLoss(),
    'WeightedBCE': WeightedBCEWithLogitsLoss(weight=10.),
    'FocalTversky': FocalTverskyLoss(gamma=1, alpha=0.5, beta=0.5, one_hot_encode=False),
    'Accuracy': Accuracy(),
    'DiceCoefficient': DiceCoefficient(ignore_background=False),
    }

metrics = MultiMetric(metrics=metrics)

# """
model = HighResolutionNet(
    config=hrnet_w48,
    inp_classes=4,
    num_classes=1,
    ratio=None,
    activation=nn.SiLU(inplace=True),
    bias=True,
    multi_scale_output=True,
    deep_supervision=True,
    )

# pretrain_path = "/itf-fi-ml/home/jonakri/ELITE/HRNet3D_pretrain/BraTS/2022-07-18/best_validation/checkpoint-best.pth"
# model.load_state_dict(torch.load(pretrain_path, map_location='cpu')['state_dict'])

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)



# train_loader = MixUP(train_loader)  # MixUP augmentation


train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           num_workers=8,
                                           batch_size=config["batch_size"],
                                           shuffle=True,
                                           )

valid_loader = torch.utils.data.DataLoader(dataset=valid_loader,
                                           num_workers=2,
                                           batch_size=1,
                                           shuffle=False,
                                           )


class LRPolicy(object):
    def __init__(self, initial, warmup_steps=10):
        self.warmup_steps = warmup_steps
        self.initial = initial

    def __call__(self, step):
        return self.initial + step/self.warmup_steps*(1 - self.initial)

warmup_steps = 50

optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["learning_rate"])

scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer,  LRPolicy(initial=1e-2, warmup_steps=warmup_steps))
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(warmup_steps - config["epochs"]))

lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    optimizer=optimizer,
    config=config,
    data_loader=train_loader,
    valid_data_loader=valid_loader,
    lr_scheduler=lr_scheduler,
    seed=None,
    # log_step=50,
    device="cuda:0",
    mixed_precision=True,
    tags=["3D"],
    project="MetSeg",
    )

# trainer.resume_checkpoint(
    # resume_model='/itf-fi-ml/home/jonakri/ELITE/HRNet3D/2022-08-25/best_validation/checkpoint-best.pth',
    # resume_metric='/itf-fi-ml/home/jonakri/ELITE/HRNet3D/2022-08-25/best_validation/statistics.json',
    # )


trainer.train()