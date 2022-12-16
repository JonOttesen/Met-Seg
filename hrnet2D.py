import torch
import monai

import numpy as np

from m_seg.dataset.containers import DatasetContainer
from m_seg.dataset.loaders import DatasetLoader

from m_seg.models import MultiLoss
from m_seg.models.losses import (
    FocalTverskyLoss,
    WeightedBCEWithLogitsLoss,
    )

from m_seg.metrics import (
    MultiMetric,
    Accuracy,
    DiceCoefficient
    )

from m_seg.trainer import Trainer

from monai.transforms import (
    CropForegroundd,
    RandRotated,
    RandFlipd,
    RandRotate90d,
    RandAdjustContrastd,
    RandHistogramShiftd,
    RandStdShiftIntensityd,
    RandScaleIntensityd,
    RandSpatialCropd,
    SpatialPadd,
    RandZoomd,
)

from m_seg.augmentation import (
    MixUP,
)

from m_seg.models.HRNet.hrnet import HighResolutionNet
from m_seg.models.HRNet.config import hrnet_w32


# Import dataset infp
train = DatasetContainer.from_json("dataset/train.json")
valid = DatasetContainer.from_json("dataset/valid.json")

# Augmentation
transforms = monai.transforms.Compose([
    RandRotated(range_x=0.4, padding_mode='zeros', mode=('bilinear', 'nearest'), keys=['image', 'mask']),

    # RandZoomd(keys=["image", "mask"], prob=0.25, min_zoom=0.85, max_zoom=1.25, mode=('bilinear', 'nearest')),
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=1),
    RandRotate90d(keys=["image", "mask"], prob=0.25, max_k=3, spatial_axes=(0, 1)),

    RandAdjustContrastd(keys=["image"], gamma=(0.5, 2), prob=0.1),
    RandHistogramShiftd(keys=["image"], num_control_points=(5, 15), prob=0.1),

    # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandStdShiftIntensityd(keys=["image"], factors=0.1, prob=0.1, nonzero=True, channel_wise=False),
    RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.1),

    # ImageInputDropout(keys=["image"], prob=0.25, num_sequences=4),  # Built into the dataloader
    CropForegroundd(keys=["image", "mask"], source_key="image", margin=5, return_coords=False, mode='constant', constant_values=0),
    SpatialPadd(keys=["image", "mask"], spatial_size=(176, 176), mode='constant', constant_values=0),
    RandSpatialCropd(keys=["image", "mask"], roi_size=(176, 176), random_center=True, random_size=False),
    ])

valid_transforms = monai.transforms.Compose([
    CropForegroundd(keys=["image", "mask"], source_key="image", margin=2, k_divisible=16, return_coords=False, mode='constant', constant_values=0),
    SpatialPadd(keys=["image", "mask"], spatial_size=(240, 240), mode='constant', constant_values=0),
    RandSpatialCropd(keys=["image", "mask"], roi_size=(240, 240), random_center=True, random_size=False),
    ])

# Order of input sequences
order_dict = {0: ('bravo', False), 1: ('t1', True), 2: ('t1', False), 3: ('flair', False)}

# Make data loader for training
train_loader = DatasetLoader(
    datasetcontainer=train,
    slices=2,
    transforms=transforms,
    sequence_order = order_dict,
    dim=0,
    input_level_dropout=True,
    )

# Make data loader for validation
valid_loader = DatasetLoader(
    datasetcontainer=valid,
    slices=2,
    transforms=valid_transforms,
    sequence_order = order_dict,
    dim=0,
    )

# Tversky don't really improve anything
# loss = [(1, monai.losses.TverskyLoss(alpha=0.7, beta=0.3, sigmoid=True, batch=True)), (1, WeightedBCEWithLogitsLoss(weight=10.))]
loss = [(1, FocalTverskyLoss(gamma=4./3, alpha=0.7, beta=0.3, one_hot_encode=False, batch=True, smooth=1e-5)), (1, WeightedBCEWithLogitsLoss(weight=10.))]

# Neat little wrapper for loss function, quite happy with it
loss = MultiLoss(losses=loss)

# Track metrics
metrics = {
    'CrossEntropy': torch.nn.BCEWithLogitsLoss(),
    'WeightedBCE': WeightedBCEWithLogitsLoss(weight=10.),
    'FocalTversky': FocalTverskyLoss(gamma=1, alpha=0.5, beta=0.5, one_hot_encode=False),
    'Accuracy': Accuracy(),
    'DiceCoefficient - 0.5': DiceCoefficient(ignore_background=False),
    'DiceCoefficient - 0.1': DiceCoefficient(ignore_background=False, treshold=0.1),
    'DiceCoefficient - 0.9': DiceCoefficient(ignore_background=False, treshold=0.9),
    }

# Wrapper for metrics
metrics = MultiMetric(metrics=metrics)

# Init model
model = HighResolutionNet(
    config=hrnet_w32,
    inp_classes=20,
    num_classes=1,
    ratio=1./8,
    norm="batch_norm",
    activation=torch.nn.SiLU(inplace=True),
    momentum=0.1,
    bias=True,
    multi_scale_output=True,
    ocr=False,
    scale_factor=2.,
    )

# Param sanity check
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)


train_weights = list()
seg_weight = 10

for i in range(len(train_loader)):
    if i in train_loader.no_seg:
        train_weights.append(1)
    else:
        train_weights.append(seg_weight)

train_loader = MixUP(train_loader)  # MixUP augmentation

train_sampler = torch.utils.data.WeightedRandomSampler(
    weights=train_weights,
    num_samples=len(train_loader),
    replacement=True,
    generator=None,
    )

# Settings, weights and biases are quite neat
config = {
    "name": "hrnet2d",
    "epochs": 150,
    "iterative": False,
    "images_pr_iteration": 20000,
    "val_images_pr_iteration": 5000,
    "batch_size": 16,
    "learning_rate": 5e-4,
    "optimizer": "AdamW",
    "lr_scheduler": "CosineAnnealingLR",
    "save_dir": "/save/weights/here/HRNet2D",
    "save_period": 20,
    "mixup": True,
    "weight": 5,
    "ratio": None,
    "seg_weight": seg_weight,
}

# Wrap loaders in pytorch dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           num_workers=8,
                                           batch_size=config["batch_size"],
                                           # shuffle=config.shuffle,
                                           sampler=train_sampler,
                                           )

valid_loader = torch.utils.data.DataLoader(dataset=valid_loader,
                                           num_workers=8,
                                           batch_size=config["batch_size"],
                                           shuffle=False,
                                           )

# warmup
class LRPolicy(object):
    def __init__(self, initial, warmup_steps=10):
        self.warmup_steps = warmup_steps
        self.initial = initial

    def __call__(self, step):
        return self.initial + step/self.warmup_steps*(1 - self.initial)

warmup_steps = 10

optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["learning_rate"])

scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer,  LRPolicy(initial=1e-2, warmup_steps=warmup_steps))
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(warmup_steps - config["epochs"]))

lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

# Init trainer
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
    tags=["2D"],
    project="MetastasesSegmentation",
    )

# Train away
trainer.train()
