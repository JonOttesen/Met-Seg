import torch.nn as nn


def _norm(norm: str, output: int, bias: bool, momentum: float = None):
    if norm == "batch_norm":
        if momentum is None:
            return nn.BatchNorm2d(
                num_features=output,
                affine=bias
                )
        return nn.BatchNorm2d(
            num_features=output,
            affine=bias,
            momentum=momentum,
            )
    elif norm == "instance_norm":
        if momentum is None:
            return nn.InstanceNorm2d(
                num_features=output,
                affine=bias
                )
        return nn.InstanceNorm2d(
            num_features=output,
            affine=bias,
            momentum=momentum,
            )
    else:
        return nn.LayerNorm(
            normalized_shape=1,
            elementwise_affine=bias,
            )
