from typing import Callable, Dict, Optional, Union, Tuple, List
from collections import defaultdict
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# from torchvision.utils import make_grid
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker

from ..base import BaseTrainer
from ..models import MultiLoss
from ..metrics import MultiMetric


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Union[MultiLoss, Callable],
                 metric_ftns: Union[MultiMetric, Dict[str, Callable]],
                 optimizer: torch.optim,
                 lr_scheduler: torch.optim.lr_scheduler,
                 config: dict,
                 entity: str,
                 project: str,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader = None,
                 seed: int = None,
                 device: str = None,
                 tags: Optional[List[str]] = None,
                 log_step: int = None,
                 mixed_precision: bool = False,
                 ):
        """
        Args:
            model (torch.nn.Module): The model to be trained
            loss_function (MultiLoss): The loss function or loss function class
            metric_ftns (MultiMetric, Dict[str, callable]): Dict or Multimetric for the metrics to be evaluated during validation
            optimizer (torch.optim): torch.optim, i.e., the optimizer class
            lr_scheduler (torch.optim.lr_scheduler): lr schedualer
            config (dict): dict of configs
            entity (str): where the wands and biases run should be stored
            project (str): name of project in weights and biases
            data_loader (torch.utils.data.dataloader): dataloader used for training
            valid_data_loader (torch.utils.data.dataloader): dataloader used for validation
            seed (int): integer seed to enforce non stochasticity,
            device (str): string of the device to be trained on, e.g., "cuda:0"
            tags (List[str]): tags in weights and biases
            log_step (int): after how many steps should you log
            mixed_precision (bool): to use or not use mixed precision
        """

        super().__init__(model=model,
                         loss_function=loss_function,
                         metric_ftns=metric_ftns,
                         optimizer=optimizer,
                         config=config,
                         lr_scheduler=lr_scheduler,
                         seed=seed,
                         device=device,
                         project=project,
                         tags=tags,
                         )

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.mixed_precision = mixed_precision

        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.images_pr_iteration = int(config['images_pr_iteration'])
        self.val_images_pr_iteration = int(config['val_images_pr_iteration'])

        self.batch_size = data_loader.batch_size
        self.len_epoch = len(data_loader) if not self.iterative else self.images_pr_iteration/self.batch_size
        self.log_step = int(self.len_epoch/4) if not isinstance(log_step, int) else int(log_step/self.batch_size)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        losses = defaultdict(list)

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    out = self.model(data)
                    loss = self._loss(out, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(data)
                loss = self._loss(out, target)

                loss.backward()
                self.optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses['loss'].append(loss)

            if batch_idx % self.log_step == 0:
                self.logger.info('Train {}: {} {} Loss: {:.6f}'.format(
                    'Epoch' if not self.iterative else 'Iteration',
                    epoch,
                    self._progress(batch_idx),
                    loss))

            if batch_idx*self.batch_size >= self.images_pr_iteration and self.iterative:
                break

        self.optimizer.zero_grad()
        losses['loss_func'] = str(self.loss_function)

        return {"loss": np.mean(losses["loss"])}

    def _loss(self, out: Union[torch.Tensor, Tuple[torch.Tensor]], target: torch.Tensor):
        
        if isinstance(out, (list, tuple)):
            output, auxiliary = out
            
            loss = self.loss_function(output, target)
            auxiliary = auxiliary if isinstance(auxiliary, list) else [auxiliary]
            for aux in auxiliary:
                loss += 0.33*self.loss_function(aux, target)

            return loss
        output = out

        return self.loss_function(output, target)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.valid_data_loader is None:
            return None

        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                out = self.model(data)


                if isinstance(out, (list, tuple)):
                    output, _ = out
                else:
                    output = out

                loss = self._loss(output, target)
                metrics['val_loss'].append(loss.item())

                for key, metric in self.metric_ftns.items():
                    if self.metrics_is_dict:
                        metrics[key].append(metric(output.cpu(), target.cpu()).item())
                    else:
                        metrics[key].append(metric(output, target).item())

                if batch_idx*self.batch_size >= self.val_images_pr_iteration and self.iterative:
                    break

        metric_dict = dict()
        for key, item in metrics.items():
            metric_dict[key] = np.mean(metrics[key])

        return metric_dict

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        elif hasattr(self.data_loader, 'batch_size'):
            current = batch_idx*self.data_loader.batch_size
            total = self.len_epoch*self.data_loader.batch_size
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
