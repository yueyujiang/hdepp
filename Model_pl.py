import torch
import torch.nn as nn
import submodule
import data
import utils
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable, Union
from torch.optim.optimizer import Optimizer
import math
import matplotlib.pyplot as plt
import io
import PIL.Image
import geoopt.manifolds.poincare.math as pmath
import geoopt
import os
import json


class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        channel = 4

        self.conv = nn.Conv1d(channel, args.h_channel, 1)
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        resblocks = []
        for i in range(args.resblock_num):
            resblocks.append(submodule.resblock(args.h_channel,
                                                args.h_channel,
                                                5, 0.3))
        self.resblocks = nn.Sequential(*resblocks)

        self.args = args
        self.train_loss = 0
        # if self.args.distance_mode == 'hyperbolic':
        #     # self.hyperbolic_layer = nn.Sequential(
        #     #     nn.Flatten(1),
        #     #     nn.Linear(args.sequence_length * args.h_channel, 64),
        #     #     utils.MobiusLinear(64, args.embedding_size)
        #     # )
        #     self.pre_layer = nn.Sequential(
        #         nn.Flatten(1),
        #         nn.Linear(args.sequence_length * args.h_channel, args.embedding_size)
        #     )
        #     # self.hyperbolic_layer = nn.nn.Sequential(
        #     #     utils.MobiusLinear(args.embedding_size, args.embedding_size, nonlin=torch.relu),
        #     #     utils.MobiusLinear(args.embedding_size, args.embedding_size, nonlin=torch.relu)
        #     # )
        #
        #     # self.register_parameter(name='c', param=nn.Parameter(torch.tensor([1.0]), requires_grad=True))
        #     self.hyperbolic_layer = utils.MobiusLinear(args.embedding_size, args.embedding_size,
        #                                                hyperbolic_bias=True, nonlin=None, c=args.c)
        #
        # else:
        self.linear = nn.Conv1d(args.h_channel,
                                args.embedding_size,
                                args.sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, channel, seq_length = x.shape
        x = self.celu(self.conv(x))
        x = self.resblocks(x)
        # if self.args.distance_mode == 'hyperbolic':
        #     x = self.pre_layer(x)
        #     x = pmath.expmap0(x)
        #     x = self.hyperbolic_layer(x)
        # else:
        x = self.linear(x).squeeze(-1)
        x = x.view(bs, self.args.embedding_size)
        return x


class model(LightningModule):
    def __init__(self, args):
        super(model, self).__init__()
        self.save_hyperparameters(args)
        torch.set_default_dtype(torch.float64)
        if not self.hparams.sequence_length:
            utils.get_seq_length(self.hparams)
        self.save_hyperparameters(self.hparams)
        self.channel = 4
        if self.hparams.distance_mode == 'hyperbolic':
            self.hparams.distance_ratio = 1
            # self.k = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
            self.k = - 1.0
        else:
            self.hparams.distance_ratio = math.sqrt(
                float(1.0 / float(self.hparams.embedding_size) / 10 * float(self.hparams.distance_alpha)))

        self.dis_loss_w = 100
        self.train_loss = []
        self.val_loss = float('inf')
        self.norm_loss = 0.0
        if self.hparams.distance_mode == 'hyperbolic':
            c = nn.Parameter(torch.tensor([float(args.c)]), requires_grad=False)
        else:
            c = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.register_parameter('c', c)
        if self.hparams.distance_mode == 'hyperbolic':
            self.hparams.weighted_method == 'fm'
        self.hparams.c = 1.0
        self.encoder = encoder(self.hparams)
        if self.hparams.distance_mode == 'hyperbolic':
            # self.encoder.pre_layer[1].weight = nn.Parameter(self.encoder.pre_layer[1].weight * 1e-5)
            # self.encoder.pre_layer[1].bias = nn.Parameter(self.encoder.pre_layer[1].bias * 1e-5)
            self.encoder.linear.weight = nn.Parameter(self.encoder.linear.weight * 1e-6)
            self.encoder.linear.bias = nn.Parameter(self.encoder.linear.bias * 0)
        self.running_loss = 0
        self.current_lr = self.hparams.lr
        self.current_clr = self.hparams.clr
        self.one_num = []
        # self.hparams.switch_epoch = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(x)
        # if self.hparams.distance_mode == 'hyperbolic':
        #     encoding = encoding * 1e-5
        return encoding

    def training_step(self, batch, batch_idx):

        nodes = batch['nodes']
        seq = batch['seqs']
        device = seq.device

        if self.hparams.distance_mode == 'hyperbolic':

            # if self.current_epoch > 500:
            #     self.hparams.weighted_method = 'fm'
            encoding = self(seq)
            n = len(encoding)
            # encoding = encoding / (encoding.norm(dim=-1, keepdim=True).detach() / math.sqrt(1 - 0.5e-4))
            #distance = pmath.dist(encoding, encoding.detach())
            #distance = pmath.dist(encoding.view(n, 1, -1), encoding.view(1, n, -1).detach())
            if self.hparams.fixed_point is not None:
                nodes += list(self.fixed_point['names'])
                encoding = torch.cat([encoding, self.fixed_point['emb'].to(device)], dim=0)
                n = len(nodes)
            gt_distance = self.train_data.true_distance(nodes, nodes).to(device)
            if (self.current_epoch + 1) % 10 == 0 and \
                self.current_epoch < self.hparams.c_epoch and \
                    self.c[0] > 0.4 and self.current_epoch > 500:
                lr = self.hparams.clr * 0.95 ** (self.current_epoch//50)
                self.current_clr = lr
            else:
                lr = 0
            # lr = 0
            if self.current_epoch < self.hparams.switch_epoch:
                formula = 1
            else:
                formula = 2
            loss, distance, scale, one_num = utils.loss(gt_distance, encoding, lr, self.c[0], formula)
            self.one_num.append(one_num)
            self.running_loss += loss
            self.c[0] = scale
            if batch_idx == 0 and (self.current_epoch + 1) % self.hparams.val_freq == 0:
                # breakpoint()
                distance = utils.hyp_dist(encoding) * self.c
                plt.figure(figsize=(5, 5))
                plt.scatter(gt_distance[torch.triu(torch.ones_like(gt_distance), diagonal=1) == 1].detach().cpu().numpy(),
                            distance[torch.triu(torch.ones_like(gt_distance), diagonal=1) == 1].detach().cpu().numpy())
                # plt.scatter(gt_distance.detach().cpu().numpy(),
                #             distance.detach().cpu().numpy())
                buf = io.BytesIO()
                plt.savefig(buf, format='jpeg')
                plt.close()
                buf.seek(0)
                image = PIL.Image.open(buf)
                image = torch.tensor(image.getdata()).view(500, 500, -1).permute(2, 0, 1)
                # breakpoint()
                self.logger.experiment.add_image("dist", image, self.current_epoch)
                self.logger.experiment.add_histogram('emb norm', (encoding**2).sum(-1).detach(), self.current_epoch)
                self.logger.experiment.add_histogram('one cnt', torch.tensor(self.one_num), self.current_epoch)
                self.logger.experiment.add_scalar('c', self.c.detach(), self.current_epoch)
                self.logger.experiment.add_scalar('clr', self.current_clr, self.current_epoch)
                self.logger.experiment.add_scalar('running loss', self.running_loss, self.current_epoch)
                self.logger.experiment.add_scalar('lr', self.current_lr, self.current_epoch)
                self.running_loss = 0
                self.one_num = []
                for n, param in self.named_parameters():
                    try:
                        self.logger.experiment.add_histogram(f'param/{n}', param, self.current_epoch)
                    except:
                        breakpoint()
                # for i, grad in enumerate(torch.autograd.grad(self.dis_loss_w * dis_loss * self.hparams.dis_loss_ratio,
                #                                              self.parameters(), retain_graph=True, allow_unused=True)):
                #     # breakpoint()
                #     try:
                #         self.logger.experiment.add_histogram(f'dis_grad/{i}', grad, self.current_epoch)
                #     except:
                #         continue
        else:
            gt_distance = self.train_data.true_distance(nodes, nodes).to(device)
            encoding = self(seq)
            distance = utils.distance(encoding, encoding.detach(),
                                      self.hparams.distance_mode) * self.hparams.distance_ratio

            not_self = torch.ones_like(distance)
            not_self[torch.arange(0, len(distance)), torch.arange(0, len(distance))] = 0

            dis_loss = utils.mse_loss(distance[not_self == 1], gt_distance[not_self == 1], self.hparams.weighted_method)
            loss = self.dis_loss_w * dis_loss * self.hparams.dis_loss_ratio

            if batch_idx == 0 and self.current_epoch % self.hparams.val_freq == 0:
                # breakpoint()
                plt.figure(figsize=(5, 5))
                plt.scatter(gt_distance[not_self == 1].detach().cpu().numpy(),
                            (distance[not_self == 1] ** 2).detach().cpu().numpy())
                buf = io.BytesIO()
                plt.savefig(buf, format='jpeg')
                plt.close()
                buf.seek(0)
                image = PIL.Image.open(buf)
                image = torch.tensor(image.getdata()).view(500, 500, -1).permute(2, 0, 1)
                # breakpoint()
                self.logger.experiment.add_image("dist", image, self.current_epoch)

        self.val_loss += loss

        return {'loss': loss}

    def training_epoch_end(
            self,
            outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        self.dis_loss_w = 100 + 1e-3 * (self.trainer.current_epoch - 1e4) * (self.trainer.current_epoch > 1e4)

        if self.current_epoch % 100 == 0:
            self.trainer.save_checkpoint(
                os.path.join(
                    self.hparams.model_dir,
                    f"epoch={self.current_epoch}.ckpt"
                )
            )

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.encoder.parameters(), lr=self.hparams.lr)
        # self.optim = [
        #     torch.optim.RMSprop(self.encoder.parameters(), lr=self.hparams.lr),
        #     torch.optim.RMSprop([self.c], lr=self.hparams.clr)
        # ]
        # return self.optim

        # return geoopt.optim.RiemannianAdam(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     betas=(0.9, 0.999),
        #     stabilize=10,
        #     # weight_decay=args.wd
        # )

    # def on_epoch_start(self) -> None:
    #     if (self.current_epoch + 1) % 10 == 0 and self.current_epoch < self.hparams.c_epoch:
    #         self.trainer.optimizers[0] = self.optim[1]
    #     else:
    #         self.trainer.optimizers[0] = self.optim[0]

    def train_dataloader(self) -> DataLoader:
        if self.hparams.fixed_point is not None:
            self.fixed_point = torch.load(self.hparams.fixed_point)
        loader = DataLoader(self.train_data,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_worker,
                            shuffle=True,
                            drop_last=True)
        return loader

    def validation_step(self, batch, batch_idx):

        return {}

    def validation_epoch_end(self, outputs):
        val_loss = self.val_loss
        # print(self.current_epoch)
        self.log('val_loss', val_loss)
        self.val_loss = 0
        self.norm_loss = 0

    def val_dataloader(self):
        # TODO: do a real train/val split
        # breakpoint()
        self.train_data = data.data(self.hparams, calculate_distance_matrix=True)
        loader = DataLoader(self.train_data,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=self.hparams.num_worker,
                            drop_last=False)
        return loader

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:

        if self.trainer.global_step % self.hparams.lr_update_freq == 0:

            # lr = 3e-5 + self.hparams.lr * (0.1 ** (epoch / self.hparams.lr_decay))
            lr = 2e-7 + self.hparams.lr * (0.1 ** (epoch / self.hparams.lr_decay))
            self.current_lr = lr
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = lr

            # for param_group in self.optim[1].param_groups:
            #     param_group['lr'] = param_group['lr'] * 0.95
            #     self.current_clr = param_group['lr']


        super(model, self).optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu,
            using_native_amp,
            using_lbfgs,
        )
