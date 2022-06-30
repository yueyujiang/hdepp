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
        if self.args.distance_mode == 'hyperbolic':
            # self.hyperbolic_layer = nn.Sequential(
            #     nn.Flatten(1),
            #     nn.Linear(args.sequence_length * args.h_channel, 64),
            #     utils.MobiusLinear(64, args.embedding_size)
            # )
            self.pre_layer = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(args.sequence_length * args.h_channel, args.embedding_size)
            )
            # self.hyperbolic_layer = nn.nn.Sequential(
            #     utils.MobiusLinear(args.embedding_size, args.embedding_size, nonlin=torch.relu),
            #     utils.MobiusLinear(args.embedding_size, args.embedding_size, nonlin=torch.relu)
            # )
            self.hyperbolic_layer = utils.MobiusLinear(args.embedding_size, args.embedding_size, hyperbolic_bias=True, nonlin=None)

        else:
            self.linear = nn.Conv1d(args.h_channel,
                                    args.embedding_size,
                                    args.sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, channel, seq_length = x.shape

        x = self.celu(self.conv(x))
        x = self.resblocks(x)
        if self.args.distance_mode == 'hyperbolic':
            x = self.pre_layer(x)
            x = pmath.expmap0(x)
            x = self.hyperbolic_layer(x)
        else:
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
        self.encoder = encoder(self.hparams)
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
            self.hparams.weighted_method == 'fm'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(x)
        return encoding

    def training_step(self, batch, batch_idx):

        nodes = batch['nodes']
        seq = batch['seqs']
        device = seq.device

        gt_distance = self.train_data.true_distance(nodes, nodes).to(device)

        # if self.hparams.distance_mode == 'hyperbolic':
        #
        #     # if self.current_epoch > 500:
        #     #     self.hparams.weighted_method = 'fm'
        #
        #     encoding = self(seq)
        #     n = len(encoding)
        #     # encoding = encoding / (encoding.norm(dim=-1, keepdim=True).detach() / math.sqrt(1 - 0.5e-4))
        #     #distance = pmath.dist(encoding, encoding.detach())
        #     #distance = pmath.dist(encoding.view(n, 1, -1), encoding.view(1, n, -1).detach())
        #     distance = utils.hyperbolic_dist(encoding, encoding)
        #     not_self = torch.ones_like(distance)
        #     not_self[torch.arange(0, len(distance)), torch.arange(0, len(distance))] = 0
        #     # breakpoint()
        #     dis_loss = utils.mse_loss(distance[not_self == 1], gt_distance[not_self == 1],
        #                               self.hparams.weighted_method, hyperbolic=True)
        #     # norm_loss = (torch.norm(encoding, dim=-1) ** 50).mean(0)
        #     # norm_loss = (torch.norm(encoding, dim=-1) - 1).abs().mean(0)
        #     norm_loss = 0
        #     loss = self.dis_loss_w * dis_loss * self.hparams.dis_loss_ratio
        #     if batch_idx == 0 and self.current_epoch % self.hparams.val_freq == 0:
        #         # breakpoint()
        #         plt.figure(figsize=(5, 5))
        #         plt.scatter(gt_distance[not_self == 1].detach().cpu().numpy(),
        #                     distance[not_self == 1].detach().cpu().numpy())
        #         # plt.scatter(gt_distance.detach().cpu().numpy(),
        #         #             distance.detach().cpu().numpy())
        #         buf = io.BytesIO()
        #         plt.savefig(buf, format='jpeg')
        #         plt.close()
        #         buf.seek(0)
        #         image = PIL.Image.open(buf)
        #         image = torch.tensor(image.getdata()).view(500, 500, -1).permute(2, 0, 1)
        #         # breakpoint()
        #         self.logger.experiment.add_image("dist", image, self.current_epoch)
        #         self.logger.experiment.add_histogram('emb norm', (encoding**2).sum(-1).detach(), self.current_epoch)
        #         for n, param in self.named_parameters():
        #             try:
        #                 self.logger.experiment.add_histogram(f'param/{n}', param, self.current_epoch)
        #             except:
        #                 breakpoint()
        #         for i, grad in enumerate(torch.autograd.grad(self.dis_loss_w * dis_loss * self.hparams.dis_loss_ratio,
        #                                                      self.parameters(), retain_graph=True, allow_unused=True)):
        #             # breakpoint()
        #             try:
        #                 self.logger.experiment.add_histogram(f'dis_grad/{i}', grad, self.current_epoch)
        #             except:
        #                 breakpoint()
        # else:
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

        if self.hparams.distance_mode == 'hyperbolic':
            return {'loss': loss}
        else:
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
        # return torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)
        return geoopt.optim.RiemannianAdam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            stabilize=10,
            # weight_decay=args.wd
        )

    def train_dataloader(self) -> DataLoader:
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
            lr = 2e-5 + self.hparams.lr * (0.1 ** (epoch / self.hparams.lr_decay))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

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
