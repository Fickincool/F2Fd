import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim
from torchmetrics.functional import peak_signal_noise_ratio

# This is just a copy from the original implementation in:
# https://github.com/NVIDIA/partialconv/tree/master/models
from F2Fd.partialconv3d import PartialConv3d
from F2Fd.partialconv2d import PartialConv2d


class Denoising_3DUNet(pl.LightningModule):
    def __init__(self, loss_fn, lr, n_features, p, n_bernoulli_samples):
        """Expected input: [B, C, S, S, S] where B the batch size, C input channels and S the subtomo length.
        The data values are expected to be standardized and [0, 1] scaled.
        """

        super().__init__()
        self.loss_fn = loss_fn
        self.lr = lr
        self.n_features = n_features
        self.p = p
        self.n_bernoulli_samples = n_bernoulli_samples
        self.in_channels = 1
        self.save_hyperparameters()

        # Encoder blocks
        self.EB0 = PartialConv3d(
            self.in_channels, self.n_features, kernel_size=3, padding=1
        )
        self.EB1 = self.encoder_block()
        self.EB2 = self.encoder_block()
        self.EB3 = self.encoder_block()
        self.EB4 = self.encoder_block()
        self.EB5 = self.encoder_block()
        self.EB6 = self.encoder_block_bottom()

        # Upsampling
        self.up54 = nn.Upsample(scale_factor=2)
        self.up43 = nn.Upsample(scale_factor=2)
        self.up32 = nn.Upsample(scale_factor=2)
        self.up21 = nn.Upsample(scale_factor=2)
        self.up10 = nn.Upsample(scale_factor=2)

        # decoder blocks
        self.DB5 = self.decoder_block(2 * n_features, 2 * n_features)
        self.DB4 = self.decoder_block(3 * n_features, 2 * n_features)
        self.DB3 = self.decoder_block(3 * n_features, 2 * n_features)
        self.DB2 = self.decoder_block(3 * n_features, 2 * n_features)
        self.DB1 = self.decoder_block_top()

        return

    def forward(self, x: torch.Tensor):
        "Input tensor of shape [batch_size, channels, tomo_side, tomo_side, tomo_side]"
        ##### ENCODER #####
        e0 = self.EB0(x)  # no downsampling, n_features = 48
        e1 = self.EB1(e0)  # downsamples 1/2
        e2 = self.EB2(e1)  # 1/4
        e3 = self.EB3(e2)  # 1/8
        e4 = self.EB4(e3)  # 1/16
        e5 = self.EB5(e4)  # 1/32
        e6 = self.EB6(e5)  # only Pconv and LReLu
        # for debugging
        # print('EB0 (no downsampling):', e0.shape)
        # print('EB1:', e1.shape)
        # print('EB2:', e2.shape)
        # print('EB3:', e3.shape)
        # print('EB4:', e4.shape)
        # print('EB5:', e5.shape)
        # print('EB6: (no downsampling)', e6.shape)

        ##### DECODER #####
        d5 = self.up54(e6)  # 1/16
        d5 = torch.concat([d5, e4], axis=1)  # 1/16, n_freatures = 96
        d5 = self.DB5(d5)  # 1/16

        d4 = self.up43(d5)  # 1/8
        d4 = torch.concat([d4, e3], axis=1)  # 1/8 n_features = 144
        d4 = self.DB4(d4)  # 1/8 n_features = 96

        d3 = self.up32(d4)  # 1/4
        d3 = torch.concat([d3, e2], axis=1)  # 1/4
        d3 = self.DB3(d3)  # 1/4

        d2 = self.up21(d3)  # 1/2
        d2 = torch.concat([d2, e1], axis=1)  # 1/2
        d2 = self.DB2(d2)  # 1/2

        d1 = self.up10(d2)
        d1 = torch.concat([d1, x], axis=1)
        x = self.DB1(d1)

        return x

    def encoder_block(self):
        layer = nn.Sequential(
            PartialConv3d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        return layer

    def encoder_block_bottom(self):
        layer = nn.Sequential(
            PartialConv3d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        return layer

    def decoder_block(self, n_features_in, n_features_out):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv3d(n_features_in, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv3d(n_features_out, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        return layer

    def decoder_block_top(self):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv3d(
                2 * self.n_features + self.in_channels, 64, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv3d(32, self.in_channels, kernel_size=3, padding=1),
            # This is in the original implementation paper, but here it doesn't help.
            # It forces data to be (almost) positive, while tomogram data is close to normal around zero.
            # nn.LeakyReLU(0.1)
        )
        return layer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        factor = 0.1

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    "min",
                    verbose=True,
                    patience=10,
                    min_lr=1e-8,
                    factor=factor,
                ),
                "monitor": "hp/train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def training_step(self, batch):
        bernoulli_subtomo, target, bernoulli_mask, gt_subtomo = batch
        pred = self(bernoulli_subtomo)
        loss = self.loss_fn(pred, target, bernoulli_mask)

        self.log(
            "hp/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if gt_subtomo is not None:
            bernoulliBatch_subtomo = self.batch2bernoulliBatch(bernoulli_subtomo)
            bernoulliBatch_pred = self.batch2bernoulliBatch(pred)
            bernoulliBatch_gt_subtomo = self.batch2bernoulliBatch(gt_subtomo)
            baseline_ssim, baseline_psnr = self.ssim_psnr_monitoring(
                bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
            )
            monitor_ssim, monitor_psnr = self.ssim_psnr_monitoring(
                bernoulliBatch_pred, bernoulliBatch_gt_subtomo
            )

            self.log(
                "ssim/baseline",
                baseline_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "ssim/predicted",
                monitor_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "psnr/baseline",
                baseline_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "psnr/predicted",
                monitor_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        tensorboard = self.logger.experiment
        tensorboard.add_histogram(
            "Intensity distribution", pred.detach().cpu().numpy().flatten()
        )

        return loss

    def batch2bernoulliBatch(self, subtomo):
        return torch.split(subtomo, self.n_bernoulli_samples)

    def ssim_psnr_monitoring(self, bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo):
        ssim_monitor = 0
        psnr_monitor = 0
        for bBatch_subtomo, bBatch_gt in zip(
            bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
        ):
            # we first normalize the images
            X = bBatch_subtomo.mean(0)
            X = (X - X.min()) / (X.max() - X.min() + 1e-8)
            Y = bBatch_gt.mean(0)
            Y = (Y - Y.min()) / (Y.max() - Y.min() + 1e-8)

            _ssim, _psnr = float(ssim(X, Y, data_range=1)), float(
                peak_signal_noise_ratio(X, Y, data_range=1)
            )
            ssim_monitor += _ssim
            psnr_monitor += _psnr

        # take the mean wrt batch
        ssim_monitor = ssim_monitor / len(bernoulliBatch_gt_subtomo)
        psnr_monitor = psnr_monitor / len(bernoulliBatch_gt_subtomo)

        return ssim_monitor, psnr_monitor


class Denoising_3DUNet_v2(Denoising_3DUNet):
    def training_step(self, batch):
        bernoulli_subtomo, target, gt_subtomo = batch
        pred = self(bernoulli_subtomo)
        loss = self.loss_fn(pred, target)

        self.log(
            "hp/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if gt_subtomo is not None:
            bernoulliBatch_subtomo = self.batch2bernoulliBatch(bernoulli_subtomo)
            bernoulliBatch_pred = self.batch2bernoulliBatch(pred)
            bernoulliBatch_gt_subtomo = self.batch2bernoulliBatch(gt_subtomo)
            baseline_ssim, baseline_psnr = self.ssim_psnr_monitoring(
                bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
            )
            monitor_ssim, monitor_psnr = self.ssim_psnr_monitoring(
                bernoulliBatch_pred, bernoulliBatch_gt_subtomo
            )

            self.log(
                "ssim/baseline",
                baseline_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "ssim/predicted",
                monitor_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "psnr/baseline",
                baseline_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "psnr/predicted",
                monitor_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        tensorboard = self.logger.experiment
        tensorboard.add_histogram(
            "Intensity distribution", pred.detach().cpu().numpy().flatten()
        )

        return loss


############################### 2D ################################################################


class Denoising_2DUNet(pl.LightningModule):
    def __init__(self, loss_fn, lr, n_features, p, n_bernoulli_samples):
        """Expected input: [B, C, S, S] where B the batch size, C input channels and S the subtomo length.
        The data values are expected to be standardized and [0, 1] scaled.
        """

        super().__init__()
        self.loss_fn = loss_fn
        self.lr = lr
        self.n_features = n_features
        self.p = p
        self.n_bernoulli_samples = n_bernoulli_samples
        self.in_channels = 1
        self.save_hyperparameters()

        # Encoder blocks
        self.EB0 = PartialConv2d(
            self.in_channels, self.n_features, kernel_size=3, padding=1
        )
        self.EB1 = self.encoder_block()
        self.EB2 = self.encoder_block()
        self.EB3 = self.encoder_block()
        self.EB4 = self.encoder_block()
        self.EB5 = self.encoder_block()
        self.EB6 = self.encoder_block_bottom()

        # Upsampling
        self.up54 = nn.Upsample(scale_factor=2)
        self.up43 = nn.Upsample(scale_factor=2)
        self.up32 = nn.Upsample(scale_factor=2)
        self.up21 = nn.Upsample(scale_factor=2)
        self.up10 = nn.Upsample(scale_factor=2)

        # decoder blocks
        self.DB5 = self.decoder_block(2 * n_features, 2 * n_features)
        self.DB4 = self.decoder_block(3 * n_features, 2 * n_features)
        self.DB3 = self.decoder_block(3 * n_features, 2 * n_features)
        self.DB2 = self.decoder_block(3 * n_features, 2 * n_features)
        self.DB1 = self.decoder_block_top()

        return

    def forward(self, x: torch.Tensor):
        "Input tensor of shape [batch_size, channels, tomo_side, tomo_side, tomo_side]"
        ##### ENCODER #####
        e0 = self.EB0(x)  # no downsampling, n_features = 48
        e1 = self.EB1(e0)  # downsamples 1/2
        e2 = self.EB2(e1)  # 1/4
        e3 = self.EB3(e2)  # 1/8
        e4 = self.EB4(e3)  # 1/16
        e5 = self.EB5(e4)  # 1/32
        e6 = self.EB6(e5)  # only Pconv and LReLu
        # for debugging
        # print('EB0 (no downsampling):', e0.shape)
        # print('EB1:', e1.shape)
        # print('EB2:', e2.shape)
        # print('EB3:', e3.shape)
        # print('EB4:', e4.shape)
        # print('EB5:', e5.shape)
        # print('EB6: (no downsampling)', e6.shape)

        ##### DECODER #####
        d5 = self.up54(e6)  # 1/16
        d5 = torch.concat([d5, e4], axis=1)  # 1/16, n_freatures = 96
        d5 = self.DB5(d5)  # 1/16

        d4 = self.up43(d5)  # 1/8
        d4 = torch.concat([d4, e3], axis=1)  # 1/8 n_features = 144
        d4 = self.DB4(d4)  # 1/8 n_features = 96

        d3 = self.up32(d4)  # 1/4
        d3 = torch.concat([d3, e2], axis=1)  # 1/4
        d3 = self.DB3(d3)  # 1/4

        d2 = self.up21(d3)  # 1/2
        d2 = torch.concat([d2, e1], axis=1)  # 1/2
        d2 = self.DB2(d2)  # 1/2

        d1 = self.up10(d2)
        d1 = torch.concat([d1, x], axis=1)
        x = self.DB1(d1)

        return x

    def encoder_block(self):
        layer = nn.Sequential(
            PartialConv2d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return layer

    def encoder_block_bottom(self):
        layer = nn.Sequential(
            PartialConv2d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        return layer

    def decoder_block(self, n_features_in, n_features_out):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv2d(n_features_in, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv2d(n_features_out, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        return layer

    def decoder_block_top(self):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv2d(
                2 * self.n_features + self.in_channels, 64, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv2d(32, self.in_channels, kernel_size=3, padding=1),
        )
        return layer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        factor = 0.1

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    "min",
                    verbose=True,
                    patience=10,
                    min_lr=1e-8,
                    factor=factor,
                ),
                "monitor": "hp/train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def training_step(self, batch):
        bernoulli_subtomo, target, bernoulli_mask, gt_subtomo = batch
        pred = self(bernoulli_subtomo)
        loss = self.loss_fn(pred, target, bernoulli_mask)

        self.log(
            "hp/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if gt_subtomo is not None:
            bernoulliBatch_subtomo = self.batch2bernoulliBatch(bernoulli_subtomo)
            bernoulliBatch_gt_subtomo = self.batch2bernoulliBatch(gt_subtomo)
            monitor_ssim = self.ssim_monitoring(
                bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
            )
            monitor_psnr = self.psnr_monitoring(
                bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
            )

            self.log(
                "hp/ssim",
                monitor_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "hp/psnr",
                monitor_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        tensorboard = self.logger.experiment
        tensorboard.add_histogram(
            "Intensity distribution", pred.detach().cpu().numpy().flatten()
        )

        return loss

    def batch2bernoulliBatch(self, subtomo):
        return torch.split(subtomo, self.n_bernoulli_samples)

    def ssim_monitoring(self, bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo):
        monitor = 0
        for bBatch_subtomo, bBatch_gt in zip(
            bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
        ):
            _monitor = ssim(bBatch_subtomo.mean((0)), bBatch_gt.mean((0)))
            monitor += _monitor

        # take the mean wrt batch
        return monitor / len(bernoulliBatch_gt_subtomo)

    def psnr_monitoring(self, bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo):
        monitor = 0
        for bBatch_subtomo, bBatch_gt in zip(
            bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
        ):
            i, j = bBatch_subtomo.mean((0)), bBatch_gt.mean(
                (0)
            )  # 1 Channel, 3D images [C, S, S, S]
            # data is standardized, we don't expect to see any value further than 10 std from the origin
            data_range = 18
            if j.abs().max() > 9:
                raise ValueError(
                    "Found input values for ground truth further than 9 std away from the origin."
                )
            _monitor = peak_signal_noise_ratio(i, j, data_range)
            monitor += _monitor

        # take the mean wrt batch
        return monitor / len(bernoulliBatch_gt_subtomo)


class Denoising_2DUNet_v2(Denoising_2DUNet):
    def training_step(self, batch):
        bernoulli_subtomo, target, gt_subtomo = batch
        pred = self(bernoulli_subtomo)
        loss = self.loss_fn(pred, target)

        self.log(
            "hp/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if gt_subtomo is not None:
            bernoulliBatch_subtomo = self.batch2bernoulliBatch(bernoulli_subtomo)
            bernoulliBatch_gt_subtomo = self.batch2bernoulliBatch(gt_subtomo)
            monitor_ssim = self.ssim_monitoring(
                bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
            )
            monitor_psnr = self.psnr_monitoring(
                bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
            )

            self.log(
                "hp/ssim",
                monitor_ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "hp/psnr",
                monitor_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        tensorboard = self.logger.experiment
        tensorboard.add_histogram(
            "Intensity distribution", pred.detach().cpu().numpy().flatten()
        )

        return loss
