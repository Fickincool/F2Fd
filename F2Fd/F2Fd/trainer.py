from ensurepip import version
import os
import yaml

from torch.utils.data import DataLoader
import torch

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from F2Fd.dataloader import singleCET_FourierDataset



class denoisingTrainer:
    def __init__(
        self,
        model,
        my_dataset,
        tensorboard_logdir,
        model_name
    ):
        super().__init__()

        self.model = model
        self.dataset = my_dataset

        # logs
        self.tensorboard_logdir = tensorboard_logdir
        self.model_name = model_name

        self.run_init_asserts()

        return

    def run_init_asserts(self):
        
        return

    def train(
        self,
        collate_fn,
        batch_size,
        epochs,
        num_gpus,
        accelerator="gpu",
        strategy="ddp",
        transform=None,
        comment=None
    ):

        print(
            "Size of dataset: %i, Steps per epoch: %i. \n"
            % (len(self.dataset), len(self.dataset) / (batch_size * num_gpus))
        )

        train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        logger = pl_loggers.TensorBoardLogger(
            self.tensorboard_logdir, name="", default_hp_metric=False
        )

        early_stop_callback = EarlyStopping(
            monitor="hp/train_loss",
            min_delta=1e-4,
            patience=100,
            verbose=True,
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [early_stop_callback, lr_monitor]

        trainer = Trainer(
            logger=logger,
            log_every_n_steps=1,
            gpus=num_gpus,
            max_epochs=epochs,
            enable_progress_bar=False,
            callbacks=callbacks,
            accelerator=accelerator,
            strategy=strategy,
        )

        trainer.fit(self.model, train_loader)

        if trainer.is_global_zero:
            #### Log additional hyperparameters #####
            version_folder = os.path.join(
                self.tensorboard_logdir, "version_%i" % self.model.logger.version
            )
            hparams_file = os.path.join(version_folder, "hparams.yaml")

            # in this dataset, we saved the samples at some point to be used later for prediction
            # this doesn't seem to yield any advantage.
            # if type(self.dataset)==singleCET_FourierDataset: 
            #     samples_file = os.path.join(version_folder, "singleCET_FourierDataset.samples")
            #     torch.save(self.dataset.fourier_samples, samples_file)

            dataset_params = ['tomo_path', 'gt_tomo_path', 'subtomo_length',
             'vol_scale_factor', 'Vmask_probability', 'Vmask_pct', 'use_deconv_data',
             'use_deconv_as_target', 'predict_simRecon']

            try:
                p = self.dataset.__dict__['p']
            except KeyError:
                p = None
                
            extra_hparams = {
                "transform": transform,
                "Dataloader": type(self.dataset),
                "Version_comment":comment,
                "Dataloader.p": p,
                'Dataloader.batch_size':batch_size,
                "dropout_p": self.model.p
            }

            for key in dataset_params:
                try:
                    extra_hparams[key] = self.dataset.__dict__[key]
                except KeyError:
                    extra_hparams[key] = None

            sdump = yaml.dump(extra_hparams)

            with open(hparams_file, "a") as fo:
                fo.write(sdump)

        return

def aggregate_bernoulliSamples(batch):
    """Concatenate batch+bernoulli samples. Shape [B*M, C, S, S, S]

    Dataset returns [M, C, S, S, S] and dataloader returns [B, M, C, S, S, S].
    This function concatenates the array in order to make a batch be the set of bernoulli samples of each of the B subtomos.
    """
    bernoulli_subtomo = torch.cat([b[0] for b in batch], axis=0)
    target = torch.cat([b[1] for b in batch], axis=0)
    bernoulli_mask = torch.cat([b[2] for b in batch], axis=0)
    
    try:
        gt_subtomo = torch.cat([b[3] for b in batch], axis=0)
    except TypeError:
        gt_subtomo = None
    
    return bernoulli_subtomo, target, bernoulli_mask, gt_subtomo

def aggregate_bernoulliSamples2(batch):
    """Concatenate batch+bernoulli samples. Shape [B*M, C, S, S, S]

    Dataset returns [M, C, S, S, S] and dataloader returns [B, M, C, S, S, S].
    This function concatenates the array in order to make a batch be the set of bernoulli samples of each of the B subtomos.
    """
    bernoulli_subtomo = torch.cat([b[0] for b in batch], axis=0)
    target = torch.cat([b[1] for b in batch], axis=0)
    
    try:
        gt_subtomo = torch.cat([b[2] for b in batch], axis=0)
    except TypeError:
        gt_subtomo = None
    
    return bernoulli_subtomo, target, gt_subtomo

def collate_for_oneBernoulliSample(batch):
    "Default pytorch collate_fn does not handle None. This ignores None values from the batch."
    batch = [list(filter(lambda x: x is not None, b)) for b in batch]
    return torch.utils.data.dataloader.default_collate(batch)