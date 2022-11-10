import os
import torch
from glob import glob
from F2Fd.utils.common import write_array
from F2Fd.utils import setup
from F2Fd.dataloader import singleCET_FourierDataset
from F2Fd.misc import parse_cet_paths
import pathlib
from time import sleep
import gc

def make_fourier_samples(args, exp_name):

    p = args["p"]  # bernoulli masking probability
    # dropout_p = args["dropout_p"]

    n_bernoulli_samples = args["n_bernoulli_samples"]
    total_samples = args["total_samples"]
    # alpha = args["alpha"]
    volumetric_scale_factor = args["volumetric_scale_factor"]
    Vmask_probability = args["Vmask_probability"]
    Vmask_pct = args["Vmask_pct"]

    subtomo_length = args["subtomo_length"]
    # n_features = args["n_features"]
    # batch_size = args["batch_size"]
    # epochs = args["epochs"]
    # lr = args["lr"]
    # num_gpus = args["num_gpus"]
    # predict_simRecon = args["predict_simRecon"]
    # use_deconv_as_target = args["use_deconv_as_target"]

    bernoulliMask_prob = args["bernoulliMask_prob"]
    input_as_target = args["input_as_target"]

    tomo_name = args["tomo_name"]

    deconv_kwargs = args["deconv_kwargs"]

    cet_path, gt_cet_path = args['cet_path'], args['gt_cet_path']
    ##################################### Model and dataloader ####################################################

    tensorboard_logdir = os.path.join(
        args["model_logdir"], "%s/%s/" % (tomo_name, exp_name)
    )
    pathlib.Path(tensorboard_logdir).mkdir(parents=True, exist_ok=True)

    # comment = args["comment"]

    sample_path = os.path.join(tensorboard_logdir, 'FourierSamples/')
    pathlib.Path(sample_path).mkdir(parents=True, exist_ok=True)

    samples_file = os.path.join(sample_path, "singleCET_FourierDataset.samples")

    if os.path.exists(samples_file):
        return samples_file

    else:
        my_dataset = singleCET_FourierDataset(
            cet_path,
            subtomo_length=subtomo_length,
            p=p,
            n_bernoulli_samples=n_bernoulli_samples,
            total_samples=total_samples,
            volumetric_scale_factor=volumetric_scale_factor,
            Vmask_probability=Vmask_probability,
            Vmask_pct=Vmask_pct,
            transform=None,
            n_shift=0,
            gt_tomo_path=gt_cet_path,
            bernoulliMask_prob=bernoulliMask_prob,
            input_as_target=input_as_target,
            **deconv_kwargs
        )

        torch.save(my_dataset.fourier_samples, samples_file)
        # print("0. before deleting my_dataset")
        del my_dataset

        gc.collect()

        # print("0. After deleting")
        sleep(10)

        return samples_file