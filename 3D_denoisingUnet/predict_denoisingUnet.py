import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from F2Fd.utils.common import write_array
from F2Fd.dataloader import (
    singleCET_dataset,
    singleCET_FourierDataset,
    singleCET_ProjectedDataset,
)
from F2Fd.predict import load_model, predict_full_tomogram
from pytorch_msssim import ssim
from torchmetrics.functional import peak_signal_noise_ratio
import sys
import json
from glob import glob
import pathlib


def parse_null_arg(arg, dtype):
    try:
        return dtype(arg)
    except ValueError:
        return None


###################### Parse arguments ###################

args = json.loads(sys.argv[1])
exp_name = sys.argv[2]

n_bernoulli_samples = args["n_bernoulli_samples_prediction"]
total_samples = args["total_samples_prediction"]

predict_N_times = args["predict_N_times"]

path_to_fourier_samples = args["path_to_fourier_samples"]
clip = args["clip"]

tomo_name = args["tomo_name"]

try:  # I used this as input to compare results with and without resampling
    # According to what I found (Notebook 6.20) resampling is significantly better
    resample_patch_each_iter = args["resample_patch_each_iter"]
except KeyError:
    resample_patch_each_iter = True

try:
    version = args["version"]
except KeyError:
    version = None

##################################### Model and dataloader ####################################################

tensorboard_logdir = os.path.join(
    args["model_logdir"], "%s/%s/" % (tomo_name, exp_name)
)
pathlib.Path(tensorboard_logdir).mkdir(parents=True, exist_ok=True)

# use the last version for the predictions
if version is None:
    version = len(glob(tensorboard_logdir + "/version*")) - 1
    version = "version_%i" % version

logdir = os.path.join(tensorboard_logdir, version)

with open(os.path.join(logdir, "experiment_args.json"), "r") as f:
    exp_args = json.load(f)

deconv_kwargs = exp_args["deconv_kwargs"]
predict_simRecon = exp_args["predict_simRecon"]
use_deconv_as_target = exp_args["use_deconv_as_target"]
bernoulliMask_prob = parse_null_arg(exp_args["bernoulliMask_prob"], float)

print("Loading model from: %s" % logdir)

model, hparams = load_model(logdir, DataParallel=True)

dataset = hparams["dataset"]

cet_path = hparams["tomo_path"]
gt_cet_path = hparams["gt_tomo_path"]
if gt_cet_path == 'null':
    gt_cet_path = None

p = float(hparams["Dataloader.p"])  # Bernoulli mask probability
dropout_p = float(hparams["p"]) # dropout probability, same as dropout_p
subtomo_length = int(hparams["subtomo_length"])
n_features = int(hparams["n_features"])
volumetric_scale_factor = parse_null_arg(hparams["vol_scale_factor"], int)
Vmask_probability = parse_null_arg(hparams["Vmask_probability"], float)
Vmask_pct = parse_null_arg(hparams["Vmask_pct"], float)
alpha = hparams["loss_fn"]["alpha"]

if dataset == "singleCET_dataset":
    my_dataset = singleCET_dataset(
        cet_path,
        subtomo_length=subtomo_length,
        p=p,
        n_bernoulli_samples=n_bernoulli_samples,
        volumetric_scale_factor=volumetric_scale_factor,
        Vmask_probability=Vmask_probability,
        Vmask_pct=Vmask_pct,
        transform=None,
        n_shift=0,
        gt_tomo_path=gt_cet_path,
        clip=clip,
        **deconv_kwargs
    )

elif dataset == "singleCET_FourierDataset":
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
        path_to_fourier_samples=path_to_fourier_samples,
        clip=clip,
        **deconv_kwargs
    )

elif dataset == "singleCET_ProjectedDataset":
    my_dataset = singleCET_ProjectedDataset(
        cet_path,
        subtomo_length=subtomo_length,
        transform=None,
        n_shift=0,
        gt_tomo_path=gt_cet_path,
        predict_simRecon=predict_simRecon,
        use_deconv_as_target=use_deconv_as_target,
        **deconv_kwargs
    )

################### Make prediction plots ########################

print("Predicting full tomogram...")

# this is taking two means: first per bernoulli batches, and then again for each time the model was run
# total predictions is the inner_range*n_bernoulli_samples
denoised_tomo = predict_full_tomogram(
    my_dataset, model, resample_patch_each_iter, N=predict_N_times
)
print("Done!")

plt.figure(figsize=(12, 8))
plt.hist(denoised_tomo.numpy().flatten(), alpha=0.5, label="Denoised")
plt.hist(my_dataset.data.numpy().flatten(), alpha=0.5, label="Original")
plt.legend()
outfile = os.path.join(logdir, "Full_tomo_histograms.png")
plt.savefig(outfile)

zidx, yidx, xidx = np.array(my_dataset.data.shape) // 2

fig, (ax0, ax1) = plt.subplots(2, 3, figsize=(25, 15))
list(map(lambda axi: axi.set_axis_off(), np.array([ax0, ax1]).ravel()))

ax0[1].set_title("Original: YX, ZX, ZY")
ax0[0].imshow(my_dataset.data[zidx])
ax0[1].imshow(my_dataset.data[:, yidx, :])
ax0[2].imshow(my_dataset.data[:, :, xidx])

ax1[1].set_title("Denoised: YX, ZX, ZY")
ax1[0].imshow(denoised_tomo[zidx])
ax1[1].imshow(denoised_tomo[:, yidx, :])
ax1[2].imshow(denoised_tomo[:, :, xidx])

plt.tight_layout()
outfile = os.path.join(logdir, "original_vs_denoised.png")
plt.savefig(outfile, dpi=200)


############### Write logs and prediction ########################

if my_dataset.gt_data is not None:
    X = denoised_tomo.unsqueeze(0)
    X = my_dataset.scale(X)
    X0 = my_dataset.data.unsqueeze(0)
    X0 = my_dataset.scale(X0)
    Y = my_dataset.gt_data.unsqueeze(0)
    Y = my_dataset.scale(Y)
    ssim_full = ssim(X, Y, data_range=1)
    ssim_full = float(ssim_full)
    ssim_baseline = ssim(X0, Y, data_range=1)
    ssim_baseline = float(ssim_baseline)

    psnr_full = peak_signal_noise_ratio(X, Y, data_range=1)
    psnr_full = float(psnr_full)
    psnr_baseline = peak_signal_noise_ratio(X0, Y, data_range=1)
    psnr_baseline = float(psnr_baseline)

    extra_hparams = {
        "full_tomo_ssim": ssim_full,
        "full_tomo_psnr": psnr_full,
        "baseline_ssim": ssim_baseline,
        "baseline_psnr": psnr_baseline,
    }
else:
    extra_hparams = {
        "full_tomo_ssim": None,
        "full_tomo_psnr": None,
        "baseline_ssim": None,
        "baseline_psnr": None,
    }

if "full_tomo_ssim" not in hparams.keys():
    sdump = yaml.dump(extra_hparams)
    hparams_file = os.path.join(tensorboard_logdir, version)
    hparams_file = os.path.join(hparams_file, "hparams.yaml")
    with open(hparams_file, "a") as fo:
        fo.write(sdump)
else:
    for k in ["full_tomo_ssim", "full_tomo_psnr", "baseline_ssim", "baseline_psnr"]:
        hparams[k] = extra_hparams[k]

    sdump = yaml.dump(hparams)
    hparams_file = os.path.join(tensorboard_logdir, version)
    hparams_file = os.path.join(hparams_file, "hparams.yaml")
    with open(hparams_file, "w") as fo:
        fo.write(sdump)

filename = cet_path.split("/")[-1].replace(".mrc", "_s2sDenoised")
filename = os.path.join(logdir, "%s.mrc" % (filename))

write_array(denoised_tomo.numpy(), filename)

# # delete the samples after prediction is finished
# if predict_on_saved_fourier_samples:
#     os.system("rm %s" %path_to_fourier_samples)
