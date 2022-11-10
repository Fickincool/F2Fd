import torch
import numpy as np
from torch.utils.data import Dataset
from F2Fd.utils.common import read_array
import tomopy.sim.project as proj
from tomopy.recon.algorithm import recon
from .deconvolution import tom_deconv_tomo
from scipy.stats import multivariate_normal
from tqdm import tqdm
from joblib import Parallel, delayed


class singleCET_dataset(Dataset):
    def __init__(
        self,
        tomo_path,
        subtomo_length,
        p,
        n_bernoulli_samples=6,
        volumetric_scale_factor=4,
        Vmask_probability=0,
        Vmask_pct=0.1,
        transform=None,
        n_shift=0,
        gt_tomo_path=None,
        clip=True,
        **deconv_kwargs
    ):
        """
        Load cryoET dataset for self2self denoising.

        The dataset consists of subtomograms of shape [M, C, S, S, S] C (equal to 1) is the number of channels and
        S is the subtomogram side length.

        - tomo_path: tomogram path
        - subtomo_length: side length of the patches to be used for training
        - p: probability of an element to be zeroed
        - volumetric_scale_factor: times the original tomogram shape will be reduced
        to take bernoulli point samples before upsampling into volumetric bernoulli blind spots.
        """
        self.tomo_path = tomo_path

        self.data = torch.tensor(read_array(tomo_path))
        
        # first deconvolve and then clip and standardize
        self.deconv_kwargs = {"vol": self.data.numpy(), **deconv_kwargs}
        self.use_deconv_data = self.check_deconv_kwargs(deconv_kwargs)
        if self.use_deconv_data:
            self.data = tom_deconv_tomo(**self.deconv_kwargs)
            self.data = torch.tensor(self.data)
        else:
            pass

        if clip:
            self.data = self.clip(self.data)
        self.data = self.standardize(self.data)

        self.gt_tomo_path = gt_tomo_path
        if gt_tomo_path is not None:
            self.gt_data = torch.tensor(read_array(gt_tomo_path))
            # the shrec data ground truth values are inverted
            if "shrec2021" in self.gt_tomo_path:
                self.gt_data = -1*self.gt_data
                self.gt_data = self.gt_data - self.gt_data.min()
            if clip:
                self.gt_data = self.clip(self.gt_data)
            self.gt_data = self.standardize(self.gt_data)
        else:
            self.gt_data = None

        self.n_shift = n_shift
        self.tomo_shape = self.data.shape
        self.subtomo_length = subtomo_length
        self.grid = self.create_grid()
        self.transform = transform  # think how to implement this
        self.p = p
        self.Vmask_pct = Vmask_pct
        self.dropout = torch.nn.Dropout(p=p)
        self.dropoutV = torch.nn.Dropout(p=self.Vmask_pct)
        self.upsample = torch.nn.Upsample(scale_factor=volumetric_scale_factor)
        self.vol_scale_factor = volumetric_scale_factor
        self.channels = 1
        self.Vmask_probability = Vmask_probability  # otherwise use Pmask

        self.n_bernoulli_samples = n_bernoulli_samples

        self.run_init_asserts()

        return

    def check_deconv_kwargs(self, deconv_kwargs):
        if bool(deconv_kwargs):
            deconv_args = [
                "angpix",
                "defocus",
                "snrfalloff",
                "deconvstrength",
                "highpassnyquist",
            ]
            for arg in deconv_args:
                if arg in self.deconv_kwargs.keys():
                    continue
                else:
                    raise KeyError(
                        'Missing required deconvolution argument: "%s"' % arg
                    )
            use_deconv_data = True
            print("Using deconvolved data for training.")

        else:
            use_deconv_data = False

        return use_deconv_data

    def run_init_asserts(self):
        if self.subtomo_length % self.vol_scale_factor != 0:
            raise ValueError(
                "Length of subtomograms must be a multiple of the volumetric scale factor."
            )
        if self.subtomo_length % 32 != 0:
            raise ValueError(
                "Length of subtomograms must be a multiple of 32 to run the network."
            )

        return

    def standardize(self, X: torch.tensor):
        mean = X.mean()
        std = X.std()

        new_X = (X - mean) / std

        return new_X

    def scale(self, X):
        scaled = (X - X.min()) / (X.max() - X.min() + 1e-8)
        return scaled

    def clip(self, X, low=0.0005, high=0.9995):
        # works with tensors =)
        return np.clip(X, np.quantile(X, low), np.quantile(X, high))

    def __len__(self):
        return len(self.grid)

    def create_Vmask(self):
        "Create volumetric blind spot random mask"
        downsampled_shape = np.array(3 * [self.subtomo_length]) // self.vol_scale_factor
        downsampled_shape = tuple([int(x) for x in downsampled_shape])

        # avoid power correction from dropout and set shape for upsampling
        bernoulli_Vmask = self.dropoutV(torch.ones(downsampled_shape)) * (
            1 - self.Vmask_pct
        )
        bernoulli_Vmask = bernoulli_Vmask.unsqueeze(0).unsqueeze(0)
        # make final shape [C, S, S, S]
        bernoulli_Vmask = self.upsample(bernoulli_Vmask).squeeze(0)

        return bernoulli_Vmask

    def create_Pmask(self):
        "Create pointed blind spot random mask"
        _shape = 3 * [self.subtomo_length]
        bernoulli_Pmask = self.dropout(torch.ones(_shape)) * (1 - self.p)
        bernoulli_Pmask = bernoulli_Pmask.unsqueeze(0)

        return bernoulli_Pmask

    def create_bernoulliMask(self):
        if np.random.uniform() < self.Vmask_probability:
            # might work as an augmentation technique.
            bernoulli_mask = self.create_Vmask()
        else:
            bernoulli_mask = self.create_Pmask()

        return bernoulli_mask

    def __getitem__(self, index: int):
        center_z, center_y, center_x = self.shift_coords(*self.grid[index])
        z_min, z_max = (
            center_z - self.subtomo_length // 2,
            center_z + self.subtomo_length // 2,
        )
        y_min, y_max = (
            center_y - self.subtomo_length // 2,
            center_y + self.subtomo_length // 2,
        )
        x_min, x_max = (
            center_x - self.subtomo_length // 2,
            center_x + self.subtomo_length // 2,
        )
        subtomo = self.data[z_min:z_max, y_min:y_max, x_min:x_max]

        if self.gt_data is not None:
            gt_subtomo = self.gt_data[z_min:z_max, y_min:y_max, x_min:x_max]
        else:
            gt_subtomo = None

        # first transform and then get samples
        if self.transform:
            subtomo, gt_subtomo = self.transform(subtomo, gt_subtomo)

        ##### One different mask per __getitem__ call
        bernoulli_mask = torch.stack(
            [self.create_bernoulliMask() for i in range(self.n_bernoulli_samples)],
            axis=0,
        )

        if gt_subtomo is not None:
            gt_subtomo = gt_subtomo.unsqueeze(0).repeat(
                self.n_bernoulli_samples, 1, 1, 1, 1
            )

        _samples = subtomo.unsqueeze(0).repeat(
            self.n_bernoulli_samples, 1, 1, 1, 1
        )  # get n samples
        bernoulli_subtomo = bernoulli_mask * _samples  # bernoulli samples
        target = (1 - bernoulli_mask) * _samples  # complement of the bernoulli sample

        return bernoulli_subtomo, target, bernoulli_mask, gt_subtomo

    def shift_coords(self, z, y, x):
        "Add random shift to coordinates"
        new_coords = []
        for idx, coord in enumerate([z, y, x]):
            shift_range = range(-self.n_shift, self.n_shift + 1)
            coord = coord + np.random.choice(shift_range)
            # Shift position if too close to border:
            if coord < self.subtomo_length // 2:
                coord = self.subtomo_length // 2
            if coord > self.tomo_shape[idx] - self.subtomo_length // 2:
                coord = self.tomo_shape[idx] - self.subtomo_length // 2
            new_coords.append(coord)

        return tuple(new_coords)

    def create_grid(self):
        """Create a possibly overlapping set of patches forming a grid that covers a tomogram"""
        dist_center = self.subtomo_length // 2  # size from center
        centers = []
        for i, coord in enumerate(self.tomo_shape):

            n_centers = int(np.ceil(coord / self.subtomo_length))
            _centers = np.linspace(
                dist_center, coord - dist_center, n_centers, dtype=int
            )

            startpoints, endpoints = _centers - dist_center, _centers + dist_center
            overlap_ratio = max(endpoints[:-1] - startpoints[1::]) / dist_center

            centers.append(_centers)

            if overlap_ratio < 0:
                raise ValueError(
                    "The tomogram is not fully covered in dimension %i." % i
                )

            # if overlap_ratio>0.5:
            #     raise ValueError('There is more than 50%% overlap between patches in dimension %i.' %i)

        zs, ys, xs = np.meshgrid(*centers, indexing="ij")
        grid = list(zip(zs.flatten(), ys.flatten(), xs.flatten()))

        return grid


class singleCET_FourierDataset(singleCET_dataset):
    def __init__(
        self,
        tomo_path,
        subtomo_length,
        p, # for the inverse mask
        n_bernoulli_samples=6,
        total_samples=100,
        volumetric_scale_factor=4,
        Vmask_probability=0, # deprecated
        Vmask_pct=0.1, # for the volumetric mask
        transform=None,
        n_shift=0,
        gt_tomo_path=None,
        input_as_target=False,
        bernoulliMask_prob=1,
        clip=True,
        path_to_fourier_samples=None,
        **deconv_kwargs
    ):
        """
        Load cryoET dataset with samples taken by Bernoulli sampling Fourier space for self2self denoising.

        The dataset consists of subtomograms of shape [M, C, S, S, S] C (equal to 1) is the number of channels and
        S is the subtomogram side length.

        - tomo_path: tomogram path
        - subtomo_length: side length of the patches to be used for training
        - p: probability of an element to be zeroed
        - volumetric_scale_factor: times the original tomogram shape will be reduced
        to take bernoulli point samples before upsampling into volumetric bernoulli blind spots.
        """
        singleCET_dataset.__init__(
            self,
            tomo_path=tomo_path,
            subtomo_length=subtomo_length,
            p=p, # Pmask probability
            n_bernoulli_samples=n_bernoulli_samples,
            volumetric_scale_factor=volumetric_scale_factor,
            Vmask_probability=Vmask_probability, # not used in this case
            Vmask_pct=Vmask_pct, # Vmask probability
            transform=transform,
            n_shift=n_shift,
            gt_tomo_path=gt_tomo_path,
            clip=clip,
            **deconv_kwargs
        )

        self.total_samples = total_samples
        self.dataF = torch.fft.rfftn(self.data)
        self.tomoF_shape = self.dataF.shape

        # here we only create one set of M bernoulli masks to be sampled from
        self.input_as_target = input_as_target
        self.bernoulliMask_prob = bernoulliMask_prob

        # I thought predicting on the same fourier samples as the ones used for training would
        # help improve performance. But it doesn't seem to be the case. It might speed up
        # training a little though by reducing time to create samples.
        if path_to_fourier_samples is not None:
            print('Found existing samples. Loading samples...')
            self.fourier_samples = torch.load(path_to_fourier_samples)
            print('Done!!')
        else:
            self.fourier_samples = self.create_FourierSamples()

        return

    def _make_shell(self, inner_radius, outer_radius, tomo_shape):
        """
        Creates a (3D) shell with given inner_radius and outer_radius centered at the middle of the array.
        """

        length = min(tomo_shape)
        if length % 2 == 1:
            length = length - 1

        mask_shape = len(tomo_shape) * [length]
        _shell_mask = np.zeros(mask_shape)

        # only do positive quadrant first
        for z in range(0, outer_radius + 1):
            for y in range(0, outer_radius + 1):
                for x in range(0, outer_radius + 1):

                    r = np.linalg.norm([z, y, x])

                    if r >= inner_radius and r < outer_radius:
                        zidx = z + length // 2
                        yidx = y + length // 2
                        xidx = x + length // 2

                        _shell_mask[zidx, yidx, xidx] = 1

        aux = (
            np.rot90(_shell_mask, axes=(0, 1))
            + np.rot90(_shell_mask, 2, axes=(0, 1))
            + np.rot90(_shell_mask, 3, axes=(0, 1))
        )

        _shell_mask = _shell_mask + aux  # this is half the volume

        aux = np.rot90(
            _shell_mask, 2, axes=(1, 2)
        )  # rotate again 180º to get full volume
        
        _shell_mask += aux
        
        return _shell_mask

    def make_shell(self, inner_radius, outer_radius, tomo_shape):
        """
        Creates a (3D) shell with given inner_radius and delta_r width centered at the middle of the array.

        """
        length = min(tomo_shape)
        if length % 2 == 1:
            length = length - 1

        _shell_mask = self._make_shell(inner_radius, outer_radius, tomo_shape)

        if inner_radius == 0:
            vol = 4 / 3 * np.pi * outer_radius**3
            pct_diff = (vol - _shell_mask.sum()) / vol
            if pct_diff > 0.1:
                print(pct_diff)
                raise ValueError("Sanity check for sphere volume not passed")

        # finally, fill the actual shape of the tomogram with the mask
        shell_mask = np.zeros(tomo_shape)
        shell_mask[
            (tomo_shape[0] - length) // 2 : (tomo_shape[0] + length) // 2,
            (tomo_shape[1] - length) // 2 : (tomo_shape[1] + length) // 2,
            (tomo_shape[2] - length) // 2 : (tomo_shape[2] + length) // 2,
        ] = _shell_mask

        return shell_mask

    def make_shell2(self, inner_radius, outer_radius, tomo_shape, factor):
        """
        Creates a (3D) shell with given inner_radius and delta_r width centered at the middle of the array.

        """
        
        length = min(tomo_shape)
        if length % 2 == 1:
            length = length - 1
            
        if 2*outer_radius>length:
            raise ValueError('Cannot fit a bigger sphere than the smallest tomogram length.')
        
        if factor % 2 == 1:
            raise ValueError('factor values must be divisible by 2')
            
        upsample = torch.nn.Upsample(scale_factor=factor)
        
        tomo_shape_down = np.array(tomo_shape)//factor
        outer_radius_down = int(np.round(outer_radius/factor))
        inner_radius_down = int(np.round(inner_radius/factor))

        _shell_mask = self._make_shell(inner_radius_down, outer_radius_down, tomo_shape_down)
        
        _shell_mask = torch.tensor(_shell_mask).unsqueeze(0).unsqueeze(0)
        _shell_mask = upsample(_shell_mask).squeeze().numpy()

        if inner_radius == 0:
            vol = 4 / 3 * np.pi * outer_radius**3
            pct_diff = (vol - _shell_mask.sum()) / vol
            if pct_diff > 0.1:
                print(pct_diff)
                raise ValueError("Percentual difference between the created sphere and the actual volume of a sphere of the given radius is bigger than 0.1")
        
        # finally, pad shape to correspond the original shape
        shape_diff = np.array(tomo_shape) - np.array(_shell_mask.shape)
        shape_diff = shape_diff//2 + 1
        _shell_mask = np.pad(_shell_mask, [(shape_diff[0], ), (shape_diff[1], ), (shape_diff[2], )])

        shell_mask =  _shell_mask[0:tomo_shape[0], 0:tomo_shape[1], 0:tomo_shape[2]]
        
        return shell_mask

    def create_Vmask(self):
        "Create volumetric blind spot random mask"
        downsampled_shape = np.array(self.tomoF_shape) // self.vol_scale_factor
        downsampled_shape = tuple(downsampled_shape)

        bernoulli_Vmask = self.dropoutV(torch.ones(downsampled_shape)) * (
            1 - self.dropoutV.p
        )
        bernoulli_Vmask = bernoulli_Vmask.unsqueeze(0).unsqueeze(0)
        bernoulli_Vmask = self.upsample(bernoulli_Vmask)
        extra_row = bernoulli_Vmask[..., -1].unsqueeze(-1)
        # make final shape [C, S, S, S] last row is to account for Nyquist Frequency
        bernoulli_Vmask = torch.cat([bernoulli_Vmask, extra_row], dim=-1).squeeze(0)

        # adjust for uneven sizes on dimension 0
        diff0 = self.tomoF_shape[0] - bernoulli_Vmask.shape[1]
        if (diff0 > 0) and (diff0 < self.vol_scale_factor):
            extra_row = bernoulli_Vmask[:, 0:diff0, ...]
            bernoulli_Vmask = torch.cat([extra_row, bernoulli_Vmask], dim=1)

        if bernoulli_Vmask[0, ...].shape != self.dataF.shape:
            raise ValueError(
                "Volumetric mask with shape %s has a different shape in the last three components as dataF with shape %s"
                % (str(bernoulli_Vmask.shape), str(self.dataF.shape))
            )

        return bernoulli_Vmask

    def create_hiFreqMask(self):
        "Randomly mask high frequencies with a sphere"
        inner = 0
        shape_vol = np.array(self.tomo_shape).prod()
        low_r = (0.05 * 3/(4*np.pi) * shape_vol)**(1/3)
        high_r = (0.1 * 3/(4*np.pi) * shape_vol)**(1/3)
        outer = np.random.uniform(low_r, high_r)
        outer = int(np.round(outer))
        
        if min(self.tomo_shape) > 400:
            # print("Using 2x downsampled shape to create spheres")
            shell_mask = self.make_shell2(inner, outer, self.tomo_shape, factor=2)
        else:
            shell_mask = self.make_shell(inner, outer, self.tomo_shape)

        shell_mask = torch.tensor(shell_mask)
        # make shell correspond to the unshifted spectrum
        shell_mask = torch.fft.ifftshift(shell_mask)
        # make it correspond to only real part of spectrum
        shell_mask = shell_mask[..., 0 : self.tomoF_shape[-1]]

        return shell_mask.float().unsqueeze(0)

    def create_Pmask(self):
        "Create pointed blind spot random mask"
        _shape = self.tomoF_shape
        # we allow power correction here: not multiplying by (1-p)
        bernoulli_Pmask = self.dropout(torch.ones(_shape)) * (1 - self.dropout.p)
        bernoulli_Pmask = bernoulli_Pmask.unsqueeze(0)

        return bernoulli_Pmask

    def create_mask(self):
        "Create a mask choosing between Bernoulli and other type. Could be volumetric or highFreq."
        # Best mask so far.
        mask = self.create_hiFreqMask() + self.create_Vmask()
        mask = torch.where(mask > 1, 1, mask)
        invMask = self.create_Pmask()
        invMask = 2*invMask - 1

        mask = invMask*mask

        assert len(mask.unique()) == 3 # -1, 0 and 1

        return mask

    def create_batchFourierSamples(self, M):
        mask = Parallel(
                n_jobs=5
                )(delayed(self.create_mask)() for i in range(M))
        mask = torch.stack(mask, axis=0)

        fourier_samples = self.dataF.unsqueeze(0).repeat(M, 1, 1, 1, 1)
        fourier_samples = fourier_samples * mask
        samples = torch.fft.irfftn(fourier_samples, dim=[-3, -2, -1])

        return samples

    def create_FourierSamples(self):
        "Create a predefined set of fourier space samples that will be sampled from on each __getitem__ call"
        print("Creating Fourier samples...")
        s = 5
        n_times = self.total_samples // s
        n_times = max([1, n_times])
        
        samples = [self.create_batchFourierSamples(s) for i in tqdm(range(n_times))]
        
        samples = torch.cat(samples)

        print("Done! Using %i Fourier samples." %len(samples))

        return samples

    def __getitem__(self, index: int):
        center_z, center_y, center_x = self.shift_coords(*self.grid[index])
        z_min, z_max = (
            center_z - self.subtomo_length // 2,
            center_z + self.subtomo_length // 2,
        )
        y_min, y_max = (
            center_y - self.subtomo_length // 2,
            center_y + self.subtomo_length // 2,
        )
        x_min, x_max = (
            center_x - self.subtomo_length // 2,
            center_x + self.subtomo_length // 2,
        )

        if self.gt_data is not None:
            gt_subtomo = self.gt_data[z_min:z_max, y_min:y_max, x_min:x_max]
        else:
            gt_subtomo = None

        if self.input_as_target:
            sample_idx = np.random.choice(
                range(len(self.fourier_samples)),
                self.n_bernoulli_samples,
                replace=False,
            )
            subtomo = self.fourier_samples[sample_idx][
                ..., z_min:z_max, y_min:y_max, x_min:x_max
            ]
            # IMPORTANT! we are mapping samples to input
            target = self.data[z_min:z_max, y_min:y_max, x_min:x_max]
            target = target.unsqueeze(0).repeat(self.n_bernoulli_samples, 1, 1, 1, 1)
        else:
            sample_idx = np.random.choice(
                range(len(self.fourier_samples)),
                2 * self.n_bernoulli_samples,
                replace=False,
            )
            samples = self.fourier_samples[sample_idx][
                ..., z_min:z_max, y_min:y_max, x_min:x_max
            ]
            # IMPORTANT! we are mapping samples to samples
            subtomo, target = torch.split(samples, self.n_bernoulli_samples)

        if gt_subtomo is not None:
            gt_subtomo = gt_subtomo.unsqueeze(0).repeat(
                self.n_bernoulli_samples, 1, 1, 1, 1
            )

        if self.transform:
            subtomo, target, gt_subtomo = self.transform(subtomo, target, gt_subtomo)

        return subtomo, target, gt_subtomo


class singleCET_ProjectedDataset(Dataset):
    def __init__(
        self,
        tomo_path,
        subtomo_length,
        transform=None,
        n_shift=0,
        gt_tomo_path=None,
        predict_simRecon=False,
        use_deconv_as_target=False,
        **deconv_kwargs
    ):
        """
        Load cryoET dataset and simulate 2 independent projections for N2N denoising. All data can be optionally deconvolved.

        The dataset consists of subtomograms of shape [C, S, S, S] C (equal to 1) is the number of channels and
        S is the subtomogram side length.

        - tomo_path: tomogram path
        - subtomo_length: side length of the patches to be used for training
        - p: probability of an element to be zeroed
        - volumetric_scale_factor: times the original tomogram shape will be reduced
        to take bernoulli point samples before upsampling into volumetric bernoulli blind spots.
        """
        self.tomo_path = tomo_path
        self.data = torch.tensor(read_array(tomo_path))
        self.data = self.clip(self.data)
        self.data = self.standardize(self.data)

        self.use_deconv_data = self.check_deconv_kwargs(deconv_kwargs)

        self.gt_tomo_path = gt_tomo_path
        if gt_tomo_path is not None:
            self.gt_data = torch.tensor(read_array(gt_tomo_path))
            self.gt_data = self.clip(self.gt_data)
            self.gt_data = self.standardize(self.gt_data)
        else:
            self.gt_data = None

        self.tomo_shape = self.data.shape
        self.subtomo_length = subtomo_length
        self.n_shift = n_shift
        self.grid = self.create_grid()
        self.transform = transform
        self.n_angles = 300
        self.shift = np.pi / np.sqrt(
            2
        )  # shift by some amount that guarantees no overlap
        self.angles0 = np.linspace(0, 2 * np.pi, self.n_angles)
        # shift the projection for the second reconstruction
        self.angles1 = np.linspace(
            0 + self.shift, 2 * np.pi + self.shift, self.n_angles
        )

        self.simRecon0 = self.make_simulated_reconstruction(self.angles0, "fbp")
        self.simRecon0 = self.standardize(self.clip(self.simRecon0))
        self.deconv_kwargs0 = {"vol": self.simRecon0, **deconv_kwargs}
        self.simRecon0 = tom_deconv_tomo(**self.deconv_kwargs0)
        self.simRecon0 = torch.tensor(self.simRecon0)

        self.predict_simRecon = predict_simRecon
        if predict_simRecon:
            self.simRecon1 = self.make_simulated_reconstruction(self.angles1, "fbp")
            self.simRecon1 = self.standardize(self.clip(self.simRecon1))
            # I map deconvolved to raw reconstruction because my idea is that this
            # prevents too much coupling of the noise somehow (??)
            if use_deconv_as_target:
                print("Using simRecon0 and deconvolved simRecon1 for training")
                self.deconv_kwargs1 = {"vol": self.simRecon1, **deconv_kwargs}
                self.simRecon1 = tom_deconv_tomo(**self.deconv_kwargs1)
            else:
                print("Using simRecon0 and simRecon1 for training")
            self.simRecon1 = torch.tensor(self.simRecon1)

        else:
            if use_deconv_as_target:
                print("Using simRecon0 and deconvolved data for training")
                self.deconv_kwargs = {"vol": self.data.numpy(), **deconv_kwargs}
                self.data = tom_deconv_tomo(**self.deconv_kwargs)
                self.data = torch.tensor(self.data)
            else:
                print("Using simRecon0 and data for training")

        self.use_deconv_as_target = use_deconv_as_target

        self.run_init_asserts()

        return

    def check_deconv_kwargs(self, deconv_kwargs):
        if bool(deconv_kwargs):
            deconv_args = [
                "angpix",
                "defocus",
                "snrfalloff",
                "deconvstrength",
                "highpassnyquist",
            ]
            for arg in deconv_args:
                if arg in deconv_kwargs.keys():
                    continue
                else:
                    raise KeyError(
                        'Missing required deconvolution argument: "%s"' % arg
                    )
            use_deconv_data = True

        else:
            use_deconv_data = False

        return use_deconv_data

    def run_init_asserts(self):
        if self.subtomo_length % 32 != 0:
            raise ValueError(
                "Length of subtomograms must be a multiple of 32 to run the network."
            )

        return

    def standardize(self, X: torch.tensor):
        mean = X.mean()
        std = X.std()

        new_X = (X - mean) / std

        return new_X

    def clip(self, X, low=0.0005, high=0.9995):
        # works with tensors =)
        return np.clip(X, np.quantile(X, low), np.quantile(X, high))

    def __len__(self):
        return len(self.grid)

    def shift_coords(self, z, y, x):
        "Add random shift to coordinates"
        new_coords = []
        for idx, coord in enumerate([z, y, x]):
            shift_range = range(-self.n_shift, self.n_shift + 1)
            coord = coord + np.random.choice(shift_range)
            # Shift position if too close to border:
            if coord < self.subtomo_length // 2:
                coord = self.subtomo_length // 2
            if coord > self.tomo_shape[idx] - self.subtomo_length // 2:
                coord = self.tomo_shape[idx] - self.subtomo_length // 2
            new_coords.append(coord)

        return tuple(new_coords)

    def create_grid(self):
        """Create a possibly overlapping set of patches forming a grid that covers a tomogram"""
        dist_center = self.subtomo_length // 2  # size from center
        centers = []
        for i, coord in enumerate(self.tomo_shape):

            n_centers = int(np.ceil(coord / self.subtomo_length))
            _centers = np.linspace(
                dist_center, coord - dist_center, n_centers, dtype=int
            )

            startpoints, endpoints = _centers - dist_center, _centers + dist_center
            overlap_ratio = max(endpoints[:-1] - startpoints[1::]) / dist_center

            centers.append(_centers)

            if overlap_ratio < 0:
                raise ValueError(
                    "The tomogram is not fully covered in dimension %i." % i
                )

            # if overlap_ratio>0.5:
            #     raise ValueError('There is more than 50%% overlap between patches in dimension %i.' %i)

        zs, ys, xs = np.meshgrid(*centers, indexing="ij")
        grid = list(zip(zs.flatten(), ys.flatten(), xs.flatten()))

        return grid

    def make_simulated_reconstruction(self, angles, algorithm):
        projection = proj.project(self.data, angles)
        reconstruction = recon(projection, angles, algorithm=algorithm)

        _shape = np.array(reconstruction.shape)
        s0 = (_shape - self.tomo_shape) // 2
        s1 = _shape - s0

        reconstruction = reconstruction[s0[0] : s1[0], s0[1] : s1[1], s0[2] : s1[2]]

        return reconstruction

    def __getitem__(self, index: int):
        center_z, center_y, center_x = self.shift_coords(*self.grid[index])

        z_min, z_max = (
            center_z - self.subtomo_length // 2,
            center_z + self.subtomo_length // 2,
        )
        y_min, y_max = (
            center_y - self.subtomo_length // 2,
            center_y + self.subtomo_length // 2,
        )
        x_min, x_max = (
            center_x - self.subtomo_length // 2,
            center_x + self.subtomo_length // 2,
        )

        if self.gt_data is not None:
            gt_subtomo = self.gt_data[z_min:z_max, y_min:y_max, x_min:x_max]
            gt_subtomo = torch.tensor(gt_subtomo).unsqueeze(0)
        else:
            gt_subtomo = None

        subtomo = self.simRecon0[z_min:z_max, y_min:y_max, x_min:x_max]
        subtomo = torch.tensor(subtomo).unsqueeze(0)
        if self.predict_simRecon:
            target = self.simRecon1[z_min:z_max, y_min:y_max, x_min:x_max]
            target = torch.tensor(target).unsqueeze(0)
        else:
            target = self.data[z_min:z_max, y_min:y_max, x_min:x_max]
            target = torch.tensor(target).unsqueeze(0)

        if self.transform is not None:
            subtomo, target, gt_subtomo = self.transform(subtomo, target, gt_subtomo)

        return subtomo, target, gt_subtomo


class randomRotation3D(object):
    def __init__(self, p):
        assert p >= 0 and p <= 1
        self.p = p

    def __call__(self, subtomo, gt_subtomo):
        "Input is a 3D ZYX (sub)tomogram"
        # 180º rotation around Y axis
        if np.random.uniform() < self.p:
            subtomo = torch.rot90(subtomo, k=2, dims=(0, 2))
            if gt_subtomo is not None:
                gt_subtomo = torch.rot90(gt_subtomo, k=2, dims=(0, 2))
        # 180º rotation around X axis
        if np.random.uniform() < self.p:
            subtomo = torch.rot90(subtomo, k=2, dims=(0, 1))
            if gt_subtomo is not None:
                gt_subtomo = torch.rot90(gt_subtomo, k=2, dims=(0, 1))
        # rotation between 90º and 270º around Z axis
        if np.random.uniform() < self.p:
            k = int(np.random.choice([1, 2, 3]))
            subtomo = torch.rot90(subtomo, k=k, dims=(1, 2))
            if gt_subtomo is not None:
                gt_subtomo = torch.rot90(gt_subtomo, k=k, dims=(1, 2))

        return subtomo, gt_subtomo

    def __repr__(self):
        return repr("randomRotation3D with probability %.02f" % self.p)


class randomRotation3D_fourierSamples(object):
    def __init__(self, p):
        assert p >= 0 and p <= 1
        self.p = p

    def make3D_rotation(self, subtomo, target, gt_subtomo):
        "3D rotation in ZYX sets of images."
        # 180º rotation around Y axis
        if np.random.uniform() < self.p:
            subtomo = torch.rot90(subtomo, k=2, dims=(0, 2))
            target = torch.rot90(target, k=2, dims=(0, 2))
            gt_subtomo = torch.rot90(gt_subtomo, k=2, dims=(0, 2))
        # 180º rotation around X axis
        if np.random.uniform() < self.p:
            subtomo = torch.rot90(subtomo, k=2, dims=(0, 1))
            target = torch.rot90(target, k=2, dims=(0, 1))
            gt_subtomo = torch.rot90(gt_subtomo, k=2, dims=(0, 1))
        # rotation between 90º and 270º around Z axis
        if np.random.uniform() < self.p:
            k = int(np.random.choice([1, 2, 3]))
            subtomo = torch.rot90(subtomo, k=k, dims=(1, 2))
            target = torch.rot90(target, k=k, dims=(1, 2))
            gt_subtomo = torch.rot90(gt_subtomo, k=k, dims=(1, 2))

        return subtomo, target, gt_subtomo

    def __call__(self, subtomo, target, gt_subtomo):
        """
        Input are of shape [M, C, S, S, S]
        First flatten the arrays, then apply the rotations on the 4D arrays, then reshape to original shape.
        """

        s, t = subtomo.flatten(start_dim=0, end_dim=1), target.flatten(
            start_dim=0, end_dim=1
        )
        if gt_subtomo is not None:
            g = gt_subtomo.flatten(start_dim=0, end_dim=1)
        else:
            g = torch.zeros_like(s)

        subtomo_rotated, target_rotated, gt_subtomo_rotated = [], [], []

        for values in zip(s, t, g):
            a, b, c = self.make3D_rotation(*values)
            subtomo_rotated.append(a)
            target_rotated.append(b)
            gt_subtomo_rotated.append(c)

        subtomo_rotated = torch.stack(subtomo_rotated).reshape(subtomo.shape)
        target_rotated = torch.stack(target_rotated).reshape(target.shape)
        gt_subtomo_rotated = torch.stack(gt_subtomo_rotated).reshape(subtomo.shape)

        # deal with no gt_subtomo case. Maybe not the best, since we calculate 1 too many rotations
        if (gt_subtomo_rotated == 0).all():
            gt_subtomo_rotated = None

        return subtomo_rotated, target_rotated, gt_subtomo_rotated

    def __repr__(self):
        return repr("randomRotation3D with probability %.02f" % self.p)
