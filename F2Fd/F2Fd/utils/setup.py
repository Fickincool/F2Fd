from cProfile import label
from pathlib import Path
from pickle import NONE
import mrcfile
import numpy as np
import sys
import os
from glob import glob

# Local
# PARENT_PATH = '/home/jeronimo/Desktop/Master_mathematics/Thesis/'
# HPC
PARENT_PATH = "/home/ubuntu/Thesis/"

# Previous training data
# TOMO_PATH = PARENT_PATH+'data/processed0/nnUnet/cET_cropped/'
# Isensee training data
TOMO_PATH = PARENT_PATH + "data/nnUnet/Task143_cryoET7/"

CRYOCARE_TOMO_PATH = PARENT_PATH + "data/nnUnet/Task143_cryoET7/imagesTr"
RAW_TOMO_PATH = (
    PARENT_PATH + "data/raw_cryo-ET/patch_creation/result/Task511_cryoET/imagesTr"
)
ISONET_TOMO_PATH = (
    PARENT_PATH
    + "data/isoNet/RAW_dataset/RAW_corrected/patch_creation/result/Task511_cryoET/imagesTr"
)
CRYOCARE_ISONET_TOMO_PATH = (
    PARENT_PATH
    + "data/isoNet/cryoCARE_dataset/cryoCARE_corrected/patch_creation/result/Task511_cryoET/imagesTr"
)

DECONV_TOMO_PATH = (
    PARENT_PATH
    + "data/isoNet/RAW_dataset/RAW_allTomos_deconv/patch_creation/result/Task511_cryoET/imagesTr"
)

N2V_TOMO_PATH = (
    PARENT_PATH
    + "data/S2SDenoising/n2v_model_logs/patch_creation/result/Task511_cryoET/imagesTr"
)

F2Fd_TOMO_PATH = (
    PARENT_PATH
    + "data/S2SDenoising/F2FDenoised/patch_creation/result/Task511_cryoET/imagesTr"
)

S2Sd_TOMO_PATH = (
    PARENT_PATH
    + "data/S2SDenoising/S2SDenoised/patch_creation/result/Task511_cryoET/imagesTr"
)


LABEL_PATH = PARENT_PATH + "data/nnUnet/Task143_cryoET7/labelsTr"


def get_tomo_indices():

    files = [f for f in Path(TOMO_PATH).glob("*tomo*")]

    # Naming convention: tomo10_bin4_denoised_0000_0-309_618-928_100-350.mrc

    # assert 1: all files start with tomo
    if len([f for f in Path(TOMO_PATH).glob("*")]) == len(files):
        pass
    else:
        print("Number of files: ", len([f for f in Path(TOMO_PATH).glob("*")]))
        print("Number of files starting with tomo: ", len(files))
        raise AssertionError

    tomo_ids, counts = np.unique(
        [f.as_posix().split("/")[-1].split("_")[0] for f in files], return_counts=True
    )

    # assert 2: all tomograms have the same number of files (including labels)
    if len(np.unique(counts)) == 1:
        pass
    else:
        print(tomo_ids, counts)
        raise AssertionError

    label_files = [
        f.as_posix() for f in Path(TOMO_PATH).glob("*tomo*") if "lbl" in f.as_posix()
    ]
    label_files = sorted(label_files)

    tomo_ids = sorted(tomo_ids)
    tomo_idx = list(range(len(tomo_ids)))
    print("Pairs of tomo IDs to indices:")
    print(list(zip(tomo_ids, tomo_idx)))

    return tomo_ids, tomo_idx


def get_paths_cryoCARE(tomo_names: list):
    """
    Get paths for data and target tomograms processed using Cryo-CARE.
    """

    path_data = []
    path_target = []
    for tomo_name in tomo_names:

        path_data += glob(os.path.join(CRYOCARE_TOMO_PATH, "%s*" % tomo_name))
        path_target += glob(os.path.join(LABEL_PATH, "%s*" % tomo_name))

    path_data, path_target = sorted(path_data), sorted(path_target)

    names_data = [x.split("/")[-1][0:15] for x in path_data]
    names_target = [x.split("/")[-1][0:15] for x in path_target]

    assert names_data == names_target  # check that tomograms correspond

    return path_data, path_target


def get_paths_rawCET(tomo_names: list):
    """
    Get paths for data and target tomograms processed using raw_tomograms.
    """

    path_data = []
    path_target = []
    for tomo_name in tomo_names:
        path_target += glob(os.path.join(LABEL_PATH, "%s*" % tomo_name))

    for x in path_target:
        patch_name = x.split("/")[-1][0:15]
        path_data += glob(os.path.join(RAW_TOMO_PATH, "%s*" % patch_name))

    path_data, path_target = sorted(path_data), sorted(path_target)

    names_data = [x.split("/")[-1][0:15] for x in path_data]
    names_target = [x.split("/")[-1][0:15] for x in path_target]

    assert names_data == names_target  # check that tomograms correspond

    return path_data, path_target


def get_paths_cryoCARE_isoNET(tomo_names: list):
    """
    Get paths for data and target tomograms processed using Isonet applied to Cryo-CARE.
    """

    path_data = []
    path_target = []
    for tomo_name in tomo_names:
        path_target += glob(os.path.join(LABEL_PATH, "%s*" % tomo_name))

    for x in path_target:
        patch_name = x.split("/")[-1][0:15]
        path_data += glob(os.path.join(CRYOCARE_ISONET_TOMO_PATH, "%s*" % patch_name))

    path_data, path_target = sorted(path_data), sorted(path_target)

    names_data = [x.split("/")[-1][0:15] for x in path_data]
    names_target = [x.split("/")[-1][0:15] for x in path_target]

    assert names_data == names_target  # check that tomograms correspond

    return path_data, path_target


def get_paths_isoNET(tomo_names: list):
    """
    Get paths for data and target tomograms processed using Isonet.
    """

    path_data = []
    path_target = []
    for tomo_name in tomo_names:
        path_target += glob(os.path.join(LABEL_PATH, "%s*" % tomo_name))

    for x in path_target:
        patch_name = x.split("/")[-1][0:15]
        path_data += glob(os.path.join(ISONET_TOMO_PATH, "%s*" % patch_name))

    path_data, path_target = sorted(path_data), sorted(path_target)

    names_data = [x.split("/")[-1][0:15] for x in path_data]
    names_target = [x.split("/")[-1][0:15] for x in path_target]

    assert names_data == names_target  # check that tomograms correspond

    return path_data, path_target

def get_paths_F2Fd(tomo_names):
    """
    Get paths for data and target tomograms denoised using our method.
    """

    path_data = []
    path_target = []
    for tomo_name in tomo_names:
        path_target += glob(os.path.join(LABEL_PATH, "%s*" % tomo_name))

    for x in path_target:
        patch_name = x.split("/")[-1][0:15]
        path_data += glob(os.path.join(F2Fd_TOMO_PATH, "%s*" % patch_name))

    path_data, path_target = sorted(path_data), sorted(path_target)

    names_data = [x.split("/")[-1][0:15] for x in path_data]
    names_target = [x.split("/")[-1][0:15] for x in path_target]

    assert names_data == names_target  # check that tomograms correspond

    return path_data, path_target

def get_paths_S2Sd(tomo_names):
    """
    Get paths for data and target tomograms denoised using our method.
    """

    path_data = []
    path_target = []
    for tomo_name in tomo_names:
        path_target += glob(os.path.join(LABEL_PATH, "%s*" % tomo_name))

    for x in path_target:
        patch_name = x.split("/")[-1][0:15]
        path_data += glob(os.path.join(S2Sd_TOMO_PATH, "%s*" % patch_name))

    path_data, path_target = sorted(path_data), sorted(path_target)

    names_data = [x.split("/")[-1][0:15] for x in path_data]
    names_target = [x.split("/")[-1][0:15] for x in path_target]

    assert names_data == names_target  # check that tomograms correspond

    return path_data, path_target


def get_paths_deconv(tomo_names):
    """
    Get paths for data and target tomograms denoised using our method.
    """

    path_data = []
    path_target = []
    for tomo_name in tomo_names:
        path_target += glob(os.path.join(LABEL_PATH, "%s*" % tomo_name))

    for x in path_target:
        patch_name = x.split("/")[-1][0:15]
        path_data += glob(os.path.join(DECONV_TOMO_PATH, "%s*" % patch_name))

    path_data, path_target = sorted(path_data), sorted(path_target)

    names_data = [x.split("/")[-1][0:15] for x in path_data]
    names_target = [x.split("/")[-1][0:15] for x in path_target]

    assert names_data == names_target  # check that tomograms correspond

    return path_data, path_target
    
def get_paths_N2V(tomo_names):
    """
    Get paths for data and target tomograms denoised using our method.
    """

    path_data = []
    path_target = []
    for tomo_name in tomo_names:
        path_target += glob(os.path.join(LABEL_PATH, "%s*" % tomo_name))

    for x in path_target:
        patch_name = x.split("/")[-1][0:15]
        path_data += glob(os.path.join(N2V_TOMO_PATH, "%s*" % patch_name))

    path_data, path_target = sorted(path_data), sorted(path_target)

    names_data = [x.split("/")[-1][0:15] for x in path_data]
    names_target = [x.split("/")[-1][0:15] for x in path_target]

    assert names_data == names_target  # check that tomograms correspond

    return path_data, path_target


def get_paths(tomo_names: list, input_type: str):

    assert type(tomo_names) == list

    if input_type == "cryoCARE":
        path_data, path_target = get_paths_cryoCARE(tomo_names)
    elif input_type == "rawCET":
        path_data, path_target = get_paths_rawCET(tomo_names)
    elif input_type == "cryoCARE+isoNET":
        path_data, path_target = get_paths_cryoCARE_isoNET(tomo_names)
    elif input_type == "isoNET":
        path_data, path_target = get_paths_isoNET(tomo_names)
    elif input_type == "F2Fd":
        path_data, path_target = get_paths_F2Fd(tomo_names)
    elif input_type == "S2Sd":
        path_data, path_target = get_paths_S2Sd(tomo_names)
    elif input_type == "N2V":
        path_data, path_target = get_paths_N2V(tomo_names)
    elif input_type == "Deconv":
        path_data, path_target = get_paths_deconv(tomo_names)
    else:
        raise NotImplementedError(
            'Only "isoNET", "cryoCARE+isoNET", "cryoCARE", "F2Fd", "S2Sd", "N2V", "Deconv" and "rawCET" input type is implemented'
        )

    return path_data, path_target
