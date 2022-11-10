import os

def download_SHREC(data_dir):
    "Download SHREC sample data. Only model 0 in this case"

    print('Downloading sample data... (takes some time)')
    if os.path.isfile('%s/shrec21_full_dataset.zip' %data_dir):
        pass
    else:     
        os.system('wget https://dataverse.nl/api/access/datafile/309088 -O %s/shrec21_full_dataset.zip' %data_dir)
    
    print('Unzipping model...')
    # unzip only the model we need, there are models 0 to 9
    os.system('unzip -n %s/shrec21_full_dataset.zip "model_0/*" -d %s' %(data_dir, data_dir))

    print('Done!\n')
    
    return


def parse_cet_paths(PARENT_PATH, tomo_name):
    
    if tomo_name.startswith("tomoPhantom"):

        cet_path = os.path.join(
            PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" % tomo_name
        )
        model_name = tomo_name.split("_")[1]

        gt_cet_path = os.path.join(
            PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomoPhantom_%s.mrc" % model_name
        )

    elif "s2sDenoised" in tomo_name:
        _name = tomo_name.split('_s2sDenoised')[0]
        _path = "data/S2SDenoising/model_logs/%s/fourierTripleMask_comparison/version_0/%s.mrc" 
        _path = _path %(_name, tomo_name)
        cet_path = os.path.join(PARENT_PATH, _path)
        __name = _name.split("_")[0]
        gt_cet_path = os.path.join(
                PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s_cryoCAREDummy.mrc" %__name
            )

    elif tomo_name.startswith("tomo"):
        if "dummy" in tomo_name:
            cet_path = os.path.join(
                PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" % tomo_name
            )
            _name = tomo_name.split("_")[0]
            gt_cet_path = os.path.join(
                PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s_cryoCAREDummy.mrc" % _name
            )
        else:
            cet_path = os.path.join(
                PARENT_PATH, "data/raw_cryo-ET/%s.mrc" % tomo_name
            )
            gt_cet_path = None

    elif tomo_name.startswith("shrec2021"):
        if "dummy" in tomo_name:
            cet_path = os.path.join(
                PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" % tomo_name
            )
            _name = tomo_name.split("_")[1]
            gt_cet_path = os.path.join(
                PARENT_PATH,
                "data/S2SDenoising/dummy_tomograms/shrec2021_%s_gtDummy.mrc" % _name,
            )
        else:
            _name = tomo_name.split('shrec2021_')[1]
            cet_path = os.path.join(
                PARENT_PATH, "data/shrec2021/%s/reconstruction.mrc" % _name
            )
            gt_cet_path = os.path.join(
                PARENT_PATH,
                "data/shrec2021/%s/grandmodel.mrc" % _name,
            )


    return cet_path, gt_cet_path