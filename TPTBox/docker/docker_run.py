#!/usr/bin/python
from __future__ import annotations

from TPTBox import *
from TPTBox.docker.docker import run_docker_on_sample_advanced, run_docker_on_ds_advanced


if __name__ == "__main__":
    import configargparse as argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-in_path", type=str, default="", help="input path to the dataset")
    parser.add_argument("-use_centroids", action="store_true", help="Docker uses centroids as input")
    parser.add_argument("-use_mask", action="store_true", help="Docker uses vert_mask as input")
    parser.add_argument("-save_subreg_msk", action="store_true", help="Saves the docker subreg mask")
    # parser.add_argument("-test_one", action="store_true", help="Runs only on the first dataset sample")
    parser.add_argument("-verbose", action="store_true")
    opt = parser.parse_args()

    in_dir_ = opt.in_path
    print("dataset in", in_dir_)
    bids_ds = bids_files.BIDS_Global_info(
        datasets=[in_dir_], parents=["rawdata", "derivatives"], additional_key=["snp", "ovl"], verbose=False, clear=True
    )

    run_docker_on_ds_advanced(
        bids_ds=bids_ds,
        docker_input_ctd=opt.use_centroids,
        docker_input_vertmsk=opt.use_mask,
        cut_sacrum_ctd=True,
        save_subreg_mask=opt.save_subreg_msk,
        save_as_sourcedocker=False,
        save_log=True,
        logger=True,
    )
