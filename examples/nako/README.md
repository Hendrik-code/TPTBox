# German National Cohort Dicom2Nii + Stitching example

This folder contains sample scrips to export the subset of NAKO files to a BIDS compliant filenames and a folder simplified folder structure.

We decided on a fixed naming schema for_
- 3D_GRE (vibe / " point dixon)
- ME_vibe (Multi echo Vibe)
- TSE SAG LWS/BWS/HWS (Sagittal spine images)
- PD FS SPC COR (proton density)
- T2 Haste composed 

We do not use "Sag T2 Spine" as it contains stitching from "TSE SAG" and the spine is deformed for better viewing but incorrect for evaluation. Use our Stitching pipeline instead. If you have other file types you have to add them in the script or the script will ask you what default naming schema you want to use.



Installing python 3.10 or higher:
```bash
pip install pydicom
pip install dicom2nifti
pip install func_timeout


pip install TPTBox
```

Making the data folder
```bash
dicom2nii_bids.py -i [Path to the dicom folder]
```
Than we provide scrips to stich T2w and Vibe images.
```bash
stitching_T2w.py -i [Path to the bids folder (dataset-nako)]
stitching_vibe.py -i [Path to the bids folder (dataset-nako)]
```
The stiched images are than under [dataset-nako]/rawdata_stiched
