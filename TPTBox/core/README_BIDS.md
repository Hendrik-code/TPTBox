# TPTBox `bids_files.py` — BIDS dataset navigation

| Symbol | Description |
|---|---|
| `BIDS_Global_info` | Scans a dataset root and indexes all BIDS files |
| `BIDS_Global_info.enumerate_subjects()` | Iterate over subjects as `(subject_id, Subject_Container)` |
| `Subject_Container` | Per-subject file index; entry point for queries |
| `Subject_Container.new_query()` | Returns a `Searchquery` for this subject |
| `BIDS_FILE` | One file parsed into BIDS entities (sub, ses, format, …) |
| `BIDS_FILE.open_nii()` | Load this file's NIfTI |
| `BIDS_FILE.get_changed_path(...)` | Derive a new path with changed BIDS entities |
| `Searchquery` | Fluent query builder: `.filter()`, `.loop_dict()`, `.first()` |
| `BIDS_Family` | `dict[str, list[BIDS_FILE]]` grouping files by format |


Loop over every T2w MRI in a dataset:
```python
from TPTBox import BIDS_Global_info, BIDS_FILE, NII

# Initialize the dataset and the folders therein to use
bids_dataset = BIDS_Global_info(["path/to/dataset"], parents=["rawdata"])

# looping over every subject in the dataset
for subject, container in bids.enumerate_subjects():
    q = container.new_query()
    q.filter("format", "T2w")
    # more filter here
    bids_families = q.loop_dict()
    # A subject can have multiple MRI images
    for bids_family in bids_families:
        # ensure this family has a T2w
        if "T2w" in bids_family:
            # get the reference to the T2w image
            t2w_ref: BIDS_FILE = bids_family["T2w"][0]
            # load the nifty
            t2w: NII = t2w.open_nii()

            # further processing or analysis that would be
            # run on every T2w MRI in this dataset
```

Investigate one BIDS_FILE and get a BIDS-compliant file path relative to it, guaranteeing a valid BIDS filename.
```python
from pathlib import Path

from TPTBox import BIDS_FILE

# Dataset root directory (must start with "dataset-")
root = Path("path/to/dataset-dsname")

# Example BIDS-compliant input file
example_file = (
    root
    / "rawdata/sub-Max-Mustermann/ses-01012026/anat/"
      "sub-Max-Mustermann_ses-01012026_acq-sag_ce-GBCA_T1w.nii.gz"
)

# Create a BIDS_FILE object
bf_file = BIDS_FILE(example_file, root)

# Access individual BIDS keys
print(f"Subject name           : {bf_file.get('sub')}")
print(f"Session                : {bf_file.get('ses')}")
print(f"Acquisition direction  : {bf_file.get('acq')}")
print(f"Contrast agent         : {bf_file.get('ce')}")
print(f"Modality               : {bf_file.bids_format}")

# Generate a new BIDS-compliant file path relative to the source file
# (keys that are not explicitly overridden remain unchanged)
seg_path = bf_file.get_changed_path(
    # File extension
    file_type="nii.gz",
    # Final suffix without a key, e.g., *_msk.nii.gz
    bids_format="msk",
    # Parent folder relative to the dataset root
    parent="derivatives",
    info={
        # Name of the segmentation
        "seg": "spine",
        # Modality from which this file was generated
        "mod": bf_file.mod,
    },
    # If True, disables sorting of keys according to the BIDS specification
    no_sorting_mode=False,
    # If True, disables strict validation against predefined key--value pairs
    non_strict_mode=False,
)

```
