from __future__ import annotations

from TPTBox import BIDS_Global_info, Subject_Container

path = "/media/data/robert/datasets/spinedata_temp"
global_info = BIDS_Global_info(
    [str(path)],
    ["rawdata", "rawdata_ct", "rawdata_dixon", "derivatives", "derivatives_spinalcord"],
    additional_key=["sequ", "seg", "ovl", "e"],
)
counts_spinalcord = 0
counts_sc_label = 0
counts_dixon = 0
for _, subject in global_info.enumerate_subjects():
    query = subject.new_query(flatten=True)
    query.filter("label", "spinalcord")
    counts_spinalcord += len(list(query.loop_list()))

    query = subject.new_query(flatten=True)
    query.filter("label", "spinalcordlabel")
    counts_dixon += len(list(query.loop_list()))

    query = subject.new_query(flatten=True)
    # It must exist a dixon
    query.filter("format", "dixon")
    # A nii.gz must exist
    query.filter("Filetype", "nii.gz")
    # Compute what dixon has a real-part image (T2 like)
    query.action(
        # Set Part key to real. Will be called if the filter = True
        action_fun=lambda x: x.set("part", "real"),
        # x is the json, because of the key="json". We return True if the json confirms that this is a real-part image
        filter_fun=lambda x: "IP" in x["ImageType"],  # type: ignore
        key="json",
        # The json is required
        required=True,
    )
    query.filter("part", "real")
    counts_sc_label += len(list(query.loop_list()))

print(counts_dixon, counts_spinalcord, counts_sc_label)

counts_sc_label = 0
for _, subject in global_info.enumerate_subjects(sort=True):
    subject: Subject_Container
    query = subject.new_query(flatten=False)
    query.filter_non_existence("label", "spinalcord")
    query.flatten()
    # query = subject.new_query(flatten=True)
    # It must exist a dixon
    query.filter("format", "dixon")
    # A nii.gz must exist
    query.filter("Filetype", "nii.gz")
    # Compute what dixon has a real-part image (T2 like)
    query.filter("json", lambda x: "IP" in x["ImageType"])  # type: ignore
    for bids_file in query.loop_list():
        print(bids_file)
        counts_sc_label += 1

print(counts_sc_label)
