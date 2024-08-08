# https://bids-specification.readthedocs.io/en/stable/02-common-principles.html
from __future__ import annotations

import json
import os
import sys
import typing
from collections.abc import Sequence
from pathlib import Path
from warnings import warn

import numpy as np

import TPTBox
from TPTBox.core.bids_constants import (
    entities,
    entities_keys,
    entity_alphanumeric,
    entity_decimal,
    entity_format,
    entity_left_right,
    entity_on_off,
    entity_parts,
    file_types,
    formats,
    formats_relaxed,
    sequence_naming_keys,
)

# ,

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))


### TODO Not implemented ###
# Subject/Session/Sequence LVL Meta data
# Recursive Meta data
# Smart rename/saving files
# Uniform Resource Indicator
# dataset_description.json (https://bids-specification.readthedocs.io/en/stable/03-modality-agnostic-files.html)
# Auto Metadata scanning (https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html)
# Typing by folder final folder
# If the session level is omitted in the folder structure, the filename MUST begin with the string sub-<label>, without ses-<label>


def validate_entities(key: str, value: str, name: str, verbose: bool):
    if not verbose:
        return True
    try:
        key = key.lower()
        if key not in entities_keys:
            print(
                f"[!] {key} is not in list of legal keys. This name '{name}' is invalid. Legal keys are: {list(entities_keys.keys())}. \nFor use see https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html"
            )
            entities_keys[key] = key
            return False
        if key in entity_alphanumeric and not value.isalnum():
            print(
                f"[!] value for {key} must be alphanumeric. This name '{name}' is invalid, with value {value}"
            )
            return False
        if key in entity_decimal and not value.isdecimal():
            print(
                f"[!] value for {key} must be decimal. This name '{name}' is invalid, with value {value}"
            )
            return False
        # if int(value) == 0:
        #    print(f"[!] value for {key} must be not 0. This name '{name}' is invalid, with value {value}")
        if key in entity_format and value not in formats_relaxed:
            print(
                f"[!] value for {key} must be a format. This name '{name}' is invalid, with value {value}"
            )
            return False
        if key in entity_on_off and value not in ["on", "off"]:
            print(
                f"[!] value for {key} must be in {['on', 'off']}. This name '{name}' is invalid, with value {value}"
            )
            return False
        if key in entity_left_right and value not in ["L", "R"]:
            print(
                f"[!] value for {key} must be in {['L', 'R']}. This name '{name}' is invalid, with value {value}"
            )
            return False
        parts = [
            "mag",
            "phase",
            "real",
            "imag",
            "inphase",
            "outphase",
            "fat",
            "water",
            "eco0-opp1",
            "eco0-opp1",
            "eco1-pip1",
            "eco2-opp2",
            "eco3-in1",
            "eco4-pop1",
            "eco5-arb1",
            "fat-outphase",
            "water-outphase",
            "water-fraction",
            "fat-fraction",
            "r2s",
        ]
        if key in entity_parts and value not in parts:
            print(
                f'[!] value for {key} must be in {parts}. This name "{name}" is invalid, with value {value}'
            )
            return False
        else:
            return True
    except Exception as e:
        print(e)
        return False


def get_values_from_name(
    path: Path | str, verbose
) -> tuple[str, dict[str, str], str, str]:
    name = Path(path).name
    bids_key, file_type = name.split(".", maxsplit=1)

    keys = bids_key.split("_")
    bids_format = keys[-1]
    if bids_format not in formats_relaxed and verbose:
        print(f"[!] Unknown format {bids_format} in file {name}", formats)
        formats_relaxed.append(bids_format)
    if file_type not in file_types and verbose:
        print(f"[!] Unknown file_type {file_type} in file {name}")

    dic = {}
    for idx, s in enumerate(keys[:-1]):
        try:
            key, value = s.split("-", maxsplit=1)
            if idx == 0 and key != "sub" and verbose:
                print(
                    f"[!] First key must be sub not {key}. This name '{name}' is invalid"
                )
            if idx != 1 and key == "ses" and verbose:
                print(f"[!] Session must be second key. This name '{name}' is invalid")

            if key in dic and verbose:
                print(
                    f"[!] {bids_key} contains copies of the same key twice. This name '{name}' is invalid"
                )

            validate_entities(key, value, name, verbose)
            dic[key] = value
        except Exception:
            if verbose:
                print(
                    f'[!] "{s}" is not a valid key/value pair. Expected "KEY-VALUE" in {name}'
                )
    return bids_format, dic, bids_key, file_type


class BIDS_Global_info:
    def __init__(
        self,
        datasets: Sequence[Path] | Sequence[str] | str | Path,
        parents: Sequence[str] | str = ["rawdata", "derivatives"],
        additional_key: Sequence[str] = ["sequ", "seg", "ovl"],
        verbose: bool = True,
        file_name_manipulation: typing.Callable[[str], str] | None = None,
        sequence_splitting_keys: list[str] | None = None,
        filter_folder: typing.Callable[[Path, int], bool] | None = None,
    ):
        """This Objects creates a datastructures reflecting BIDS-folders.

        Args:
            datasets (typing.List[str]): List of dataset paths
            parents (typing.List[str]): List of parents (like ["rawdata","sourcedata","derivatives"])
            additional_key (list, optional): Additional keys that are not in the default BIDS but should not raise a warning. Defaults to ["sequ", "seg", "ovl"].
            filter_folder: Filter function, input is the path of the folder and the level of the folder structure. Return True if we should continue searching
                        Example:
                        filter_folder = lambda p, lvl: True if (lvl != 2 or p.name in ["sub-123","sub-456"]) else False
        """
        self.count_file = 0
        if sequence_splitting_keys is None:
            from TPTBox.core.bids_constants import sequence_splitting_keys

            self.sequence_splitting_keys = sequence_splitting_keys

        self.sequence_splitting_keys = sequence_splitting_keys
        if isinstance(datasets, Path | str):
            datasets = [datasets]  # type: ignore
        if isinstance(parents, str):
            parents = [parents]
        assert isinstance(datasets, Sequence), "datasets is not a list"
        assert isinstance(parents, Sequence), "parents is not a list"
        self.__bids_list: dict = {}

        self.file_name_manipulation = file_name_manipulation
        # Validate
        for ds in datasets:
            ds_path = Path(ds) if isinstance(ds, str) else ds
            if not ds_path.name.startswith("dataset-"):
                print(f"[!] Dataset {ds_path.name} does not start with 'dataset-'")
        for ps in parents:
            if not any(ps.startswith(lp) for lp in parents):
                print(f"[!] Parentfolder {ps} is not a legal name")

        self.datasets = datasets
        self.parents = parents
        self.subjects: dict[str, Subject_Container] = {}
        self.verbose = verbose

        for k in additional_key:
            if k not in entities:
                entities[k] = k
            if k not in entities_keys:
                entities_keys[k] = k

        # Search and add files
        for ds in datasets:
            if not Path(ds).exists():
                raise FileNotFoundError(ds)
            for ps in parents:
                path = Path(ds, ps)
                if path.exists():
                    self.search_folder(path, ds, filter_folder)
        self.entities_keys = entities_keys

    def search_folder(self, path: Path, ds, filter_folder) -> None:
        def scantree(path, lvl=1):
            """Recursively yield DirEntry objects for given directory."""
            for entry in os.scandir(path):
                if entry.is_dir(follow_symlinks=False):
                    if filter_folder is not None and not filter_folder(
                        Path(entry.path), lvl
                    ):
                        continue
                    yield from scantree(entry.path, lvl=lvl + 1)
                else:
                    yield entry

        for entry in scantree(path):
            if entry.is_file():
                if entry.name[0] == ".":
                    continue
                self.add_file_2_subject(Path(entry.path), ds)

    def add_file_2_subject(self, bids: BIDS_FILE | Path, ds=None) -> None:
        if isinstance(bids, Path) and "DS_Store" in bids.name:
            return
        if ds is None:
            if isinstance(bids, BIDS_FILE):
                ds = bids.dataset
            else:
                raise AssertionError("Dataset-path required")
        if isinstance(bids, Path):
            try:
                bids_key, file_type = bids.name.split(".", maxsplit=1)
            except Exception:
                print("[!] skip file with out a type declaration:", bids.name)
                # raise e
                return

            if bids_key in self._global_bids_list:
                self._global_bids_list[bids_key].add_file(bids)
                return
            bids = BIDS_FILE(
                bids,
                ds,
                verbose=self.verbose,
                file_name_manipulation=self.file_name_manipulation,
            )
        subject = bids.info.get("sub", "unsorted")
        if subject not in self.subjects:
            self.subjects[subject] = Subject_Container(
                subject, self.sequence_splitting_keys
            )
        self.count_file += 1
        print(
            f"Found: {subject}, total file keys {(self.count_file)},  total subjects = {len(self.subjects)}    ",
            end="\r",
        )
        self.subjects[subject].add(bids)

    def enumerate_subjects(self, sort=False) -> list[tuple[str, Subject_Container]]:
        # TODO Enumerate should put out numbers...
        if sort:
            return sorted(self.subjects.items())
        return self.subjects.items()  # type: ignore

    def iter_subjects(self, sort=False) -> list[tuple[str, Subject_Container]]:
        if sort:
            return sorted(self.subjects.items())
        return self.subjects.items()  # type: ignore

    def __len__(self):
        return len(self.subjects)

    def __str__(self):
        return (
            "BIDS_Global_info: parents="
            + str(self.parents)
            + f"\nDatasets = {self.datasets}"
        )

    @property
    def _global_bids_list(self):
        return self.__bids_list


class Subject_Container:
    def __init__(self, name, sequence_splitting_keys: list[str]) -> None:
        self.name = name
        self.sequences: dict[str, list[BIDS_FILE]] = {}
        self.sequence_splitting_keys = sequence_splitting_keys.copy()

    def get_sequence_name(self, bids: BIDS_FILE):
        key_values = []
        for key in self.sequence_splitting_keys:
            key_values.append(bids.info[key]) if key in bids.info else None
        key = str.join("_", key_values)
        # sequence_names are only unique in the same session
        # ses = bids.info.get("ses", None)
        append_id = ""
        # idx = 1
        # This code fixes that in different sessions/same patient the filename can reuse splitting-keys
        # while True:
        #    if key + append_id in self.sequences:
        #        other_ses = self.sequences[key + append_id][0].info.get("ses", None)
        #        if ses == other_ses:
        #            break
        #        idx += 1
        #        append_id = f"_{idx}"
        #    else:
        #        break
        return key + append_id

    def add(self, bids: BIDS_FILE) -> None:
        sequ = self.get_sequence_name(bids)
        self.sequences.setdefault(sequ, [])
        if bids not in self.sequences[sequ]:
            self.sequences[sequ].append(bids)
        bids.set_subject(self)

    def new_query(self, flatten=False) -> Searchquery:
        """Make a new search_query

        Args:
            flatten (bool, optional): If you look for single file set flatten to True,
            If you want to find related files use flatten False and generate a dictionary with get_sequence_files after filtering. Defaults to False.

        Returns:
            Searchquery
        """
        return Searchquery(self, flatten)

    def get_sequence_files(
        self,
        sequ: str,
        key_transform: typing.Callable[[BIDS_FILE], str | None] | None = None,
        key_addendum: list[str] | None = None,
        alternative_sequ_list: list[BIDS_FILE] | None = None,
    ) -> BIDS_Family:
        """Returns a dictionary of all files the related sequence.
        Args:
                        sequ (str): key of the sequence
                        key_transform: function that maps BIDS_FILE to family key
                        key_addendum: list of keys that are used as sequence_naming_keys
                        alternative_sequ_list: when passed, is used instead of the self.sequences

        Returns:
            BIDS_Family that contains a data dictionary.
            dict: The key is the 'format' word (ct, dixon, snp,...), except special naming keys are passed.
            Default naming keys: ["seg", "label"] (see sequence_naming_keys for the up-to-date list)
            All default naming keys as well as those passed in key_addendum are appended as key-value to the family-key.
            Example:
            ses-123_msk.nii.gz will get key msk
            seg-subreg_msk.nii.gz will get the key msk_seg-subreg
            ses-123_T1c.nii.gz will get key T1c
        """
        out_dict: dict[str, list[BIDS_FILE]] = {}
        sequ_list: list[BIDS_FILE] = self.sequences[sequ]
        if alternative_sequ_list is not None:
            sequ_list = alternative_sequ_list
        for s in sequ_list:
            key = s.format

            seq_naming_keys = sequence_naming_keys.copy()
            if key_addendum is not None:
                seq_naming_keys += [
                    k for k in key_addendum if k not in sequence_naming_keys
                ]

            for k in seq_naming_keys:
                if k in s.info:
                    key = key + "_" + k + "-" + s.info[k]

            if key_transform is not None:
                out = key_transform(s)
                if isinstance(out, str):
                    key = out

            if key in out_dict:
                s1 = out_dict[key]
                s1.append(s)
            else:
                out_dict[key] = [s]
        return BIDS_Family(out_dict, self.sequence_splitting_keys)


class BIDS_FILE:
    def __init__(
        self,
        file: Path | str,
        dataset: Path | str,
        verbose=True,
        bids_ds: BIDS_Global_info | None = None,
        file_name_manipulation: typing.Callable[[str], str] | None = None,
    ):
        """A multi-file representation. It holds the path to Bids-files with the same identifier (filename excluding the file type).
        It can hold the reference to the nii.gz, json, etc at the same time.

        The following fields are imported and can be accessed.
        self.format (str): The last value determining its use/modalities like T1w, msk, dixon
        self.info (dict[str,str]): key,values of the bids file. Can be changed, but will be only reflected in get_changed_path/find_changed_path if the from_info is set True.
                                    Oder of insert matter (requires python >3.7). get/set/loop are for this dict
        self.BIDS_key(str): Hashing key.
        self.file (dict[str,Path]): keys are file extensions and values are paths to those files with given extension
                                    The first file is added by this init, second by add_file-function. This is usually done by the tree generation of BIDS_Global_info
                                    Self generated BIDS_FILE will not automatically have the json/nii.gz pair, except when their are in the same folder.
        self.dataset (Path): root-dataset name. All should paths will start from here.

        Args:
            file (Path): Path to the file in BIDS format.
            dataset (Path): Top Folder of the dataset
            verbose (bool, optional): You will be informed of non-conform Bids names/keys. Defaults to True.
        """
        file = Path(file)
        self.dataset = Path(dataset)
        self.verbose = verbose
        if file_name_manipulation is not None:
            if "WS_" in str(file):
                file.rename(file.parent / Path(file_name_manipulation(file.name)))
            name = file_name_manipulation(file.name)
        else:
            name = file.name
        self.format, self.info, self.BIDS_key, file_type = get_values_from_name(
            name, verbose
        )

        if bids_ds is not None:
            bids_ds.add_file_2_subject(bids=self, ds=self.dataset)
        self.file = {
            file_type: Path(file),
        }
        bids_key, _ = Path(file).name.split(".", maxsplit=1)
        for file_type in ["nii.gz", "json", "png"]:
            if file_type in self.file:
                continue
            p = Path(Path(file).parent, bids_key + "." + file_type)
            if p.exists():
                self.file[file_type] = p
        self.file = dict(sorted(self.file.items()))

    def __str__(self) -> str:
        s = f"{self.BIDS_key}.{list(self.file.keys())}\t parent = {self.get_parent()}"
        return s

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return self.BIDS_key.__hash__()

    def exists(self):
        if "nii.gz" in self.file:
            return self.file["nii.gz"].exists()
        else:
            return self.file[next(iter(self.file.keys()))].exists()

    def unlink(self, missing_ok=True):
        for f in self.file.values():
            f.unlink(missing_ok=missing_ok)

    def __lt__(self, other):
        return self.BIDS_key < other.BIDS_key

    def __len__(self):
        return 1

    def __getitem__(self, key):
        assert key == 0, key
        return self

    def __iter__(self):
        return iter((self,))

    def __eq__(self, key):
        if hasattr(key, "BIDS_key"):
            return self.BIDS_key == key.BIDS_key
        else:
            return False

    def set_subject(self, sub: Subject_Container):
        self.subject = sub

    def set(self, key, value):
        validate_entities(key, value, f"..._{key}-{value}_...", self.verbose)
        self.info[key] = value

    def get(self, key, default=None):
        if key in self.info:
            return self.info[key]
        return default

    def loop_keys(self):
        return self.info.items()

    def remove(self, key):
        assert key != "sub", "not allowed to remove subject name"
        return self.info.pop(key)

    def add_file(
        self,
        path: Path,
        bids_ds: BIDS_Global_info | None = None,
    ):
        bids_key, file_type = Path(path).name.split(".", maxsplit=1)

        assert (
            bids_key == self.BIDS_key
        ), f"only aligned data aka same name different file type: {bids_key} != {self.BIDS_key}"
        bids_dic_file = self.file
        if file_type not in self.file:
            bids_dic_file[file_type] = path
            if bids_ds is not None:
                bids_ds._global_bids_list[bids_key].file = dict(
                    sorted(bids_dic_file.items())
                )
        self.file = dict(sorted(bids_dic_file.items()))

    def rename_files(self, path: Path | str, ending=".nii.gz"):
        path = str(path)
        assert path.endswith(
            ending
        ), f"set 'ending' to the part after the '.'\n {path} does not end with {ending}"
        path = path.replace(ending, "")
        for key, value in self.file.items():
            p = Path(path + "." + key)
            value.rename(p)

    def get_path_decomposed(self, file_type=None) -> tuple[Path, str, str, str]:
        if file_type is None:
            file_type = next(iter(self.file.keys()))
        folder_list = (
            str(self.file[file_type].relative_to(self.dataset))
            .replace("\\\\", "/")
            .replace("\\", "/")
            .split("/")
        )
        parent = folder_list[0]
        subpath = folder_list[1:-1]
        filename = folder_list[-1]
        # print(parent, subpath, filename)
        return self.dataset, parent, str.join("/", subpath), filename

    @property
    def parent(self):
        return self.get_parent()

    @property
    def bids_format(self):
        return self.format

    def get_parent(self, file_type=None):
        return self.get_path_decomposed(file_type)[1]

    def get_changed_bids(
        self,
        file_type: str | None = "nii.gz",
        bids_format: str | None = None,
        parent: str = "derivatives",
        path: str | None = None,
        info: dict | None = None,
        from_info=False,
        auto_add_run_id=False,
        additional_folder: str | None = None,
        dataset_path: str | None = None,
        make_parent=True,
    ):
        ds = dataset_path if dataset_path is not None else self.get_path_decomposed()[0]
        return BIDS_FILE(
            self.get_changed_path(
                file_type=file_type,
                bids_format=bids_format,
                parent=parent,
                path=path,
                info=info,
                from_info=from_info,
                auto_add_run_id=auto_add_run_id,
                additional_folder=additional_folder,
                dataset_path=dataset_path,
                make_parent=make_parent,
            ),
            ds,
        )

    def get_changed_path(
        self,
        file_type: str | None = "nii.gz",
        bids_format: str | None = None,
        parent: str = "derivatives",
        path: str | None = None,
        info: dict | None = None,
        from_info=False,
        auto_add_run_id=False,
        additional_folder: str | None = None,
        dataset_path: str | None = None,
        make_parent=True,
        no_sorting_mode: bool = False,
        non_strict_mode: bool = False,
    ) -> Path:
        """
        Changes part of the path to generate new flies. The new parent will be derivatives as a default.
        Examples:
        subreg_path = ct_bids.get_changed_path(file_type="nii.gz",parent = "derivatives",info={"seg": "subreg"}, format="cdt")

        Args:
            file_type (str | None, optional): Override the file type, like nii.gz to json Defaults to "nii.gz".

            format (str | None, optional): Changes the "format key" like ct, msk, T1w. Defaults to None.

            parent (str, optional): derivatives or rawdata or any parent folder. Defaults to "derivatives".

            path (str | None, optional): If set: replaces the path from parent to file. Use {key} to get dynamic paths like {/sub-{sub}/ses-{ses}}. Defaults to None.

            info (dict | None, optional): Provide additional key,value pairs like {'seg':'subreg'}. Defaults to None.

            from_info (bool, optional): False: The key,value pair from the original file name are used. True: The key,value of the python object are used. Defaults to False.

            auto_add_run_id (bool, optional): Adds a run-{new id} tag, that is not occupied. Defaults to False.

            additional_folder (str | None, optional): add an additional folder towards path. Defaults to None.

            dataset_path (str | None, optional): Override the dataset_path. Defaults to None.

            no_sorting_mode (bool): If true, will keep the order of the origin nii. Defaults to False

        Returns:
            _type_: _description_
        """ """"""
        if info is None:
            info = {}
        if isinstance(file_type, str) and file_type.startswith("."):
            file_type = file_type[1:]
        path = self.insert_info_into_path(path)
        additional_folder = (
            self.insert_info_into_path(additional_folder)
            if additional_folder is not None
            else None
        )
        ds_path, same_parent, same_path, old_filename = self.get_path_decomposed()
        if from_info:
            same_info = self.info
            same_format = self.format
            same_filetype = None
        else:
            same_format, same_info, _, same_filetype = get_values_from_name(
                old_filename, self.verbose
            )  # Oder of keys is deterministic for python >3.7
        while True:
            file_name = ""
            ## Info
            final_info = {}
            for key, value in same_info.items():
                if key in info:
                    value = info[key]  # noqa: PLW2901
                if value is not None:
                    # file_name += f"{key}-{value}_"
                    if non_strict_mode:
                        validate_entities(
                            key, value, f"..._{key}-{value}_...", verbose=True
                        )
                    else:
                        assert validate_entities(
                            key, value, f"..._{key}-{value}_...", verbose=True
                        )
                    final_info[key] = value
            for key, value in info.items():
                # New Keys are getting checked!
                if non_strict_mode:
                    (
                        print(
                            f"[!] {key} is not in list of legal keys. This name '{key}' is invalid. Legal keys are: {list(entities_keys.keys())}. \nFor use see https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html"
                        )
                        if key not in entities_keys
                        else None
                    )
                else:
                    assert (
                        key in entities_keys
                    ), f"[!] {key} is not in list of legal keys. This name '{key}' is invalid. Legal keys are: {list(entities_keys.keys())}. \nFor use see https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html"
                # validate_entities(key, value, f"..._{key}-{value}_...", self.verbose)
                if key in same_info:
                    continue
                if value is not None:
                    assert validate_entities(key, value, f"..._{key}-{value}_...", True)
                    final_info[key] = value
                # file_name += f"{key}-{value}_"
            # sort by order
            keys_order = final_info.keys()
            if not no_sorting_mode:
                entity_keys = list(entities_keys.keys())
                keys_order = sorted(
                    final_info.keys(),
                    key=lambda x: entity_keys.index(x)
                    if x in entity_keys
                    else list(final_info.keys()).index(x) + len(entity_keys),
                )
            for key in keys_order:
                file_name += f"{key}-{final_info[key]}_"
            # End Info
            bids_format = bids_format if bids_format is not None else same_format
            file_type = file_type if file_type is not None else same_filetype
            assert (
                file_type in file_types
            ), f"[!] {file_type} is not in list of file types. Legal file types are: {list(file_types)}. \nFor use see https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html"
            if bids_format not in formats:
                raise ValueError(
                    f"[!] {bids_format} is not in list of formats. Legal formats are: {list(formats)}. \nFor use see https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html"
                )
            file_name += f"{bids_format}.{file_type}"

            out_path = Path(
                dataset_path if dataset_path is not None else ds_path,
                parent if parent is not None else same_parent,
                path if path is not None else same_path,
                additional_folder if additional_folder is not None else "",
                file_name,
            )
            if make_parent:
                out_path.parent.mkdir(exist_ok=True, parents=True)
            if not auto_add_run_id:
                return out_path
            if not out_path.exists():
                return out_path
            if "run" in info:
                info["run"] += 1
            else:
                info["run"] = 2

    def save_changed_path(
        self,
        bids_format: str | None = None,
        parent: str = "derivatives",
        path: str | None = None,
        info: dict | None = None,
        from_info=False,
        auto_add_run_id=False,
        additional_folder: str | None = None,
        dataset_path: str | None = None,
        non_strict_mode: bool = False,
    ) -> None:
        import shutil

        for key, value in self.file.items():
            out = self.get_changed_path(
                key,
                bids_format=bids_format,
                parent=parent,
                path=path,
                from_info=from_info,
                info=info,
                auto_add_run_id=auto_add_run_id,
                additional_folder=additional_folder,
                dataset_path=dataset_path,
                make_parent=True,
                non_strict_mode=non_strict_mode,
            )
            shutil.copy2(value, out)

    def find_changed_path(
        self,
        bids_ds: BIDS_Global_info,
        bids_format: str | None = None,
        info: dict | None = None,
        from_info=False,
    ) -> BIDS_FILE | None:
        if info is None:
            info = {}

        if from_info:
            same_info = self.info
            same_format = self.format
        else:
            _, _, _, old_filename = self.get_path_decomposed()
            same_format, same_info, _, _ = get_values_from_name(
                old_filename, self.verbose
            )  # Oder of keys is deterministic for python >3.7
        file_name = ""
        for key, value in same_info.items():
            if key in info:
                value = info[key]  # noqa: PLW2901
            file_name += f"{key}-{value}_"
        for key, value in info.items():
            validate_entities(key, value, f"..._{key}-{value}_...", self.verbose)
            if key in same_info:
                continue
            file_name += f"{key}-{value}_"

        file_name += f"{bids_format if bids_format is not None else same_format}"
        return bids_ds._global_bids_list.get(file_name)

    def insert_info_into_path(self, path):
        """Helper function. Automatically replaces {key} with  values from the self.info dict in a string. Like:
        f"sub-{sub}" --> "sub-patient001"
        f"{sub}/ses-{ses}/sub-{sub}_ses-{ses}_label-heart_msk.nii.gz" --> "sub-patient001"
        """
        if path is None:
            return None
        path = str(path)
        while "{" in path:
            left, right = path.split("{", maxsplit=1)
            middle, right = right.split("}", maxsplit=1)
            a = self.info.get(middle, None)
            if a is None:
                warn(f"{middle} not found in {self}", stacklevel=3)
                a = middle
            path = left + a + right
        return path

    def get_sequence_files(
        self,
        key_transform: typing.Callable[[BIDS_FILE], str | None] | None = None,
        key_addendum: list[str] | None = None,
    ):
        """
        See Sequence.get_sequence_files()
        The BIDS_file must be part of a Sequence-family. Usually automatically generated by tree generation of BIDS_Global_info

        Returns:
                dict:
        """

        assert hasattr(
            self, "subject"
        ), "The BIDS_file must be part of a Sequence-family. Usually automatically generated by tree generation of BIDS_Global_info"
        sequ = self.subject.get_sequence_name(self)
        return self.subject.get_sequence_files(
            sequ, key_transform=key_transform, key_addendum=key_addendum
        )

    def open_nii_reorient(self, axcodes_to=("P", "I", "R"), verbose=False):
        return self.open_nii().reorient_(axcodes_to, verbose=verbose)

    def has_json(self) -> bool:
        return "json" in self.file

    def open_json(self) -> dict:
        with open(self.file["json"]) as f:
            return json.load(f)

    def open_poi(self, nii: TPTBox.Image_Reference | None = None):
        from TPTBox import load_poi

        try:
            ctd = load_poi(self.file["json"])
            if (
                ctd.zoom is None
                or ctd.shape is None
                or ctd.rotation is None
                or ctd.origin is None
                or ctd.orientation is None
            ):
                if nii is None and "ctd.json" in str(self.file["json"]):
                    p = Path(str(self.file["json"]).replace("ctd.json", "msk.nii.gz"))
                    nii = p if p.exists() else nii
                assert (
                    nii is not None
                ), "This file has no zoom info. Use open_ctd(self, nii) with a image reference (BIDS_FILE/PATH) with the same nii"
                nii = TPTBox.to_nii(nii)
                assert isinstance(nii, TPTBox.NII)
                ctd.zoom = nii.zoom
                ctd.shape = nii.shape
                ctd.rotation = nii.rotation
                ctd.origin = nii.origin
                ctd.orientation = nii.orientation
        except KeyError as e:
            raise ValueError(
                f"json not present. Found only {self.file.keys()}\t{self.file}\n\n{self}"
            ) from e
        return ctd

    def open_ctd(self, nii: TPTBox.Image_Reference | None = None):
        return self.open_poi(nii)

    def has_nii(self) -> bool:
        return "nii.gz" in self.file

    def open_nii(self):
        try:
            from TPTBox import NII

            return NII.load_bids(self)
        except KeyError as e:
            raise ValueError(
                f"nii.gz not present. Found only {self.file.keys()}\t{self.file}\n\n{self}"
            ) from e

    def has_npz(self) -> bool:
        return "npz" in self.file

    def open_npz(self) -> dict[str, np.ndarray]:
        return dict(np.load(self.file["npz"], allow_pickle=False))  # type: ignore

    def open(self, filetype, _internal=False) -> Path | TPTBox.NII | dict | None:
        if not _internal:
            warn("open is deprecated.", DeprecationWarning, stacklevel=2)
        if filetype not in self.file:
            return None
        if filetype == "json":
            return self.open_json()
        if filetype == "nii.gz":
            return self.open_nii()
        return self.file[filetype]

    def do_filter(
        self,
        key: str,
        constrain: list[str] | str | typing.Callable[[str | object], bool],
        required=False,
    ):
        """
        Returns True/False if the  key,constrain is matched
        If a key is not present the inverse of the "required" value is returned

        Args:
            key (str): The key for which we filter. Can be "format", a filetype, a key from the info-dict
                    In case of filetype + constrain is a callable you get a opened Nifti, opened json or Path
            constrain (str | typing.Callable[[str  |  object], bool]):
                    If a string is given: An exact string match is looked up
                    If a callable: The function is called with the value of the key.
            required (bool, optional): If True: A key must exist or the family/file is filtered.
                    If False: Only if the key exist the family/file will be considers for filtering. Defaults to True.
        """
        key = key.lower()
        if key == "":
            return False
        if key == "format":
            value = self.format
        elif key == "filetype":
            value = list(self.file.keys())
            if isinstance(constrain, str) and constrain.startswith("."):
                constrain = constrain[1:]
        elif key == "parent":
            value = self.get_parent()
        elif key == "self":
            value = self
        elif key in self.info:
            value = self.info[key]
        elif key in self.file:
            value = self.open(key, _internal=True)
        else:
            return not required

        if not isinstance(value, list):
            value = [value]
        for v in value:
            if isinstance(constrain, list):
                if v in constrain:
                    return True
            elif isinstance(constrain, str):
                if v == constrain:
                    return True
            elif constrain(v):
                return True
        return False

    def get_interpolation_order(self) -> int:
        """Returns 0 if the file is a mask or segmentation
        Returns 3 if the file is a image.

        Returns:
            int: interpolation_order
        """
        return 0 if self.format == "msk" or "label" in self.info else 3


class Searchquery:
    def __init__(self, subj: Subject_Container, flatten=False) -> None:
        """Filter for specific files.

        Args:
            subj (Subject_Container): The Subject class in with filter and action is applied
            flatten (bool, optional): If true, you are filtering for single files. If False you looking for sequences. Defaults to False.
        """
        self.candidates: list[BIDS_FILE] | dict[str, list[BIDS_FILE]] = []
        if flatten:
            for value_lists in subj.sequences.values():
                for value in value_lists:
                    self.candidates.append(value)
        else:
            self.candidates = subj.sequences.copy()

        self.subject = subj
        self._flatten = flatten

    @classmethod
    def from_BIDS_Family(cls, fam: BIDS_Family):
        dic = fam.data_dict
        any_file = dic[next(iter(dic.keys()))][0]
        sub = any_file.subject
        query = Searchquery(subj=sub, flatten=False)
        query._filter_fam_id(fam)
        # query.candidates = dic.copy()
        return query

    def _filter_fam_id(self, fam: BIDS_Family):
        self.unflatten()
        dic = fam.data_dict
        any_file = dic[next(iter(dic.keys()))][0]
        subject_id = any_file.subject.get_sequence_name(any_file)
        c = self.candidates
        self.candidates = {}
        self.candidates[subject_id] = c[subject_id]  # type: ignore

    def copy(self):
        copy = Searchquery(self.subject, self._flatten)
        copy.candidates = self.candidates.copy()
        return copy

    def flatten(self):
        """
        Transform from multi-file-mode to single file-mode
        """
        if self._flatten:
            return
        a: list[BIDS_FILE] = []
        assert isinstance(self.candidates, dict)
        for value_lists in self.candidates.values():
            for value in value_lists:
                a.append(value)  # noqa: PERF402
        self.candidates = a
        self._flatten = True

    def unflatten(self):
        """
        Transforms from single file-mode to multi-file-mode. Filtered Objects are still removed
        """
        if not self._flatten:
            return
        a = {}
        assert isinstance(self.candidates, list)
        for value in self.candidates:
            key = self.subject.get_sequence_name(value)
            a.setdefault(key, [])
            a[key].append(value)
        self.candidates = a
        self._flatten = False

    def filter_self(
        self, filter_fun: typing.Callable[[BIDS_FILE], bool], required=True
    ) -> None:
        return self.filter("self", filter_fun, required=required)  # type: ignore

    def filter_json(
        self, filter_fun: typing.Callable[[dict], bool], required=True
    ) -> None:
        return self.filter("json", filter_fun, required=required)  # type: ignore

    def filter(
        self,
        key: str,
        filter_fun: list[str] | str | typing.Callable[[str | object], bool],
        required=True,
    ):
        """Remove family/file from the Searchquery if:
                        (unflatten-mode) NO single file exist in the family returns True
                        (unflatten-mode) the filter_fun returns False

                        If a key is not present the inverse of the "required" value is returned

        Args:
                        key (str): The key for which we filter. Can be "format", a filetype, a key from the info-dict
                                    In case of filetype + filter_fun is a callable you get a opened Nifti, opened json or Path
                        filter_fun (str | typing.Callable[[str  |  object], bool]):
                                    If a string is given: An exact string match is looked up
                                    If a callable: The function is called with the value of the key.
                        required (bool, optional): If True: A key must exist or the family/file is filtered.
                                    If False: Only if the key exist the family/file will be considers for filtering. Defaults to True.
        """
        if self._flatten:
            assert isinstance(self.candidates, list)
            for bids_file in self.candidates.copy():
                if not bids_file.do_filter(key, filter_fun, required=required):
                    self.candidates.remove(bids_file)
        else:
            assert isinstance(self.candidates, dict)
            for sequences, bids_files in self.candidates.copy().items():
                # print(sequences, list(bids_file.do_filter(key, filter_fun, required=required) for bids_file in bids_files))
                if not any(
                    bids_file.do_filter(key, filter_fun, required=required)
                    for bids_file in bids_files
                ):
                    self.candidates.pop(sequences)

    def filter_format(
        self, filter_fun: list[str] | str | typing.Callable[[str | object], bool]
    ):
        if isinstance(filter_fun, list):
            return self.filter_format(lambda x: x in filter_fun)
        return self.filter("format", filter_fun=filter_fun, required=True)

    def filter_filetype(
        self, filter_fun: str | typing.Callable[[str | object], bool], required=True
    ):
        return self.filter("filetype", filter_fun=filter_fun, required=required)

    def filter_non_existence(
        self,
        key: str,
        filter_fun: str | typing.Callable[[str | object], bool] = lambda x: True,  # noqa: ARG005
        required=True,
    ) -> None:
        """Remove family/file from the Searchquery if:

            (unflatten-mode) ANY single file exist in the family returns True

            (unflatten-mode) the filter_fun returns True

            If a key is not present the inverse of the "required" value is returned

        Args:
            key (str): The key for which we filter. Can be "format", a filetype, a key from the info-dict
                        In case of filetype + filter_fun is a callable you get a opened Nifti, opened json or Path
            filter_fun (str | typing.Callable[[str  |  object], bool]):
                If a string is given: An exact string match is looked up
                If a callable: The function is called with the value of the key.
            required (bool, optional): If True: A key must exist or the family/file is filtered.
                If False: Only if the key exist the family/file will be considers for filtering. Defaults to True.
        """
        if self._flatten:
            assert isinstance(self.candidates, list)
            for bids_file in self.candidates.copy():
                if bids_file.do_filter(key, filter_fun, required=required):
                    self.candidates.remove(bids_file)
        else:
            assert isinstance(self.candidates, dict)
            for sequences, bids_files in self.candidates.copy().items():
                # print(sequences, list(bids_file.do_filter(key, filter_fun, required=required) for bids_file in bids_files))
                if any(
                    bids_file.do_filter(key, filter_fun, required=required)
                    for bids_file in bids_files
                ):
                    self.candidates.pop(sequences)

    def filter_dixon_only_inphase(self):
        def json_filter(x):
            return (
                "ImageType" not in x
                or "W" not in x["ImageType"]
                and "F" not in x["ImageType"]
                and "FAT" not in x["ImageType"]
                and "WATER" not in x["ImageType"]
                and "OP" not in x["ImageType"]
            )

        def lam_filter(x):
            return (
                x.upper() != "W"
                and x.upper() != "F"
                and x.upper() != "FAT"
                and x.upper() != "WATER"
                and x.upper() != "OP"
                and x.upper() != "OPP"
                and x.upper() != "OUTPHASE"
            )

        self.filter_json(json_filter, required=False)
        self.filter("rec", lam_filter, required=False)  # type: ignore DEPRECATED
        self.filter("part", lam_filter, required=False)  # type: ignore
        self.filter("acq", lam_filter, required=False)  # type: ignore DEPRECATED

    def filter_dixon_water(self, _keys=None):
        if _keys is None:
            _keys = ["W", "WATER"]
        assert self._flatten

        def json_filter(x):
            return "ImageType" not in x or all(k in x["ImageType"] for k in _keys)

        def lam_filter(x):
            return any(k == x.upper() for k in _keys)

        self.filter_json(json_filter, required=False)
        self.filter("rec", lam_filter, required=False)  # type: ignore
        self.filter("part", lam_filter, required=False)  # type: ignore

    def filter_dixon_fat(self):
        self.filter_dixon_water(_keys=["F", "FAT"])

    def filter_dixon_outphase(self):
        self.filter_dixon_water(_keys=["OP", "OPP", "OUTPHASE"])

    def action(
        self,
        action_fun: typing.Callable[[BIDS_FILE], None],
        filter_fun: str | typing.Callable[[str | object], bool] = lambda x: True,  # noqa: ARG005
        key: str = "",
        required: bool = True,
        all_in_sequence=False,
    ):
        """When the filter_function is True the action_fun is applied on the BIDS_File

        Args:
            action_fun (typing.Callable[[BIDS_FILE], None]): If the filter-function return True: The function is called with the BIDS_file as an argument
            key (str): The key for which we filter. Can be "format", a filetype, a key from the info-dict
                        In case of filetype + filter_fun is a callable you get a opened Nifti, opened json or Path
            filter_fun (str | typing.Callable[[str  |  object], bool]):
                        If a string is given: An exact string match is looked up
                        If a callable: The function is called with the value of the key.
            required (bool): _description_. Defaults to True.
                all_in_sequence (bool, optional): If True, the action_fun is called also on all family files. Defaults to False.
        """
        assert not (self._flatten and all_in_sequence)
        if self._flatten:
            assert isinstance(self.candidates, list)
            for bids_file in self.candidates.copy():
                if bids_file.do_filter(key, filter_fun, required=required):
                    action_fun(bids_file)
        else:
            assert isinstance(self.candidates, dict)
            for bids_files in self.candidates.copy().values():
                if all_in_sequence:  # noqa: SIM102
                    if any(
                        bids_file.do_filter(key, filter_fun, required=required)
                        for bids_file in bids_files
                    ):
                        for bids_file in bids_files:
                            action_fun(bids_file)

                for bids_file in bids_files:
                    if bids_file.do_filter(key, filter_fun, required=required):
                        action_fun(bids_file)

    def __str__(self) -> str:
        if self._flatten:
            s = f"Filter of {self.subject.name}\n"
            assert isinstance(self.candidates, list)
            for bids_file in self.candidates:
                s += "\t"
                s += str(bids_file)
                s += "\n"
            return s
        else:
            s = f"Filter of {self.subject.name}\n"
            assert isinstance(self.candidates, dict)
            for sequences, bids_files in self.candidates.items():
                s += f"\tsequences {sequences}\n"
                for bids_file in bids_files:
                    s += "\t\t"
                    s += str(bids_file)
                    s += "\n"
            return s

    def loop_list(self, sort=False) -> typing.Iterator[BIDS_FILE]:
        """Returns an iterator. Flatten must be True
        Args:
            sort (bool, optional): Sort alphabetically. Defaults to False.

        Returns:
            typing.Iterator[BIDS_FILE]: _description_
        """
        assert isinstance(
            self.candidates, list
        ), "call flatten() before looping as a list"
        if sort:
            return sorted(self.candidates.__iter__())  # type: ignore
        return self.candidates.__iter__()

    def loop_dict(
        self,
        sort=False,
        key_transform: typing.Callable[[BIDS_FILE], str | None] | None = None,
        key_addendum: list[str] | None = None,  # type: ignore
    ) -> typing.Iterator[BIDS_Family]:
        """Returns an iterator. Flatten must be False: it iterates over all families, where the return is the dict from the get_sequence_files function

        Args:
            sort (bool, optional): Sort alphabetically. Defaults to False.
            key_transform (typing.Callable[[BIDS_FILE], str | None]): provide alternative dict name for certain fils, if default should be used return None
        Returns:
            typing.Iterator[typing.Dict[str, BIDS_FILE | list[BIDS_FILE]]]
        """
        assert not self._flatten, "call unflatten first"
        assert isinstance(self.candidates, dict)
        l: typing.Iterator[BIDS_Family] = (
            self.subject.get_sequence_files(
                sequ,
                key_transform=key_transform,
                key_addendum=key_addendum,
                alternative_sequ_list=values,
            )
            for sequ, values in self.candidates.items()
        )
        if sort:
            l = sorted(l)  # type: ignore
        return l


class BIDS_Family:
    def __init__(
        self,
        family_data: dict[str, list[BIDS_FILE]],
        sequence_splitting_keys: list[str],
    ):
        k = []
        for x in family_data.values():
            for y in x:
                assert y.BIDS_key not in k, family_data
                k.append(y.BIDS_key)
        self.sequence_splitting_keys = sequence_splitting_keys.copy()
        self.data_dict = family_data.copy()
        self.family_id = self.get_identifier()

    def __getitem__(self, item: str) -> list[BIDS_FILE]:
        try:
            return self.data_dict[item]
        except KeyError as e:
            raise KeyError(
                f"BIDS_Family does not contain key {item}, only {self.keys()}"
            ) from e

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def __str__(self) -> str:
        s = ""
        for k, v in self:
            s += f"{k:30} : {v}\n"
        return s[:-1]

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, key: str | list[str]):
        if isinstance(key, list):
            contains = True
            for k in key:
                contains = contains and k in self.data_dict
            return contains
        return key in self.data_dict

    def __len__(self) -> int:
        # number of underlying BIDS files
        return sum([v for k, v in self.get_key_len().items()])

    def __iter__(self):
        return iter(self.data_dict.items())

    def __hash__(self) -> int:
        return str(self).__hash__()

    def __lt__(self, other):
        return str(self) < str(other)

    def get_identifier(self):
        first_e = self.data_dict[next(iter(self.data_dict.keys()))][0]
        if "sub" not in first_e.info:
            print(
                f"family_id, no sub-key, got {first_e.info} and data_dict {list(self.data_dict.keys())}"
            )
            identifier = "sub-404"
        else:
            identifier = "sub-" + first_e.info["sub"]
        for s in first_e.info.keys():
            if s in self.sequence_splitting_keys:
                identifier += "_" + s + "-" + first_e.info[s]
        return identifier

    def items(self):
        return self.data_dict.items()

    def keys(self):
        return self.data_dict.keys()

    def sort(self):
        self.data_dict = dict(sorted(self.data_dict.items()))

    def values(self):
        return list(self.data_dict.values())

    def new_query(self, flatten=False):
        q = Searchquery.from_BIDS_Family(self)
        if flatten:
            q.flatten()
        return q

    def get_key_len(self) -> dict[str, int]:
        return {k: len(v) for k, v in self}

    def get_format_len(self):
        format_len = {}
        for k, v in self:
            bids_format = k.split("_")[0]
            if bids_format not in format_len:
                format_len[bids_format] = (0, 0)
            format_len[bids_format] = (
                format_len[bids_format][0] + 1,
                format_len[bids_format][1] + len(v),
            )
        return format_len

    def get_files_with_multiples(self):
        return self.get_files(key=[k for k, v in self.get_key_len().items() if v > 1])

    def get_files(self, key: list[str] | str | None = None):
        if key is None:
            key = [k for k, v in self]
        if isinstance(key, str):
            key = [key]
        family_dict_files: dict[str, list[dict]] = {}
        for k in key:
            family_dict_files[k] = [b.file for b in self[k]]
        return family_dict_files

    def get(self, item: str | list[str], default=None) -> list[BIDS_FILE] | None:
        if not isinstance(item, list):
            item = [item]
        for i in item:
            if i in self:
                return self[i]
        return default

    def get_bids_files_as_dict(self, keys: list[str]) -> dict[str, BIDS_FILE]:
        """Checks each entry of the list, if everything is there, loads everything, one for each entry in keys

        Args:
            keys: list of (keys or list of keys)

        Returns:
            If all keys are present, returns dictionary[key, BIDS_File], else, returns first key that is not found (str)
        """
        loaded_files: dict[str, BIDS_FILE] = {}
        for k in keys:
            if k not in self:
                raise KeyError(f"{k} not in {self.get_key_len()}")
            bids_f = self.get(k)[0]  # type: ignore
            loaded_files[k] = bids_f
        return loaded_files


if __name__ == "__main__":
    global_info = BIDS_Global_info(
        ["/media/robert/Expansion/dataset-Testset"],
        ["sourcedata", "rawdata", "rawdata_ct", "rawdata_dixon", "derivatives"],
    )
    for _, subject in global_info.enumerate_subjects():
        query = subject.new_query()
        # It must exist a dixon and a msk
        query.filter("format", "dixon")
        # A nii.gz must exist
        query.filter("Filetype", "nii.gz")
        query.filter("format", "msk")
        # Example of lamda function filtering
        query.filter("sequ", lambda x: int(x) == 303, required=True)  # type: ignore
        print(query)
        for bids_file in query.loop_dict():
            # Do something with the files.
            dixon = bids_file["dixon"]
            assert isinstance(dixon, list)
            dixon[0].get_sequence_files()
            break
        query = subject.new_query(flatten=True)
        # A nii.gz must exist
        query.filter("Filetype", "json")
        # It must be a ct
        query.filter("format", "ct")
        # Example of lamda function filtering
        query.filter("sequ", lambda x: x != "None" and int(x) > 200, required=True)  # type: ignore
        # query.filter("self",)
        print(query)

# Example output
# Filter of spinegan0042
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0042; Session : 20220517;     Sequence : 301; e : 2;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0042; Session : 20220517;     Sequence : 301; e : 3;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0042; Session : 20220517;     Sequence : 302; e : 2;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0042; Session : 20220517;     Sequence : 302; e : 3;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0042; Session : 20220517;     Sequence : 303; e : 2;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0042; Session : 20220517;     Sequence : 303; e : 3;
#
# Filter of spinegan0042
#        Format : ct;    Filetype : ['json', 'nii.gz']   Subject : spinegan0042; Session : 20220517;     Sequence : 406;
#        Format : ct;    Filetype : ['json', 'nii.gz']   Subject : spinegan0042; Session : 20220517;     Sequence : 206;
#
# Filter of spinegan0026
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 301; e : 2;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 301; e : 3;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 302; e : 2;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 302; e : 3;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 303; e : 2;
#        Format : dixon; Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 303; e : 3;
#
# Filter of spinegan0026
#        Format : ct;    Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210109;     Sequence : 203;
#        Format : ct;    Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 305;
#        Format : ct;    Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 204;
#        Format : ct;    Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210109;     Sequence : 205;
#        Format : ct;    Filetype : ['json', 'nii.gz']   Subject : spinegan0026; Session : 20210111;     Sequence : 205;
