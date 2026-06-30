# https://bids-specification.readthedocs.io/en/stable/02-common-principles.html
from __future__ import annotations

import json
import os
import random
import sys
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

import TPTBox

if TYPE_CHECKING:
    from TPTBox.core.nii_poi_abstract import Grid
from TPTBox.core.bids_constants import (
    entities,
    entities_keys,
    entity_alphanumeric,
    entity_decimal,
    entity_format,
    entity_left_right,
    entity_on_off,
    file_types,
    formats,
    formats_relaxed,
    sequence_naming_keys,
)

_supported_nii_files = ["nii.gz", "nii", "mkd"]
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


def validate_entities(key: str, value: str, name: str, verbose: bool) -> bool:
    """Validate a BIDS key-value entity pair against the BIDS specification.

    Args:
        key: The BIDS entity key (e.g. ``"sub"``, ``"ses"``, ``"seg"``).
        value: The value associated with the key.
        name: The full filename, used for human-readable error messages.
        verbose: If ``True``, print warnings for invalid entities.

    Returns:
        ``True`` when the entity is valid or ``verbose`` is ``False``;
        ``False`` when a violation is detected.
    """
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
            print(f"[!] value for {key} must be alphanumeric. This name '{name}' is invalid, with value {value}")
            return False
        if key in entity_decimal and not value.isdecimal():
            print(f"[!] value for {key} must be decimal. This name '{name}' is invalid, with value {value}")
            return False
        # if int(value) == 0:
        #    print(f"[!] value for {key} must be not 0. This name '{name}' is invalid, with value {value}")
        if key in entity_format and value not in formats_relaxed:
            print(f"[!] value for {key} must be a format. This name '{name}' is invalid, with value {value}")
            return False
        if key in entity_on_off and value not in ["on", "off"]:
            print(f"[!] value for {key} must be in {['on', 'off']}. This name '{name}' is invalid, with value {value}")
            return False
        if key in entity_left_right and value not in ["L", "R"]:
            print(f"[!] value for {key} must be in {['L', 'R']}. This name '{name}' is invalid, with value {value}")
            return False
        # parts = [
        #    "mag",
        #    "phase",
        #    "real",
        #    "imag",
        #    "inphase",
        #    "outphase",
        #    "fat",
        #    "water",
        #    "eco0-opp1",
        #    "eco0-opp1",
        #    "eco1-pip1",
        #    "eco2-opp2",
        #    "eco3-in1",
        #    "eco4-pop1",
        #    "eco5-arb1",
        #    "fat-outphase",
        #    "water-outphase",
        #    "water-fraction",
        #    "fat-fraction",
        #    "r2s",
        # ]
        # if key in entity_parts and value not in parts:
        #    print(f'[!] value for {key} must be in {parts}. This name "{name}" is invalid, with value {value}')
        #    return False
        else:
            return True
    except Exception as e:
        print(e)
        return False


def get_values_from_name(path: Path | str, verbose: bool) -> tuple[str, dict[str, str], str, str]:
    """Parse a BIDS-formatted filename into its constituent components.

    Splits a filename like ``sub-001_ses-01_T1w.nii.gz`` into the BIDS
    format label, a key-value entity dictionary, the BIDS key stem, and
    the file-type extension.

    Args:
        path: Path to (or plain name of) a BIDS file.
        verbose: If ``True``, print warnings for non-conformant filenames.

    Returns:
        A 4-tuple ``(bids_format, info_dict, bids_key, file_type)`` where

        * ``bids_format`` is the trailing label (e.g. ``"T1w"``),
        * ``info_dict`` maps entity keys to values
          (e.g. ``{"sub": "001", "ses": "01"}``),
        * ``bids_key`` is the full stem without the extension
          (e.g. ``"sub-001_ses-01_T1w"``),
        * ``file_type`` is the extension (e.g. ``"nii.gz"``).
    """
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
                print(f"[!] First key must be sub not {key}. This name '{name}' is invalid")
            if idx != 1 and key == "ses" and verbose:
                print(f"[!] Session must be second key. This name '{name}' is invalid")

            if key in dic and verbose:
                print(f"[!] {bids_key} contains copies of the same key twice. This name '{name}' is invalid")

            validate_entities(key, value, name, verbose)
            dic[key] = value
        except Exception:
            if verbose:
                print(f'[!] "{s}" is not a valid key/value pair. Expected "KEY-VALUE" in {name}')
    return bids_format, dic, bids_key, file_type


def Buffered_BIDS_Global_info(
    datasets: Sequence[Path | str] | str | Path,
    parents: Sequence[str] | str = ["rawdata", "derivatives"],
    additional_key: Sequence[str] = ["sequ", "seg", "ovl"],
    verbose: bool = True,
    file_name_manipulation: typing.Callable[[str], str] | None = None,
    sequence_splitting_keys: list[str] | None = None,
    filter_file: typing.Callable[[Path], bool] | None = None,
    max_age_days: int = 30,
    recompute_parents: list[str] | None = None,
) -> BIDS_Global_info:
    """Create a :class:`BIDS_Global_info` object backed by an on-disk file-path cache.

    Scans each ``<dataset>/<parent>`` folder and serialises the discovered
    file paths to a hidden pickle file (``.filepaths``) so that subsequent
    calls can skip the directory walk.  The cache is automatically
    invalidated when it is older than ``max_age_days`` days.

    Args:
        datasets: One or more dataset root directories (must contain a
            ``dataset-`` prefix in their name per BIDS convention).
        parents: Parent sub-folders to search inside each dataset, e.g.
            ``["rawdata", "derivatives"]``.
        additional_key: Extra BIDS entity keys beyond the official spec that
            should not trigger validation warnings.
        verbose: Print progress and cache-status messages.
        file_name_manipulation: Optional callable applied to each filename
            before BIDS parsing, e.g. to normalise non-conformant names.
        sequence_splitting_keys: Keys used to group files into sequences
            (families).  Defaults to the library-level constant when
            ``None``.
        filter_file: Optional predicate; if provided, only paths for which
            the function returns ``True`` are included.
        max_age_days: Number of days after which the on-disk cache is
            considered stale and regenerated.  Defaults to ``30``.
        recompute_parents: Parent names for which the cache should always be
            rebuilt even if a valid cache exists.

    Returns:
        A fully initialised :class:`BIDS_Global_info` instance.
    """
    import pickle

    if recompute_parents is None:
        recompute_parents = []
    buffer_name = ".filepaths"
    if isinstance(datasets, (str, Path)):
        datasets = [datasets]
    files = {ds: [] for ds in datasets}

    def save_buffer(f: Path, buffer_name: str) -> list[Path]:
        """Scan *f*, persist the discovered file list to a pickle cache, and return it."""
        global _cont  # noqa: PLW0603
        new_buffer = [Path(f.path) for f in _scan_tree(f, verbose=True) if Path(f.path).is_file()]
        try:
            with open(str(f / buffer_name), "wb") as b:
                pickle.dump(new_buffer, b)
                print("\n[ ] Save Buffer:", f) if verbose else None
        except OSError:
            print("Saving not allowed")

        _cont = 0
        return new_buffer

    for dataset in datasets:
        for parent in parents:
            assert "/" not in parent, "only top parent folder allowed"
            folder = Path(dataset, parent)
            if not folder.exists():
                print("[ ] Dose not exist:", (folder), f"{' ':20}") if verbose else None
                continue
            if (folder / buffer_name).exists():
                import datetime

                file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(folder / buffer_name))
                today = datetime.datetime.today()

                age = today - file_mod_time
                if age.days >= int(max_age_days):
                    (
                        print(
                            "[ ] Delete Buffer - to old:",
                            (folder / buffer_name),
                            f"{' ':20}",
                        )
                        if verbose
                        else None
                    )
                    (folder / buffer_name).unlink()
            if (folder / buffer_name).exists() and parent not in recompute_parents:
                with open((folder / buffer_name), "rb") as b:
                    l = pickle.load(b)
                    (
                        print(
                            f"[{len(l):8}] Read Buffer:",
                            (folder / buffer_name),
                            f"{' ':20}",
                        )
                        if verbose
                        else None
                    )
                    files[dataset] += l
            else:
                (
                    print(
                        f"[{_cont:8}] Create new Buffer:",
                        (folder / buffer_name),
                        f"{' ':20}",
                        end="\r",
                    )
                    if verbose
                    else None
                )
                files[dataset] += save_buffer((folder), buffer_name)
    if filter_file is not None:
        files: dict[Path | str, list[Path]] = {d: [g for g in f if filter_file(g)] for d, f in files.items()}
    return BIDS_Global_info(
        datasets,
        parents,
        additional_key,
        verbose=verbose,
        file_name_manipulation=file_name_manipulation,
        sequence_splitting_keys=sequence_splitting_keys,
        filter_folder=lambda _x, _y: False,
        additional_file_list=files,
    )


_cont = 0


def _scan_tree(path, lvl=1, filter_folder=lambda _x, _y: True, verbose=False):
    """Recursively yield DirEntry objects for given directory."""
    global _cont  # noqa: PLW0603
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            if filter_folder is not None and not filter_folder(Path(entry.path), lvl):
                continue
            yield from _scan_tree(entry.path, lvl=lvl + 1, verbose=verbose)
        elif entry.name[0] != ".":
            if verbose:
                print(f"[{_cont:8}]", end="\r")
                _cont += 1

            yield entry


class BIDS_Global_info:
    """Global index of a BIDS dataset, mapping subjects to their files across multiple dataset roots."""

    def __init__(
        self,
        datasets: Sequence[Path | str] | Sequence[Path] | Sequence[str] | str | Path,
        parents: Sequence[str] | str = ["rawdata", "derivatives"],
        additional_key: Sequence[str] = ["sequ", "seg", "ovl"],
        verbose: bool = True,
        file_name_manipulation: typing.Callable[[str], str] | None = None,
        sequence_splitting_keys: list[str] | None = None,
        filter_folder: typing.Callable[[Path, int], bool] | None = None,
        additional_file_list: dict[str | Path, list[Path]] | None = None,
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
        if isinstance(datasets, (Path, str)):
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
        if additional_file_list is not None:
            for ds, file_list in additional_file_list.items():
                for f in file_list:
                    # if Path(f).is_file():
                    self.add_file_2_subject(f, ds)

        self.entities_keys = entities_keys

    def search_folder(self, path: Path, ds, filter_folder) -> None:
        """Recursively scan *path* and register every file found with the global info.

        Args:
            path: Directory to scan.
            ds: Dataset root path associated with the scanned folder.
            filter_folder: Callable ``(path, level) -> bool``; folders for
                which this returns ``False`` are skipped.
        """
        for entry in _scan_tree(path, filter_folder=filter_folder):
            if entry.is_file():
                self.add_file_2_subject(Path(entry.path), ds)

    def add_file_2_subject(self, bids: BIDS_FILE | Path, ds: Path | str | None = None) -> None:
        """Parse a file path (or pre-built :class:`BIDS_FILE`) and add it to the correct subject bucket.

        Args:
            bids: Either a raw filesystem path or an already-constructed
                :class:`BIDS_FILE` instance.
            ds: Dataset root path.  Required when *bids* is a plain
                :class:`~pathlib.Path`; inferred automatically when *bids*
                is a :class:`BIDS_FILE`.

        Raises:
            AssertionError: If *bids* is a :class:`~pathlib.Path` and *ds*
                is ``None``.
        """
        if isinstance(bids, Path) and "DS_Store" in bids.name:
            return
        if ds is None:
            if isinstance(bids, BIDS_FILE):
                ds = bids.dataset
            else:
                raise AssertionError("Dataset-path required")

        if isinstance(bids, (Path, str)):
            try:
                bids_key, file_type = str(bids).rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)
                # print(bids_key)
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
            self.subjects[subject] = Subject_Container(subject, self.sequence_splitting_keys)
        self.count_file += 1
        (
            print(
                f"Found: {subject}, total file keys {(self.count_file)},  total subjects = {len(self.subjects)}    ",
                end="\r",
            )
            if self.verbose
            else None
        )
        self.subjects[subject].add(bids)

    def enumerate_subjects(self, sort: bool = False, shuffle: bool = False) -> list[tuple[str, Subject_Container]]:
        """Return all subject identifiers together with their :class:`Subject_Container`.

        Args:
            sort: If ``True``, return subjects sorted alphabetically by
                subject ID.
            shuffle: If ``True``, return subjects in a random order.
                Mutually exclusive with *sort* (sort takes precedence).

        Returns:
            A list of ``(subject_id, Subject_Container)`` pairs.
        """
        # TODO Enumerate should put out numbers...
        if sort:
            return sorted(self.subjects.items())
        if shuffle:
            s = list(self.subjects.items())
            random.shuffle(s)
            return s
        return self.subjects.items()  # type: ignore

    def iter_subjects(self, sort: bool = False, shuffle: bool = False) -> list[tuple[str, Subject_Container]]:
        """Iterate over all subjects (alias for :meth:`enumerate_subjects` without shuffle).

        Args:
            sort: If ``True``, return subjects sorted alphabetically by
                subject ID.

        Returns:
            A list of ``(subject_id, Subject_Container)`` pairs.
        """
        if sort:
            return sorted(self.subjects.items())
        if shuffle:
            s = list(self.subjects.items())
            random.shuffle(s)
            return s
        return self.subjects.items()  # type: ignore

    def __len__(self):
        return len(self.subjects)

    def __str__(self):
        return "BIDS_Global_info: parents=" + str(self.parents) + f"\nDatasets = {self.datasets}"

    @property
    def _global_bids_list(self) -> dict:
        """Internal mapping from BIDS key stem to :class:`BIDS_FILE` instances."""
        return self.__bids_list


class Subject_Container:
    """Container for all BIDS files belonging to a single subject, grouped by sequence."""

    def __init__(self, name, sequence_splitting_keys: list[str]) -> None:
        self.name = name
        self.sequences: dict[str, list[BIDS_FILE]] = {}
        self.sequence_splitting_keys = sequence_splitting_keys.copy()

    def get_sequence_name(self, bids: BIDS_FILE) -> str:
        """Derive the sequence-bucket key for a given BIDS file.

        Combines the values of :attr:`sequence_splitting_keys` that are
        present in *bids* into a single underscore-joined string.

        Args:
            bids: The file whose sequence name should be resolved.

        Returns:
            A string key identifying the sequence bucket, e.g.
            ``"ses-01_sequ-303"``.
        """
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
        """Register a BIDS file with this subject container.

        Places *bids* into the correct sequence bucket (determined by
        :meth:`get_sequence_name`) and sets the back-reference on the file.

        Args:
            bids: The BIDS file to add.
        """
        sequ = self.get_sequence_name(bids)
        self.sequences.setdefault(sequ, [])
        if bids not in self.sequences[sequ]:
            self.sequences[sequ].append(bids)
        bids.set_subject(self)

    def new_query(self, flatten=False) -> Searchquery:
        """Make a new search_query.

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
                seq_naming_keys += [k for k in key_addendum if k not in sequence_naming_keys]

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
    """Representation of a single BIDS-compliant file with parsed entities and dataset context."""

    def __init__(
        self,
        file: Path | str,
        dataset: Path | str,
        verbose=True,
        bids_ds: BIDS_Global_info | None = None,
        file_name_manipulation: typing.Callable[[str], str] | None = None,
    ):
        """Multi-file BIDS record sharing the same identifier (all extensions of one file stem).

        Holds references to `.nii.gz`, `.json`, etc. simultaneously.

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
        file = Path(file) if not isinstance(file, Path) else file
        self.dataset = Path(dataset) if not isinstance(dataset, Path) else dataset
        self.verbose = verbose
        if file_name_manipulation is not None:
            if "WS_" in str(file):
                file.rename(file.parent / Path(file_name_manipulation(file.name)))
            name = file_name_manipulation(file.name)
        else:
            name = file.name
        self.format, self.info, self.BIDS_key, file_type = get_values_from_name(name, verbose)

        if bids_ds is not None:
            bids_ds.add_file_2_subject(bids=self, ds=self.dataset)
        self.file = {file_type: file}
        bids_key, _ = file.name.split(".", maxsplit=1)
        for file_type in ["nii.gz", "json", "png"]:
            if file_type in self.file:
                continue
            if os.path.exists(os.path.join(file.parent, bids_key + "." + file_type)):
                self.file[file_type] = Path(file.parent, bids_key + "." + file_type)
        self.file = dict(sorted(self.file.items()))

    def get_file(self, ending: str = "json", default: Path | None = None) -> Path | None:
        """Return the path for a given file extension, or *default* if absent.

        Args:
            ending: File extension to look up, e.g. ``"json"`` or
                ``"nii.gz"``.
            default: Value returned when the extension is not present.

        Returns:
            The :class:`~pathlib.Path` for the requested extension, or
            *default*.
        """
        return self.file.get(ending, default)

    def __str__(self) -> str:
        s = f"{self.BIDS_key}.{list(self.file.keys())}\t parent = {self.get_parent()}"
        return s

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return self.BIDS_key.__hash__()

    def exists(self) -> bool:
        """Return ``True`` when the primary file (preferring ``nii.gz``) exists on disk.

        Returns:
            ``True`` if the file exists, ``False`` otherwise.
        """
        if "nii.gz" in self.file:
            return self.file["nii.gz"].exists()
        else:
            return self.file[next(iter(self.file.keys()))].exists()

    def unlink(self, missing_ok: bool = True) -> None:
        """Delete all files associated with this BIDS entry from disk.

        Args:
            missing_ok: If ``True``, suppress errors when a file does not
                exist.  Passed directly to :meth:`pathlib.Path.unlink`.
        """
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

    def set_subject(self, sub: Subject_Container) -> None:
        """Attach a back-reference to the owning :class:`Subject_Container`.

        Args:
            sub: The subject container that owns this file.
        """
        self.subject = sub

    def set(self, key: str, value: str) -> None:
        """Set a BIDS entity key-value pair on this file, validating the entity.

        Args:
            key: BIDS entity key (e.g. ``"seg"``).
            value: Value to assign to the key.
        """
        validate_entities(key, value, f"..._{key}-{value}_...", self.verbose)
        self.info[key] = value

    def get(self, key: str, default: str | None = None) -> str | None:
        """Return the value for a BIDS entity key, or *default* if not present.

        Args:
            key: BIDS entity key to look up (e.g. ``"sub"``, ``"ses"``).
            default: Fallback value when *key* is absent.

        Returns:
            The entity value string, or *default*.
        """
        if key in self.info:
            return self.info[key]
        return default

    def loop_keys(self) -> typing.ItemsView[str, str]:
        """Return all BIDS entity key-value pairs for this file.

        Returns:
            A view of ``(key, value)`` pairs from the :attr:`info` dictionary.
        """
        return self.info.items()

    def remove(self, key: str) -> str:
        """Remove and return a BIDS entity key from :attr:`info`.

        Args:
            key: Entity key to remove.  Must not be ``"sub"``.

        Returns:
            The value that was associated with *key*.

        Raises:
            AssertionError: If *key* is ``"sub"`` (subject ID cannot be
                removed).
            KeyError: If *key* is not present in :attr:`info`.
        """
        assert key != "sub", "not allowed to remove subject name"
        return self.info.pop(key)

    def add_file(
        self,
        path: Path,
        bids_ds: BIDS_Global_info | None = None,
    ) -> None:
        """Associate an additional file extension with this BIDS entry.

        Used to register companion files (e.g. a ``.json`` sidecar alongside
        a ``nii.gz``) that share the same BIDS key stem.

        Args:
            path: Path to the companion file.  Its stem must match
                :attr:`BIDS_key`.
            bids_ds: If provided, the global registry is updated so that the
                merged file dictionary is reflected there as well.

        Raises:
            AssertionError: If the stem of *path* does not match
                :attr:`BIDS_key`.
        """
        bids_key, file_type = Path(path).name.split(".", maxsplit=1)

        assert bids_key == self.BIDS_key, f"only aligned data aka same name different file type: {bids_key} != {self.BIDS_key}"
        bids_dic_file = self.file
        if file_type not in self.file:
            bids_dic_file[file_type] = path
            if bids_ds is not None:
                bids_ds._global_bids_list[bids_key].file = dict(sorted(bids_dic_file.items()))
        self.file = dict(sorted(bids_dic_file.items()))

    def rename_files(self, path: Path | str, ending: str = ".nii.gz") -> None:
        """Rename all associated files on disk to a new base path.

        The *ending* suffix is stripped from *path* to obtain the base stem,
        then each extension in :attr:`file` is appended.

        Args:
            path: Target path including the primary extension (e.g.
                ``/out/sub-001_T1w.nii.gz``).
            ending: Extension that terminates *path* and that will be
                stripped before adding per-extension suffixes.

        Raises:
            AssertionError: If *path* does not end with *ending*.
        """
        path = str(path)
        assert path.endswith(ending), f"set 'ending' to the part after the '.'\n {path} does not end with {ending}"
        path = path.replace(ending, "")
        for key, value in self.file.items():
            p = Path(path + "." + key)
            value.rename(p)

    def symlink_files(self, path: Path | str, ending: str = ".nii.gz", exist_ok: bool = False) -> None:
        """Create symbolic links for all associated files at a new base path.

        Equivalent to :meth:`rename_files` but creates symlinks rather than
        moving files.  Existing correct symlinks are silently skipped.

        Args:
            path: Target path including the primary extension (e.g.
                ``/out/sub-001_T1w.nii.gz``).
            ending: Extension used to compute the base stem; a leading dot is
                added automatically if absent.

        Raises:
            AssertionError: If *path* does not end with *ending*, or if an
                existing symlink at the target points elsewhere.
        """
        ending = ending if ending[0] == "." else "." + ending
        path = str(path)
        assert path.endswith(ending), f"set 'ending' to the part after the '.'\n {path} does not end with {ending}"
        path = path.replace(ending, "")
        for key, value in self.file.items():
            p = Path(path + "." + key)

            if os.path.islink(p):
                assert Path(os.readlink(p)) == value, f"{p} exists"
                continue
            if exist_ok and p.exists():
                continue

            os.symlink(value, p)

    def get_path_decomposed(self, file_type: str | None = None) -> tuple[Path, str, str, str]:
        """Decompose the file path relative to the dataset root.

        Args:
            file_type: Extension key to use when selecting which path to
                decompose (e.g. ``"nii.gz"``).  Defaults to the first
                extension in :attr:`file`.

        Returns:
            A 4-tuple ``(dataset_path, parent, sub_path, filename)`` where

            * ``dataset_path`` is the dataset root :class:`~pathlib.Path`,
            * ``parent`` is the top-level folder (e.g. ``"rawdata"``),
            * ``sub_path`` is the intermediate path (e.g.
              ``"sub-001/ses-01"``),
            * ``filename`` is the bare filename including extension.
        """
        if file_type is None:
            file_type = next(iter(self.file.keys()))
        folder_list = str(self.file[file_type].relative_to(self.dataset)).replace("\\\\", "/").replace("\\", "/").split("/")
        parent = folder_list[0]
        subpath = folder_list[1:-1]
        filename = folder_list[-1]
        # print(parent, subpath, filename)
        return self.dataset, parent, str.join("/", subpath), filename

    @property
    def parent(self) -> str:
        """Top-level parent folder name (e.g. ``"rawdata"`` or ``"derivatives"``)."""
        return self.get_parent()

    @property
    def bids_format(self) -> str:
        """Alias for :attr:`format`; the BIDS modality/format label (e.g. ``"T1w"``)."""
        return self.format

    @property
    def mod(self) -> str | None:
        """Modality label, resolving ``"msk"`` to the underlying ``mod`` entity value.

        Returns:
            The ``mod`` entity value for mask files, or :attr:`bids_format`
            for all other formats.
        """
        mod = self.bids_format
        if mod == "msk":
            return self.get("mod")
        return mod

    def get_parent(self, file_type: str | None = None) -> str:
        """Return the top-level parent folder name for this file.

        Args:
            file_type: Extension key used to select which path to inspect.
                Defaults to the first extension in :attr:`file`.

        Returns:
            The parent folder name, e.g. ``"rawdata"`` or ``"derivatives"``.
        """
        return self.get_path_decomposed(file_type)[1]

    def get_changed_bids(
        self,
        file_type: str | None = "nii.gz",
        bids_format: str | None = None,
        parent: str = "derivatives",
        path: str | None = None,
        info: dict | None = None,
        from_info: bool = False,
        auto_add_run_id: bool = False,
        additional_folder: str | None = None,
        dataset_path: str | None = None,
        make_parent: bool = False,
        non_strict_mode: bool = False,
    ) -> BIDS_FILE:
        """Construct a new :class:`BIDS_FILE` pointing to a derived output path.

        Delegates path construction to :meth:`get_changed_path` and wraps the
        result in a :class:`BIDS_FILE` instance.

        Args:
            file_type: Target file extension (e.g. ``"nii.gz"``).
            bids_format: Override the format/modality label.
            parent: Target parent folder (e.g. ``"derivatives"``).
            path: Override the intermediate sub-path; supports ``{key}``
                template substitution.
            info: Additional or overriding entity key-value pairs.
            from_info: If ``True``, use the in-memory :attr:`info` dict
                instead of the original filename entities.
            auto_add_run_id: Append an auto-incremented ``run`` tag when the
                target path already exists.
            additional_folder: Extra folder appended between *path* and the
                filename.
            dataset_path: Override the dataset root.
            make_parent: Create parent directories if they do not exist.
            non_strict_mode: Relax BIDS entity validation.

        Returns:
            A new :class:`BIDS_FILE` pointing to the derived output.
        """
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
                non_strict_mode=non_strict_mode,
            ),
            ds,
        )

    def get_changed_path(  # noqa: C901
        self,
        file_type: str | None = "nii.gz",
        bids_format: str | None = None,
        parent: str = "derivatives",
        path: str | None = None,
        info: dict | None = None,
        from_info=False,
        auto_add_run_id=False,
        additional_folder: str | None = None,
        dataset_path: str | Path | None = None,
        make_parent=False,
        no_sorting_mode: bool = False,
        non_strict_mode: bool = False,
    ) -> Path:
        """Changes part of the path to generate new flies. The new parent will be derivatives as a default.

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
        """
        if info is None:
            info = {}
        if non_strict_mode and not self.BIDS_key.startswith("sub"):
            info["sub"] = self.BIDS_key.replace("_", "-").replace(".", "-")
        else:
            # replace _ with - in all info
            self.info = {k: v.replace("_", "-") for k, v in self.info.items()}
        if isinstance(file_type, str) and file_type.startswith("."):
            file_type = file_type[1:]
        path = self.insert_info_into_path(path)
        additional_folder = self.insert_info_into_path(additional_folder) if additional_folder is not None else None
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
                        validate_entities(key, value, f"..._{key}-{value}_...", verbose=True)
                    else:
                        assert validate_entities(key, value, f"..._{key}-{value}_...", verbose=True)
                    final_info[key] = value.replace("_", "-")
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
                    assert key in entities_keys, (
                        f"[!] {key} is not in list of legal keys. This name '{key}' is invalid. Legal keys are: {list(entities_keys.keys())}. \nFor use see https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html"
                    )
                # validate_entities(key, value, f"..._{key}-{value}_...", self.verbose)
                if key in same_info:
                    continue
                if value is not None:
                    if not non_strict_mode:
                        assert validate_entities(key, value, f"..._{key}-{value}_...", True), f"..._{key}-{value}_..."
                    final_info[key] = value
                # file_name += f"{key}-{value}_"
            # sort by order
            keys_order = final_info.keys()
            if not no_sorting_mode:
                entity_keys = list(entities_keys.keys())
                keys_order = sorted(
                    final_info.keys(),
                    key=lambda x: entity_keys.index(x) if x in entity_keys else list(final_info.keys()).index(x) + len(entity_keys),
                )
            for key in keys_order:
                file_name += f"{key}-{final_info[key]}_"
            # End Info
            bids_format = bids_format if bids_format is not None else same_format
            file_type = file_type if file_type is not None else same_filetype
            assert file_type in file_types, (
                f"[!] {file_type} is not in list of file types. Legal file types are: {list(file_types)}. \nFor use see https://bids-specification.readthedocs.io/en/stable/99-appendices/09-entities.html"
            )
            if bids_format not in formats and not non_strict_mode:
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
        from_info: bool = False,
        auto_add_run_id: bool = False,
        additional_folder: str | None = None,
        dataset_path: str | None = None,
        non_strict_mode: bool = False,
    ) -> None:
        """Copy all associated files to their derived output paths.

        Calls :meth:`get_changed_path` for every extension in :attr:`file`
        and copies each source file to the resulting destination using
        :func:`shutil.copy2`.

        Args:
            bids_format: Override the format/modality label.
            parent: Target parent folder (e.g. ``"derivatives"``).
            path: Override the intermediate sub-path; supports ``{key}``
                template substitution.
            info: Additional or overriding entity key-value pairs.
            from_info: If ``True``, use the in-memory :attr:`info` dict
                instead of the original filename entities.
            auto_add_run_id: Append an auto-incremented ``run`` tag when the
                target path already exists.
            additional_folder: Extra folder appended between *path* and the
                filename.
            dataset_path: Override the dataset root.
            non_strict_mode: Relax BIDS entity validation.
        """
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
        from_info: bool = False,
    ) -> BIDS_FILE | None:
        """Look up an already-registered derived file in the global BIDS index.

        Constructs the expected BIDS key for the derived file (applying the
        same entity overrides as :meth:`get_changed_path`) and returns the
        matching :class:`BIDS_FILE` from *bids_ds* if it exists.

        Args:
            bids_ds: The global BIDS dataset to search.
            bids_format: Override the format/modality label.
            info: Additional or overriding entity key-value pairs.
            from_info: If ``True``, use the in-memory :attr:`info` dict
                instead of the original filename entities.

        Returns:
            The matching :class:`BIDS_FILE` if found, otherwise ``None``.
        """
        if info is None:
            info = {}

        if from_info:
            same_info = self.info
            same_format = self.format
        else:
            _, _, _, old_filename = self.get_path_decomposed()
            same_format, same_info, _, _ = get_values_from_name(old_filename, self.verbose)  # Oder of keys is deterministic for python >3.7
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

    def insert_info_into_path(self, path: str | None) -> str | None:
        """Replace ``{key}`` placeholders in *path* with entity values from :attr:`info`.

        Example::

            # With self.info = {"sub": "patient001", "ses": "01"}
            bids.insert_info_into_path("sub-{sub}/ses-{ses}")
            # -> "sub-patient001/ses-01"

        Args:
            path: Template string containing ``{key}`` placeholders, or
                ``None``.

        Returns:
            The expanded string, or ``None`` when *path* is ``None``.
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
    ) -> BIDS_Family:
        """Return the :class:`BIDS_Family` that this file belongs to.

        Delegates to :meth:`Subject_Container.get_sequence_files` using the
        sequence name resolved from this file's entities.  The file must be
        part of a subject sequence (i.e. registered via
        :class:`BIDS_Global_info`).

        Args:
            key_transform: Optional callable mapping a :class:`BIDS_FILE` to
                a custom family key string; return ``None`` to use the default
                key.
            key_addendum: Extra entity keys appended to the family key beyond
                the default :attr:`sequence_naming_keys`.

        Returns:
            The :class:`BIDS_Family` containing all related files for this
            sequence.

        Raises:
            AssertionError: If this file has no owning
                :class:`Subject_Container`.
        """
        assert hasattr(self, "subject"), (
            "The BIDS_file must be part of a Sequence-family. Usually automatically generated by tree generation of BIDS_Global_info"
        )
        sequ = self.subject.get_sequence_name(self)
        return self.subject.get_sequence_files(sequ, key_transform=key_transform, key_addendum=key_addendum)

    def open_nii_reorient(self, axcodes_to: tuple[str, ...] = ("P", "I", "R"), verbose: bool = False) -> TPTBox.NII:
        """Open the NIfTI file and reorient it in-place to the target axis codes.

        Args:
            axcodes_to: Desired orientation as a tuple of axis code strings,
                e.g. ``("P", "I", "R")``.
            verbose: If ``True``, print reorientation details.

        Returns:
            The reoriented :class:`~TPTBox.NII` volume.
        """
        return self.open_nii().reorient_(axcodes_to, verbose=verbose)

    def has_json(self) -> bool:
        """Return ``True`` when a JSON sidecar is registered for this file."""
        return "json" in self.file

    def open_json(self) -> dict:
        """Load and return the JSON sidecar as a dictionary.

        Returns:
            Parsed JSON contents.

        Raises:
            KeyError: If no JSON file is registered in :attr:`file`.
        """
        with open(self.file["json"]) as f:
            return json.load(f)

    def open_poi(self, nii: TPTBox.Image_Reference | None = None) -> TPTBox.POI:
        """Load the associated JSON file as a :class:`~TPTBox.POI` (point-of-interest) object.

        If the POI lacks spatial metadata (zoom, shape, etc.) a reference
        NIfTI image is required to fill in those fields.

        Args:
            nii: Optional image reference used to supply missing spatial
                metadata.  Required when the JSON does not embed grid info.

        Returns:
            The loaded :class:`~TPTBox.POI` object with complete spatial
            metadata.

        Raises:
            ValueError: If no JSON file is present or the POI lacks spatial
                metadata and *nii* is ``None``.
        """
        from TPTBox import POI

        try:
            ctd = POI.load(self.file["json"], allow_global=True)
            if ctd.zoom is None or ctd.shape is None or ctd.rotation is None or ctd.origin is None or ctd.orientation is None:
                if nii is None and "ctd.json" in str(self.file["json"]):
                    p = Path(str(self.file["json"]).replace("ctd.json", "msk.nii.gz"))
                    nii = p if p.exists() else nii
                assert nii is not None, (
                    "This file has no zoom info. Use open_ctd(self, nii) with a image reference (BIDS_FILE/PATH) with the same nii"
                )
                nii = TPTBox.to_nii(nii)
                assert isinstance(nii, TPTBox.NII)
                ctd.zoom = nii.zoom
                ctd.shape = nii.shape
                ctd.rotation = nii.rotation
                ctd.origin = nii.origin
                ctd.orientation = nii.orientation
        except KeyError as e:
            raise ValueError(f"json not present. Found only {self.file.keys()}\t{self.file}\n\n{self}") from e
        return ctd

    def open_ctd(self, nii: TPTBox.Image_Reference | None = None) -> TPTBox.POI:
        """Alias for :meth:`open_poi`; load the centroid JSON as a :class:`~TPTBox.POI`.

        Args:
            nii: Optional image reference for missing spatial metadata.

        Returns:
            The loaded :class:`~TPTBox.POI` object.
        """
        return self.open_poi(nii)

    def has_nii(self) -> bool:
        """Return ``True`` when at least one supported NIfTI extension is registered."""
        return any(a in self.file for a in _supported_nii_files)

    def open_nii(self) -> TPTBox.NII:
        """Load the NIfTI file into a :class:`~TPTBox.NII` object.

        Returns:
            The loaded :class:`~TPTBox.NII` volume.

        Raises:
            ValueError: If no NIfTI file (``nii.gz`` / ``nii``) is present.
        """
        try:
            from TPTBox import NII

            return NII.load_bids(self)
        except KeyError as e:
            raise ValueError(f"nii.gz not present. Found only {self.file.keys()}\t{self.file}\n\n{self}") from e

    def get_grid_info(self, add_grid_info_to_json: bool = True) -> Grid | None:
        """Return the spatial grid metadata for this file.

        Looks up the grid info in the associated JSON sidecar.  If the JSON
        does not contain grid information the NIfTI is loaded, the grid is
        computed, and the result is written back to the JSON when
        *add_grid_info_to_json* is ``True``.

        Args:
            add_grid_info_to_json: If ``True``, persist newly computed grid
                info to the JSON sidecar.

        Returns:
            A :class:`~TPTBox.core.nii_poi_abstract.Grid` instance, or
            ``None`` if no NIfTI file is present.
        """
        from TPTBox.core.dicom.dicom_extract import _add_grid_info_to_json
        from TPTBox.core.nii_poi_abstract import Grid

        nii_file = self.get_nii_file()
        if nii_file is None:
            return None
        if not self.has_json():
            self.file["json"] = Path(str(nii_file).split(".")[0] + ".json")
        return Grid(**_add_grid_info_to_json(nii_file, self.file["json"], add=add_grid_info_to_json)["grid"])

    def get_nii_file(self) -> Path | None:
        """Return the path to the first available NIfTI file.

        Checks the supported NIfTI extensions (``nii.gz``, ``nii``, ``mkd``)
        in order and returns the first one present.

        Returns:
            The :class:`~pathlib.Path` to the NIfTI file, or ``None`` if
            none of the supported extensions are registered.
        """
        for key in _supported_nii_files:
            if key in self.file:
                return self.file[key]
        return None

    def has_npz(self) -> bool:
        """Return ``True`` when an NPZ array file is registered for this entry."""
        return "npz" in self.file

    def open_npz(self) -> dict[str, np.ndarray]:
        """Load the associated NPZ file and return its contents as a dictionary.

        Returns:
            A dictionary mapping array names to :class:`numpy.ndarray` objects.

        Raises:
            KeyError: If no NPZ file is registered in :attr:`file`.
        """
        return dict(np.load(self.file["npz"], allow_pickle=False))  # type: ignore

    def open(self, filetype: str, _internal: bool = False) -> Path | TPTBox.NII | dict | None:
        """Open a file by extension, returning the appropriate Python object.

        .. deprecated::
            Use :meth:`open_nii`, :meth:`open_json`, or :attr:`file` directly.

        Args:
            filetype: Extension to open (e.g. ``"nii.gz"``, ``"json"``).
            _internal: Suppress the deprecation warning when called internally.

        Returns:
            * A :class:`~TPTBox.NII` for NIfTI extensions.
            * A :class:`dict` for JSON files.
            * A :class:`~pathlib.Path` for all other file types.
            * ``None`` if *filetype* is not registered in :attr:`file`.
        """
        if not _internal:
            warn("open is deprecated.", DeprecationWarning, stacklevel=2)
        if filetype not in self.file:
            return None
        if filetype == "json":
            return self.open_json()
        if filetype in _supported_nii_files:
            return self.open_nii()
        return self.file[filetype]

    def do_filter(
        self,
        key: str,
        constrain: list[str] | str | typing.Callable[[str | object], bool],
        required: bool = False,
    ) -> bool:
        """Check whether this file satisfies a key/constraint filter.

        If the *key* is not present in this file, the inverse of *required*
        is returned (i.e. absent keys pass when ``required=False`` and fail
        when ``required=True``).

        Args:
            key: The entity to inspect.  Special values: ``"format"`` checks
                :attr:`format`; ``"filetype"`` checks registered extensions;
                ``"parent"`` checks the parent folder; ``"self"`` passes the
                whole :class:`BIDS_FILE`.  Any other value is looked up in
                :attr:`info`, then in :attr:`file` (opening the file when a
                callable is given).
            constrain: Matching criterion — a string for exact match, a list
                of strings for membership, or a callable predicate.
            required: If ``True``, files for which *key* is absent are
                rejected.  Defaults to ``False``.

        Returns:
            ``True`` when the constraint is satisfied, ``False`` otherwise.
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
        """Return the interpolation order for this file (0 for masks/segmentations, 3 for images).

        Returns:
            int: interpolation_order
        """
        return 0 if self.format == "msk" or "label" in self.info else 3

    def get_frame_of_reference_uid(self, default: str | None = None) -> str | None:
        """Return a short hash identifying the world-space frame of reference.

        Reads ``FrameOfReferenceUID`` from the JSON sidecar and converts it
        to an 8-character base-36 string.  Falls back to the ``res`` or
        ``ses`` entity, and finally to *default*, when no JSON is available.

        Args:
            default: Value returned when the frame of reference cannot be
                determined.

        Returns:
            An 8-character base-36 hash string, a BIDS entity value, or
            *default*.
        """
        import hashlib

        length = 8
        try:
            uid = self.open_json()["FrameOfReferenceUID"]
        except Exception:
            return self.get("res", self.get("ses", default))
        # Hash the UID
        hash_bytes = hashlib.sha1(uid.encode("utf-8")).digest()
        # Convert to integer, then to base36
        num = int.from_bytes(hash_bytes, "big")
        base36 = ""
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        while num:
            num, i = divmod(num, 36)
            base36 = chars[i] + base36
        return base36[:length]

    def get_identifier(self, sequence_splitting_keys: list[str]) -> str:
        """Generates an identifier for the BIDS_FILE based on subject and splitting keys.

        Args:
            sequence_splitting_keys (list[str]): list of keys to use for splitting
        """
        if "sub" not in self.info:
            print(f"family_id, no sub-key, got {self.info}")
            identifier = "sub-404"
        else:
            identifier = "sub-" + self.info["sub"]
        for s in self.info.keys():
            if s in sequence_splitting_keys:
                identifier += "_" + s + "-" + self.info[s]
        return identifier


class Searchquery:
    """Query builder for filtering and retrieving BIDS files from a Subject_Container."""

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
    def from_BIDS_Family(cls, fam: BIDS_Family) -> Searchquery:
        """Construct a :class:`Searchquery` pre-filtered to a single :class:`BIDS_Family`.

        Args:
            fam: The family whose sequence bucket will be the sole candidate.

        Returns:
            A new :class:`Searchquery` containing only the sequence that
            produced *fam*.
        """
        dic = fam.data_dict
        any_file = dic[next(iter(dic.keys()))][0]
        sub = any_file.subject
        query = Searchquery(subj=sub, flatten=False)
        query._filter_fam_id(fam)
        # query.candidates = dic.copy()
        return query

    def _filter_fam_id(self, fam: BIDS_Family) -> None:
        """Restrict candidates to the single sequence bucket that produced *fam*."""
        self.unflatten()
        dic = fam.data_dict
        any_file = dic[next(iter(dic.keys()))][0]
        subject_id = any_file.subject.get_sequence_name(any_file)
        c = self.candidates
        self.candidates = {}
        self.candidates[subject_id] = c[subject_id]  # type: ignore

    def copy(self) -> Searchquery:
        """Return a shallow copy of this query with the same candidates.

        Returns:
            A new :class:`Searchquery` sharing the same subject and flatten
            mode, with an independent copy of the candidates collection.
        """
        copy = Searchquery(self.subject, self._flatten)
        copy.candidates = self.candidates.copy()
        return copy

    def flatten(self) -> None:
        """Transform from multi-file-mode (sequence buckets) to single-file-mode.

        After calling this method, :attr:`candidates` is a flat
        :class:`list` of :class:`BIDS_FILE` instances and
        :meth:`loop_list` can be used to iterate over them.
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

    def unflatten(self) -> None:
        """Transform from single-file-mode back to multi-file-mode (sequence buckets).

        Previously filtered files remain excluded.  After calling this method
        :meth:`loop_dict` can be used to iterate over sequence families.
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

    def filter_self(self, filter_fun: typing.Callable[[BIDS_FILE], bool], required: bool = True) -> None:
        """Filter candidates by applying *filter_fun* directly to each :class:`BIDS_FILE`.

        Args:
            filter_fun: Predicate receiving a :class:`BIDS_FILE`; return
                ``True`` to keep the file.
            required: Passed to :meth:`filter`.
        """
        return self.filter("self", filter_fun, required=required)  # type: ignore

    def filter_json(self, filter_fun: typing.Callable[[dict], bool], required: bool = True) -> None:
        """Filter candidates by applying *filter_fun* to the parsed JSON sidecar.

        Args:
            filter_fun: Predicate receiving the JSON dictionary; return
                ``True`` to keep the file.
            required: If ``True``, files without a JSON sidecar are removed.
        """
        return self.filter("json", filter_fun, required=required)  # type: ignore

    def filter(
        self,
        key: str,
        filter_fun: list[str] | str | typing.Callable[[str | object], bool],
        required: bool = True,
    ) -> None:
        """Remove candidates from the query that do not satisfy the filter.

        In flatten mode (single-file mode), individual files that do not pass
        are removed.  In unflatten mode (sequence mode), a sequence bucket is
        removed when *no* file in the bucket satisfies the filter.

        Args:
            key: The entity to inspect.  Special values: ``"format"``,
                ``"filetype"``, ``"parent"``, ``"self"``; otherwise looked up
                in :attr:`BIDS_FILE.info` or :attr:`BIDS_FILE.file`.
            filter_fun: Matching criterion — a string for exact match, a list
                of accepted strings, or a callable predicate.
            required: If ``True``, candidates where *key* is absent are
                removed.  If ``False``, absent keys are silently kept.
                Defaults to ``True``.
        """
        if self._flatten:
            assert isinstance(self.candidates, list)
            # list comprehension is O(n); the old copy()+list.remove() loop was O(n^2)
            self.candidates = [f for f in self.candidates if f.do_filter(key, filter_fun, required=required)]
        else:
            assert isinstance(self.candidates, dict)
            self.candidates = {
                seq: bids_files
                for seq, bids_files in self.candidates.items()
                if any(f.do_filter(key, filter_fun, required=required) for f in bids_files)
            }

    def filter_format(self, filter_fun: list[str] | str | typing.Callable[[str | object], bool]) -> None:
        """Keep only files whose format label satisfies *filter_fun*.

        Args:
            filter_fun: A format string, a list of accepted format strings,
                or a callable predicate applied to the format label.
        """
        if isinstance(filter_fun, list):
            return self.filter_format(lambda x: x in filter_fun)
        return self.filter("format", filter_fun=filter_fun, required=True)

    def filter_filetype(self, filter_fun: list[str] | str | typing.Callable[[str | object], bool], required: bool = True) -> None:
        """Keep only files that have a matching file extension.

        Args:
            filter_fun: An extension string (e.g. ``"nii.gz"``), a list of
                accepted extensions, or a callable predicate.
            required: If ``True``, files without any registered extension
                are removed.
        """
        return self.filter("filetype", filter_fun=filter_fun, required=required)

    def filter_non_existence(
        self,
        key: str,
        filter_fun: str | typing.Callable[[str | object], bool] = lambda x: True,  # noqa: ARG005
        required=True,
    ) -> None:
        """Remove family/file from the Searchquery if the filter condition is met.

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
            # list comprehension is O(n); the old copy()+list.remove() loop was O(n^2)
            self.candidates = [f for f in self.candidates if not f.do_filter(key, filter_fun, required=required)]
        else:
            assert isinstance(self.candidates, dict)
            self.candidates = {
                seq: bids_files
                for seq, bids_files in self.candidates.items()
                if not any(f.do_filter(key, filter_fun, required=required) for f in bids_files)
            }

    def filter_dixon_only_inphase(self) -> None:
        """Remove Dixon files that are fat, water, out-of-phase, or difference images.

        Retains only in-phase (or unlabelled) Dixon acquisitions by
        inspecting both the JSON ``ImageType`` field and the ``rec``,
        ``acq``, and ``part`` entities.
        """

        def json_filter(x: dict) -> bool:
            """Return True if JSON ImageType does not indicate a fat/water/outphase channel."""
            return "ImageType" not in x or (
                "W" not in x["ImageType"]
                and "F" not in x["ImageType"]
                and "FAT" not in x["ImageType"]
                and "WATER" not in x["ImageType"]
                and "OP" not in x["ImageType"]
            )

        def lam_filter(x: str) -> bool:
            """Return True if the entity value does not indicate a non-inphase Dixon channel."""
            return (
                x.upper() != "W"
                and x.upper() != "F"
                and x.upper() != "FAT"
                and x.upper() != "WATER"
                and x.upper() != "OP"
                and x.upper() != "OPP"
                and x.upper() != "OUTPHASE"
                and x.upper() != "DIFFERENCE"
            )

        self.filter_json(json_filter, required=False)
        self.filter("rec", lam_filter, required=False)  # type: ignore DEPRECATED
        self.filter("acq", lam_filter, required=False)  # type: ignore DEPRECATED
        self.filter("part", "inphase", required=False)  # type: ignore

    def filter_dixon_water(self, _keys: list[str] | None = None) -> None:
        """Keep only Dixon water-channel images (requires flatten mode).

        Args:
            _keys: Image-type labels that identify the water channel.
                Defaults to ``["W", "WATER"]``.
        """
        if _keys is None:
            _keys = ["W", "WATER"]
        assert self._flatten

        def json_filter(x: dict) -> bool:
            """Return True if JSON ImageType contains all required channel keys."""
            return "ImageType" not in x or all(k in x["ImageType"] for k in _keys)

        def lam_filter(x: str) -> bool:
            """Return True if the entity value matches one of the target channel keys."""
            return any(k == x.upper() for k in _keys)

        self.filter_json(json_filter, required=False)
        self.filter("rec", lam_filter, required=False)  # type: ignore
        self.filter("part", lam_filter, required=False)  # type: ignore

    def filter_dixon_fat(self) -> None:
        """Keep only Dixon fat-channel images (requires flatten mode)."""
        self.filter_dixon_water(_keys=["F", "FAT"])

    def filter_dixon_outphase(self) -> None:
        """Keep only Dixon out-of-phase images (requires flatten mode)."""
        self.filter_dixon_water(_keys=["OP", "OPP", "OUTPHASE"])

    def action(
        self,
        action_fun: typing.Callable[[BIDS_FILE], None],
        filter_fun: str | typing.Callable[[str | object], bool] = lambda x: True,  # noqa: ARG005
        key: str = "",
        required: bool = True,
        all_in_sequence: bool = False,
    ) -> None:
        """Apply *action_fun* to every candidate file that passes the filter.

        Args:
            action_fun: Callable invoked with each matching
                :class:`BIDS_FILE` as its sole argument.
            key: The entity to inspect for filtering (same semantics as in
                :meth:`filter`).  An empty string matches everything.
            filter_fun: Matching criterion — a string for exact match or a
                callable predicate.  Defaults to a predicate that always
                returns ``True`` (act on all candidates).
            required: If ``True``, candidates where *key* is absent are
                skipped.  Defaults to ``True``.
            all_in_sequence: In unflatten mode, when ``True`` and at least
                one file in a sequence bucket matches, *action_fun* is called
                on *every* file in that bucket.  Defaults to ``False``.

        Raises:
            AssertionError: If both :attr:`_flatten` and *all_in_sequence*
                are ``True`` (incompatible combination).
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
                    if any(bids_file.do_filter(key, filter_fun, required=required) for bids_file in bids_files):
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
        """Returns an iterator. Flatten must be True.

        Args:
            sort (bool, optional): Sort alphabetically. Defaults to False.

        Returns:
            typing.Iterator[BIDS_FILE]: _description_
        """
        assert isinstance(self.candidates, list), "call flatten() before looping as a list"
        if sort:
            return sorted(self.candidates.__iter__())  # type: ignore
        return self.candidates.__iter__()

    def loop_dict(
        self,
        sort=False,
        key_transform: typing.Callable[[BIDS_FILE], str | None] | None = None,
        key_addendum: list[str] | None = None,  # type: ignore
    ) -> typing.Iterator[BIDS_Family]:
        """Returns an iterator. Flatten must be False: it iterates over all families, where the return is the dict from the get_sequence_files function.

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
    """A group of related BIDS files sharing the same sequence-splitting key values."""

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
            raise KeyError(f"BIDS_Family does not contain key '{item}', only {list(self.keys())}; {self.family_id=}") from e

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

    def get_identifier(self) -> str:
        """Return the subject+sequence identifier string for this family.

        Delegates to :meth:`BIDS_FILE.get_identifier` on the first file in
        the family.

        Returns:
            A string like ``"sub-001_ses-01_sequ-303"`` derived from the
            sequence splitting keys.
        """
        first_e = self.data_dict[next(iter(self.data_dict.keys()))][0]
        return first_e.get_identifier(self.sequence_splitting_keys)

    def items(self) -> typing.ItemsView[str, list[BIDS_FILE]]:
        """Return ``(format_key, [BIDS_FILE, ...])`` pairs from the family dictionary."""
        return self.data_dict.items()

    def keys(self) -> typing.KeysView[str]:
        """Return the format keys present in this family."""
        return self.data_dict.keys()

    def sort(self) -> None:
        """Sort the family dictionary alphabetically by format key in-place."""
        self.data_dict = dict(sorted(self.data_dict.items()))

    def values(self) -> list[list[BIDS_FILE]]:
        """Return all lists of :class:`BIDS_FILE` objects in this family."""
        return list(self.data_dict.values())

    def new_query(self, flatten: bool = False) -> Searchquery:
        """Create a :class:`Searchquery` scoped to this family's sequence.

        Args:
            flatten: If ``True``, the query starts in single-file mode.

        Returns:
            A :class:`Searchquery` pre-filtered to this family's sequence
            bucket.
        """
        q = Searchquery.from_BIDS_Family(self)
        if flatten:
            q.flatten()
        return q

    def get_key_len(self) -> dict[str, int]:
        """Return the number of :class:`BIDS_FILE` instances for each format key.

        Returns:
            A dictionary mapping format key to the count of associated files.
        """
        return {k: len(v) for k, v in self}

    def get_format_len(self) -> dict[str, tuple[int, int]]:
        """Return per-base-format counts aggregated across all sub-keys.

        Groups family keys by their base format (the part before the first
        ``_``) and accumulates ``(key_count, file_count)`` tuples.

        Returns:
            A dictionary mapping base-format label to a
            ``(number_of_keys, total_file_count)`` tuple.
        """
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

    def get_files_with_multiples(self) -> dict[str, list[dict]]:
        """Return file-path dictionaries for all format keys that have more than one file.

        Returns:
            A dictionary mapping format key to a list of :attr:`BIDS_FILE.file`
            dicts, restricted to keys that hold more than one file.
        """
        return self.get_files(key=[k for k, v in self.get_key_len().items() if v > 1])

    def get_files(self, key: list[str] | str | None = None) -> dict[str, list[dict]]:
        """Return the raw file-path dictionaries for one or more format keys.

        Args:
            key: A single format key, a list of format keys, or ``None`` to
                include all keys.

        Returns:
            A dictionary mapping each requested format key to a list of
            :attr:`BIDS_FILE.file` dicts (one per associated
            :class:`BIDS_FILE`).
        """
        if key is None:
            key = [k for k, v in self]
        if isinstance(key, str):
            key = [key]
        family_dict_files: dict[str, list[dict]] = {}
        for k in key:
            family_dict_files[k] = [b.file for b in self[k]]
        return family_dict_files

    def get(self, item: str | list[str], default: list[BIDS_FILE] | None = None) -> list[BIDS_FILE] | None:
        """Return the list of :class:`BIDS_FILE` instances for the first matching key.

        When *item* is a list the method tries each element in order and
        returns the first hit.

        Args:
            item: A single format key or an ordered list of candidate keys.
            default: Value returned when none of the keys are found.

        Returns:
            The list of :class:`BIDS_FILE` instances, or *default*.
        """
        if not isinstance(item, list):
            item = [item]
        for i in item:
            if i in self:
                return self[i]
        return default

    def get_bids_files_as_dict(self, keys: list[str]) -> dict[str, BIDS_FILE]:
        """Checks each entry of the list, if everything is there, loads everything, one for each entry in keys.

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
