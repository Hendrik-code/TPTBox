from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Union

import torch
from torch.backends import cudnn

if TYPE_CHECKING:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# nnunetv2-2.5.2 or higher
@dataclass(slots=True)
class Config:
    """Config how the nnUnet is ran."""

    dataset_id: int
    out_base: Path = Path("/media/data/lisa/code/nnUnet")
    gpus: list[str] = field(default_factory=lambda: ["0"])
    big_model: bool = True
    planner: str | None = None
    num_epochs: int = 250
    num_iterations_per_epoch: int = 1000
    num_val_iterations_per_epoch: int = 50

    num_folds: int = 0
    start_fold: int = 0

    patch_size: tuple[int, ...] | None = None
    batch_size: int | None = None
    batch_size_val: int = 1

    gpu_memory_target: int | None = None
    overwrite_target_spacing: list[float] | None = None

    verify_dataset_integrity: bool = True
    preprocess: bool | None = None
    compress = True
    debug: bool = False
    configurations: list[str] = field(
        default_factory=lambda: ["3d_fullres"]
    )  # default=["2d", "3d_fullres", "3d_lowres","3d_cascade_fullres"],
    num_processes: tuple[int] = (4,)  # [32] # [8, 4, 8]
    verbose = False

    @property
    def plans(self) -> str:
        """Return plan name."""
        if self.planner is not None:
            return self.planner
        return "nnUNetPlannerResEncL" if self.big_model else "nnUNetPlannerResEncM"

    @property
    def dataset_folder(self) -> str:
        """Get dataset folder name."""
        return f"Dataset{self.dataset_id:03}"

    @property
    def single_gpu(self) -> bool:
        """Test if this is single GPU."""
        return len(self.gpus) == 1


def _run_training_highjack(self: nnUNetTrainer) -> None:
    self.on_train_start()

    for epoch in range(self.current_epoch, self.num_epochs):
        t = time()
        self.on_epoch_start()

        self.on_train_epoch_start()
        train_outputs = []

        for batch_id in range(self.num_iterations_per_epoch):
            x = time() - t
            print(
                f"{epoch}:{batch_id:04}/{self.num_iterations_per_epoch:04}",
                " time:",
                str(timedelta(seconds=x)),
                "ETA:",
                str(timedelta(seconds=x / (max(1, batch_id)) * self.num_iterations_per_epoch)),
                end="\r",
            )
            train_outputs.append(self.train_step(next(self.dataloader_train)))  # type: ignore
        print(f"{epoch}:{batch_id:05}/{self.num_iterations_per_epoch:05}", " time", str(timedelta(seconds=time() - t)))
        self.on_train_epoch_end(train_outputs)
        torch.cuda.empty_cache()
        t = time()
        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):
                print(f"{batch_id:05}/{self.num_val_iterations_per_epoch:05}", " time", str(timedelta(seconds=time() - t)), end="\r")

                val_outputs.append(self.validation_step(next(self.dataloader_val)))  # type: ignore
            self.on_validation_epoch_end(val_outputs)

        self.on_epoch_end()
        l = list(self.dataset_json["labels"].keys())
        self.print_to_log_file(
            "Dice",
            ", ".join([f"{l[e]}:{i:.3f}" for e, i in enumerate(self.logger.my_fantastic_logging["dice_per_class_or_region"][-1], 1)]),
        )
    self.on_train_end()


def _run_training(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    pretrained_weights: str | None = None,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: torch.device = torch.device("cuda"),  # noqa: B008
    num_iterations_per_epoch=250,
    num_val_iterations_per_epoch=50,  # 50
    num_epochs=250,  # 1000
    oversample_foreground_percent=0.33,
    current_epoch=0,
    enable_deep_supervision=True,
    save_every=1,  # 50
):

    from nnunetv2.run.run_training import get_trainer_from_args, join, maybe_load_checkpoint

    if plans_identifier == "nnUNetPlans":
        print(
            "\n############################\n"
            "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
            "Please consider using those instead! "
            "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
            "\n############################\n"
        )
    if isinstance(fold, str) and fold != "all":
        try:
            fold = int(fold)
        except ValueError:
            print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
            raise

    if val_with_best:
        assert not disable_checkpointing, "--val_best is not compatible with --disable_checkpointing"

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name, plans_identifier, device=device)

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name, plans_identifier, device=device)
    nnunet_trainer.oversample_foreground_percent = oversample_foreground_percent
    nnunet_trainer.num_val_iterations_per_epoch = num_val_iterations_per_epoch
    nnunet_trainer.num_epochs = num_epochs
    nnunet_trainer.current_epoch = current_epoch
    nnunet_trainer.num_iterations_per_epoch = num_iterations_per_epoch
    nnunet_trainer.enable_deep_supervision = enable_deep_supervision  # type: ignore
    nnunet_trainer.save_every = save_every
    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (continue_training and only_run_validation), "Cannot set --c and --val flag at the same time. Dummy."

    maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not only_run_validation:
        _run_training_highjack(nnunet_trainer)

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
    nnunet_trainer.perform_actual_validation(export_validation_probabilities)


class NNUNetRunner:
    """Runs the nnUnet training."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ----------------------------------------------------------
    # Environment
    # ----------------------------------------------------------

    def _setup_environment(self):
        os.environ["nnUNet_raw"] = str(self.cfg.out_base / "nnUNet_raw")  # noqa: SIM112
        os.environ["nnUNet_preprocessed"] = str(self.cfg.out_base / "nnUNet_preprocessed")  # noqa: SIM112
        os.environ["nnUNet_results"] = str(self.cfg.out_base / "nnUNet_results")  # noqa: SIM112
        os.environ["nnUNet_n_proc_DA"] = "40"  # noqa: SIM112

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------

    def _load_dataset_json(self) -> dict:
        ds_file = self.cfg.out_base / "nnUNet_raw" / self.cfg.dataset_folder / "dataset.json"
        with open(ds_file) as f:
            return json.load(f)

    # ----------------------------------------------------------
    # Planning
    # ----------------------------------------------------------

    def _preprocess(self):
        if self.cfg.preprocess is None:
            plan_file = self.cfg.out_base / "nnUNet_preprocessed" / self.cfg.dataset_folder / f"{self.cfg.plans}.json"
            self.cfg.preprocess = not (plan_file).exists()
        if not self.cfg.preprocess:
            return

        import nnunetv2.experiment_planning.plan_and_preprocess_api as pp

        print("Extract_fingerprints...")
        pp.extract_fingerprints([self.cfg.dataset_id], "DatasetFingerprintExtractor", 8, self.cfg.verify_dataset_integrity, False, False)
        print("Plan Experiments...")
        pp.plan_experiments(
            [self.cfg.dataset_id],
            self.cfg.plans,
            self.cfg.gpu_memory_target,
            "DefaultPreprocessor",
            self.cfg.overwrite_target_spacing,
            self.cfg.plans,
        )
        print("Preprocessing...")

        from TPTBox.core.internal.train_nnUnet.fastProcessor import preprocess

        preprocess(
            [self.cfg.dataset_id], self.cfg.plans, self.cfg.configurations, self.cfg.num_processes, self.cfg.compress, self.cfg.verbose
        )

    # ----------------------------------------------------------
    # Plans patching
    # ----------------------------------------------------------

    def _patch_plans(self):
        plan_file = self.cfg.out_base / "nnUNet_preprocessed" / self.cfg.dataset_folder / f"{self.cfg.plans}.json"

        if not plan_file.exists():
            return

        with open(plan_file) as f:
            plans = json.load(f)

        changed = False

        if self.cfg.patch_size is not None:
            plans["configurations"]["3d_fullres"]["patch_size"] = list(self.cfg.patch_size)

            changed = True

        if self.cfg.batch_size is not None:
            plans["configurations"]["3d_fullres"]["batch_size"] = self.cfg.batch_size

            changed = True

        if changed:
            with open(plan_file, "w") as f:
                json.dump(plans, f, indent=2)

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def _train_fold(self, fold: int | str):

        # from nnunetv2.run.run_training import run_training

        print(f"Training fold {fold}")

        best_checkpoints = list(
            Path(self.cfg.out_base / "nnUNet_results").glob(f"Dataset{self.cfg.dataset_id:03}*/*_3d_full*/fold_{fold}/checkpoint_best.pth")
        )
        print("existing Trainigs: ", best_checkpoints)
        _run_training(
            dataset_name_or_id=self.cfg.dataset_folder,
            configuration="3d_fullres",
            fold=fold,
            trainer_class_name="nnUNetTrainer",
            plans_identifier=self.cfg.plans,
            num_iterations_per_epoch=self.cfg.num_iterations_per_epoch,
            num_epochs=self.cfg.num_epochs,
            continue_training=len(best_checkpoints) != 0,
        )

    # ----------------------------------------------------------
    # Multiprocessing
    # ----------------------------------------------------------

    def _train(self):

        if self.cfg.single_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpus[0]

            if self.cfg.num_folds == 0:
                self._train_fold("all")
            else:
                for fold in range(self.cfg.start_fold, self.cfg.num_folds):
                    self._train_fold(fold)
            return

        folds = list(
            range(
                self.cfg.start_fold,
                max(self.cfg.num_folds, len(self.cfg.gpus)),
            )
        )

        def worker(args):
            fold, gpu = args

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu

            self._train_fold(fold)

        jobs = [(fold, self.cfg.gpus[i % len(self.cfg.gpus)]) for i, fold in enumerate(folds)]

        with Pool(len(self.cfg.gpus)) as p:
            p.map(worker, jobs)

    # ----------------------------------------------------------
    # Main
    # ----------------------------------------------------------

    def run(self) -> None:
        """Starts and runs the training."""
        self._setup_environment()

        ds = self._load_dataset_json()

        self.cfg.overwrite_target_spacing = ds.get("spacing", self.cfg.overwrite_target_spacing)

        self._preprocess()

        self._patch_plans()

        self._train()


def parse_args() -> Config:
    """Arg parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-id", "-id", required=True, type=int)
    parser.add_argument("--base", "-b", required=True, type=str)
    parser.add_argument("--gpu", nargs="+", default=["0"])
    parser.add_argument("--epochs", "-e", type=int, default=250)
    parser.add_argument("--epoch-len", "-el", type=int, default=1000)
    parser.add_argument("--planner", default=None)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--num-folds", type=int, default=0)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--patch-size", nargs="+", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--force-preprocessing", action="store_true")
    parser.add_argument("--num_processes", type=int, default=4)  # 32 on server

    args = parser.parse_args()
    preprocess = None
    if args.skip_preprocessing:
        preprocess = False
    if args.force_preprocessing:
        preprocess = True
    return Config(
        out_base=Path(args.base),
        dataset_id=args.dataset_id,
        gpus=args.gpu,
        big_model=not args.small,
        planner=args.planner,
        num_epochs=args.epochs,
        num_folds=args.num_folds,
        start_fold=args.start_fold,
        num_iterations_per_epoch=args.epoch_len,
        patch_size=tuple(args.patch_size) if args.patch_size else None,
        batch_size=args.batch_size,
        preprocess=preprocess,
        num_processes=(args.num_processes,),
    )


def main() -> None:
    """Main."""
    cfg = parse_args()

    runner = NNUNetRunner(cfg)

    runner.run()


if __name__ == "__main__":
    main()
    # /media/data/anaconda3/envs/py3.12/bin/python /media/data/lisa/code/scripts/train_nnUnet/train.py -id 7 --gpu 0 -e 300 -el 1000 --force-preprocessing --num_processes 32
