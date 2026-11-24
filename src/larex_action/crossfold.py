from calamari_ocr.ocr.training.cross_fold_trainer import (
        CrossFold,
        SavedCalamariModel,
        CrossFoldTrainerParams,
        CrossFoldTrainer,
        )
from tfaip.util.tfaipargparse import post_init
import multiprocessing
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import sys
import json
import logging
from copy import deepcopy
from calamari_ocr.ocr.training.params import (
    CalamariDefaultTrainerPipelineParams,
)
from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessorParams
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.utils.multiprocessing import prefix_run_command, run
from models import eval_model_from_input, best_models_dir_from_input


def train_individual_model(run_args):
    # same as calamari, just added the "canceled" event
    args = run_args["args"]
    train_args_json = run_args["json"]
    fold_logger = logging.getLogger(f"FOLD {run_args['id']}")
    for handler in fold_logger.handlers:
        handler.terminator = ""

    for out, err in run(
        prefix_run_command(
            [
                sys.executable,
                "-u",
                run_args["train_script"],
                train_args_json,
            ],
            run_args.get("run", None),
            {"threads": run_args.get("num_threads", -1)},
        ),
        verbose=run_args.get("verbose", False),
    ):
        # Print the output of the thread
        if run_args.get("verbose", False):
            if out:
                fold_logger.info(out.rstrip("\n"))
            if err:
                fold_logger.info(err.rstrip("\n"))
        canceled = run_args.get("canceled", None)
        if canceled is not None and canceled.is_set():
            return args

    return args


def cf_params_from_input(action_input, traindata) -> CrossFoldTrainerParams:
    parameters = getattr(action_input, "parameters", {}) or {}
    train_dict = CrossFoldTrainerParams().to_dict()
    for k, v in parameters.items():
        dct = train_dict
        ks = k.split(".")
        for subkey in ks[:-1]:
            if subkey in dct:
                dct = dct[subkey]
        if ks[-1] in dct:
            dct[ks[-1]] = v
    cfparams = CrossFoldTrainerParams.from_dict(train_dict)
    n_augmentations = parameters.get("n_augmentations", 0)
    if n_augmentations > 0:
        for dp in cfparams.trainer.scenario.data.pre_proc.processors_of_type(AugmentationProcessorParams):
            dp.n_augmentations = n_augmentations
    weights = eval_model_from_input(action_input, single=False, none=[])
    devices = [int(i) for i in os.getenv("CUDA_VISIBLE_DEVICES", "").strip().split(",")]
    cfparams.trainer.device.gpus = devices
    if weights and cfparams.trainer.network is None:
        with open(weights[0]) as f:
            wdict = json.load(f)
        cfparams.trainer.network = wdict.get("network", None)
    cfparams.weights = weights
    cfparams.best_models_dir = best_models_dir_from_input(action_input)
    cfparams.trainer.gen.train = traindata
    cfparams.trainer.scenario.data.__post_init__()
    cfparams.trainer.scenario.__post_init__()
    cfparams.trainer.__post_init__()
    return cfparams


class LarexCrossFoldTrainer(CrossFoldTrainer):
    async def run(self, ctx):
        temporary_dir = tempfile.mkdtemp(prefix="larex-action-calamari")
        fold_file = os.path.join(temporary_dir, "folds.json")
        cross_fold = CrossFold(
            n_folds=self.params.n_folds,
            data_generator_params=self.params.trainer.gen.train,
            output_dir=temporary_dir,
        )
        await ctx.heartbeat(0, "Preparing training data", raise_on_cancel=True)
        cross_fold.create_folds(self.params.trainer.progress_bar)
        cross_fold.write_folds_to_json(fold_file)

        # Create the json argument file for each individual training
        run_args = []
        seed = self.params.trainer.random_seed or -1
        folds_to_run = self.params.single_fold if len(self.params.single_fold) > 0 else range(len(cross_fold.folds))
        await ctx.heartbeat(0, "Preparing folds", raise_on_cancel=True)
        for fold in folds_to_run:
            train_files = cross_fold.train_files(fold)
            test_files = cross_fold.test_files(fold)
            path = os.path.join(temporary_dir, "fold_{}.json".format(fold))
            with open(path, "w") as f:
                trainer_params = deepcopy(self.params.trainer)
                trainer_params.gen = CalamariDefaultTrainerPipelineParams(
                    train=trainer_params.gen.train,
                    val=deepcopy(trainer_params.gen.train),
                    setup=trainer_params.gen.setup,
                )
                tp = trainer_params.gen.train.to_dict()
                del tp["__cls__"]
                tp["files"] = train_files
                trainer_params.gen.train = Hdf5.from_dict(tp)
                vp = trainer_params.gen.val.to_dict()
                del vp["__cls__"]
                vp["files"] = test_files
                trainer_params.gen.val = Hdf5.from_dict(vp)

                trainer_params.scenario.id = fold
                trainer_params.progress_bar_mode = 2
                trainer_params.output_dir = os.path.join(temporary_dir, "fold_{}".format(fold))
                trainer_params.early_stopping.best_model_output_dir = self.params.best_models_dir
                trainer_params.early_stopping.best_model_name = ""
                best_model_prefix = self.params.best_model_label.format(id=fold)
                trainer_params.best_model_prefix = best_model_prefix

                if self.params.visible_gpus:
                    assert trainer_params.device.gpus is None, "Using visible_gpus with device.gpus is not supported"
                    trainer_params.device.gpus = [self.params.visible_gpus[fold % len(self.params.visible_gpus)]]

                if seed >= 0:
                    trainer_params.random_seed = seed + fold

                if len(self.params.weights) == 1:
                    trainer_params.warmstart.model = self.params.weights[0]
                elif len(self.params.weights) > 1:
                    trainer_params.warmstart.model = self.params.weights[fold]

                # start from scratch via None
                if trainer_params.warmstart.model:
                    if (
                        len(trainer_params.warmstart.model.strip()) == 0
                        or trainer_params.warmstart.model.upper() == "NONE"
                    ):
                        trainer_params.warmstart.model = None
                    else:
                        # access model once to upgrade the model if necessary
                        # (can not be performed in parallel if multiple folds use the same model)
                        SavedCalamariModel(trainer_params.warmstart.model)

                post_init(trainer_params)

                json.dump(
                    trainer_params.to_dict(),
                    f,
                    indent=4,
                )

            run_args.append(
                {
                    "json": path,
                    "args": trainer_params,
                    "id": fold,
                    "train_script": self.train_script_path,
                    "run": self.params.run_train,
                    "verbose": True,
                }
            )
        
        await ctx.heartbeat(10, "Training", raise_on_cancel=True)

        def _cleanup():
            if not self.params.keep_temporary_files:
                import shutil
                shutil.rmtree(temporary_dir)
        
        with multiprocessing.Manager() as manager:
            canceled = manager.Event()
            for arg in run_args:
                arg["canceled"] = canceled
            with ThreadPoolExecutor(max_workers=self.params.max_parallel_models) as exc:
                futures = [exc.submit(train_individual_model, run_arg) for run_arg in run_args]
                try:
                    while True:
                        status = [f.done() for f in futures]
                        if all(status):
                            _cleanup()
                            await ctx.complete(message="Training process finished")
                            return
                        perc = 10 + int(.9 * len([x for x in status if x]) / len(status))
                        try:
                            await ctx.heartbeat(progress_percent=perc,
                                                status_message="Training in progress",
                                                raise_on_cancel=True)
                        except Exception as e:
                            for f in futures:
                                f.cancel()
                            canceled.set()
                            _cleanup()
                            raise e
                        sleep(10)
                except KeyboardInterrupt as e:
                    ctx.heartbeat(status_message="Training interrupted.",
                                  status="failed")
                    canceled.set()
                    _cleanup()
                    raise e
