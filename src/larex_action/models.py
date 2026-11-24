from pathlib import Path
from larex_actions import ParameterChoice
from calamari_ocr.utils import glob_all
import os

CHECKPOINT = "/storage/workspace/shared/project/textploring/models/"\
             + "calamari_tf-keras_nuernberg1/*.ckpt.json"
CHECKPOINT = os.getenv("CALAMARI_DEFAULT_MODEL", "").strip()
MODEL_DIRECTORY = os.getenv("CALAMARI_MODEL_DIRECTORY", "").strip()


def discover_calamari_ocr_models(
    model_directory: str | Path | None = MODEL_DIRECTORY, default=CHECKPOINT
) -> list[ParameterChoice]:
    choices = [ParameterChoice(value=default, label="Processor default",
                               default=True)]
    roots = []
    if model_directory:
        roots.append(Path(model_directory).expanduser())
    discovered: dict[Path, str] = {}
    for configured_root in roots:
        try:
            root = configured_root.resolve(strict=True)
        except OSError:
            continue
        if not root.is_dir():
            continue
        candidates = set(c.parent for c in root.rglob("*.ckpt.json"))
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_dir() or not resolved.is_relative_to(root):
                continue
            if resolved in discovered:
                continue
            relative = resolved.relative_to(root).as_posix()
            discovered[resolved / "*.ckpt.json"] = f"{relative} "\
                                                   + f"({root.name})"[:256]
    for path, label in sorted(discovered.items(),
                              key=lambda item: (item[1].casefold(),
                                                str(item[0])))[:999]:
        choices.append(ParameterChoice(value=str(path), label=label))
    return choices


def rglob(d):
    # recursively scan Path d for dirs not containing *.ckpt*
    if list(d.glob("*.ckpt*")):
        return []
    res = [d]
    for sd in d.glob("*/"):
        res += rglob(sd)
    return res


def discover_model_dirs(
    model_directory: str | Path | None = MODEL_DIRECTORY
) -> list[ParameterChoice]:
    print("default:", model_directory)
    mdir = Path(model_directory).expanduser().resolve(strict=True)
    assert mdir.is_dir()
    choices = [ParameterChoice(value=str(mdir), label="/",
                               default=True)]
    subdirs = sorted(list(d for d in set(rglob(mdir)) if d != mdir))
    for sd in subdirs:
        relative = sd.relative_to(mdir).as_posix()
        choices.append(ParameterChoice(value=str(sd), label="/" + relative))
    return choices


def ocr_model_from_input(action_input) -> str:
    parameters = getattr(action_input, "parameters", {}) or {}
    value = parameters.get("ocrModel", "")
    if not isinstance(value, str):
        raise TypeError("ocrModel must be a string.")
    return value.strip() or CHECKPOINT


def eval_model_from_input(action_input, single=True, none=None) -> str:
    parameters = getattr(action_input, "parameters", {}) or {}
    value = parameters.get("ocrModel", "none")
    if value == "none":
        return none
    else:
        mdls = glob_all(value)
        if not mdls:
            raise ValueError("Could not find any model for path »"+value+"«.")
        else:
            if single:
                return mdls[0]
            else:
                return mdls


def best_models_dir_from_input(action_input) -> str:
    parameters = getattr(action_input, "parameters", {}) or {}
    mdir = parameters.get("modelDir", MODEL_DIRECTORY)
    mname = parameters.get("modelName", "model")
    mpath = Path(mdir).expanduser().resolve(strict=True)
    bestpath = mpath / mname

    def is_valid(p):
        if p.parent != mpath:
            return False
        if len(list(p.glob("*"))) != len(list(p.glob("*.ckpt*"))):
            return False
        return True
    if is_valid(bestpath):
        return bestpath.as_posix()
    else:
        return None
