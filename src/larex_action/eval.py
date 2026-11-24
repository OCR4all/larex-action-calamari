from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.evaluator import Evaluator, EvaluatorParams
from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.scenario import CalamariScenario

from reader import params_from_input
from models import (eval_model_from_input,
                    discover_calamari_ocr_models)
from functools import partial
import tfaip.util.logging
import os
from copy import deepcopy
from paiargparse.dataclass_extractor import extract_args_of_dataclass

from larex_actions import (ActionContext,
                           EvaluationMetric,
                           EvaluationReport,
                           EvaluationSample,
                           EvaluationTable,
                           EvaluationTableColumn,
                           EvaluationTableRow)
from larex_actions.fastapi import create_larex_action_app
import re
import logging

logger = tfaip.util.logging.logger(__name__)

PROCESSOR_ID = os.getenv("LAREX_PROCESSOR_ID", "calamari-eval").strip()
DISPATCH_SECRET_ENV = "LAREX_DISPATCH_HMAC_SECRET"
PROGRESS_COMPLETE_BEFORE_UPLOAD = 95


def configure_sdk_transport_logging() -> None:
    enabled = os.getenv("LAREX_SDK_TRANSPORT_LOGGING", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    sdk_logger = logging.getLogger("larex_actions.transport")
    sdk_logger.setLevel(logging.DEBUG)
    if not sdk_logger.hasHandlers():
        sdk_logger.addHandler(logging.StreamHandler())
        sdk_logger.propagate = False


configure_sdk_transport_logging()


def eval_params_from_input(action_input) -> EvaluatorParams:
    parameters = getattr(action_input, "parameters", {}) or {}
    eval_args = extract_args_of_dataclass(EvaluatorParams)
    eval_dict = {arg.name: parameters.get(arg.name, arg.default)
                 for arg in eval_args}
    return EvaluatorParams.from_dict(eval_dict)


def page_progress(completed_pages: int, total_pages: int) -> int:
    return int(completed_pages / total_pages * PROGRESS_COMPLETE_BEFORE_UPLOAD)


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return stem[:96] or "page"


def create_tables(r):
    gt_files = r["ids"]
    # all files
    sorted_lines = sorted(zip(r["single"], gt_files), key=lambda a: -a[0][1])
    samples = []
    for (len_gt, errs, sync_errs, confusion, (gt, pred)), gt_file in sorted_lines:
        p_id, l_id = gt_file.split("/", 1)
        s = EvaluationSample(
            id=gt_file,
            inputId=p_id,
            targetId=l_id,
            fields={"reference": gt.strip(),
                    "prediction": pred.strip(),
                    "confusions": "{}".format(confusion)},
            metrics=[
                EvaluationMetric(
                    key="len", label="LEN", value=len_gt,
                    format="INTEGER", direction="NEUTRAL"),
                EvaluationMetric(
                    key="err", label="ERR", value=errs,
                    format="INTEGER", direction="LOWER_IS_BETTER"),
                EvaluationMetric(
                    key="cer", label="CER", value=errs / max(len(gt), len(pred), 1),
                    format="PERCENT", direction="LOWER_IS_BETTER"),
                EvaluationMetric(
                    key="rel_err", label="REL. ERR", value=errs / r["total_char_errs"] if r["total_char_errs"] > 0 else 0,
                    format="NUMBER", direction="LOWER_IS_BETTER"),
                EvaluationMetric(
                    key="sync_err", label="SYNC ERR", value=sync_errs,
                    format="INTEGER", direction="LOWER_IS_BETTER"),
                ]
            )
        samples.append(s)

    # total confusions
    columns = [
            EvaluationTableColumn(key="gt", label="GT", type="STRING"),
            EvaluationTableColumn(key="pred", label="PRED", type="STRING"),
            EvaluationTableColumn(key="count", label="COUNT", type="INTEGER"),
            EvaluationTableColumn(key="pcer", label="PERC (CER)", type="NUMBER"),
            ]

    keys = sorted(r["confusion"].items(), key=lambda item: -item[1])
    rows = [EvaluationTableRow(values={"gt": "{" + gt + "}",
                                       "pred": "{" + pred + "}",
                                       "count": count,
                                       "pcer": count * max(len(gt), len(pred)) / r["total_sync_errs"]
                                       })
            for i, ((gt, pred), count) in enumerate(keys)
            ]
    conftbl = EvaluationTable(key="confusions",
                              title="Confusions",
                              columns=columns,
                              rows=rows,
                              )
    return samples, [conftbl]


async def process_run(ctx: ActionContext) -> None:
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.complete(
            ctx.result_builder(), "Calamari OCR received no pages."
        )
        return
    xml_bytes = {page.id: await ctx.download_bytes(page.xml[0])
                 for page in action_input.pages}
    httpxclient = ctx.client._client
    ctx.client._client = None
    ctx_clientless = deepcopy(ctx)
    ctx.client._client = httpxclient
    await ctx.check_cancelled()
    checkpoint = eval_model_from_input(action_input)
    if checkpoint is not None:
        saved_model = SavedCalamariModel(checkpoint, auto_update=True)
        trainer_params = CalamariScenario.trainer_cls().params_cls().from_dict(saved_model.dict)
        data_params = trainer_params.scenario.data
    else:
        data_params = Data.default_params()
    data = Data(data_params)

    pred_data = params_from_input(ctx_clientless, action_input,
                                  xml_bytes=xml_bytes)
    gt_data = params_from_input(ctx_clientless, action_input,
                                gt=True, xml_bytes=xml_bytes)
    eval_params = eval_params_from_input(action_input)
    evaluator = Evaluator(eval_params, data=data)
    # evaluator.preload_gt(gt_dataset=gt_data)
    r = evaluator.run(gt_dataset=gt_data, pred_dataset=pred_data)
    await ctx.check_cancelled()
    print(
        "Got mean normalized label error rate of {:.2%} ({} errs, {} total chars, {} sync errs)".format(
            r["avg_ler"], r["total_char_errs"], r["total_chars"], r["total_sync_errs"]
        )
    )
    samples, tables = create_tables(r)
    report = EvaluationReport(
        profile="larex.ocr-recognition",
        profileVersion=1,
        title="Recognition quality",
        summary=[EvaluationMetric(
            key="cer", label="CER", value=r["avg_ler"],
            format="PERCENT", direction="LOWER_IS_BETTER"),
            EvaluationMetric(
            key="total_char_errs", label="errors", value=r["total_char_errs"],
            format="NUMBER", direction="LOWER_IS_BETTER"),
            EvaluationMetric(
            key="total_chars", label="characters", value=r["total_chars"],
            format="NUMBER", direction="NEUTRAL"),
            EvaluationMetric(
            key="total_sync_errs", label="sync errors", value=r["total_sync_errs"],
            format="NUMBER", direction="LOWER_IS_BETTER"),
                 ],
        tables=tables,
        samples=samples
    )
    await ctx.complete_evaluation(report, "Evaluation complete")


app = create_larex_action_app(
    processor_id=PROCESSOR_ID,
    dispatch_secret_env=DISPATCH_SECRET_ENV,
    handler=process_run,
    processor_capabilities={"evaluationReports": True},
    parameter_value_providers={
        "calamariOCRModels": partial(discover_calamari_ocr_models, default="none"),
    },
)
