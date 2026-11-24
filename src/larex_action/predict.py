from calamari_ocr.ocr.predict.predictor import MultiPredictor
from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.ocr.voting import VoterParams
from calamari_ocr.utils import glob_all
from reader import params_from_input
from models import discover_calamari_ocr_models, ocr_model_from_input, CHECKPOINT
import tfaip.util.logging
import os
from copy import deepcopy

from larex_actions import ActionContext, ActionTargetSelection
from larex_actions.fastapi import create_larex_action_app
import re
import logging

logger = tfaip.util.logging.logger(__name__)

MODEL_DIRECTORY = os.getenv("CALAMARI_MODEL_DIRECTORY", "").strip()
PROCESSOR_ID = os.getenv("LAREX_PROCESSOR_ID", "calamari-ocr").strip()
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


def page_progress(completed_pages: int, total_pages: int) -> int:
    return int(completed_pages / total_pages * PROGRESS_COMPLETE_BEFORE_UPLOAD)


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return stem[:96] or "page"


def result_message(xml_count: int) -> str:
    return f"Calamari OCR produced {xml_count} PAGE XML file(s)."


def ats_to_dict(ats: ActionTargetSelection | None) -> dict:
    # transform ats to dict page_id -> (type, ids)
    if ats.type == "REGION":
        return {p.page_id: (ats.type, p.region_ids) for p in ats.pages}
    if ats.type == "TEXT_LINE":
        return {p.page_id: (ats.type, p.text_line_ids) for p in ats.pages}
    return {}


class Predictor:
    def __init__(self, checkpoint=CHECKPOINT):
        self.loaded_model = None
        self.load_model(checkpoint)

    def load_model(self, checkpoint):
        if checkpoint == self.loaded_model:
            return
        self.loaded_model = checkpoint
        if not checkpoint.endswith(".json"):
            checkpoint += ".json"
        checkpoint = glob_all(checkpoint)
        checkpoint = [cp[:-5] for cp in checkpoint]

        self.predictor = MultiPredictor.from_paths(
            checkpoints=checkpoint,
            voter_params=VoterParams(),
            predictor_params=PredictorParams(),
        )

    def predict(self, ctx: ActionContext, action_input):
        httpxclient = ctx.client._client
        ctx.client._client = None
        ctx_clientless = deepcopy(ctx)
        ctx.client._client = httpxclient

        data = params_from_input(ctx_clientless, action_input)
        self.predictor.data._pipelines.clear()
        do_prediction = self.predictor.predict(data)
        pipeline = self.predictor.data.get_or_create_pipeline(
            self.predictor.params.pipeline, data
        )
        reader = pipeline.reader()
        if len(reader) == 0:
            logger.info("Empty dataset.")
            return []

        avg_sentence_confidence = 0
        n_predictions = 0

        reader.prepare_store()

        samples = []
        sentences = []
        for s in do_prediction:
            _, (_, prediction), meta = s.inputs, s.outputs, s.meta
            sample = reader.sample_by_id(meta["id"])
            n_predictions += 1
            sentence = prediction.sentence

            avg_sentence_confidence += prediction.avg_char_probability

            samples.append(sample)
            sentences.append(sentence)
            reader.store_text_prediction(prediction, meta["id"],
                                         output_dir=None)
            cur_res = reader.result_pop()
            if cur_res is not None:
                yield cur_res

        logger.info(
            "Average sentence confidence: {:.2%}".format(
                avg_sentence_confidence / n_predictions
            )
        )
        reader.store()
        cur_res = reader.result_pop()
        if cur_res is not None:
            yield cur_res

        logger.info("All prediction files written")


async def process_run(ctx: ActionContext) -> None:
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.complete(
            ctx.result_builder(), "Calamari OCR received no pages."
        )
        return
    predictor.load_model(ocr_model_from_input(action_input))

    total = len(action_input.pages)
    xml_count = 0

    await ctx.heartbeat(
        page_progress(0, total),
        f"Predicting page 1/{total}: {action_input.pages[0].name}",
        raise_on_cancel=True,
    )
    for page_id, content in predictor.predict(ctx, action_input):
        await ctx.check_cancelled()
        page = [p for p in action_input.pages if p.id == page_id][0]
        index = action_input.pages.index(page) + 1
        while index - 1 > xml_count:
            logger.info(f"Skipping page {xml_count + 1} of {total}")
            await ctx.submit_page_results(
                action_input.pages[xml_count].id,
                page_status="SKIPPED",
                reason="No segmented text lines found."
                    )
            xml_count += 1

        results = ctx.result_builder()
        results.add_xml_bytes(
            page_id=page.id,
            content=content,
            file_name=f"{safe_stem(page.name or page.id)}.xml",
        )
        await ctx.submit_page_results(
                page.id,
                results,
                f"Finished OCR for page {index}/{total}",
        )
        xml_count += 1

        if index < total:
            logger.info("Waiting for calamari results "
                        + f"(page {index+1}/{total})")
            await ctx.heartbeat(
                page_progress(index, total),
                f"Predicting page {index + 1}/{total}: "
                + "{action_input.pages[index].name}",
                raise_on_cancel=True,
            )
    while total > xml_count:
        logger.info(f"Skipping page {xml_count + 1} of {total}")
        await ctx.submit_page_results(
            action_input.pages[xml_count].id,
            ctx.result_builder(),
            "Skipped: no segmented text lines found."
            )
        xml_count += 1
    logger.info(f"All resulting pages (total: {xml_count}) sent.")
    await ctx.complete(message=result_message(total))


predictor = Predictor()
app = create_larex_action_app(
    processor_id=PROCESSOR_ID,
    dispatch_secret_env=DISPATCH_SECRET_ENV,
    handler=process_run,
    parameter_value_providers={
        "calamariOCRModels": discover_calamari_ocr_models,
    },
)
