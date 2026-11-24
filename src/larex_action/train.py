from reader import params_from_input
from models import discover_calamari_ocr_models, discover_model_dirs
import tfaip.util.logging
import os
from copy import deepcopy
from functools import partial
from larex_actions import ActionContext
from larex_actions.fastapi import create_larex_action_app
import logging
from crossfold import cf_params_from_input, LarexCrossFoldTrainer

logger = tfaip.util.logging.logger(__name__)


PROCESSOR_ID = os.getenv("LAREX_PROCESSOR_ID", "calamari-train").strip()
DISPATCH_SECRET_ENV = "LAREX_DISPATCH_HMAC_SECRET"


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


async def process_run(ctx: ActionContext) -> None:
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.complete(
            ctx.result_builder(), "Calamari OCR received no pages."
        )
        return
    httpxclient = ctx.client._client
    ctx.client._client = None
    ctx_clientless = deepcopy(ctx)
    ctx.client._client = httpxclient

    data = params_from_input(ctx_clientless, action_input, gt=True)
    cfparams = cf_params_from_input(action_input, data)
    if cfparams.best_models_dir is None:
        await ctx.heartbeat(status_message="Model name invalid!",
                            status="failed")
        raise ValueError("Invalid model name")
    trainer = LarexCrossFoldTrainer(cfparams)
    await trainer.run(ctx)


app = create_larex_action_app(
    processor_id=PROCESSOR_ID,
    dispatch_secret_env=DISPATCH_SECRET_ENV,
    handler=process_run,
    parameter_value_providers={
        "calamariOCRModels": partial(discover_calamari_ocr_models,
                                     default="none"),
        "calamariModelDirs": discover_model_dirs,
    },
)
