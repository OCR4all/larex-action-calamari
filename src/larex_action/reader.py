from calamari_ocr.ocr.dataset.datareader.pagexml.reader import (
    PageXML,
    PageXMLReader,
    PageXMLDatasetLoader,
    InputSample,
    SampleMeta,
)
import tfaip.util.logging
import httpx
from io import BytesIO
from tfaip.data.pipeline.definitions import (
        PipelineMode, INPUT_PROCESSOR, TARGETS_PROCESSOR)
from dataclasses import dataclass
from random import shuffle
from larex_actions import ActionContext, ActionTargetSelection
from paiargparse import pai_dataclass
from paiargparse.dataclass_extractor import extract_args_of_dataclass
from lxml import etree
import numpy as np

logger = tfaip.util.logging.logger(__name__)


def ats_to_dict(ats: ActionTargetSelection | None) -> dict:
    # transform ats to dict page_id -> (type, ids)
    if ats is None:
        return {}
    if ats.type == "REGION":
        return {p.page_id: (ats.type, p.region_ids) for p in ats.pages}
    if ats.type == "TEXT_LINE":
        return {p.page_id: (ats.type, p.text_line_ids) for p in ats.pages}
    return {}


def sync_download_bytes(client, file):
    attempt = 1
    while attempt < 10:
        attempt += 1
        try:
            client._validate_download_url(file.download_url)
            response = client._client.get(file.download_url,
                                          headers=client._auth_headers())
            response.raise_for_status()
            return response.content
        except Exception as exc:
            logger.warning(f"Download failed (attempt {attempt-1}/10) with exception: ",
                           type(exc).__name__)
            continue
    raise exc


def params_from_input(ctx: ActionContext, action_input,
                      gt=False, xml_bytes=None) -> dict:
    parameters = getattr(action_input, "parameters", {}) or {}
    larex_args = extract_args_of_dataclass(Larex)
    lrx_dict = {arg.name: parameters.get(arg.name, arg.default)
                for arg in larex_args}
    lrx_dict["images"] = [p for p in action_input.pages
                          if len(p.xml) and len(p.images)]
    lrx_dict["targets"] = action_input.target_selection
    lrx_dict["ctx"] = ctx
    lrx_dict["xml_bytes"] = xml_bytes or {}
    if gt and "gt_index" in parameters:
        lrx_dict["text_index"] = parameters["gt_index"]
    return Larex.from_dict(lrx_dict)


class LarexXMLDatasetLoader(PageXMLDatasetLoader):
    def __init__(
        self,
        mode: PipelineMode,
        ctx: ActionContext,
        non_existing_as_empty: bool,
        text_index: int,
        xml_bytes: dict = None,
        skip_invalid: bool = True,
        skip_commented: bool = False,
    ):
        super().__init__(
            mode, non_existing_as_empty, text_index, skip_invalid,
            skip_commented=False
        )
        self.ctx = ctx
        self.xml_bytes = xml_bytes or {}

    def _select_lines(self, root, ns, selection):
        stype, slist = selection
        if stype == "PAGE":
            return root.findall(".//ns:TextLine", namespaces=ns)
        elif stype == "REGION":
            return [
                tl
                for r in slist
                for tl in root.findall(
                    f'.//ns:TextRegion[@id="{r}"]/ns:TextLine', namespaces=ns
                )
            ]
        elif stype == "TEXT_LINE":
            return [
                line
                for line in root.findall(".//ns:TextLine", namespaces=ns)
                if line.attrib.get("id", None) in slist
            ]

    def load(self, page, cache=None, selection=("PAGE", [])):
        if cache is None:
            cache = {}
        page_id = page.id
        if page_id not in cache:
            xml_bytes = self.xml_bytes.get(page_id,
                                           sync_download_bytes(self.ctx.client,
                                                               page.xml[0]))
            cache[page_id] = etree.fromstring(
                xml_bytes,
                parser=etree.XMLParser(resolve_entities=False,
                                       no_network=True),
            )
        self.root = cache[page_id]
        root = self.root
        ns = {"ns": root.nsmap[root.prefix]}
        page_el = root.find(".//ns:Page", namespaces=ns)
        img_w = int(page_el.attrib.get("imageWidth"))

        textlines = self._select_lines(root, ns, selection)
        for textline in textlines:
            text = None
            if self.mode in TARGETS_PROCESSOR:
                tequivs = textline.findall('./ns:TextEquiv'
                                           + f'[@index="{self.text_index}"]',
                                           namespaces=ns)

                if not tequivs:
                    tequivs = [te for te in
                               textline.findall("./ns:TextEquiv",
                                                namespaces=ns)
                               if "index" not in te.attrib]

                if len(tequivs) > 1:
                    logger.warning("PageXML is invalid: TextLine includes "
                                   + "TextEquivs with non unique ids")

                if self.skip_commented and len(textline.attrib.get("comments",
                                                                   "")):
                    continue

                if tequivs is not None and len(tequivs) > 0:
                    te = tequivs[0]
                    uc = te.find("./ns:Unicode", namespaces=ns)
                    text = uc.text if uc is not None else ""
                    if text is None:
                        # Handle empty tag as empty string,
                        # not as "not existing"
                        text = ""
                else:
                    text = None

                if text is None:
                    if self.skip_invalid:
                        continue
                    elif self._non_existing_as_empty:
                        text = ""
                    else:
                        raise Exception("Empty text field")
            if self.mode in {PipelineMode.TRAINING, PipelineMode.EVALUATION}:
                if len(text) == 0:
                    # Empty lines cannot be used for training (CTC-loss can
                    # not be computed)
                    continue

            # add text loading here
            yield {
                "page_id": page_id,
                "ns": ns,
                "rtype": textline.getparent().attrib.get("type", ""),
                "xml_element": (
                    textline if self.mode == PipelineMode.PREDICTION else None
                ),
                # "larex_page": page,
                "id": "{}/{}".format(page_id, textline.attrib.get("id")),
                "base_name": textline.attrib.get("id"),
                "coords": textline.find("./ns:Coords",
                                        namespaces=ns).attrib.get(
                    "points"
                ),
                "orientation": float(textline.getparent(
                        ).attrib.get("orientation", 0)),
                "img_width": img_w,
                "text": text,
            }


@pai_dataclass
@dataclass
class Larex(PageXML):
    ctx: ActionContext = None
    targets: ActionTargetSelection = None
    xml_bytes: dict = None

    def __len__(self):
        return len(self.images)

    @staticmethod
    def cls():
        return LarexReader

    def select(self, indices):
        raise NotImplementedError

    def to_prediction(self):
        return self

    def prepare_for_mode(self, mode):
        if self.ctx.client._client is None:
            self.ctx.client._client = httpx.Client(timeout=120.0)


class LarexReader(PageXMLReader):
    def __init__(self, mode, params):
        super().__init__(mode, params)
        self.etrees = {}
        self.target_dict = ats_to_dict(params.targets)
        self.current_result = None
        for page in params.images:
            loader = LarexXMLDatasetLoader(
                self.mode,
                params.ctx,
                params.non_existing_as_empty,
                params.text_index,
                params.xml_bytes,
                params.skip_invalid,
                params.skip_commented,
            )
            for sample in loader.load(
                page, selection=self.target_dict.get(page.id, ("PAGE", []))
            ):
                self.add_sample(sample)
            self.etrees[page.id] = loader.root
        # store which pagexml was stored last, to check when a file is ready to
        # be written during sequential prediction
        self._last_page_id = None

    def result_pop(self):
        res = self.current_result
        self.current_result = None
        return res

    def _store_page(self, extension, page_id):
        if self.current_result is not None:
            raise ValueError(
                "LarexReader.current_result has to be read and reset with"
                + ".result_pop() before a new page can be saved!"
            )
        et = self.etrees[page_id]
        self.current_result = (
            page_id,
            etree.tostring(et.getroottree(), encoding="utf-8",
                           xml_declaration=True),
        )

    def _sample_iterator(self):
        all_samples = enumerate(self.params.images)
        if self.mode == PipelineMode.TRAINING:
            all_samples = list(all_samples)
            shuffle(all_samples)
        return all_samples

    def _load_image(self, page):
        img_bytes = sync_download_bytes(self.params.ctx.client, page.images[0])
        return self._image_loader.load_image(BytesIO(img_bytes))

    def _load_sample(self, i_page, text_only):
        loader = LarexXMLDatasetLoader(
            self.mode,
            self.params.ctx,
            self.params.non_existing_as_empty,
            self.params.text_index,
            self.params.xml_bytes,
            self.params.skip_invalid,
            self.params.skip_commented,
        )

        idx, page = i_page
        img = None
        for i, sample in enumerate(loader.load(
            page,
            cache=self.etrees,
            selection=self.target_dict.get(page.id, ("PAGE", [])),
        )):
            fold_id = (idx + i) % self.params.n_folds \
                    if self.params.n_folds > 0 else -1
            text = sample["text"]
            orientation = sample["orientation"]
            if not text_only and self.mode in INPUT_PROCESSOR:
                if img is None:
                    img = self._load_image(page)

                ly, lx = img.shape[:2]

                # rotate by orientation angle in clockwise direction to
                # correct present skew
                angle = orientation if orientation and orientation % 360 != 0 \
                    else 0

                line_img = PageXMLReader.cutout(
                    img,
                    sample["coords"],
                    mode=self.params.cut_mode,
                    angle=angle,
                    cval=None,
                    scale=lx / sample["img_width"],
                ).copy()

                # add padding as required from normal files
                if self.params.pad:
                    line_img = np.pad(
                        line_img,
                        self.params.pad,
                        mode="constant",
                        constant_values=line_img.max(initial=0),
                    )
            else:
                line_img = None

            yield InputSample(
                line_img, text, SampleMeta(id=sample["id"], fold_id=fold_id)
            )
