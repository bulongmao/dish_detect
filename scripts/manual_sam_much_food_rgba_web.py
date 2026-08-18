#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import io
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from segment_anything import SamPredictor, sam_model_registry

import manual_sam_much_food_rgba_gradio as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web" / "manual_sam"
MAX_SESSIONS = 12
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

app = FastAPI(title="SAM Food Studio", version="1.0.0")
app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

SESSIONS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
SAM_LOCK = threading.RLock()


class LoadRequest(BaseModel):
    image_name: str


class ProcessRequest(BaseModel):
    session_id: str
    use_fill_holes: bool = True
    close_kernel: int = Field(default=3, ge=0, le=15)
    erode_iterations: int = Field(default=1, ge=0, le=3)


class ClickRequest(ProcessRequest):
    x: float
    y: float
    label_mode: Literal["前景点", "背景点"] = "前景点"


class CandidateRequest(ProcessRequest):
    candidate_index: int = Field(default=0, ge=0, le=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HTML + CSS + JavaScript 版人工 SAM 食物分割工具"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=core.DEFAULT_INPUT_IMAGE_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=core.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=core.find_default_checkpoint(),
    )
    parser.add_argument(
        "--model-type",
        choices=sorted(sam_model_registry.keys()),
        default=None,
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def image_data_url(image_rgb: Optional[np.ndarray]) -> Optional[str]:
    if image_rgb is None:
        return None
    buffer = io.BytesIO()
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(
        buffer,
        format="WEBP",
        quality=92,
        method=4,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def decode_uploaded_image(data: bytes, file_name: str) -> np.ndarray:
    if not data:
        raise ValueError("上传文件为空，请重新选择图片")

    try:
        with Image.open(io.BytesIO(data)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            return np.array(normalized, dtype=np.uint8, copy=True)
    except (OSError, UnidentifiedImageError, ValueError):
        encoded = np.frombuffer(data, dtype=np.uint8)
        decoded = core.cv2.imdecode(encoded, core.cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(
                f"无法解析图片“{file_name}”。请使用 JPG、PNG、WEBP 或 BMP，"
                "手机 HEIC 图片请先另存为 JPG/PNG。"
            )
        if decoded.ndim == 2:
            decoded = core.cv2.cvtColor(decoded, core.cv2.COLOR_GRAY2RGB)
        elif decoded.shape[2] == 4:
            decoded = core.cv2.cvtColor(decoded, core.cv2.COLOR_BGRA2RGB)
        else:
            decoded = core.cv2.cvtColor(decoded, core.cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(decoded, dtype=np.uint8)


def persist_uploaded_bytes(data: bytes, file_name: str):
    image_rgb = decode_uploaded_image(data, file_name)
    png_bytes = core.encode_rgb_png(image_rgb)
    upload_root = core.INPUT_IMAGE_ROOT / core.UPLOAD_SUBDIR
    upload_root.mkdir(parents=True, exist_ok=True)
    stem = core.sanitize_upload_stem(file_name)
    destination = upload_root / f"{stem}.png"
    suffix = 2
    while destination.exists() and destination.read_bytes() != png_bytes:
        destination = upload_root / f"{stem}_{suffix}.png"
        suffix += 1
    if not destination.exists():
        destination.write_bytes(png_bytes)
    return destination.resolve(), image_rgb


def get_state(session_id: str) -> Dict[str, Any]:
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="会话已失效，请重新加载图片")
    SESSIONS.move_to_end(session_id)
    return state


def create_session(path: Path, image_name: str, image: np.ndarray):
    session_id = uuid.uuid4().hex
    state = core.empty_state()
    state["image_name"] = image_name
    state["image_path"] = str(path)
    core.set_predictor_image(str(path), image)
    SESSIONS[session_id] = state
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return session_id, state


def response_payload(
    session_id: str,
    state: Dict[str, Any],
    image: np.ndarray,
    status: str,
    use_fill_holes: bool = False,
    close_kernel: int = 0,
    erode_iterations: int = 0,
    preview: Optional[np.ndarray] = None,
):
    scores = state.get("scores")
    return {
        "session_id": session_id,
        "image_name": state.get("image_name", ""),
        "image": image_data_url(
            core.overlay_image(
                image,
                state,
                use_fill_holes,
                close_kernel,
                erode_iterations,
            )
        ),
        "preview": image_data_url(preview),
        "status": status,
        "candidate_index": int(state.get("candidate_index", 0)),
        "scores": (
            [round(float(score), 4) for score in scores]
            if scores is not None
            else []
        ),
    }


def process_values(request: ProcessRequest):
    return (
        bool(request.use_fill_holes),
        int(request.close_kernel),
        int(request.erode_iterations),
    )


@app.get("/")
def index():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/styles.css", include_in_schema=False)
def styles():
    return FileResponse(WEB_ROOT / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def javascript():
    return FileResponse(
        WEB_ROOT / "app.js",
        media_type="application/javascript",
    )


@app.get("/api/health")
def health():
    return {
        "ok": core.PREDICTOR is not None,
        "device": str(next(core.PREDICTOR.model.parameters()).device)
        if core.PREDICTOR is not None
        else None,
        "input_root": str(core.INPUT_IMAGE_ROOT),
        "output_root": str(core.OUTPUT_ROOT),
    }


@app.get("/api/images")
def list_images():
    paths = core.collect_images(core.INPUT_IMAGE_ROOT)
    return {
        "images": [
            path.relative_to(core.INPUT_IMAGE_ROOT).as_posix()
            for path in paths
        ]
    }


@app.post("/api/load")
def load_image(request: LoadRequest):
    with SAM_LOCK:
        try:
            path = core.resolve_input_image(request.image_name)
            if not path.is_file():
                raise ValueError(f"图片不存在：{path}")
            image = core.load_rgb(path)
            session_id, state = create_session(
                path,
                request.image_name,
                image,
            )
            return response_payload(
                session_id,
                state,
                image,
                f"已加载：{request.image_name}",
            )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/upload")
def upload_image(file: UploadFile = File(...)):
    original_name = Path(file.filename or "uploaded_image.png").name
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片超过 64 MB，请压缩后上传")

    with SAM_LOCK:
        try:
            path, image = persist_uploaded_bytes(data, original_name)
            image_name = path.relative_to(core.INPUT_IMAGE_ROOT).as_posix()
            session_id, state = create_session(path, image_name, image)
            return response_payload(
                session_id,
                state,
                image,
                "本地图片已上传并加载\n"
                f"原图：{path}\n"
                f"分割结果将保存到：{core.OUTPUT_ROOT}",
            )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/click")
def click_image(request: ClickRequest):
    with SAM_LOCK:
        state = get_state(request.session_id)
        image = core.load_rgb(Path(state["image_path"]))
        x = int(np.clip(request.x, 0, image.shape[1] - 1))
        y = int(np.clip(request.y, 0, image.shape[0] - 1))
        state["points"].append([x, y])
        state["labels"].append(1 if request.label_mode == "前景点" else 0)
        core.predict_state(state, image)
        fill_holes, close_kernel, erode_iterations = process_values(request)
        return response_payload(
            request.session_id,
            state,
            image,
            core.status_text(state),
            fill_holes,
            close_kernel,
            erode_iterations,
        )


@app.post("/api/candidate")
def choose_candidate(request: CandidateRequest):
    with SAM_LOCK:
        state = get_state(request.session_id)
        masks = state.get("masks")
        if masks is not None and len(masks) > 0:
            state["candidate_index"] = int(
                np.clip(request.candidate_index, 0, len(masks) - 1)
            )
        image = core.load_rgb(Path(state["image_path"]))
        fill_holes, close_kernel, erode_iterations = process_values(request)
        return response_payload(
            request.session_id,
            state,
            image,
            core.status_text(state),
            fill_holes,
            close_kernel,
            erode_iterations,
        )


@app.post("/api/undo")
def undo(request: ProcessRequest):
    with SAM_LOCK:
        state = get_state(request.session_id)
        if state["points"]:
            state["points"].pop()
            state["labels"].pop()
        image = core.load_rgb(Path(state["image_path"]))
        core.predict_state(state, image)
        fill_holes, close_kernel, erode_iterations = process_values(request)
        return response_payload(
            request.session_id,
            state,
            image,
            core.status_text(state),
            fill_holes,
            close_kernel,
            erode_iterations,
        )


@app.post("/api/clear")
def clear(request: ProcessRequest):
    with SAM_LOCK:
        state = get_state(request.session_id)
        state["points"] = []
        state["labels"] = []
        state["masks"] = None
        state["scores"] = None
        state["candidate_index"] = 0
        image = core.load_rgb(Path(state["image_path"]))
        fill_holes, close_kernel, erode_iterations = process_values(request)
        return response_payload(
            request.session_id,
            state,
            image,
            core.status_text(state),
            fill_holes,
            close_kernel,
            erode_iterations,
        )


@app.post("/api/commit")
def commit(request: ProcessRequest):
    with SAM_LOCK:
        state = get_state(request.session_id)
        mask = core.current_mask(state)
        image = core.load_rgb(Path(state["image_path"]))
        fill_holes, close_kernel, erode_iterations = process_values(request)
        if mask is None:
            return response_payload(
                request.session_id,
                state,
                image,
                "当前没有可加入的 SAM Mask",
                fill_holes,
                close_kernel,
                erode_iterations,
            )
        if state["committed_mask"] is None:
            state["committed_mask"] = mask.copy()
        else:
            state["committed_mask"] = np.logical_or(
                state["committed_mask"],
                mask,
            )
        state["points"] = []
        state["labels"] = []
        state["masks"] = None
        state["scores"] = None
        state["candidate_index"] = 0
        return response_payload(
            request.session_id,
            state,
            image,
            "已加入最终 Mask；可继续点击下一块食物",
            fill_holes,
            close_kernel,
            erode_iterations,
        )


@app.post("/api/reset")
def reset(request: ProcessRequest):
    with SAM_LOCK:
        state = get_state(request.session_id)
        image_path = state["image_path"]
        image_name = state["image_name"]
        image = core.load_rgb(Path(image_path))
        new_state = core.empty_state()
        new_state["image_path"] = image_path
        new_state["image_name"] = image_name
        core.set_predictor_image(image_path, image)
        SESSIONS[request.session_id] = new_state
        return response_payload(
            request.session_id,
            new_state,
            image,
            "已重置全部选择",
        )


@app.post("/api/preview")
def preview(request: ProcessRequest):
    with SAM_LOCK:
        state = get_state(request.session_id)
        image = core.load_rgb(Path(state["image_path"]))
        fill_holes, close_kernel, erode_iterations = process_values(request)
        return response_payload(
            request.session_id,
            state,
            image,
            core.status_text(state),
            fill_holes,
            close_kernel,
            erode_iterations,
        )


@app.post("/api/save")
def save(request: ProcessRequest):
    with SAM_LOCK:
        state = get_state(request.session_id)
        fill_holes, close_kernel, erode_iterations = process_values(request)
        preview_image, status = core.save_result(
            state,
            fill_holes,
            close_kernel,
            erode_iterations,
        )
        image = core.load_rgb(Path(state["image_path"]))
        return response_payload(
            request.session_id,
            state,
            image,
            status,
            fill_holes,
            close_kernel,
            erode_iterations,
            preview_image,
        )


def main():
    args = parse_args()
    public_host = (
        "sam-food"
        if args.host in {"127.0.0.1", "0.0.0.0"}
        else args.host
    )
    public_url = f"http://{public_host}"
    if args.port != 80:
        public_url += f":{args.port}"
    core.configure_runtime_paths(
        args.input_root,
        args.output_root,
        args.checkpoint,
        args.model_type,
    )
    core.INPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    core.ensure_dirs()
    if not WEB_ROOT.is_dir():
        raise FileNotFoundError(f"网页资源目录不存在：{WEB_ROOT}")
    if not core.SAM_CHECKPOINT.is_file():
        raise FileNotFoundError(f"SAM 权重不存在：{core.SAM_CHECKPOINT}")

    device = core.resolve_device(args.device)
    print("=" * 56)
    print("SAM Food Studio")
    print("Input:", core.INPUT_IMAGE_ROOT)
    print("Output:", core.OUTPUT_ROOT)
    print("Checkpoint:", core.SAM_CHECKPOINT)
    print("Model type:", core.SAM_MODEL_TYPE)
    print("Device:", device)
    print("URL:", public_url)
    print("=" * 56)

    if args.check_only:
        print("检查通过：路径、依赖和计算设备均可用。")
        return

    sam = sam_model_registry[core.SAM_MODEL_TYPE](
        checkpoint=str(core.SAM_CHECKPOINT)
    )
    sam.to(device=device)
    sam.eval()
    core.PREDICTOR = SamPredictor(sam)

    if args.open_browser:
        import webbrowser

        threading.Timer(
            1.5,
            lambda: webbrowser.open(public_url),
        ).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
