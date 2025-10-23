# realtime_inference.py
import argparse
import time
import threading
from typing import Generator, List, Optional

import gradio as gr
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLOE
import cv2
import torch  # 🔹 GPU 캐시 정리를 위해 추가

# Orbbec SDK (optional)
try:
    import pyorbbecsdk  # type: ignore
    from pyorbbecsdk import (
        Pipeline, Config, OBError,
        OBSensorType, OBFormat, FrameSet
    )  # type: ignore
    HAS_OBSDK = True
except Exception:
    HAS_OBSDK = False


# =========================
# Utils
# =========================
def normalize_prompts(text: str) -> List[str]:
    if not text:
        return []
    for s in [",", "\n", ";"]:
        text = text.replace(s, "|")
    return [t.strip() for t in text.split("|") if t.strip()]


def annotate(image_pil: Image.Image, det: sv.Detections, class_names: List[str]) -> Image.Image:
    res_wh = image_pil.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=res_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=res_wh)

    labels = []
    has_names = "class_name" in det.data
    confs = det.confidence if det.confidence is not None else []
    for i, conf in enumerate(confs):
        if has_names:
            name = det.data["class_name"][i]
        else:
            cid = int(det.class_id[i]) if det.class_id is not None else -1
            name = class_names[cid] if 0 <= cid < len(class_names) else f"id:{cid}"
        labels.append(f"{name} {float(conf):.2f}")

    canvas = image_pil.copy()
    canvas = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(scene=canvas, detections=det)
    canvas = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(scene=canvas, detections=det)
    canvas = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True)\
        .annotate(scene=canvas, detections=det, labels=labels)
    return canvas


# =========================
# Camera backends
# =========================
class CameraBase:
    def open(self): ...
    def read(self) -> Optional[np.ndarray]: ...
    def close(self): ...


class OrbbecCamera(CameraBase):
    """
    COLOR 프로파일: 1280x720@30 요청 (장치 포맷에 맞춰 read에서 RGB로 변환)
    """
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        if not HAS_OBSDK:
            raise RuntimeError("pyorbbecsdk not available")
        self.width, self.height, self.fps = width, height, fps
        self.pipeline, self.config = None, None

        # 안전 매핑: 환경에 따라 이름 다를 수 있음
        F = OBFormat

        def fmt(name: str, fallback: str):
            return getattr(F, name, getattr(F, fallback))

        self.FMT_RGB  = fmt("RGB888", "RGB")
        self.FMT_YUYV = fmt("YUYV", "YUYV")
        self.FMT_MJPG = fmt("MJPG", "MJPG")
        self.FMT_BGRA = fmt("BGRA", "BGRA")
        self.FMT_NV12 = fmt("NV12", "NV12")

    def open(self):
        pipeline = Pipeline()
        config = Config()
        plist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

        preferred = [
            (self.FMT_RGB,  self.width, self.height, self.fps),
            (self.FMT_YUYV, self.width, self.height, self.fps),
            (self.FMT_MJPG, self.width, self.height, self.fps),
            (self.FMT_BGRA, self.width, self.height, self.fps),
            (self.FMT_NV12, self.width, self.height, self.fps),
        ]

        chosen = None
        for fmt, w, h, fps in preferred:
            try:
                chosen = plist.get_video_stream_profile(w, h, fmt, fps)
                print(f"[INFO] Using profile: {{type: OB_STREAM_COLOR, format: {fmt.name}, width: {w}, height: {h}, fps: {fps}}}")
                break
            except OBError:
                continue

        if chosen is None:
            chosen = plist.get_default_video_stream_profile()
            print("[WARN] Preferred profiles unavailable. Fallback:", chosen)

        config.enable_stream(chosen)
        pipeline.start(config)
        self.pipeline, self.config = pipeline, config

    def read(self) -> Optional[np.ndarray]:
        fs: FrameSet = self.pipeline.wait_for_frames(100)
        if fs is None:
            return None
        cf = fs.get_color_frame()
        if cf is None:
            return None

        try:
            fmt = cf.get_format()
        except Exception:
            fmt = None

        buf = np.frombuffer(cf.get_data(), dtype=np.uint8)
        h, w = cf.get_height(), cf.get_width()

        if fmt in (self.FMT_RGB, getattr(OBFormat, "RGB", self.FMT_RGB)):
            if buf.size < h * w * 3:
                return None
            return buf.reshape((h, w, 3))

        if fmt == self.FMT_YUYV:
            if buf.size < h * w * 2:
                return None
            yuyv = buf.reshape((h, w, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)

        if fmt == self.FMT_MJPG:
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if fmt == self.FMT_BGRA:
            if buf.size < h * w * 4:
                return None
            bgra = buf.reshape((h, w, 4))
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

        if fmt == self.FMT_NV12:
            if buf.size < int(h * w * 1.5):
                return None
            nv12 = buf.reshape((int(h * 1.5), w))
            return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)

        try:
            return buf.reshape((h, w, 3))
        except Exception:
            return None

    def close(self):
        if self.pipeline:
            self.pipeline.stop()
        self.pipeline, self.config = None, None


class OpenCVCamera(CameraBase):
    def __init__(self, index: int, width: int, height: int, fps: int):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.cap = None

    def open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}")
        self.cap = cap

    def read(self) -> Optional[np.ndarray]:
        ok, frame_bgr = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.cap:
            self.cap.release()
        self.cap = None


def make_camera(prefer_orbbec: bool, cam_index: int, w: int, h: int, fps: int) -> CameraBase:
    if HAS_OBSDK and prefer_orbbec:
        return OrbbecCamera(w, h, fps)
    return OpenCVCamera(cam_index, w, h, fps)


# =========================
# Preview manager (항상 on)
# =========================
class PreviewManager:
    def __init__(self, prefer_orbbec: bool, cam_index: int, w: int, h: int, fps: int):
        self.cam = make_camera(prefer_orbbec, cam_index, w, h, fps)
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.cam.open()

        def _loop():
            while self.running:
                try:
                    f = self.cam.read()
                    if f is not None:
                        with self.frame_lock:
                            self.latest_frame = f
                    else:
                        time.sleep(0.004)
                except Exception:
                    time.sleep(0.01)
            self.cam.close()

        self.thread = threading.Thread(target=_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cam.close()

    def get_latest(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()


# =========================
# Inference Controller
# =========================
class InferenceController:
    def __init__(self, checkpoint: str, device: str, pm: PreviewManager):
        self.checkpoint = checkpoint
        self.device = device
        self.pm = pm

        self.model: Optional[YOLOE] = None
        self.prompts: List[str] = ["person"]
        self.conf: float = 0.25
        self.iou: float = 0.45
        self.infer_running = False

        # 프롬프트 변경 직후 워밍업 프레임 수 (잔상 제거)
        self._warmup_frames_default = 3
        self._warmup_left = 0

        self._model_lock = threading.Lock()

    def _new_model(self) -> YOLOE:
        """매번 새로운 모델 생성 (이전 상태/임베딩 완전 초기화)"""
        # 기존 모델 정리
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 새 모델 로드
        m = YOLOE(self.checkpoint)
        m.to(self.device)
        return m

    def start_infer(self, prompts_text: str, conf: float, iou: float):
        plist = normalize_prompts(prompts_text) or ["person"]
        self.prompts = plist
        self.conf = float(conf)
        self.iou = float(iou)

        with self._model_lock:
            # 🔑 핵심: 매번 새로운 인스턴스로 교체
            self.model = self._new_model()
            # 텍스트 임베딩 재설정
            self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))

        self.infer_running = True
        self._warmup_left = self._warmup_frames_default

    def stop_infer(self):
        self.infer_running = False
        self._warmup_left = 0
        # 🔑 모델 내려서 잔상/누적 상태 제거
        with self._model_lock:
            if self.model is not None:
                try:
                    del self.model
                except Exception:
                    pass
                self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_once(self, frame_rgb: np.ndarray) -> Image.Image:
        """infer_running이면 모델 추론 후 annotate, 아니면 원본 프레임 반환."""
        pil_in = Image.fromarray(frame_rgb)
        if not self.infer_running:
            return pil_in

        # 프롬프트 변경/시작 직후 몇 프레임 드롭(버퍼/동기화)
        if self._warmup_left > 0:
            self._warmup_left -= 1
            return pil_in

        try:
            with self._model_lock:
                if self.model is None:
                    return pil_in
                results = self.model.predict(pil_in, conf=self.conf, iou=self.iou, verbose=False)
            det = sv.Detections.from_ultralytics(results[0])

            # class_name 보강
            if "class_name" not in det.data:
                names = []
                for cid in (det.class_id or []):
                    if cid is None or int(cid) >= len(self.prompts) or int(cid) < 0:
                        names.append("unknown")
                    else:
                        names.append(self.prompts[int(cid)])
                det.data["class_name"] = np.array(names, dtype=object)

            # 검출 0개여도 항상 PIL 반환
            return annotate(pil_in, det, self.prompts) if len(det) > 0 else pil_in

        except Exception:
            # 에러 시에도 스트림 유지
            return pil_in


# =========================
# Gradio App
# =========================
def build_app(checkpoint: str, device: str, prefer_orbbec: bool, cam_index: int, w: int, h: int, fps: int):

    pm = PreviewManager(prefer_orbbec, cam_index, w, h, fps)
    ctrl = InferenceController(checkpoint, device, pm)

    with gr.Blocks(title="YOLO-E Live Inference (Single View)") as demo:
        with gr.Row():
            with gr.Column(scale=5):
                status_md = gr.Markdown("**Not inferencing**")
                live_out = gr.Image(type="pil", label="Live View (raw or annotated)", streaming=True)

            with gr.Column(scale=3):
                promtp_md = gr.Markdown("**Put your text prompts**")
                prompts = gr.Textbox(label="Text Prompts", value="person,bus", lines=4)
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence")
                iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU")
                with gr.Row():
                    btn_start = gr.Button("Start Inference", variant="primary")
                    btn_stop = gr.Button("Stop Inference", variant="stop")

        # 단일 스트림(상시 동작, 좌측 창만 갱신)
        def unified_stream() -> Generator[Image.Image, None, None]:
            pm.start()
            # 첫 프레임 빠르게 채움
            yield Image.fromarray(np.zeros((h if h > 0 else 720, w if w > 0 else 1280, 3), dtype=np.uint8))
            while True:
                frame = pm.get_latest()
                if frame is None:
                    time.sleep(0.006)
                    continue
                img = ctrl.run_once(frame)
                yield img

        demo.load(fn=unified_stream, inputs=None, outputs=[live_out])

        # 상태 토글
        def on_start(prompts_text, conf_val, iou_val):
            ctrl.start_infer(prompts_text, conf_val, iou_val)
            return gr.update(value="**Inference running...**")

        def on_stop():
            ctrl.stop_infer()
            return gr.update(value="**Not inferencing**")

        btn_start.click(fn=on_start, inputs=[prompts, conf, iou], outputs=[status_md])
        btn_stop.click(fn=on_stop, inputs=None, outputs=[status_md])

    return demo


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="yoloe-v8l-seg.pt")
    ap.add_argument("--device", type=str, default="cuda:0")  # 또는 cpu
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--prefer-orbbec", action="store_true")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(
        checkpoint=args.checkpoint,
        device=args.device,
        prefer_orbbec=args.prefer_orbbec,
        cam_index=args.camera_index,
        w=args.width,
        h=args.height,
        fps=args.fps,
    )
    app.queue(max_size=32)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=False,
        share=False,
        max_threads=2,
    )
