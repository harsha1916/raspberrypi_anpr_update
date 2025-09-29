#!/usr/bin/env python3
# Raspberry Pi 5 | Python 3.11.x
import os, sys, time, signal, threading
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

# ==========================
# HARD-CODE CONFIG
# ==========================
MODEL_PATH = "plate-yolo-data-384-without-rect-3.onnx"

# Choose source: "rtsp" or "webcam"
SOURCE      = "rtsp"          # "rtsp" | "webcam"
RTSP_URL    = "rtsp://admin:admin@192.168.1.201:554/avstream/channel=1/stream=0.sdp"
WEBCAM_ID   = 0               # /dev/video0
WEBCAM_SIZE = (640, 480)      # requested capture size

# Decode downscale (reduces CPU for RTSP)
USE_GSTREAMER = True
GST_SIZE      = (640, 360)    # decoded size (w,h) for RTSP

# Output & view
OUTPUT_MP4  = ""              # e.g., "/home/pi5/plates.mp4" or "" to disable
SHOW_WINDOW = True
TARGET_FPS  = 0               # 0 = unlimited; else cap (e.g., 12)

# Detection thresholds
CONF_THRES  = 0.25
IOU_THRES   = 0.45
PAD_VALUE   = 114

# If model trained grayscale but exported 3ch, replicating gray→3 often improves scores
REPLICATE_GRAY_TO_3 = True

# If model has multiple classes and you know the plate class id, set it; else leave None
TARGET_CLASS_ID = None

# Debug prints
DEBUG = False
def dprint(*a):
    if DEBUG: print("[DEBUG]", *a)

# Keep math libs from oversubscribing threads
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ==========================
# Utils
# ==========================
def letterbox(img, new_shape, pad_value=114):
    Ht, Wt = new_shape
    h, w = img.shape[:2]
    r = min(Ht / h, Wt / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (Ht - nh) // 2
    bottom = Ht - nh - top
    left = (Wt - nw) // 2
    right = Wt - nw - left
    if img.ndim == 2:
        out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=pad_value)
    else:
        out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    return out

def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0: return []
    boxes = boxes.astype(np.float32)
    x1,y1,x2,y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        ovr = inter/(areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds+1]
    return keep

def plausible_pick(xyxy_list, scores, W, H):
    """Pick the conversion giving boxes of reasonable size (not tiny/huge/outside)."""
    best = None; best_score = -1
    for xyxy in xyxy_list:
        if xyxy is None or len(xyxy)==0: continue
        w = np.maximum(0, xyxy[:,2]-xyxy[:,0])
        h = np.maximum(0, xyxy[:,3]-xyxy[:,1])
        area = (w*h) / (W*H + 1e-9)
        valid = (w>2) & (h>2) & (area < 0.6)
        score = valid.mean() + 0.05*(scores.mean() if len(scores) else 0.0)
        if score > best_score:
            best = xyxy; best_score = score
    return best

# ==========================
# YOLO6 model wrapper (output [1, 9072, 6])
# ==========================
class PlateYOLO6:
    def __init__(self, model_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.out_name = self.sess.get_outputs()[0].name

        ishape = self.inp.shape  # [1,3,384,384]
        self.in_h = ishape[2] if isinstance(ishape[2], int) else 384
        self.in_w = ishape[3] if isinstance(ishape[3], int) else 384
        print(f"[INFO] Model: input 3x{self.in_h}x{self.in_w}, output {self.out_name}")

    def preprocess(self, frame_bgr):
        # GRAY->3 helps when training data was grayscale but model expects 3ch
        if REPLICATE_GRAY_TO_3:
            g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            img3 = cv2.merge([g,g,g])
        else:
            img3 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        im = letterbox(img3, (self.in_h, self.in_w), PAD_VALUE)
        x = im.astype(np.float32) / 255.0
        x = np.transpose(x, (2,0,1))[None,...]   # [1,3,H,W]
        return x

    def postprocess(self, out, frame_shape):
        H, W = frame_shape[:2]
        y = out[0]                    # [N,6]
        if y.ndim != 2 or y.shape[1] != 6:
            return []

        # Interpret as [x, y, w, h, obj, cls_prob_or_id]
        boxes = y[:, :4].astype(np.float32)
        obj   = y[:, 4].astype(np.float32)
        last  = y[:, 5].astype(np.float32)

        # If last ∈ [0,1], treat it as class prob; else as class id
        if np.all((last >= 0.0) & (last <= 1.0)):
            scores = obj * last
            cls_ids = (last > 0.5).astype(np.int32)  # dummy if single class
        else:
            scores = obj
            cls_ids = np.rint(last).astype(np.int32)

        if TARGET_CLASS_ID is not None:
            cls_ids = np.full_like(cls_ids, TARGET_CLASS_ID)

        # Filter by conf
        keep = scores >= CONF_THRES
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if boxes.size == 0:
            return []

        # Your debug suggested "model_pixels" → abs @ net size
        # Try both center and top-left using model pixel scale, pick plausible
        cx, cy, w, h = boxes.T
        # center → xyxy (scale from net to image)
        xyxy_center = np.stack(
            [(cx - w/2) * (W/self.in_w),
             (cy - h/2) * (H/self.in_h),
             (cx + w/2) * (W/self.in_w),
             (cy + h/2) * (H/self.in_h)], 1)
        # top-left → xyxy (scale from net to image)
        xyxy_tleft = np.stack(
            [cx * (W/self.in_w),
             cy * (H/self.in_h),
             (cx + w) * (W/self.in_w),
             (cy + h) * (H/self.in_h)], 1)

        xyxy = plausible_pick([xyxy_center, xyxy_tleft], scores, W, H)
        if xyxy is None:
            return []

        # Clip
        xyxy[:,0::2] = np.clip(xyxy[:,0::2], 0, W-1)
        xyxy[:,1::2] = np.clip(xyxy[:,1::2], 0, H-1)

        # NMS
        keep_idx = nms(xyxy, scores, IOU_THRES)
        xyxy = xyxy[keep_idx]; scores = scores[keep_idx]; cls_ids = cls_ids[keep_idx]

        return [(int(a),int(b),int(c),int(d), float(s), int(k))
                for (a,b,c,d), s, k in zip(xyxy, scores, cls_ids)]

    def infer(self, frame_bgr):
        x = self.preprocess(frame_bgr)
        y = self.sess.run([self.out_name], {self.inp.name: x})[0]
        return self.postprocess(y, frame_bgr.shape)

# ==========================
# Frame readers
# ==========================
class RTSPReader(threading.Thread):
    def __init__(self, url, reconnect_delay=1.0, size=(640,360)):
        super().__init__(daemon=True)
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.size = size
        self.cap = None
        self.latest = None
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()

    # ---- helpers ----
    @staticmethod
    def _has_gstreamer():
        try:
            info = cv2.getBuildInformation()
            return "GStreamer" in info and "YES" in info.split("GStreamer")[1].splitlines()[0]
        except Exception:
            return False

    def _gst_try_open(self, pipe_desc):
        cap = cv2.VideoCapture(pipe_desc, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] GStreamer OK: {pipe_desc}")
                return cap
            cap.release()
        print(f"[WARN] GStreamer failed: {pipe_desc}")
        return None

    def _ffmpeg_try_open(self, url, use_tcp=True, bufsize=3):
        # FFMPEG backend; try forcing TCP (more stable over WAN)
        url2 = url
        if use_tcp and ("?" not in url2):
            url2 = url + "?rtsp_transport=tcp"
        cap = cv2.VideoCapture(url2, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, bufsize))
            # Some OpenCV builds support timeouts:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)   # 3s
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)   # 3s
        except Exception:
            pass
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] FFMPEG OK ({'TCP' if use_tcp else 'UDP'}): {url2}")
                return cap
            cap.release()
        print(f"[WARN] FFMPEG failed ({'TCP' if use_tcp else 'UDP'}): {url2}")
        return None

    def open(self):
        w, h = self.size

        # 1) GStreamer paths (if available)
        if self._has_gstreamer():
            print("[INFO] OpenCV reports GStreamer=YES; trying pipelines…")

            # Prefer TCP first (more reliable), then UDP
            tcp_flag = "protocols=tcp"
            udp_flag = "protocols=udp"

            # (a) Hardware decode on Pi5 (v4l2h264dec). If H.265, change to v4l2h265dec.
            pipes = [
                # TCP, v4l2 decoder
                (f"rtspsrc location={self.url} {tcp_flag} latency=200 ! "
                 f"rtpjitterbuffer drop-on-late=true ! rtph264depay ! h264parse ! "
                 f"v4l2h264dec ! videoconvert ! videoscale ! "
                 f"video/x-raw, width={w}, height={h}, format=BGR ! "
                 f"appsink max-buffers=1 drop=true sync=false"),

                # TCP, software decode (avdec_h264)
                (f"rtspsrc location={self.url} {tcp_flag} latency=200 ! "
                 f"rtpjitterbuffer drop-on-late=true ! rtph264depay ! h264parse ! "
                 f"avdec_h264 ! videoconvert ! videoscale ! "
                 f"video/x-raw, width={w}, height={h}, format=BGR ! "
                 f"appsink max-buffers=1 drop=true sync=false"),

                # TCP, generic decodebin (lets GStreamer pick)
                (f"rtspsrc location={self.url} {tcp_flag} latency=200 ! "
                 f"rtpjitterbuffer drop-on-late=true ! "
                 f"rtph264depay ! h264parse ! decodebin ! videoconvert ! videoscale ! "
                 f"video/x-raw, width={w}, height={h}, format=BGR ! "
                 f"appsink max-buffers=1 drop=true sync=false"),

                # UDP variants (lower latency, less reliable over Wi-Fi)
                (f"rtspsrc location={self.url} {udp_flag} latency=100 ! "
                 f"rtpjitterbuffer drop-on-late=true ! rtph264depay ! h264parse ! "
                 f"avdec_h264 ! videoconvert ! videoscale ! "
                 f"video/x-raw, width={w}, height={h}, format=BGR ! "
                 f"appsink max-buffers=1 drop=true sync=false"),
            ]
            for p in pipes:
                cap = self._gst_try_open(p)
                if cap is not None:
                    return cap
            print("[WARN] All GStreamer attempts failed; falling back to FFMPEG…")
        else:
            print("[INFO] OpenCV built without GStreamer (or not detected). Using FFMPEG fallback…")

        # 2) FFMPEG fallbacks (TCP then UDP)
        cap = self._ffmpeg_try_open(self.url, use_tcp=True, bufsize=3)
        if cap is not None:
            return cap
        cap = self._ffmpeg_try_open(self.url, use_tcp=False, bufsize=3)
        if cap is not None:
            return cap

        print("[ERROR] Could not open RTSP with GStreamer or FFMPEG.")
        return None

    def run(self):
        while not self.stop_evt.is_set():
            if self.cap is None:
                self.cap = self.open()
                if self.cap is None:
                    time.sleep(self.reconnect_delay); continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                try: self.cap.release()
                except Exception: pass
                self.cap = None
                time.sleep(self.reconnect_delay); continue
            with self.lock:
                self.latest = frame

    def get_latest(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.stop_evt.set()
        try:
            if self.cap is not None: self.cap.release()
        except Exception:
            pass


class WebcamReader(threading.Thread):
    def __init__(self, cam_id=0, size=(640,480)):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.size = size
        self.cap = None
        self.latest = None
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()

    def open(self):
        # On Windows, CAP_DSHOW is often the most reliable
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        if self.size:
            w, h = self.size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap if cap.isOpened() else None

    def run(self):
        while not self.stop_evt.is_set():
            if self.cap is None:
                self.cap = self.open()
                if self.cap is None:
                    time.sleep(0.5); continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                try: self.cap.release()
                except Exception: pass
                self.cap = None
                time.sleep(0.2); continue
            with self.lock:
                self.latest = frame

    def get_latest(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.stop_evt.set()
        try:
            if self.cap is not None: self.cap.release()
        except Exception:
            pass

# ==========================
# Main
# ==========================
def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Loading model …")
    model = PlateYOLO6(MODEL_PATH)

    # Choose reader
    if SOURCE.lower() == "rtsp":
        print(f"[INFO] Source: RTSP ({RTSP_URL})  decode {GST_SIZE if USE_GSTREAMER else 'default'}")
        reader = RTSPReader(RTSP_URL)
    elif SOURCE.lower() == "webcam":
        print(f"[INFO] Source: Webcam (/dev/video{WEBCAM_ID}) size={WEBCAM_SIZE}")
        reader = WebcamReader(WEBCAM_ID, WEBCAM_SIZE)
    else:
        print("[ERROR] SOURCE must be 'rtsp' or 'webcam'")
        sys.exit(2)

    reader.start()

    writer = None
    fps_tick = time.time(); frames=0; fps=0.0
    frame_interval = 1.0/TARGET_FPS if TARGET_FPS>0 else 0.0

    def cleanup(*_):
        reader.stop(); reader.join(timeout=2.0)
        if writer is not None: writer.release()
        if SHOW_WINDOW: cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("[INFO] Running … (ESC to quit)")
    try:
        while True:
            t0 = time.time()
            frame = reader.get_latest()
            if frame is None:
                time.sleep(0.01); continue

            H,W = frame.shape[:2]
            # Lazy-init writer
            if writer is None and OUTPUT_MP4:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, 20.0, (W, H))
                if not writer.isOpened():
                    print("[WARN] Could not open writer; disabling save.")
                    writer = None

            # Inference
            dets = model.infer(frame)

            # Draw
            if len(dets)==0 and DEBUG:
                cv2.putText(frame, "NO DETECTIONS", (10, H-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            for (x1,y1,x2,y2,sc,clsid) in dets:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                lbl = f"PLATE {sc:.2f}" if TARGET_CLASS_ID is None else f"CLS{clsid} {sc:.2f}"
                cv2.putText(frame, lbl, (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

            # FPS
            frames += 1
            if (time.time()-fps_tick) >= 1.0:
                fps = frames / (time.time()-fps_tick)
                fps_tick = time.time(); frames = 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if writer is not None: writer.write(frame)
            if SHOW_WINDOW:
                cv2.imshow("Plate Detection (Pi5)", frame)
                if cv2.waitKey(1) & 0xFF == 27: break  # ESC

            if frame_interval>0:
                dt = time.time()-t0
                if dt < frame_interval: time.sleep(frame_interval-dt)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
