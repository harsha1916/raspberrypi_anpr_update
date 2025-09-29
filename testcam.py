#!/usr/bin/env python3
# Works on Windows laptop (webcam) AND Raspberry Pi (RTSP/webcam)
import os, sys, time, signal, threading
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

# ==========================
# HARD-CODE CONFIG
# ==========================
# -- Detector (already working for you) --
MODEL_DET_PATH = r"plate-yolo-data-384-without-rect-3.onnx"  # <-- EDIT (Windows) or "/home/pi5/..."
CONF_THRES     = 0.25
IOU_THRES      = 0.45
PAD_VALUE      = 114
REPLICATE_GRAY_TO_3 = True   # helps if detector trained on grayscale

# -- OCR: choose "onnx", "tesseract", or "none"
USE_OCR = "onnx"   # "onnx" | "tesseract" | "none"

# ONNX OCR (CRNN/CTC style) paths
OCR_MODEL_PATH  = r"plate-yolo-data-384-without-rect-3.onnx"   # <-- EDIT if you have it
LABELS_PATH     = r"labels.txt"      # one symbol per line (no blank)

# Tesseract note:
# Windows: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
# and set TESSERACT_EXE below if needed, e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_EXE   = r""  # leave empty if tesseract is on PATH

# -- Source select --
SOURCE       = "webcam"  # "webcam" | "rtsp"
WEBCAM_ID    = 0
WEBCAM_SIZE  = (640, 480)  # or (640,480)

RTSP_URL     = "rtsp://user:pass@CAMERA_IP:554/Streaming/Channels/102"
USE_GSTREAMER = False         # Windows: False
GST_SIZE      = (640, 360)    # used if USE_GSTREAMER=True (Linux/Pi)

# -- UI / Recording --
SHOW_WINDOW  = True
OUTPUT_MP4   = ""   # e.g. r"C:\temp\plates_out.mp4" or "" to disable
TARGET_FPS   = 0    # 0 = unlimited

DEBUG = False
def dprint(*a):
    if DEBUG: print("[DEBUG]", *a)

# modest threading for math libs (Pi-friendly, harmless on Windows)
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
    best = None; best_score = -1.0
    for xyxy in xyxy_list:
        if xyxy is None or len(xyxy)==0: continue
        w = np.maximum(0, xyxy[:,2]-xyxy[:,0])
        h = np.maximum(0, xyxy[:,3]-xyxy[:,1])
        area = (w*h)/(W*H+1e-9)
        valid = (w>6) & (h>6) & (area < 0.6)
        score = valid.mean() + 0.05*(scores.mean() if len(scores) else 0.0)
        if score > best_score:
            best, best_score = xyxy, score
    return best

# ==========================
# Detector (output [1, 9072, 6])
# ==========================
class PlateDetector:
    def __init__(self, model_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.out_name = self.sess.get_outputs()[0].name
        ishape = self.inp.shape  # [1,3,384,384]
        self.in_h = ishape[2] if isinstance(ishape[2], int) else 384
        self.in_w = ishape[3] if isinstance(ishape[3], int) else 384
        print(f"[INFO] Detector input 3x{self.in_h}x{self.in_w}, output {self.out_name}")

    def preprocess(self, frame_bgr):
        if REPLICATE_GRAY_TO_3:
            g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            img3 = cv2.merge([g,g,g])
        else:
            img3 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        im = letterbox(img3, (self.in_h, self.in_w), PAD_VALUE)
        x = im.astype(np.float32)/255.0
        x = np.transpose(x, (2,0,1))[None,...]  # [1,3,H,W]
        return x

    def postprocess(self, out, frame_shape):
        H,W = frame_shape[:2]
        y = out[0]  # [N,6]
        if y.ndim != 2 or y.shape[1] != 6: return []
        boxes = y[:, :4].astype(np.float32)
        obj   = y[:, 4].astype(np.float32)
        last  = y[:, 5].astype(np.float32)

        # last is prob (0..1) in your dump; use obj*prob
        if np.all((last >= 0.0) & (last <= 1.0)):
            scores = obj * last
            cls_ids = (last > 0.5).astype(np.int32)
        else:
            scores = obj
            cls_ids = np.rint(last).astype(np.int32)

        keep = scores >= CONF_THRES
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if boxes.size == 0: return []

        # These coords are in model pixels @ 384×384. Try both center and top-left, scale to frame.
        cx,cy,w,h = boxes.T
        xyxy_center = np.stack(
            [(cx - w/2)*(W/self.in_w),
             (cy - h/2)*(H/self.in_h),
             (cx + w/2)*(W/self.in_w),
             (cy + h/2)*(H/self.in_h)], 1)
        xyxy_tleft = np.stack(
            [cx*(W/self.in_w),
             cy*(H/self.in_h),
             (cx+w)*(W/self.in_w),
             (cy+h)*(H/self.in_h)], 1)

        xyxy = plausible_pick([xyxy_center, xyxy_tleft], scores, W, H)
        if xyxy is None: return []
        xyxy[:,0::2] = np.clip(xyxy[:,0::2], 0, W-1)
        xyxy[:,1::2] = np.clip(xyxy[:,1::2], 0, H-1)

        idx = nms(xyxy, scores, IOU_THRES)
        xyxy, scores, cls_ids = xyxy[idx], scores[idx], cls_ids[idx]
        return [(int(a),int(b),int(c),int(d), float(s), int(k)) for (a,b,c,d),s,k in zip(xyxy, scores, cls_ids)]

    def infer(self, frame_bgr):
        x = self.preprocess(frame_bgr)
        y = self.sess.run([self.out_name], {self.inp.name: x})[0]
        return self.postprocess(y, frame_bgr.shape)

# ==========================
# OCR (ONNX CRNN + CTC greedy)
# ==========================
class OCROnnx:
    def __init__(self, model_path, labels_path):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"OCR model not found: {model_path}")
        self.labels = self._read_labels(labels_path)
        self.blank_idx = len(self.labels)  # assume C = len(labels)+1 with blank at last; we’ll auto-calc later
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.inp  = self.sess.get_inputs()[0]
        ishape = self.inp.shape  # try [1,1,H,W] or [1,H,W,1]
        self.nchw = True
        if len(ishape)==4:
            # guess nchw if channel is at index 1
            self.nchw = (not isinstance(ishape[1], str)) and (ishape[1] in (1,3))
            if self.nchw:
                self.in_c = ishape[1] if isinstance(ishape[1], int) else 1
                self.in_h = ishape[2] if isinstance(ishape[2], int) else 160
                self.in_w = ishape[3] if isinstance(ishape[3], int) else 256
            else:
                self.in_h = ishape[1] if isinstance(ishape[1], int) else 160
                self.in_w = ishape[2] if isinstance(ishape[2], int) else 256
                self.in_c = ishape[3] if isinstance(ishape[3], int) else 1
        else:
            # fallback
            self.in_c, self.in_h, self.in_w = 1, 160, 256
        print(f"[INFO] OCR input {'NCHW' if self.nchw else 'NHWC'} C={self.in_c} H={self.in_h} W={self.in_w}")

    @staticmethod
    def _read_labels(path):
        if not Path(path).exists():
            # safe fallback: digits + uppercase
            print(f"[WARN] labels.txt missing at {path}; using 0-9A-Z fallback")
            return list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _prep_roi(self, roi_bgr):
        # Convert to grayscale, normalize contrast a bit
        g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.bilateralFilter(g, 5, 30, 30)  # denoise but keep edges
        # Resize with padding to expected input
        Ht, Wt = self.in_h, self.in_w
        h, w = g.shape[:2]
        r = min(Ht/h, Wt/w)
        nh, nw = int(round(h*r)), int(round(w*r))
        g2 = cv2.resize(g, (nw, nh), cv2.INTER_LINEAR)
        top = (Ht - nh)//2; bottom = Ht - nh - top
        left = (Wt - nw)//2; right = Wt - nw - left
        g2 = cv2.copyMakeBorder(g2, top,bottom,left,right, cv2.BORDER_CONSTANT, value=0)
        x = (g2.astype(np.float32)/255.0)
        if self.nchw:
            x = x[None, None, ...]  # [1,1,H,W]
        else:
            x = x[..., None][None, ...]  # [1,H,W,1]
        return x

    def _ctc_greedy(self, logits):
        # Normalize to [T, C]
        arr = np.asarray(logits)
        # squeeze batch dims of size 1
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        if arr.ndim == 3:
            # [T,1,C] or [1,T,C]
            if arr.shape[1] == 1:
                arr = np.transpose(arr, (1,0,2))[0]  # [T,C]
            elif arr.shape[0] == 1:
                arr = arr[0]
        if arr.ndim != 2:
            # sometimes OCR exports as [C, T]; transpose if needed
            if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
        T, C = arr.shape
        # guess blank index
        if C == len(self.labels)+1:
            blank = C-1
        elif C == len(self.labels):
            blank = 0  # many toolchains place blank at 0 if labels already include it
        else:
            blank = C-1
        pred = arr.argmax(axis=-1).astype(np.int64).flatten()
        out = []
        prev = -1
        for p in pred.tolist():
            if p != blank and p != prev:
                out.append(p)
            prev = p
        # safe map
        text = "".join(self.labels[i] for i in out if 0 <= i < len(self.labels))
        return text

    def infer(self, roi_bgr):
        x = self._prep_roi(roi_bgr)
        y = self.sess.run(None, {self.inp.name: x})
        logits = y[0] if len(y)==1 else y[0]
        return self._ctc_greedy(logits)

# ==========================
# Frame readers
# ==========================
class WebcamReader(threading.Thread):
    def __init__(self, cam_id=0, size=(1280,720)):
        super().__init__(daemon=True)
        self.cam_id = cam_id; self.size = size
        self.cap = None; self.latest = None
        self.lock = threading.Lock(); self.stop_evt = threading.Event()

    def open(self):
        # Windows: CAP_DSHOW tends to be reliable; try CAP_MSMF if needed
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        if self.size:
            w,h = self.size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
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

class RTSPReader(threading.Thread):
    def __init__(self, url, reconnect_delay=1.0):
        super().__init__(daemon=True)
        self.url = url; self.reconnect_delay = reconnect_delay
        self.cap = None; self.latest = None
        self.lock = threading.Lock(); self.stop_evt = threading.Event()

    def open(self):
        if USE_GSTREAMER:
            w,h = GST_SIZE
            gst = (f"rtspsrc location={self.url} latency=100 ! "
                   f"rtph264depay ! h264parse ! avdec_h264 ! "
                   f"videoscale ! video/x-raw, width={w}, height={h}, format=BGR ! "
                   f"appsink drop=true sync=false")
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        return cap if cap.isOpened() else None

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

# ==========================
# Main
# ==========================
def main():
    # Detector
    if not Path(MODEL_DET_PATH).exists():
        print(f"[ERROR] Detector model not found: {MODEL_DET_PATH}", file=sys.stderr); sys.exit(1)
    det = PlateDetector(MODEL_DET_PATH)

    # OCR
    ocr = None
    if USE_OCR.lower() == "onnx":
        try:
            ocr = OCROnnx(OCR_MODEL_PATH, LABELS_PATH)
        except Exception as e:
            print(f"[WARN] ONNX OCR unavailable: {e}. Falling back to Tesseract if configured.")
            USE = "tesseract"
        else:
            USE = "onnx"
    elif USE_OCR.lower() == "tesseract":
        USE = "tesseract"
    else:
        USE = "none"

    if USE == "tesseract":
        try:
            import pytesseract
            if TESSERACT_EXE:
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
        except Exception as e:
            print(f"[WARN] pytesseract not available: {e}. OCR disabled.")
            USE = "none"

    # Source
    if SOURCE.lower() == "webcam":
        print(f"[INFO] Source: webcam id={WEBCAM_ID} size={WEBCAM_SIZE}")
        reader = WebcamReader(WEBCAM_ID, WEBCAM_SIZE)
    else:
        print(f"[INFO] Source: RTSP {RTSP_URL}")
        reader = RTSPReader(RTSP_URL)
    reader.start()

    # Writer
    writer = None
    fps_tick = time.time(); frames=0; fps=0.0
    frame_interval = 1.0/TARGET_FPS if TARGET_FPS>0 else 0.0

    def cleanup(*_):
        reader.stop(); reader.join(timeout=2.0)
        if writer is not None: writer.release()
        if SHOW_WINDOW: cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("[INFO] Running… (ESC to quit)")
    try:
        while True:
            t0 = time.time()
            frame = reader.get_latest()
            if frame is None:
                time.sleep(0.01); continue

            H,W = frame.shape[:2]
            if writer is None and OUTPUT_MP4:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, 20.0, (W, H))
                if not writer.isOpened():
                    print("[WARN] Could not open writer; disabling save.")
                    writer = None

            # Detect
            dets = det.infer(frame)

            # Draw + OCR
            for (x1,y1,x2,y2,sc,clsid) in dets:
                # pad crop a little (helps OCR)
                px = int(0.03*(x2-x1)); py = int(0.15*(y2-y1))  # more padding vertically
                xa = max(0, x1 - px); xb = min(W-1, x2 + px)
                ya = max(0, y1 - py); yb = min(H-1, y2 + py)
                roi = frame[ya:yb, xa:xb].copy()

                plate_text = ""
                if USE == "onnx" and ocr is not None:
                    plate_text = ocr.infer(roi)
                elif USE == "tesseract":
                    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    g = cv2.bilateralFilter(g, 5, 30, 30)
                    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    import pytesseract
                    # Whitelist typical plate chars (tweak for your locale)
                    config = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    plate_text = pytesseract.image_to_string(bw, config=config).strip()

                # Draw
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                label = f"{plate_text}" if plate_text else f"PLATE {sc:.2f}"
                cv2.putText(frame, label, (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

            # FPS
            frames += 1
            if (time.time()-fps_tick) >= 1.0:
                fps = frames / (time.time()-fps_tick)
                fps_tick = time.time(); frames = 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if writer is not None: writer.write(frame)
            if SHOW_WINDOW:
                cv2.imshow("Plate + OCR", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if frame_interval>0:
                dt = time.time()-t0
                if dt < frame_interval: time.sleep(frame_interval-dt)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
