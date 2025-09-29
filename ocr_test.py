#!/usr/bin/env python3
import os, sys, time, signal, threading, re, collections
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
from paddleocr import PaddleOCR

# ========= Config (edit as needed) =========
MODEL_DET_PATH = r"plate-yolo-data-384-without-rect-3.onnx"

SOURCE       = "webcam"   # "webcam" | "rtsp"
WEBCAM_ID    = 0
WEBCAM_SIZE  = (640, 480)

RTSP_URL     = "rtsp://user:pass@CAMERA_IP:554/Streaming/Channels/102"
USE_GSTREAMER = False
GST_SIZE      = (640, 360)

SHOW_WINDOW  = True
OUTPUT_MP4   = ""         # e.g. r"C:\temp\plates.mp4" or "" to disable
TARGET_FPS   = 0          # 0 = unlimited

# Detector thresholds
CONF_THRES   = 0.25
IOU_THRES    = 0.45
PAD_VALUE    = 114
REPLICATE_GRAY_TO_3 = True

# PaddleOCR
PADDLE_LANG  = "en"
PADDLE_THREADS = 4
PADDLE_REC_MODEL_DIR = None  # None = built-in

# Async OCR behavior
OCR_MAX_FPS        = 4.0      # upper bound for OCR rate
OCR_TOPK_PER_FRAME = 1        # submit at most N plates per frame (usually 1 is enough)
OCR_MATCH_IOU      = 0.3      # to attach cached OCR text to current boxes
RESULT_CACHE_SIZE  = 32       # recent OCR results kept for matching
DEBUG = False
def dprint(*a):
    if DEBUG: print("[DEBUG]", *a)

# Tame math libs (good on Pi, harmless on Windows)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["FLAGS_minloglevel"] = "2"

# ========= Utils =========
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
        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
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

def iou(a, b):
    # a,b = (x1,y1,x2,y2)
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-9
    return inter/ua

# ========= Detector (ONNX, [1,3,384,384] -> [1,9072,6] in model pixels) =========
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
        scores = obj * last if np.all((last >= 0.0) & (last <= 1.0)) else obj

        keep = scores >= CONF_THRES
        boxes, scores = boxes[keep], scores[keep]
        if boxes.size == 0: return []

        cx,cy,w,h = boxes.T
        # center
        xyxy_center = np.stack(
            [(cx - w/2)*(W/self.in_w), (cy - h/2)*(H/self.in_h),
             (cx + w/2)*(W/self.in_w), (cy + h/2)*(H/self.in_h)], 1)
        # top-left
        xyxy_tleft  = np.stack(
            [cx*(W/self.in_w), cy*(H/self.in_h),
             (cx+w)*(W/self.in_w), (cy+h)*(H/self.in_h)], 1)

        # choose plausible (area sanity)
        def plaus(xyxy):
            if xyxy is None or len(xyxy)==0: return -1
            wv = np.maximum(0, xyxy[:,2]-xyxy[:,0]); hv = np.maximum(0, xyxy[:,3]-xyxy[:,1])
            area = (wv*hv)/(W*H+1e-9)
            return ((wv>6)&(hv>6)&(area<0.6)).mean()
        xyxy = xyxy_center if plaus(xyxy_center) >= plaus(xyxy_tleft) else xyxy_tleft

        xyxy[:,0::2] = np.clip(xyxy[:,0::2], 0, W-1)
        xyxy[:,1::2] = np.clip(xyxy[:,1::2], 0, H-1)

        idx = nms(xyxy, scores, IOU_THRES)
        xyxy, scores = xyxy[idx], scores[idx]
        return [(int(a),int(b),int(c),int(d), float(s)) for (a,b,c,d),s in zip(xyxy, scores)]

    def infer(self, frame_bgr):
        x = self.preprocess(frame_bgr)
        y = self.sess.run([self.out_name], {self.inp.name: x})[0]
        return self.postprocess(y, frame_bgr.shape)

# ========= Indian plate normalization (LL DD [L{1,2}] DDDD) =========
LETTER_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGIT_SET  = set("0123456789")
CONFUSION_TO_DIGIT  = {'O':'0','D':'0','Q':'0','I':'1','L':'1','|':'1','!':'1','Z':'2','S':'5','B':'8'}
CONFUSION_TO_LETTER = {'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B'}

def normalize_indian_plate(raw: str, pretty_space=False) -> str:
    if not raw: return ""
    s = re.sub(r"[^A-Za-z0-9]", "", raw.upper())
    if len(s) < 7 or len(s) > 10:
        return s
    chars = list(s)
    # LL
    for i in range(min(2,len(chars))):
        c = chars[i]
        if c not in LETTER_SET and c in CONFUSION_TO_LETTER:
            chars[i] = CONFUSION_TO_LETTER[c]
    # DD
    for i in range(2, min(4,len(chars))):
        c = chars[i]
        if c not in DIGIT_SET and c in CONFUSION_TO_DIGIT:
            chars[i] = CONFUSION_TO_DIGIT[c]
    rem = chars[4:]
    if len(rem) >= 4:
        # last 4 digits
        for j in range(len(rem)-4, len(rem)):
            c = rem[j]
            if c not in DIGIT_SET and c in CONFUSION_TO_DIGIT:
                rem[j] = CONFUSION_TO_DIGIT[c]
        series_len = min(max(0, len(rem)-4), 2)
        ser = rem[:series_len]
        for k in range(len(ser)):
            c = ser[k]
            if c not in LETTER_SET and c in CONFUSION_TO_LETTER:
                ser[k] = CONFUSION_TO_LETTER[c]
            if ser[k] in DIGIT_SET:
                ser[k] = CONFUSION_TO_LETTER.get(ser[k], ser[k])
        fixed = chars[:4] + ser + rem[series_len:]
    else:
        fixed = chars
    out = "".join(fixed)
    return out[:4] + " " + out[4:] if pretty_space and len(out)>=6 else out

# ========= PaddleOCR rec-only =========
class PaddleRec:
    def __init__(self, lang="en", model_dir=None, threads=4):
        self.ocr = PaddleOCR(
            use_angle_cls=True, lang=lang, det=False, rec=True,
            rec_model_dir=model_dir, use_gpu=False, cpu_threads=threads, drop_score=0.5
        )
        print(f"[INFO] PaddleOCR rec-only init: lang={lang} threads={threads}")

    def rec_single(self, roi_bgr):
        img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        r = self.ocr.ocr(img, det=False, rec=True, cls=True)
        if not r: return ""
        first = r[0][0] if isinstance(r[0], (list,tuple)) else ["",0.0]
        txt = first[0] if isinstance(first, (list,tuple)) and len(first)>0 else ""
        score = float(first[1]) if isinstance(first, (list,tuple)) and len(first)>1 else 0.0
        return normalize_indian_plate(txt) if score>=0.25 else ""

# ========= Async OCR worker (latest-only, frame dropping) =========
class AsyncOCR:
    """
    - Maintains ONLY the latest submitted crops (up to N top-k per frame).
    - If new crops come while OCR is busy, old ones are discarded.
    - Caches recent (bbox,text) with timestamps; main loop matches by IoU.
    """
    def __init__(self, threads=1, max_fps=4.0):
        self.rec = PaddleRec(PADDLE_LANG, PADDLE_REC_MODEL_DIR, threads=PADDLE_THREADS)
        self.latest = None  # ([(roi_bgr, bbox)], t_submit)
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.results = collections.deque(maxlen=RESULT_CACHE_SIZE)  # (bbox, text, t)
        self.min_interval = 1.0/max_fps if max_fps>0 else 0.0
        self.last_run = 0.0
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, crops_with_boxes):
        """
        crops_with_boxes: list of (roi_bgr, (x1,y1,x2,y2))
        Only the latest set is kept; older pending work is dropped.
        """
        if not crops_with_boxes: return
        now = time.time()
        if self.min_interval>0 and (now - self.last_run) < self.min_interval:
            return  # rate limit
        with self.lock:
            self.latest = (crops_with_boxes, now)

    def _loop(self):
        while not self.stop_evt.is_set():
            task = None
            with self.lock:
                if self.latest is not None:
                    task = self.latest
                    self.latest = None    # drop any older pending work
            if task is None:
                time.sleep(0.005)
                continue
            crops_with_boxes, tsub = task
            self.last_run = time.time()

            # Run OCR sequentially (1 thread) to keep CPU low; you can parallelize inside if needed.
            for roi, bbox in crops_with_boxes:
                # light enhancement helps
                g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                g = cv2.bilateralFilter(g, 5, 30, 30)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                g = clahe.apply(g)
                roi_enh = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                txt = self.rec.rec_single(roi_enh)
                if txt:
                    self.results.append((bbox, txt, time.time()))
        # end while

    def get_text_for_bbox(self, bbox):
        # find best IoU match from recent results
        best_txt, best_iou = "", 0.0
        for bb, txt, _ in list(self.results)[::-1]:  # newest first
            ov = iou(bbox, bb)
            if ov > best_iou and ov >= OCR_MATCH_IOU:
                best_iou, best_txt = ov, txt
                if ov > 0.8: break
        return best_txt

    def stop(self):
        self.stop_evt.set()
        self.worker.join(timeout=1.0)

# ========= Readers =========
class WebcamReader(threading.Thread):
    def __init__(self, cam_id=0, size=(640,480)):
        super().__init__(daemon=True)
        self.cam_id = cam_id; self.size = size
        self.cap = None; self.latest = None
        self.lock = threading.Lock(); self.stop_evt = threading.Event()

    def open(self):
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)  # on Windows; try CAP_MSMF if needed
        if self.size:
            w,h = self.size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap if cap.isOpened() else None

    def run(self):
        while not self.stop_evt.is_set():
            if self.cap is None:
                self.cap = self.open()
                if self.cap is None: time.sleep(0.5); continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                try: self.cap.release()
                except: pass
                self.cap = None
                time.sleep(0.1); continue
            with self.lock:
                self.latest = frame

    def get_latest(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.stop_evt.set()
        try:
            if self.cap is not None: self.cap.release()
        except: pass

class RTSPReader(threading.Thread):
    def __init__(self, url, reconnect_delay=0.5):
        super().__init__(daemon=True)
        self.url = url; self.reconnect_delay = reconnect_delay
        self.cap = None; self.latest=None
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
                if self.cap is None: time.sleep(self.reconnect_delay); continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                try: self.cap.release()
                except: pass
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
        except: pass

# ========= Main =========
def main():
    if not Path(MODEL_DET_PATH).exists():
        print(f"[ERROR] Detector model not found: {MODEL_DET_PATH}", file=sys.stderr); sys.exit(1)
    det = PlateDetector(MODEL_DET_PATH)

    # Source
    if SOURCE.lower()=="webcam":
        print(f"[INFO] Source: webcam id={WEBCAM_ID} size={WEBCAM_SIZE}")
        reader = WebcamReader(WEBCAM_ID, WEBCAM_SIZE)
    else:
        print(f"[INFO] Source: RTSP {RTSP_URL}")
        reader = RTSPReader(RTSP_URL)
    reader.start()

    # Async OCR worker
    ocrw = AsyncOCR(threads=1, max_fps=OCR_MAX_FPS)

    writer=None
    fps_tick=time.time(); frames=0; fps=0.0
    frame_interval = 1.0/TARGET_FPS if TARGET_FPS>0 else 0.0

    def cleanup(*_):
        reader.stop(); reader.join(timeout=1.5)
        ocrw.stop()
        if writer is not None: writer.release()
        if SHOW_WINDOW: cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup); signal.signal(signal.SIGTERM, cleanup)

    print("[INFO] Running… (ESC to quit)")
    try:
        while True:
            t0 = time.time()
            frame = reader.get_latest()
            if frame is None:
                time.sleep(0.002); continue

            H,W = frame.shape[:2]
            if writer is None and OUTPUT_MP4:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, 20.0, (W, H))
                if not writer.isOpened(): print("[WARN] writer open failed"); writer=None

            # 1) Fast detection (main thread)
            dets = det.infer(frame)

            # 2) Submit up to TOPK crops to OCR worker (non-blocking, latest-only)
            #    Pick largest/most confident to stabilize text
            if dets:
                dets_sorted = sorted(dets, key=lambda d:(d[4], (d[2]-d[0])*(d[3]-d[1])), reverse=True)
                submit = []
                for (x1,y1,x2,y2,sc) in dets_sorted[:OCR_TOPK_PER_FRAME]:
                    px = int(0.03*(x2-x1)); py = int(0.20*(y2-y1))
                    xa, xb = max(0, x1-px), min(W-1, x2+px)
                    ya, yb = max(0, y1-py), min(H-1, y2+py)
                    roi = frame[ya:yb, xa:xb]
                    if roi.size>0:
                        submit.append((roi, (x1,y1,x2,y2)))
                ocrw.submit(submit)

            # 3) Draw detections + attach latest OCR text from cache by IoU
            for (x1,y1,x2,y2,sc) in dets:
                txt = ocrw.get_text_for_bbox((x1,y1,x2,y2))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                label = txt if len(txt)>=3 else f"PLATE {sc:.2f}"
                cv2.putText(frame, label, (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

            # 4) FPS overlay
            frames += 1
            if (time.time()-fps_tick) >= 1.0:
                fps = frames / (time.time()-fps_tick)
                fps_tick = time.time(); frames=0
            cv2.putText(frame, f"FPS(det): {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if writer is not None: writer.write(frame)
            if SHOW_WINDOW:
                cv2.imshow("Async OCR (drop stale) - Plates", frame)
                if cv2.waitKey(1) & 0xFF == 27: break

            if frame_interval>0:
                dt = time.time()-t0
                if dt < frame_interval: time.sleep(frame_interval-dt)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
