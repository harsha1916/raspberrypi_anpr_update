#!/usr/bin/env python3
import os, sys, time, signal, threading, collections, re
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

# ====== CONFIG ======
MODEL_PATH   = "plate-yolo-data-384-without-rect-3.onnx"

# Source
SOURCE       = "rtsp"   # "rtsp" | "webcam"
RTSP_URL     = "rtsp://admin:admin@192.168.1.201:554/avstream/channel=1/stream=1.sdp"
USE_GSTREAMER= True
GST_SIZE     = (640, 360)

WEBCAM_ID    = 0
WEBCAM_SIZE  = (640, 480)

# UI / Output
SHOW_WINDOW  = True
OUTPUT_MP4   = ""   # e.g. "/home/pi5/plates_sort.mp4"
TARGET_FPS   = 0

# Detector thresholds
CONF_THRES   = 0.25
IOU_THRES    = 0.45
PAD_VALUE    = 114
REPLICATE_GRAY_TO_3 = True

# SORT params (default good start)
SORT_MAX_AGE    = 10
SORT_MIN_HITS   = 3
SORT_IOU_THRESH = 0.3

# Async OCR (PaddleOCR)
USE_PADDLE_OCR  = True     # set False if you haven’t installed paddleocr on Windows
PADDLE_LANG     = "en"
PADDLE_THREADS  = 4
PADDLE_REC_MODEL_DIR = None  # None = built-in

# OCR rate limit & stability
OCR_MAX_FPS           = 3.0      # total OCR rate cap (background thread)
OCR_PER_TRACK_COOLDOWN= 0.7      # seconds between OCR tries per track
OCR_TOPK_TRACKS       = 2        # max tracks to submit per frame
PLATE_STABLE_WINDOW   = 5        # keep last N OCR reads per track
PLATE_ACCEPT_COUNT    = 2        # require this many matches in window to “lock”

DEBUG = False
def dprint(*a):
    if DEBUG: print("[DEBUG]", *a)

# tame math libs
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["FLAGS_minloglevel"] = "2"

# ====== SORT import ======
try:
    from sort import Sort  # clone https://github.com/abewley/sort and put sort.py beside this script
except ImportError:
    print("[ERROR] Could not import 'sort'. Place sort.py from abewley/sort next to this script.")
    sys.exit(2)

# ====== Utils ======
def letterbox(img, new_shape, pad_value=114):
    Ht, Wt = new_shape
    h, w = img.shape[:2]
    r = min(Ht/h, Wt/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    im = cv2.resize(img, (nw, nh), cv2.INTER_LINEAR)
    top = (Ht - nh)//2; bottom = Ht - nh - top
    left = (Wt - nw)//2; right = Wt - nw - left
    if img.ndim == 2:
        out = cv2.copyMakeBorder(im, top,bottom,left,right, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        out = cv2.copyMakeBorder(im, top,bottom,left,right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    return out

def nms(boxes, scores, iou_thres=0.45):
    if len(boxes)==0: return []
    boxes = boxes.astype(np.float32)
    x1,y1,x2,y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size>0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        inter = w*h
        ovr = inter/(areas[i]+areas[order[1:]]-inter + 1e-9)
        inds = np.where(ovr<=iou_thres)[0]
        order = order[inds+1]
    return keep

def plausible_pick(xyxy_list, scores, W, H):
    best, best_score = None, -1
    for xyxy in xyxy_list:
        if xyxy is None or len(xyxy)==0: continue
        w = np.maximum(0, xyxy[:,2]-xyxy[:,0])
        h = np.maximum(0, xyxy[:,3]-xyxy[:,1])
        area = (w*h)/(W*H+1e-9)
        valid = (w>6)&(h>6)&(area<0.6)
        score = valid.mean() + 0.05*(scores.mean() if len(scores) else 0.0)
        if score>best_score:
            best, best_score = xyxy, score
    return best

# ====== Detector (ONNX, [1,3,384,384] -> [1,9072,6]) ======
class PlateYOLO6:
    def __init__(self, model_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.out_name = self.sess.get_outputs()[0].name
        ishape = self.inp.shape
        self.in_h = ishape[2] if isinstance(ishape[2], int) else 384
        self.in_w = ishape[3] if isinstance(ishape[3], int) else 384
        print(f"[INFO] Model: input 3x{self.in_h}x{self.in_w}, output {self.out_name}")

    def preprocess(self, frame_bgr):
        if REPLICATE_GRAY_TO_3:
            g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            img3 = cv2.merge([g,g,g])
        else:
            img3 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        im = letterbox(img3, (self.in_h, self.in_w), PAD_VALUE)
        x = im.astype(np.float32)/255.0
        x = np.transpose(x, (2,0,1))[None,...]
        return x

    def postprocess(self, out, frame_shape):
        H,W = frame_shape[:2]
        y = out[0]  # [N,6]
        if y.ndim != 2 or y.shape[1] != 6: return []
        boxes = y[:, :4].astype(np.float32)
        obj   = y[:, 4].astype(np.float32)
        last  = y[:, 5].astype(np.float32)
        scores = obj * last if np.all((last>=0.0)&(last<=1.0)) else obj
        keep = scores >= CONF_THRES
        boxes, scores = boxes[keep], scores[keep]
        if boxes.size == 0: return []

        cx,cy,w,h = boxes.T
        xyxy_c = np.stack([(cx-w/2)*(W/self.in_w),(cy-h/2)*(H/self.in_h),
                           (cx+w/2)*(W/self.in_w),(cy+h/2)*(H/self.in_h)],1)
        xyxy_t = np.stack([cx*(W/self.in_w),cy*(H/self.in_h),
                           (cx+w)*(W/self.in_w),(cy+h)*(H/self.in_h)],1)
        xyxy = plausible_pick([xyxy_c, xyxy_t], scores, W, H)
        if xyxy is None: return []
        xyxy[:,0::2] = np.clip(xyxy[:,0::2], 0, W-1)
        xyxy[:,1::2] = np.clip(xyxy[:,1::2], 0, H-1)
        idx = nms(xyxy, scores, IOU_THRES)
        xyxy, scores = xyxy[idx], scores[idx]
        # return tuples x1,y1,x2,y2,score
        return [(int(a),int(b),int(c),int(d),float(s)) for (a,b,c,d),s in zip(xyxy, scores)]

    def infer(self, frame_bgr):
        x = self.preprocess(frame_bgr)
        y = self.sess.run([self.out_name], {self.inp.name: x})[0]
        return self.postprocess(y, frame_bgr.shape)

# ====== Indian plate normalization (same as earlier) ======
LETTER_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGIT_SET  = set("0123456789")
CONFUSION_TO_DIGIT  = {'O':'0','D':'0','Q':'0','I':'1','L':'1','|':'1','!':'1','Z':'2','S':'5','B':'8'}
CONFUSION_TO_LETTER = {'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B'}
def normalize_indian_plate(raw: str) -> str:
    if not raw: return ""
    s = re.sub(r"[^A-Za-z0-9]", "", raw.upper())
    if len(s) < 7 or len(s) > 10: return s
    chars = list(s)
    for i in range(min(2,len(chars))):
        c = chars[i]
        if c not in LETTER_SET and c in CONFUSION_TO_LETTER:
            chars[i] = CONFUSION_TO_LETTER[c]
    for i in range(2, min(4,len(chars))):
        c = chars[i]
        if c not in DIGIT_SET and c in CONFUSION_TO_DIGIT:
            chars[i] = CONFUSION_TO_DIGIT[c]
    rem = chars[4:]
    if len(rem) >= 4:
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
    return "".join(fixed)

# ====== PaddleOCR (recognizer-only) ======
if USE_PADDLE_OCR:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("[ERROR] paddleocr not installed. `pip install paddleocr` (and paddlepaddle) or set USE_PADDLE_OCR=False.")
        sys.exit(2)

class PaddleRec:
    def __init__(self, lang="en", model_dir=None, threads=4):
        if not USE_PADDLE_OCR:
            self.ocr = None
            return
        self.ocr = PaddleOCR(
            use_angle_cls=True, lang=lang, det=False, rec=True,
            rec_model_dir=model_dir, use_gpu=False, cpu_threads=threads, drop_score=0.5
        )
        print(f"[INFO] PaddleOCR rec-only: lang={lang} threads={threads}")

    def rec_single(self, roi_bgr):
        # light enhance
        g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.bilateralFilter(g, 5, 30, 30)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(g)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if not self.ocr: return ""
        r = self.ocr.ocr(img, det=False, rec=True, cls=True)
        if not r: return ""
        first = r[0][0] if isinstance(r[0], (list,tuple)) else ["",0.0]
        txt = first[0] if isinstance(first, (list,tuple)) and len(first)>0 else ""
        score = float(first[1]) if isinstance(first, (list,tuple)) and len(first)>1 else 0.0
        return normalize_indian_plate(txt) if score>=0.25 else ""

# ====== Async OCR worker keyed by track_id ======
class AsyncOCR:
    def __init__(self, max_fps=3.0):
        self.rec = PaddleRec(PADDLE_LANG, PADDLE_REC_MODEL_DIR, threads=PADDLE_THREADS)
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.queue_latest = {}   # track_id -> latest roi
        self.cooldown = {}       # track_id -> last_time
        self.results = {}        # track_id -> deque of recent texts
        self.min_interval = 1.0/max_fps if max_fps>0 else 0.0
        self._last_cycle = 0.0
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, track_id, roi):
        now = time.time()
        last = self.cooldown.get(track_id, 0.0)
        if (now - last) < OCR_PER_TRACK_COOLDOWN:
            return
        with self.lock:
            self.queue_latest[track_id] = roi

    def _loop(self):
        while not self.stop_evt.is_set():
            if self.min_interval>0 and (time.time()-self._last_cycle) < self.min_interval:
                time.sleep(0.003); continue
            self._last_cycle = time.time()

            batch = None
            with self.lock:
                if self.queue_latest:
                    # take a snapshot and clear (drop stale)
                    batch = list(self.queue_latest.items())
                    self.queue_latest.clear()
            if not batch:
                time.sleep(0.003); continue

            for tid, roi in batch:
                txt = self.rec.rec_single(roi)
                self.cooldown[tid] = time.time()
                if txt:
                    dq = self.results.setdefault(tid, collections.deque(maxlen=PLATE_STABLE_WINDOW))
                    dq.append(txt)

    def get_stable_text(self, track_id):
        dq = self.results.get(track_id)
        if not dq or len(dq)==0:
            return "", 0
        # majority vote
        counts = {}
        for t in dq:
            if t: counts[t] = counts.get(t,0)+1
        if not counts: return "", 0
        best_txt = max(counts.items(), key=lambda kv: kv[1])[0]
        return (best_txt, counts[best_txt])

    def stop(self):
        self.stop_evt.set()
        self.worker.join(timeout=1.0)

# ====== Readers ======
class RTSPReader(threading.Thread):
    def __init__(self, url, reconnect_delay=0.6, size=(640,360)):
        super().__init__(daemon=True)
        self.url=url; self.reconnect_delay=reconnect_delay; self.size=size
        self.cap=None; self.latest=None
        self.lock=threading.Lock(); self.stop_evt=threading.Event()

    @staticmethod
    def _has_gst():
        try:
            info = cv2.getBuildInformation()
            return "GStreamer" in info and "YES" in info.split("GStreamer")[1].splitlines()[0]
        except Exception: return False

    def open(self):
        w,h = self.size
        if USE_GSTREAMER and self._has_gst():
            pipes = [
                (f"rtspsrc location={self.url} protocols=tcp latency=200 ! "
                 f"rtpjitterbuffer drop-on-late=true ! rtph264depay ! h264parse ! "
                 f"avdec_h264 ! videoconvert ! videoscale ! "
                 f"video/x-raw, width={w}, height={h}, format=BGR ! "
                 f"appsink max-buffers=1 drop=true sync=false")
            ]
            for p in pipes:
                cap = cv2.VideoCapture(p, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    ok,_=cap.read()
                    if ok:
                        print("[INFO] GStreamer pipeline OK")
                        return cap
                if cap: cap.release()
            print("[WARN] GStreamer failed; trying FFMPEG…")
        cap = cv2.VideoCapture(self.url + ("?rtsp_transport=tcp" if "?rtsp_transport" not in self.url else ""), cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        except Exception: pass
        if cap.isOpened():
            ok,_=cap.read()
            if ok:
                print("[INFO] FFMPEG RTSP OK")
                return cap
            cap.release()
        print("[ERROR] RTSP open failed")
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
        except Exception: pass

class WebcamReader(threading.Thread):
    def __init__(self, cam_id=0, size=(640,480)):
        super().__init__(daemon=True)
        self.cam_id=cam_id; self.size=size
        self.cap=None; self.latest=None
        self.lock=threading.Lock(); self.stop_evt=threading.Event()

    def open(self):
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
                if self.cap is None: time.sleep(0.5); continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                try: self.cap.release()
                except Exception: pass
                self.cap=None; time.sleep(0.2); continue
            with self.lock:
                self.latest = frame

    def get_latest(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.stop_evt.set()
        try:
            if self.cap is not None: self.cap.release()
        except Exception: pass

# ====== Main ======
def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    det = PlateYOLO6(MODEL_PATH)

    # Reader
    if SOURCE.lower()=="rtsp":
        print(f"[INFO] Source: RTSP {RTSP_URL}")
        reader = RTSPReader(RTSP_URL, size=GST_SIZE)
    else:
        print(f"[INFO] Source: Webcam id={WEBCAM_ID} size={WEBCAM_SIZE}")
        reader = WebcamReader(WEBCAM_ID, WEBCAM_SIZE)
    reader.start()

    # SORT tracker
    tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESH)

    # Async OCR worker
    ocrw = AsyncOCR(max_fps=OCR_MAX_FPS)

    writer=None
    fps_tick=time.time(); frames=0; fps=0.0
    frame_interval = 1.0/TARGET_FPS if TARGET_FPS>0 else 0.0

    def cleanup(*_):
        reader.stop(); reader.join(timeout=1.5)
        ocrw.stop()
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
                time.sleep(0.003); continue

            H,W = frame.shape[:2]
            if writer is None and OUTPUT_MP4:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, 20.0, (W,H))
                if not writer.isOpened(): print("[WARN] writer failed"); writer=None

            # Detect
            dets = det.infer(frame)  # list of (x1,y1,x2,y2,score)

            # SORT expects ndarray [[x1,y1,x2,y2,score], ...]
            sort_in = np.array([[x1,y1,x2,y2,sc] for (x1,y1,x2,y2,sc) in dets], dtype=np.float32) if dets else np.empty((0,5), dtype=np.float32)
            tracks = tracker.update(sort_in)  # returns [[x1,y1,x2,y2,track_id], ...]

            # Pick up to N tracks to OCR (largest/most confident detections)
            # Create a map from bbox to score for sorting
            score_map = {(x1,y1,x2,y2):sc for (x1,y1,x2,y2,sc) in dets}
            track_list = []
            for x1,y1,x2,y2,tid in tracks:
                x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2); tid=int(tid)
                sc = score_map.get((x1,y1,x2,y2), 0.0)
                track_list.append((x1,y1,x2,y2,tid,sc))
            # sort by score*area
            track_list.sort(key=lambda t: ((t[5])*((t[2]-t[0])*(t[3]-t[1]))), reverse=True)

            # Submit limited crops to OCR worker (latest-only per track)
            for x1,y1,x2,y2,tid,sc in track_list[:OCR_TOPK_TRACKS]:
                px = int(0.03*(x2-x1)); py = int(0.20*(y2-y1))
                xa, xb = max(0, x1-px), min(W-1, x2+px)
                ya, yb = max(0, y1-py), min(H-1, y2+py)
                roi = frame[ya:yb, xa:xb]
                if roi.size>0:
                    ocrw.submit(tid, roi)

            # Draw tracks + stabilized text
            for x1,y1,x2,y2,tid,sc in track_list:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                text, votes = ocrw.get_stable_text(tid)
                if text and votes >= PLATE_ACCEPT_COUNT:
                    label = f"ID{tid} {text}"
                    color = (0,200,0)
                else:
                    label = f"ID{tid} reading..."
                    color = (0,255,255)
                cv2.putText(frame, label, (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # FPS
            frames += 1
            if (time.time()-fps_tick) >= 1.0:
                fps = frames / (time.time()-fps_tick)
                fps_tick = time.time(); frames=0
            cv2.putText(frame, f"FPS(det): {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if writer is not None: writer.write(frame)
            if SHOW_WINDOW:
                cv2.imshow("Plates + SORT + Async OCR", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if frame_interval>0:
                dt = time.time()-t0
                if dt < frame_interval: time.sleep(frame_interval-dt)

    finally:
        cleanup()

if __name__ == "__main__":
    main()
