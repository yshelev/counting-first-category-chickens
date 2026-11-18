import os
import time
import json
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from dotenv import load_dotenv
import requests

load_dotenv()

print("LKRLEWKL:WKERL:KWLERKWLERKW:LEKRWL:")
RTSP_URL = os.getenv("RTSP_URL")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best.pt")
CONF = float(os.getenv("CONF", 0.4))
LINE_FRAC = float(os.getenv("LINE_FRAC", 0.5))
PLUMP_THIN = float(os.getenv("PLUMP_THIN", 0.30))
DICE_MIN = float(os.getenv("DICE_MIN", 0.60))
DEFECT_RED = float(os.getenv("DEFECT_RED", 0.015))
DEFECT_YELLOW = float(os.getenv("DEFECT_YELLOW", 0.020))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 1))
API_URL = os.getenv("SERVER_URL", "http://server:8000") + "/data/bird"

model = YOLO(MODEL_PATH, task="segment") if os.path.exists(MODEL_PATH) else YOLO("yolov8n-seg.pt")
print(model.task)
print("HELLO WORKER")
def crop_to_mask(m):
    ys, xs = np.where(m > 0)
    if len(ys) == 0: return None
    return m[ys.min():ys.max()+1, xs.min():xs.max()+1]

def canonical(mask_full):
    c = crop_to_mask(mask_full)
    if c is None or c.shape[0] < 30: return None
    ys, xs = np.where(c > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    rot = np.degrees(np.arctan2(major[1], major[0])) - 90.0
    pad = int(max(c.shape))
    cp = cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    M = cv2.getRotationMatrix2D((mean[0] + pad, mean[1] + pad), rot, 1.0)
    r = cv2.warpAffine(cp, M, (cp.shape[1], cp.shape[0]), flags=cv2.INTER_NEAREST)
    return crop_to_mask(r)

def color_ratio(frame, mb):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    red = (((H <= 8) | (H >= 160)) & (S >= 110) & (V >= 40))
    yellow = ((H >= 18) & (H <= 38) & (S >= 80) & (V >= 80))
    tot = max(1, mb.sum())
    return {
        "red": float((red & mb).sum()) / tot,
        "yellow": float((yellow & mb).sum()) / tot
    }

def verdict(plump, red, yellow):
    if red >= DEFECT_RED or yellow >= DEFECT_YELLOW:
        return "брак"
    if plump < PLUMP_THIN:
        return "брак"
    return "1"

def make_thumb(frame, mb):
    ys, xs = np.where(mb)
    crop = frame[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape[:2]
    scale = 96 / max(h, w)
    crop = cv2.resize(crop, (max(1, int(w*scale)), max(1, int(h*scale))))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode() if ok else None

def send_bird(bird_data):
    try:
        requests.post(API_URL, json=bird_data, timeout=1)
    except Exception as e:
        print(f"Send error: {e}")

def process_video():
    print("QWEQWEQWEQWEQWEQW")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"Failed to open RTSP: {RTSP_URL}")
        return

    W = int(cap.get(3))
    H = int(cap.get(4))
    x_line = int(W * LINE_FRAC)
    prev, seen = {}, set()
    order = 0
    frame_count = 0

    print(f"Started processing RTSP stream. Line at x={x_line}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream disconnected, reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        conf=CONF, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue

        ids = r.boxes.id.int().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().numpy()
        print("rboxes", r.boxes)
        print("rmasks", r.masks)
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None

        for k, (box, tid) in enumerate(zip(xyxy, ids)):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            print(masks)
            if (tid in prev and tid not in seen and
                prev[tid] < x_line <= cx and
                masks is not None and k < len(masks)):

                mb = cv2.resize(masks[k], (W, H)) > 0.5
                can = canonical(mb.astype(np.uint8))
                if can is None:
                    prev[tid] = cx
                    continue

                order += 1
                norm = (cv2.resize(can, (96, 192),
                                   interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
                plump = float((can > 0).sum(axis=1).max()) / can.shape[0]
                cr = color_ratio(frame, mb)

                bird = {
                    "n": order,
                    "id": int(tid),
                    "plump": plump,
                    "red": cr["red"],
                    "yellow": cr["yellow"],
                    "grade": verdict(plump, cr["red"], cr["yellow"]),
                    "thumb": make_thumb(frame, mb),
                }
                seen.add(tid)
                send_bird(bird)
                print(f"BIRD {tid} SENT! Order: {order}", flush=True)

            prev[tid] = cx

            
if __name__ == "__main__":
    process_video()