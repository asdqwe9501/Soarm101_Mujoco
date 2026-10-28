"""색 가드 보정: 실제 손목 카메라 사진에서 현재 ROI·HSV 임계값이 공 색을 맞게 판별하는지 확인한다.

1) 촬영 (실제 손목 카메라):
   python tools/guard_calibrate.py capture --wrist-cam 1 --out data/guard_calib
   공을 집게 사이(잡기 직전 위치)에 두고 색 이름(red / black / blue / none)을 입력하면 한 장 저장한다.
   none은 공 없이 책상만 보이는 사진이다. 색마다 조명·위치를 바꿔 10장 이상 찍는다.
2) 확인 (저장한 사진, 파일 이름이 색으로 시작: red_003.png, blue_010.png, none_001.png):
   python tools/guard_calibrate.py check --images data/guard_calib
   사진마다 판별 결과와 색별 픽셀 비율을 출력하고, ROI와 색 마스크를 겹친 이미지를 <images>/overlay/에 저장한다.
   맞지 않으면 policies/color_guard.py의 ROI, MIN_FRAC, DOMINANCE, _color_masks 임계값을 조정한다.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policies.color_guard import ROI, _color_masks, _drop_border_blobs, classify_wrist, wrist_color_fractions  # noqa: E402

LABELS = {"red": "red", "black": "black", "blue": "sky blue", "none": None}
TINT = {"red": (255, 0, 0), "black": (255, 255, 0), "sky blue": (0, 160, 255)}


def overlay(img):
    out = img.astype(np.float32).copy()
    for c, m in _color_masks(img).items():
        m = _drop_border_blobs(m)
        out[m] = 0.4 * out[m] + 0.6 * np.array(TINT[c])
    out = out.astype(np.uint8)
    r, cidx = ROI
    out[r.start:r.stop, [cidx.start, cidx.stop - 1]] = (0, 255, 0)
    out[[r.start, r.stop - 1], cidx.start:cidx.stop] = (0, 255, 0)
    return out


def check(images):
    files = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    (Path(images) / "overlay").mkdir(exist_ok=True)
    conf = Counter()
    for p in files:
        label = p.stem.split("_")[0].lower()
        if label not in LABELS:
            print(f"건너뜀 (이름이 red/black/blue/none으로 시작하지 않음): {p.name}")
            continue
        img = np.asarray(Image.open(p).convert("RGB"))
        if img.shape[:2] != (480, 640):
            print(f"경고: {p.name} 크기 {img.shape[:2]} (ROI는 480×640 기준)")
        seen = classify_wrist(img)
        want = LABELS[label]
        conf[(str(want), str(seen))] += 1
        frac = {k: round(v, 3) for k, v in wrist_color_fractions(img).items()}
        print(f"{'OK ' if seen == want else 'XX '} {p.name}: 정답 {want}, 판별 {seen}, 비율 {frac}")
        Image.fromarray(overlay(img)).save(Path(images) / "overlay" / p.name)
    n = sum(conf.values())
    ok = sum(v for (w, s), v in conf.items() if w == s)
    wrong_color = sum(v for (w, s), v in conf.items() if s != "None" and w != s)
    print(f"\n정확도 {ok}/{n}. 다른 색으로 잘못 판별(가드가 맞는 공을 거부하게 됨): {wrong_color}")
    print("정답 → 판별:", dict(conf))
    print(f"겹친 이미지: {Path(images) / 'overlay'} (초록 사각형 = ROI)")


def capture(wrist_cam, out):
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=int(wrist_cam) if str(wrist_cam).isdigit() else wrist_cam,
                                          fps=30, width=640, height=480))
    cam.connect()
    Path(out).mkdir(parents=True, exist_ok=True)
    count = Counter(p.stem.split("_")[0] for p in Path(out).glob("*.png"))
    try:
        while True:
            label = input("색 (red / black / blue / none, 빈 줄: 종료) > ").strip().lower()
            if not label:
                break
            if label not in LABELS:
                print("red / black / blue / none 중 하나를 입력하세요")
                continue
            img = np.asarray(cam.read())
            count[label] += 1
            path = Path(out) / f"{label}_{count[label]:03d}.png"
            Image.fromarray(img).save(path)
            print(f"  저장 {path.name}: 지금 판별 {classify_wrist(img)}")
    finally:
        cam.disconnect()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--wrist-cam", required=True)
    c.add_argument("--out", default="data/guard_calib")
    k = sub.add_parser("check")
    k.add_argument("--images", default="data/guard_calib")
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.wrist_cam, args.out)
    else:
        check(args.images)


if __name__ == "__main__":
    main()
