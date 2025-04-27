"""face_crop.py – find_face_square(image, side_max, pad_px=80)

Returns (x0, y0, x1, y1) in ORIGINAL pixels, expanded outward by
`pad_px` on every side so the crop isn’t tight on the face.
If no face is detected, returns None.
"""
from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

_mp_fd = mp.solutions.face_detection


def find_face_square(
    pil_img: Image.Image,
    side_max: int,
    pad_px: int = 300,           # ← adjust this for more/less breathing room
) -> Tuple[int, int, int, int] | None:
    img_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    with _mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
        res = fd.process(img_bgr[:, :, ::-1])  # MediaPipe expects RGB
        if not res.detections:
            return None

        # largest face by box area
        det = max(
            res.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        )
        box = det.location_data.relative_bounding_box
        x0 = int(box.xmin * w)
        y0 = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        # base square side = max face dimension + padding*2
        side = max(bw, bh) + 2 * pad_px
        side = min(side, side_max, w, h)      # never exceed limits

        # center square on the face
        cx = x0 + bw // 2
        cy = y0 + bh // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)

        # if clamped, re-square by shifting the opposite edge
        x0, y0 = x1 - side, y1 - side

        return (x0, y0, x1, y1)
