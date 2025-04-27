#!/usr/bin/env python3
"""
LoRA-dataset prep GUI
————————
• Full-image preview (Tkinter Canvas)
• Square crop: draggable inside, resizable by corner handles
• Stops at image edge, always square
• Saved PNG is resampled to `crop_size` (settings.yaml)
• Keys: ← → browse | Enter save | s skip
"""

# ── stdlib / deps ───────────────────────────────────────────────────────────
from pathlib import Path
import sys
import yaml                # pip install pyyaml
from PIL import Image, ImageTk  # pip install pillow
import tkinter as tk
from tkinter import messagebox


from face_crop import find_face_square
from captioner import caption

# ── settings ───────────────────────────────────────────────────────────────
CFG_FILE = "settings.yaml"
DEFAULTS = dict(
    input_folder="./input",
    output_folder="./output",
    crop_size=512,
    global_tags="photo of jonathanzxyz",
)


def load_cfg():
    cfg = DEFAULTS.copy()
    if Path(CFG_FILE).is_file():
        cfg.update(yaml.safe_load(Path(CFG_FILE).read_text()) or {})
    cfg["input_folder"] = str(Path(cfg["input_folder"]).expanduser())
    cfg["output_folder"] = str(Path(cfg["output_folder"]).expanduser())
    return cfg


CFG = load_cfg()


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])


# ── CropBox helper ─────────────────────────────────────────────────────────
class CropBox:
    HANDLE_R = 6  # px radius of handles (display coords)

    def __init__(self, canvas, img_w, img_h, scale):
        self.cv = canvas
        self.iw, self.ih, self.s = img_w, img_h, scale
        side = min(CFG["crop_size"], img_w, img_h)
        pad_x = (img_w - side) // 2
        pad_y = (img_h - side) // 2
        self.box = [pad_x, pad_y, pad_x + side, pad_y + side]  # xyxy in orig px
        self.drag_mode = None  # "move" or handle idx 0-3
        self.start_xy = (0, 0)
        self.start_box = self.box.copy()
        self.redraw()

        # one-time canvas-wide events
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<B1-Motion>", self.on_drag)
        self.cv.bind("<ButtonRelease-1>", lambda _e: setattr(self, "drag_mode", None))

    # ── draw / redraw ────────────────────────────────────────────────────
    def redraw(self):
        self.cv.delete("crop")
        x0, y0, x1, y1 = [c * self.s for c in self.box]
        # main rectangle
        self.cv.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="lime",
            width=2,
            tags="crop",
        )
        # handles
        self.handles = []
        for hx, hy in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            hid = self.cv.create_rectangle(
                hx - self.HANDLE_R,
                hy - self.HANDLE_R,
                hx + self.HANDLE_R,
                hy + self.HANDLE_R,
                fill="lime",
                outline="black",
                tags="crop",
            )
            self.handles.append(hid)

    # ── event helpers ───────────────────────────────────────────────────
    def on_press(self, ev):
        self.start_xy = (ev.x, ev.y)
        self.start_box = self.box.copy()

        # check if on a handle first
        for i, hid in enumerate(self.handles):
            x0, y0, x1, y1 = self.cv.coords(hid)
            if x0 <= ev.x <= x1 and y0 <= ev.y <= y1:
                self.drag_mode = i  # handle index
                return
        # else inside rect? -> move
        rx0, ry0, rx1, ry1 = [c * self.s for c in self.box]
        if rx0 <= ev.x <= rx1 and ry0 <= ev.y <= ry1:
            self.drag_mode = "move"

    def on_drag(self, ev):
        if self.drag_mode is None:
            return
        dx = (ev.x - self.start_xy[0]) / self.s
        dy = (ev.y - self.start_xy[1]) / self.s
        b = self.start_box.copy()

        if self.drag_mode == "move":
            b = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
            b = self._clamp_move(b)
        else:
            b = self._resize(b, dx, dy, self.drag_mode)
        self.box = b
        self.redraw()

    # ── math -------------------------------------------------------------
    def _clamp_move(self, b):
        side = b[2] - b[0]
        x0 = max(0, min(b[0], self.iw - side))
        y0 = max(0, min(b[1], self.ih - side))
        return [x0, y0, x0 + side, y0 + side]

    def _resize(self, b, dx, dy, idx):
        delta = max(dx, dy, key=abs)
        if idx == 0:  # TL
            b[0] += delta
            b[1] += delta
        elif idx == 1:  # TR
            b[2] += delta
            b[1] -= delta
        elif idx == 2:  # BR
            b[2] += delta
            b[3] += delta
        elif idx == 3:  # BL
            b[0] -= delta
            b[3] += delta

        # enforce square + bounds
        side = b[2] - b[0]
        side = max(10, side)
        if b[0] < 0:
            b[0], b[2] = 0, side
        if b[1] < 0:
            b[1], b[3] = 0, side
        if b[2] > self.iw:
            b[2], b[0] = self.iw, self.iw - side
        if b[3] > self.ih:
            b[3], b[1] = self.ih, self.ih - side
        b[2] = b[0] + side
        b[3] = b[1] + side
        return b

    # public
    def get_box(self):
        return tuple(map(int, self.box))


# ── Main application ───────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LoRA crop-tag helper")
        self.geometry("1200x800")

        self.img_paths = list_images(CFG["input_folder"])
        if not self.img_paths:
            messagebox.showinfo("No images", "Put JPG/PNG files in input folder.")
            self.destroy()
            return
        self.idx = 0

        self.saved = set()             #  <-- NEW  (indices of images you've saved)

        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        bottom = tk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(bottom, text="Caption:").pack(side=tk.LEFT, padx=4)
        self.caption = tk.StringVar()
        tk.Entry(bottom, textvariable=self.caption, width=70).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        for lbl, cmd in (
            ("◀ Prev", self.prev_img),
            ("Next ▶", self.next_img),
            ("✅ Save", self.save_img),
            ("Skip", self.skip_img),
        ):
            tk.Button(bottom, text=lbl, command=cmd).pack(side=tk.RIGHT, padx=2, pady=4)

        # keys
        self.bind("<Left>", lambda _e: self.prev_img())
        self.bind("<Right>", lambda _e: self.next_img())
        self.bind("<Return>", lambda _e: self.save_img())
        self.bind("s", lambda _e: self.skip_img())

        Path(CFG["output_folder"]).mkdir(parents=True, exist_ok=True)
        self.photo = None
        self.cropper = None
        self.load_img()

    # ── image cycle ────────────────────────────────────────────────────
    def load_img(self):
        path = self.img_paths[self.idx]
        self.orig = Image.open(path).convert("RGB")

        # ── auto-face crop ──────────────────────────────────────────────
        side_limit = min(self.orig.width, self.orig.height)   # ← NEW: allow big squares
        face_box   = find_face_square(self.orig, side_limit)  #    (was CFG["crop_size"])
        self.initial_face_box = face_box                      # None if no face found

        # ── fit to window ───────────────────────────────────────────────
        self.update_idletasks()
        max_w = self.canvas.winfo_width() - 20 or 1000
        max_h = self.canvas.winfo_height() - 20 or 700
        scale = min(max_w / self.orig.width, max_h / self.orig.height, 1)
        disp  = self.orig.resize(
            (int(self.orig.width * scale), int(self.orig.height * scale))
        )
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(10, 10, anchor=tk.NW, image=self.photo, tags="img")

        # ── cropper ─────────────────────────────────────────────────────
        self.cropper = CropBox(self.canvas, self.orig.width, self.orig.height, scale)
        if self.initial_face_box:
            self.cropper.box = list(self.initial_face_box)
            self.cropper.redraw()

        # ── caption ─────────────────────────────────────────────────────
        
        ph    = caption(self.orig)
        gtags = CFG["global_tags"]
        self.caption.set(f"{gtags}, {ph}" if gtags else ph)

        self.title(f"[{self.idx+1}/{len(self.img_paths)}] – {path.name}")


    # ── actions ────────────────────────────────────────────────────────
    def save_img(self):
        box = self.cropper.get_box()
        crop = (
            self.orig.crop(box)
            .resize((CFG["crop_size"], CFG["crop_size"]), Image.LANCZOS)
        )
        stem = self.img_paths[self.idx].stem
        out_dir = Path(CFG["output_folder"])
        crop.save(out_dir / f"{stem}.png")

        text = self.caption.get().strip()
        if CFG["global_tags"] and CFG["global_tags"] not in text:
            text = f"{text}, {CFG['global_tags']}".strip(", ")
        (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")

        self.saved.add(self.idx)   #  <-- NEW

        self.next_img()

    def skip_img(self):
        self.next_img()

    def next_img(self):
        if self.idx < len(self.img_paths) - 1:
            self.idx += 1
            self.load_img()
        else:  # already at last image
            if len(self.saved) == len(self.img_paths):
                messagebox.showinfo("Done", "All images saved. Bye!")
                self.destroy()
            else:
                messagebox.showinfo(
                    "Reached end",
                    "You’re on the last image.\nSave it (Enter) or go back (←) – "
                    "the program will exit only after **all** images are saved.",
                )

    def prev_img(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_img()


# ── run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        App().mainloop()
    except KeyboardInterrupt:
        pass
