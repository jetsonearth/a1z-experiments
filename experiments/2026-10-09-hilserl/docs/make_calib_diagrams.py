"""Draw two calibration diagrams for the USB-port RL setup (labels in Simplified Chinese).

Outputs (written next to this script):
  calib-annotated.png  - side-view photo, upscaled 2x, with calibration annotations
  calib-heights.png    - side-view height schematic (not to scale)

Run with:
  /Users/ZhaobangJetWu/a1z-experiments/.venv-pipeline/bin/python -I make_calib_diagrams.py
"""

import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PHOTO = os.path.join(HERE, "side-view.jpg")
OUT_ANNOTATED = os.path.join(HERE, "calib-annotated.png")
OUT_HEIGHTS = os.path.join(HERE, "calib-heights.png")

# PingFang.ttc is not available on this machine (PIL cannot open it), so use the
# Simplified Chinese face of STHeiti (index 1 = Heiti SC) instead.
FONT_CANDIDATES = [
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 1),
    ("/System/Library/Fonts/STHeiti Light.ttc", 1),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
]


def load_font(size):
    for path, index in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    raise SystemExit("No usable Chinese font found")


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def dashed_line(draw, p0, p1, fill, width=3, dash=16, gap=10):
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    t = 0.0
    while t < length:
        a = t
        b = min(t + dash, length)
        draw.line(
            [(x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b)],
            fill=fill,
            width=width,
        )
        t += dash + gap


def dashed_rect(draw, box, fill, width=4, dash=16, gap=10):
    x0, y0, x1, y1 = box
    dashed_line(draw, (x0, y0), (x1, y0), fill, width, dash, gap)
    dashed_line(draw, (x1, y0), (x1, y1), fill, width, dash, gap)
    dashed_line(draw, (x1, y1), (x0, y1), fill, width, dash, gap)
    dashed_line(draw, (x0, y1), (x0, y0), fill, width, dash, gap)


def leader(draw, p0, p1, color):
    """Thin leader line with a dark halo so it reads over the photo."""
    draw.line([p0, p1], fill=(0, 0, 0, 150), width=7)
    draw.line([p0, p1], fill=color, width=3)
    r = 5
    draw.ellipse([p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r], fill=color)


def text_size(font, text):
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def label_box(draw, x, y, lines, accent, font, pad_x=18, pad_y=12, gap=10, accent_w=6):
    """Dark translucent label box with a coloured left strip and white text.

    Returns the box rectangle (x0, y0, x1, y1) so leaders can attach to it.
    """
    widths = []
    heights = []
    for line in lines:
        w, h = text_size(font, line)
        widths.append(w)
        heights.append(h)
    line_h = max(heights)
    body_w = max(widths)
    total_h = line_h * len(lines) + gap * (len(lines) - 1)
    x1 = x + accent_w + pad_x + body_w + pad_x
    y1 = y + pad_y + total_h + pad_y
    draw.rounded_rectangle([x, y, x1, y1], radius=10, fill=(14, 14, 14, 190))
    draw.rectangle([x, y + 8, x + accent_w, y1 - 8], fill=accent)
    ty = y + pad_y
    for line in lines:
        draw.text((x + accent_w + pad_x, ty), line, font=font, fill=(255, 255, 255, 255))
        ty += line_h + gap
    return (x, y, x1, y1)


# ---------------------------------------------------------------------------
# Diagram 1: annotated photo
# ---------------------------------------------------------------------------

SCALE = 2  # photo is upscaled 2x; the coordinates below are in ORIGINAL pixels


def S(x, y):
    """Original-photo coordinate -> upscaled coordinate."""
    return (x * SCALE, y * SCALE)


def make_annotated():
    photo = Image.open(SRC_PHOTO).convert("RGBA")
    base = photo.resize((640 * SCALE, 480 * SCALE), Image.LANCZOS)
    W, H = base.size

    # Key positions in original photo pixels (verified on a gridded crop).
    target = S(437, 247)        # target port (plug currently plugged in here)
    mouth = S(436, 250)         # plug tip just inside the mouth (key 2)
    blue_left = S(403, 238)     # key 1: between target and neighbour slot
    blue_right = S(470, 258)    # key 1: between target and hub's right end
    yellow_c = S(437, 223.5)    # centre of the start-height band above the port

    RED = (230, 40, 40, 255)
    GREEN = (40, 200, 80, 255)
    BLUE = (30, 144, 255, 255)
    YELLOW = (255, 210, 0)
    ORANGE = (255, 140, 0, 255)

    font = load_font(26)
    font_small = load_font(22)

    # --- Layer 1: zones (yellow start band, orange safety box) -----------
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rx, ry = 25 * SCALE, 11.5 * SCALE
    d.ellipse(
        [yellow_c[0] - rx, yellow_c[1] - ry, yellow_c[0] + rx, yellow_c[1] + ry],
        fill=(YELLOW[0], YELLOW[1], YELLOW[2], 95),
        outline=(YELLOW[0], YELLOW[1], YELLOW[2], 230),
        width=3,
    )
    half = 40 * SCALE
    safe_box = (target[0] - half, target[1] - half, target[0] + half, target[1] + half)
    dashed_rect(d, safe_box, fill=ORANGE, width=4)
    base = Image.alpha_composite(base, layer)

    # --- Layer 2: markers -------------------------------------------------
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    red_r = 40
    d.ellipse(
        [target[0] - red_r, target[1] - red_r, target[0] + red_r, target[1] + red_r],
        outline=RED,
        width=5,
    )
    gr = 12
    d.ellipse(
        [mouth[0] - gr, mouth[1] - gr, mouth[0] + gr, mouth[1] + gr],
        outline=GREEN,
        width=4,
    )
    for bx, by in (blue_left, blue_right):
        r = 9
        d.ellipse(
            [bx - r, by - r, bx + r, by + r],
            fill=BLUE,
            outline=(255, 255, 255, 255),
            width=3,
        )
    base = Image.alpha_composite(base, layer)

    # --- Layer 3 + 4: labels and leaders ---------------------------------
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Label 1 (red, top-left): target port.
    box1 = label_box(
        d, 40, 36,
        ["目标口（标定前先把插头拔出来）"],
        accent=RED, font=font,
    )
    leader(d, (box1[2], (box1[1] + box1[3]) // 2), (target[0], target[1] - red_r + 2), RED)

    # Label 4 (yellow, left): random start band.
    box4 = label_box(
        d, 40, 124,
        ["每轮随机起点：口上方 4–12 mm、", "水平 10 mm 以内"],
        accent=(YELLOW[0], YELLOW[1], YELLOW[2], 255), font=font,
    )
    leader(
        d,
        (box4[2], (box4[1] + box4[3]) // 2),
        (yellow_c[0] - rx, yellow_c[1]),
        (YELLOW[0], YELLOW[1], YELLOW[2], 255),
    )

    # Label 5 (orange, upper right): safety box. Vertical leader down to the box top.
    box5 = label_box(
        d, 880, 140,
        ["安全范围：", "水平 ±15 mm"],
        accent=ORANGE, font=font,
    )
    leader(d, (box5[0] + 50, box5[3]), (safe_box[2] - 20, safe_box[1]), ORANGE)

    # Label 2 (green, right): key 2 at the mouth.
    box2 = label_box(
        d, 960, 372,
        ["按 2：插头刚插进口、", "不往下推（按 2 次）"],
        accent=GREEN, font=font,
    )
    leader(d, (box2[0], box2[1] + 60), (mouth[0] + gr + 2, mouth[1] - 6), GREEN)

    # Label 3 (blue, bottom right): key 1 on the flat plastic beside the port.
    box3 = label_box(
        d, 800, 820,
        ["按 1：插头平放在口旁边的塑料上", "（左右各 1 次），别落进别的口"],
        accent=BLUE, font=font,
    )
    leader(d, (box3[0] + 70, box3[1]), (blue_left[0] + 2, blue_left[1] + 10), BLUE)
    leader(d, (box3[0] + 220, box3[1]), (blue_right[0] + 2, blue_right[1] + 10), BLUE)

    # Footnote (bottom left).
    box6 = label_box(
        d, 40, 900,
        ["毫米与像素的换算是估计值，仅作示意"],
        accent=(170, 170, 170, 255), font=font_small,
    )

    base = Image.alpha_composite(base, layer)
    base.convert("RGB").save(OUT_ANNOTATED, "PNG")
    print("wrote", OUT_ANNOTATED)


# ---------------------------------------------------------------------------
# Diagram 2: height schematic (not to scale)
# ---------------------------------------------------------------------------

def make_heights():
    W, H = 1100, 750
    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    font_title = load_font(30)
    font = load_font(21)
    font_small = load_font(20)

    BLUE = (25, 113, 194, 255)
    ORANGE = (240, 140, 0, 255)
    GREEN = (47, 158, 68, 255)
    YELLOW = (255, 212, 59)
    INK = (33, 37, 41, 255)

    # Vertical levels (y grows downward). Surface is at y = 420.
    Y_PLUS20 = 260
    Y_PLUS12 = 324
    Y_PLUS4 = 388
    Y_SURF = 420
    Y_MOUTH = 440      # just inside the slot (key 2)
    Y_DEEP = 512       # mouth - 9 mm (safety box bottom)
    Y_HUB_BOTTOM = 690

    HUB_X0, HUB_X1 = 200, 560
    SLOT_X0, SLOT_X1 = 370, 390
    LINE_X0, LINE_X1 = 140, 600
    LABEL_X = 620

    # --- Zones and lines (layer, so the translucent band blends) ---------
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rectangle([LINE_X0, Y_PLUS12, LINE_X1, Y_PLUS4], fill=(YELLOW[0], YELLOW[1], YELLOW[2], 120))
    img = Image.alpha_composite(img, layer)

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Hub block (front elevation) and its port slot.
    d.rounded_rectangle([HUB_X0, Y_SURF, HUB_X1, Y_HUB_BOTTOM], radius=12, fill=(34, 34, 34, 255))
    d.rectangle([SLOT_X0, Y_SURF, SLOT_X1, 540], fill=(236, 239, 242, 255))
    # Subtle highlight on the top edge of the hub.
    d.line([(HUB_X0 + 12, Y_SURF + 3), (HUB_X1 - 12, Y_SURF + 3)], fill=(70, 70, 70, 255), width=2)

    # Level lines.
    dashed_line(d, (LINE_X0, Y_PLUS20), (LINE_X1, Y_PLUS20), ORANGE, width=3, dash=14, gap=9)
    dashed_line(d, (LINE_X0, Y_MOUTH), (LINE_X1, Y_MOUTH), GREEN, width=3, dash=14, gap=9)
    dashed_line(d, (LINE_X0, Y_DEEP), (LINE_X1, Y_DEEP), ORANGE, width=3, dash=14, gap=9)
    d.line([(LINE_X0, Y_SURF), (LINE_X1, Y_SURF)], fill=BLUE, width=4)

    # Plug: grey body above the surface, tip just inside the slot at the mouth level.
    d.rectangle([373, 300, 387, Y_MOUTH], fill=(134, 142, 150, 255), outline=(52, 58, 64, 255), width=2)
    d.rectangle([376, Y_MOUTH - 14, 384, Y_MOUTH], fill=(206, 212, 218, 255))
    img = Image.alpha_composite(img, layer)

    # Start band border (drawn on top of the translucent fill).
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.line([(LINE_X0, Y_PLUS12), (LINE_X1, Y_PLUS12)], fill=(YELLOW[0], YELLOW[1], YELLOW[2], 255), width=2)
    d.line([(LINE_X0, Y_PLUS4), (LINE_X1, Y_PLUS4)], fill=(YELLOW[0], YELLOW[1], YELLOW[2], 255), width=2)
    img = Image.alpha_composite(img, layer)

    # --- Title and labels -------------------------------------------------
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((W // 2, 40), "标定和 RL 范围（侧视示意，不按比例）", font=font_title,
           fill=INK, anchor="mm")

    d.text((LABEL_X, Y_PLUS20), "表面 + 20 mm：安全盒顶部", font=font, fill=ORANGE, anchor="lm")
    d.text((LABEL_X, (Y_PLUS12 + Y_PLUS4) // 2), "表面 + 4~12 mm：每轮起点（随机）",
           font=font, fill=INK, anchor="lm")
    d.text((LABEL_X, Y_SURF - 8), "表面（按 1 量这里的高度）", font=font, fill=BLUE, anchor="lb")
    d.text((LABEL_X, Y_MOUTH + 12), "口部（按 2：刚进口 = 论文的 TARGET_POSE）",
           font=font, fill=GREEN, anchor="lm")
    d.text((LABEL_X, Y_DEEP), "口部 − 9 mm：安全盒底部（够插到底）",
           font=font, fill=ORANGE, anchor="lm")

    # Dotted arrow from far above, down to just above the start band.
    arrow_x = 380
    y = 128
    while y < 282:
        d.ellipse([arrow_x - 2.5, y - 2.5, arrow_x + 2.5, y + 2.5], fill=INK)
        y += 11
    d.polygon([(arrow_x - 9, 280), (arrow_x + 9, 280), (arrow_x, 296)], fill=INK)
    d.text((arrow_x + 22, 112), "几十厘米外 → 脚本/IK 移到起点附近（不归 RL 管）",
           font=font_small, fill=INK, anchor="lm")

    # Bracket on the left spanning from the start band down to the safety-box bottom.
    bx = 116
    d.line([(bx, Y_PLUS12), (bx, Y_DEEP)], fill=INK, width=3)
    d.line([(bx, Y_PLUS12), (bx + 10, Y_PLUS12)], fill=INK, width=3)
    d.line([(bx, Y_DEEP), (bx + 10, Y_DEEP)], fill=INK, width=3)
    mid = (Y_PLUS12 + Y_DEEP) // 2
    d.text((12, mid - 16), "RL 负责", font=font_small, fill=INK, anchor="lm")
    d.text((12, mid + 14), "的这一段", font=font_small, fill=INK, anchor="lm")

    img = Image.alpha_composite(img, layer)
    img.convert("RGB").save(OUT_HEIGHTS, "PNG")
    print("wrote", OUT_HEIGHTS)


if __name__ == "__main__":
    make_annotated()
    make_heights()
