#!/usr/bin/env python3
import csv
import html
import sys
from pathlib import Path


WIDTH = 1400
HEIGHT = 1100
PADDING = 60
MAX_RENDER_POINTS = 180000


def project(value, vmin, vmax, start, end):
    if vmax <= vmin:
        return (start + end) * 0.5
    t = (value - vmin) / (vmax - vmin)
    return start + t * (end - start)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: render_cluster_snapshot.py INPUT.csv OUTPUT.svg TITLE", file=sys.stderr)
        return 1

    input_csv = Path(sys.argv[1])
    output_svg = Path(sys.argv[2])
    title = sys.argv[3]

    points = []
    with input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((
                float(row["x"]),
                float(row["y"]),
                int(row["r"]),
                int(row["g"]),
                int(row["b"]),
            ))

    if not points:
        raise RuntimeError(f"no points in {input_csv}")

    if len(points) > MAX_RENDER_POINTS:
        step = max(1, len(points) // MAX_RENDER_POINTS)
        points = points[::step]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    pad_x = x_span * 0.05
    pad_y = y_span * 0.05
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    inner_w = WIDTH - 2 * PADDING
    inner_h = HEIGHT - 2 * PADDING

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
    parts.append('<rect width="100%" height="100%" fill="#0d0f12"/>')
    parts.append(f'<text x="{PADDING}" y="36" fill="#ffffff" font-size="24" font-family="Arial, sans-serif">{html.escape(title)}</text>')
    parts.append(f'<rect x="{PADDING}" y="{PADDING}" width="{inner_w}" height="{inner_h}" fill="#0d0f12" stroke="#33404d" stroke-width="1"/>')

    for x, y, r, g, b in points:
        px = project(x, x_min, x_max, PADDING, WIDTH - PADDING)
        py = project(y, y_min, y_max, HEIGHT - PADDING, PADDING)
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="0.8" fill="rgb({r},{g},{b})" />'
        )

    parts.append(f'<text x="{PADDING}" y="{HEIGHT - 18}" fill="#9fb0c2" font-size="16" font-family="Arial, sans-serif">x range: {x_min:.1f} to {x_max:.1f} m</text>')
    parts.append(f'<text x="{WIDTH - 420}" y="{HEIGHT - 18}" fill="#9fb0c2" font-size="16" font-family="Arial, sans-serif">y range: {y_min:.1f} to {y_max:.1f} m</text>')
    parts.append('</svg>')

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(parts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
