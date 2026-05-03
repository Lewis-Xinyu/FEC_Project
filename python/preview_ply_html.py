#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go


PLY_TO_NUMPY = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a PLY point cloud as a local interactive HTML preview."
    )
    parser.add_argument("ply_path", type=Path, help="Input .ply file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path. Defaults next to the input file.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=180_000,
        help="Maximum points to draw in the browser.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.3,
        help="Rendered marker size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Sampling seed.",
    )
    return parser.parse_args()


def read_header_lines(path: Path) -> tuple[list[str], int]:
    header_lines: list[str] = []
    offset = 0
    with path.open("rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            offset += len(line)
            text = line.decode("ascii", errors="strict").strip()
            header_lines.append(text)
            if text == "end_header":
                break
    return header_lines, offset


def extract_vertex_schema(header_lines: Iterable[str]) -> tuple[str, int, list[tuple[str, str]]]:
    fmt = ""
    in_vertex = False
    vertex_count = 0
    props: list[tuple[str, str]] = []
    for line in header_lines:
        if line.startswith("format "):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError("Invalid PLY format line")
            fmt = parts[1]
            continue
        if line.startswith("element "):
            parts = line.split()
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) == 3:
                prop_type, prop_name = parts[1], parts[2]
                props.append((prop_name, prop_type))
                continue
            if len(parts) == 5 and parts[1] == "list":
                raise ValueError("PLY vertex list properties are not supported")
    if not fmt or vertex_count <= 0 or not props:
        raise ValueError("Invalid PLY: missing vertex schema")
    return fmt, vertex_count, props


def read_binary_vertices(
    path: Path, offset: int, vertex_count: int, props: list[tuple[str, str]]
) -> np.ndarray:
    dtype_fields = []
    for name, prop_type in props:
        if prop_type not in PLY_TO_NUMPY:
            raise ValueError(f"Unsupported PLY property type: {prop_type}")
        dtype_fields.append((name, "<" + PLY_TO_NUMPY[prop_type]))
    dtype = np.dtype(dtype_fields)
    with path.open("rb") as fh:
        fh.seek(offset)
        data = np.fromfile(fh, dtype=dtype, count=vertex_count)
    return np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32, copy=False)


def read_ascii_vertices(
    path: Path, offset: int, vertex_count: int, props: list[tuple[str, str]]
) -> np.ndarray:
    prop_names = [name for name, _ in props]
    try:
        x_idx, y_idx, z_idx = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
    except ValueError as exc:
        raise ValueError("PLY vertex properties must include x/y/z") from exc
    rows: list[list[float]] = []
    with path.open("rb") as fh:
        fh.seek(offset)
        for _ in range(vertex_count):
            line = fh.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading ASCII vertices")
            parts = line.decode("ascii", errors="strict").strip().split()
            rows.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
    return np.asarray(rows, dtype=np.float32)


def load_xyz(path: Path) -> np.ndarray:
    header_lines, offset = read_header_lines(path)
    fmt, vertex_count, props = extract_vertex_schema(header_lines)
    prop_names = [name for name, _ in props]
    if not {"x", "y", "z"}.issubset(prop_names):
        raise ValueError("PLY vertex properties must include x/y/z")
    if fmt == "binary_little_endian":
        return read_binary_vertices(path, offset, vertex_count, props)
    if fmt == "ascii":
        return read_ascii_vertices(path, offset, vertex_count, props)
    raise ValueError(f"Unsupported PLY format: {fmt}")


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def axis_ranges(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    pad = spans * 0.03
    return (
        (float(mins[0] - pad[0]), float(maxs[0] + pad[0])),
        (float(mins[1] - pad[1]), float(maxs[1] + pad[1])),
        (float(mins[2] - pad[2]), float(maxs[2] + pad[2])),
    )


def write_html(
    out_path: Path, src_path: Path, all_points: np.ndarray, render_points: np.ndarray, point_size: float
) -> None:
    x_range, y_range, z_range = axis_ranges(all_points)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=render_points[:, 0],
                y=render_points[:, 1],
                z=render_points[:, 2],
                mode="markers",
                marker={
                    "size": point_size,
                    "color": render_points[:, 2],
                    "colorscale": "Turbo",
                    "opacity": 0.85,
                },
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        title={
            "text": (
                f"{html.escape(src_path.name)}"
                f"<br><sup>showing {len(render_points):,} / {len(all_points):,} points</sup>"
            )
        },
        scene={
            "xaxis": {"title": "X", "range": x_range, "backgroundcolor": "#f7f7f7"},
            "yaxis": {"title": "Y", "range": y_range, "backgroundcolor": "#f7f7f7"},
            "zaxis": {"title": "Z", "range": z_range, "backgroundcolor": "#f7f7f7"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        paper_bgcolor="#ffffff",
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    ply_path = args.ply_path.expanduser().resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    out_path = args.out.expanduser().resolve() if args.out else ply_path.with_suffix(".preview.html")
    points = load_xyz(ply_path)
    render_points = sample_points(points, args.max_points, args.seed)
    write_html(out_path, ply_path, points, render_points, args.point_size)
    print(f"Input: {ply_path}")
    print(f"Points: {len(points):,}")
    print(f"Rendered: {len(render_points):,}")
    print(f"HTML: {out_path}")


if __name__ == "__main__":
    main()
