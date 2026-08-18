"""Render advisor-ready 3-D expert-map scenes from the frozen TRAIN payload.

This is a visualization-only producer.  It reads map geometry from the frozen
map manifest and trajectory positions from the frozen TRAIN NPZ.  It does not
start a simulator, open the sealed evaluation payload, train a model, or alter
scientific data.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "artifacts" / "s8r2_10k"
OUT_ROOT = ROOT / "artifacts" / "s8r5_rendered_maps"
FIG_ROOT = OUT_ROOT / "figures"
META_ROOT = OUT_ROOT / "metadata"

MAP_MANIFEST = DATA_ROOT / "map_manifest.json"
TASK_MANIFEST = DATA_ROOT / "task_manifest.csv"
TRAIN_PAYLOAD = DATA_ROOT / "train.npz"

WIDTH_PX = 2560
HEIGHT_PX = 1440
FIG_SIZE = (16.0, 9.0)
DPI = WIDTH_PX / FIG_SIZE[0]

SCENE_SELECTIONS = [
    {
        "figure_name": "render_double_gate_hero",
        "family": "DOUBLE_GATE",
        "map_id": "DOUBLE_GATE_075",
        "task_id": "DOUBLE_GATE_075_TASK_08",
        "selection_reason": "固定 TRAIN 双门地图；轨迹展示横向与高度变化并穿过两组门结构，适合作为主视觉，但不声称最难或最优。",
        "azim": -62,
        "elev": 24,
        "accent": "#36D7E8",
    },
    {
        "figure_name": "render_sbend_hero",
        "family": "SBEND_RANDOM",
        "map_id": "SBEND_RANDOM_023",
        "task_id": "SBEND_RANDOM_023_TASK_05",
        "selection_reason": "沿用冻结 TRAIN family 代表样本；三段交替障碍形成清晰 S 型横向绕障展示。",
        "azim": -66,
        "elev": 25,
        "accent": "#FFB45E",
    },
    {
        "figure_name": "render_chicane_or_detour_hero",
        "family": "CHICANE_RANDOM",
        "map_id": "CHICANE_RANDOM_090",
        "task_id": "CHICANE_RANDOM_090_TASK_02",
        "selection_reason": "采用冻结 TRAIN 的 CHICANE_RANDOM 代表样本；连续左右偏置让多次路线调整在空间视角下清晰可见。",
        "azim": -58,
        "elev": 23,
        "accent": "#A88CFF",
    },
    {
        "figure_name": "render_clutter_or_gate_hero",
        "family": "MIXED_CLUTTER",
        "map_id": "MIXED_CLUTTER_078",
        "task_id": "MIXED_CLUTTER_078_TASK_04",
        "selection_reason": "采用冻结 TRAIN 的 MIXED_CLUTTER 代表样本；障碍数量更多、交替排列更复杂，适合研究问题背景页。",
        "azim": -63,
        "elev": 24,
        "accent": "#F06C8D",
    },
]

FAMILY_COLORS = {
    "DOUBLE_GATE": "#35C9D8",
    "SBEND_RANDOM": "#F4A55B",
    "CHICANE_RANDOM": "#A88CFF",
    "MIXED_CLUTTER": "#F06C8D",
}
ROUTE_COLORS = ["#36D7E8", "#FFB45E", "#9C8CFF", "#F06C8D", "#63E6A5", "#FFE27A", "#7EA9FF", "#F49AC2", "#B6F36B", "#D6B4FF"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tasks() -> dict[str, dict[str, str]]:
    with TASK_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["task_id"]: row for row in rows}


def require_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    required = (MAP_MANIFEST, TASK_MANIFEST, TRAIN_PAYLOAD)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen S8-R2 inputs: {missing}")
    maps = {row["map_id"]: row for row in json.loads(MAP_MANIFEST.read_text(encoding="utf-8"))}
    tasks = read_tasks()
    for item in SCENE_SELECTIONS:
        task = tasks[item["task_id"]]
        if task["split"] != "train":
            raise AssertionError(f"selected task escaped TRAIN: {item['task_id']}")
        if item["map_id"] not in maps or task["map_id"] != item["map_id"]:
            raise AssertionError(f"map/task mismatch: {item['task_id']}")
    return maps, tasks


def build_episode_index() -> dict[str, tuple[int, int]]:
    with np.load(TRAIN_PAYLOAD, allow_pickle=False) as data:
        task_ids = data["task_ids"].astype(str)
        offsets = data["episode_offsets"].astype(np.int64)
        lengths = data["episode_lengths"].astype(np.int64)
    if len(task_ids) != 8000 or len(set(task_ids)) != 8000:
        raise AssertionError("unexpected TRAIN episode index")
    return {task_id: (int(offset), int(length)) for task_id, offset, length in zip(task_ids, offsets, lengths)}


def extract_positions(task_ids: list[str]) -> dict[str, np.ndarray]:
    index = build_episode_index()
    missing = sorted(set(task_ids) - set(index))
    if missing:
        raise AssertionError(f"selected TRAIN tasks missing: {missing}")
    result: dict[str, np.ndarray] = {}
    with np.load(TRAIN_PAYLOAD, allow_pickle=False) as data:
        positions = data["positions"]
        for task_id in task_ids:
            offset, length = index[task_id]
            result[task_id] = np.asarray(positions[offset : offset + length], dtype=np.float32).copy()
    return result


def box_faces(center: np.ndarray, size: np.ndarray) -> list[list[tuple[float, float, float]]]:
    lo = center - size / 2.0
    hi = center + size / 2.0
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def shade(base: str, factor: float) -> str:
    rgb = np.asarray(mcolors.to_rgb(base))
    return mcolors.to_hex(np.clip(rgb * factor, 0.0, 1.0))


def draw_ground(ax: Any, bounds: dict[str, list[float]]) -> None:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0 = bounds["z"][0]
    xx, yy = np.meshgrid(np.linspace(x0, x1, 25), np.linspace(y0, y1, 19))
    zz = np.full_like(xx, z0)
    ax.plot_surface(xx, yy, zz, color="#0A1D2D", alpha=0.72, linewidth=0, shade=False)
    for x in np.linspace(x0, x1, 17):
        ax.plot([x, x], [y0, y1], [z0 + 0.004, z0 + 0.004], color="#1A4A63", alpha=0.33, linewidth=0.5)
    for y in np.linspace(y0, y1, 13):
        ax.plot([x0, x1], [y, y], [z0 + 0.004, z0 + 0.004], color="#1A4A63", alpha=0.33, linewidth=0.5)


def draw_obstacles(ax: Any, map_record: dict[str, Any], accent: str) -> None:
    for index, obstacle in enumerate(map_record["obstacles"]):
        center = np.asarray(obstacle["center"], dtype=float)
        size = np.asarray(obstacle["size"], dtype=float)
        base = shade(accent, 0.78 if index % 2 else 0.92)
        faces = box_faces(center, size)
        face_colors = [shade(base, factor) for factor in (0.62, 1.18, 0.82, 0.92, 0.72, 1.03)]
        collection = Poly3DCollection(
            faces,
            facecolors=face_colors,
            edgecolors=shade(accent, 1.15),
            linewidths=0.8,
            alpha=0.94,
        )
        ax.add_collection3d(collection)
        # A restrained top-edge highlight gives the blocks a rendered-object feel
        # without changing the recorded geometry.
        hi = center + np.asarray([0.0, 0.0, size[2] / 2.0])
        ax.plot(
            [hi[0] - size[0] / 2, hi[0] + size[0] / 2],
            [hi[1] - size[1] / 2, hi[1] - size[1] / 2],
            [hi[2], hi[2]],
            color="#FFFFFF",
            alpha=0.32,
            linewidth=1.2,
        )


def draw_route(ax: Any, positions: np.ndarray, color: str, *, gradient: bool = True, linewidth: float = 3.0) -> None:
    points = positions[:, :3]
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#02080F", alpha=0.85, linewidth=linewidth + 4.5, solid_capstyle="round")
    if gradient:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        collection = Line3DCollection(segments, cmap="turbo", linewidths=linewidth, alpha=0.98)
        collection.set_array(np.linspace(0.08, 0.95, len(segments)))
        ax.add_collection3d(collection)
    else:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.98, linewidth=linewidth, solid_capstyle="round")
    ax.scatter(*points[0], s=70, color="#FFFFFF", edgecolors=color, linewidths=2.0, depthshade=False, zorder=12)
    ax.scatter(*points[-1], s=92, color="#63E6A5", edgecolors="#D7FFE7", linewidths=1.5, depthshade=False, zorder=12)


def draw_drone_marker(ax: Any, position: np.ndarray, color: str) -> None:
    x, y, z = position[:3]
    span = 0.20
    ax.scatter([x], [y], [z], s=40, color="#F8FBFF", edgecolors=color, linewidths=1.5, depthshade=False, zorder=14)
    ax.plot([x - span, x + span], [y, y], [z, z], color=color, linewidth=2.3, alpha=0.95)
    ax.plot([x, x], [y - span, y + span], [z, z], color=color, linewidth=2.3, alpha=0.95)
    for dx, dy in ((-span, 0), (span, 0), (0, -span), (0, span)):
        ax.scatter([x + dx], [y + dy], [z], s=16, color=color, edgecolors="#FFFFFF", linewidths=0.6, depthshade=False, zorder=15)


def style_axes(ax: Any, bounds: dict[str, list[float]], azim: float, elev: float) -> None:
    ax.set_facecolor("#071724")
    ax.figure.patch.set_facecolor("#06111D")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_zlim(bounds["z"])
    ax.set_box_aspect((8, 6, 3.0))
    ax.set_xlabel("X / forward", color="#B8D4E3", labelpad=8, fontsize=10)
    ax.set_ylabel("Y / lateral", color="#B8D4E3", labelpad=8, fontsize=10)
    ax.set_zlabel("Z / height", color="#B8D4E3", labelpad=6, fontsize=10)
    ax.tick_params(colors="#89AFC2", labelsize=8, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.025, 0.09, 0.14, 0.82))
        axis.pane.set_edgecolor((0.10, 0.25, 0.33, 0.7))
    ax.grid(True, color="#1B485D", alpha=0.42, linewidth=0.6)


def render_scene(item: dict[str, Any], map_record: dict[str, Any], positions: np.ndarray, output: Path, *, title_suffix: str = "") -> None:
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor="#06111D")
    ax = fig.add_subplot(111, projection="3d")
    bounds = map_record["world_bounds"]
    accent = item["accent"]
    style_axes(ax, bounds, item["azim"], item["elev"])
    draw_ground(ax, bounds)
    draw_obstacles(ax, map_record, accent)
    draw_route(ax, positions, accent, gradient=True, linewidth=3.0)
    draw_drone_marker(ax, positions[int(len(positions) * 0.72)], accent)
    start, goal = positions[0], positions[-1]
    ax.text(start[0], start[1], start[2] + 0.18, "START", color="#FFFFFF", fontsize=9, weight="bold")
    ax.text(goal[0], goal[1], goal[2] + 0.18, "GOAL", color="#8FF1B7", fontsize=9, weight="bold")
    fig.text(0.055, 0.928, item["family"].replace("_", " "), color=accent, fontsize=14, weight="bold", family="DejaVu Sans")
    fig.text(0.055, 0.883, title_suffix or "EXPERT TRAJECTORY / TRAIN SPLIT", color="#F4F8FB", fontsize=25, weight="bold")
    fig.text(0.055, 0.847, f"{item['map_id']}  ·  {item['task_id']}  ·  FROZEN TRAIN EXPERT", color="#A9C6D4", fontsize=10)
    fig.text(0.945, 0.928, "S8-R5B  /  RENDERED MAPS", color="#6F94A6", fontsize=9, ha="right", family="DejaVu Sans")
    fig.text(0.945, 0.075, "OBSTACLE GEOMETRY = FROZEN MAP MANIFEST   |   TRAJECTORY = FROZEN TRAIN PAYLOAD", color="#6F94A6", fontsize=8.5, ha="right")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.82)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches=None)
    plt.close(fig)


def render_multi_route(item: dict[str, Any], map_record: dict[str, Any], routes: list[tuple[str, np.ndarray]], output: Path) -> None:
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor="#06111D")
    ax = fig.add_subplot(111, projection="3d")
    style_axes(ax, map_record["world_bounds"], -62, 24)
    draw_ground(ax, map_record["world_bounds"])
    draw_obstacles(ax, map_record, item["accent"])
    for index, (_, points) in enumerate(routes):
        draw_route(ax, points, ROUTE_COLORS[index % len(ROUTE_COLORS)], gradient=False, linewidth=1.65)
    handles = [plt.Line2D([0], [0], color=ROUTE_COLORS[i], lw=2, label=f"task {i:02d}") for i in range(len(routes))]
    ax.legend(handles=handles, ncol=5, loc="upper left", bbox_to_anchor=(0.02, 0.97), labelcolor="#D8EBF3", facecolor="#102B3C", framealpha=0.88, fontsize=8)
    fig.text(0.055, 0.928, "DOUBLE GATE  /  MULTI-ROUTE", color="#35C9D8", fontsize=14, weight="bold")
    fig.text(0.055, 0.883, "10 REAL TRAIN EXPERT ROUTES / ONE MAP", color="#F4F8FB", fontsize=25, weight="bold")
    fig.text(0.055, 0.847, "DOUBLE_GATE_075  ·  task_00–task_09  ·  FIXED OBSTACLE GEOMETRY", color="#A9C6D4", fontsize=10)
    fig.text(0.945, 0.075, "EVERY ROUTE IS READ DIRECTLY FROM THE FROZEN TRAIN PAYLOAD", color="#6F94A6", fontsize=8.5, ha="right")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.82)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def make_overview(paths: list[tuple[str, Path]], output: Path) -> None:
    canvas = Image.new("RGB", (WIDTH_PX * 3 // 2, HEIGHT_PX * 3 // 2), "#06111D")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    tile_w = canvas.width // 2
    tile_h = canvas.height // 2
    for index, (label, path) in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_w - 18, tile_h - 18), Image.Resampling.LANCZOS)
            x = (index % 2) * tile_w + (tile_w - image.width) // 2
            y = (index // 2) * tile_h + (tile_h - image.height) // 2
            canvas.paste(image, (x, y))
            draw.rounded_rectangle((x + 16, y + 16, x + 16 + 250, y + 58), radius=10, fill="#06111D", outline="#6D9CAE", width=1)
            draw.text((x + 30, y + 24), label, fill="#F4F8FB", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False)


def main() -> None:
    maps, tasks = require_inputs()
    selected_ids = [item["task_id"] for item in SCENE_SELECTIONS]
    multi_map_tasks = [f"DOUBLE_GATE_075_TASK_{slot:02d}" for slot in range(10)]
    positions = extract_positions(selected_ids + multi_map_tasks)

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for item in SCENE_SELECTIONS:
        render_scene(item, maps[item["map_id"]], positions[item["task_id"]], FIG_ROOT / f"{item['figure_name']}.png", title_suffix={
            "DOUBLE_GATE": "DOUBLE GATE TRAVERSAL / HERO SCENE",
            "SBEND_RANDOM": "S-BEND AVOIDANCE / HERO SCENE",
            "CHICANE_RANDOM": "CHICANE DETOUR / HERO SCENE",
            "MIXED_CLUTTER": "MIXED CLUTTER / HERO SCENE",
        }[item["family"]])

    multi_item = {
        "figure_name": "render_same_map_multi_trajectories",
        "family": "DOUBLE_GATE",
        "map_id": "DOUBLE_GATE_075",
        "task_id": "; ".join(multi_map_tasks),
        "split": "train",
        "accent": "#35C9D8",
        "selection_reason": "固定同一张 TRAIN 双门地图，叠加该地图上的 10 条真实 TRAIN 专家轨迹，展示 start-goal 多样性。",
        "source_dataset": "S8-R2 10K procedural GCOPTER expert dataset",
        "rendering_tool": "Matplotlib 3D Poly3DCollection + Line3DCollection",
        "output_resolution": [WIDTH_PX, HEIGHT_PX],
        "based_on_real_trajectory": True,
    }
    render_multi_route(multi_item, maps["DOUBLE_GATE_075"], [(task_id, positions[task_id]) for task_id in multi_map_tasks], FIG_ROOT / "render_same_map_multi_trajectories.png")

    drone_item = dict(SCENE_SELECTIONS[3])
    drone_item["figure_name"] = "render_single_scene_with_drone_marker"
    render_scene(drone_item, maps[drone_item["map_id"]], positions[drone_item["task_id"]], FIG_ROOT / "render_single_scene_with_drone_marker.png", title_suffix="DRONE MARKER / SINGLE SCENE")

    hero_paths = [
        ("DOUBLE GATE", FIG_ROOT / "render_double_gate_hero.png"),
        ("S-BEND", FIG_ROOT / "render_sbend_hero.png"),
        ("CHICANE", FIG_ROOT / "render_chicane_or_detour_hero.png"),
        ("MIXED CLUTTER", FIG_ROOT / "render_clutter_or_gate_hero.png"),
    ]
    overview_path = FIG_ROOT / "rendered_expert_map_overview.png"
    make_overview(hero_paths, overview_path)

    figures: list[dict[str, Any]] = []
    for item in SCENE_SELECTIONS:
        figures.append({
            "figure_name": item["figure_name"],
            "family": item["family"],
            "map_id": item["map_id"],
            "task_id": item["task_id"],
            "split": "train",
            "source_dataset": "S8-R2 10K procedural GCOPTER expert dataset",
            "selection_reason": item["selection_reason"],
            "rendering_tool": "Matplotlib 3D Poly3DCollection + Line3DCollection",
            "output_resolution": [WIDTH_PX, HEIGHT_PX],
            "based_on_real_trajectory": True,
        })
    figures.extend([
        multi_item,
        {
            "figure_name": "render_single_scene_with_drone_marker",
            "family": drone_item["family"],
            "map_id": drone_item["map_id"],
            "task_id": drone_item["task_id"],
            "split": "train",
            "source_dataset": "S8-R2 10K procedural GCOPTER expert dataset",
            "selection_reason": "在 MIXED_CLUTTER TRAIN 主场景中加入位于真实轨迹上的简洁无人机标识，仅做视觉提示。",
            "rendering_tool": "Matplotlib 3D Poly3DCollection + Line3DCollection",
            "output_resolution": [WIDTH_PX, HEIGHT_PX],
            "based_on_real_trajectory": True,
        },
        {
            "figure_name": "rendered_expert_map_overview",
            "family": "DOUBLE_GATE; SBEND_RANDOM; CHICANE_RANDOM; MIXED_CLUTTER",
            "map_id": "MULTI_SCENE_OVERVIEW",
            "task_id": "; ".join(item["task_id"] for item in SCENE_SELECTIONS),
            "split": "train",
            "source_dataset": "S8-R2 10K procedural GCOPTER expert dataset",
            "selection_reason": "将四张已生成的 TRAIN 场景主图按 2×2 拼接，供 PPT 快速总览。",
            "rendering_tool": "PIL composition of four TRAIN-derived rendered figures",
            "output_resolution": [WIDTH_PX * 3 // 2, HEIGHT_PX * 3 // 2],
            "based_on_real_trajectory": True,
        },
    ])

    source_hashes = {
        "map_manifest.json": sha256_file(MAP_MANIFEST),
        "task_manifest.csv": sha256_file(TASK_MANIFEST),
        "train.npz": sha256_file(TRAIN_PAYLOAD),
    }
    for figure in figures:
        path = FIG_ROOT / f"{figure['figure_name']}.png"
        figure["file_sha256"] = sha256_file(path)

    metadata = {
        "task": "S8-R5B-RENDERED-EXPERT-TRAJECTORY-MAPS-FOR-ADVISOR-V1",
        "test_payload_accessed": False,
        "model_training_run_count": 0,
        "scientific_contract_changed": False,
        "source_payloads_opened": ["artifacts/s8r2_10k/train.npz"],
        "source_hashes": source_hashes,
        "figures": figures,
        "selection_policy": "deterministic fixed TRAIN map/task IDs; no random selection; no TEST payload access",
        "rendering_notes": [
            "Obstacle centers and sizes are read directly from map_manifest.json.",
            "Trajectory polyline vertices are read directly from train.npz positions.",
            "Ground plane, materials, lighting-like face shading, glow underlay, labels, and drone marker are non-data visual layers.",
            "No trajectory smoothing, hand-drawn route, obstacle relocation, simulator screenshot, or model output is used.",
        ],
    }
    META_ROOT.mkdir(parents=True, exist_ok=True)
    (META_ROOT / "render_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(figures), "output": str(OUT_ROOT), "test_payload_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
