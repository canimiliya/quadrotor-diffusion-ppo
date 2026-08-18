"""Create advisor-ready S8-R5 figures from the frozen S8-R2 expert dataset.

This script is visualization-only.  It never opens the sealed TEST payload,
never trains a model, and only extracts trajectory arrays from train.npz.
Dataset-wide counts and distributions come from frozen metadata files.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


# Editable SVG text is mandatory for the slide-ready vector exports.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "s8r5-expert-visualization-v1"
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.9,
    "legend.frameon": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


ROOT = Path(__file__).resolve().parents[1]
# Exact frozen 18-ray contract from envs/observation.py, repeated here so the
# visualization-only script does not import PyBullet or start simulation code.
LOCAL_RAY_DIRECTIONS = np.asarray([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
], dtype=float)
LOCAL_RAY_DIRECTIONS /= np.linalg.norm(LOCAL_RAY_DIRECTIONS, axis=1, keepdims=True)

DATA_ROOT = ROOT / "artifacts" / "s8r2_10k"
OUT_ROOT = ROOT / "artifacts" / "s8r5_visualization"
FIG_ROOT = OUT_ROOT / "figures"
META_ROOT = OUT_ROOT / "metadata"

SUMMARY_PATH = DATA_ROOT / "dataset_summary.json"
MAP_MANIFEST_PATH = DATA_ROOT / "map_manifest.json"
TASK_MANIFEST_PATH = DATA_ROOT / "task_manifest.csv"
DIVERSITY_PATH = DATA_ROOT / "diversity_statistics.csv"
TRAIN_PATH = DATA_ROOT / "train.npz"

FAMILIES = (
    "SINGLE_BLOCK", "OFFSET_BLOCK", "DOUBLE_BLOCK", "ALTERNATING_BLOCKS",
    "SBEND_RANDOM", "CHICANE_RANDOM", "NARROW_GATE", "DOUBLE_GATE",
    "LATERAL_DETOUR", "MIXED_CLUTTER",
)
HERO_TASK_ID = "DOUBLE_GATE_075_TASK_08"
HORIZON = 16
RAY_RANGE_M = 3.0
SLIDE_SIZE = (13.333, 7.5)

COLORS = {
    "blue": "#0F4D92",
    "blue_soft": "#78A6D0",
    "teal": "#42949E",
    "green": "#2E8B57",
    "green_soft": "#AADCA9",
    "red": "#B64342",
    "red_soft": "#E9A6A1",
    "gold": "#E3A21A",
    "violet": "#7C6CCF",
    "ink": "#272727",
    "mid": "#767676",
    "light": "#D8D8D8",
    "pale": "#F3F6F9",
}
XYZ_COLORS = (COLORS["blue"], COLORS["teal"], COLORS["red"])
TASK_COLORS = tuple(plt.get_cmap("tab10")(i) for i in range(10))
BLOCK_COLORS = ("#0F4D92", "#42949E", "#8BCF8B", "#7C6CCF", "#E3A21A", "#E9A6A1")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_vector(value: str) -> np.ndarray:
    result = np.asarray(json.loads(value), dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"invalid 3-vector: {value}")
    return result


def pretty_family(name: str) -> str:
    return name.replace("_", " ").title()


def robust_distance(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, float]:
    values = np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=float)
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = np.where(q75 - q25 > 1.0e-12, q75 - q25, 1.0)
    scores = np.abs((values - median) / scale).sum(axis=1)
    return {str(row["map_id"]): float(score) for row, score in zip(rows, scores)}


def select_family_representatives(
    task_rows: list[dict[str, str]], diversity_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Choose a median-geometry TRAIN map, then its most legible real path."""
    task_by_id = {row["task_id"]: row for row in task_rows}
    train_map_ids = {row["map_id"] for row in task_rows if row["split"] == "train"}
    map_rows = [
        row for row in diversity_rows
        if not row.get("task_id") and row["map_id"] in train_map_ids
    ]
    trajectory_rows = [
        row for row in diversity_rows
        if row.get("task_id") and task_by_id[row["task_id"]]["split"] == "train"
    ]
    by_family_maps: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_map_traj: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in map_rows:
        by_family_maps[row["family"]].append(row)
    for row in trajectory_rows:
        by_map_traj[row["map_id"]].append(row)

    fields = ("obstacle_count", "obstacle_width_mean", "obstacle_depth_mean", "obstacle_height_mean")
    selections: list[dict[str, Any]] = []
    for family in FAMILIES:
        candidates = by_family_maps[family]
        scores = robust_distance(candidates, fields)
        map_id = min((row["map_id"] for row in candidates), key=lambda x: (scores[x], x))
        traj = max(
            by_map_traj[map_id],
            key=lambda row: (float(row["path_straight_ratio"]), row["task_id"]),
        )
        selections.append({
            "family": family,
            "map_id": map_id,
            "task_id": traj["task_id"],
            "geometry_representativeness_score": scores[map_id],
            "path_straight_ratio": float(traj["path_straight_ratio"]),
            "selection_rule": "TRAIN map nearest family median geometry by IQR-scaled L1; within-map maximum path/straight ratio",
        })
    if {row["family"] for row in selections} != set(FAMILIES):
        raise AssertionError("family representative selection is incomplete")
    return selections


def load_inputs() -> dict[str, Any]:
    required = (SUMMARY_PATH, MAP_MANIFEST_PATH, TASK_MANIFEST_PATH, DIVERSITY_PATH, TRAIN_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen S8-R2 inputs: {missing}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    maps = json.loads(MAP_MANIFEST_PATH.read_text(encoding="utf-8"))
    tasks = read_csv(TASK_MANIFEST_PATH)
    diversity = read_csv(DIVERSITY_PATH)
    if summary["total_maps"] != 1000 or summary["total_trajectories"] != 10000:
        raise AssertionError("unexpected frozen dataset scale")
    if not summary.get("test_split_sealed"):
        raise AssertionError("TEST split is not marked sealed")
    if len(maps) != 1000 or len(tasks) != 10000:
        raise AssertionError("manifest cardinality mismatch")

    task_by_id = {row["task_id"]: row for row in tasks}
    map_by_id = {row["map_id"]: row for row in maps}
    hero = task_by_id[HERO_TASK_ID]
    if hero["split"] != "train":
        raise AssertionError("hero task must be TRAIN")
    hero_map_tasks = sorted(
        [row for row in tasks if row["map_id"] == hero["map_id"]],
        key=lambda row: int(row["task_slot"]),
    )
    if len(hero_map_tasks) != 10 or {row["split"] for row in hero_map_tasks} != {"train"}:
        raise AssertionError("same-map panel must contain exactly ten TRAIN tasks")

    representatives = select_family_representatives(tasks, diversity)
    for row in representatives:
        if task_by_id[row["task_id"]]["split"] != "train":
            raise AssertionError("family representative escaped TRAIN")
    return {
        "summary": summary,
        "maps": maps,
        "tasks": tasks,
        "diversity": diversity,
        "task_by_id": task_by_id,
        "map_by_id": map_by_id,
        "hero": hero,
        "hero_map_tasks": hero_map_tasks,
        "representatives": representatives,
    }


def build_episode_index(train_path: Path) -> dict[str, dict[str, Any]]:
    with np.load(train_path, allow_pickle=False) as data:
        task_ids = data["task_ids"].astype(str)
        map_ids = data["map_ids"].astype(str)
        offsets = data["episode_offsets"].astype(np.int64)
        lengths = data["episode_lengths"].astype(np.int64)
        task_indices = data["task_indices"].astype(np.int64)
    if len(task_ids) != 8000:
        raise AssertionError("TRAIN episode count mismatch")
    return {
        task_id: {
            "offset": int(offset), "length": int(length), "map_id": map_id,
            "task_slot": int(slot), "episode_index": int(index),
        }
        for index, (task_id, map_id, offset, length, slot) in enumerate(
            zip(task_ids, map_ids, offsets, lengths, task_indices)
        )
    }


def extract_train_payload(inputs: dict[str, Any]) -> dict[str, Any]:
    index = build_episode_index(TRAIN_PATH)
    wanted = {row["task_id"] for row in inputs["representatives"]}
    wanted.update(row["task_id"] for row in inputs["hero_map_tasks"])
    wanted.add(HERO_TASK_ID)
    missing = sorted(wanted - set(index))
    if missing:
        raise AssertionError(f"selected TRAIN tasks missing from train.npz: {missing}")

    selected_positions: dict[str, np.ndarray] = {}
    with np.load(TRAIN_PATH, allow_pickle=False) as data:
        positions = data["positions"]
        for task_id in wanted:
            episode = index[task_id]
            start, stop = episode["offset"], episode["offset"] + episode["length"]
            selected_positions[task_id] = np.asarray(positions[start:stop], dtype=np.float32).copy()
        del positions
        gc.collect()

        hero_episode = index[HERO_TASK_ID]
        hs, he = hero_episode["offset"], hero_episode["offset"] + hero_episode["length"]
        hero_velocities = np.asarray(data["velocities"][hs:he], dtype=np.float32).copy()
        hero_times = np.asarray(data["times"][hs:he], dtype=np.float32).copy()

        # Choose the real local observation with the most obstacle-visible rays.
        # Ties prefer the smaller minimum fraction and then the earlier frame.
        observations = data["observations"]
        sample_candidates: list[tuple[int, float, int, str, np.ndarray]] = []
        for task in inputs["hero_map_tasks"]:
            task_id = task["task_id"]
            episode = index[task_id]
            length = episode["length"]
            usable = np.asarray(
                observations[episode["offset"]: episode["offset"] + length - HORIZON + 1],
                dtype=np.float32,
            )
            ray_values = usable[:, 16:34]
            visible = (ray_values < 0.999999).sum(axis=1)
            min_ray = ray_values.min(axis=1)
            order = np.lexsort((np.arange(len(usable)), min_ray, -visible))
            local_index = int(order[0])
            sample_candidates.append((
                int(visible[local_index]), float(min_ray[local_index]), local_index,
                task_id, usable[local_index].copy(),
            ))
        sample_candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        visible_count, minimum_ray, sample_step, sample_task_id, sample_observation = sample_candidates[0]
        del observations
        gc.collect()

        sample_episode = index[sample_task_id]
        sample_global = sample_episode["offset"] + sample_step
        actions = data["actions"]
        hero_actions = np.asarray(actions[hs:he], dtype=np.float32).copy()
        action_window = np.asarray(actions[sample_global:sample_global + HORIZON], dtype=np.float32).copy()
        action_norms = np.linalg.norm(np.asarray(actions, dtype=np.float32), axis=1)
        histogram_counts, histogram_edges = np.histogram(action_norms, bins=np.linspace(0.0, 1.0001, 51))
        action_quantiles = np.quantile(action_norms, (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0))
        action_violation_count = int(np.count_nonzero(action_norms > 1.0 + 1.0e-6))
        del actions, action_norms
        gc.collect()

    if action_window.shape != (HORIZON, 3):
        raise AssertionError("sample H=16 action window is incomplete")
    return {
        "episode_index": index,
        "positions": selected_positions,
        "hero_velocities": hero_velocities,
        "hero_times": hero_times,
        "hero_actions": hero_actions,
        "sample_task_id": sample_task_id,
        "sample_step": sample_step,
        "sample_observation": sample_observation,
        "sample_action_window": action_window,
        "visible_ray_count": visible_count,
        "minimum_ray_fraction": minimum_ray,
        "action_histogram_counts": histogram_counts,
        "action_histogram_edges": histogram_edges,
        "action_quantiles": action_quantiles,
        "action_violation_count": action_violation_count,
    }


def save_figure(fig: plt.Figure, name: str) -> dict[str, Any]:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    svg_path = FIG_ROOT / f"{name}.svg"
    png_path = FIG_ROOT / f"{name}.png"
    fig.savefig(svg_path, facecolor="white", metadata={"Date": None})
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    with Image.open(png_path) as image:
        width, height = image.size
    return {
        "name": name,
        "png": str(png_path.relative_to(ROOT)).replace("\\", "/"),
        "svg": str(svg_path.relative_to(ROOT)).replace("\\", "/"),
        "png_sha256": sha256_file(png_path),
        "svg_sha256": sha256_file(svg_path),
        "pixel_size": [width, height],
    }


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.04, y=0.975, ha="left", fontsize=20, fontweight="bold", color=COLORS["ink"])
    fig.text(0.04, 0.925, subtitle, ha="left", va="top", fontsize=10.5, color=COLORS["mid"])


def add_topdown_obstacles(ax: plt.Axes, map_spec: dict[str, Any], *, alpha: float = 0.72) -> None:
    for obstacle in map_spec["obstacles"]:
        center = np.asarray(obstacle["center"], dtype=float)
        size = np.asarray(obstacle["size"], dtype=float)
        rect = patches.Rectangle(
            (center[0] - size[0] / 2.0, center[1] - size[1] / 2.0),
            size[0], size[1], facecolor=COLORS["red_soft"], edgecolor=COLORS["red"],
            linewidth=1.1, alpha=alpha, zorder=1,
        )
        ax.add_patch(rect)


def style_topdown(ax: plt.Axes, world_bounds: dict[str, list[float]], *, compact: bool = False) -> None:
    ax.set_xlim(*world_bounds["x"])
    ax.set_ylim(*world_bounds["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#FBFCFD")
    ax.grid(color="#D8DEE6", linewidth=0.55, alpha=0.65)
    ax.set_xticks((-1, 1, 3, 5, 7))
    ax.set_yticks((-3, 0, 3))
    if compact:
        ax.tick_params(labelsize=7, length=2)


def cuboid_faces(center: np.ndarray, size: np.ndarray) -> list[list[np.ndarray]]:
    low, high = center - size / 2.0, center + size / 2.0
    x0, y0, z0 = low
    x1, y1, z1 = high
    vertices = np.asarray([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    return [[vertices[i] for i in face] for face in (
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    )]


def add_cuboid_obstacles(ax: plt.Axes, map_spec: dict[str, Any], *, alpha: float = 0.38) -> None:
    for obstacle in map_spec["obstacles"]:
        center = np.asarray(obstacle["center"], dtype=float)
        size = np.asarray(obstacle["size"], dtype=float)
        collection = Poly3DCollection(
            cuboid_faces(center, size), facecolors=COLORS["red_soft"],
            edgecolors=COLORS["red"], linewidths=0.75, alpha=alpha,
        )
        ax.add_collection3d(collection)


def style_3d(ax: plt.Axes, world_bounds: dict[str, list[float]], *, local_center: np.ndarray | None = None) -> None:
    if local_center is None:
        ax.set_xlim(*world_bounds["x"])
        ax.set_ylim(*world_bounds["y"])
        ax.set_zlim(*world_bounds["z"])
        ax.set_box_aspect((8, 6, 2.8))
    else:
        ax.set_xlim(max(world_bounds["x"][0], local_center[0] - 2.8), min(world_bounds["x"][1], local_center[0] + 2.8))
        ax.set_ylim(max(world_bounds["y"][0], local_center[1] - 2.8), min(world_bounds["y"][1], local_center[1] + 2.8))
        ax.set_zlim(*world_bounds["z"])
        ax.set_box_aspect((5.6, 5.6, 2.8))
    ax.set_xlabel("x [m]", labelpad=6)
    ax.set_ylabel("y [m]", labelpad=6)
    ax.set_zlabel("z [m]", labelpad=4)
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.grid(True, alpha=0.22)


def quaternion_to_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def figure_family_overview(inputs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 5, figsize=SLIDE_SIZE)
    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.12, top=0.84, wspace=0.16, hspace=0.27)
    add_title(
        fig,
        "Ten procedural environment families — one real TRAIN expert per family",
        "Representative map geometry is selected deterministically; obstacles are true AABB footprints and paths are frozen rollout positions.",
    )
    for ax, selection in zip(axes.flat, inputs["representatives"]):
        map_spec = inputs["map_by_id"][selection["map_id"]]
        task = inputs["task_by_id"][selection["task_id"]]
        positions = payload["positions"][selection["task_id"]]
        add_topdown_obstacles(ax, map_spec)
        ax.plot(positions[:, 0], positions[:, 1], color=COLORS["blue"], linewidth=1.7, zorder=3)
        start, goal = parse_vector(task["start"]), parse_vector(task["goal"])
        ax.scatter(start[0], start[1], s=32, marker="o", color=COLORS["green"], edgecolor="white", linewidth=0.6, zorder=4)
        ax.scatter(goal[0], goal[1], s=48, marker="*", color=COLORS["gold"], edgecolor=COLORS["ink"], linewidth=0.4, zorder=4)
        style_topdown(ax, map_spec["world_bounds"], compact=True)
        ax.set_title(pretty_family(selection["family"]), fontsize=10.5, fontweight="bold", pad=4)
        ax.text(0.02, 0.03, selection["map_id"], transform=ax.transAxes, fontsize=6.8, color=COLORS["mid"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2})
    axes[1, 0].set_xlabel("x [m]")
    axes[1, 0].set_ylabel("y [m]")
    legend = [
        patches.Patch(facecolor=COLORS["red_soft"], edgecolor=COLORS["red"], label="Obstacle footprint"),
        Line2D([0], [0], color=COLORS["blue"], lw=2, label="Expert rollout"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["gold"], markeredgecolor=COLORS["ink"], label="Goal"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.035), fontsize=9)
    fig.text(0.985, 0.035, "10 families • 100 maps/family • 10 trajectories/map", ha="right", fontsize=9, color=COLORS["mid"])
    return save_figure(fig, "expert_family_overview")


def figure_ten_tasks(inputs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    map_spec = inputs["map_by_id"][inputs["hero"]["map_id"]]
    fig = plt.figure(figsize=SLIDE_SIZE)
    ax = fig.add_axes([0.07, 0.18, 0.86, 0.65])
    add_title(
        fig,
        "One map, ten distinct start–goal tasks",
        f"All paths below are accepted TRAIN rollouts on the unchanged geometry of {map_spec['map_id']} ({pretty_family(map_spec['family'])}).",
    )
    add_topdown_obstacles(ax, map_spec, alpha=0.82)
    line_handles = []
    for task in inputs["hero_map_tasks"]:
        slot = int(task["task_slot"])
        color = TASK_COLORS[slot]
        positions = payload["positions"][task["task_id"]]
        start, goal = parse_vector(task["start"]), parse_vector(task["goal"])
        ax.plot(positions[:, 0], positions[:, 1], color=color, linewidth=2.0, alpha=0.92, zorder=3)
        ax.scatter(start[0], start[1], s=34, marker="o", facecolor="white", edgecolor=color, linewidth=1.6, zorder=4)
        ax.scatter(goal[0], goal[1], s=58, marker="*", facecolor=color, edgecolor=COLORS["ink"], linewidth=0.45, zorder=4)
        ax.text(start[0] - 0.06, start[1] - 0.16, f"S{slot}", color=color, fontsize=7.5, ha="right", fontweight="bold")
        ax.text(goal[0] + 0.07, goal[1] + 0.08, f"G{slot}", color=color, fontsize=7.5, ha="left", fontweight="bold")
        line_handles.append(Line2D([0], [0], color=color, lw=2.2, label=f"Task {slot}"))
    style_topdown(ax, map_spec["world_bounds"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    marker_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["ink"], label="Exact start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["gold"], markeredgecolor=COLORS["ink"], label="Exact goal"),
    ]
    fig.legend(handles=line_handles + marker_handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.035), fontsize=8.3)
    return save_figure(fig, "ten_tasks_same_map")


def figure_single_trajectory(inputs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    task = inputs["hero"]
    map_spec = inputs["map_by_id"][task["map_id"]]
    positions = payload["positions"][HERO_TASK_ID]
    start, goal = parse_vector(task["start"]), parse_vector(task["goal"])
    fig = plt.figure(figsize=SLIDE_SIZE)
    ax = fig.add_axes([0.05, 0.08, 0.79, 0.80], projection="3d")
    add_title(
        fig,
        "A real three-dimensional GCOPTER expert rollout",
        "The blue curve is the recorded closed-loop TRAIN trajectory; the dashed segment is only the computed start–goal reference for visual comparison.",
    )
    add_cuboid_obstacles(ax, map_spec, alpha=0.46)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=COLORS["blue"], lw=3.0, label="Expert rollout", zorder=5)
    ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], color=COLORS["mid"],
            lw=1.3, ls="--", alpha=0.78, label="Straight start–goal reference")
    ax.scatter(*start, s=70, color=COLORS["green"], edgecolor="white", linewidth=0.7, label="Start", depthshade=False)
    ax.scatter(*goal, s=105, marker="*", color=COLORS["gold"], edgecolor=COLORS["ink"], linewidth=0.55, label="Goal", depthshade=False)
    style_3d(ax, map_spec["world_bounds"])
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.96), fontsize=8.5)

    stats_row = next(row for row in inputs["diversity"] if row.get("task_id") == HERO_TASK_ID)
    fig.text(0.79, 0.68, "TRACEABLE SAMPLE", fontsize=9, fontweight="bold", color=COLORS["blue"])
    fig.text(0.79, 0.62, f"Split\nTRAIN\n\nMap\n{task['map_id']}\n\nTask\n{task['task_id']}", fontsize=9.3, color=COLORS["ink"], va="top")
    fig.text(0.79, 0.27,
             f"Steps  {len(positions):,}\nDuration  {payload['hero_times'][-1]:.2f} s\nPath / straight  {float(stats_row['path_straight_ratio']):.3f}\nCollisions  0",
             fontsize=9.3, color=COLORS["ink"], va="top",
             bbox={"boxstyle": "round,pad=0.55", "facecolor": COLORS["pale"], "edgecolor": COLORS["light"]})
    return save_figure(fig, "single_expert_trajectory_3d")


def figure_timeseries(inputs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    positions = payload["positions"][HERO_TASK_ID]
    velocities = payload["hero_velocities"]
    actions = payload["hero_actions"]
    times = payload["hero_times"]
    fig, axes = plt.subplots(3, 1, figsize=SLIDE_SIZE, sharex=True, gridspec_kw={"height_ratios": (1.15, 1.0, 1.0)})
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.82, hspace=0.18)
    add_title(
        fig,
        "One expert rollout is a synchronized state–action time series",
        f"Same TRAIN trajectory as the 3D view: {HERO_TASK_ID} • {len(times):,} control steps • 48 Hz.",
    )
    labels = ("x", "y", "z")
    for axis, values, ylabel in zip(axes, (positions, velocities, actions),
                                    ("Position [m]", "Velocity [m/s]", "Normalized action")):
        for index, (label, color) in enumerate(zip(labels, XYZ_COLORS)):
            axis.plot(times, values[:, index], color=color, lw=1.8, label=label)
        axis.axhline(0.0, color=COLORS["light"], lw=0.8, zorder=0)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#DFE4EA", linewidth=0.6, alpha=0.7)
        axis.legend(loc="upper right", ncol=3, fontsize=8.5)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_ylim(-1.05, 1.05)
    axes[-1].text(0.01, 0.08, "u = world-frame target velocity / 0.801 m/s", transform=axes[-1].transAxes,
                  fontsize=8.5, color=COLORS["mid"])
    return save_figure(fig, "expert_trajectory_timeseries")


def sample_ray_geometry(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observation = payload["sample_observation"]
    position = observation[0:3].astype(float)
    goal_relative = observation[3:6].astype(float)
    quaternion = observation[9:13].astype(float)
    ray_values = observation[16:34].astype(float)
    rotation = quaternion_to_matrix_xyzw(quaternion)
    world_directions = np.asarray(LOCAL_RAY_DIRECTIONS, dtype=float) @ rotation.T
    endpoints = position[None, :] + RAY_RANGE_M * ray_values[:, None] * world_directions
    return position, goal_relative, ray_values, endpoints


def figure_sample_format(inputs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    observation = payload["sample_observation"]
    action_window = payload["sample_action_window"]
    task = inputs["task_by_id"][payload["sample_task_id"]]
    map_spec = inputs["map_by_id"][task["map_id"]]
    position, goal_relative, ray_values, endpoints = sample_ray_geometry(payload)

    fig = plt.figure(figsize=SLIDE_SIZE)
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, bottom=0.09, top=0.80,
                          width_ratios=(0.92, 1.45), height_ratios=(1.05, 0.95), hspace=0.30, wspace=0.22)
    add_title(
        fig,
        "What one expert-data sample contains",
        "A real TRAIN frame supplies a 34D deployable observation and a 3D normalized world-frame velocity label; sequence learners use the next H=16 labels.",
    )
    fig.text(0.67, 0.865, "34D observation", ha="center", fontsize=11, color="white",
             bbox={"boxstyle": "round,pad=0.35", "facecolor": COLORS["blue"], "edgecolor": "none"})
    fig.text(0.83, 0.865, "3D action", ha="center", fontsize=11, color="white",
             bbox={"boxstyle": "round,pad=0.35", "facecolor": COLORS["teal"], "edgecolor": "none"})
    fig.text(0.93, 0.865, "H=16", ha="center", fontsize=11, color=COLORS["ink"],
             bbox={"boxstyle": "round,pad=0.35", "facecolor": COLORS["green_soft"], "edgecolor": "none"})

    ax_scene = fig.add_subplot(gs[0, 0])
    add_topdown_obstacles(ax_scene, map_spec, alpha=0.42)
    for index, endpoint in enumerate(endpoints):
        color = COLORS["red"] if ray_values[index] < 0.999999 else COLORS["blue_soft"]
        ax_scene.plot([position[0], endpoint[0]], [position[1], endpoint[1]], color=color,
                      lw=1.25 if ray_values[index] < 0.999999 else 0.75, alpha=0.82)
    arrow_scale = min(1.7, max(0.8, float(np.linalg.norm(goal_relative)) * 0.3))
    goal_dir = goal_relative / max(float(np.linalg.norm(goal_relative)), 1.0e-9)
    ax_scene.arrow(position[0], position[1], goal_dir[0] * arrow_scale, goal_dir[1] * arrow_scale,
                   color=COLORS["gold"], width=0.025, head_width=0.18, length_includes_head=True, zorder=5)
    ax_scene.scatter(position[0], position[1], s=70, color=COLORS["ink"], edgecolor="white", linewidth=0.8, zorder=6)
    style_topdown(ax_scene, map_spec["world_bounds"], compact=True)
    ax_scene.set_title("Actual local sensing frame (top-down projection)", fontsize=10, fontweight="bold")
    ax_scene.text(0.02, 0.03, "coral: obstacle hit   blue-grey: clear to 3 m   gold: goal direction",
                  transform=ax_scene.transAxes, fontsize=6.8, color=COLORS["mid"],
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2})

    ax_action = fig.add_subplot(gs[1, 0])
    ax_action.set_aspect("equal")
    action = action_window[0]
    ax_action.axhline(0, color=COLORS["light"], lw=0.8)
    ax_action.axvline(0, color=COLORS["light"], lw=0.8)
    unit_circle = patches.Circle((0, 0), 1, facecolor=COLORS["pale"], edgecolor=COLORS["mid"], ls="--", lw=1)
    ax_action.add_patch(unit_circle)
    ax_action.arrow(0, 0, float(action[0]), float(action[1]), color=COLORS["teal"], width=0.025,
                    head_width=0.13, length_includes_head=True)
    ax_action.set_xlim(-1.15, 1.15)
    ax_action.set_ylim(-1.15, 1.15)
    ax_action.set_xlabel(r"$u_x$")
    ax_action.set_ylabel(r"$u_y$")
    ax_action.set_title("3D action label (xy projection)", fontsize=10, fontweight="bold")
    ax_action.text(0.02, 0.04,
                   f"u = [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}]\nAction norm = {np.linalg.norm(action):.3f}; z component shown numerically",
                   transform=ax_action.transAxes, fontsize=8.2, color=COLORS["ink"],
                   bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["light"]})

    ax_vector = fig.add_subplot(gs[0, 1])
    ax_vector.set_xlim(0, 34)
    ax_vector.set_ylim(0, 2.45)
    ax_vector.axis("off")
    groups = [
        ("position", 3, observation[0:3]),
        ("goal − position", 3, observation[3:6]),
        ("linear velocity", 3, observation[6:9]),
        ("quaternion", 4, observation[9:13]),
        ("angular velocity", 3, observation[13:16]),
        ("obstacle rays", 18, observation[16:34]),
    ]
    cursor = 0
    for color, (label, width, values) in zip(BLOCK_COLORS, groups):
        ax_vector.add_patch(patches.Rectangle((cursor, 1.32), width, 0.58, facecolor=color, edgecolor="white", lw=1))
        ax_vector.text(cursor + width / 2, 1.61, str(width), ha="center", va="center", color="white",
                       fontsize=9, fontweight="bold")
        ax_vector.text(cursor + width / 2, 1.18, label, ha="center", va="top", fontsize=7.2,
                       rotation=22 if width <= 4 else 0, color=COLORS["ink"])
        cursor += width
    ax_vector.text(0, 2.18, "Contiguous 34D vector — exact field order and dimensions", fontsize=10.5, fontweight="bold")
    ax_vector.text(0, 0.57,
                   f"position = {np.array2string(observation[0:3], precision=2)}\n"
                   f"goal − position = {np.array2string(observation[3:6], precision=2)}\n"
                   f"velocity = {np.array2string(observation[6:9], precision=2)}",
                   fontsize=8.3, family="monospace", va="center")
    ax_vector.text(18.0, 0.57,
                   f"quaternion = {np.array2string(observation[9:13], precision=2)}\n"
                   f"angular velocity = {np.array2string(observation[13:16], precision=2)}\n"
                   f"rays: min/median/max = {ray_values.min():.3f} / {np.median(ray_values):.3f} / {ray_values.max():.3f}",
                   fontsize=8.3, family="monospace", va="center")

    ax_window = fig.add_subplot(gs[1, 1])
    image = ax_window.imshow(action_window.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                             interpolation="nearest", extent=(0.5, 16.5, 2.5, -0.5))
    ax_window.set_yticks((0, 1, 2), (r"$u_x$", r"$u_y$", r"$u_z$"))
    ax_window.set_xticks((1, 4, 8, 12, 16))
    ax_window.set_xlabel("Future action index within the same episode")
    ax_window.set_title("Real H=16 × 3 target window for BC / Diffusion", fontsize=10, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax_window, fraction=0.025, pad=0.02)
    colorbar.set_label("normalized action", fontsize=8)
    ax_window.text(0.01, -0.34, "No padding • no episode crossing • only the first predicted action is executed at deployment",
                   transform=ax_window.transAxes, fontsize=8, color=COLORS["mid"])
    fig.text(0.97, 0.03, "Not included in observation: full obstacle map, future expert path, planner state or corridor coefficients",
             ha="right", fontsize=8.6, color=COLORS["red"])
    return save_figure(fig, "expert_sample_format")


def figure_statistics(inputs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    task_by_id = inputs["task_by_id"]
    trajectory_rows = [row for row in inputs["diversity"] if row.get("task_id")]
    if len(trajectory_rows) != 10000:
        raise AssertionError("trajectory metadata row count mismatch")
    lengths = np.asarray([float(row["episode_steps"]) for row in trajectory_rows])
    distances = np.asarray([float(row["start_goal_distance"]) for row in trajectory_rows])
    ratios = np.asarray([float(row["path_straight_ratio"]) for row in trajectory_rows])
    summary = inputs["summary"]

    fig = plt.figure(figsize=SLIDE_SIZE)
    add_title(
        fig,
        "Frozen expert dataset: scale, balance and quality-gated diversity",
        "Dataset-wide trajectory distributions use frozen metadata; the action-norm distribution uses every TRAIN transition only.",
    )
    cards = [
        ("1,000", "procedural maps"),
        ("10,000", "expert trajectories"),
        ("6,120,725", "total transitions"),
        ("10", "environment families"),
        ("0", "collisions / ground / nonfinite / clipping"),
    ]
    for index, (value, label) in enumerate(cards):
        x = 0.04 + index * 0.19
        card = patches.FancyBboxPatch((x, 0.77), 0.17, 0.105, transform=fig.transFigure,
                                      boxstyle="round,pad=0.008", facecolor=COLORS["pale"],
                                      edgecolor=COLORS["light"], linewidth=0.8)
        fig.add_artist(card)
        fig.text(x + 0.085, 0.828, value, ha="center", va="center", fontsize=16,
                 fontweight="bold", color=COLORS["blue"] if index < 4 else COLORS["green"])
        fig.text(x + 0.085, 0.785, label, ha="center", va="center", fontsize=7.6, color=COLORS["mid"])

    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, bottom=0.10, top=0.70, hspace=0.42, wspace=0.34)
    ax_len = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_ratio = fig.add_subplot(gs[0, 2])
    ax_action = fig.add_subplot(gs[1, 0])
    ax_family = fig.add_subplot(gs[1, 1:])

    hist_style = {"color": COLORS["blue_soft"], "edgecolor": "white", "linewidth": 0.5}
    ax_len.hist(lengths, bins=32, **hist_style)
    ax_len.set_xlabel("Trajectory length [steps]")
    ax_len.set_ylabel("Trajectories")
    ax_len.set_title("a  Length distribution", loc="left", fontweight="bold")
    ax_len.axvline(np.median(lengths), color=COLORS["red"], lw=1.4, ls="--")

    ax_dist.hist(distances, bins=32, **hist_style)
    ax_dist.set_xlabel("Start–goal distance [m]")
    ax_dist.set_ylabel("Trajectories")
    ax_dist.set_title("b  Task-distance distribution", loc="left", fontweight="bold")
    ax_dist.axvline(np.median(distances), color=COLORS["red"], lw=1.4, ls="--")

    ax_ratio.hist(ratios, bins=32, **hist_style)
    ax_ratio.set_xlabel("Path length / start–goal distance")
    ax_ratio.set_ylabel("Trajectories")
    ax_ratio.set_title("c  Geometric path ratio", loc="left", fontweight="bold")
    ax_ratio.axvline(np.median(ratios), color=COLORS["red"], lw=1.4, ls="--")

    edges = payload["action_histogram_edges"]
    counts = payload["action_histogram_counts"]
    centers, widths = (edges[:-1] + edges[1:]) / 2.0, np.diff(edges)
    percentages = counts / counts.sum() * 100.0
    ax_action.bar(centers, percentages, width=widths, color=COLORS["teal"], edgecolor="white", linewidth=0.35)
    ax_action.set_xlabel(r"Action norm $\Vert u \Vert_2$")
    ax_action.set_ylabel("TRAIN transitions [%]")
    ax_action.set_title("d  Action-norm distribution", loc="left", fontweight="bold")
    ax_action.set_xlim(0, 1.02)
    ax_action.text(0.02, 0.96, "Includes zero-action settle periods", transform=ax_action.transAxes,
                   fontsize=7.5, color=COLORS["mid"], va="top")

    family_counts = summary["family_trajectory_counts"]
    labels = [pretty_family(name).replace(" ", "\n") for name in FAMILIES]
    values = [family_counts[name] for name in FAMILIES]
    bars = ax_family.bar(range(len(FAMILIES)), values, color=COLORS["blue"], edgecolor="white", linewidth=0.5)
    ax_family.set_xticks(range(len(FAMILIES)), labels, fontsize=7)
    ax_family.set_ylabel("Trajectories")
    ax_family.set_ylim(0, 1160)
    ax_family.set_title("e  Balanced family coverage", loc="left", fontweight="bold")
    for bar, value in zip(bars, values):
        ax_family.text(bar.get_x() + bar.get_width() / 2, value + 28, f"{value:,}", ha="center", fontsize=7.2)
    for axis in (ax_len, ax_dist, ax_ratio, ax_action, ax_family):
        axis.grid(axis="y", color="#E1E6EB", linewidth=0.55, alpha=0.65)
    fig.text(0.97, 0.045,
             f"Map-level leakage: {summary['map_overlaps']['train_val'] + summary['map_overlaps']['train_test'] + summary['map_overlaps']['val_test']}  •  "
             f"Candidate acceptance: {summary['acceptance_rate'] * 100:.4f}%  •  TEST remains sealed",
             ha="right", fontsize=8.5, color=COLORS["mid"])
    return save_figure(fig, "expert_dataset_statistics")


def figure_observation_rays(inputs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    task = inputs["task_by_id"][payload["sample_task_id"]]
    map_spec = inputs["map_by_id"][task["map_id"]]
    position, goal_relative, ray_values, endpoints = sample_ray_geometry(payload)
    fig = plt.figure(figsize=SLIDE_SIZE)
    ax = fig.add_axes([0.04, 0.07, 0.77, 0.80], projection="3d")
    add_title(
        fig,
        "What the student senses at one instant: goal information + 18 obstacle rays",
        "All rays are decoded from one real TRAIN observation. Faded cuboids are explanatory context and are not passed to the student as a full map.",
    )
    add_cuboid_obstacles(ax, map_spec, alpha=0.20)
    hit_drawn = clear_drawn = False
    for value, endpoint in zip(ray_values, endpoints):
        hit = value < 0.999999
        color = COLORS["red"] if hit else COLORS["blue_soft"]
        label = "Obstacle hit" if hit and not hit_drawn else "Clear to 3 m" if (not hit and not clear_drawn) else None
        ax.plot([position[0], endpoint[0]], [position[1], endpoint[1]], [position[2], endpoint[2]],
                color=color, lw=2.3 if hit else 1.0, alpha=0.95 if hit else 0.55, label=label)
        hit_drawn = hit_drawn or hit
        clear_drawn = clear_drawn or (not hit)
    goal_norm = float(np.linalg.norm(goal_relative))
    direction = goal_relative / max(goal_norm, 1.0e-9)
    ax.quiver(*position, *(direction * 1.6), color=COLORS["gold"], linewidth=2.8, arrow_length_ratio=0.16, label="Goal direction")
    ax.scatter(*position, s=95, color=COLORS["ink"], edgecolor="white", linewidth=0.8, depthshade=False, label="Drone")
    style_3d(ax, map_spec["world_bounds"], local_center=position)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.96), fontsize=8.5)

    sample_time = payload["sample_step"] / 48.0
    fig.text(0.78, 0.68, "ACTUAL OBSERVATION", fontsize=9, fontweight="bold", color=COLORS["blue"])
    fig.text(0.78, 0.62,
             f"Task\n{payload['sample_task_id']}\n\nStep / time\n{payload['sample_step']} / {sample_time:.3f} s\n\nVisible obstacle rays\n{payload['visible_ray_count']} / 18\n\nNearest ray distance\n{payload['minimum_ray_fraction'] * RAY_RANGE_M:.3f} m",
             fontsize=9.2, va="top", color=COLORS["ink"])
    fig.text(0.78, 0.24,
             "Student receives:\n• absolute position\n• goal-relative position\n• velocities + attitude\n• 18 local ray fractions\n\nStudent does not receive:\n• full obstacle list\n• expert future path\n• planner/corridor state",
             fontsize=8.8, va="top", color=COLORS["ink"],
             bbox={"boxstyle": "round,pad=0.55", "facecolor": COLORS["pale"], "edgecolor": COLORS["light"]})
    return save_figure(fig, "observation_ray_visualization")


def draw_pipeline_arrow(ax: plt.Axes, x0: float, x1: float, y: float = 0.51) -> None:
    ax.annotate("", xy=(x1, y), xytext=(x0, y), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": COLORS["mid"], "mutation_scale": 14})


def figure_pipeline(inputs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    task = inputs["hero"]
    map_spec = inputs["map_by_id"][task["map_id"]]
    positions = payload["positions"][HERO_TASK_ID]
    action_window = payload["sample_action_window"]
    observation = payload["sample_observation"]
    train_windows = sum(max(0, row["length"] - HORIZON + 1) for row in payload["episode_index"].values())

    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    ax.set_axis_off()
    add_title(
        fig,
        "How frozen GCOPTER rollouts become later BC / Diffusion training inputs",
        "This is a data-preparation bridge, not a model-result figure; S8-R5 performs no BC, Diffusion or PPO training.",
    )
    stages = [
        (0.035, 0.22, "1", "Map + expert rollout"),
        (0.245, 0.15, "2", "Episode arrays"),
        (0.425, 0.15, "3", "Episode-bounded windows"),
        (0.605, 0.15, "4", "Observation / action pair"),
        (0.785, 0.18, "5", "Learning inputs"),
    ]
    for x, width, number, title in stages:
        card = patches.FancyBboxPatch((x, 0.20), width, 0.57, transform=ax.transAxes,
                                      boxstyle="round,pad=0.012", facecolor="#FBFCFD",
                                      edgecolor=COLORS["light"], linewidth=1.1)
        ax.add_patch(card)
        ax.text(x + 0.018, 0.73, number, transform=ax.transAxes, fontsize=10, color="white", fontweight="bold",
                bbox={"boxstyle": "circle,pad=0.25", "facecolor": COLORS["blue"], "edgecolor": "none"})
        ax.text(x + width / 2, 0.665, title, transform=ax.transAxes, fontsize=8.6, fontweight="bold", ha="center")
    for x0, x1 in ((0.225, 0.242), (0.395, 0.422), (0.575, 0.602), (0.755, 0.782)):
        draw_pipeline_arrow(ax, x0, x1)

    mini = ax.inset_axes([0.055, 0.29, 0.175, 0.37])
    add_topdown_obstacles(mini, map_spec, alpha=0.65)
    mini.plot(positions[:, 0], positions[:, 1], color=COLORS["blue"], lw=1.5)
    style_topdown(mini, map_spec["world_bounds"], compact=True)
    mini.set_xticks([])
    mini.set_yticks([])
    mini.set_xlabel("real TRAIN trajectory", fontsize=7)

    ax.text(0.265, 0.59, "observations  (L × 34)\nactions       (L × 3)\npositions     (L × 3)\ntimes          (L)\n\nepisode offset + length",
            transform=ax.transAxes, fontsize=8.3, family="monospace", va="top", color=COLORS["ink"])
    ax.text(0.265, 0.27, "One task = one episode\nNo timestep sharing across splits", transform=ax.transAxes,
            fontsize=7.6, color=COLORS["mid"])

    for index, y in enumerate((0.56, 0.49, 0.42)):
        ax.add_patch(patches.Rectangle((0.448 + index * 0.018, y), 0.105, 0.055, transform=ax.transAxes,
                                       facecolor=COLORS["green_soft"], edgecolor=COLORS["green"], alpha=0.88))
        ax.text(0.500 + index * 0.018, y + 0.027, "16 × 3", transform=ax.transAxes,
                ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(0.50, 0.34, "obs[t]  →  actions[t:t+16]", transform=ax.transAxes, ha="center", fontsize=8.2)
    ax.text(0.50, 0.28, "No padding\nNo episode crossing", transform=ax.transAxes, ha="center", fontsize=7.6, color=COLORS["mid"])

    cursor = 0.625
    for color, width in zip(BLOCK_COLORS, (3, 3, 3, 4, 3, 18)):
        scaled = width / 34 * 0.11
        ax.add_patch(patches.Rectangle((cursor, 0.57), scaled, 0.06, transform=ax.transAxes,
                                       facecolor=color, edgecolor="white", linewidth=0.5))
        cursor += scaled
    ax.text(0.68, 0.65, "34D observation", transform=ax.transAxes, ha="center", fontsize=8.3, fontweight="bold")
    ax.annotate("", xy=(0.72, 0.43), xytext=(0.64, 0.43), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": COLORS["teal"]})
    ax.text(0.68, 0.47, f"3D action\n{np.array2string(action_window[0], precision=2)}", transform=ax.transAxes,
            ha="center", fontsize=7.8)
    ax.text(0.68, 0.28, "Actual values from one TRAIN frame", transform=ax.transAxes,
            ha="center", fontsize=7.4, color=COLORS["mid"])

    ax.text(0.805, 0.60, "Behavior cloning", transform=ax.transAxes, fontsize=9, fontweight="bold", color=COLORS["blue"])
    ax.text(0.805, 0.54, "34D condition\n→ deterministic H=16 × 3 target", transform=ax.transAxes, fontsize=7.8)
    ax.text(0.805, 0.43, "Diffusion policy", transform=ax.transAxes, fontsize=9, fontweight="bold", color=COLORS["violet"])
    ax.text(0.805, 0.37, "34D condition + noisy action sequence\n→ denoised H=16 × 3 target", transform=ax.transAxes, fontsize=7.8)
    ax.text(0.805, 0.26, "INPUTS ONLY\nno training or model comparison here", transform=ax.transAxes,
            fontsize=7.6, color=COLORS["red"], fontweight="bold")

    ax.text(0.50, 0.10,
            f"Frozen TRAIN provides {train_windows:,} valid H=16 windows from 8,000 whole episodes  •  TEST payload remains unopened",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=COLORS["ink"],
            bbox={"boxstyle": "round,pad=0.45", "facecolor": COLORS["pale"], "edgecolor": COLORS["light"]})
    return save_figure(fig, "expert_to_learning_pipeline")


def quantiles(values: np.ndarray) -> dict[str, float]:
    names = ("min", "q25", "median", "q75", "q95", "max")
    numbers = np.quantile(values, (0.0, 0.25, 0.5, 0.75, 0.95, 1.0))
    return {name: float(number) for name, number in zip(names, numbers)}


def write_metadata(inputs: dict[str, Any], payload: dict[str, Any], figures: list[dict[str, Any]]) -> None:
    META_ROOT.mkdir(parents=True, exist_ok=True)
    summary = inputs["summary"]
    task_by_id = inputs["task_by_id"]
    trajectory_rows = [row for row in inputs["diversity"] if row.get("task_id")]
    lengths = np.asarray([float(row["episode_steps"]) for row in trajectory_rows])
    distances = np.asarray([float(row["start_goal_distance"]) for row in trajectory_rows])
    ratios = np.asarray([float(row["path_straight_ratio"]) for row in trajectory_rows])
    train_windows = sum(max(0, row["length"] - HORIZON + 1) for row in payload["episode_index"].values())

    source_hashes = {
        "dataset_summary.json": sha256_file(SUMMARY_PATH),
        "map_manifest.json": sha256_file(MAP_MANIFEST_PATH),
        "task_manifest.csv": sha256_file(TASK_MANIFEST_PATH),
        "diversity_statistics.csv": sha256_file(DIVERSITY_PATH),
        "train.npz": sha256_file(TRAIN_PATH),
    }
    if source_hashes["train.npz"] != summary["dataset_sha256"]["train"]:
        raise AssertionError("TRAIN NPZ hash differs from frozen dataset summary")
    if source_hashes["map_manifest.json"] != summary["map_manifest_sha256"]:
        raise AssertionError("map manifest hash differs from frozen summary")
    if source_hashes["task_manifest.csv"] != summary["task_manifest_sha256"]:
        raise AssertionError("task manifest hash differs from frozen summary")

    representatives = []
    for row in inputs["representatives"]:
        episode = payload["episode_index"][row["task_id"]]
        representatives.append({**row, **episode, "split": task_by_id[row["task_id"]]["split"]})
    selection = {
        "task": "S8-R5-EXPERT-DATA-VISUALIZATION-FOR-ADVISOR-BRIEFING-V1",
        "random_seed": None,
        "random_selection_used": False,
        "family_representatives": representatives,
        "same_map": {
            "map_id": inputs["hero"]["map_id"],
            "family": inputs["hero"]["family"],
            "split": "train",
            "task_ids": [row["task_id"] for row in inputs["hero_map_tasks"]],
            "selection_reason": "fixed TRAIN double-gate map with ten tasks, two gate structures, and clear xy/z task diversity",
        },
        "hero_trajectory": {
            "task_id": HERO_TASK_ID,
            "map_id": inputs["hero"]["map_id"],
            "split": "train",
            **payload["episode_index"][HERO_TASK_ID],
            "selection_reason": "same-map narrative continuity plus visible lateral/vertical gate traversal; illustration only, not a performance extremum claim",
        },
        "observation_sample": {
            "task_id": payload["sample_task_id"],
            "map_id": task_by_id[payload["sample_task_id"]]["map_id"],
            "split": "train",
            **payload["episode_index"][payload["sample_task_id"]],
            "step": payload["sample_step"],
            "time_s": payload["sample_step"] / 48.0,
            "visible_obstacle_rays": payload["visible_ray_count"],
            "minimum_ray_fraction": payload["minimum_ray_fraction"],
            "minimum_ray_distance_m": payload["minimum_ray_fraction"] * RAY_RANGE_M,
            "selection_rule": "within the selected TRAIN map, maximize rays < 1; tie by minimum ray fraction, earliest step, then task ID; require a complete H=16 window",
        },
        "test_payload_accessed": False,
    }
    statistics = {
        "dataset_scale": {
            "maps": summary["total_maps"],
            "trajectories": summary["total_trajectories"],
            "transitions": summary["total_transitions"],
            "families": len(summary["family_counts"]),
            "train_trajectories": summary["train_trajectories"],
            "train_transitions": summary["split_stats"]["train"]["transitions"],
            "train_h16_windows": train_windows,
        },
        "quality": {
            "collision_count": summary["collision_count"],
            "ground_count": summary["ground_count"],
            "nonfinite_count": summary["nonfinite_count"],
            "action_support_violation": summary["action_support_violation"],
            "environment_clipping": summary["environment_clipping"],
            "map_overlaps": summary["map_overlaps"],
            "action_violation_count_recomputed_on_train": payload["action_violation_count"],
        },
        "dataset_metadata_distributions": {
            "trajectory_steps": quantiles(lengths),
            "start_goal_distance_m": quantiles(distances),
            "path_straight_ratio": quantiles(ratios),
        },
        "train_transition_action_norm_quantiles": {
            name: float(value) for name, value in zip(
                ("min", "q25", "median", "q75", "q95", "q99", "max"), payload["action_quantiles"]
            )
        },
        "action_norm_note": "TRAIN transition distribution includes zero-action initialization/settle periods.",
        "family_trajectory_counts": summary["family_trajectory_counts"],
    }
    manifest = {
        "task": "S8-R5-EXPERT-DATA-VISUALIZATION-FOR-ADVISOR-BRIEFING-V1",
        "status": "PENDING_INDEPENDENT_VERIFICATION",
        "backend": "Python / Matplotlib only",
        "core_conclusion": "The frozen S8-R2 corpus contains diverse, traceable GCOPTER expert rollouts and real 34D-to-3D/H16 samples that document the data foundation for later learning; these are not model-result figures.",
        "figure_archetypes": {
            "expert_family_overview": "light image plate + trajectory evidence",
            "ten_tasks_same_map": "single hero map plate",
            "single_expert_trajectory_3d": "3D image plate",
            "expert_trajectory_timeseries": "quantitative grid",
            "expert_sample_format": "schematic-led composite",
            "expert_dataset_statistics": "asymmetric quantitative composite",
            "observation_ray_visualization": "3D schematic-led evidence",
            "expert_to_learning_pipeline": "schematic-led process",
        },
        "export_contract": {"primary": "editable SVG", "ppt_preview": "300 dpi PNG", "canvas_inches": list(SLIDE_SIZE)},
        "source_hashes": source_hashes,
        "source_payloads_opened": ["artifacts/s8r2_10k/train.npz"],
        "metadata_inputs": [
            "artifacts/s8r2_10k/dataset_summary.json",
            "artifacts/s8r2_10k/map_manifest.json",
            "artifacts/s8r2_10k/task_manifest.csv",
            "artifacts/s8r2_10k/diversity_statistics.csv",
        ],
        "test_payload_accessed": False,
        "training_or_model_evaluation_run": False,
        "scientific_contract_modified": False,
        "figures": figures,
        "review_risks_addressed": [
            "All trajectory panels assert TRAIN split before loading arrays.",
            "Exact manifest goals are drawn separately from rollout endpoints.",
            "Dataset-wide metadata and TRAIN-only raw action statistics are labelled separately.",
            "Obstacle rays are described as local sensing while absolute position remains part of the observation.",
            "No model outcome or causal performance claim is shown.",
        ],
    }
    for name, value in (("selection_manifest.json", selection), ("statistics.json", statistics), ("figure_manifest.json", manifest)):
        (META_ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT,
                        help="Accepted for explicit reproducibility; must resolve to this checkout.")
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT.resolve():
        raise ValueError(f"this script is bound to its checkout: {ROOT}")

    inputs = load_inputs()
    payload = extract_train_payload(inputs)
    figures = [
        figure_family_overview(inputs, payload),
        figure_ten_tasks(inputs, payload),
        figure_single_trajectory(inputs, payload),
        figure_timeseries(inputs, payload),
        figure_sample_format(inputs, payload),
        figure_statistics(inputs, payload),
        figure_observation_rays(inputs, payload),
        figure_pipeline(inputs, payload),
    ]
    write_metadata(inputs, payload, figures)
    print(json.dumps({
        "task": "S8-R5-EXPERT-DATA-VISUALIZATION-FOR-ADVISOR-BRIEFING-V1",
        "figures": [row["png"] for row in figures],
        "test_payload_accessed": False,
        "training_run": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
