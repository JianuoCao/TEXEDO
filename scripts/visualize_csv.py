"""Render 36-dim G1 motions to MuJoCo MP4 videos (headless, EGL backend).

Each ``(T, 36)`` motion is played back on the Unitree G1 model and rendered to an
``.mp4`` with MuJoCo's offscreen EGL renderer (no X server needed). This ports the
proven renderer from ``GenMimic/scripts/visualize_csv_egl.py`` so the candidates
produced by ``pipeline.generate`` get real robot videos instead of static plots.

    python scripts/visualize_csv.py --input motion.csv --output-dir viz/
    python scripts/visualize_csv.py --input-dir candidates/ --output-dir candidates/

Input formats
-------------
* ``.csv`` — already in *CSV ordering* (root_pos, quat **xyzw**, joints reordered to
  the MuJoCo qpos order). Rendered as-is. This is what ``generator/demo.py`` writes.
* ``.npy`` — raw *NPZ ordering* (root_pos, quat **wxyz**, joints in NPZ order). It is
  converted to CSV ordering on the fly before rendering.

The G1 model XML defaults to the in-repo asset ``assets/robot/g1/g1_29dof_rev_1_0.xml``
(resolved via ``textseedo.paths``); override with ``--model``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# EGL must be selected before mujoco is imported.
os.environ["MUJOCO_GL"] = "egl"

import numpy as np

# Resolve the G1 model from the in-repo assets so nothing outside the repo is referenced.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from textseedo.paths import assets

DEFAULT_MODEL_PATH = str(assets("robot", "g1", "g1_29dof_rev_1_0.xml"))

# NPZ joint order -> CSV (MuJoCo qpos) joint order. Mirrors
# generator/mgpt/archs/fsq_arch.py:convert_to_csv_format.
_NPZ_TO_CSV = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
]


def _npz_to_csv_format(data: np.ndarray) -> np.ndarray:
    """Convert a (T, 36) motion from NPZ ordering to CSV ordering.

    Quaternion wxyz -> xyzw, and joints reordered to the MuJoCo qpos order.
    """
    out = np.zeros_like(data)
    out[:, :3] = data[:, :3]            # root position
    out[:, 3] = data[:, 4]              # x
    out[:, 4] = data[:, 5]              # y
    out[:, 5] = data[:, 6]              # z
    out[:, 6] = data[:, 3]             # w
    joints = data[:, 7:36]
    for i, src in enumerate(_NPZ_TO_CSV):
        out[:, 7 + i] = joints[:, src]
    return out


def _load_motion_csv(path: Path) -> np.ndarray:
    """Load a motion file and return it in CSV ordering, shape (T, 36)."""
    if path.suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
        if arr.shape[1] >= 37:
            arr = arr[:, -36:]
        return _npz_to_csv_format(arr)
    if path.suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",")
        arr = np.atleast_2d(arr)
        if arr.shape[1] >= 37:  # drop a leading frame-index column if present
            arr = arr[:, -36:]
        return np.asarray(arr, dtype=np.float32)
    raise ValueError(f"Unsupported file type: {path}")


def _apply_white_floor_gray_grid(m) -> None:
    """Use a white background/floor with a light gray grid."""
    import mujoco

    # White skybox/background.
    for tex_id in range(m.ntex):
        if m.tex_type[tex_id] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            adr = m.tex_adr[tex_id]
            size = m.tex_height[tex_id] * m.tex_width[tex_id] * m.tex_nchannel[tex_id]
            m.tex_data[adr:adr + size] = 255

    # White ground texture with thin gray tile borders.
    ground_tex_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "groundplane")
    if ground_tex_id >= 0:
        width = m.tex_width[ground_tex_id]
        height = m.tex_height[ground_tex_id]
        channels = m.tex_nchannel[ground_tex_id]
        adr = m.tex_adr[ground_tex_id]
        tile = np.full((height, width, channels), 255, dtype=np.uint8)
        grid_color = 210
        tile[0, :, :] = grid_color
        tile[-1, :, :] = grid_color
        tile[:, 0, :] = grid_color
        tile[:, -1, :] = grid_color
        m.tex_data[adr:adr + height * width * channels] = tile.reshape(-1)

    ground_mat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
    if ground_mat_id >= 0:
        m.mat_rgba[ground_mat_id] = np.array([1.0, 1.0, 1.0, 1.0])
        m.mat_reflectance[ground_mat_id] = 0.0

    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        m.geom_rgba[floor_id] = np.array([1.0, 1.0, 1.0, 1.0])

    m.vis.rgba.haze[:] = np.array([1.0, 1.0, 1.0, 1.0])
    try:
        m.vis.rgba.fog[:] = np.array([1.0, 1.0, 1.0, 1.0])
    except AttributeError:
        pass


def render_motion(
    data: np.ndarray,
    output_path: str,
    fps: int = 50,
    resolution=(960, 720),
    model_path: str = None,
    track_body: str = "pelvis",
    track_distance: float = 3.0,
    track_azimuth: float = 90.0,
    track_elevation: float = -25.0,
    lookat_height: float = 0.35,
    dynamic_root_camera: bool = False,
    use_xml_camera: str = None,
    crf: int = 20,
    preset: str = "medium",
) -> bool:
    """Render a (T, 36) CSV-ordered motion to an MP4 via EGL offscreen rendering."""
    import mujoco
    import cv2

    try:
        data = np.asarray(data, dtype=np.float64)

        # AMASS Z offset: lift onto the floor if the root starts near z=0.
        if data[0, 2] < 0.1:
            data[:, 2] += 0.793

        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        m = mujoco.MjModel.from_xml_path(model_path)
        d = mujoco.MjData(m)
        m.opt.timestep = 1.0 / fps

        # Clean look: no shadows / ground reflection.
        if m.nlight:
            m.light_castshadow[:] = 0
        if m.nmat:
            m.mat_reflectance[:] = 0.0
        _apply_white_floor_gray_grid(m)

        # Camera setup.
        dynamic_camera = None
        if use_xml_camera is not None:
            camera_arg = use_xml_camera
        elif dynamic_root_camera:
            dynamic_camera = mujoco.MjvCamera()
            dynamic_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            dynamic_camera.distance = track_distance
            dynamic_camera.azimuth = track_azimuth
            dynamic_camera.elevation = track_elevation
            camera_arg = dynamic_camera
        elif track_body is not None:
            body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, track_body)
            if body_id < 0:
                raise ValueError(f"Body '{track_body}' not found in model")
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = body_id
            cam.distance = track_distance
            cam.azimuth = track_azimuth
            cam.elevation = track_elevation
            camera_arg = cam
        else:
            camera_arg = "follow"

        width, height = resolution
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        renderer = mujoco.Renderer(m, height=height, width=width)

        temp_path = output_path.replace(".mp4", "_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {temp_path}")

        print(f"  Rendering {len(data)} frames at {fps} FPS, {width}x{height}...")
        for i, frame in enumerate(data):
            # qpos free joint: [pos(3), quat_wxyz(4), joints(29)].
            d.qpos[:36] = frame
            d.qpos[3] = frame[6]      # w
            d.qpos[4:7] = frame[3:6]  # xyz
            mujoco.mj_forward(m, d)
            if dynamic_camera is not None:
                dynamic_camera.lookat[:] = d.qpos[:3]
                dynamic_camera.lookat[2] += lookat_height
            renderer.update_scene(d, camera=camera_arg)
            img = renderer.render()
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(data)} ({100*(i+1)/len(data):.1f}%)")

        writer.release()
        renderer.close()

        # Re-encode to H.264 for portability; fall back to the raw mp4v file.
        try:
            subprocess.run(
                ["ffmpeg", "-i", temp_path, "-vcodec", "libx264", "-crf", str(crf),
                 "-preset", preset, "-pix_fmt", "yuv420p", "-y", output_path],
                check=True, capture_output=True, text=True,
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            msg = "ffmpeg not found" if isinstance(e, FileNotFoundError) else "ffmpeg failed"
            print(f"  Warning: {msg}, keeping mp4v output")
            if os.path.exists(temp_path):
                os.replace(temp_path, output_path)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ {output_path} ({size_mb:.2f} MB)")
        return True

    except Exception as e:  # noqa: BLE001 - report and continue with other clips
        print(f"  ✗ Error rendering {output_path}: {e}")
        return False


def _render_file(args: tuple) -> tuple[str, bool]:
    (
        file_path,
        output_dir,
        fps,
        width,
        height,
        model_path,
        dynamic_root_camera,
        use_xml_camera,
        track_distance,
        track_azimuth,
        track_elevation,
        lookat_height,
        crf,
        preset,
    ) = args
    file_path = Path(file_path)
    out_mp4 = str(Path(output_dir) / f"{file_path.stem}.mp4")
    motion = _load_motion_csv(file_path)
    ok = render_motion(
        motion,
        out_mp4,
        fps=fps,
        resolution=(width, height),
        model_path=model_path,
        dynamic_root_camera=dynamic_root_camera,
        use_xml_camera=use_xml_camera,
        track_distance=track_distance,
        track_azimuth=track_azimuth,
        track_elevation=track_elevation,
        lookat_height=lookat_height,
        crf=crf,
        preset=preset,
    )
    return str(file_path), ok


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render 36-dim G1 motions to MuJoCo MP4 videos (EGL headless)"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="Single .npy/.csv motion")
    g.add_argument("--input-dir", type=Path, help="Directory of motions")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default=None,
                   help=f"Robot model XML (default: {DEFAULT_MODEL_PATH})")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--prefer-npy", action="store_true",
                   help="When both .csv and .npy exist for a stem, render from .npy")
    p.add_argument("--xml-camera", type=str, default=None,
                   help="Use a named camera from the XML (e.g. 'follow') instead of "
                        "the default pelvis-tracking camera")
    p.add_argument("--free-camera", action="store_true",
                   help="Use the old root-following free camera instead of pelvis tracking")
    p.add_argument("--track-distance", type=float, default=3.0)
    p.add_argument("--track-azimuth", type=float, default=90.0)
    p.add_argument("--track-elevation", type=float, default=-25.0)
    p.add_argument("--lookat-height", type=float, default=0.35,
                   help="Meters above the root position that the dynamic camera looks at")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", type=str, default="medium")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel render workers for --input-dir")
    args = p.parse_args()

    if args.input:
        files = [args.input]
    else:
        csvs = sorted(args.input_dir.glob("*.csv"))
        npys = sorted(args.input_dir.glob("*.npy"))
        if args.prefer_npy:
            files = npys or csvs
        else:
            # Prefer .csv (already CSV-ordered); add .npy only for stems without a .csv.
            csv_stems = {f.stem for f in csvs}
            files = csvs + [f for f in npys if f.stem not in csv_stems]
    if not files:
        raise SystemExit("No motion files (.csv/.npy) found")

    model_path = args.model or DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        raise SystemExit(
            f"G1 model XML not found at {model_path}.\n"
            "Fetch the in-repo robot asset with:\n"
            "    python scripts/download_assets.py --only g1_robot\n"
            "or pass an explicit --model /path/to/g1_29dof_rev_1_0.xml"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_args = [
        (
            str(f),
            str(args.output_dir),
            args.fps,
            args.width,
            args.height,
            model_path,
            args.free_camera and args.xml_camera is None,
            args.xml_camera,
            args.track_distance,
            args.track_azimuth,
            args.track_elevation,
            args.lookat_height,
            args.crf,
            args.preset,
        )
        for f in files
    ]

    ok = 0
    workers = max(1, args.workers)
    if workers == 1 or len(files) == 1:
        for item in worker_args:
            _, rendered = _render_file(item)
            ok += int(rendered)
    else:
        print(f"Rendering {len(files)} files with {workers} EGL worker processes...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_render_file, item) for item in worker_args]
            for future in as_completed(futures):
                file_path, rendered = future.result()
                ok += int(rendered)
                status = "✓" if rendered else "✗"
                print(f"{status} {Path(file_path).name}")
    print(f"Done: {ok}/{len(files)} rendered -> {args.output_dir}")


if __name__ == "__main__":
    main()
