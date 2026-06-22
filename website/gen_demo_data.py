#!/usr/bin/env python3
"""Build demo_data.js for the interactive demo in index.html.

Walks videos/web_cases/prompt_*/ folders, reads each prompt text and its
metrics_summary.csv, and emits a compact JS array assigned to
window.DEMO_DATA. One entry per prompt, with up to 64 variant records.

Score definitions (per project spec):
  R_dyn  = R_dyn_Q
  R_text = exp(-R_sem_matching_score)
"""
import csv
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "videos", "web_cases")
OUT = os.path.join(HERE, "demo_data.js")


def fnum(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def main():
    prompts = []
    folders = sorted(
        d for d in os.listdir(CASES)
        if d.startswith("prompt_") and os.path.isdir(os.path.join(CASES, d))
    )
    for folder in folders:
        m = re.match(r"prompt_(\d+)$", folder)
        if not m:
            continue
        pid = m.group(1)
        fdir = os.path.join(CASES, folder)

        txt_path = os.path.join(fdir, f"prompt_{pid}.txt")
        try:
            with open(txt_path, encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            text = ""

        csv_path = os.path.join(fdir, "metrics_summary.csv")
        if not os.path.isfile(csv_path):
            continue
        viddir = os.path.join(fdir, "video")

        variants = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("motion_key") or "").strip()
                if not key:
                    continue
                if not os.path.isfile(os.path.join(viddir, key + ".mp4")):
                    continue
                r_sem = fnum(row, "R_sem_matching_score")
                variants.append({
                    "key": key,
                    "rdyn": round(fnum(row, "R_dyn_Q"), 4),
                    "rtext": round(math.exp(-r_sem), 4),
                    "succ": int(fnum(row, "success")),
                    "jae": round(fnum(row, "joint_angle_error"), 4),
                    "mpjpe": round(fnum(row, "mpjpe_l"), 2),
                    "accel": round(fnum(row, "accel_dist"), 2),
                    "vel": round(fnum(row, "vel_dist"), 2),
                    "done": (row.get("done_reason") or "").strip(),
                })
        if not variants:
            continue
        prompts.append({"id": pid, "text": text, "variants": variants})

    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.DEMO_DATA = " + payload + ";\n")

    total = sum(len(p["variants"]) for p in prompts)
    print(f"Wrote {OUT}")
    print(f"  prompts: {len(prompts)}")
    print(f"  variants total: {total}")
    if prompts:
        s = prompts[0]
        print(f"  sample[0]: id={s['id']} variants={len(s['variants'])} text={s['text']!r}")
        print(f"  sample variant: {s['variants'][0]}")


if __name__ == "__main__":
    main()
