#!/usr/bin/env python3
"""Check all bundled Qgain plans without loading datasets or using a GPU."""
import argparse
import json

import _bootstrap
from pmgs.allocation import load_allocation
from pmgs.workflow import SCENES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=list(SCENES) + ["all"], default=["all"])
    args = parser.parse_args()
    scenes = SCENES if "all" in args.scenes else args.scenes
    rows = []
    for scene in scenes:
        counts, entry = load_allocation(scene)
        rows.append(dict(scene=scene, faces=len(counts), gaussians=int(counts.sum()),
                         maximum_per_face=int(counts.max()), sha256=entry["sha256"], status="passed"))
    print(json.dumps(dict(status="passed", allocations=rows), indent=2))


if __name__ == "__main__":
    main()
