"""Assemble the bundled edit animation's PNG frames into video and GIF."""
import importlib.util
from pathlib import Path
import sys


def main():
    example = Path(__file__).resolve().parents[1] / 'examples' / 'lego_bucket_lift'
    sys.path.insert(0, str(example))
    spec = importlib.util.spec_from_file_location('_pmgs_animation_assembly_entry', str(example / 'assemble.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == '__main__':
    main()
