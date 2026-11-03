#!/usr/bin/env python3
"""Render a PMGS checkpoint without computing metrics or initializing LPIPS."""
from evaluate import main


if __name__ == "__main__":
    main(render_only_default=True)
