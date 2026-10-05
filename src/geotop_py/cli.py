"""Command-line runner for GEOtop-format 1D cases.

Read geotop.inpts and its inputs, simulate each point, and write the requested
output tables and profiles. EnergyBalance must be enabled; WaterBalance may
be enabled or disabled.
"""

from __future__ import annotations

import argparse
import sys
import time

from .pipeline import run_simulation


class _Progress:
    """Small dependency-free terminal progress reporter."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.tty = self.stream.isatty()
        self.started = time.monotonic()
        self.last_draw = 0.0
        self.last_bucket = -1

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, round(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return (f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours
                else f"{minutes:02d}:{seconds:02d}")

    def __call__(self, completed: int, total: int, date) -> None:
        now = time.monotonic()
        fraction = completed / total if total else 1.0
        elapsed = now - self.started
        eta = elapsed * (total - completed) / completed if completed else 0.0

        if self.tty:
            if completed < total and now - self.last_draw < 0.1:
                return
            width = 30
            filled = min(width, round(width * fraction))
            bar = "#" * filled + "-" * (width - filled)
            line = (f"\r[{bar}] {fraction:6.1%}  {completed}/{total}  "
                    f"{date:%Y-%m-%d %H:%M}  "
                    f"elapsed {self._duration(elapsed)}  "
                    f"ETA {self._duration(eta)}")
            self.stream.write(line)
            if completed >= total:
                self.stream.write("\n")
            self.stream.flush()
            self.last_draw = 0.0 if completed >= total else now
        else:
            # Keep redirected logs compact: one message per 10 percent.
            bucket = min(10, int(fraction * 10))
            if bucket == self.last_bucket and completed < total:
                return
            self.stream.write(
                f"GEOtopPY: {fraction:5.1%} ({completed}/{total}), "
                f"simulation time {date:%Y-%m-%d %H:%M}, "
                f"ETA {self._duration(eta)}\n")
            self.stream.flush()
            self.last_bucket = bucket

    def close(self) -> None:
        if self.tty and self.last_draw:
            self.stream.write("\n")
            self.stream.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="GEOtopPY",
        description="GEOtop-compatible 1D runner (WaterBalance 0 or 1).")
    ap.add_argument("working_dir",
                    help="directory containing geotop.inpts and its input files")
    ap.add_argument("--suffix", default="",
                    help="write to output_tabs<suffix>/ and output_prof<suffix>/ "
                         "(default: write to the paths specified in geotop.inpts)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    progress = None if args.quiet else _Progress()
    try:
        run_simulation(args.working_dir, suffix=args.suffix,
                       verbose=not args.quiet, progress=progress)
    except (ValueError, FileNotFoundError, KeyError, NotImplementedError) as e:
        print(f"GEOtopPY: error: {e}", file=sys.stderr)
        return 1
    finally:
        if progress is not None:
            progress.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
