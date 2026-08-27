#!/usr/bin/env python3
"""Bundle the graded files for this assignment into one zip for Gradescope.

Your implementations live in the `xag329a_a3` package rather than in the
notebook, so Gradescope needs both the notebook and that folder. This builds an
archive holding exactly those, and flags anything that looks unfinished before
you upload.

Run it from the repo root:

    uv run python make_submission.py

Then upload the `a3_submission.zip` it writes.
"""

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ASSIGNMENT_ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ASSIGNMENT_ROOT / "src" / "submission"
NOTEBOOK = "assignment3.ipynb"
PACKAGE = "xag329a_a3"
# APIManager reads this from the directory above the package, so it travels with
# the submission to keep the upload runnable on its own.
EXTRA_FILES = ("weather_codes.json",)

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_NAMES = {".DS_Store"}

# The marker is assembled rather than written out, because the packaging step that
# builds this repo deletes everything between a literal STARTS and ENDS marker in
# any .py file. Spelling either one in full here would truncate this script.
_MARKER = "### TODO: YOUR CODE {} HERE"
TODO_BLOCK = re.compile(
    _MARKER.format("STARTS") + r"(.*?)" + _MARKER.format("ENDS"), re.DOTALL
)
ACCURACY = re.compile(r"Accuracy\s*=\s*([0-9]*\.?[0-9]+)\s*%")

# The autograder reads Part 3's score out of these two cells, found by tag.
GRADED_CELLS = {
    "zero_shot_eval": "the zero-shot evaluation in Part 3",
    "agentic_pipeline_eval": "the run_pipeline evaluation at the end of Part 3",
}
FULL_DATASET = 15

_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"


def _ok(message):
    print(f"{_GREEN}{message}{_RESET}")


def _warn(message):
    print(f"{_YELLOW}warning: {message}{_RESET}")


def _die(message):
    print(f"{_RED}error: {message}{_RESET}")
    raise SystemExit(1)


def _keep(path):
    if path.name in SKIP_NAMES or path.suffix in SKIP_SUFFIXES:
        return False
    return not any(part in SKIP_DIRS for part in path.parts)


def _collect():
    """Return the files to archive, or exit if a required one is missing."""
    notebook = SUBMISSION_DIR / NOTEBOOK
    package = SUBMISSION_DIR / PACKAGE

    if not notebook.is_file():
        _die(
            f"Could not find {NOTEBOOK} in {SUBMISSION_DIR}.\n"
            "Run this script from the repo root, and do not rename the notebook."
        )
    if not package.is_dir():
        _die(
            f"Could not find the {PACKAGE}/ folder in {SUBMISSION_DIR}.\n"
            "That folder holds your implementations, so it has to be submitted too."
        )

    paths = [notebook]
    paths += sorted(p for p in package.rglob("*") if p.is_file() and _keep(p))
    paths += [
        SUBMISSION_DIR / name
        for name in EXTRA_FILES
        if (SUBMISSION_DIR / name).is_file()
    ]
    return paths


def _check_todos(paths):
    """Warn about TODO blocks that are still empty."""
    unfinished = Counter()
    for path in paths:
        if path.suffix != ".py":
            continue
        for body in TODO_BLOCK.findall(path.read_text(encoding="utf-8")):
            if not body.strip():
                unfinished[path.relative_to(SUBMISSION_DIR).as_posix()] += 1

    if not unfinished:
        return
    total = sum(unfinished.values())
    _warn(f"{total} TODO block(s) are still empty and will score zero:")
    for name, count in sorted(unfinished.items()):
        print(f"    {name}: {count}")


def _stream_text(cell):
    text = ""
    for output in cell.get("outputs", []) or []:
        if output.get("output_type") == "stream":
            chunk = output.get("text", "")
            text += "".join(chunk) if isinstance(chunk, list) else chunk
    return text


def _check_eval_outputs(notebook_path):
    """Warn when a graded evaluation cell has nothing for the autograder to read.

    Part 3 is scored from the accuracy saved in the notebook rather than re-run,
    so a cell that was never run, or was run after the last save, scores zero.
    """
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _die(f"{NOTEBOOK} is not readable as notebook JSON: {exc}")

    tagged = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for tag in cell.get("metadata", {}).get("tags", []) or []:
            if tag in GRADED_CELLS:
                tagged[tag] = cell

    for tag, description in GRADED_CELLS.items():
        cell = tagged.get(tag)
        if cell is None:
            _warn(
                f"Could not find {description}. That cell carries a tag the autograder "
                "needs, and retyping the cell removes it. Copy the cell back in from a "
                "fresh copy of the notebook and run it."
            )
            continue

        text = _stream_text(cell)
        found = ACCURACY.findall(text)
        if not found:
            _warn(
                f"{description} has no accuracy in its saved output, so it will score "
                "zero. Run the cell, wait for it to finish, then save the notebook."
            )
            continue

        answered = text.count("Query:")
        if answered < FULL_DATASET:
            _warn(
                f"{description} reports {found[-1]}% but covers only {answered} of the "
                f"{FULL_DATASET} questions in the dataset. Run the cell that sets "
                "DEBUG_MODE = False first, then re-run the evaluation and save."
            )
        else:
            _ok(f"{description}: {found[-1]}% over {answered} questions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default=ASSIGNMENT_ROOT / "a3_submission.zip",
        type=Path,
        help="Where to write the archive (default: a3_submission.zip in the repo root).",
    )
    output = parser.parse_args().output

    paths = _collect()
    _check_todos(paths)
    _check_eval_outputs(SUBMISSION_DIR / NOTEBOOK)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(SUBMISSION_DIR).as_posix())

    size_kb = output.stat().st_size / 1024
    _ok(f"\nWrote {output.name} ({len(paths)} files, {size_kb:.0f} KB)")
    print(f"    {NOTEBOOK}")
    print(f"    {PACKAGE}/ ({sum(1 for p in paths if PACKAGE in p.parts)} files)")
    for name in EXTRA_FILES:
        if (SUBMISSION_DIR / name).is_file():
            print(f"    {name}")
    print(f"\nUpload {output} to Gradescope.")


if __name__ == "__main__":
    main()
