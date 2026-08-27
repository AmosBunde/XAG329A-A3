# Defines helper functions for the autograder.
#
# A3's student code lives in the xag329a_a3 package rather than in notebook
# cells, so the grader imports it directly and needs none of A2's notebook
# loading machinery. What remains here reads the output of a tagged eval cell,
# which is how notebook-reported accuracies get graded.

import json
import os
import re


def _at_sentence_start(text):
    # str.capitalize would lowercase the rest, turning "Part 3" into "part 3".
    return text[:1].upper() + text[1:]


def _find_tagged_cell(ipynb_path, tag):
    with open(ipynb_path) as f:
        ipynb = json.load(f)

    for cell in ipynb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", []) or []
        if tag in tags:
            return cell
    return None


def stdout_from_tagged_cell(ipynb_path, tag, description=None):
    """
    Return non-empty stdout lines from the code cell tagged `tag`.

    `description` names the cell the way a student would recognize it, since
    cell tags are invisible in the notebook UI and mean nothing to them.
    """
    error_file = os.path.basename(ipynb_path)
    where = description or f"the cell tagged '{tag}'"
    tg_cell = _find_tagged_cell(ipynb_path, tag)
    if tg_cell is None:
        raise ValueError(
            f"Could not find {where} in {error_file}. That cell carries a hidden tag "
            f"('{tag}') that the autograder needs, and deleting the cell or pasting "
            "its code into a new one removes the tag. Copy the cell back in from a "
            "fresh handout notebook, re-run it, and save."
        )

    if not tg_cell.get("outputs"):
        raise ValueError(
            f"{_at_sentence_start(where)} in {error_file} has no saved output, so there is "
            "nothing to grade. Run that cell, wait for it to finish, save the "
            "notebook, and submit the saved copy."
        )

    stdout = []
    for output in tg_cell["outputs"]:
        # Prefer stdout streams. Some notebook editors strip the `name` field
        # when saving; treat unnamed streams as stdout, but skip stderr.
        if output.get("output_type") != "stream":
            continue
        name = output.get("name")
        if name == "stderr":
            continue
        if name not in (None, "stdout"):
            continue
        text = output.get("text", "")
        if isinstance(text, list):
            stdout.extend(text)
        elif isinstance(text, str):
            stdout.extend(text.split("\n"))
        else:
            raise ValueError(f"Error processing stdout in {error_file} (tag={tag}).")

    return [x.strip() for x in stdout if len(x.strip()) > 0]


_ACCURACY_RE = re.compile(r"Accuracy\s*=\s*([0-9]*\.?[0-9]+)\s*%")


def accuracy_from_tagged_cell(ipynb_path, tag, description=None):
    """
    Return the accuracy, in percent, printed by the code cell tagged `tag`.

    The eval cells end with `Accuracy = 66.7%`, so the last such line is the
    summary for the whole run.
    """
    error_file = os.path.basename(ipynb_path)
    where = description or f"the cell tagged '{tag}'"
    lines = stdout_from_tagged_cell(ipynb_path, tag, description)

    found = _ACCURACY_RE.findall("\n".join(lines))
    if not found:
        raise ValueError(
            f"{_at_sentence_start(where)} in {error_file} ran, but its output has no "
            "'Accuracy = ...%' line. The autograder reads your score from that line, "
            "so keep the cell's final print statement as it was written, re-run the "
            "cell, and save the notebook."
        )
    return float(found[-1])


def json_from_tagged_cell(ipynb_path, tag):
    """
    Parse the JSON object printed by the code cell tagged `tag`.

    Scans stdout for valid JSON objects via JSONDecoder.raw_decode so braces
    in earlier LLM/LaTeX output do not poison extraction. Prefers a dict whose
    "section" matches `tag`; otherwise uses the last valid dict. If the chosen
    object has a "section" field, it must match `tag`.
    """
    error_file = os.path.basename(ipynb_path)
    lines = stdout_from_tagged_cell(ipynb_path, tag)
    text = "\n".join(lines)

    # Eval cells print noisy prediction traces before the summary JSON; a greedy
    # \{...\} match can start on LaTeX braces. Decode from each '{' instead.
    decoder = json.JSONDecoder()
    candidates = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            candidates.append(data)

    if not candidates:
        raise ValueError(
            f"No JSON object found in stdout for tagged cell '{tag}' in {error_file}. "
            "Make sure the cell prints a JSON summary via json.dumps(...)."
        )

    matching = [d for d in candidates if d.get("section") == tag]
    data = matching[-1] if matching else candidates[-1]
    if "section" in data and data["section"] != tag:
        raise ValueError(
            f"JSON section mismatch for tag '{tag}': got {data['section']!r}."
        )
    return data
