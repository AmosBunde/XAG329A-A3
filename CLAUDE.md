# CLAUDE.md for XAG329A-A3

Standing rules for all work in this repository. These rules apply to every session and every message unless the user explicitly overrides one in the message at hand.

## 1. Where code may be written

- Only write code between the `### TODO: YOUR CODE STARTS HERE` and `### TODO: YOUR CODE ENDS HERE` fences in these three files:
  - `src/submission/xag329a_a3/multi_lm_agent.py`
  - `src/submission/xag329a_a3/api_manager.py`
  - `src/submission/xag329a_a3/DeepResearchAgent.py`
- Never change function signatures.
- Never change code outside the fences.
- Never change the tests (`src/submission/xag329a_a3/tests/`), the grader (`src/grader.py`, `src/graderUtil.py`, `src/autograde_utils.py`), or `src/submission/xag329a_a3/evaluation.py`.

## 2. Secrets

- A `.env` at the repo root already holds `OPENROUTER_API_KEY`, `TAVILY_API_KEY`, `POLYGON_API_KEY` and `WOLFRAM_APP_ID`.
- Never print a key value, in a terminal, a log, a report, or anywhere else.
- Never copy a key into a notebook cell or any tracked file.
- Never commit `.env`. It must stay untracked.

## 3. External API calls

- Never run a notebook cell or script that calls OpenRouter or any other external API unless the user explicitly asks for it in that same message.
- The self-checks in `src/submission/xag329a_a3/tests/` and `src/grader.py` are offline. They patch out the model calls and may be run freely.

## 4. Read before writing

Before writing any code, read all three of these files and map every graded test to the TODO it exercises:

- `src/grader.py`
- `src/submission/xag329a_a3/tests/test.py`
- `src/submission/prompt_help.md`

## 5. Definition of done for code

- All nine self-checks in `src/submission/xag329a_a3/tests/` pass.
- `src/grader.py` reports 32/32.
- Report the actual numbers from the runs, not a summary.

## 6. Tooling

- Use `uv` for everything: `uv sync`, `uv run python ...`, `uv run pytest ...`.
- Do not `pip install` into the system interpreter.

## 7. Writing style

- No contractions and no em dashes in any text written for the user, including code comments and reports.
