# XAG329A3 – Self-Improving AI Agents – Part 3 Final Assignment

This assignment builds an end-to-end agentic workflow by combining LLMs with external APIs and self-improvement techniques.

## Overview

You will build an API-augmented LLM pipeline and then layer self-improvement methods on top of it:

0. LLM Performance with a Single Call (5 points)
1. API-augmented LLM pipeline (15 points)
2. Self-improvement techniques (20 points)
3. Building an LLM Agentic Workflow (10 points)

**Total: 50 points.** Part 4 (Deep Research Agent) in the notebook is optional and ungraded.

## Submission

Your graded code for this assignment lives in the `xag329a_a3` package, not in the
notebook, so Gradescope needs both. The easiest way to send them is to let the
helper script bundle them for you, from the repo root:

```bash
uv run python make_submission.py
```

That writes `a3_submission.zip` holding `assignment3.ipynb`, the `xag329a_a3/`
package, and `weather_codes.json`. Upload that single file to Gradescope.

The script also checks your work before zipping: it lists any TODO blocks you have
not filled in and any evaluation cell whose accuracy is missing from the saved
notebook. Both would cost you points, so it is worth running once even if you plan
to upload by hand.

To upload the files directly instead:

- Upload **`assignment3.ipynb`** and the **`xag329a_a3/` folder** (both from `src/submission/`).
- **Do not rename either one** — the notebook must be named exactly `assignment3.ipynb`,
  and the package folder exactly `xag329a_a3`.
- The autograder rejects a notebook-only upload, because your implementations of
  `route_query`, `decompose_and_fuse`, `iterative_refine`, and the rest are in the
  package files.

Either way, Part 3 is graded from the accuracy saved in your notebook rather than
re-run, so before you submit, run the two Part 3 evaluation cells — the zero-shot
one and the `run_pipeline` one at the end — and save the notebook afterwards. They
come after the cell that sets `DEBUG_MODE = False`, so they cover the whole dataset,
which is what Part 3 is scored on. A cell you never ran, or ran after your last save,
has no output for the autograder to read and scores zero. `make_submission.py` warns
you when that has happened.

## Checking your work as you go

Each part has a self-check cell in the notebook that exercises your code with a fake
API manager and canned model replies. They cost nothing to run and make no API calls,
so use them before spending credits on a full evaluation:

```python
from xag329a_a3.tests import test_api_routing  # or any other helper

test_api_routing()
```

They check that the pieces are wired together the way the rest of the pipeline
expects. They do not judge answer quality, and passing them is not the same as
passing the graded tests.

They are also how you reproduce a Gradescope failure without waiting on Gradescope:
each graded test names the helper that covers the same ground, so a failure on, say,
the API routing test tells you to run `test_api_routing()` and read the same problem
described locally.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.
Think of `uv` as a fast replacement for `pip` + `venv`: one tool creates a virtual environment, installs packages from `pyproject.toml` / `uv.lock`, and runs commands inside that environment.

We only support `uv` for this assignment (not Conda).

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS, you can also use Homebrew:

```bash
brew install uv
```

After installing, restart your terminal (or open a new one), then confirm it works:

```bash
uv --version
```

### 2. Install the project dependencies

From the root of this repository (the folder that contains `pyproject.toml`), run:

```bash
uv sync
```

This creates a local `.venv/` folder and installs everything listed in `pyproject.toml` (locked by `uv.lock`).
You usually do **not** need to activate the environment yourself — use `uv run ...` (or select the `.venv` kernel in your editor).

### 3. Set up API keys

This assignment uses four APIs, and you need to sign up for each of them. **Only OpenRouter is paid** — the other three have free tiers that are sufficient here. The weather tool needs no key at all, because [Open-Meteo](https://open-meteo.com/) is keyless.

1. **[OpenRouter](https://openrouter.ai/) API** (`OPENROUTER_API_KEY`) — LLM inference for every model in this assignment. **Paid**: add credits before running.
2. **[Tavily](https://www.tavily.com/) API** (`TAVILY_API_KEY`) — web search. The "Researcher" plan is free with limited usage.
3. **[Polygon (Massive)](https://massive.com/) API** (`POLYGON_API_KEY`) — financial / stock data. The "Stock Basic" plan is free with limited usage.
4. **[Wolfram Alpha](https://developer.wolframalpha.com/) API** (`WOLFRAM_APP_ID`) — mathematical calculations. The free APP ID plan has limited usage.

[OpenRouter](https://openrouter.ai/) is a unified API gateway that routes requests to many model providers through one interface and API key. We use it in this assignment so students can access both open-source and closed (proprietary) models without configuring each provider separately. For this assignment, however, we will only use specific open-source models (see [Models and roles](#models-and-roles) and [Useful links](#useful-links)).

Once you have all four, add them to your environment:

1. Copy `.env.example` to `.env`
2. Edit `.env` and fill in the four variables listed above. Step-by-step signup instructions for each API are in [Additional information](#additional-information).
3. The notebook loads these with:

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

`find_dotenv()` searches upward from the notebook, so keeping `.env` at the repository root works even though the notebook lives in `src/submission/`.

### 4. Open the notebook (recommended: VS Code)

The easiest way to run the assignment locally is in **[VS Code](https://code.visualstudio.com/)**:

1. Install the **[Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)** extension (search "Jupyter" by Microsoft in Extensions)
2. Open this folder in VS Code (the repo root that contains `pyproject.toml`)
3. Open `src/submission/assignment3.ipynb`
4. Select the kernel / interpreter created by `uv`:
   - Choose **Python Environments** → the `.venv` in this project
   - Or run the kernel registration command below and pick `xag329a-a3`
5. Run cells as usual

See also: [VS Code Jupyter notebooks docs](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

### Alternative: Jupyter in the browser

If you prefer classic [Jupyter Notebook](https://jupyter.org/), from the repo root:

```bash
uv run jupyter notebook src/submission/assignment3.ipynb
```

`uv run` runs the command inside the project environment, so imports like `from xag329a_a3.api_manager import APIManager` should work.

See also: [Jupyter Notebook documentation](https://docs.jupyter.org/en/latest/).

### API test notebooks

The `src/submission/eval-search-apis/` folder has one small notebook per API. Each calls `load_dotenv()`, reads its own variable from `.env`, and makes a live request, so you can check a key in isolation. If a notebook fails, the problem is the key or the account rather than your assignment code.

- [`tavily.ipynb`](src/submission/eval-search-apis/tavily.ipynb) — Tavily web search (`TAVILY_API_KEY`)
- [`polygon.ipynb`](src/submission/eval-search-apis/polygon.ipynb) — Polygon / Massive stock data (`POLYGON_API_KEY`)
- [`wolframalpha.ipynb`](src/submission/eval-search-apis/wolframalpha.ipynb) — Wolfram Alpha computation (`WOLFRAM_APP_ID`)
- [`open-metro.ipynb`](src/submission/eval-search-apis/open-metro.ipynb) — Open-Meteo weather and Nominatim geocoding (no key required)

Run one with:

```bash
uv run jupyter notebook src/submission/eval-search-apis/tavily.ipynb
```

These notebooks also document each provider's free-tier plan and where to find your key, and they are the quickest way to confirm your `.env` is set up correctly.

### Troubleshooting

**Kernel / wrong environment**: If cells fail with `ModuleNotFoundError`, make sure the notebook kernel is the project `.venv` (or `xag329a-a3`), then re-run:

```bash
uv sync
uv run python -m ipykernel install --user --name xag329a-a3 --display-name "xag329a-a3"
```

**Package import errors**: From the repo root:

```bash
uv sync
```

**Missing API tokens**: If you see an error about `OPENROUTER_API_KEY`, `TAVILY_API_KEY`, `POLYGON_API_KEY`, or `WOLFRAM_APP_ID`, confirm `.env` exists (copied from `.env.example`), the variable names match exactly, and you ran `load_dotenv(find_dotenv())` in the notebook. To isolate which key is at fault, run the matching notebook from [API test notebooks](#api-test-notebooks).

**`weather_codes.json` not found**: The `get_weather` tool reads `weather_codes.json` from the directory that contains the `xag329a_a3/` package, so keep the file at `src/submission/weather_codes.json`. If you run `open-metro.ipynb` from the `eval-search-apis/` folder, it reads `../weather_codes.json`.

## Project structure

```
.
├── pyproject.toml              # dependencies / packaging (install from repo root)
├── uv.lock                     # locked dependency versions (used by `uv sync`)
├── .python-version             # default Python for `uv` (3.11)
├── .env.example                # copy to `.env` and add your four API keys
├── make_submission.py          # bundles your notebook + package into a zip for Gradescope
└── src/submission/
    ├── assignment3.ipynb       # main assignment notebook
    ├── weather_codes.json      # WMO weather-code lookup, read by the get_weather tool
    ├── eval-search-apis/       # one small notebook per API, to verify your keys
    └── xag329a_a3/             # Python package (api_manager, multi_lm_agent, utils, evaluation, wolfram_api)
```

After `uv sync`, import the package as `xag329a_a3` (not via a relative path into `src/submission`).

## Usage

```python
import os
from xag329a_a3.api_manager import APIManager
from xag329a_a3.multi_lm_agent import MultiLMAgent
from xag329a_a3.evaluation import prepare_dataset, evaluate_qa

api_manager = APIManager(
    tavily_api_key=os.environ["TAVILY_API_KEY"],
    polygon_api_key=os.environ["POLYGON_API_KEY"],
    wolfram_app_id=os.environ["WOLFRAM_APP_ID"],
)

agent = MultiLMAgent(api_manager)
dataset = prepare_dataset(question_type="math", debug_mode=True)
```

## Models and roles

Use the LiteLLM model IDs below (the `openrouter/` prefix is required). Roles match the notebook constants.

- `openrouter/meta-llama/llama-3.1-8b-instruct` — cheap default for zero-shot `generate`, API routing, and Part 4 baseline; also a fusion sample model and the Part 3 fusion model. [meta-llama/llama-3.1-8b-instruct](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct)
- `openrouter/meta-llama/llama-3.3-70b-instruct` — stronger single-API / decomposition / iterative refinement, including Part 3. [meta-llama/llama-3.3-70b-instruct](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct)
- `openrouter/openai/gpt-oss-20b` — LLM-as-judge (`evaluate_qa`); also the second fusion sample model. [openai/gpt-oss-20b](https://openrouter.ai/openai/gpt-oss-20b)
- `openrouter/google/gemma-3-4b-it` — optional cheaper stand-in while developing. [google/gemma-3-4b-it](https://openrouter.ai/google/gemma-3-4b-it)

The two fusion sample models are the default for `MultiLMAgent(fusion_sample_models=...)`, matching the notebook's `FUSION_SAMPLE_MODELS`.

## Features

- **Four external tools**: Tavily web search, Polygon (Massive) stock data, Wolfram Alpha computation, and keyless Open-Meteo weather
- **LLM-based API routing**: structured-output parsing selects the tool and extracts its parameters from a natural-language query
- **Self-improvement methods**: query decomposition, multi-model fusion, and iterative self-refinement
- **Agentic workflow**: the full pipeline combined and evaluated end to end
- **LLM-as-judge evaluation**: `evaluate_qa` grades free-form answers against reference values

## Requirements

- Python 3.10–3.12 (default: 3.11 via `.python-version`; 3.13 is unsupported)
- OpenRouter API key (paid)
- Tavily API key, Polygon (Massive) API key, Wolfram Alpha app ID (all free tiers)
- Required packages listed in `pyproject.toml` (installed via `uv sync`)

## Additional information

### Steps to create an OpenRouter API key

1. **Create an account / sign in**
   - Go to [openrouter.ai](https://openrouter.ai/) and sign up or log in. ([docs](https://openrouter.ai/docs))

2. **Open API Keys**
   - Go to [openrouter.ai/keys](https://openrouter.ai/keys)

3. **Create a new key**
   - Click **Create Key**, give it a name (e.g., `xag329a`), and copy the key immediately

4. **Add credits**
   - OpenRouter is a paid API gateway; add credits in your account billing settings so requests can succeed

5. **Use the key in your app**
   - Put it in `.env` as `OPENROUTER_API_KEY=...`, or export it in your shell:
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```
   ([OpenRouter docs](https://openrouter.ai/docs/quickstart))

### Steps to get a free Tavily API key

Tavily is the web search API used by the `tavily_search` tool.

1. **Create a free account**
   - Go to [tavily.com](https://www.tavily.com/) and sign up. The **"Researcher"** plan is free with limited usage.

2. **Open your API keys**
   - Go to [app.tavily.com/home](https://app.tavily.com/home)

3. **Copy the key and add it to `.env`**
   - Put it in `.env` as `TAVILY_API_KEY=...`

4. **Verify it works**
   - Run [`src/submission/eval-search-apis/tavily.ipynb`](src/submission/eval-search-apis/tavily.ipynb)

See also: [Tavily Python SDK reference](https://docs.tavily.com/sdk/python/reference).

### Steps to get a free Polygon (Massive) API key

Polygon, now at massive.com, is the financial data API used by the `get_stock_data` tool.

1. **Create a free account**
   - Go to [massive.com](https://massive.com/) and sign up. The free **"Stock Basic"** plan is enough for this assignment.

2. **Open your API keys**
   - Once logged in, go to [polygon.io/dashboard/keys](https://polygon.io/dashboard/keys)

3. **Copy the key and add it to `.env`**
   - Put it in `.env` as `POLYGON_API_KEY=...`

4. **Verify it works**
   - Run [`src/submission/eval-search-apis/polygon.ipynb`](src/submission/eval-search-apis/polygon.ipynb)

Note: the Python client is imported as `from massive import RESTClient` following the rename.

### Steps to get a free Wolfram Alpha app ID

Wolfram Alpha is the computation API used by the `compute` tool.

1. **Create a free account**
   - Go to [developer.wolframalpha.com](https://developer.wolframalpha.com/) and sign up. The free APP ID plan has limited usage.

2. **Create an AppID**
   - Go to [developer.wolframalpha.com/access](https://developer.wolframalpha.com/access) and create an AppID. This assignment calls the Full Results API (`/v2/query`).

3. **Copy the AppID and add it to `.env`**
   - Put it in `.env` as `WOLFRAM_APP_ID=...`. Note this is an **app ID** rather than a bearer token, so the variable name ends in `_APP_ID`.

4. **Verify it works**
   - Run [`src/submission/eval-search-apis/wolframalpha.ipynb`](src/submission/eval-search-apis/wolframalpha.ipynb)

### Weather: no API key needed

The `get_weather` tool needs no signup. It geocodes the location with [Nominatim](https://nominatim.org/) (OpenStreetMap) and then reads historical weather from the [Open-Meteo](https://open-meteo.com/) archive API, both of which are keyless. See [`src/submission/eval-search-apis/open-metro.ipynb`](src/submission/eval-search-apis/open-metro.ipynb).

### Useful links

- [Jupyter Notebook](https://jupyter.org/) / [docs](https://docs.jupyter.org/en/latest/)
- [VS Code](https://code.visualstudio.com/) / [Jupyter in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [OpenRouter](https://openrouter.ai/) / [API keys](https://openrouter.ai/keys) / [Docs](https://openrouter.ai/docs) / [LiteLLM OpenRouter provider](https://docs.litellm.ai/docs/providers/openrouter)
- [Tavily](https://www.tavily.com/) / [API keys](https://app.tavily.com/home) / [Python SDK reference](https://docs.tavily.com/sdk/python/reference)
- [Polygon / Massive](https://massive.com/) / [API keys](https://polygon.io/dashboard/keys)
- [Wolfram Alpha](https://developer.wolframalpha.com/) / [AppID access](https://developer.wolframalpha.com/access)
- [Open-Meteo](https://open-meteo.com/) (weather, no key) / [Nominatim](https://nominatim.org/) (geocoding, no key)
