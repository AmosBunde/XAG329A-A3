"""Self-check helpers for the assignment TODOs.

These use a fake API manager and canned model replies, so you can check the
wiring of each part without spending API credits or waiting on the network.
They check plumbing, not answer quality: passing here means the pieces are
connected the way the rest of the pipeline expects, not that your prompts are
good. The graded tests are separate.

Run them from the notebook cell that follows each part, or directly:

```python
from xag329a_a3.tests import test_generate

test_generate()
```
"""

import sys
import traceback
from contextlib import contextmanager
from functools import wraps

_GREEN = "\033[92m"  # bright green
_RED = "\033[91m"
_RESET = "\033[0m"


def _print_pass(message: str) -> None:
    print(f"{_GREEN}{message}{_RESET}")


def _print_fail(message: str) -> None:
    print(f"{_RED}{message}{_RESET}")


def _report(label: str, errcount: int) -> None:
    if errcount == 0:
        _print_pass(f"No errors detected for `{label}`")
    else:
        _print_fail(f"{errcount} error(s) detected for `{label}`")


@contextmanager
def _patched(module, name, replacement):
    """Swap a module-level function, restoring it even if a check raises.

    Without the restore, one failed check would leave the package monkeypatched
    for the rest of the notebook session.
    """
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


def _guarded(func):
    """Turn an unexpected exception into a readable failure line.

    An unfinished TODO usually surfaces as a NameError or TypeError from inside
    the student's own code; reporting it with its location is more useful here
    than letting a traceback replace the rest of the checks.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            frames = traceback.extract_tb(sys.exc_info()[2])
            where = ""
            if frames:
                last = frames[-1]
                where = f" (raised at {last.filename.split('/')[-1]}:{last.lineno})"
            _print_fail(
                f"Error for `{func.__name__}`: {type(exc).__name__}: {exc}{where}"
            )
            _print_fail(
                f"Stopped `{func.__name__}` early; fix the error above and re-run."
            )

    return wrapper


####################
# Test fixtures    #
####################


class FakeAPIManager:
    """Stands in for `APIManager` so an agent can be built without API keys.

    `MultiLMAgent` only reads `api_functions` off the manager and calls
    `route_query`, so nothing here touches the network.
    """

    def __init__(self, result=None):
        self.api_functions = {
            "tavily_search": (lambda **kw: [], None),
            "get_stock_data": (lambda **kw: {}, None),
            "compute": (lambda **kw: {}, None),
            "get_weather": (lambda **kw: {}, None),
        }
        self.router_model = "fake/router-model"
        self.queries = []
        self._result = result or {
            "results": "Paris was 21C at noon",
            "api_used": "get_weather",
            "params": {"location": "Paris, France", "date": "2025-08-27"},
        }

    def route_query(self, query, model=None):
        self.queries.append(query)
        if callable(self._result):
            return self._result(query)
        return self._result


class ScriptedGenerate:
    """Stands in for `agent.generate`, handing back canned replies.

    Replies come from `replies` in order, or from `reply(prompt, model)` when a
    check needs the answer to depend on what has happened so far. Records every
    call so a check can assert which model was asked and what prompt it got.
    """

    def __init__(self, replies=(), reply=None):
        self._replies = list(replies)
        self._reply = reply
        self.calls = []

    def __call__(self, query, model=None, temperature=None):
        self.calls.append({"prompt": query, "model": model, "temperature": temperature})
        if self._reply is not None:
            return self._reply(query, model)
        if not self._replies:
            return "no scripted reply left"
        return self._replies.pop(0)

    @property
    def models(self):
        return [call["model"] for call in self.calls]

    @property
    def prompts(self):
        return [call["prompt"] for call in self.calls]


_REQUIRED_AGENT_ATTRS = (
    "api_manager",
    "decomposition_model",
    "iterative_refinement_model",
    "fusion_model",
    "fusion_sample_models",
    "generation_temp",
)


def _make_agent(**kwargs):
    """Build a `MultiLMAgent` wired to a `FakeAPIManager`."""
    from xag329a_a3.multi_lm_agent import MultiLMAgent

    manager = FakeAPIManager(result=kwargs.pop("route_result", None))
    return MultiLMAgent(manager, **kwargs), manager


def _not_ready(agent, label):
    """Report an incomplete `__init__` instead of failing later and elsewhere."""
    missing = [name for name in _REQUIRED_AGENT_ATTRS if not hasattr(agent, name)]
    if missing:
        _print_fail(
            f"Cannot run `{label}`: `MultiLMAgent.__init__` never set "
            f"{', '.join(missing)}. Finish the Part 0 TODO first."
        )
        return True
    return False


def _bare_api_manager():
    """Build an `APIManager` with only what routing touches.

    `route_query` and `_parse_query_params` use `router_model` and
    `api_functions` and nothing else, so `__init__` is skipped entirely: no
    Tavily, Massive, Wolfram, or geocoder client is constructed and no API key
    is needed.
    """
    from xag329a_a3.api_manager import (
        APIManager,
        ComputeParams,
        GetWeatherParams,
        StockDataParams,
        TavilySearchParams,
    )

    manager = APIManager.__new__(APIManager)
    manager.router_model = "fake/router-model"
    invoked = []

    def make_tool(name):
        def tool(**kwargs):
            invoked.append((name, kwargs))
            return {"value": f"{name} ran"}

        tool.__doc__ = f"Fake {name} tool used by the self-checks."
        tool.__name__ = name
        return tool

    manager.api_functions = {
        "tavily_search": (make_tool("tavily_search"), TavilySearchParams),
        "get_stock_data": (make_tool("get_stock_data"), StockDataParams),
        "compute": (make_tool("compute"), ComputeParams),
        "get_weather": (make_tool("get_weather"), GetWeatherParams),
    }
    return manager, invoked


####################
# Part 0           #
####################


@_guarded
def test_agent_init():
    """Check that `__init__` stores the models and manager it is handed."""
    from xag329a_a3.multi_lm_agent import MultiLMAgent

    manager = FakeAPIManager()
    agent = MultiLMAgent(
        manager,
        decomposition_model="fake/decomposition",
        iterative_refinement_model="fake/refinement",
        fusion_model="fake/fusion",
        fusion_sample_models=("fake/sample-a", "fake/sample-b"),
        generation_temp=0.35,
    )

    expected = {
        "api_manager": manager,
        "decomposition_model": "fake/decomposition",
        "iterative_refinement_model": "fake/refinement",
        "fusion_model": "fake/fusion",
        "generation_temp": 0.35,
    }
    errcount = 0
    for name, want in expected.items():
        if not hasattr(agent, name):
            errcount += 1
            _print_fail(
                f"Error for `MultiLMAgent.__init__`: `self.{name}` was never set."
            )
        elif getattr(agent, name) != want:
            errcount += 1
            _print_fail(
                f"Error for `MultiLMAgent.__init__`: expected `self.{name}` "
                f"to be {want!r}, got {getattr(agent, name)!r}."
            )

    if not hasattr(agent, "fusion_sample_models"):
        errcount += 1
        _print_fail(
            "Error for `MultiLMAgent.__init__`: `self.fusion_sample_models` "
            "was never set."
        )
    elif list(agent.fusion_sample_models) != ["fake/sample-a", "fake/sample-b"]:
        errcount += 1
        _print_fail(
            "Error for `MultiLMAgent.__init__`: expected "
            "`self.fusion_sample_models` to keep the two models passed in, got "
            f"{agent.fusion_sample_models!r}."
        )

    _report("MultiLMAgent.__init__", errcount)


@_guarded
def test_generate():
    """Check that `generate` forwards model / temperature and returns the reply."""
    from xag329a_a3 import multi_lm_agent as module

    agent, _ = _make_agent(generation_temp=0.55)
    if _not_ready(agent, "MultiLMAgent.generate"):
        return

    recorded = {}

    def fake_generate_llm(model, messages, temperature=0.7, *args, **kwargs):
        recorded["model"] = model
        recorded["messages"] = messages
        recorded["temperature"] = temperature
        return "FAKE REPLY"

    errcount = 0
    with _patched(module, "generate_llm", fake_generate_llm):
        result = agent.generate("What is 40 + 2?", model="fake/model-x")

        if result != "FAKE REPLY":
            errcount += 1
            _print_fail(
                "Error for `generate`: expected the text returned by "
                f"`generate_llm` ('FAKE REPLY'), got {result!r}."
            )
        if recorded.get("model") != "fake/model-x":
            errcount += 1
            _print_fail(
                "Error for `generate`: expected the `model` argument to be "
                f"passed through to `generate_llm`, got {recorded.get('model')!r}."
            )
        if recorded.get("temperature") != 0.55:
            errcount += 1
            _print_fail(
                "Error for `generate`: when `temperature` is None it should "
                "fall back to `self.generation_temp` (0.55), got "
                f"{recorded.get('temperature')!r}."
            )
        messages = recorded.get("messages") or []
        joined = " ".join(
            str(m.get("content", "")) for m in messages if isinstance(m, dict)
        )
        if "What is 40 + 2?" not in joined:
            errcount += 1
            _print_fail(
                "Error for `generate`: the query should appear in the messages "
                f"sent to `generate_llm`, got {messages!r}."
            )

    # `generate_llm` returns None when its retries are exhausted; `generate`
    # has to pass that None along rather than inventing an answer.
    with _patched(module, "generate_llm", lambda *a, **k: None):
        failed = agent.generate("anything", model="fake/model-x")
        if failed is not None:
            errcount += 1
            _print_fail(
                "Error for `generate`: expected None when `generate_llm` "
                f"fails, got {failed!r}."
            )

    _report("generate", errcount)


####################
# Part 1           #
####################


@_guarded
def test_api_routing():
    """Check that routing picks a tool, parses its params, and calls it."""
    from xag329a_a3 import api_manager as module
    from xag329a_a3.api_manager import APIResponse, ComputeParams

    manager, invoked = _bare_api_manager()
    structured_calls = []

    def fake_generate_structured(model, messages, schema_model, temperature=0.0):
        structured_calls.append(schema_model)
        if schema_model is APIResponse:
            return APIResponse(api_name="compute")
        if schema_model is ComputeParams:
            return ComputeParams(wolfram_query="40 + 2")
        return None

    errcount = 0
    with _patched(module, "generate_structured", fake_generate_structured):
        params = manager._parse_query_params("What is 40 + 2?", "compute")
        if params != {"wolfram_query": "40 + 2"}:
            errcount += 1
            _print_fail(
                "Error for `_parse_query_params`: expected the validated "
                "params as a plain dict {'wolfram_query': '40 + 2'}, got "
                f"{params!r}. Remember to return `.model_dump()` rather than "
                "the Pydantic object."
            )

        structured_calls.clear()
        invoked.clear()
        result = manager.route_query("What is 40 + 2?")

        if APIResponse not in structured_calls:
            errcount += 1
            _print_fail(
                "Error for `route_query`: the tool choice should come from a "
                "`generate_structured` call using the `APIResponse` schema; "
                f"schemas requested were {structured_calls!r}."
            )
        if result.get("api_used") != "compute":
            errcount += 1
            _print_fail(
                "Error for `route_query`: expected `api_used` to be the tool "
                f"the router chose ('compute'), got {result.get('api_used')!r}."
            )
        if result.get("params") != {"wolfram_query": "40 + 2"}:
            errcount += 1
            _print_fail(
                "Error for `route_query`: expected `params` to be the parsed "
                f"arguments, got {result.get('params')!r}."
            )
        if result.get("results") != {"value": "compute ran"}:
            errcount += 1
            _print_fail(
                "Error for `route_query`: expected `results` to hold the tool "
                f"output, got {result.get('results')!r}."
            )
        if invoked != [("compute", {"wolfram_query": "40 + 2"})]:
            errcount += 1
            _print_fail(
                "Error for `route_query`: expected the chosen tool to be "
                f"called once with the parsed params, got {invoked!r}."
            )

    # A router that names a tool which does not exist must be reported, not
    # called; the same goes for a routing call that comes back empty.
    with _patched(
        module,
        "generate_structured",
        lambda model, messages, schema_model, temperature=0.0: (
            APIResponse(api_name="not_a_tool") if schema_model is APIResponse else None
        ),
    ):
        unknown = manager.route_query("What is 40 + 2?")
        if "error" not in unknown:
            errcount += 1
            _print_fail(
                "Error for `route_query`: expected an 'error' key when the "
                f"router names an unknown tool, got {unknown!r}."
            )

    with _patched(
        module, "generate_structured", lambda *a, **k: None
    ):
        failed = manager.route_query("What is 40 + 2?")
        if "error" not in failed:
            errcount += 1
            _print_fail(
                "Error for `route_query`: expected an 'error' key when the "
                f"routing call fails, got {failed!r}."
            )

    _report("route_query", errcount)


@_guarded
def test_single_lm_with_single_api_call():
    """Check the one-tool-call, one-LM-call pipeline."""
    agent, manager = _make_agent()
    if _not_ready(agent, "single_LM_with_single_API_call"):
        return

    scripted = ScriptedGenerate(["Paris was 21C. Final answer: 21C"])
    agent.generate = scripted

    errcount = 0
    result = agent.single_LM_with_single_API_call(
        "How warm was Paris on 2025-08-27?", "fake/model-x"
    )

    if manager.queries != ["How warm was Paris on 2025-08-27?"]:
        errcount += 1
        _print_fail(
            "Error for `single_LM_with_single_API_call`: expected the original "
            f"query to be routed once, got {manager.queries!r}."
        )
    if not scripted.calls:
        errcount += 1
        _print_fail(
            "Error for `single_LM_with_single_API_call`: expected one call to "
            "`self.generate` after routing, got none."
        )
    else:
        prompt = scripted.prompts[0]
        if "How warm was Paris on 2025-08-27?" not in prompt:
            errcount += 1
            _print_fail(
                "Error for `single_LM_with_single_API_call`: the prompt must "
                "include the original query, since the model never saw it."
            )
        if "Paris was 21C at noon" not in prompt:
            errcount += 1
            _print_fail(
                "Error for `single_LM_with_single_API_call`: the prompt must "
                "include the tool output, since the model cannot see the API "
                "result otherwise."
            )
    if result != "Paris was 21C. Final answer: 21C":
        errcount += 1
        _print_fail(
            "Error for `single_LM_with_single_API_call`: expected the model's "
            f"answer to be returned, got {result!r}."
        )

    # Tools fail in normal operation, a rate limit or an unknown location for
    # instance. Reporting that, or asking the model to answer without the tool,
    # are both fine; crashing is not, since one bad question would end a whole
    # evaluation run.
    error_agent, _ = _make_agent(
        route_result={
            "results": None,
            "api_used": "get_weather",
            "params": {"location": "Atlantis"},
            "error": "Location not found",
        }
    )
    error_agent.generate = ScriptedGenerate(["I could not find that."])
    error_result = error_agent.single_LM_with_single_API_call("anything", "fake/model")
    if not isinstance(error_result, str) or not error_result.strip():
        errcount += 1
        _print_fail(
            "Error for `single_LM_with_single_API_call`: when the API call fails "
            "this should still return a non-empty string, since `evaluate_qa` "
            f"grades whatever comes back, but it returned {error_result!r}."
        )

    _report("single_LM_with_single_API_call", errcount)


####################
# Part 2           #
####################


@_guarded
def test_query_decomposition():
    """Check the decomposition prompt and the sub-query call."""
    agent, _ = _make_agent(decomposition_model="fake/decomposition")
    if _not_ready(agent, "_get_sub_queries"):
        return

    errcount = 0

    prompt = agent._get_query_decomposition_prompt("How warm was Paris and Tokyo?")
    if not isinstance(prompt, str) or not prompt.strip():
        errcount += 1
        _print_fail(
            "Error for `_get_query_decomposition_prompt`: expected a non-empty "
            f"prompt string, got {prompt!r}."
        )
    else:
        if "How warm was Paris and Tokyo?" not in prompt:
            errcount += 1
            _print_fail(
                "Error for `_get_query_decomposition_prompt`: the user query "
                "has to appear in the prompt."
            )
        if "<sub-query>" not in prompt:
            errcount += 1
            _print_fail(
                "Error for `_get_query_decomposition_prompt`: the prompt has "
                "to ask for <sub-query>...</sub-query> tags, which is what "
                "`_get_sub_queries` parses."
            )

    scripted = ScriptedGenerate(
        [
            "Here you go:\n"
            "<sub-query>weather in Paris on 2025-08-27</sub-query>\n"
            "<sub-query>weather in Tokyo on 2025-08-27</sub-query>\n"
            "<sub-query>weather in Lima on 2025-08-27</sub-query>"
        ]
    )
    agent.generate = scripted
    sub_queries = agent._get_sub_queries("How warm was Paris and Tokyo?")

    expected = [
        "weather in Paris on 2025-08-27",
        "weather in Tokyo on 2025-08-27",
        "weather in Lima on 2025-08-27",
    ]
    if sub_queries != expected:
        errcount += 1
        _print_fail(
            "Error for `_get_sub_queries`: expected the tagged sub-queries "
            f"{expected!r}, got {sub_queries!r}."
        )
    if scripted.models and scripted.models[0] != "fake/decomposition":
        errcount += 1
        _print_fail(
            "Error for `_get_sub_queries`: decomposition should use "
            "`self.decomposition_model` rather than `generate`'s default, got "
            f"{scripted.models[0]!r}."
        )
    if scripted.prompts and "<sub-query>" not in scripted.prompts[0]:
        errcount += 1
        _print_fail(
            "Error for `_get_sub_queries`: the prompt sent to the model should "
            "be the one built by `_get_query_decomposition_prompt`."
        )

    # The cap keeps a chatty model from fanning out into unbounded API calls.
    agent.generate = ScriptedGenerate(
        [
            "<sub-query>one</sub-query><sub-query>two</sub-query>"
            "<sub-query>three</sub-query>"
        ]
    )
    capped = agent._get_sub_queries("anything", max_sub_queries=2)
    if capped != ["one", "two"]:
        errcount += 1
        _print_fail(
            "Error for `_get_sub_queries`: expected `max_sub_queries=2` to cap "
            f"the list at ['one', 'two'], got {capped!r}."
        )

    agent.generate = ScriptedGenerate([None])
    empty = agent._get_sub_queries("anything")
    if empty != []:
        errcount += 1
        _print_fail(
            "Error for `_get_sub_queries`: expected [] when the decomposition "
            f"call fails, got {empty!r}."
        )

    _report("_get_sub_queries", errcount)


@_guarded
def test_synthesis_prompt():
    """Check that the synthesis prompt carries the evidence and drops errors."""
    agent, _ = _make_agent()
    if _not_ready(agent, "_get_synthesis_prompt"):
        return

    decomposed = [
        {
            "sub_query": "weather in Paris",
            "api": "get_weather",
            "params": {"location": "Paris, France"},
            "results": "21C and clear",
        },
        {
            "sub_query": "stock price of TSLA",
            "api": "get_stock_data",
            "params": {"ticker": "TSLA"},
            "results": "stale quote nobody should trust",
            "error": "massive rate limited the request",
        },
    ]

    errcount = 0
    prompt = agent._get_synthesis_prompt("What was the weather in Paris?", decomposed)

    if not isinstance(prompt, str) or not prompt.strip():
        errcount += 1
        _print_fail(
            "Error for `_get_synthesis_prompt`: expected a non-empty prompt "
            f"string, got {prompt!r}."
        )
    else:
        if "What was the weather in Paris?" not in prompt:
            errcount += 1
            _print_fail(
                "Error for `_get_synthesis_prompt`: the original query has to "
                "appear in the prompt."
            )
        if "21C and clear" not in prompt:
            errcount += 1
            _print_fail(
                "Error for `_get_synthesis_prompt`: the results of successful "
                "sub-queries have to appear in the prompt."
            )
        # The failed step is identified by its own marker text rather than by
        # its error message, which a correct implementation never copies into
        # the prompt either way.
        leaked = [
            marker
            for marker in ("stale quote nobody should trust", "get_stock_data")
            if marker in prompt
        ]
        if leaked:
            errcount += 1
            _print_fail(
                "Error for `_get_synthesis_prompt`: steps with an 'error' key "
                "have nothing to contribute and should be left out, but "
                f"{leaked!r} still appears in the prompt."
            )

    # Decomposition can come back empty, and this still has to produce a
    # sensible prompt rather than a bare fragment.
    empty_prompt = agent._get_synthesis_prompt("What was the weather in Paris?", [])
    if not isinstance(empty_prompt, str) or "What was the weather in Paris?" not in (
        empty_prompt or ""
    ):
        errcount += 1
        _print_fail(
            "Error for `_get_synthesis_prompt`: with no sub-query results the "
            "prompt should still ask the original query, got "
            f"{empty_prompt!r}."
        )

    _report("_get_synthesis_prompt", errcount)


@_guarded
def test_decompose_and_fuse():
    """Check that sampling across models happens, then a fusion call."""
    agent, manager = _make_agent(
        decomposition_model="fake/decomposition",
        fusion_model="fake/fusion",
        fusion_sample_models=("fake/sample-a", "fake/sample-b"),
    )
    if _not_ready(agent, "decompose_and_fuse"):
        return

    scripted = ScriptedGenerate(
        [
            "<sub-query>weather in Paris</sub-query>",
            "SAMPLE-ONE says 21C",
            "SAMPLE-TWO says 21C",
            "FUSED ANSWER: 21C",
        ]
    )
    agent.generate = scripted

    errcount = 0
    result = agent.decompose_and_fuse("How warm was Paris?")

    if len(scripted.calls) < 4:
        errcount += 1
        _print_fail(
            "Error for `decompose_and_fuse`: expected at least 4 generate "
            "calls (decompose, one per fusion sample model, then fuse), got "
            f"{len(scripted.calls)}."
        )
    else:
        sampled = set(scripted.models[1:3])
        if sampled != {"fake/sample-a", "fake/sample-b"}:
            errcount += 1
            _print_fail(
                "Error for `decompose_and_fuse`: the synthesis prompt should "
                "go to every model in `self.fusion_sample_models`; models "
                f"asked were {sampled!r}."
            )
        if scripted.models[-1] != "fake/fusion":
            errcount += 1
            _print_fail(
                "Error for `decompose_and_fuse`: the final fusion call should "
                f"use `self.fusion_model`, got {scripted.models[-1]!r}."
            )
        fusion_prompt = scripted.prompts[-1]
        for sample in ("SAMPLE-ONE says 21C", "SAMPLE-TWO says 21C"):
            if sample not in fusion_prompt:
                errcount += 1
                _print_fail(
                    "Error for `decompose_and_fuse`: the fusion prompt has to "
                    f"show the sampled answers; {sample!r} is missing."
                )
        if "How warm was Paris?" not in fusion_prompt:
            errcount += 1
            _print_fail(
                "Error for `decompose_and_fuse`: the fusion prompt has to "
                "include the original query alongside the sampled answers."
            )

    if manager.queries != ["weather in Paris"]:
        errcount += 1
        _print_fail(
            "Error for `decompose_and_fuse`: expected the sub-query to be "
            f"routed through the API manager, got {manager.queries!r}."
        )
    if result != "FUSED ANSWER: 21C":
        errcount += 1
        _print_fail(
            "Error for `decompose_and_fuse`: expected the fused reply to be "
            f"returned, got {result!r}."
        )

    _report("decompose_and_fuse", errcount)


@_guarded
def test_iterative_refine():
    """Check that each pass carries history forward and the answer is unwrapped."""
    agent, manager = _make_agent(iterative_refinement_model="fake/refinement")
    if _not_ready(agent, "iterative_refine"):
        return

    scripted = ScriptedGenerate(
        [
            "<sub-query>weather in Paris on 2025-08-27</sub-query>",
            "<final_answer>21C</final_answer>",
        ]
    )
    agent.generate = scripted

    errcount = 0
    result = agent.iterative_refine("How warm was Paris on 2025-08-27?")

    if result != "21C":
        errcount += 1
        _print_fail(
            "Error for `iterative_refine`: expected the text inside "
            f"<final_answer> tags ('21C'), got {result!r}."
        )
    if manager.queries != ["weather in Paris on 2025-08-27"]:
        errcount += 1
        _print_fail(
            "Error for `iterative_refine`: expected the sub-query the model "
            f"asked for to be routed, got {manager.queries!r}."
        )
    if scripted.models and scripted.models[0] != "fake/refinement":
        errcount += 1
        _print_fail(
            "Error for `iterative_refine`: deciding the next action is the "
            "reasoning step, so use `self.iterative_refinement_model`, got "
            f"{scripted.models[0]!r}."
        )
    if len(scripted.calls) < 2:
        errcount += 1
        _print_fail(
            "Error for `iterative_refine`: expected a second pass after the "
            f"sub-query was answered, got {len(scripted.calls)} call(s)."
        )
    else:
        second_prompt = scripted.prompts[1]
        if "weather in Paris on 2025-08-27" not in second_prompt:
            errcount += 1
            _print_fail(
                "Error for `iterative_refine`: the second prompt has to "
                "include the history gathered so far, otherwise every pass "
                "repeats the same first step."
            )
        if "Paris was 21C at noon" not in second_prompt:
            errcount += 1
            _print_fail(
                "Error for `iterative_refine`: the history in the prompt has "
                "to carry the sub-query results, not just the sub-queries."
            )

    _report("iterative_refine", errcount)


####################
# Part 3           #
####################


@_guarded
def test_run_pipeline():
    """Check that the pipeline can work through a multi-step question.

    Whichever components the pipeline drives is up to you, so this only checks
    that it keeps going until the model has what it asked for. The fake model
    hands back the answer once three sub-queries have been answered, offering
    every step still outstanding each time, so a pipeline that gathers them one
    at a time and one that fans out in parallel both get there.
    """
    query = (
        "Over the first three days of August 2025, did the morning (9AM) "
        "temperature in Miami ever drop below 70 Fahrenheit?"
    )
    steps_needed = 3
    agent, manager = _make_agent(
        route_result=lambda sub_query: {
            "results": f"evidence for {sub_query}",
            "api_used": "get_weather",
            "params": {"sub_query": sub_query},
        }
    )
    if _not_ready(agent, "run_pipeline"):
        return

    def reply(prompt, model=None):
        done = len(manager.queries)
        if done >= steps_needed:
            return "<final_answer>42</final_answer>"
        return "".join(
            f"<sub-query>step {i + 1}</sub-query>" for i in range(done, steps_needed)
        )

    scripted = ScriptedGenerate(reply=reply)
    agent.generate = scripted

    errcount = 0
    result = agent.run_pipeline(query)

    if not scripted.calls:
        errcount += 1
        _print_fail(
            "Error for `run_pipeline`: it never asked a model anything, so it "
            "cannot have answered the query. Drive one of the pipelines built "
            "earlier, or one of your own."
        )
    if len(manager.queries) < steps_needed:
        errcount += 1
        _print_fail(
            f"Error for `run_pipeline`: the model asked for {steps_needed} pieces "
            f"of evidence before it would answer, but only {len(manager.queries)} "
            "reached the API manager. A pipeline that stops after one API call "
            "cannot answer a question spanning three days -- check the iteration "
            "budget you give the refinement loop."
        )
    if not isinstance(result, str) or "42" not in (result or ""):
        errcount += 1
        _print_fail(
            "Error for `run_pipeline`: once its sub-queries were answered the "
            "model replied <final_answer>42</final_answer>, so '42' should come "
            f"back from `run_pipeline`, but it returned {result!r}."
        )

    _report("run_pipeline", errcount)
