#!/usr/bin/env python3
"""Gradescope / local grader for XAG329A Assignment 3.

Every test runs the submitted code against fakes: a stand-in API manager and
canned model replies. Nothing reaches the network, so a failure here describes
the submission rather than the weather at OpenRouter. Part 3's accuracy is the
one exception, and it is read out of the notebook the student saved rather than
re-run.

Failure messages are written for the student: what was expected, what actually
happened, and where to look. The self-check helpers in
`xag329a_a3/tests/test.py` cover the same ground as the basic tests, so most
failures can be reproduced in the notebook without waiting on Gradescope, and
the messages below say which helper to run.
"""

import os
import sys
import unittest
from contextlib import contextmanager

from graderUtil import graded, CourseTestRunner, GradedTestCase, HiddenPrints
from autograde_utils import accuracy_from_tagged_cell, stdout_from_tagged_cell

### BEGIN_HIDE ###
### END_HIDE ###

SUBMISSION_NOTEBOOK = os.path.join(
    os.path.dirname(__file__), "submission", "assignment3.ipynb"
)

# Cell tags are invisible in the notebook UI, so failures name these cells the
# way a student sees them instead.
PIPELINE_CELL = "the run_pipeline evaluation cell at the end of Part 3"
ZERO_SHOT_CELL = "the zero-shot evaluation cell in Part 3"

# An unpatched `generate_llm` checks for a key before anything else, and a
# missing key would report itself as a setup problem. A dummy one lets the
# network guard below be what reports the mistake instead.
os.environ.setdefault("OPENROUTER_API_KEY", "gradescope-dummy-key")

# Ensure the student submission package (xag329a_a3) is importable.
_SUBMISSION_DIR = os.path.join(os.path.dirname(__file__), "submission")
if _SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, _SUBMISSION_DIR)


class _AttemptedNetworkCall(BaseException):
    """Raised when a graded test path tries to reach a real LLM endpoint.

    Derives from BaseException rather than Exception on purpose: `generate_llm`
    wraps its request in `except Exception` and converts any failure into a
    `None` return, which would silently mask the guard instead of failing the
    test with this message.
    """


def _guard_network_calls(utils_module):
    """Make any unpatched LLM request fail immediately instead of retrying.

    `_make_completion_request` is wrapped in a tenacity retry that backs off up
    to 60s over 5 attempts. Left in place, one accidental real call would blow
    the per-test timeout and be reported as a timeout rather than as the missing
    fake it actually is.
    """

    def _raise(args):
        raise _AttemptedNetworkCall(
            "A graded test attempted a real LLM API call with args "
            f"{args!r}. Tests must patch generate_llm / generate_structured, "
            "or inject a fake api_manager, and never reach the network."
        )

    utils_module._make_completion_request = _raise


# Import the student package once, at grader import time. Two reasons: the
# LiteLLM import is slow enough to eat a per-test timeout, and a submission that
# fails to import should produce a readable failure on every test rather than
# breaking unittest discovery. Only the graded modules are imported, so a broken
# optional Part 4 cannot take the whole grader down.
_IMPORT_ERROR = None
try:
    from xag329a_a3 import utils as _utils
    from xag329a_a3 import api_manager as _api_manager
    from xag329a_a3 import multi_lm_agent as _multi_lm_agent

    _guard_network_calls(_utils)
except Exception as exc:  # noqa: BLE001 - surfaced per-test below
    _IMPORT_ERROR = exc


############
# FIXTURES #
############


@contextmanager
def _patched(module, name, replacement):
    """Swap a module-level function, restoring it even if a test fails."""
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


def _short(value, limit=240):
    """Render a value for a failure message without pasting a whole prompt."""
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} characters total)"


class _FakeAPIManager:
    """Stands in for APIManager so an agent can be built without API keys.

    `MultiLMAgent` only stores the manager, reads `api_functions` off it, and
    calls `route_query`, so nothing here touches the network. `result` may be a
    callable to vary the answer per sub-query.
    """

    DEFAULT_RESULT = {
        "results": "Paris was 21C at noon",
        "api_used": "get_weather",
        "params": {"location": "Paris, France", "date": "2025-08-27"},
    }

    def __init__(self, result=None):
        self.api_functions = {
            name: (lambda **kwargs: {}, None)
            for name in ("tavily_search", "get_stock_data", "compute", "get_weather")
        }
        self.router_model = "fake/router-model"
        self.queries = []
        self._result = self.DEFAULT_RESULT if result is None else result

    def route_query(self, query, model=None):
        self.queries.append(query)
        if callable(self._result):
            return self._result(query)
        return self._result


class _ScriptedGenerate:
    """Stands in for `agent.generate`, handing back canned replies.

    Replies are taken from `replies` in order; `by_model` answers based on which
    model was asked, which keeps a test from depending on the order a
    submission happens to make its calls. Every call is recorded so a test can
    assert which model saw which prompt.
    """

    def __init__(self, replies=(), by_model=None, default=None):
        self._replies = list(replies)
        self._by_model = by_model
        self.default = default
        self.calls = []

    def __call__(self, query, model=None, temperature=None):
        self.calls.append({"prompt": query, "model": model, "temperature": temperature})
        if self._by_model is not None:
            return self._by_model(query, model)
        if self._replies:
            return self._replies.pop(0)
        return self.default

    @property
    def models(self):
        return [call["model"] for call in self.calls]

    @property
    def prompts(self):
        return [call["prompt"] for call in self.calls]


def _bare_api_manager():
    """Build an APIManager with only what routing touches.

    `route_query` and `_parse_query_params` read `router_model` and
    `api_functions` and nothing else, so `__init__` is skipped entirely: no
    Tavily, Massive, Wolfram, or geocoder client is constructed and no API key
    is needed. Returns the manager and the list its fake tools append to.
    """
    manager = _api_manager.APIManager.__new__(_api_manager.APIManager)
    manager.router_model = "fake/router-model"
    invoked = []

    def make_tool(name):
        def tool(**kwargs):
            invoked.append((name, kwargs))
            return {"value": f"{name} ran"}

        tool.__doc__ = f"Fake {name} tool used by the grader."
        tool.__name__ = name
        return tool

    manager.api_functions = {
        "tavily_search": (make_tool("tavily_search"), _api_manager.TavilySearchParams),
        "get_stock_data": (make_tool("get_stock_data"), _api_manager.StockDataParams),
        "compute": (make_tool("compute"), _api_manager.ComputeParams),
        "get_weather": (make_tool("get_weather"), _api_manager.GetWeatherParams),
    }
    return manager, invoked


def _fusion_replies(agent):
    """Canned replies for the three stages of `decompose_and_fuse`.

    The decomposition call is answered whatever model it used, since which model
    that should be is graded in Part 2a; these tests are only about sampling
    across the fusion models and fusing the results.
    """

    def reply_for(prompt, model):
        if model in tuple(agent.fusion_sample_models):
            return f"SAMPLE from {model}: 21C"
        if model == agent.fusion_model:
            return "FUSED ANSWER: 21C"
        return "<sub-query>weather in Paris on 2025-08-27</sub-query>"

    return reply_for


_REQUIRED_AGENT_ATTRS = (
    "api_manager",
    "decomposition_model",
    "iterative_refinement_model",
    "fusion_model",
    "fusion_sample_models",
    "generation_temp",
)


class _A3TestCase(GradedTestCase):
    """Shared fixtures, plus failure messages aimed at the student."""

    # Our messages already say what was expected and what happened, so drop
    # unittest's generated "1.0 != 0.55" tail.
    longMessage = False

    def setUp(self):
        self._where = None
        self._self_check = None
        if _IMPORT_ERROR is not None:
            self.fail(
                "Could not import the xag329a_a3 package from the submission: "
                f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}. Submit the whole "
                "xag329a_a3/ folder alongside the notebook, and make sure it runs in "
                "the notebook before submitting."
            )

    def context(self, where=None, self_check=None):
        """Record where the student should look when this test fails."""
        self._where = where
        self._self_check = self_check

    @property
    def hint(self):
        parts = []
        if self._where:
            parts.append(f"This is graded from {self._where}.")
        if self._self_check:
            name = self._self_check
            parts.append(
                "You can reproduce this in the notebook with "
                f"`from xag329a_a3.tests import {name}; {name}()`."
            )
        return " ".join(parts)

    def _formatMessage(self, msg, standardMsg):
        # unittest's assertions format the message and then hand it to fail(),
        # which formats it again, so only append a hint that is not there yet.
        text = super()._formatMessage(msg, standardMsg)
        hint = self.hint
        if hint and hint not in (text or ""):
            return f"{text}\n{hint}"
        return text

    def fail(self, msg=None):
        super().fail(self._formatMessage(msg, ""))

    def feedback(self, text):
        """Show `text` on Gradescope even when the test passes."""
        method = getattr(type(self), self._testMethodName)
        setattr(method, "__student_feedback__", text)

    @contextmanager
    def no_crash(self, what, fix):
        """Turn an exception out of student code into a readable failure.

        A missing None check usually surfaces as an AttributeError or TypeError
        deep in the call, which tells a student nothing on its own.
        """
        try:
            yield
        except self.failureException:
            raise
        except Exception as exc:
            self.fail(f"{what} raised {type(exc).__name__}: {exc}. {fix}")

    def make_agent(self, route_result=None, **kwargs):
        """Build a MultiLMAgent wired to a fake API manager."""
        settings = {
            "decomposition_model": "fake/decomposition",
            "iterative_refinement_model": "fake/refinement",
            "fusion_model": "fake/fusion",
            "fusion_sample_models": ("fake/sample-a", "fake/sample-b"),
        }
        settings.update(kwargs)
        manager = _FakeAPIManager(result=route_result)
        try:
            agent = _multi_lm_agent.MultiLMAgent(manager, **settings)
        except TypeError as exc:
            self.fail(
                f"`MultiLMAgent(...)` rejected the arguments the notebook passes it: "
                f"{exc}. Leave `__init__`'s signature as it was given and fill in only "
                "its body."
            )
        return agent, manager

    def assertAgentReady(self, agent, needs=()):
        """Fail early, and in one place, when `__init__` is unfinished."""
        missing = [
            name for name in (needs or _REQUIRED_AGENT_ATTRS) if not hasattr(agent, name)
        ]
        if missing:
            attrs = ", ".join(f"self.{name}" for name in missing)
            self.fail(
                f"`MultiLMAgent.__init__` never set {attrs}, so the rest of the agent "
                "has nothing to work with. Finish the Part 0 TODO in "
                "xag329a_a3/multi_lm_agent.py first: it is what stores the api manager "
                "and the model names every later part reads."
            )


#########
# TESTS #
#########


class Test_0(_A3TestCase):
    """Part 0: LLM performance with a single call (5 points)."""

    GENERATE = "the `generate` TODO in xag329a_a3/multi_lm_agent.py (Part 0)"
    INIT = "the `__init__` TODO in xag329a_a3/multi_lm_agent.py (Part 0)"

    @graded(timeout=30)
    def test_0(self):
        """0-0-basic: generate sends the query to the requested model and returns its reply"""
        self.context(where=self.GENERATE, self_check="test_generate")
        agent, _ = self.make_agent(generation_temp=0.55)
        self.assertAgentReady(agent, ("generation_temp",))

        seen = {}

        def fake_generate_llm(model, messages, temperature=0.7, *args, **kwargs):
            seen["model"] = model
            seen["messages"] = messages
            seen["temperature"] = temperature
            return "FAKE REPLY"

        with _patched(_multi_lm_agent, "generate_llm", fake_generate_llm):
            with self.no_crash(
                "`generate`",
                "It should build a messages list from the query and hand it to "
                "`generate_llm`.",
            ):
                with HiddenPrints():
                    reply = agent.generate("What is 40 + 2?", model="fake/model-x")

        self.assertTrue(
            seen,
            "`generate` never called `generate_llm`, so the query was never sent to a "
            "model. Build a messages list from the query and pass it to `generate_llm` "
            "from utils.py.",
        )
        self.assertEqual(
            reply,
            "FAKE REPLY",
            "`generate` should hand back the text the model produced. `generate_llm` "
            f"returned 'FAKE REPLY' here, but `generate` returned {_short(reply)}.",
        )
        self.assertEqual(
            seen.get("model"),
            "fake/model-x",
            "`generate` was called with model='fake/model-x', so that is the model "
            f"`generate_llm` should be asked for, but it received "
            f"{_short(seen.get('model'))}. Pass the `model` argument through instead of "
            "naming a model yourself.",
        )
        self.assertEqual(
            seen.get("temperature"),
            0.55,
            "With `temperature` left as None, `generate` falls back to "
            "`self.generation_temp`, which is 0.55 on this agent, but `generate_llm` was "
            f"asked for temperature={_short(seen.get('temperature'))}. Pass the local "
            "`temperature` variable along rather than a literal or nothing at all.",
        )
        messages = seen.get("messages") or []
        joined = " ".join(
            str(m.get("content", "")) for m in messages if isinstance(m, dict)
        )
        self.assertIn(
            "What is 40 + 2?",
            joined,
            "The query has to appear in the messages sent to `generate_llm`, since the "
            f"model cannot answer a question it never sees. Messages sent: "
            f"{_short(messages)}.",
        )

        with _patched(_multi_lm_agent, "generate_llm", lambda *a, **k: None):
            with HiddenPrints():
                failed = agent.generate("anything", model="fake/model-x")
        self.assertIsNone(
            failed,
            "`generate_llm` returns None when its retries are exhausted, a rate limit "
            "for instance. `generate` has to pass that None along so callers can tell a "
            f"failed call from an answer, but it returned {_short(failed)}.",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1(_A3TestCase):
    """Part 1: API-augmented LLM pipeline (15 points)."""

    ROUTING = (
        "the `route_query` and `_parse_query_params` TODOs in "
        "xag329a_a3/api_manager.py (Part 1e)"
    )
    SINGLE_CALL = (
        "the `single_LM_with_single_API_call` TODO in "
        "xag329a_a3/multi_lm_agent.py (Part 1f)"
    )

    def route_and_params(self, tool="compute", params=None):
        """A `generate_structured` stand-in that routes to `tool`, then fills it."""
        params = params or _api_manager.ComputeParams(wolfram_query="40 + 2")
        schemas = []

        def fake_generate_structured(model, messages, schema_model, temperature=0.0):
            schemas.append(schema_model)
            if schema_model is _api_manager.APIResponse:
                return _api_manager.APIResponse(api_name=tool)
            if schema_model is type(params):
                return params
            return None

        return fake_generate_structured, schemas

    @graded(timeout=30)
    def test_0(self):
        """1-0-basic: route_query picks a tool, fills its arguments, and runs it"""
        self.context(where=self.ROUTING, self_check="test_api_routing")
        manager, invoked = _bare_api_manager()
        fake_generate_structured, schemas = self.route_and_params()

        with _patched(_api_manager, "generate_structured", fake_generate_structured):
            with self.no_crash(
                "`route_query`",
                "It should ask a model for a tool name, parse that tool's arguments, and "
                "leave the name in `function_name` and the arguments in `params`.",
            ):
                with HiddenPrints():
                    result = manager.route_query("What is 40 + 2?")

        self.assertTrue(
            schemas,
            "`route_query` never called `generate_structured`, so no model chose a tool. "
            "The tool choice is the first of its two model-driven steps.",
        )
        self.assertIn(
            _api_manager.APIResponse,
            schemas,
            "The tool choice should come from a `generate_structured` call using the "
            "`APIResponse` schema, which is what constrains the reply to one of the four "
            f"tool names. The schemas requested were {_short(schemas)}.",
        )
        self.assertIsInstance(
            result,
            dict,
            f"`route_query` should return a dict, but returned {_short(result)}.",
        )
        self.assertEqual(
            result.get("api_used"),
            "compute",
            "The router chose 'compute' here, so that is the tool `route_query` should "
            f"report using, but `api_used` came back as {_short(result.get('api_used'))}. "
            "Leave the chosen name in `function_name`.",
        )
        self.assertEqual(
            result.get("params"),
            {"wolfram_query": "40 + 2"},
            "`params` should be the arguments parsed for the chosen tool, but came back "
            f"as {_short(result.get('params'))}. Leave them in `params` so the code below "
            "your TODO can call the tool with them.",
        )
        self.assertEqual(
            result.get("results"),
            {"value": "compute ran"},
            "`results` should hold what the tool returned, but came back as "
            f"{_short(result.get('results'))}. This usually means `function_name` or "
            "`params` was not what the call below your TODO expected.",
        )
        self.assertEqual(
            invoked,
            [("compute", {"wolfram_query": "40 + 2"})],
            "The chosen tool should be called exactly once, with the parsed arguments. "
            f"Tool calls recorded: {_short(invoked)}.",
        )

    @graded(timeout=30)
    def test_1(self):
        """1-1-basic: _parse_query_params returns the validated arguments as a dict"""
        self.context(where=self.ROUTING, self_check="test_api_routing")
        manager, _ = _bare_api_manager()
        asked = []

        def fake_generate_structured(model, messages, schema_model, temperature=0.0):
            asked.append(schema_model)
            if schema_model is _api_manager.GetWeatherParams:
                return _api_manager.GetWeatherParams(
                    location="Paris, France", date="2025-08-27", hour="12"
                )
            return None

        with _patched(_api_manager, "generate_structured", fake_generate_structured):
            with self.no_crash(
                "`_parse_query_params`",
                "It should ask `generate_structured` for the named tool's arguments and "
                "return them as a plain dict.",
            ):
                with HiddenPrints():
                    params = manager._parse_query_params(
                        "How warm was Paris on 2025-08-27?", "get_weather"
                    )

        self.assertTrue(
            asked,
            "`_parse_query_params` never called `generate_structured`, so nothing filled "
            "in the tool's arguments. `generate_structured` asks the model for JSON and "
            "validates it against a Pydantic schema in one call, so the work here is the "
            "prompt rather than the parsing.",
        )
        self.assertIn(
            _api_manager.GetWeatherParams,
            asked,
            "The schema to validate against is the one paired with the requested "
            "function in `self.api_functions`; 'get_weather' pairs with "
            f"`GetWeatherParams`. The schemas requested were {_short(asked)}.",
        )
        self.assertEqual(
            params,
            {"location": "Paris, France", "date": "2025-08-27", "hour": "12"},
            "`_parse_query_params` should hand back a plain dict of the validated "
            f"arguments, but returned {_short(params)}. Call `.model_dump()` on the "
            "object `generate_structured` returns rather than returning the object "
            "itself.",
        )

        with _patched(_api_manager, "generate_structured", lambda *a, **k: None):
            with self.no_crash(
                "`_parse_query_params`",
                "`generate_structured` returns None when the model does not answer or "
                "its reply fails validation; check for that before using the result.",
            ):
                with HiddenPrints():
                    failed = manager._parse_query_params("anything", "compute")
        self.assertIsNone(
            failed,
            "When `generate_structured` comes back None, `_parse_query_params` should "
            f"return None so its caller can report the failure, but it returned "
            f"{_short(failed)}.",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=30)
    def test_3(self):
        """1-3-basic: single_LM_with_single_API_call answers from the tool's output"""
        self.context(
            where=self.SINGLE_CALL, self_check="test_single_lm_with_single_api_call"
        )
        agent, manager = self.make_agent()
        self.assertAgentReady(agent)

        scripted = _ScriptedGenerate(["Paris was 21C. Final answer: 21C"])
        agent.generate = scripted
        query = "How warm was Paris on 2025-08-27?"

        with self.no_crash(
            "`single_LM_with_single_API_call`",
            "It is two moves: route the query through `self.api_manager`, then ask "
            "`self.generate` to turn the tool output into an answer.",
        ):
            with HiddenPrints():
                result = agent.single_LM_with_single_API_call(query, "fake/model-x")

        self.assertEqual(
            manager.queries,
            [query],
            "The original query should be routed through the API manager exactly once; "
            "`route_query` already picks the tool and runs it. Queries routed: "
            f"{_short(manager.queries)}.",
        )
        self.assertTrue(
            scripted.calls,
            "After routing, an LM has to turn the raw tool output into an answer, but "
            "`self.generate` was never called. The tool output on its own is not an "
            "answer to the question.",
        )
        prompt = scripted.prompts[0]
        self.assertIn(
            query,
            prompt,
            "The prompt has to include the original query, since the model answering it "
            f"never saw it. Prompt sent: {_short(prompt)}.",
        )
        self.assertIn(
            "Paris was 21C at noon",
            prompt,
            "The prompt has to include the tool's output, since the model has no other "
            f"way to see what the API returned. Prompt sent: {_short(prompt)}.",
        )
        self.assertEqual(
            result,
            "Paris was 21C. Final answer: 21C",
            "The model's answer is what gets graded, so return it unchanged; this "
            f"returned {_short(result)}.",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_2(_A3TestCase):
    """Part 2: self-improvement techniques (20 points)."""

    DECOMPOSE = (
        "the query decomposition TODOs in xag329a_a3/multi_lm_agent.py (Part 2a)"
    )
    SYNTHESIS = (
        "the `_get_synthesis_prompt` TODO in xag329a_a3/multi_lm_agent.py (Part 2b)"
    )
    FUSE = "the `decompose_and_fuse` TODO in xag329a_a3/multi_lm_agent.py (Part 2c)"
    REFINE = "the `iterative_refine` TODO in xag329a_a3/multi_lm_agent.py (Part 2d)"

    DECOMPOSED_STEPS = (
        {
            "sub_query": "weather in Paris on 2025-08-27",
            "api": "get_weather",
            "params": {"location": "Paris, France", "date": "2025-08-27"},
            "results": "21C and clear",
        },
        {
            "sub_query": "closing price of TSLA on 2025-08-27",
            "api": "get_stock_data",
            "params": {"ticker": "TSLA", "date": "2025-08-27"},
            "results": "stale quote nobody should trust",
            "error": "the stock API rate limited the request",
        },
    )

    @graded(timeout=30)
    def test_0(self):
        """2-0-basic: the decomposition prompt carries the query and asks for tagged sub-queries"""
        self.context(where=self.DECOMPOSE, self_check="test_query_decomposition")
        agent, _ = self.make_agent()
        self.assertAgentReady(agent)
        query = "Which city was windier on 2025-08-01: Chicago or Boston?"

        with self.no_crash(
            "`_get_query_decomposition_prompt`",
            "It should return the prompt string that asks a model to split the query up.",
        ):
            with HiddenPrints():
                prompt = agent._get_query_decomposition_prompt(query)

        self.assertIsInstance(
            prompt,
            str,
            "`_get_query_decomposition_prompt` should return the prompt as a string, but "
            f"returned {_short(prompt)}.",
        )
        self.assertTrue(
            prompt.strip(),
            "`_get_query_decomposition_prompt` returned an empty string, so the model "
            "would be asked nothing at all.",
        )
        self.assertIn(
            query,
            prompt,
            "The query being decomposed has to appear in the prompt; the model cannot "
            f"split up a question it never sees. Prompt built: {_short(prompt)}.",
        )
        self.assertIn(
            "<sub-query>",
            prompt,
            "The prompt has to ask for each sub-query wrapped in "
            "<sub-query>...</sub-query> tags, because that is what `_get_sub_queries` "
            f"reads the sub-queries out of. Prompt built: {_short(prompt)}.",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=30)
    def test_2(self):
        """2-2-basic: the synthesis prompt carries the query and the gathered evidence"""
        self.context(where=self.SYNTHESIS, self_check="test_synthesis_prompt")
        agent, _ = self.make_agent()
        self.assertAgentReady(agent)
        query = "How warm was Paris on 2025-08-27?"

        with self.no_crash(
            "`_get_synthesis_prompt`",
            "It should return a prompt built from the query and the sub-query results.",
        ):
            with HiddenPrints():
                prompt = agent._get_synthesis_prompt(query, list(self.DECOMPOSED_STEPS))

        self.assertIsInstance(
            prompt,
            str,
            f"`_get_synthesis_prompt` should return a string, but returned "
            f"{_short(prompt)}.",
        )
        self.assertTrue(
            prompt.strip(),
            "`_get_synthesis_prompt` returned an empty string, so the model would be "
            "asked nothing at all.",
        )
        self.assertIn(
            query,
            prompt,
            "The original query has to appear in the prompt: the model reading it never "
            f"saw the question. Prompt built: {_short(prompt)}.",
        )
        self.assertIn(
            "21C and clear",
            prompt,
            "The results of the sub-queries that succeeded have to appear in the prompt, "
            "since they are the only evidence the model gets. Prompt built: "
            f"{_short(prompt)}.",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=30)
    def test_4(self):
        """2-4-basic: decompose_and_fuse samples every fusion model, then fuses their answers"""
        self.context(where=self.FUSE, self_check="test_decompose_and_fuse")
        agent, manager = self.make_agent()
        self.assertAgentReady(agent)
        query = "How warm was Paris on 2025-08-27?"

        scripted = _ScriptedGenerate(by_model=_fusion_replies(agent))
        agent.generate = scripted

        with self.no_crash(
            "`decompose_and_fuse`",
            "It is three stages: decompose and synthesize, sample one answer per model "
            "in `self.fusion_sample_models`, then fuse those answers with "
            "`self.fusion_model`.",
        ):
            with HiddenPrints():
                result = agent.decompose_and_fuse(query)

        asked = scripted.models
        for model in tuple(agent.fusion_sample_models):
            self.assertIn(
                model,
                asked,
                f"Every model in `self.fusion_sample_models` should be asked for an "
                f"answer, but {model!r} never was. The variety that makes fusion worth "
                f"doing comes from asking different models. Models asked: "
                f"{_short(asked)}.",
            )
        self.assertEqual(
            asked[-1],
            agent.fusion_model,
            "The last call should be the fusion call, using `self.fusion_model`, since "
            f"its reply is what this method returns. Models asked: {_short(asked)}.",
        )
        self.assertEqual(
            manager.queries,
            ["weather in Paris on 2025-08-27"],
            "The sub-queries from decomposition have to be routed through the API "
            f"manager, otherwise there is no evidence to fuse over. Queries routed: "
            f"{_short(manager.queries)}.",
        )
        self.assertEqual(
            result,
            "FUSED ANSWER: 21C",
            "The fusion model's reply is what gets graded, so return it unchanged; this "
            f"returned {_short(result)}.",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=30)
    def test_6(self):
        """2-6-basic: iterative_refine runs a sub-query, then returns the final answer"""
        self.context(where=self.REFINE, self_check="test_iterative_refine")
        agent, manager = self.make_agent(iterative_refinement_model="fake/refinement")
        self.assertAgentReady(agent)
        query = "How warm was Paris on 2025-08-27?"

        scripted = _ScriptedGenerate(
            [
                "<sub-query>weather in Paris on 2025-08-27</sub-query>",
                "<final_answer>21C</final_answer>",
            ],
            default="<final_answer>21C</final_answer>",
        )
        agent.generate = scripted

        with self.no_crash(
            "`iterative_refine`",
            "Each pass is two moves: build the prompt with "
            "`_get_iterative_refinement_prompt`, then ask the model what to do next.",
        ):
            with HiddenPrints():
                result = agent.iterative_refine(query)

        self.assertTrue(
            scripted.calls,
            "`iterative_refine` never called `self.generate`, so no model was ever asked "
            "what to do next.",
        )
        self.assertEqual(
            scripted.models[0],
            "fake/refinement",
            "Deciding whether to answer or keep digging is the reasoning step here, so "
            "use the model `__init__` stored in `self.iterative_refinement_model`; it "
            f"asked {_short(scripted.models[0])}.",
        )
        self.assertEqual(
            manager.queries,
            ["weather in Paris on 2025-08-27"],
            "The sub-query the model asked for has to be routed through the API manager, "
            f"otherwise the next pass has nothing new to work with. Queries routed: "
            f"{_short(manager.queries)}.",
        )
        self.assertEqual(
            result,
            "21C",
            "Once the model answers, the text inside the <final_answer> tags is what "
            f"gets graded; this returned {_short(result)}. The most common cause is a "
            "prompt the model cannot follow, or the loop stopping before it answered.",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_3(_A3TestCase):
    """Part 3: building an LLM agentic workflow (10 points)."""

    PIPELINE = "the `run_pipeline` TODO in xag329a_a3/multi_lm_agent.py (Part 3)"
    # A real multi-step question from the evaluation set: three days of weather
    # and then an answer.
    MULTI_STEP_QUERY = (
        "Over the first three days of August 2025, did the morning (9AM) "
        "temperature in Miami ever drop below 70 Fahrenheit?"
    )

    def staged_agent(self, steps_needed):
        """An agent whose fake model answers once `steps_needed` queries are routed.

        Each reply offers every step still outstanding, so a pipeline that fans
        out in parallel and one that works through them one at a time both reach
        the answer. What is being measured is whether the pipeline can gather
        that much evidence at all, not how it chooses to.
        """
        manager = _FakeAPIManager(
            result=lambda query: {
                "results": f"evidence for {query}",
                "api_used": "get_weather",
                "params": {"sub_query": query},
            }
        )
        agent, _ = self.make_agent()
        agent.api_manager = manager

        def reply_for(prompt, model):
            done = len(manager.queries)
            if done >= steps_needed:
                return "<final_answer>42</final_answer>"
            return "".join(
                f"<sub-query>step {i + 1}</sub-query>" for i in range(done, steps_needed)
            )

        scripted = _ScriptedGenerate(by_model=reply_for)
        agent.generate = scripted
        return agent, manager, scripted

    @graded(timeout=60)
    def test_0(self):
        """3-0-basic: run_pipeline answers a query by way of the components it drives"""
        self.context(where=self.PIPELINE, self_check="test_run_pipeline")
        agent, _, scripted = self.staged_agent(steps_needed=1)

        with self.no_crash(
            "`run_pipeline`",
            "It should drive the components built earlier in the assignment and return "
            "the answer they produce.",
        ):
            with HiddenPrints():
                result = agent.run_pipeline(self.MULTI_STEP_QUERY)

        self.assertTrue(
            scripted.calls,
            "`run_pipeline` never asked a model anything, so it cannot have answered the "
            "query. It should drive one of the pipelines built earlier, or one of your "
            "own.",
        )
        self.assertIsInstance(
            result,
            str,
            "`run_pipeline` should return the answer as a string, since `evaluate_qa` "
            f"grades the text it returns; it returned {_short(result)}.",
        )
        self.assertIn(
            self.MULTI_STEP_QUERY,
            " ".join(str(p) for p in scripted.prompts),
            "The query has to reach a model somewhere in the pipeline, but none of the "
            "prompts contained it.",
        )
        self.assertIn(
            "42",
            result,
            "The fake model answered '42' once it had the evidence it asked for, so "
            f"that answer should come back from `run_pipeline`; it returned "
            f"{_short(result)}. An answer that never arrives usually means the loop gave "
            "up, or that the model's reply was dropped instead of returned.",
        )

    @graded(timeout=60)
    def test_1(self):
        """3-1-basic: run_pipeline gives the pipeline room for a multi-step question"""
        self.context(where=self.PIPELINE, self_check="test_run_pipeline")
        agent, manager, scripted = self.staged_agent(steps_needed=3)

        budget = {}
        refine = getattr(agent, "iterative_refine", None)
        if callable(refine):

            def recording_refine(query, max_iterations=3, *args, **kwargs):
                budget["max_iterations"] = max_iterations
                return refine(query, max_iterations, *args, **kwargs)

            agent.iterative_refine = recording_refine

        with self.no_crash(
            "`run_pipeline`",
            "A question spanning three days needs one API call per day and then one more "
            "step to answer.",
        ):
            with HiddenPrints():
                result = agent.run_pipeline(self.MULTI_STEP_QUERY)

        self.assertTrue(
            scripted.calls,
            "`run_pipeline` never asked a model anything, so it cannot have worked "
            "through a multi-step question. It should drive one of the pipelines built "
            "earlier, or one of your own.",
        )
        if "max_iterations" in budget:
            self.assertGreaterEqual(
                budget["max_iterations"],
                4,
                "This question needs three API calls and then one more step to answer "
                "them, so a refinement loop needs a budget of at least 4; "
                f"`run_pipeline` allowed {budget['max_iterations']}. A failed sub-query "
                "costs an iteration too, so leave some room above the minimum.",
            )
        routed = len(manager.queries)
        self.assertGreaterEqual(
            routed,
            3,
            "The fake model asked for three pieces of evidence before it would answer, "
            f"but only {routed} of them reached the API manager. A pipeline that stops "
            "after one API call cannot answer a question that spans three days.",
        )
        self.assertIn(
            "42",
            result,
            "Once its three sub-queries were answered the fake model gave the final "
            f"answer '42', but `run_pipeline` returned {_short(result)}. The usual cause "
            "is running out of iterations before the model got to answer.",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


if __name__ == "__main__":
    # Student-readable local run (also used by make grade_A3 via graderUtil).
    assignment = unittest.TestSuite()
    assignment.addTest(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    )
    CourseTestRunner(gradescope=False).run(assignment)
