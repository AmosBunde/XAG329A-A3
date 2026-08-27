import os
from typing import Dict, List, Optional, Type, TypeVar, Union

import litellm
from litellm import completion
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_random_exponential

SchemaModel = TypeVar("SchemaModel", bound=BaseModel)
ResponseFormat = Union[Dict, Type[BaseModel]]

# Silence LiteLLM's "Provider List" / feedback banner prints on stderr.
litellm.suppress_debug_info = True
litellm.set_verbose = False


def generate_llm(
    model: str,
    messages: List[Dict],
    temperature: float = 0.7,
    response_format: Optional[ResponseFormat] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """
    Generate a response from an OpenRouter model via LiteLLM.

    Returns the model's reply as a string, or None if all retries fail, so a
    throttled call is never mistaken for an answer.

    For structured output, pass a Pydantic model class as ``response_format``.
    Use ``generate_structured`` instead if you want the reply validated.

    Example message:

    ```python
        messages = [{"role": "system", "content": "You're a helpful assistant."}, {"role": "user", "content": query}]
    ```
    """

    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError(
            "OPENROUTER_API_KEY environment variable must be set."
        )

    args = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        args["response_format"] = _normalize_response_format(response_format)
    if max_tokens:
        args["max_tokens"] = max_tokens

    try:
        return _make_completion_request(args)
    except Exception as e:
        print(f"API call failed: {e}")
        print(f"\tArguments: {args}")
        print("Final attempt failed. Returning None.")
        return None


def generate_structured(
    model: str,
    messages: List[Dict],
    schema_model: Type[SchemaModel],
    temperature: float = 0.0,
) -> Optional[SchemaModel]:
    """
    Ask a model for JSON matching ``schema_model`` and validate the reply.

    Returns an instance of ``schema_model``, or None if the model did not
    respond or answered with something the schema rejects.
    """
    content = generate_llm(model, messages, temperature, response_format=schema_model)
    if not content:
        print(f"No response from {model}.")
        return None

    try:
        return schema_model.model_validate_json(content)
    except ValidationError as e:
        print(f"{model} returned an unusable response: {e}")
        print(f"\tResponse: {content}")
        return None


def _normalize_response_format(response_format: ResponseFormat) -> Dict:
    """
    Accept the several JSON-schema shapes and return the one OpenRouter expects.

    OpenRouter requires the schema nested under a named ``json_schema`` key
    (https://openrouter.ai/docs/guides/features/structured-outputs), so a
    Pydantic model class or a flat ``{"type": "json_schema", "schema": ...}``
    is wrapped here.

    ``strict`` is left off because these schemas have fields with defaults,
    which the strict subset disallows, and because enforcement depends on which
    provider OpenRouter routes to. Callers should still validate the response.
    """
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
            },
        }

    if response_format.get("type") != "json_schema":
        return response_format
    if "json_schema" in response_format:
        return response_format

    schema = response_format.get("schema")
    if schema is None:
        return response_format

    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_format.get("name", "response"),
            "strict": response_format.get("strict", False),
            "schema": schema,
        },
    }


# OpenRouter rate limits can take tens of seconds to clear, so the backoff is
# allowed to grow to a minute and spread over 5 attempts.
@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(5))
def _make_completion_request(args: Dict) -> str:
    """Makes a completion request with retry logic."""
    response = completion(**args)
    return response["choices"][0]["message"]["content"]
