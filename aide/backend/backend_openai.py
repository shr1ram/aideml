"""Backend for OpenAI API."""

import json
import logging
import os
import re
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("aide")

OPENAI_BASE_URL = "https://api.openai.com/v1"

# A dropped/half-open connection to a custom proxy (e.g. LiteLLM) must NOT
# hang the agent forever. Give the client a hard request timeout and a few
# retries so a stalled call fails fast and AIDE's own loop can recover.
OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "4"))

_client: openai.OpenAI = None  # type: ignore
_custom_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    # Use real OpenAI API with proper API key, explicitly override base_url
    api_key = os.getenv("OPENAI_API_KEY")
    _client = openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL,
                            timeout=OPENAI_REQUEST_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)


@once
def _setup_custom_client():
    global _custom_client
    # Only create custom client if base URL is set
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if base_url:
        _custom_client = openai.OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=OPENAI_REQUEST_TIMEOUT, max_retries=OPENAI_MAX_RETRIES,
        )


def _streamed_chat_completion(client, messages, **kwargs):
    """chat.completions with stream=True, reassembled into a non-streaming-shaped
    object (.choices[0].message.content/.tool_calls + .usage).

    The team LiteLLM proxy sits behind an Alibaba gateway that 504s after ~180s on
    *idle* non-streaming connections. Reasoning models (Kimi) often think longer
    than that before the first byte, so a plain create() hangs then 504s. Streaming
    keeps bytes flowing so the gateway never reaps the socket.
    """
    from types import SimpleNamespace

    kwargs.pop("stream", None)
    stream = client.chat.completions.create(
        messages=messages, stream=True,
        stream_options={"include_usage": True}, **kwargs,
    )
    content_parts, tool_calls = [], {}
    finish_reason, usage = None, None
    model = kwargs.get("model")
    resp_id = system_fingerprint = created = None
    for chunk in stream:
        if getattr(chunk, "model", None):
            model = chunk.model
        if getattr(chunk, "id", None):
            resp_id = chunk.id
        if getattr(chunk, "system_fingerprint", None):
            system_fingerprint = chunk.system_fingerprint
        if getattr(chunk, "created", None):
            created = chunk.created
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        for tc in (getattr(delta, "tool_calls", None) or []):
            # Some providers stream tool_call deltas with index=None; fall back to
            # positional so distinct calls don't collide on a single None key.
            idx = tc.index if tc.index is not None else len(tool_calls)
            slot = tool_calls.setdefault(idx, {"id": None, "name": "", "args": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["args"] += tc.function.arguments
    msg_tool_calls = None
    if tool_calls:
        msg_tool_calls = [
            SimpleNamespace(
                id=slot["id"], type="function",
                function=SimpleNamespace(name=slot["name"], arguments=slot["args"]),
            )
            for _, slot in sorted(tool_calls.items())
        ]
    message = SimpleNamespace(
        content="".join(content_parts) if content_parts else None,
        tool_calls=msg_tool_calls, role="assistant",
    )
    usage_estimated = False
    if usage is None:
        # The proxy didn't return a usage chunk (stream_options unsupported).
        # Don't silently record 0 tokens — that corrupts cost data. Estimate from
        # char counts (~4 chars/token) and flag it so the trace knows it's approximate.
        def _toklen(s):
            return max(0, len(s or "") // 4)
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        usage = SimpleNamespace(
            prompt_tokens=_toklen(" " * prompt_chars),
            completion_tokens=_toklen("".join(content_parts)),
            total_tokens=0,
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        usage_estimated = True
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
        model=model,
        id=resp_id,
        system_fingerprint=system_fingerprint,
        created=created,
        usage_estimated=usage_estimated,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    If the model doesn't support function calling, gracefully degrade to text generation.
    """
    # Setup clients
    _setup_openai_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    if "max_tokens" in filtered_kwargs:
        filtered_kwargs["max_output_tokens"] = filtered_kwargs.pop("max_tokens")

    if (
        re.match(r"^o\d", filtered_kwargs["model"])
        or filtered_kwargs["model"] == "codex-mini-latest"
    ):
        filtered_kwargs.pop("temperature", None)

    # Use different API based on whether this is a non-OpenAI model with custom base URL
    model_name = filtered_kwargs.get("model", "")
    is_openai_model = re.match(r"^(gpt-|o\d-|codex-mini-latest$)", model_name)
    use_chat_api = os.getenv("OPENAI_BASE_URL") is not None and not is_openai_model

    if use_chat_api:
        _setup_custom_client()
        # Standard chat completions API (for local servers).
        # chat.completions expects `max_tokens`, but above we renamed it to
        # `max_output_tokens` for the responses API — undo that for this path.
        if "max_output_tokens" in filtered_kwargs:
            filtered_kwargs["max_tokens"] = filtered_kwargs.pop("max_output_tokens")
        messages = opt_messages_to_list(system_message, user_message)
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
    else:
        # OpenAI responses API (for official OpenAI models)
        messages = opt_messages_to_list(system_message, user_message)
        # Convert to the responses API format
        for i in range(len(messages)):
            messages[i]["content"] = [
                {"type": "input_text", "text": messages[i]["content"]}
            ]
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_responses_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_responses_tool_choice_dict

    logger.info(f"OpenAI API request: system={system_message}, user={user_message}")

    t0 = time.time()

    # Attempt the API call
    try:
        if use_chat_api:
            # Use custom client if available, otherwise fall back to default
            client_to_use = _custom_client if _custom_client else _client
            response = backoff_create(
                _streamed_chat_completion,
                OPENAI_TIMEOUT_EXCEPTIONS,
                client_to_use,
                messages,
                **filtered_kwargs,
            )
        else:
            response = backoff_create(
                _client.responses.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                input=messages,
                **filtered_kwargs,
            )
    except openai.BadRequestError as e:
        # Check whether the error indicates that function calling is not supported
        if "function calling" in str(e).lower() or "tools" in str(e).lower():
            logger.warning(
                "Function calling was attempted but is not supported by this model. "
                "Falling back to plain text generation."
            )
            # Remove function-calling parameters and retry
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)

            # Retry without function calling
            if use_chat_api:
                # Use custom client if available, otherwise fall back to default
                client_to_use = _custom_client if _custom_client else _client
                response = backoff_create(
                    _streamed_chat_completion,
                    OPENAI_TIMEOUT_EXCEPTIONS,
                    client_to_use,
                    messages,
                    **filtered_kwargs,
                )
            else:
                response = backoff_create(
                    _client.responses.create,
                    OPENAI_TIMEOUT_EXCEPTIONS,
                    input=messages,
                    **filtered_kwargs,
                )
        else:
            # If it's some other error, re-raise
            raise

    req_time = time.time() - t0

    # Parse the output based on API type
    if use_chat_api:
        # Chat completions API response
        message = response.choices[0].message

        if (
            hasattr(message, "tool_calls")
            and message.tool_calls
            and func_spec is not None
        ):
            # Function call found
            tool_call = message.tool_calls[0]
            if tool_call.function.name == func_spec.name:
                try:
                    output = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as ex:
                    logger.error(
                        "Error decoding function arguments:\n"
                        f"{tool_call.function.arguments}"
                    )
                    raise ex
            else:
                logger.warning(
                    f"Function name mismatch: expected {func_spec.name}, "
                    f"got {tool_call.function.name}. Fallback to text."
                )
                output = message.content
        else:
            # No function call, use regular text output
            output = message.content

        in_tokens = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens
    else:
        # Responses API response
        if (
            hasattr(response, "output")
            and response.output is not None
            and len(response.output) > 0
        ):
            # Look for function calls in the response output items
            function_call_item = None
            for output_item in response.output:
                if hasattr(output_item, "type") and output_item.type == "function_call":
                    function_call_item = output_item
                    break

            if function_call_item is not None:
                # Function call found
                if func_spec is not None and function_call_item.name == func_spec.name:
                    try:
                        output = json.loads(function_call_item.arguments)
                    except json.JSONDecodeError as ex:
                        logger.error(
                            "Error decoding function arguments:\n"
                            f"{function_call_item.arguments}"
                        )
                        raise ex
                else:
                    # Function name mismatch or no func_spec
                    if func_spec is not None:
                        logger.warning(
                            f"Function name mismatch: expected {func_spec.name}, "
                            f"got {function_call_item.name}. Fallback to text."
                        )
                    output = response.output_text
            else:
                # No function call, use regular text output
                output = response.output_text
        else:
            # Fallback to output_text
            output = response.output_text

        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens

    info = {
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "model": response.model,
        "created": getattr(response, "created", None),
    }

    logger.info(
        f"OpenAI API call completed - {response.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"OpenAI API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
