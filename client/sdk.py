"""Python client SDK for the deployed Document Analyst (Part 3).

Talks directly to the Model Serving REST invocation endpoint
(POST /serving-endpoints/{name}/invocations). Uses httpx directly
(rather than the openai SDK) so retry/backoff, timeout, and raw SSE
parsing are all under our own control, per this task's requirements.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx


class AnalystClientError(Exception):
    """Wraps any HTTP-level failure talking to the deployed endpoint."""

    def __init__(self, message: str, status_code: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)


_RETRYABLE_STATUS_CODES = {429, 503}


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.host:
            raise AnalystClientError(
                "No Databricks host provided (pass host= or set DATABRICKS_HOST)."
            )
        if not self.token:
            raise AnalystClientError(
                "No Databricks token provided (pass token= or set DATABRICKS_TOKEN)."
            )

        self._invoke_url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        self._status_url = f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # -- internal helpers -----------------------------------------------

    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Perform an HTTP request with exponential backoff on 429/503."""
        start = time.time()
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(method, url, headers=self._headers, **kwargs)
            except httpx.TimeoutException as e:
                elapsed = time.time() - start
                raise TimeoutError(
                    f"Request to {url} timed out after {elapsed:.2f}s "
                    f"(timeout={self.timeout}s): {e}"
                ) from e

            if response.status_code < 400:
                return response

            request_id = response.headers.get("x-request-id")

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                sleep_s = (2 ** attempt) + 0.1 * attempt
                attempt += 1
                time.sleep(sleep_s)
                continue

            try:
                body = response.json()
                message = body.get("message") or body.get("error") or response.text
            except (json.JSONDecodeError, ValueError):
                message = response.text

            last_error = AnalystClientError(
                message=message,
                status_code=response.status_code,
                request_id=request_id,
            )
            break

        if last_error is not None:
            raise last_error

        raise AnalystClientError("Request failed for an unknown reason.")

    @staticmethod
    def _extract_content(response_json) -> str:
        """Pull the assistant's final answer out of the endpoint's response.

        This models-from-code LangChain endpoint returns MLflow's native
        predictions-style payload — a list wrapping the raw AnalystState
        dict — not a strict OpenAI ChatCompletion object. The synthesizer
        (Task 1.6) guarantees the last entry in `messages` is the AIMessage
        with the final answer, so we read it from there directly.
        """
        if isinstance(response_json, list):
            if not response_json:
                return ""
            response_json = response_json[0]

        if "choices" in response_json:
            choices = response_json.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") or {}
            return message.get("content", "") or ""

        messages = response_json.get("messages") or []
        if not messages:
            return ""

        last_message = messages[-1]
        if isinstance(last_message, dict):
            return last_message.get("content", "") or ""
        return getattr(last_message, "content", "") or ""

    # -- public API -------------------------------------------------------

    def ask(self, question: str) -> str:
        """Send a question to the deployed endpoint and return the final answer."""
        payload = {"messages": [{"role": "user", "content": question}]}
        response = self._request_with_retry("POST", self._invoke_url, json=payload)
        return self._extract_content(response.json())

    def ask_streaming(self, question: str) -> Iterator[str]:
        """Yield text chunks as they arrive from the endpoint.

        Caveat: a models-from-code LangChain endpoint returns a single
        completion unless it implements predict_stream — some endpoints
        reject streaming requests outright (HTTP 400, "This endpoint does
        not support streaming"), others might silently return a single
        non-incremental chunk. This handles both: it tries a real SSE
        stream first, and falls back to a plain non-streaming call
        (yielding the full answer once) if streaming isn't supported.
        """
        payload = {
            "messages": [{"role": "user", "content": question}],
            "stream": True,
        }

        start = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST", self._invoke_url, headers=self._headers, json=payload
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        request_id = response.headers.get("x-request-id")
                        try:
                            body = response.json()
                            message = body.get("message") or body.get("error") or response.text
                        except (json.JSONDecodeError, ValueError):
                            message = response.text

                        if "does not support streaming" in message.lower():
                            # Endpoint explicitly rejects streaming — fall
                            # back to a plain non-streaming call below.
                            pass
                        else:
                            raise AnalystClientError(
                                message=message,
                                status_code=response.status_code,
                                request_id=request_id,
                            )
                    else:
                        content_type = response.headers.get("content-type", "")
                        got_any_chunk = False

                        if "text/event-stream" in content_type:
                            for line in response.iter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                data_str = line[len("data:"):].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except (json.JSONDecodeError, ValueError):
                                    continue

                                choices = chunk.get("choices") or []
                                if not choices:
                                    continue
                                delta = choices[0].get("delta") or {}
                                piece = delta.get("content")
                                if piece:
                                    got_any_chunk = True
                                    yield piece

                            if got_any_chunk:
                                return

                        full_bytes = response.read()
                        try:
                            body = json.loads(full_bytes)
                        except (json.JSONDecodeError, ValueError):
                            body = None

                        if body is not None:
                            answer = self._extract_content(body)
                            if answer:
                                yield answer
                            return

        except httpx.TimeoutException as e:
            elapsed = time.time() - start
            raise TimeoutError(
                f"Streaming request timed out after {elapsed:.2f}s "
                f"(timeout={self.timeout}s): {e}"
            ) from e

        # Fallback: endpoint rejected streaming outright — issue a normal
        # non-streaming request and yield the full answer as one chunk.
        answer = self.ask(question)
        if answer:
            yield answer
            
    def health_check(self) -> bool:
        """Return True only if the endpoint reports state.ready == READY."""
        try:
            response = self._request_with_retry("GET", self._status_url)
        except (AnalystClientError, TimeoutError):
            return False

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return False

        state = body.get("state") or {}
        ready = state.get("ready")
        return str(ready).upper() == "READY"