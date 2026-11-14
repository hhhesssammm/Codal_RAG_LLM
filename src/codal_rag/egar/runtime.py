"""Per-question budgets. Timeout workers are daemon/read-only and never retried."""
from __future__ import annotations

import queue
import threading
import time

from codal_rag.egar.evidence import token_upper_bound


class BudgetExceeded(RuntimeError):
    pass


class OperationTimeout(BudgetExceeded):
    pass


class QuestionBudget:
    def __init__(self, config, *, clock=time.monotonic):
        self.config, self.clock = config, clock
        self.started = clock()
        self.llm_calls = 0
        self.input_upper_bound = self.output_upper_bound = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.usage_complete = True

    def remaining(self):
        return self.config.timeout_seconds - (self.clock() - self.started)

    def run(self, operation):
        timeout = min(self.config.call_timeout_seconds, self.remaining())
        if timeout <= 0:
            raise OperationTimeout("Question time budget exhausted")
        mailbox = queue.Queue(maxsize=1)
        def worker():
            try:
                mailbox.put((True, operation()))
            except Exception as exc:
                mailbox.put((False, exc))
        threading.Thread(target=worker, daemon=True).start()
        try:
            success, result = mailbox.get(timeout=timeout)
        except queue.Empty:
            raise OperationTimeout("Operation timed out; no retry was issued") from None
        if not success:
            raise result
        return result

    def metrics(self):
        exact = self.usage_complete
        cost = None
        if exact and (self.config.input_cost_per_million or self.config.output_cost_per_million):
            cost = (self.prompt_tokens * self.config.input_cost_per_million +
                    self.completion_tokens * self.config.output_cost_per_million) / 1_000_000
        return {"latency_seconds": self.clock() - self.started, "llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens if exact else None,
                "completion_tokens": self.completion_tokens if exact else None,
                "total_inference_tokens": self.prompt_tokens + self.completion_tokens if exact else None,
                "input_token_upper_bound": self.input_upper_bound,
                "output_token_upper_bound": self.output_upper_bound,
                "estimated_cost": cost, "cost_basis": "configured_price_per_million_provider_tokens",
                "token_budget_counter": "utf8_byte_upper_bound"}


class BudgetedLLM:
    def __init__(self, client, budget):
        self.client, self.budget = client, budget

    def generate_json(self, system_prompt, user_prompt):
        import json
        b = self.budget
        for attempt in range(b.config.max_retries + 1):
            if b.llm_calls >= b.config.max_llm_calls:
                raise BudgetExceeded("Maximum LLM calls reached")
            b.llm_calls += 1
            b.input_upper_bound += token_upper_bound(system_prompt + user_prompt)
            try:
                result = b.run(lambda: self.client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt))
                b.output_upper_bound += token_upper_bound(json.dumps(result, ensure_ascii=False))
                usage = getattr(self.client, "last_usage", None)
                if usage and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                    b.prompt_tokens += usage["prompt_tokens"]
                    b.completion_tokens += usage["completion_tokens"]
                else:
                    b.usage_complete = False
                return result
            except OperationTimeout:
                b.usage_complete = False
                raise
            except (ValueError, RuntimeError, OSError):
                b.usage_complete = False
                if attempt == b.config.max_retries:
                    raise
