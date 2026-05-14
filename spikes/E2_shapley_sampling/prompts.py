"""Handcrafted prompt set for the E2 Shapley sampling spike.

Nine prompts total: three RAG-style, three instruction-following, three
reasoning. Each category contains one short (~1k tokens), one medium
(~4k tokens), and one long (~16k tokens) entry.

Token estimates use the four-characters-per-token heuristic. The
notebook reports the actual Anthropic token counts after the first run.

The expected_topic field is a short string the notebook uses to score
"did the answer stay on topic" without requiring a downstream judge.

This file is intentionally self-contained: no external data, no
templating.
"""

from __future__ import annotations

from typing import TypedDict

# A repeating block used to inflate prompt length while keeping the
# semantic structure recognizable. The block is plausible context text;
# the prompts that include it cite it explicitly so the model has a
# reason to engage with it rather than skip it.
_FILLER_BLOCK = (
    "The historical record shows that small operational changes compound "
    "over long horizons. Documented audit trails support pattern recognition "
    "across decades of program activity. The relevant case studies are "
    "indexed in the appendix under the standard taxonomy. Replication "
    "studies confirm the primary findings with effect sizes consistent "
    "with the original reports. Cross-validation across independent teams "
    "is documented in the supplementary materials. "
)


def _grow(base: str, target_chars: int) -> str:
    """Pad ``base`` with the filler block until it exceeds ``target_chars``.

    The resulting string ends with the filler text; callers append the
    final question after the filler so the model sees the question last.
    """
    out = base
    while len(out) < target_chars:
        out += _FILLER_BLOCK
    return out


class PromptEntry(TypedDict):
    """One record in the spike's prompt set."""

    id: str
    length_bucket: str
    category: str
    prompt: str
    expected_topic: str


_RAG_1K = (
    "Context document:\n\n"
    "Apollo 11 launched on July 16, 1969 from Kennedy Space Center. The "
    "crew consisted of Neil Armstrong, Buzz Aldrin, and Michael Collins. "
    "Armstrong and Aldrin landed in the Sea of Tranquility on July 20. "
    "Collins remained in lunar orbit aboard the Columbia command module. "
    "The mission returned to Earth on July 24, splashing down in the "
    "Pacific Ocean. The samples returned included approximately 21.5 "
    "kilograms of lunar material. "
)

_RAG_4K_BASE = (
    "Context document:\n\n"
    "The Voyager program consists of two robotic probes, Voyager 1 and "
    "Voyager 2, launched by NASA in 1977. Voyager 1 was launched on "
    "September 5, 1977 and Voyager 2 on August 20, 1977. Both probes were "
    "designed to take advantage of a planetary alignment that occurs once "
    "every 175 years. Voyager 2 remains the only spacecraft to have "
    "visited Uranus and Neptune. Voyager 1 crossed into interstellar "
    "space in August 2012. Voyager 2 followed in November 2018. "
)

_RAG_16K_BASE = (
    "Context document:\n\n"
    "The Cassini-Huygens mission was a joint NASA, ESA, and ASI mission "
    "to study Saturn and its system. The spacecraft launched on October "
    "15, 1997 and arrived at Saturn on July 1, 2004. The Huygens probe "
    "separated from the orbiter on December 25, 2004 and landed on Titan "
    "on January 14, 2005. The orbiter continued to operate until "
    "September 15, 2017, when it was deliberately deorbited into Saturn's "
    "atmosphere. Mission highlights include the discovery of plumes on "
    "Enceladus, detailed mapping of Titan's surface, and identification "
    "of new Saturnian moons. "
)

_INSTR_1K = (
    "Write a Python function named compute_factorial that returns the "
    "factorial of a non-negative integer n. The function must raise "
    "ValueError on negative inputs. Use an iterative implementation, "
    "not recursion. Include a one-line docstring."
)

_INSTR_4K_BASE = (
    "You are a code-review assistant. Review the following pull request "
    "description and produce a structured report with three sections: "
    "Strengths, Concerns, Suggested Changes. Be concrete and reference "
    "specific items in the description. Do not invent code that is not "
    "described.\n\nPull request description:\n\n"
    "Title: Add retry logic to the upload pipeline.\n\n"
    "This PR adds exponential backoff with jitter to the S3 upload path. "
    "The new RetryPolicy class lives in src/uploader/retry.py and is "
    "wired into Uploader.put(). Maximum retries is 5; initial delay is "
    "100ms; jitter is uniform in [0, 50ms]. Tests cover the happy path, "
    "the all-retries-exhausted case, and the partial-success case. "
    "Coverage on src/uploader/ is 92 percent. The integration test runs "
    "against a localstack S3 mock. "
)

_INSTR_16K_BASE = (
    "You are reviewing a long technical document. Produce a 200-word "
    "summary that names the document's central claim, lists three "
    "supporting points, and identifies one gap or weakness. Stay "
    "grounded in the document; do not invent claims.\n\nDocument:\n\n"
    "This report describes a deployment of distributed tracing across "
    "the order-management platform. The platform spans 47 services, "
    "deployed on Kubernetes across three regions. Tracing was added "
    "incrementally over six months. The team chose OpenTelemetry over "
    "vendor-specific SDKs to preserve portability. Sampling is head-based "
    "at the edge with a 10 percent base rate; tail-based sampling at the "
    "collector promotes traces flagged by error spans or high-latency "
    "outliers. Storage uses ClickHouse with a 14-day retention. The "
    "primary outcome was a 40 percent reduction in mean time to detect "
    "for cross-service incidents. The team reports two open challenges: "
    "cardinality control on span attributes, and the propagation of "
    "tracing context across asynchronous message queues. "
)

_REASON_1K = (
    "A train leaves city A at 9:00 AM traveling at 60 km/h. A second "
    "train leaves city B at 10:00 AM traveling at 80 km/h toward city A. "
    "The two cities are 280 km apart along the same straight track. At "
    "what time do the trains meet, and how far from city A is the "
    "meeting point? Show your reasoning step by step."
)

_REASON_4K_BASE = (
    "Consider the following scenario and answer the question that "
    "follows.\n\nScenario:\n\n"
    "A small library uses a deterministic check-out policy. Patrons may "
    "borrow at most 5 books at a time. The maximum loan period is 21 "
    "days. Each renewal extends the loan period by 14 days, up to a "
    "maximum of two renewals per book. A book on hold for another "
    "patron may not be renewed. Patrons with overdue books may not "
    "borrow new books until the overdue items are returned. Late fees "
    "accrue at 25 cents per day per book, capped at 10 dollars per "
    "book. The library forgives fees if the patron donates a "
    "replacement book of equivalent value. "
)

_REASON_16K_BASE = (
    "You are presented with a long policy document. Identify any "
    "internal inconsistencies and explain them. Stay grounded in the "
    "text; do not invent rules.\n\nPolicy document:\n\n"
    "All employees must complete annual compliance training by December "
    "31. Training assignments are issued in the prior January. Managers "
    "are responsible for verifying completion of their reports. "
    "Employees who join after July 1 have until March 31 of the "
    "following year to complete the training. The annual deadline is "
    "December 15 for employees in the EU region due to local regulatory "
    "filings. Late completion incurs a 1 percent reduction in the "
    "year-end performance bonus, calculated against the base bonus. "
    "Employees who have not completed training by the deadline are "
    "marked non-compliant and lose access to customer data systems "
    "until completion. "
)


PROMPTS: list[PromptEntry] = [
    {
        "id": "rag-1k",
        "length_bucket": "1k",
        "category": "rag",
        "prompt": _RAG_1K
        + "\n\nQuestion: How many kilograms of lunar material did Apollo 11 return?",
        "expected_topic": "21.5 kilograms",
    },
    {
        "id": "rag-4k",
        "length_bucket": "4k",
        "category": "rag",
        "prompt": _grow(_RAG_4K_BASE, 4000 * 4)
        + "\n\nQuestion: When did Voyager 1 cross into interstellar space?",
        "expected_topic": "August 2012",
    },
    {
        "id": "rag-16k",
        "length_bucket": "16k",
        "category": "rag",
        "prompt": _grow(_RAG_16K_BASE, 16000 * 4)
        + "\n\nQuestion: On what date did the Huygens probe land on Titan?",
        "expected_topic": "January 14, 2005",
    },
    {
        "id": "instr-1k",
        "length_bucket": "1k",
        "category": "instruction",
        "prompt": _INSTR_1K,
        "expected_topic": "iterative factorial with ValueError",
    },
    {
        "id": "instr-4k",
        "length_bucket": "4k",
        "category": "instruction",
        "prompt": _grow(_INSTR_4K_BASE, 4000 * 4) + "\n\nProduce the review report now.",
        "expected_topic": "structured review with Strengths, Concerns, Suggested Changes",
    },
    {
        "id": "instr-16k",
        "length_bucket": "16k",
        "category": "instruction",
        "prompt": _grow(_INSTR_16K_BASE, 16000 * 4) + "\n\nProduce the 200-word summary now.",
        "expected_topic": "distributed tracing deployment summary",
    },
    {
        "id": "reason-1k",
        "length_bucket": "1k",
        "category": "reasoning",
        "prompt": _REASON_1K,
        "expected_topic": "trains meet at 11:00 AM, 120 km from city A",
    },
    {
        "id": "reason-4k",
        "length_bucket": "4k",
        "category": "reasoning",
        "prompt": _grow(_REASON_4K_BASE, 4000 * 4)
        + "\n\nQuestion: A patron has three overdue books. The patron "
        "wants to renew a fourth book that is on hold for another "
        "patron. Apply the policy and state the outcome with reasoning.",
        "expected_topic": "cannot renew, two policy reasons",
    },
    {
        "id": "reason-16k",
        "length_bucket": "16k",
        "category": "reasoning",
        "prompt": _grow(_REASON_16K_BASE, 16000 * 4)
        + "\n\nQuestion: List any internal inconsistencies in the "
        "policy and explain each in two sentences.",
        "expected_topic": "deadline conflict for EU and late-joiner cases",
    },
]


def by_id(prompt_id: str) -> PromptEntry:
    """Look up a prompt by id.

    Args:
        prompt_id: One of the ids defined in :data:`PROMPTS`.

    Returns:
        The matching :class:`PromptEntry`.

    Raises:
        KeyError: If no prompt with the given id exists.
    """
    for entry in PROMPTS:
        if entry["id"] == prompt_id:
            return entry
    raise KeyError(f"no prompt with id={prompt_id!r}")
