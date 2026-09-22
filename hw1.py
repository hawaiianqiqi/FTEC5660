#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    import os

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    model_name = "deepseek-v4-flash-vision-exp"

    system_template = (
        "You are a meticulous receipt-reading assistant. Each request shows you exactly "
        "one supermarket receipt image. You reply with a single strict JSON object and "
        "nothing else. You never invent a value: if a figure is genuinely unreadable, use "
        "null, and you never explain yourself."
    )

    # The JSON schema is deliberately kept out of the templates below: it is passed in as
    # the ``instruction`` variable. ChatPromptTemplate parses curly braces in *templates*
    # as placeholders, so literal braces must never appear here.
    human_template = "Receipt file: {filename}\n\n{instruction}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            (
                "human",
                [
                    {"type": "text", "text": human_template},
                    {"type": "image_url", "image_url": {"url": "{image}"}},
                ],
            ),
        ]
    )

    base_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": 0,
        "max_retries": 2,
        "timeout": 120,
    }

    # ``deepseek-v4-flash-vision-exp`` is served through a course endpoint, so allow the
    # base URL to be supplied by the environment. Its exact keyword differs across
    # langchain-deepseek releases, hence the fallback chain.
    api_base = (
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )
    candidates: list[dict[str, Any]] = []
    if api_base:
        candidates.append({**base_kwargs, "api_base": api_base})
        candidates.append({**base_kwargs, "base_url": api_base})
    else:
        candidates.append(dict(base_kwargs))
    # Last resort: some proxies reject the optional knobs entirely.
    candidates.append({"model": model_name})

    model = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            model = ChatDeepSeek(**candidate)
            break
        except TypeError as exc:
            last_error = exc
    if model is None:
        raise last_error or RuntimeError("could not initialise ChatDeepSeek")

    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    import json
    import sys

    samples_per_receipt = 2  # independent readings required to agree
    max_rounds = 2  # a third reading is only requested to break a tie
    max_concurrency = 12
    tolerance = Decimal("0.06")

    schema = (
        "{\n"
        '  "items": [{"name": "<item text>", "price": <positive number>}],\n'
        '  "discount_lines": [{"label": "<discount text>", "amount": <positive number>}],\n'
        '  "subtotal": <number>,\n'
        '  "rounding": <number>,\n'
        '  "paid": <number>\n'
        "}"
    )
    rules = (
        "Rules:\n"
        '1. "items": every purchased line that has a positive price. List all of them and '
        "skip none.\n"
        '2. "discount_lines": every line that REDUCES the bill - promotions, coupons, member '
        'discounts, app discounts, packaging-damage deductions and percentage discounts such '
        'as "5% OFF". Give each amount as a POSITIVE number (drop the minus sign). List all '
        "of them.\n"
        '3. "subtotal": the value printed on the SUBTOTAL line. It is already after discounts.\n'
        '4. "rounding": the value printed on the ROUNDING line. It may be negative. Use 0 when '
        "the receipt has no rounding line.\n"
        '5. "paid": the final amount actually paid - the last line of the receipt, i.e. the '
        "payment-method line such as OCTOPUS, CASH or EPS. This is the value AFTER rounding.\n"
        '6. Never put the ROUNDING line inside "discount_lines".\n'
        "7. Plain decimal numbers only: no currency symbol, no thousands separator, no "
        'percent signs. Only "rounding" may be negative.\n'
        '8. Check your own arithmetic before answering: sum(items.price) - '
        'sum(discount_lines.amount) must equal "subtotal", and "subtotal" + "rounding" must '
        'equal "paid". If either check fails, re-read the image and fix it.\n'
        "9. Reply with the JSON object alone - no prose, no markdown code fences, no comments."
    )
    base_instruction = (
        "Read the receipt image and return one JSON object with exactly this shape:\n\n"
        + schema
        + "\n\n"
        + rules
    )

    def instruction_for(note: str, previous: str) -> str:
        if not note:
            return base_instruction
        extra = (
            "\n\nCorrection needed: a previous attempt on this same image was rejected "
            "because " + note + "."
        )
        if previous:
            extra += "\nThat attempt returned:\n" + previous[:1500]
        extra += "\nRe-read the image very carefully and fix exactly that problem."
        return base_instruction + extra

    def parse_object(text: str) -> dict[str, Any] | None:
        """Recover the JSON object from a model reply that may still be fenced."""
        if not text:
            return None
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        candidates = [cleaned]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            candidates.append(cleaned[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def money(value: Any) -> Decimal | None:
        """Coerce a model-supplied amount into an exact Decimal, or None."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float, str)):
            raw = str(value)
            for token in ("HK$", "HKD", "$", ",", " "):
                raw = raw.replace(token, "")
            raw = raw.strip().strip("()")
            if not raw:
                return None
            try:
                return Decimal(raw)
            except InvalidOperation:
                return None
        return None

    def line_total(data: dict[str, Any], key: str, field: str) -> tuple[Decimal, int]:
        """Sum a list of records, tolerating rows shaped as dicts or bare numbers."""
        rows = data.get(key)
        total = Decimal("0")
        counted = 0
        if isinstance(rows, list):
            for row in rows:
                amount = money(row.get(field) if isinstance(row, dict) else row)
                if amount is not None:
                    total += abs(amount)
                    counted += 1
        return total, counted

    def problems_with(data: Any) -> list[str]:
        """Return the reasons this extraction cannot be trusted (empty means good)."""
        if not isinstance(data, dict):
            return ["the reply was not a JSON object"]
        found: list[str] = []
        subtotal = money(data.get("subtotal"))
        rounding = money(data.get("rounding"))
        paid = money(data.get("paid"))
        items_sum, item_count = line_total(data, "items", "price")
        discount_sum, _ = line_total(data, "discount_lines", "amount")

        if item_count == 0:
            found.append("no priced item lines were listed")
        if subtotal is None:
            found.append("the SUBTOTAL value was missing or unreadable")
        if paid is None:
            found.append("the final paid amount was missing or unreadable")

        if subtotal is not None and item_count:
            items_less_discounts = items_sum - discount_sum
            if abs(items_less_discounts - subtotal) > tolerance:
                found.append(
                    "sum(items) - sum(discounts) is "
                    + str(items_less_discounts)
                    + " but SUBTOTAL reads "
                    + str(subtotal)
                )
        if subtotal is not None and paid is not None:
            rounding_value = rounding if rounding is not None else Decimal("0")
            if abs(subtotal + rounding_value - paid) > tolerance:
                found.append(
                    "SUBTOTAL + ROUNDING is "
                    + str(subtotal + rounding_value)
                    + " but the paid amount reads "
                    + str(paid)
                )
        return found

    def is_fatal(exc: BaseException) -> bool:
        """Auth, permission and missing-model errors will never fix themselves."""
        name = type(exc).__name__
        if any(token in name for token in ("Authentication", "Permission", "NotFound")):
            return True
        return getattr(exc, "status_code", None) in (401, 403, 404)

    def answer_key(data: dict[str, Any]) -> tuple[Any, ...]:
        """This receipt's contribution to both answers, for comparing two readings."""
        subtotal = money(data.get("subtotal"))
        rounding = money(data.get("rounding"))
        paid = money(data.get("paid"))
        if paid is None and subtotal is not None:
            paid = subtotal + (rounding if rounding is not None else Decimal("0"))
        discount_sum, _ = line_total(data, "discount_lines", "amount")
        return (paid, subtotal + discount_sum if subtotal is not None else None, subtotal)

    def settle(state: dict[str, Any]) -> dict[str, Any] | None:
        """Return the agreed reading, or None when another sample is still needed.

        The model is not deterministic even at temperature 0, and a misread can
        stay self-consistent: dropping one item line and one discount line of the
        same value keeps ``sum(items) - sum(discounts)`` intact while understating
        the no-discount total. Agreement between two independent readings is what
        catches that; validation alone cannot.
        """
        valid = [sample["data"] for sample in state["samples"] if sample["valid"]]
        exhausted = state["stop"] or state["rounds"] >= max_rounds

        if not valid:
            if not exhausted:
                return None
            for sample in reversed(state["samples"]):
                if sample["data"] is not None:
                    return sample["data"]
            return {}

        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for data in valid:
            groups.setdefault(answer_key(data), []).append(data)
        for group in groups.values():
            if len(group) >= samples_per_receipt:
                return group[0]
        if not exhausted:
            return None
        return max(groups.values(), key=len)[0]

    states = [
        {
            "path": path,
            "image": image_data_url(path),
            "samples": [],
            "note": "",
            "previous": "",
            "rounds": 0,
            "stop": False,
            "data": None,
        }
        for path in images
    ]

    pending = list(range(len(states)))
    for round_index in range(max_rounds):
        if not pending:
            break
        # Round one reads every receipt twice, in the same parallel batch, so the
        # extra sample costs latency only. Later rounds re-read just the stragglers.
        plan: list[int] = []
        for index in pending:
            plan.extend([index] * (samples_per_receipt if round_index == 0 else 1))
        batch_inputs = [
            {
                "filename": states[index]["path"].name,
                "image": states[index]["image"],
                "instruction": instruction_for(
                    states[index]["note"], states[index]["previous"]
                ),
            }
            for index in plan
        ]
        try:
            outputs = chain.batch(
                batch_inputs,
                config={"max_concurrency": max_concurrency},
                return_exceptions=True,
            )
        except Exception as exc:  # noqa: BLE001 - a broken batch must never kill the run
            outputs = [exc] * len(batch_inputs)
        if not isinstance(outputs, list) or len(outputs) != len(batch_inputs):
            outputs = [outputs] * len(batch_inputs)

        for index in set(plan):
            states[index]["rounds"] += 1

        for index, output in zip(plan, outputs):
            state = states[index]
            if isinstance(output, BaseException):
                state["samples"].append({"data": None, "valid": False})
                if is_fatal(output):
                    state["stop"] = True
                    state["note"] = "the model call failed (" + type(output).__name__ + ")"
                continue
            text = response_text(output)
            data = parse_object(text)
            found = (
                problems_with(data)
                if data is not None
                else ["the reply contained no parsable JSON object"]
            )
            state["samples"].append({"data": data, "valid": not found})
            if not found:
                state["note"] = ""
                state["previous"] = ""
            else:
                state["previous"] = text
                state["note"] = "; ".join(found)

        still_pending: list[int] = []
        for index in pending:
            state = states[index]
            chosen = settle(state)
            if chosen is None:
                # Every reading so far was individually valid but they disagree.
                if not state["note"]:
                    state["note"] = (
                        "an independent read of this same receipt returned different figures - "
                        "re-read the image and double-check every item price and every discount line"
                    )
                    state["previous"] = ""
                still_pending.append(index)
                continue
            state["data"] = chosen
        pending = still_pending

    total_paid = Decimal("0")
    total_without_discounts = Decimal("0")
    unusable: list[str] = []

    for state in states:
        data = state["data"] if isinstance(state["data"], dict) else {}
        subtotal = money(data.get("subtotal"))
        rounding = money(data.get("rounding"))
        paid = money(data.get("paid"))
        discount_sum, _ = line_total(data, "discount_lines", "amount")
        items_sum, item_count = line_total(data, "items", "price")

        if paid is None and subtotal is not None:
            paid = subtotal + (rounding if rounding is not None else Decimal("0"))

        if subtotal is not None:
            # SUBTOTAL plus every discount line added back as a positive number.
            without_discounts = subtotal + discount_sum
        elif item_count:
            # Equivalent route: the sum of the original positive item prices.
            without_discounts = items_sum
        else:
            without_discounts = None

        if paid is None or without_discounts is None:
            unusable.append(state["path"].name)

        total_paid += paid if paid is not None else Decimal("0")
        total_without_discounts += (
            without_discounts if without_discounts is not None else Decimal("0")
        )

    for state in states:
        if state["note"]:
            print(
                "warning: {} was not fully verified ({})".format(
                    state["path"].name, state["note"]
                ),
                file=sys.stderr,
            )
    if unusable:
        print(
            "warning: no usable extraction for " + ", ".join(unusable),
            file=sys.stderr,
        )

    # Formatting happens in Python so each response carries exactly one amount.
    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discounts:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
