"""
Calculator tools for mathematical operations.

All tools are decorated with @tool for LangChain integration
and automatic LangSmith tracing.
"""

from __future__ import annotations

import math
from datetime import datetime

from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: A mathematical expression string (e.g., '2 + 3 * 4', '100 / 7').

    Returns:
        The result of the calculation as a formatted string.
    """
    # Whitelist of safe operations
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "len": len,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "ceil": math.ceil, "floor": math.floor,
        "pi": math.pi, "e": math.e,
    }

    try:
        # Remove potentially dangerous characters
        cleaned = expression.replace("__", "").replace("import", "").replace("exec", "")
        result = eval(cleaned, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"{result:,.4f}" if isinstance(result, float) else f"{result:,}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


@tool
def percentage(value: float, total: float) -> str:
    """Calculate what percentage a value is of a total.

    Args:
        value: The part value.
        total: The total/whole value.

    Returns:
        The percentage formatted as a string.
    """
    if total == 0:
        return "Error: Cannot calculate percentage with total of 0."
    pct = (value / total) * 100
    return f"{value:,.2f} is {pct:.2f}% of {total:,.2f}"


@tool
def average(numbers: list[float]) -> str:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        numbers: List of numbers to average.

    Returns:
        The average formatted as a string.
    """
    if not numbers:
        return "Error: Cannot calculate average of empty list."
    avg = sum(numbers) / len(numbers)
    return f"Average of {len(numbers)} values: {avg:,.2f}"


@tool
def growth_rate(old_value: float, new_value: float) -> str:
    """Calculate the growth rate between two values.

    Args:
        old_value: The original/starting value.
        new_value: The final/ending value.

    Returns:
        Growth rate as a percentage string.
    """
    if old_value == 0:
        return "Error: Cannot calculate growth rate from a starting value of 0."
    rate = ((new_value - old_value) / old_value) * 100
    direction = "increase" if rate > 0 else "decrease"
    return (
        f"Growth from {old_value:,.2f} to {new_value:,.2f}: "
        f"{abs(rate):.2f}% {direction}"
    )


@tool
def revenue_growth(revenues: list[float]) -> str:
    """Calculate period-over-period revenue growth rates.

    Args:
        revenues: List of revenue values in chronological order.

    Returns:
        Growth rates for each period transition.
    """
    if len(revenues) < 2:
        return "Error: Need at least 2 revenue values to calculate growth."

    results = []
    for i in range(1, len(revenues)):
        if revenues[i - 1] == 0:
            results.append(f"Period {i}: Cannot calculate (previous value is 0)")
            continue
        rate = ((revenues[i] - revenues[i - 1]) / revenues[i - 1]) * 100
        results.append(
            f"Period {i-1}→{i}: ${revenues[i-1]:,.2f} → ${revenues[i]:,.2f} = {rate:+.2f}%"
        )

    overall = ((revenues[-1] - revenues[0]) / revenues[0]) * 100 if revenues[0] != 0 else 0
    results.append(f"Overall growth: {overall:+.2f}%")
    avg_growth = overall / (len(revenues) - 1)
    results.append(f"Average growth per period: {avg_growth:+.2f}%")

    return "\n".join(results)


@tool
def date_difference(date1: str, date2: str) -> str:
    """Calculate the number of days between two dates.

    Args:
        date1: First date in YYYY-MM-DD format.
        date2: Second date in YYYY-MM-DD format.

    Returns:
        The difference in days, weeks, and months.
    """
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        delta = abs((d2 - d1).days)
        weeks = delta / 7
        months = delta / 30.44  # Average days per month

        return (
            f"Difference between {date1} and {date2}:\n"
            f"  {delta} days\n"
            f"  {weeks:.1f} weeks\n"
            f"  {months:.1f} months"
        )
    except ValueError as e:
        return f"Error parsing dates: {str(e)}. Use YYYY-MM-DD format."


@tool
def token_counter(text: str) -> str:
    """Estimate the number of tokens in a text string.

    Uses a simple approximation: ~4 characters per token for English text.

    Args:
        text: The text to count tokens for.

    Returns:
        Estimated token count.
    """
    char_count = len(text)
    word_count = len(text.split())
    estimated_tokens = max(char_count // 4, word_count)

    return (
        f"Text statistics:\n"
        f"  Characters: {char_count:,}\n"
        f"  Words: {word_count:,}\n"
        f"  Estimated tokens: ~{estimated_tokens:,}"
    )


# List of all calculator tools for easy import
CALCULATOR_TOOLS = [
    calculate,
    percentage,
    average,
    growth_rate,
    revenue_growth,
    date_difference,
    token_counter,
]
