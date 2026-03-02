"""Utility functions for the calculator demo project."""
import math
from typing import Union

Number = Union[int, float]


def is_prime(n: int) -> bool:
    """Return True if n is a prime number."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def clamp(value: Number, min_val: Number, max_val: Number) -> Number:
    """Clamp value between min_val and max_val."""
    return max(min_val, min(value, max_val))


def percent(part: Number, whole: Number) -> float:
    """Return what percentage part is of whole."""
    if whole == 0:
        raise ValueError("Whole cannot be zero")
    return (part / whole) * 100.0


def factorial(n: int) -> int:
    """Return n! (factorial). n must be a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return math.factorial(n)


def average(numbers: list[Number]) -> float:
    """Return the arithmetic mean of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot compute average of empty list")
    return sum(numbers) / len(numbers)


def round_to(value: float, decimals: int = 2) -> float:
    """Round value to the given number of decimal places."""
    return round(value, decimals)
