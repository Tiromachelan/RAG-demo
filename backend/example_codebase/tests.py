"""Unit tests for the calculator and utils modules."""
import pytest
from calculator import Calculator
from utils import is_prime, clamp, percent, factorial, average, round_to


class TestCalculator:
    def setup_method(self):
        self.calc = Calculator()

    def test_add(self):
        assert self.calc.add(2, 3) == 5

    def test_subtract(self):
        assert self.calc.subtract(10, 4) == 6

    def test_multiply(self):
        assert self.calc.multiply(3, 7) == 21

    def test_divide(self):
        assert self.calc.divide(10, 2) == 5.0

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(5, 0)

    def test_power(self):
        assert self.calc.power(2, 10) == 1024

    def test_history(self):
        self.calc.add(1, 2)
        self.calc.multiply(3, 4)
        assert len(self.calc.get_history()) == 2

    def test_clear_history(self):
        self.calc.add(1, 1)
        self.calc.clear_history()
        assert self.calc.get_history() == []


class TestUtils:
    def test_is_prime(self):
        assert is_prime(7) is True
        assert is_prime(4) is False
        assert is_prime(1) is False

    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_percent(self):
        assert percent(25, 100) == 25.0
        with pytest.raises(ValueError):
            percent(1, 0)

    def test_factorial(self):
        assert factorial(5) == 120
        assert factorial(0) == 1

    def test_average(self):
        assert average([1, 2, 3, 4, 5]) == 3.0
        with pytest.raises(ValueError):
            average([])

    def test_round_to(self):
        assert round_to(3.14159, 2) == 3.14
