"""Simple calculator module — used as the demo codebase for the RAG coding agent."""


class Calculator:
    """A basic calculator that supports add, subtract, multiply, divide, and history."""

    def __init__(self):
        self.history = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def power(self, base: float, exponent: float) -> float:
        result = base ** exponent
        self.history.append(f"{base} ** {exponent} = {result}")
        return result

    def clear_history(self):
        self.history = []

    def get_history(self) -> list[str]:
        return list(self.history)
