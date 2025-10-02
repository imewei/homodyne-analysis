"""
Function Composition Framework for Homodyne Analysis
====================================================

This module provides function composition patterns and utilities for improved
readability, testability, and modularity in the homodyne analysis package.

Implements functional programming patterns including:
- Function composition and pipelining
- Higher-order functions for data processing
- Monadic error handling patterns
- Composable validation chains
- Functional configuration management

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import TypeVar

logger = logging.getLogger(__name__)

# Type variables for generic function composition
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Result(Generic[T]):
    """
    Monadic Result type for composable error handling.

    This class provides a functional approach to error handling that allows
    chaining operations without explicit error checking at each step.
    """

    def __init__(self, value: T | None = None, error: Exception | None = None):
        self._value = value
        self._error = error
        self._is_success = error is None

    @classmethod
    def success(cls, value: T) -> Result[T]:
        """Create a successful result."""
        return cls(value=value)

    @classmethod
    def failure(cls, error: Exception) -> Result[T]:
        """Create a failed result."""
        return cls(error=error)

    @property
    def is_success(self) -> bool:
        """Check if the result is successful."""
        return self._is_success

    @property
    def is_failure(self) -> bool:
        """Check if the result is a failure."""
        return not self._is_success

    @property
    def value(self) -> T:
        """Get the value if successful, raise exception if failed."""
        if self._is_success:
            return self._value
        raise self._error

    @property
    def error(self) -> Exception | None:
        """Get the error if failed, None if successful."""
        return self._error

    def map(self, func: Callable[[T], U]) -> Result[U]:
        """Apply function to value if successful, otherwise propagate error."""
        if self._is_success:
            try:
                return Result.success(func(self._value))
            except Exception as e:
                return Result.failure(e)
        return Result.failure(self._error)

    def flat_map(self, func: Callable[[T], Result[U]]) -> Result[U]:
        """Apply function that returns Result, avoiding nested Results."""
        if self._is_success:
            try:
                return func(self._value)
            except Exception as e:
                return Result.failure(e)
        return Result.failure(self._error)

    def filter(
        self, predicate: Callable[[T], bool], error_msg: str = "Filter failed"
    ) -> Result[T]:
        """Filter value with predicate, fail if predicate returns False."""
        if self._is_success:
            try:
                if predicate(self._value):
                    return self
                return Result.failure(ValueError(error_msg))
            except Exception as e:
                return Result.failure(e)
        return Result.failure(self._error)

    def or_else(self, default: T) -> T:
        """Get value if successful, otherwise return default."""
        return self._value if self._is_success else default

    def or_else_get(self, func: Callable[[], T]) -> T:
        """Get value if successful, otherwise call function to get default."""
        return self._value if self._is_success else func()


def compose(*functions: Callable) -> Callable:
    """
    Compose functions from right to left: compose(f, g, h)(x) = f(g(h(x))).

    Parameters
    ----------
    *functions : Callable
        Functions to compose

    Returns
    -------
    Callable
        Composed function

    Examples
    --------
    >>> add_one = lambda x: x + 1
    >>> multiply_two = lambda x: x * 2
    >>> subtract_three = lambda x: x - 3
    >>> composed = compose(add_one, multiply_two, subtract_three)
    >>> composed(10)  # add_one(multiply_two(subtract_three(10))) = add_one(multiply_two(7)) = add_one(14) = 15
    15
    """
    if not functions:
        return lambda x: x

    def composed_function(x):
        result = x
        for func in reversed(functions):
            result = func(result)
        return result

    return composed_function


def pipe(*functions: Callable) -> Callable:
    """
    Pipe functions from left to right: pipe(f, g, h)(x) = h(g(f(x))).

    Parameters
    ----------
    *functions : Callable
        Functions to pipe

    Returns
    -------
    Callable
        Piped function

    Examples
    --------
    >>> add_one = lambda x: x + 1
    >>> multiply_two = lambda x: x * 2
    >>> subtract_three = lambda x: x - 3
    >>> piped = pipe(add_one, multiply_two, subtract_three)
    >>> piped(10)  # subtract_three(multiply_two(add_one(10))) = subtract_three(multiply_two(11)) = subtract_three(22) = 19
    19
    """
    if not functions:
        return lambda x: x

    def piped_function(x):
        result = x
        for func in functions:
            result = func(result)
        return result

    return piped_function


def curry(func: Callable) -> Callable:
    """
    Convert a function to its curried version.

    Parameters
    ----------
    func : Callable
        Function to curry

    Returns
    -------
    Callable
        Curried function

    Examples
    --------
    >>> def add(x, y, z):
    ...     return x + y + z
    >>> curried_add = curry(add)
    >>> add_5_and_3 = curried_add(5)(3)
    >>> result = add_5_and_3(2)  # 5 + 3 + 2 = 10
    10
    """
    sig = inspect.signature(func)
    param_count = len(sig.parameters)

    def curried(*args):
        if len(args) >= param_count:
            return func(*args[:param_count])
        return lambda *more_args: curried(*(args + more_args))

    return curried


def partial_right(func: Callable, *args, **kwargs) -> Callable:
    """
    Partial application from the right (fix rightmost arguments).

    Parameters
    ----------
    func : Callable
        Function to partially apply
    *args
        Arguments to fix from the right
    **kwargs
        Keyword arguments to fix

    Returns
    -------
    Callable
        Partially applied function
    """

    def partial_func(*left_args, **left_kwargs):
        combined_kwargs = {**kwargs, **left_kwargs}
        return func(*(left_args + args), **combined_kwargs)

    return partial_func


def retry_on_failure(max_attempts: int = 3, delay: float = 0.1) -> Callable:
    """
    Decorator for retrying function calls on failure.

    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts
    delay : float
        Delay between attempts in seconds

    Returns
    -------
    Callable
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time

            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.debug(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}"
                        )
                        time.sleep(delay * (2**attempt))  # Exponential backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper

    return decorator


def memoize(func: Callable) -> Callable:
    """
    Memoization decorator for caching function results.

    Parameters
    ----------
    func : Callable
        Function to memoize

    Returns
    -------
    Callable
        Memoized function
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    wrapper.cache = cache
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


@dataclass
class Pipeline:
    """
    Functional pipeline for composing operations with error handling.

    This class provides a fluent interface for building complex data processing
    pipelines while maintaining composability and testability.
    """

    steps: list[Callable] = None
    error_handler: Callable[[Exception], Any] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []

    def add_step(self, func: Callable, *args, **kwargs) -> Pipeline:
        """Add a processing step to the pipeline."""
        if args or kwargs:
            step = functools.partial(func, *args, **kwargs)
        else:
            step = func

        return Pipeline(steps=self.steps + [step], error_handler=self.error_handler)

    def add_validation(
        self, predicate: Callable[[Any], bool], error_msg: str
    ) -> Pipeline:
        """Add a validation step to the pipeline."""

        def validate(data):
            if not predicate(data):
                raise ValueError(error_msg)
            return data

        return self.add_step(validate)

    def add_transform(self, func: Callable) -> Pipeline:
        """Add a transformation step to the pipeline."""
        return self.add_step(func)

    def add_side_effect(self, func: Callable) -> Pipeline:
        """Add a side effect (logging, monitoring) that doesn't modify data."""

        def side_effect_wrapper(data):
            func(data)
            return data

        return self.add_step(side_effect_wrapper)

    def with_error_handler(self, handler: Callable[[Exception], Any]) -> Pipeline:
        """Set error handler for the pipeline."""
        return Pipeline(steps=self.steps, error_handler=handler)

    def execute(self, initial_value: T) -> Result[T]:
        """Execute the pipeline on initial value."""
        try:
            result = initial_value
            for step in self.steps:
                result = step(result)
            return Result.success(result)

        except Exception as e:
            if self.error_handler:
                try:
                    handled_result = self.error_handler(e)
                    return Result.success(handled_result)
                except Exception as handler_error:
                    return Result.failure(handler_error)
            else:
                return Result.failure(e)

    def execute_async(self, initial_value: T) -> Result[T]:
        """Execute pipeline asynchronously (placeholder for future async support)."""
        # For now, just execute synchronously
        # Future implementation could use asyncio for async operations
        return self.execute(initial_value)


class ConfigurablePipeline:
    """
    Configuration-driven pipeline for complex data processing workflows.

    This class allows building pipelines from configuration specifications,
    making it easy to modify processing workflows without code changes.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.registry = {}
        self._register_default_functions()

    def _register_default_functions(self):
        """Register default functions available for pipeline configuration."""
        import numpy as np

        # Mathematical operations
        self.registry.update(
            {
                "add": lambda x, y: x + y,
                "multiply": lambda x, y: x * y,
                "power": lambda x, y: x**y,
                "log": np.log,
                "exp": np.exp,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "mean": np.mean,
                "std": np.std,
                "min": np.min,
                "max": np.max,
            }
        )

        # Validation functions
        self.registry.update(
            {
                "is_positive": lambda x: np.all(x > 0),
                "is_finite": lambda x: np.all(np.isfinite(x)),
                "has_shape": lambda x, shape: x.shape == shape,
                "is_not_empty": lambda x: len(x) > 0,
            }
        )

    def register_function(self, name: str, func: Callable):
        """Register a custom function for use in pipelines."""
        self.registry[name] = func

    def build_pipeline(self) -> Pipeline:
        """Build pipeline from configuration."""
        pipeline = Pipeline()

        steps = self.config.get("steps", [])
        for step_config in steps:
            step_type = step_config.get("type")
            func_name = step_config.get("function")
            args = step_config.get("args", [])
            kwargs = step_config.get("kwargs", {})

            if func_name not in self.registry:
                raise ValueError(f"Unknown function: {func_name}")

            func = self.registry[func_name]

            if step_type == "transform":
                pipeline = pipeline.add_transform(
                    functools.partial(func, *args, **kwargs)
                )
            elif step_type == "validation":
                error_msg = step_config.get(
                    "error_message", f"Validation failed: {func_name}"
                )
                predicate = functools.partial(func, *args, **kwargs)
                pipeline = pipeline.add_validation(predicate, error_msg)
            elif step_type == "side_effect":
                pipeline = pipeline.add_side_effect(
                    functools.partial(func, *args, **kwargs)
                )

        # Add error handler if specified
        error_handler_name = self.config.get("error_handler")
        if error_handler_name and error_handler_name in self.registry:
            pipeline = pipeline.with_error_handler(self.registry[error_handler_name])

        return pipeline


def create_validation_chain(
    *validators: Callable[[Any], bool],
) -> Callable[[Any], Result[Any]]:
    """
    Create a validation chain that applies multiple validators in sequence.

    Parameters
    ----------
    *validators : Callable[[Any], bool]
        Validation functions that return True if valid

    Returns
    -------
    Callable[[Any], Result[Any]]
        Validation chain function
    """

    def validate(data):
        result = Result.success(data)
        for i, validator in enumerate(validators):
            result = result.filter(validator, f"Validation {i + 1} failed")
            if result.is_failure:
                break
        return result

    return validate


def safe_divide(x: float, y: float) -> Result[float]:
    """
    Safe division that returns Result type.

    Example of how to create composable functions with error handling.
    """
    if y == 0:
        return Result.failure(ZeroDivisionError("Division by zero"))
    return Result.success(x / y)


def safe_sqrt(x: float) -> Result[float]:
    """
    Safe square root that returns Result type.

    Example of how to create composable functions with error handling.
    """
    if x < 0:
        return Result.failure(ValueError("Cannot take square root of negative number"))

    import math

    return Result.success(math.sqrt(x))


# Example usage and demonstration functions
def demonstrate_composition_patterns():
    """
    Demonstrate various function composition patterns.

    This function shows how to use the composition framework for
    improved readability and testability.
    """
    print("Function Composition Framework Demonstration")
    print("=" * 50)

    # 1. Basic function composition
    print("\n1. Basic Function Composition:")
    add_one = lambda x: x + 1
    multiply_two = lambda x: x * 2

    composed = compose(add_one, multiply_two)
    piped = pipe(multiply_two, add_one)

    print(f"compose(add_one, multiply_two)(5) = {composed(5)}")  # (5 * 2) + 1 = 11
    print(f"pipe(multiply_two, add_one)(5) = {piped(5)}")  # (5 * 2) + 1 = 11

    # 2. Result monadic operations
    print("\n2. Monadic Error Handling:")
    result = (
        Result.success(16)
        .flat_map(safe_sqrt)  # sqrt(16) = 4
        .flat_map(lambda x: safe_divide(x, 2))  # 4 / 2 = 2
        .map(lambda x: x * 3)
    )  # 2 * 3 = 6

    if result.is_success:
        print(f"Success: {result.value}")
    else:
        print(f"Error: {result.error}")

    # 3. Pipeline usage
    print("\n3. Pipeline Processing:")
    import numpy as np

    data_pipeline = (
        Pipeline()
        .add_validation(lambda x: len(x) > 0, "Data cannot be empty")
        .add_transform(np.array)
        .add_validation(lambda x: np.all(np.isfinite(x)), "Data must be finite")
        .add_transform(lambda x: x * 2)
        .add_side_effect(lambda x: print(f"Processing array of shape {x.shape}"))
        .add_transform(np.mean)
    )

    test_data = [1, 2, 3, 4, 5]
    pipeline_result = data_pipeline.execute(test_data)

    if pipeline_result.is_success:
        print(f"Pipeline result: {pipeline_result.value}")
    else:
        print(f"Pipeline error: {pipeline_result.error}")


if __name__ == "__main__":
    demonstrate_composition_patterns()
