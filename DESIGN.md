
# Design Decisions

## Using strings instead of tuples to specify shapes

- Pros:
  - more concise, fewer paranthesis makes it more human readable.
  - can use more expressive syntax, e.g. named variadic dims (WIP).
- Cons:
  - need string validation at runtime
  - more prone to errors.

## Decorator instead of type hints

- Pros:
  - this library is for runtime checking, not static analysis
  - type hints would interfere with existing type hints and static
    analyzers
- Cons:
  - more verbose, adds visual noise
  - if you change argument order or refactor, you need to remember
    to make the same changes to the decorator, which is error prone.
