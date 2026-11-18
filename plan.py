"""Immutable logical operators and a small, explicit predicate language."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import operator
from typing import Literal
from schema import columns_tuple, comparison_scalar, projected_schema, require_columns

Scalar = str | int | float | bool
Operator = Literal['==', '!=', '<', '<=', '>', '>=']
OPERATORS = {'==': operator.eq, '!=': operator.ne, '<': operator.lt,
             '<=': operator.le, '>': operator.gt, '>=': operator.ge}


class Predicate(ABC):
    @abstractmethod
    def columns(self) -> frozenset[str]: ...

    @abstractmethod
    def to_arrow(self, schema=None): ...


@dataclass(frozen=True)
class Comparison(Predicate):
    column: str
    op: Operator
    value: Scalar

    def __post_init__(self):
        if not isinstance(self.column, str) or not self.column:
            raise ValueError('Invalid column')
        if self.op not in ('==', '!=', '<', '<=', '>', '>='):
            raise ValueError('Unsupported comparison')
        if type(self.value) not in (str, int, float, bool):
            raise TypeError('Only non-null scalar literals are supported')
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError('Nonfinite literals are not supported')

    def columns(self):
        return frozenset((self.column,))

    def to_arrow(self, schema=None):
        import pyarrow.dataset as ds

        value = self.value
        if schema is not None:
            require_columns(self.columns(), schema.names)
            value = ds.scalar(comparison_scalar(schema.field(self.column), self.op, value))
        return OPERATORS[self.op](ds.field(self.column), value)

    def __str__(self):
        return f'{self.column} {self.op} {self.value!r}'


@dataclass(frozen=True)
class And(Predicate):
    left: Predicate
    right: Predicate

    def columns(self):
        return self.left.columns() | self.right.columns()

    def to_arrow(self, schema=None):
        return self.left.to_arrow(schema) & self.right.to_arrow(schema)

    def __str__(self):
        return f'({self.left}) AND ({self.right})'


class LogicalPlan(ABC):
    @abstractmethod
    def output_schema(self): ...

    def output_columns(self) -> tuple[str, ...]:
        return tuple(self.output_schema().names)

    def execute(self):
        self.output_schema()
        return self._execute()

    @abstractmethod
    def _execute(self): ...

    @abstractmethod
    def explain(self, depth: int = 0) -> str: ...


@dataclass(frozen=True)
class Filter(LogicalPlan):
    child: LogicalPlan
    predicate: Predicate

    def output_schema(self):
        schema = self.child.output_schema()
        self.predicate.to_arrow(schema)
        return schema

    def _execute(self):
        table = self.child._execute()
        return table.filter(self.predicate.to_arrow(table.schema))

    def explain(self, depth=0):
        return '  ' * depth + f'Filter({self.predicate})\n' + self.child.explain(depth + 1)


@dataclass(frozen=True)
class Projection(LogicalPlan):
    child: LogicalPlan
    columns: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, 'columns', columns_tuple(self.columns))

    def output_schema(self):
        return projected_schema(self.child.output_schema(), self.columns)

    def _execute(self):
        return self.child._execute().select(list(self.columns))

    def explain(self, depth=0):
        return '  ' * depth + f'Projection({", ".join(self.columns)})\n' + self.child.explain(depth + 1)
