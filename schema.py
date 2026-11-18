"""Logical column validation, independent of the storage backend."""
from collections.abc import Iterable


def columns_tuple(columns: Iterable[str]) -> tuple[str, ...]:
    if isinstance(columns, str):
        raise TypeError('Pass a sequence of column names, not a string')
    names = tuple(columns)
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('At least one nonempty column name is required')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate column names are not supported')
    return names


def require_columns(requested: Iterable[str], available: Iterable[str]) -> None:
    missing = set(requested) - set(available)
    if missing:
        raise ValueError('Unknown columns: ' + ', '.join(sorted(missing)))


def projected_schema(schema, names):
    """Keep field types, nullability, metadata, and requested order."""
    import pyarrow as pa

    require_columns(names, schema.names)
    return pa.schema([schema.field(name) for name in names], metadata=schema.metadata)


def comparison_scalar(field, op, value):
    """Bind supported literals to the exact physical field type."""
    import math
    import pyarrow as pa

    dtype = field.type
    floating = pa.types.is_float32(dtype) or pa.types.is_float64(dtype)
    valid = (
        (pa.types.is_integer(dtype) and type(value) is int)
        or (floating and type(value) in (int, float))
        or ((pa.types.is_string(dtype) or pa.types.is_large_string(dtype)) and type(value) is str)
        or (pa.types.is_boolean(dtype) and type(value) is bool and op in ("==", "!="))
    )
    if not valid:
        raise ValueError(f"Incompatible literal or operator for column {field.name!r} ({dtype})")
    try:
        scalar = pa.scalar(value, type=dtype)
        if floating:
            represented = scalar.as_py()
            if not math.isfinite(represented) or (type(value) is int and represented != value):
                raise ValueError
        return scalar
    except (OverflowError, ValueError, pa.ArrowException) as exc:
        raise ValueError(f"Literal is outside the supported range for column {field.name!r} ({dtype})") from exc
