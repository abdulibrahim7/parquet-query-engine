"""Local Parquet scans backed by Apache Arrow Dataset."""
from dataclasses import dataclass
from pathlib import Path

from plan import And, Comparison, LogicalPlan, Predicate
from schema import columns_tuple, projected_schema


def split_predicate(predicate, schema):
    """Separate statistics-safe conjuncts from filters that require decoded rows."""
    import pyarrow as pa

    if isinstance(predicate, And):
        left_scan, left_residual = split_predicate(predicate.left, schema)
        right_scan, right_residual = split_predicate(predicate.right, schema)
        return (conjoin(left_scan, right_scan), conjoin(left_residual, right_residual))
    if isinstance(predicate, Comparison):
        if predicate.op == '!=' and pa.types.is_floating(schema.field(predicate.column).type):
            # Min/max statistics can omit NaN although NaN != a finite value.
            return None, predicate
        return predicate, None
    return None, predicate


def conjoin(left, right):
    if left is None:
        return right
    return left if right is None else And(left, right)


@dataclass(frozen=True)
class ParquetScan(LogicalPlan):
    path: str | Path
    columns: tuple[str, ...] | None = None
    predicate: Predicate | None = None

    def __post_init__(self):
        if self.columns is not None:
            object.__setattr__(self, 'columns', columns_tuple(self.columns))

    def _dataset(self):
        import pyarrow.dataset as ds

        path = Path(self.path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != '.parquet':
            raise ValueError('Expected a local .parquet file')
        return ds.dataset(str(path), format='parquet')

    def _validate(self, dataset):
        import pyarrow as pa

        schema = dataset.schema
        columns_tuple(schema.names)
        if any(pa.types.is_nested(field.type) for field in schema):
            raise ValueError('Nested Parquet columns are not supported')
        if self.predicate is not None:
            self.predicate.to_arrow(schema)
        return schema if self.columns is None else projected_schema(schema, self.columns)

    def output_schema(self):
        return self._validate(self._dataset())

    def execute(self):
        return self._scan(self._dataset())

    def _execute(self):
        return self.execute()

    def _scan(self, dataset):
        import pyarrow as pa

        output = self._validate(dataset)
        pushed, residual = split_predicate(self.predicate, dataset.schema)
        expression = pushed.to_arrow(dataset.schema) if pushed is not None else None
        if residual is None:
            return dataset.to_table(columns=output.names, filter=expression)

        # Decode only output and residual columns. Drop the latter before retaining batches.
        columns = list(dict.fromkeys(output.names + sorted(residual.columns())))
        scanner = dataset.scanner(columns=columns, filter=expression, batch_size=16_384,
                                  batch_readahead=1, fragment_readahead=1)
        residual_expression = residual.to_arrow(dataset.schema)
        batches = (
            batch.filter(residual_expression).select(output.names) for batch in scanner.to_batches()
        )
        return pa.Table.from_batches(batches, schema=output)

    def row_groups(self):
        """Return (candidate, total) row groups based on Arrow metadata pruning."""
        dataset = self._dataset()
        self._validate(dataset)
        pushed, _ = split_predicate(self.predicate, dataset.schema)
        expression = pushed.to_arrow(dataset.schema) if pushed is not None else None
        fragments = list(dataset.get_fragments())
        return (sum(len(fragment.split_by_row_group(filter=expression)) for fragment in fragments),
                sum(fragment.num_row_groups for fragment in fragments))

    def explain(self, depth=0):
        columns = '*' if self.columns is None else ', '.join(self.columns)
        predicate = 'none' if self.predicate is None else str(self.predicate)
        return '  ' * depth + f'ParquetScan({Path(self.path).name}, columns=[{columns}], predicate={predicate})'
