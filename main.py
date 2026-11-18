"""Run a local Parquet query with optional column and comparison filters."""
import argparse
import json
import math
import sys
from plan import Comparison, Filter, Projection
from parquet import ParquetScan
from optimizer import optimize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', help='Local .parquet file')
    parser.add_argument('--columns', nargs='+', help='Selected output columns')
    parser.add_argument('--where', nargs=3, action='append', metavar=('COLUMN', 'OP', 'JSON_VALUE'), default=[])
    parser.add_argument('--no-optimize', action='store_true')
    parser.add_argument('--row-groups', action='store_true', help='Report metadata-based candidate row groups')
    parser.add_argument('--limit', type=int, default=10, help='Displayed rows only (not an execution limit)')
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    import pyarrow as pa

    try:
        plan = ParquetScan(args.file)
        for name, op, value in args.where:
            plan = Filter(plan, Comparison(name, op, json.loads(value)))
        if args.columns:
            plan = Projection(plan, tuple(args.columns))
        print('Original plan:\n' + plan.explain())
        if not args.no_optimize:
            plan = optimize(plan)
            print('\nOptimized plan:\n' + plan.explain())
        if args.row_groups:
            scan = plan
            while isinstance(scan, (Filter, Projection)):
                scan = scan.child
            candidate, total = scan.row_groups()
            print(f"\nCandidate row groups: {candidate}/{total}")
        table = plan.execute()
        print(f'\nRows: {table.num_rows}; columns: {table.num_columns}')
        rows = table.slice(0, args.limit).to_pylist()
        for row in rows:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    row[key] = str(value)
        print(json.dumps(rows, indent=2, default=str, allow_nan=False))
        return 0
    except (OSError, TypeError, ValueError, pa.ArrowException) as exc:
        print(f'Query failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
