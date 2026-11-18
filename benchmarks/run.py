"""Benchmark execution against a deterministic wide Parquet fixture."""
import argparse
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq

from optimizer import optimize
from parquet import ParquetScan
from plan import Comparison, Filter, Projection


def make_fixture(path, rows, group_size, width):
    randomizer = random.Random(412)
    schema = pa.schema([('row_id', pa.int64()), ('lap', pa.int32())]
                       + [(f'payload_{i:02}', pa.float64()) for i in range(width)])
    with pq.ParquetWriter(path, schema, compression='snappy', use_dictionary=False) as writer:
        for start in range(0, rows, group_size):
            count = min(group_size, rows - start)
            stop = start + count
            arrays = [pa.array(range(start, stop), type=pa.int64()),
                      pa.array((i % 97 for i in range(start, stop)), type=pa.int32())]
            arrays += [pa.array([randomizer.random() for _ in range(count)]) for _ in range(width)]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema), row_group_size=group_size)


def query(path, threshold):
    return Projection(Filter(ParquetScan(path), Comparison('row_id', '>=', threshold)), ('lap',))


def peak_rss():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return peak
    if sys.platform.startswith('linux'):
        return peak * 1024
    return None


def worker(path, threshold, mode):
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    plan = query(path, int(threshold))
    if mode == 'optimized':
        plan = optimize(plan)
    start = perf_counter()
    table = plan.execute()
    elapsed = perf_counter() - start
    print(json.dumps({'seconds': elapsed, 'rows': table.num_rows,
                      'peak_arrow_bytes': pa.default_memory_pool().max_memory(),
                      'peak_rss_bytes': peak_rss()}))


def summarize(trials):
    return {key: statistics.median([t[key] for t in trials]) if trials[0][key] is not None else None
            for key in ('seconds', 'peak_arrow_bytes', 'peak_rss_bytes')}


def measure(path, threshold, repeats):
    baseline = query(path, threshold)
    optimized = optimize(baseline)
    reference = baseline.execute()
    if not reference.equals(optimized.execute(), check_metadata=True):
        raise RuntimeError('Benchmark results differ')
    expected = reference.num_rows
    del reference
    candidate, total = optimized.row_groups()
    trials = {'baseline': [], 'optimized': []}
    for i in range(repeats):
        modes = tuple(trials)
        if i % 2:
            modes = modes[::-1]
        for mode in modes:
            sample = json.loads(subprocess.check_output(
                [sys.executable, '-m', 'benchmarks.run', '--worker', str(path), str(threshold), mode],
                text=True, timeout=120,
            ))
            if sample['rows'] != expected:
                raise RuntimeError('Timed trial returned an unexpected row count')
            trials[mode].append(sample)
    return {'output_rows': expected, 'optimized_candidate_row_groups': candidate,
            'total_row_groups': total, 'median': {mode: summarize(samples) for mode, samples in trials.items()},
            'trials': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=400_000)
    parser.add_argument('--row-group-size', type=int, default=20_000)
    parser.add_argument('--payload-columns', type=int, default=14)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    if not (1 <= args.row_group_size < args.rows and args.payload_columns >= 0 and args.repeats >= 1):
        parser.error('Use at least two row groups, a nonnegative payload width, and at least one repeat')

    with tempfile.TemporaryDirectory(prefix='parquet-bench-') as directory:
        path = Path(directory) / 'synthetic.parquet'
        make_fixture(path, args.rows, args.row_group_size, args.payload_columns)
        report = {
            'environment': {'platform': platform.system(), 'architecture': platform.machine(),
                            'python': platform.python_version(), 'pyarrow': pa.__version__,
                            'arrow_cpu_threads': 2, 'arrow_io_threads': 2},
            'fixture': {'rows': args.rows, 'row_group_size': args.row_group_size,
                        'payload_columns': args.payload_columns, 'parquet_bytes': path.stat().st_size,
                        'repeats_per_mode': args.repeats},
            'workloads': {'selective': measure(path, args.rows * 9 // 10, args.repeats),
                          'broad': measure(path, 0, args.repeats)},
        }
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
