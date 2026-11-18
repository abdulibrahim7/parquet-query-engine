# Benchmark method

Run `python -m benchmarks.run --output docs/benchmark-results.json` from the repository root. The harness generates an ordered integer row ID, a lap column, and 14 irrelevant Float64 payload columns with a fixed pseudo-random seed. It writes the data in 20 Snappy-compressed groups of 20,000 rows to a temporary Parquet file. The recorded file was 46,555,713 bytes.

Both workloads apply a filter to the row ID and return the lap column. The selective query returns the last 40,000 rows. The broad query returns all 400,000 rows. The latter shows what happens when all row groups remain candidates but projection can still avoid decoding the unused columns.

For each workload, the driver checks equality of values, order, and schema metadata before timing. Each trial uses a fresh subprocess. Execution time excludes process startup, imports, fixture creation, and optimization and includes result materialization. Worker order alternates and the validation reads the fixture before timing, so these are warm-cache results. The recorded environment was macOS arm64, Python 3.12.16, PyArrow 26.0.1, with two Arrow CPU and two Arrow I/O threads per worker.

Arrow peak measures allocations tracked by the default Arrow memory pool. Peak RSS is the operating system's process high-water mark where supported and includes interpreter and library overhead. It is reported as unavailable on Windows. The row-group counts are metadata candidates. The harness does not measure physical bytes read.

This benchmark compares the project's own full-materialization baseline with its optimized scan. It is intentionally wide and ordered to expose the differences between the two execution paths. The measured results apply to this fixture, environment, and query shape.
