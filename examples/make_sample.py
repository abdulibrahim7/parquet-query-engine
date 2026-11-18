"""Generate a small synthetic telemetry dataset, not real client data."""
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    destination = Path(__file__).with_name('sample.parquet')
    table = pa.table({'lap': [1, 1, 2, 2, 3, 3], 'corner': ['T1', 'T2'] * 3,
                      'speed_kph': [105, 130, 115, 142, 120, 150],
                      'time_loss_s': [0.4, 0.1, 0.2, 0.0, 0.1, 0.0]})
    pq.write_table(table, destination, row_group_size=2)
    print(destination)


if __name__ == '__main__':
    main()
