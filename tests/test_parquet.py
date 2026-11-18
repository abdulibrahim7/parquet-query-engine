import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from plan import And, Comparison, Filter, Projection
from parquet import ParquetScan
from optimizer import optimize


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'data.parquet'
    pq.write_table(pa.table({'lap': [1, 2, 3, 4, 5], 'speed': [90, 120, None, 150, 110],
                             'corner': ['T1', 'T2', 'T1', 'T2', 'T3']}), path, row_group_size=2)
    return ParquetScan(path)


@pytest.mark.parametrize('make_plan', [
    lambda s: s,
    lambda s: Filter(s, Comparison('speed', '>', 100)),
    lambda s: Projection(s, ('corner', 'lap')),
    lambda s: Projection(Filter(s, Comparison('speed', '>', 100)), ('lap',)),
    lambda s: Filter(Filter(s, Comparison('speed', '>', 100)), Comparison('lap', '<', 5)),
    lambda s: Filter(Projection(s, ('corner', 'speed')), Comparison('speed', '>=', 110)),
    lambda s: Filter(s, Comparison('speed', '>', 999)),
    lambda s: Filter(s, Comparison('corner', '==', 'T2')),
])
def test_equivalent(source, make_plan):
    before = make_plan(source)
    assert before.execute().equals(optimize(before).execute(), check_metadata=True)


def test_expected_rows_and_order(source):
    result = optimize(Projection(Filter(source, Comparison('speed', '>', 100)), ('corner', 'lap'))).execute()
    assert result.column_names == ['corner', 'lap']
    assert result.to_pylist() == [{'corner': 'T2', 'lap': 2}, {'corner': 'T2', 'lap': 4}, {'corner': 'T3', 'lap': 5}]


def test_invalid_filter_hidden_column(source):
    with pytest.raises(ValueError, match='Unknown columns'):
        optimize(Filter(Projection(source, ('lap',)), Comparison('speed', '>', 100)))


def test_invalid_nested_projection(source):
    with pytest.raises(ValueError, match='Unknown columns'):
        optimize(Projection(Projection(source, ('lap',)), ('speed',)))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ParquetScan(tmp_path / 'missing.parquet').execute()


@pytest.mark.parametrize('op,expected', [('!=', [1, 2, 4]), ('==', [5]), ('<', [1]), ('>=', [2, 4, 5])])
def test_null_comparisons(source, op, expected):
    query = Projection(Filter(source, Comparison('speed', op, 110)), ('lap',))
    baseline = query.execute()
    assert baseline['lap'].to_pylist() == expected
    assert baseline.equals(optimize(query).execute(), check_metadata=True)


def test_filter_order_and_projection_schema(tmp_path):
    path = tmp_path / 'metadata.parquet'
    schema = pa.schema([
        pa.field('id', pa.int32(), nullable=False, metadata={b'role': b'index'}),
        pa.field('value', pa.float32(), nullable=True),
        pa.field('extra', pa.string()),
    ], metadata={b'source': b'synthetic'})
    pq.write_table(pa.Table.from_arrays([
        pa.array([1, 2, 3, 4], type=pa.int32()),
        pa.array([0.1, None, 0.1, 0.2], type=pa.float32()),
        pa.array(['x', 'y', 'z', 'q']),
    ], schema=schema), path, row_group_size=2)
    query = Projection(Filter(Projection(ParquetScan(path), ('id', 'value')),
                              And(Comparison('value', '==', 0.1), Comparison('id', '>', 1))), ('id',))
    expected = query.execute()
    actual = optimize(query).execute()
    assert actual['id'].to_pylist() == [3]
    assert expected.equals(actual, check_metadata=True)
    assert actual.schema.metadata == schema.metadata
    assert actual.schema.field('id').equals(schema.field('id'), check_metadata=True)


def test_row_group_pruning_and_hidden_filter_columns(source):
    query = Projection(Filter(source, Comparison('speed', '>=', 150)), ('lap',))
    assert source.row_groups() == (3, 3)
    scan = optimize(query)
    assert scan.row_groups() == (1, 3)
    assert scan.execute().to_pylist() == [{'lap': 4}]


def test_no_statistics_means_conservative_candidate_count(tmp_path):
    path = tmp_path / 'no-stats.parquet'
    pq.write_table(pa.table({'id': list(range(8))}), path, row_group_size=2, write_statistics=False)
    query = Filter(ParquetScan(path), Comparison('id', '>=', 6))
    scan = optimize(query)
    assert scan.row_groups() == (4, 4)
    assert scan.execute().equals(query.execute())


def test_floating_not_equal_preserves_nan_and_prunes_safe_conjuncts(tmp_path):
    path = tmp_path / 'nan.parquet'
    pq.write_table(pa.table({'id': [0, 1, 2, 3, 4, 5],
                             'value': [1., float('nan'), 1., None, 2., 2.],
                             'unused': ['x'] * 6}), path, row_group_size=2)
    source = ParquetScan(path)
    query = Projection(Filter(source, Comparison('value', '!=', 1.)), ('id',))
    baseline = query.execute()
    assert baseline['id'].to_pylist() == [1, 4, 5]
    assert optimize(query).execute().equals(baseline)
    combined = Projection(Filter(Filter(source, Comparison('id', '<', 4)),
                                 Comparison('value', '!=', 1.)), ('id',))
    scan = optimize(combined)
    assert scan.row_groups() == (2, 3)
    assert scan.execute().to_pylist() == [{'id': 1}]
    assert scan.execute().equals(combined.execute())


@pytest.mark.parametrize('name,op,literal', [
    ('lap', '==', True), ('lap', '==', 1.0), ('lap', '==', 2**70),
    ('corner', '==', 2), ('speed', '>', '100'),
])
def test_invalid_type_rejected_by_both_paths(source, name, op, literal):
    query = Filter(source, Comparison(name, op, literal))
    for operation in (query.execute, lambda: optimize(query)):
        with pytest.raises(ValueError, match='literal|Literal'):
            operation()


def test_empty_result_has_expected_schema(source):
    query = Projection(Filter(source, Comparison('speed', '>', 999)), ('corner', 'lap'))
    actual = optimize(query).execute()
    assert actual.num_rows == 0
    assert actual.schema.equals(query.output_schema(), check_metadata=True)


def test_cli_row_groups_and_baseline(source, capsys):
    from main import main
    args = [str(source.path), '--columns', 'lap', '--where', 'speed', '>=', '150', '--row-groups']
    assert main(args) == 0
    assert 'Candidate row groups: 1/3' in capsys.readouterr().out
    assert main(args + ['--no-optimize']) == 0
    assert 'Candidate row groups: 3/3' in capsys.readouterr().out


def test_all_null_row_group_does_not_match(tmp_path):
    path = tmp_path / 'null-group.parquet'
    pq.write_table(pa.table({'id': [0, 1, 2, 3], 'value': pa.array([None, None, 1, 2], type=pa.int64())}),
                   path, row_group_size=2)
    source = ParquetScan(path)
    query = Projection(Filter(source, Comparison('value', '!=', 1)), ('id',))
    assert query.execute().to_pylist() == [{'id': 3}]
    assert optimize(query).execute().equals(query.execute())


def test_unsupported_physical_types_are_validated(tmp_path):
    path = tmp_path / 'half.parquet'
    pq.write_table(pa.table({'value': pa.array([1., 2.], type=pa.float16())}), path)
    query = Filter(ParquetScan(path), Comparison('value', '>', 1.))
    for operation in (query.execute, lambda: optimize(query)):
        with pytest.raises(ValueError, match='Incompatible literal'):
            operation()


def test_nested_input_is_rejected(tmp_path):
    path = tmp_path / 'nested.parquet'
    pq.write_table(pa.table({'nested': [[1, 2]]}), path)
    with pytest.raises(ValueError, match='Nested Parquet columns'):
        ParquetScan(path).execute()


def test_boolean_predicates_and_inappropriate_operators(tmp_path):
    path = tmp_path / 'boolean.parquet'
    pq.write_table(pa.table({'id': [0, 1, 2, 3], 'flag': [True, None, False, True]}),
                   path, row_group_size=2)
    source = ParquetScan(path)
    query = Projection(Filter(source, Comparison('flag', '==', True)), ('id',))
    assert query.execute().to_pylist() == [{'id': 0}, {'id': 3}]
    assert optimize(query).execute().equals(query.execute())
    for op, value in [('>', True), ('==', 1)]:
        with pytest.raises(ValueError, match='Incompatible literal'):
            optimize(Filter(source, Comparison('flag', op, value)))
