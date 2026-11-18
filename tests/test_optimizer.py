import pytest
from plan import And, Comparison, Filter, Projection
from parquet import ParquetScan
from optimizer import optimize, filter_combine
from schema import columns_tuple, require_columns

S = ParquetScan('not-needed.parquet')
P = Comparison('speed', '>', 100)
Q = Comparison('lap', '==', 2)


def opt(plan):
    return optimize(plan, validate=False)


def test_filter_combine():
    assert filter_combine(Filter(Filter(S, P), Q)) == Filter(S, And(P, Q))


def test_predicate_pushdown():
    assert opt(Filter(S, P)) == ParquetScan(S.path, predicate=P)


def test_projection_pushdown():
    assert opt(Projection(S, ('lap', 'speed'))) == ParquetScan(S.path, ('lap', 'speed'))


def test_both_pushdowns_hidden_filter_column():
    assert opt(Projection(Filter(S, P), ('lap',))) == ParquetScan(S.path, ('lap',), P)


def test_filter_above_projection():
    assert opt(Filter(Projection(S, ('lap', 'speed')), P)) == ParquetScan(S.path, ('lap', 'speed'), P)


def test_nested_projection():
    assert opt(Projection(Projection(S, ('lap', 'speed')), ('speed',))) == ParquetScan(S.path, ('speed',))


def test_nested_filter():
    assert opt(Filter(Filter(S, P), Q)) == ParquetScan(S.path, predicate=And(P, Q))


def test_invalid_filter_not_pushed_past_projection():
    result = opt(Filter(Projection(S, ('lap',)), P))
    assert isinstance(result, Filter)


def test_invalid_projection_not_expanded():
    assert isinstance(opt(Projection(Projection(S, ('lap',)), ('speed',))), Projection)


def test_idempotent():
    result = opt(Projection(Filter(Filter(S, P), Q), ('lap',)))
    assert opt(result) == result


@pytest.mark.parametrize('columns', [(), ('a', 'a'), ('',), 'abc'])
def test_invalid_columns(columns):
    with pytest.raises((TypeError, ValueError)):
        columns_tuple(columns)


def test_missing_columns():
    with pytest.raises(ValueError, match='Unknown columns'):
        require_columns(['missing'], ['lap'])


@pytest.mark.parametrize('op,value', [('???', 1), ('==', None), ('==', float('nan')), ('==', [])])
def test_invalid_comparison(op, value):
    with pytest.raises((TypeError, ValueError)):
        Comparison('lap', op, value)
