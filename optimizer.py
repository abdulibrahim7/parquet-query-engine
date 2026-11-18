"""Rule-based optimizer for a small unary relational algebra."""
from dataclasses import replace
from plan import And, Filter, LogicalPlan, Projection
from parquet import ParquetScan


def filter_combine(plan: LogicalPlan) -> LogicalPlan:
    if isinstance(plan, Filter) and isinstance(plan.child, Filter):
        return Filter(plan.child.child, And(plan.child.predicate, plan.predicate))
    return plan


def predicate_pushdown(plan: LogicalPlan) -> LogicalPlan:
    if not isinstance(plan, Filter):
        return plan
    child = plan.child
    if isinstance(child, Projection) and plan.predicate.columns().issubset(child.columns):
        return Projection(Filter(child.child, plan.predicate), child.columns)
    if isinstance(child, ParquetScan):
        if child.columns is not None and not plan.predicate.columns().issubset(child.columns):
            return plan
        combined = plan.predicate if child.predicate is None else And(child.predicate, plan.predicate)
        return replace(child, predicate=combined)
    return plan


def projection_pushdown(plan: LogicalPlan) -> LogicalPlan:
    if not isinstance(plan, Projection):
        return plan
    child = plan.child
    if isinstance(child, Projection) and set(plan.columns).issubset(child.columns):
        return Projection(child.child, plan.columns)
    if isinstance(child, ParquetScan):
        if child.columns is not None and not set(plan.columns).issubset(child.columns):
            return plan
        return replace(child, columns=plan.columns)
    return plan


def _rewrite(plan: LogicalPlan) -> LogicalPlan:
    if isinstance(plan, (Filter, Projection)):
        plan = replace(plan, child=_rewrite(plan.child))
    for rule in (filter_combine, predicate_pushdown, projection_pushdown):
        plan = rule(plan)
    return plan


def optimize(plan: LogicalPlan, *, validate: bool = True) -> LogicalPlan:
    """Validate the original schema before rewriting, then run to a fixed point.

    validate=False is intended for structural tests on plans without actual files.
    Do not use it to execute an unvalidated plan.
    """
    expected = plan.output_schema() if validate else None
    # Each rule eliminates a node or moves a filter below a projection.
    for _ in range(1000):
        result = _rewrite(plan)
        if result == plan:
            if validate and not result.output_schema().equals(expected, check_metadata=True):
                raise RuntimeError('Optimizer changed output schema')
            return result
        plan = result
    raise RuntimeError('Optimizer failed to converge')
