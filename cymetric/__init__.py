"""Cymetric: The Cyclus Analysis Toolkit"""
from __future__ import unicode_literals, print_function

try:
    from cymetric.cyclus import Datum, FullBackend, SqliteBack, Hdf5Back, \
        Recorder
    from cymetric.typesystem import *  # only grabs code generated defintiions
    from cymetric.tools import dbopen
    from cymetric.schemas import schema, canon_dbtype, canon_shape, \
        canon_column, canon_name
    from cymetric.root_metrics import root_metric
    from cymetric.metrics import Metric, metric
    from cymetric.evaluator import METRIC_REGISTRY, register_metric, \
        raw_to_series, Evaluator, eval
    from cymetric.execution import ExecutionContext, exec_code
    from cymetric.eco_inputs import default_cap_overnight
    from cymetric.eco_metrics import capital_cost
    from cymetric.cash_flows import lcoe_plot
except ImportError:
    # again with the wacky CI issues
    from .cyclus import Datum, FullBackend, SqliteBack, Hdf5Back, \
        Recorder
    from .typesystem import *  # only grabs code generated defintiions
    from .tools import dbopen
    from .schemas import schema, canon_dbtype, canon_shape, \
        canon_column, canon_name
    from .root_metrics import root_metric
    from .metrics import Metric, metric
    from .evaluator import METRIC_REGISTRY, register_metric, \
        raw_to_series, Evaluator, eval
    from .execution import ExecutionContext, exec_code
    from .eco_inputs import default_cap_overnight
    from .eco_metrics import capital_cost
    from .cash_flows import lcoe_plot