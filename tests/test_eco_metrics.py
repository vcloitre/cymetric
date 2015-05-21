"""Tests for metrics. These test metric calculation functions unbound to any 
database. This makes writing the tests easier in a unit test like fashion.
"""
from __future__ import print_function, unicode_literals
from uuid import UUID

import nose
from nose.tools import assert_equal, assert_less
from nose.plugins.skip import SkipTest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from cymetric import cyclus
from cymetric import eco_metrics
from cymetric.tools import raw_to_series, ensure_dt_bytes




def test_capital_cost():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 9, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 10, 343.75),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 11, 687.5),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 12, 1031.25),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 13, 1375.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 14, 1718.75),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 15, 2062.5),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 16, 2406.25),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 17, 2750.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 18, 1375.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 19, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, -8, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, -7, 750.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, -6, 1500.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, -5, 2250.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, -4, 3000.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, -3, 3750.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, -2, 4500.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, -1, 5250.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, 0, 6000.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, 1, 3000.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, 2, 0)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('AgentId', '<i8'), ('Time','<i8'),
             ('CashFlow', '<f8')]))
        )
    power = pd.DataFrame(np.array([
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, 12, 3),
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, 12, 4),
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 20, 12, 5),
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 5.5, 210),
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 5.5, 211),
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 5.5, 212),
          (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 5.5, 213)
          ], dtype=ensure_dt_bytes([
                  ('SimId', 'O'), ('AgentId', '<i8'), ('Value', '<f8'),
                  ('Time', '<i8')]))
          )
    entry = pd.DataFrame(np.array([
          (13, ':cycamore:Reactor', 210),
          (20, ':cycamore:Reactor', 3),
          (4, ':cycamore:Sink', 1)
          ], dtype=ensure_dt_bytes([('AgentId', '<i8'), ('Spec', 'O'),
                  ('EnterTime', '<i8')]))
          )
    s1 = power.set_index(['SimId', 'AgentId', 'Value'])['Time']
    s2 = entry.set_index(['AgentId', 'Spec'])['EnterTime']
    series = [s1, s2]
    obs = eco_metrics.capital_cost.func(series)
    assert_frame_equal(exp, obs)


def test_fuel_cost():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 1, 'uox', 
        29641.600000000002, 46),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 11, 3, 'mox', 0, 9)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('TransactionId', '<i8'), ('ReceiverId','<i8'),
             ('Commodity', 'O'), ('Cost', '<f8'), ('Time', '<i8')]))
        )
    resources = pd.DataFrame(np.array([
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 12.56),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 5.5),
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('ResourceId', '<i8'), ('Quantity', 
                      '<f8'),]))
              )
    transactions = pd.DataFrame(np.array([
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 1, 27,
                 'uox', 46),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 11, 3, 13, 
                 'mox', 9)
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('TransactionId', '<i8'),
                         ('ReceiverId', '<i8'), ('ResourceId', '<i8'), 
                         ('Commodity', 'O'), ('Time', '<i8')]))
                 )
    s1 = resources.set_index(['SimId', 'ResourceId'])['Quantity']
    s2 = transactions.set_index(['SimId', 'TransactionId', 'ReceiverId', 
       'ResourceId', 'Commodity'])['Time']
    series = [s1, s2]
    obs = eco_metrics.fuel_cost.func(series)
    assert_frame_equal(exp, obs)


def test_decommissioning_cost():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 19),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/49, 20),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/49*2, 21),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7*3/7, 22),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7*4/7, 23),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7*5/7, 24),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7*6/7, 25),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7, 26),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7*6/7, 27),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7*5/7, 28),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7*4/7, 29),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/7*3/7, 30),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/49*2, 31),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 7500/49, 32),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 33)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('AgentId', '<i8'), ('DecomPayment','<f8'),
             ('Time', '<i8')]))
        )
    decom = pd.DataFrame(np.array([
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 234),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 450)
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('AgentId', '<i8'), ('DecomTime', 
                      '<i8'),]))
              )
    power = pd.DataFrame(np.array([
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 10.0),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 11.3)
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('AgentId', '<i8'), ('Value', 
                         '<f8')]))
                 )
    entry = pd.DataFrame(np.array([
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 
                 ':cycamore:Reactor'),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 
                 ':cycamore:Reactor'),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 
                 ':cycamore:Sink')
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('AgentId', '<i8'),
                         ('Spec', 'O')]))
                 )
    s1 = decom.set_index(['SimId', 'AgentId'])['DecomTime']
    s2 = power.set_index(['SimId', 'AgentId'])['Value']
    s3 = entry.set_index(['SimId', 'AgentId'])['Spec']
    series = [s1, s2, s3]
    obs = eco_metrics.decommissioning_cost.func(series)
    assert_frame_equal(exp, obs)


def test_operation_maintenance():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 232.3),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 2, 232.3),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 8, 400)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('AgentId', '<i8'), ('Time', '<i8'),
             ('O&MPayment','<f8')]))
        )
    power = pd.DataFrame(np.array([
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 2, 2.323),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 3, 2.323),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 32, 2.323),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 100, 4)
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('AgentId', '<i8'), ('Time', '<i8'),
                      ('Value', '<f8')]))
              )
    s1 = power.set_index(['SimId', 'AgentId', 'Time'])['Value']
    series = [s1]
    obs = eco_metrics.operation_maintenance.func(series)
    assert_frame_equal(exp, obs)


if __name__ == "__main__":
    nose.runmodule()