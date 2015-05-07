"""Tests for metrics. These test metric calculation functions
unbound to any database. This makes writing the tests easier in a unit
test like fashion.
"""
from __future__ import print_function, unicode_literals
from uuid import UUID

import nose
from nose.tools import assert_equal, assert_less
from nose.plugins.skip import SkipTest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

try:
    from pyne import data
    HAVE_PYNE = True
except ImportError:
    HAVE_PYNE = False

from cymetric import cyclus
from cymetric import metrics
from cymetric.tools import raw_to_series, ensure_dt_bytes


def test_agents():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 22, 'Region', ':agents:NullRegion', 'USA', -1, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 23, 'Inst', ':agents:NullInst', 'utility', 22, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 24, 'Facility', ':agents:Source', 'MineU235', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 25, 'Facility', ':agents:Source', 'U238', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 26, 'Facility', ':agents:Source', 'DU', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 'Facility', ':agents:Source', 'DU2', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 28, 'Facility', ':Brightlite:FuelfabFacility', 'LWR Fuel FAb', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 29, 'Facility', ':Brightlite:ReprocessFacility', 'LWR Seperation', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 30, 'Facility', ':Brightlite:ReprocessFacility', 'FR Reprocess', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 31, 'Facility', ':agents:Sink', 'SINK', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 32, 'Facility', ':Brightlite:FuelfabFacility', 'FR Fuel Fab', 23, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 33, 'Inst', ':cycamore:DeployInst', 'utility2', 22, -1, 0, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 'Facility', ':Brightlite:ReactorFacility', 'LWR', 33, -1, 5, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 35, 'Facility', ':Brightlite:ReactorFacility', 'LWR', 33, -1, 5, 120.0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 36, 'Facility', ':Brightlite:ReactorFacility', 'FRx', 33, -1, 10, 120.0),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('AgentId', '<i8'), 
                ('Kind', 'O'), ('Spec', 'O'), ('Prototype', 'O'), 
                ('ParentId', '<i8'), ('Lifetime', '<i8'), 
                ('EnterTime', '<i8'), ('ExitTime', '<f8')]))
        )
    agent_entry = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 22, 'Region', ':agents:NullRegion', 'USA', -1, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 23, 'Inst', ':agents:NullInst', 'utility', 22, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 24, 'Facility', ':agents:Source', 'MineU235', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 25, 'Facility', ':agents:Source', 'U238', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 26, 'Facility', ':agents:Source', 'DU', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 'Facility', ':agents:Source', 'DU2', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 28, 'Facility', ':Brightlite:FuelfabFacility', 'LWR Fuel FAb', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 29, 'Facility', ':Brightlite:ReprocessFacility', 'LWR Seperation', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 30, 'Facility', ':Brightlite:ReprocessFacility', 'FR Reprocess', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 31, 'Facility', ':agents:Sink', 'SINK', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 32, 'Facility', ':Brightlite:FuelfabFacility', 'FR Fuel Fab', 23, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 33, 'Inst', ':cycamore:DeployInst', 'utility2', 22, -1, 0),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 'Facility', ':Brightlite:ReactorFacility', 'LWR', 33, -1, 5),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 35, 'Facility', ':Brightlite:ReactorFacility', 'LWR', 33, -1, 5),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 36, 'Facility', ':Brightlite:ReactorFacility', 'FRx', 33, -1, 10),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('AgentId', '<i8'), 
                ('Kind', 'O'), ('Spec', 'O'), ('Prototype', 'O'), 
                ('ParentId', '<i8'), ('Lifetime', '<i8'), 
                ('EnterTime', '<i8')]))
        )
    info = pd.DataFrame({'Duration': {0: 120}, 
        'SimId': {0: UUID('f22f2281-2464-420a-8325-37320fd418f8')}, 
        })
    dur = info.set_index(['SimId'])
    series = [raw_to_series(agent_entry, ['SimId', 'AgentId'], col) \
              for col in ('Kind', 'Spec', 'Prototype', 'ParentId', 
                          'Lifetime', 'EnterTime')]
    series += [None, None, dur]
    obs = metrics.agents.func(series)
    assert_frame_equal(exp, obs)


def test_materials():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922350000, 0.04),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922380000, 1.94),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 942390000, 0.01),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('ResourceId', '<i8'), ('ObjId', '<i8'), 
                ('TimeCreated', '<i8'), ('NucId', '<i8'), ('Mass', '<f8')]))
        )
    res = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 9, 7, 'Material', 1, 2, 'kg', 5, 0, 0),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('ResourceId', '<i8'), ('ObjId', '<i8'), 
                ('Type', 'O'), ('TimeCreated', '<i8'), ('Quantity', '<i8'), 
                ('Units', 'O'), ('QualId', '<i8'), ('Parent1', '<i8'), 
                ('Parent2', '<i8')]))
        )
    comps = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 922350000, 0.02),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 922380000, 0.97),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 942390000, 0.005),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('NucId', '<i8'), 
                ('MassFrac', '<f8')]))
        )
    s1 = res.set_index(['SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated'])['Quantity']
    s2 = comps.set_index(['SimId', 'QualId', 'NucId'])['MassFrac']
    series = [s1,s2]
    obs = metrics.materials.func(series)
    assert_frame_equal(exp, obs)


def test_activity():
    if not HAVE_PYNE:
        raise SkipTest
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922350000, 3197501.3876324706),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922380000, 24126337.066086654),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 942390000, 22949993169.28023),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('ResourceId', '<i8'), ('ObjId', '<i8'), 
                ('TimeCreated', '<i8'), ('NucId', '<i8'), ('Activity', '<f8')]))
        )

    mass = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922350000, 0.04),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922380000, 1.94),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 942390000, 0.01),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('ResourceId', '<i8'), ('ObjId', '<i8'), 
                ('TimeCreated', '<i8'), ('NucId', '<i8'), ('Mass', '<f8')]))
        )
    series = [mass.set_index(['SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated', 'NucId'])['Mass']]
    obs = metrics.activity.func(series)
    assert_frame_equal(exp, obs)


def test_decayheat():
    if not HAVE_PYNE:
        raise SkipTest
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922350000, 2.3944723480343003e-12),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922380000, 1.6505536389997207e-11),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 942390000, 1.92784802432412e-08),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('ResourceId', '<i8'), ('ObjId', '<i8'), 
                ('TimeCreated', '<i8'), ('NucId', '<i8'), ('DecayHeat', '<f8')]))
        )

    act = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922350000, 3197501.3876324706),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 922380000, 24126337.066086654),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 9, 7, 1, 942390000, 22949993169.28023),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('ResourceId', '<i8'), ('ObjId', '<i8'), 
                ('TimeCreated', '<i8'), ('NucId', '<i8'), ('Activity', '<f8')]))
        )

    series = [act.set_index(['SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated', \
                  'NucId'])['Activity']]
    obs = metrics.decay_heat.func(series)
    assert_frame_equal(exp, obs)


#################################
####### FCO METRICS TESTS #######
#################################

def test_fco_u_mined():
    exp = pd.DataFrame(np.array([(0, 3.780034), (1, 2.185349)], 
        dtype=ensure_dt_bytes([('Year', '<i8'), ('FcoUMined', '<f8')]))
        )
    mats = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 7, 3, 3, 922350000, 8.328354),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 7, 3, 3, 922380000, 325.004979),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 6, 8, 4, 3, 922350000, 11.104472),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 6, 8, 4, 3, 922380000, 322.228861),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 7, 9, 5, 12, 922350000, 11.104472),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 7, 9, 5, 12, 922380000, 322.228861),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('ResourceId', '<i8'),
                ('ObjId', '<i8'), ('TimeCreated', '<i8'), ('NucId', '<i8'), 
                ('Mass', '<f8')]))
        )
    trans = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 1, 7, 'LWR Fuel'),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 2, 8, 'LWR Fuel'),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 3, 9, 'LWR Fuel'),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('TransactionId', '<i8'), ('ResourceId', '<i8'), 
                ('Commodity', 'O')]))
        )
    s1 = mats.set_index(['SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated', 'NucId'])['Mass']
    s2 = trans.set_index(['SimId', 'TransactionId', 'ResourceId'])['Commodity']
    series = [s1,s2]
    obs = metrics.fco_u_mined.func(series)
    assert_frame_equal(exp, obs)


def test_fco_fuel_loading():
    exp = pd.DataFrame(np.array([(0, 0.666666), (1, 0.333333)], 
        dtype=ensure_dt_bytes([('Year', '<i8'), ('FuelLoading', '<f8')]))
        )
    mats = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 7, 3, 3, 922350000, 8.328354),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, 7, 3, 3, 922380000, 325.004979),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 6, 8, 4, 3, 922350000, 11.104472),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 6, 8, 4, 3, 922380000, 322.228861),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 7, 9, 5, 12, 922350000, 11.104472),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 7, 9, 5, 12, 922380000, 322.228861),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('QualId', '<i8'), ('ResourceId', '<i8'),
                ('ObjId', '<i8'), ('TimeCreated', '<i8'), ('NucId', '<i8'), 
                ('Mass', '<f8')]))
        )
    trans = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 1, 7, 'LWR Fuel'),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 2, 8, 'FR Fuel'),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 3, 9, 'FR Fuel'),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('TransactionId', '<i8'), ('ResourceId', '<i8'), 
                ('Commodity', 'O')]))
        )
    s1 = mats.set_index(['SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated', 'NucId'])['Mass']
    s2 = trans.set_index(['SimId', 'TransactionId', 'ResourceId'])['Commodity']
    series = [s1,s2]
    obs = metrics.fco_fuel_loading.func(series)
    assert_frame_equal(exp, obs)


def test_fco_electricity_gen():
    exp = pd.DataFrame(np.array([(0, 3), (1, 10)], 
        dtype=ensure_dt_bytes([('Year', '<i8'), ('Power', '<f8')]))
        )
    tsp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 1, 3, 1000),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 2, 3, 2000),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 3, 12, 10000),
        ], dtype=ensure_dt_bytes([
                ('SimId', 'O'), ('AgentId', '<i8'), ('Time', '<i8'), 
                ('Value', '<f8')]))
        )
    series = [tsp.set_index(['SimId', 'AgentId', 'Time'])['Value']]
    obs = metrics.fco_electricity_gen.func(series)
    assert_frame_equal(exp, obs)


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
    obs = metrics.capital_cost.func(series)
    assert_frame_equal(exp, obs)


def test_fuel_cost():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 1, 'uox', 29641.600000000002, 46),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 11, 3, 'mox', 0, 9)
        ], dtype=ensure_dt_bytes([
             ('SimId','O'), ('TransactionId', '<i8'), ('ReceiverId','<i8'),
             ('Commodity', 'O'), ('Cost', '<f8'), ('Time', '<i8')]))
        )
    resources = pd.DataFrame(np.array([
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 12.56),
              (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 5.5),
              ], dtype=ensure_dt_bytes([
                      ('SimId', 'O'), ('ResourceId', '<i8'), ('Quantity', '<f8'),]))
              )
    transactions = pd.DataFrame(np.array([
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 34, 1, 27, 'uox', 46),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 11, 3, 13, 'mox', 9)
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('TransactionId', '<i8'), ('ReceiverId', '<i8'),
                         ('ResourceId', '<i8'), ('Commodity', 'O'), ('Time', '<i8')]))
                 )
    s1 = resources.set_index(['SimId', 'ResourceId'])['Quantity']
    s2 = transactions.set_index(['SimId', 'TransactionId', 'ReceiverId', 'ResourceId', 'Commodity'])['Time']
    series = [s1, s2]
    obs = metrics.fuel_cost.func(series)
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
                      ('SimId', 'O'), ('AgentId', '<i8'), ('DecomTime', '<i8'),]))
              )
    power = pd.DataFrame(np.array([
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 10.0),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, 11.3)
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('AgentId', '<i8'), ('Value', '<f8')]))
                 )
    entry = pd.DataFrame(np.array([
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 27, ':cycamore:Reactor'),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, ':cycamore:Reactor'),
                 (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 5, ':cycamore:Sink')
                 ], dtype=ensure_dt_bytes([
                         ('SimId', 'O'), ('AgentId', '<i8'),
                         ('Spec', 'O')]))
                 )
    s1 = decom.set_index(['SimId', 'AgentId'])['DecomTime']
    s2 = power.set_index(['SimId', 'AgentId'])['Value']
    s3 = entry.set_index(['SimId', 'AgentId'])['Spec']
    series = [s1, s2, s3]
    obs = metrics.decommissioning_cost.func(series)
    assert_frame_equal(exp, obs)


def test_operation_maintenance():
    exp = pd.DataFrame(np.array([
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 0, 232.3),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 2, 232.3),
        (UUID('f22f2281-2464-420a-8325-37320fd418f8'), 13, 8, 400)
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
    obs = metrics.operation_maintenance.func(series)
    assert_frame_equal(exp, obs)


if __name__ == "__main__":
    nose.runmodule()
