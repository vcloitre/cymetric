#####economic metrics for nuclear power plants#####

from __future__ import print_function, unicode_literals

import inspect

import numpy as np
import pandas as pd


try:
    from cymetric.metrics import metric
    from cymetric import cyclus
    from cymetric import schemas
    from cymetric import typesystem as ts
    from cymetric import tools
    from cymetric.evaluator import register_metric
    from cymetric.eco_inputs import overnight
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from .metrics import metric
    from . import cyclus
    from . import schemas
    from . import typesystem as ts
    from . import tools
    from .evaluator import register_metric
    from .eco_inputs import overnight


## The actual metrics ##


_ccdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Value'), 'Time'), ('AgentEntry', ('AgentId', 'Spec'), 'EnterTime')]

_ccschema = [('SimId', ts.UUID), ('AgentId', ts.INT),
             ('Time', ts.INT), ('CashFlow', ts.DOUBLE)]

@metric(name='CapitalCost', depends=_ccdeps, schema=_ccschema)
def capital_cost(series):
    """cap_cost returns the cash flows per YEAR related to the capital costs of
    a reactor. The overnight cost comes from WEO 2014 (cost in the US). The 
    timeframe for the payment of the capital costs is drawn from of a graph
    from EDF. Next steps could be make the price depends on the reactor 
    technology and make the payment more realistic, ie include interest rates, 
    improve the linear model and finally be able to fetch data"""
    model_choice = input("How do you want to model the capital costs (linear, or random) ? ")
    payment_begin = input("How many years before commissioning should the payment begin ? (give an integer or write \"random\") ")
    payment_duration = input("How many years do you want the payment to last ? (give an integer or write \"random)\" ")
    s_cost = capital_shape(payment_begin, payment_duration, model_choice)
    f_power = series[0].reset_index()
    f_entry = series[1].reset_index()
    f_entry = f_entry[f_entry.Spec == ":cycamore:Reactor"]
    id_reactors = f_entry["AgentId"].values
    f_entry = pd.DataFrame([f_entry.EnterTime, f_entry.AgentId]).transpose()
    f_entry = f_entry.set_index(['AgentId'])
    f_entry['Capacity'] = pd.Series()
    rtn = pd.DataFrame()
    for i in id_reactors:
        f_entry['Capacity'][i] = f_power[f_power.AgentId == i][f_power.Time ==
                               f_entry.EnterTime[i]].Value.iloc[0]
        s_cost2 = s_cost * overnight * f_entry['Capacity'][i]
        rtn = pd.concat([rtn,pd.DataFrame({'AgentId': i, 'Time': pd.Series(list(range(payment_duration)))-payment_begin+f_entry.EnterTime[i]//12, 'CashFlow' : s_cost2})], ignore_index=True)
    rtn['SimId']=f_power.SimId.iloc[0]
    cols = rtn.columns.tolist()
    cols=cols[3:]+cols[:1]+cols[2:3]+cols[1:2]
    rtn = rtn[cols]
    return rtn

del _ccdeps, _ccschema


_fcdeps = [('Resources', ('SimId', 'ResourceId'), 'Quantity'), ('Transactions',
        ('SimId', 'TransactionId', 'ReceiverId', 'ResourceId', 'Commodity'), 
        'Time')]

_fcschema = [('SimId', ts.UUID), ('TransactionId', ts.INT), ('ReceiverId', 
          ts.INT), ('Commodity', ts.STRING), ('Cost', ts.DOUBLE), ('Time', 
          ts.INT)]

@metric(name='FuelCost', depends=_fcdeps, schema=_fcschema)
def fuel_cost(series):
    """fuel_cost returns the cash flows related to the fuel costs for power 
    plants.
    """
    fuel_price = 2360 # $/kg
    # see http://www.world-nuclear.org/info/Economic-Aspects/Economics-of-Nuclear-Power/
    f_resources = series[0].reset_index().set_index(['ResourceId'])
    f_transactions = series[1].reset_index().set_index(['ResourceId'])
    f_transactions['Quantity'] = f_resources['Quantity']
    f_transactions['Cost'] = f_transactions['Quantity']*fuel_price*(
    					   f_transactions['Commodity']=='uox')
    del f_transactions['Quantity']
    rtn = f_transactions.reset_index()
    cols = rtn.columns.tolist()
    cols = cols[1:5]+cols[6:]+cols[5:6]
    rtn = rtn[cols]
    return rtn

del _fcdeps, _fcschema


_dcdeps = [('DecomSchedule', ('SimId', 'AgentId'), 'DecomTime'),
        ('TimeSeriesPower', ('SimId', 'AgentId'), 'Value'), ('AgentEntry',
        ('SimId', 'AgentId'), 'Spec')]

_dcschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('DecomPayment',
          ts.DOUBLE), ('Time', ts.INT)]

@metric(name='DecommissioningCost', depends=_dcdeps, schema=_dcschema)
def decommissioning_cost(series):
    """decom
    """
    cost = 750 # decommission cost in $/kW d'Haeseler
    duration = 15 # decommission last about 15 yrs
    f_decom = series[0].reset_index().set_index('AgentId')
    f_power = series[1].reset_index()
    f_entry = series[2].reset_index()
    id_reactors = f_entry[f_entry.Spec==":cycamore:Reactor"]["AgentId"].values
    id_decom = f_decom.index.tolist()
    id_decom_reac = [val for val in id_reactors if val in id_decom]
    rtn = pd.DataFrame()
    for i in id_decom_reac:
        s_cost = pd.Series(list(range(duration)))
        s_cost = s_cost.apply(lambda x: 4*cost*f_power[f_power['AgentId']==i][
               'Value'].iloc[0]/((duration-1)**2)*x*(x<=duration/2)-4*cost*
               f_power[f_power['AgentId']==i]['Value'].iloc[0]/((duration-
               1)**2)*(x-dur)) # end missing
        rtn = pd.concat([rtn,pd.DataFrame({'AgentId': i, 'Time': list(range(
            duration))+f_decom.DecomTime[i]//12, 'DecomPayment': s_cost})], 
            ignore_index=True)
    rtn['SimId'] = f_decom['SimId'].iloc[0]
    cols = rtn.columns.tolist()
    cols = cols[-1:]+cols[:-1]
    rtn = rtn[cols]
    return rtn

del _dcdeps, _dcschema


c_omdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Time'), 'Value')]

_omschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('Time', ts.INT), 
          ('O&MPayment', ts.DOUBLE)]

@metric(name='OperationMaintenance', depends=_omdeps, schema=_omschema)
def operation_maintenance(series):
    """O&M
    """
    cost = 100 # $/kW/an
    rtn = series[0].reset_index()
    rtn.Time=rtn.Time//12
    rtn =  rtn.drop_duplicates(cols=['AgentId', 'Time'], take_last=True)
    rtn['O&MPayment'] = rtn['Value']*cost
    rtn = rtn.reset_index()
    del rtn['Value'], rtn['index']
    return rtn

del _omdeps, _omschema
