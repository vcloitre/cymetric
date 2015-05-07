"""A collection of metrics that come stock with cymetric.
"""
from __future__ import print_function, unicode_literals
import inspect

import numpy as np
import pandas as pd

try:
    from pyne import data
    import pyne.enrichment as enr
    HAVE_PYNE = True
except ImportError:
    HAVE_PYNE = False

try:
    from cymetric import cyclus
    from cymetric import schemas
    from cymetric import typesystem as ts
    from cymetric import tools
    from cymetric.evaluator import register_metric
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from . import cyclus
    from . import schemas
    from . import typesystem as ts
    from . import tools
    from .evaluator import register_metric


class Metric(object):
    """Metric class"""
    dependencies = NotImplemented
    schema = NotImplemented
    
    def __init__(self, db):
        self.db = db

    @property
    def name(self):
        return self.__class__.__name__


def _genmetricclass(f, name, depends, scheme):
    """Creates a new metric class with a given name, dependencies, and schema.
    
    Parameters
    ----------
    name : str
        Metric name
    depends : list of lists (table name, tuple of indices, column name)
        Dependencies on other database tables (metrics or root metrics)
    scheme : list of tuples (column name, data type)
        Schema for metric
    """
    if not isinstance(scheme, schemas.schema):
        scheme = schemas.schema(scheme)

    class Cls(Metric):
        dependencies = depends
        schema = scheme
        func = staticmethod(f)

        __doc__ = inspect.getdoc(f)

        def __init__(self, db):
            """Constructor for metric object in database."""
            super(Cls, self).__init__(db)

        def __call__(self, series, conds=None, known_tables=None, *args, **kwargs):
            """Computes metric for given input data and conditions."""
            # FIXME test if I already exist in the db, read in if I do
            if known_tables is None:
                known_tables = self.db.tables()
            if self.name in known_tables:
                return self.db.query(self.name, conds=conds)
            return f(series)

    Cls.__name__ = str(name)
    register_metric(Cls)
    return Cls


def metric(name=None, depends=NotImplemented, schema=NotImplemented):
    """Decorator that creates metric class from a function or class."""
    def dec(f):
        clsname = name or f.__name__
        return _genmetricclass(f=f, name=clsname, scheme=schema, depends=depends)
    return dec


#
# The actual metrics
#

# Material Mass (quantity * massfrac)
_matdeps = [('Resources', ('SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated'), 
                'Quantity'),
            ('Compositions', ('SimId', 'QualId', 'NucId'), 'MassFrac')]

_matschema = (('SimId', ts.UUID), ('QualId', ts.INT), 
              ('ResourceId', ts.INT), ('ObjId', ts.INT), 
              ('TimeCreated', ts.INT), ('NucId', ts.INT), 
              ('Mass', ts.DOUBLE))

@metric(name='Materials', depends=_matdeps, schema=_matschema)
def materials(series):
    """Materials metric returns the material mass (quantity of material in 
    Resources times the massfrac in Compositions) indexed by the SimId, QualId, 
    ResourceId, ObjId, TimeCreated, and NucId.
    """
    x = pd.merge(series[0].reset_index(), series[1].reset_index(), 
            on=['SimId', 'QualId'], how='inner').set_index(['SimId', 'QualId', 
                'ResourceId', 'ObjId','TimeCreated', 'NucId'])
    y = x['Quantity'] * x['MassFrac']
    y.name = 'Mass'
    z = y.reset_index()
    return z

del _matdeps, _matschema


# Activity (mass * decay_const / atomic_mass)
_actdeps = [('Materials', ('SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated', 'NucId'), 'Mass')]

_actschema = [('SimId', ts.UUID), ('QualId', ts.INT), 
              ('ResourceId', ts.INT), ('ObjId', ts.INT), 
              ('TimeCreated', ts.INT), ('NucId', ts.INT), 
              ('Activity', ts.DOUBLE)]

@metric(name='Activity', depends=_actdeps, schema=_actschema)
def activity(series):
    """Activity metric returns the instantaneous activity of a nuclide 
    in a material (material mass * decay constant / atomic mass) 
    indexed by the SimId, QualId, ResourceId, ObjId, TimeCreated, and NucId.
    """
    tools.raise_no_pyne('Activity could not be computed', HAVE_PYNE)
    mass = series[0]
    act = []
    for (simid, qual, res, obj, time, nuc), m in mass.iteritems():
        val = (1000 * data.N_A * m * data.decay_const(nuc) \
              / data.atomic_mass(nuc))
        act.append(val)
    act = pd.Series(act, index=mass.index)
    act.name = 'Activity'
    rtn = act.reset_index()
    return rtn

del _actdeps, _actschema


# DecayHeat (activity * q_value)
_dhdeps = [('Activity', ('SimId', 'QualId', 'ResourceId', 'ObjId', 'TimeCreated', 'NucId'),
               'Activity')]

_dhschema = [('SimId', ts.UUID), ('QualId', ts.INT), 
             ('ResourceId', ts.INT), ('ObjId', ts.INT), 
             ('TimeCreated', ts.INT), ('NucId', ts.INT), 
             ('DecayHeat', ts.DOUBLE)]

@metric(name='DecayHeat', depends=_dhdeps, schema=_dhschema)
def decay_heat(series):
    """Decay heat metric returns the instantaneous decay heat of a nuclide 
    in a material (Q value * activity) indexed by the SimId, QualId, 
    ResourceId, ObjId, TimeCreated, and NucId.
    """
    tools.raise_no_pyne('DecayHeat could not be computed', HAVE_PYNE)
    act = series[0]
    dh = []
    for (simid, qual, res, obj, time, nuc), a in act.iteritems():
        val = (data.MeV_per_MJ * a * data.q_val(nuc))
        dh.append(val)
    dh = pd.Series(dh, index=act.index)
    dh.name = 'DecayHeat'
    rtn = dh.reset_index()
    return rtn

del _dhdeps, _dhschema


# Agents

_agentsdeps = [
    ('AgentEntry', ('SimId', 'AgentId'), 'Kind'),
    ('AgentEntry', ('SimId', 'AgentId'), 'Spec'),
    ('AgentEntry', ('SimId', 'AgentId'), 'Prototype'),
    ('AgentEntry', ('SimId', 'AgentId'), 'ParentId'),
    ('AgentEntry', ('SimId', 'AgentId'), 'Lifetime'),
    ('AgentEntry', ('SimId', 'AgentId'), 'EnterTime'),
    ('AgentExit', ('SimId', 'AgentId'), 'ExitTime'),
    ('DecomSchedule', ('SimId', 'AgentId'), 'DecomTime'),
    ('Info', ('SimId',), 'Duration'),
    ]

_agentsschema = schemas.schema([
    ('SimId', ts.UUID), ('AgentId', ts.INT), ('Kind', ts.STRING), 
    ('Spec', ts.STRING), ('Prototype', ts.STRING), ('ParentId', ts.INT), 
    ('Lifetime', ts.INT), ('EnterTime', ts.INT), ('ExitTime', ts.INT),
    ])

@metric(name='Agents', depends=_agentsdeps, schema=_agentsschema)
def agents(series):
    """Computes the Agents table. This is tricky because both the AgentExit
    table and the DecomSchedule table may not be present in the database.
    Furthermore, the Info table does not contain the AgentId column. This
    computation handles the calculation of the ExitTime in the face a 
    significant amounts of missing data.
    """
    mergeon  = ['SimId', 'AgentId']
    idx = series[0].index
    df = series[0].reset_index()
    for s in series[1:6]:
        df = pd.merge(df, s.reset_index(), on=mergeon)
    agent_exit = series[6]
    if agent_exit is None:
        agent_exit = pd.Series(index=idx, data=[np.nan]*len(idx))
        agent_exit.name = 'ExitTime'
    else:
        agent_exit = agent_exit.reindex(index=idx)
    df = pd.merge(df, agent_exit.reset_index(), on=mergeon)
    decom_time = series[7]
    if decom_time is not None:
        df = tools.merge_and_fillna_col(df, decom_time.reset_index(), 
                                        'ExitTime', 'DecomTime', on=mergeon)
    duration = series[8]
    df = tools.merge_and_fillna_col(df, duration.reset_index(), 
                                    'ExitTime', 'Duration', on=['SimId'])
    return df

del _agentsdeps, _agentsschema


#########################
## FCO-related metrics ##
#########################

# U Resources Mined [t] 
_udeps= [('Materials', ('ResourceId', 'ObjId', 'TimeCreated', 'NucId'), 'Mass'),
         ('Transactions', ('ResourceId', ), 'Commodity')]

_uschema = [('Year', ts.INT), ('FcoUMined', ts.DOUBLE)]

@metric(name='FcoUMined', depends=_udeps, schema=_uschema)
def fco_u_mined(series):
    """FcoUMined metric returns the uranium mined in tonnes for each year 
    in a 200-yr simulation. This is written for FCO databases that use the 
    Bright-lite Fuel Fab(i.e., the U235 and U238 are given separately in the 
    FCO simulations)."""
    mass = pd.merge(series[0].reset_index(), series[1].reset_index(), 
            on=['ResourceId'], how='inner').set_index(['ObjId', 
                'TimeCreated', 'NucId'])
    u = []
    prods = {}
    mass235 = {}
    m = mass[mass['Commodity'] == 'LWR Fuel']
    for (obj, _, nuc), value in m.iterrows():
        prods[obj] = prods.get(obj, 0.0) + value['Mass']
        if nuc==922350000:
            mass235[obj] = value['Mass']
    for obj, m235 in mass235.items():
        x_prod = m235 / prods[obj]
        feed = enr.feed(0.0072, x_prod, 0.0025, product=prods[obj]) / 1000
        u.append(feed)
    m = m.groupby(level=['ObjId', 'TimeCreated'])['Mass'].sum()
    m = m.reset_index()
    # sum by years (12 time steps)
    u = pd.DataFrame(data={'Year': m.TimeCreated.apply(lambda x: x//12), 
                           'FcoUMined': u}, columns=['Year', 'FcoUMined'])
    u = u.groupby('Year').sum()
    rtn = u.reset_index()
    return rtn

del _udeps, _uschema


# Electricity Generated [GWe-y]
_egdeps = [('TimeSeriesPower', ('Time',), 'Value'),]

_egschema = [('Year', ts.INT), ('Power', ts.DOUBLE)]

@metric(name='FcoElectricityGen', depends=_egdeps, schema=_egschema)
def fco_electricity_gen(series):
    """FcoElectricityGen metric returns the electricity generated in GWe-y 
    in a 200-yr simulation. This is written for the purpose of FCO databases.
    """
    elec = series[0].reset_index()
    # sum by years (12 time steps)
    elec = pd.DataFrame(data={'Year': elec.Time.apply(lambda x: x//12), 
                              'Power': elec.Value.apply(lambda x: x/1000)}, 
                        columns=['Year', 'Power'])
    elec = elec.groupby('Year').sum()
    rtn = elec.reset_index()
    return rtn

del _egdeps, _egschema


# Annual Fuel Loading Rate [tHM/y]
_fldeps = [('Materials', ('ResourceId', 'TimeCreated'), 'Mass'),
          ('Transactions', ('ResourceId',), 'Commodity')]

_flschema = [('Year', ts.INT), ('FuelLoading', ts.DOUBLE)]

@metric(name='FcoFuelLoading', depends=_fldeps, schema=_flschema)
def fco_fuel_loading(series):
    """FcoFuelLoading metric returns the fuel loaded in tHM/y in a 200-yr 
    simulation. This is written for FCO databases.
    """
    mass = pd.merge(series[0].reset_index(), series[1].reset_index(),
            on=['ResourceId'], how='inner').set_index(['TimeCreated'])
    mass = mass.query('Commodity == ["LWR Fuel", "FR Fuel"]')
    mass = mass.groupby(mass.index)['Mass'].sum()
    # sum by years (12 time steps)
    mass.index = map(lambda x: x//12, mass.index)
    mass.index.name = 'Year'
    mass.name = 'FuelLoading'
    mass = mass.reset_index()
    # kg to t
    mass.FuelLoading = mass.FuelLoading / 1000
    return mass

del _fldeps, _flschema

#####economic metrics for nuclear power plants#####

_ccdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Value'), 'Time'), ('AgentEntry', ('AgentId', 'Spec'), 'EnterTime')]

_ccschema = [('SimId', ts.UUID), ('AgentId', ts.INT),
             ('Time', ts.INT), ('CashFlow', ts.DOUBLE)]

@metric(name='CapitalCost', depends=_ccdeps, schema=_ccschema)
def capital_cost(series):
    """cap_cost returns the cash flows per YEAR related to the capital costs of a reactor.
    overnight cost comes from WEO 2014 (cost in the US). The timeframe for the payment of
    the capital costs is drawn from of a graph from EDF. Next steps could be make the price
    depends on the reactor technology and make the payment more realistic, ie include
    interest rates, improve the linear model and finally be able to fetch data"""
    overnight=5000 #capital cost overnight $/kW
    pay_beg=8 #payment begins 8 years before commissioning
    pay_end=2 #payment ends 2 years after commissioning
    f_power=series[0].reset_index()
    f_entry=series[1].reset_index()
    id_reactors=f_entry[f_entry.Spec==":cycamore:Reactor"]["AgentId"].values
    df=pd.DataFrame([f_entry[f_entry.Spec==':cycamore:Reactor'].EnterTime,f_entry[f_entry.Spec==':cycamore:Reactor'].AgentId]).transpose()
    df=df.set_index(['AgentId'])
    df['Capacity']=pd.Series()
    rtn=pd.DataFrame()
    for i in id_reactors:
        df['Capacity'][i]=f_power[f_power.AgentId==i][f_power.Time==df.EnterTime[i]].Value.iloc[0]
        s_cost=pd.Series(list(range(pay_beg+pay_end+1)))
        s_cost=s_cost.apply(lambda x: overnight*df.Capacity[i]/((pay_beg+pay_end)*pay_beg)*x*(x<=8)-overnight*df.Capacity[i]/((pay_beg+pay_end)*pay_end)*(x-pay_beg-pay_end)*(x>8))
        rtn= pd.concat([rtn,pd.DataFrame({'AgentId': i, 'Time': list(range(pay_beg+pay_end+1))+df.EnterTime[i]//12-pay_beg, 'CashFlow': s_cost})], ignore_index=True)
    rtn['SimId']=f_power.SimId.iloc[0]
    cols = rtn.columns.tolist()
    cols=cols[3:]+cols[:1]+cols[2:3]+cols[1:2]
    rtn = rtn[cols]
    return rtn

del _ccdeps, _ccschema


_fcdeps = [('Resources', ('SimId', 'ResourceId'), 'Quantity'), ('Transactions', ('SimId', 'TransactionId', 'ReceiverId', 'ResourceId', 'Commodity'), 'Time')]

_fcschema = [('SimId', ts.UUID), ('TransactionId', ts.INT),
             ('ReceiverId', ts.INT), ('Commodity', ts.STRING), ('Cost', ts.DOUBLE), ('Time', ts.INT)]

@metric(name='FuelCost', depends=_fcdeps, schema=_fcschema)
def fuel_cost(series):
    """fuel_cost returns the cash flows related to the fuel costs for power plants.
    """
    fuel_price = 2360 # $/kg, see http://www.world-nuclear.org/info/Economic-Aspects/Economics-of-Nuclear-Power/
    f_resources = series[0].reset_index().set_index(['ResourceId'])
    f_transactions = series[1].reset_index().set_index(['ResourceId'])
    f_transactions['Quantity'] = f_resources['Quantity']
    f_transactions['Cost'] = f_transactions['Quantity']*fuel_price*(f_transactions['Commodity']=='uox')
    del f_transactions['Quantity']
    rtn = f_transactions.reset_index()
    cols = rtn.columns.tolist()
    cols = cols[1:5]+cols[6:]+cols[5:6]
    rtn = rtn[cols]
    return rtn

del _fcdeps, _fcschema


_dcdeps = [('DecomSchedule', ('SimId', 'AgentId'), 'DecomTime'), ('TimeSeriesPower', ('SimId', 'AgentId'), 'Value'),
          ('AgentEntry', ('SimId', 'AgentId'), 'Spec')]

_dcschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('DecomPayment', ts.DOUBLE), ('Time', ts.INT)]

@metric(name='DecommissioningCost', depends=_dcdeps, schema=_dcschema)
def decommissioning_cost(series):
    """decom
    """
    cost = 750 # decommission cost in $/kW d'Haeseler
    duration = 15 # decommission last about 15 yrs
    f_decom = series[0].reset_index().set_index('AgentId')
    f_power = series[1].reset_index()
    ff_entry = series[2].reset_index()
    id_reactors = f_entry[f_entry.Spec==":cycamore:Reactor"]["AgentId"].values
    id_decom = f_decom.index.tolist()
    id_decom_reac = [val for val in id_reactors if val in id_decom]
    rtn = pd.DataFrame()
    for i in id_decom_reac:
        s_cost=pd.Series(list(range(duration)))
        s_cost=s_cost.apply(lambda x: 4*cost*f_power[f_power['AgentId']==i]['Value'].iloc[0]/((duration-1)**2)*x*(x<=duration/2)-4*cost*f_power[f_power['AgentId']==i]['Value'].iloc[0]/((duration-1)**2)*(x-duration+1)*(x>duration/2))
        rtn = pd.concat([rtn,pd.DataFrame({'AgentId': i, 'Time': list(range(duration))+f_decom.DecomTime[i]//12, 'DecomCost': s_cost})], ignore_index=True)
    rtn['SimId'] = f_decom['SimId'].iloc[0]
    cols = rtn.columns.tolist()
    cols = cols[-1:]+cols[:-1]
    rtn = rtn[cols]
    return rtn
    

del _dcdeps, _dcschema
