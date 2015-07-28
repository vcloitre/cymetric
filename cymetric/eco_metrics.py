#####economic metrics for nuclear power plants#####

from __future__ import print_function, unicode_literals

import inspect

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import math
import os


try:
    from cymetric.metrics import metric
    from cymetric import cyclus
    from cymetric import schemas
    from cymetric import typesystem as ts
    from cymetric import tools
    from cymetric.evaluator import register_metric
    from cymetric.eco_inputs import capital_shape, rapid_cap_begin, rapid_cap_duration, slow_cap_begin, slow_cap_duration, default_cap_begin, default_cap_duration, default_cap_overnight, default_cap_shape, default_discount_rate
except ImportError:
    # some wacky CI paths prevent absolute importing, try relative
    from .metrics import metric
    from . import cyclus
    from . import schemas
    from . import typesystem as ts
    from . import tools
    from .evaluator import register_metric
    from .eco_inputs import capital_shape, rapid_cap_begin, rapid_cap_duration, slow_cap_begin, slow_cap_duration, default_cap_begin, default_cap_duration, default_cap_overnight, default_cap_shape, default_discount_rate

xml_inputs = 'parameters.xml' # temporary solution : always store an xml file in your working directory that you will have to use. This file have to be known

## The actual metrics ##


_ccdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Value'), 'Time'), ('AgentEntry', ('AgentId', 'ParentId', 'Spec'), 'EnterTime'), ('Info', ('InitialYear', 'InitialMonth'), 'Duration'), ('EconomicInfo', (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Capital', 'Begin'), ('Capital', 'Duration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost')), ('Finance','DiscountRate'))]

_ccschema = [('SimId', ts.UUID), ('AgentId', ts.INT),
             ('Time', ts.INT), ('Payment', ts.DOUBLE)]

@metric(name='CapitalCost', depends=_ccdeps, schema=_ccschema)
def capital_cost(series):
    """cap_cost returns the cash flows per YEAR (MAYBE BETTER PER MONTH FOR THE 
    RANDOM ANALYSIS) related to the capital costs of a reactor. The overnight_cost 
    cost comes from WEO 2014 (cost in the US). The timeframe for the payment of 
    the capital costs is drawn from of a graph from EDF. Next steps could be 
    make the price depends on the reactor technology and make the payment more 
    realistic, ie include interest rates, improve the linear model and finally 
    be able to fetch data"""
    f_power = series[0].reset_index()
    f_entry = series[1].reset_index()
    f_info = series[2].reset_index()
    f_ecoi = series[3].reset_index()
    tuples = (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Capital', 'Begin'), ('Capital', 'Duration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Finance','DiscountRate'))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    f_ecoi.columns = index
    f_ecoi = f_ecoi.set_index(('Agent', 'AgentId'))
    simDuration = f_info['Duration'].iloc[0]
    #std=3.507*12
    #var=std**2
    f_entry = pd.DataFrame([f_entry.EnterTime, f_entry.AgentId]).transpose()
    f_entry = f_entry.set_index(['AgentId'])
    agentIds = f_ecoi.index
    rtn = pd.DataFrame()
    for id in agentIds:
    	tmp = f_ecoi.loc[id]
    	if 'REACTOR' in tmp.loc[('Agent', 'Prototype')].upper():
    		deviation = tmp.loc[('Capital', 'Deviation')]
    		variance = deviation ** 2
    		deviation = int(np.random.poisson(variance) - variance)
    		begin = tmp.loc[('Capital', 'Begin')] + deviation
    		duration = tmp.loc[('Capital', 'Duration')] + 2 * deviation
    		overnightCost = tmp.loc[('Capital', 'OvernightCost')]
    		cashFlowShape = capital_shape(begin, duration)
    		powerCapacity = max(f_power[f_power.AgentId==id]['Value'])
    		discountRate = tmp.loc[('Finance','DiscountRate')]
    		cashFlow = np.around(cashFlowShape * overnightCost * powerCapacity, 3)
    		cashFlow *= ((1 + discountRate) ** math.ceil(duration / 12) - 1) / (discountRate * math.ceil(duration / 12))
    		tmp = pd.DataFrame({'AgentId': id, 'Time': pd.Series(list(range(duration + 1))) - begin + f_entry.EnterTime[id], 'Payment' : cashFlow})
    		rtn = pd.concat([rtn, tmp], ignore_index=True)
    rtn['SimId'] = f_power['SimId'].iloc[0]
    subset = rtn.columns.tolist()
    subset = subset[3:] + subset[:1] + subset[2:3] + subset[1:2]
    rtn = rtn[subset]
    rtn = rtn[rtn['Time'].apply(lambda x: x >= 0 and x < simDuration)]
    rtn = rtn.reset_index()
    del rtn['index']
    return rtn

del _ccdeps, _ccschema


_fcdeps = [('Resources', ('SimId', 'ResourceId'), 'Quantity'), ('Transactions',
        ('SimId', 'TransactionId', 'ReceiverId', 'ResourceId', 'Commodity'), 
        'Time')]

_fcschema = [('SimId', ts.UUID), ('TransactionId', ts.INT), ('AgentId', 
          ts.INT), ('Commodity', ts.STRING), ('Payment', ts.DOUBLE), ('Time', 
          ts.INT)]

@metric(name='FuelCost', depends=_fcdeps, schema=_fcschema)
def fuel_cost(series):
    """fuel_cost returns the cash flows related to the fuel costs for power 
    plants.
    """
    fuel_price = 2360 # $/kg
    # see http://www.world-nuclear.org/info/Economic-Aspects/Economics-of-Nuclear-Power/
    # need to add a dictionnary with diff commodities and prices (uox, wast etc..)
    f_resources = series[0].reset_index().set_index(['ResourceId'])
    f_transactions = series[1].reset_index().set_index(['ResourceId'])
    f_transactions['Quantity'] = f_resources['Quantity']
    f_transactions['Payment'] = f_transactions['Quantity'] * fuel_price
    # * (f_transactions['Commodity']=='uox' or f_transactions['Commodity']=='mox')
    del f_transactions['Quantity']
    rtn = f_transactions.reset_index()
    subset = rtn.columns.tolist()
    subset = subset[1:5]+subset[6:]+subset[5:6]
    rtn = rtn[subset]
    rtn.columns = ['SimId', 'TransactionId', 'AgentId', 'Commodity', 'Payment', 'Time']
    return rtn

del _fcdeps, _fcschema


_dcdeps = [ ('TimeSeriesPower', ('SimId', 'AgentId'), 'Value'),
			('AgentEntry', ('EnterTime', 'Lifetime', 'AgentId'), 'Spec'),
			('Info', ('InitialYear', 'InitialMonth'), 'Duration')]

_dcschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('Payment',
          ts.DOUBLE), ('Time', ts.INT)]

@metric(name='DecommissioningCost', depends=_dcdeps, schema=_dcschema)
def decommissioning_cost(series):
    """decom
    """
    cost = 750000 # decommission cost in $/MW d'Haeseler
    duration = 150 # decommission lasts about 15 yrs
    if series[0].empty:
    	return pd.DataFrame()
    f_power = series[0].reset_index()
    f_power = f_power[f_power['Value'].apply(lambda x: x > 0)]
    f_entry = series[1].reset_index()
    f_info = series[2].reset_index()
    sim_duration = f_info['Duration'].iloc[0]
    f_entry = f_entry[(f_entry['EnterTime'] + f_entry['Lifetime']).apply(lambda x: x < sim_duration)] # only reactors that will be decommissioned
    id_reactors = f_entry[f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
    rtn = pd.DataFrame()
    for i in id_reactors:
        s_cost = capital_shape(duration // 2, duration-1, 'triangle')
        s_cost = s_cost * f_power[f_power.AgentId==i]['Value'].iloc[0] * cost
        entrytime = f_entry[f_entry.AgentId==i]['EnterTime'].iloc[0]
        lifetime = f_entry[f_entry.AgentId==i]['Lifetime'].iloc[0]
        rtn = pd.concat([rtn,pd.DataFrame({'AgentId': i, 'Time': list(range(lifetime + entrytime, lifetime + entrytime + duration)), 'Payment': s_cost})], ignore_index=True)
    rtn['SimId'] = f_power['SimId'].iloc[0]
    subset = rtn.columns.tolist()
    subset = subset[-1:]+subset[:-1]
    rtn = rtn[subset]
    return rtn[rtn['Time'].apply(lambda x: x >= 0 and x < sim_duration)]

del _dcdeps, _dcschema


_omdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Time'), 'Value')]

_omschema = [('SimId', ts.UUID), ('AgentId', ts.INT), ('Time', ts.INT), 
          ('Payment', ts.DOUBLE)]

@metric(name='OperationMaintenance', depends=_omdeps, schema=_omschema)
def operation_maintenance(series):
    """O&M
    """
    cost = 15 # $/MWh
    rtn = series[0].reset_index()
    #rtn.Time=rtn.Time//12
    #rtn =  rtn.drop_duplicates(subset=['AgentId', 'Time'], take_last=True) useful when the Time is in years but useless when months
    rtn['Payment'] = rtn['Value'] * 8760 / 12 * cost
    rtn = rtn.reset_index()
    del rtn['Value'], rtn['index']
    return rtn

del _omdeps, _omschema


_eideps = [('AgentEntry', ('AgentId', 'Prototype'), 'ParentId')]

_eischema = [('AgentId', ts.INT), ('Prototype', ts.STRING), ('ParentId', ts.INT), ('BeginMonth', ts.INT), ('EndMonth', ts.INT), ('DiscountRate', ts.DOUBLE)]
		
@metric(name='EconomicInfo', depends=_eideps, schema=_eischema)
def economic_info(series):
    """Write the economic parameters in the database
    """
    tuples = [('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Agent', 'ParentId'), ('Finance','ReturnOnDebt'), ('Finance','ReturnOnEquity'), ('Finance','TaxRate'), ('Finance','DiscountRate'), ('Capital', 'Begin'), ('Capital', 'Duration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Decommissioning', 'Duration'), ('Decommissioning', 'OvernightCost'), ('OperationMaintenance', 'FixedCost'), ('OperationMaintenance', 'VariableCost'), ('Fuel', 'Cost'), ('Fuel', 'WasteFee'), ('Truncation', 'Begin'), ('Truncation', 'End')]
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    rtn = pd.DataFrame(index=index)
    ser = pd.Series(False, index=['finance', 'capital', 'decommissioning', 'operationmMaintenance', 'fuel'])
    f_entry = series[0].reset_index()
    agent_index = f_entry.reset_index().set_index('AgentId')['index']
    rtn = rtn.T
    rtn[('Agent', 'Prototype')] = f_entry['Prototype']
    rtn[('Agent', 'AgentId')] = f_entry['AgentId']
    rtn[('Agent', 'ParentId')] = f_entry['ParentId']
    xml_inputs = 'parameters.xml'
    tree = ET.parse(xml_inputs)
    root = tree.getroot()
    truncation = root.find('truncation')
    rtn[('Truncation', 'Begin')] = int(truncation.find('simulation_begin').text)
    rtn[('Truncation', 'End')] = int(truncation.find('simulation_end').text)
    finance = root.find('finance')
    if not finance == None:
    	ser['finance'] = True
    	rtn.loc[:, ('Finance', 'TaxRate')] = float(finance.find('tax_rate').text)
    	rtn.loc[:, ('Finance','ReturnOnDebt')] = float(finance.find('return_on_debt').text)
    	rtn.loc[:, ('Finance','ReturnOnEquity')] = float(finance.find('return_on_equity').text)
    capital = root.find('capital')
    if not capital == None:
    	ser['capital'] = True
    	rtn.loc[:, ('Capital', 'Begin')] = int(capital.find('begin').text)
    	rtn.loc[:, ('Capital', 'Duration')] = int(capital.find('duration').text)
    	rtn.loc[:, ('Capital', 'Deviation')] = float(capital.find('deviation').text)
    	rtn.loc[:, ('Capital', 'OvernightCost')] = int(capital.find('overnight_cost').text)
    decommissioning = root.find('decommissioning')
    if not decommissioning == None:
    	ser['decommissioning'] = True
    	rtn.loc[:, ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    	rtn.loc[:, ('Decommissioning', 'OvernightCost')] = int(decommissioning.find('overnight_cost').text)
    operation_maintenance = root.find('operation_maintenance')
    if not operation_maintenance == None:
    	ser['operationMaintenance'] = True
    	rtn.loc[:, ('OperationMaintenance', 'FixedCost')] = int(operation_maintenance.find('fixed').text)
    	rtn.loc[:, ('OperationMaintenance', 'VariableCost')] = int(operation_maintenance.find('variable').text)
    fuel = root.find('fuel')
    if not fuel == None:
    	ser['fuel'] = True
    	rtn.loc[:, ('Fuel', 'Cost')] = int(fuel.find('cost').text)
    	rtn.loc[:, ('Fuel', 'WasteFee')] = int(fuel.find('waste_fee').text)
    # discount rate is only possible at sim or reg level
    for region in root.findall('region'):
    	id_region = int(region.find('id').text)
    	if 'finance' in ser[ser==False]:
    		finance = region.find('finance')
    		if not finance == None:
    			return_on_debt = float(finance.find('return_on_debt').text)
    			return_on_equity = float(finance.find('return_on_equity').text)
    			tax_rate = float(finance.find('tax_rate').text)
    			rtn.loc[agent_index[id_region], ('Finance', 'TaxRate')] = tax_rate
    			rtn.loc[agent_index[id_region], ('Finance','ReturnOnDebt')] = return_on_debt
    			rtn.loc[agent_index[id_region],('Finance','ReturnOnEquity')] = return_on_equity
    			for id_institution in f_entry[f_entry.ParentId==id_region]['AgentId'].tolist():
    				rtn.loc[agent_index[id_institution], ('Finance', 'TaxRate')] = tax_rate
    				rtn.loc[agent_index[id_institution], ('Finance','ReturnOnDebt')] = return_on_debt
    				rtn.loc[agent_index[id_institution], ('Finance','ReturnOnEquity')] = return_on_equity
    				for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    					rtn.loc[agent_index[id_reactor], ('Finance', 'TaxRate')] = tax_rate
    					rtn.loc[agent_index[id_reactor], ('Finance','ReturnOnDebt')] = return_on_debt
    					rtn.loc[agent_index[id_reactor], ('Finance','ReturnOnEquity')] = return_on_equity
    	if 'capital' in ser[ser==False]:
    		capital = region.find('capital')
    		if capital is not None:
    			begin = int(capital.find('begin').text)
    			duration = int(capital.find('duration').text)
    			deviation = float(capital.find('deviation').text)
    			overnight_cost = int(capital.find('overnight_cost').text)
    			rtn.loc[agent_index[id_region], ('Capital', 'Begin')] = begin
    			rtn.loc[agent_index[id_region], ('Capital', 'Duration')] = duration
    			rtn.loc[agent_index[id_region], ('Capital', 'Deviation')] = deviation
    			rtn.loc[agent_index[id_region], ('Capital', 'OvernightCost')] = overnight_cost
    			for id_institution in f_entry[f_entry.ParentId==id_region]['AgentId'].tolist():
    				rtn.loc[agent_index[id_institution], ('Capital', 'Begin')] = begin
    				rtn.loc[agent_index[id_institution], ('Capital', 'Duration')] = duration
    				rtn.loc[agent_index[id_institution], ('Capital', 'Deviation')] = deviation
    				rtn.loc[agent_index[id_institution], ('Capital', 'OvernightCost')] = overnight_cost
    				for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    					rtn.loc[agent_index[id_reactor], ('Capital', 'Begin')] = begin
    					rtn.loc[agent_index[id_reactor], ('Capital', 'Duration')] = duration
    					rtn.loc[agent_index[id_reactor], ('Capital', 'Deviation')] = deviation
    					rtn.loc[agent_index[id_reactor], ('Capital', 'OvernightCost')] = overnight_cost
    	if 'decommissioning' in ser[ser==False]:
    		decommissioning = region.find('decommissioning')
    		if decommissioning is not None:
    			duration = int(decommissioning.find('duration').text)
    			overnight_cost = int(decommissioning.find('overnight_cost').text)
    			rtn.loc[agent_index[id_region], ('Decommissioning', 'Duration')] = duration
    			rtn.loc[agent_index[id_region], ('Decommissioning', 'OvernightCost')] = overnight_cost
    			for id_institution in f_entry[f_entry.ParentId==id_region]['AgentId'].tolist():
    				rtn.loc[agent_index[id_institution], ('Decommissioning', 'Duration')] = duration
    				rtn.loc[agent_index[id_institution], ('Decommissioning', 'OvernightCost')] = overnight_cost
    				for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    					rtn.loc[agent_index[id_reactor], ('Decommissioning', 'Duration')] = duration
    					rtn.loc[agent_index[id_reactor], ('Decommissioning', 'OvernightCost')] = overnight_cost
    	if 'operationMaintenance' in ser[ser==False]:
    		operation_maintenance = region.find('operation_maintenance')
    		if operation_maintenance is not None:
    			fixed = int(operation_maintenance.find('fixed').text)
    			variable = int(operation_maintenance.find('variable').text)
    			rtn.loc[agent_index[id_region], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agent_index[id_region], ('OperationMaintenance', 'VariableCost')] = variable
    			for id_institution in f_entry[f_entry.ParentId==id_region]['AgentId'].tolist():
    				rtn.loc[agent_index[id_institution], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agent_index[id_institution], ('OperationMaintenance', 'VariableCost')] = variable
    				for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    					rtn.loc[agent_index[id_reactor], ('OperationMaintenance', 'FixedCost')] = fixed
    					rtn.loc[agent_index[id_reactor], ('OperationMaintenance', 'VariableCost')] = variable
    	if 'fuel' in ser[ser==False]:
    		fuel = region.find('fuel')
    		if fuel is not None:
    			cost = int(fuel.find('cost').text)
    			waste_fee = int(fuel.find('waste_fee').text)
    			rtn.loc[agent_index[id_region], ('Fuel', 'Cost')] = cost
    			rtn.loc[agent_index[id_region], ('Fuel', 'WasteFee')] = waste_fee
    			for id_institution in f_entry[f_entry.ParentId==id_region]['AgentId'].tolist():
    				rtn.loc[agent_index[id_institution], ('Fuel', 'Cost')] = cost
    				rtn.loc[agent_index[id_institution], ('Fuel', 'WasteFee')] = waste_fee
    				for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    					rtn.loc[agent_index[id_reactor], ('Fuel', 'Cost')] = cost
    					rtn.loc[agent_index[id_reactor], ('Fuel', 'WasteFee')] = waste_fee
    	for institution in region.findall('institution'):
    		id_institution = int(institution.find('id').text)
    		finance = institution.find('finance')
    		if finance is not None:
    			return_on_debt = float(finance.find('return_on_debt').text)
    			return_on_equity = float(finance.find('return_on_equity').text)
    			tax_rate = float(finance.find('tax_rate').text)
    			rtn.loc[agent_index[id_institution], ('Finance', 'TaxRate')] = tax_rate
    			rtn.loc[agent_index[id_institution], ('Finance','ReturnOnDebt')] = return_on_debt
    			rtn.loc[agent_index[id_institution],('Finance','ReturnOnEquity')] = return_on_equity
    			for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    				rtn.loc[agent_index[id_reactor], ('Finance', 'TaxRate')] = tax_rate
    				rtn.loc[agent_index[id_reactor], ('Finance','ReturnOnDebt')] = return_on_debt
    				rtn.loc[agent_index[id_reactor], ('Finance','ReturnOnEquity')] = return_on_equity
    		capital = institution.find('capital')
    		if capital is not None:
    			begin = int(capital.find('begin').text)
    			duration = int(capital.find('duration').text)
    			deviation = float(capital.find('deviation').text)
    			overnight_cost = int(capital.find('overnight_cost').text)
    			rtn.loc[agent_index[id_institution], ('Capital', 'Begin')] = begin
    			rtn.loc[agent_index[id_institution], ('Capital', 'Duration')] = duration
    			rtn.loc[agent_index[id_institution], ('Capital', 'Deviation')] = deviation
    			rtn.loc[agent_index[id_institution], ('Capital', 'OvernightCost')] = overnight_cost
    			for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    				rtn.loc[agent_index[id_reactor], ('Capital', 'Begin')] = begin
    				rtn.loc[agent_index[id_reactor], ('Capital', 'Duration')] = duration
    				rtn.loc[agent_index[id_reactor], ('Capital', 'Deviation')] = deviation
    				rtn.loc[agent_index[id_reactor], ('Capital', 'OvernightCost')] = overnight_cost
    		decommissioning = institution.find('decommissioning')
    		if decommissioning is not None:
    			duration = int(decommissioning.find('duration').text)
    			overnight_cost = int(decommissioning.find('overnight_cost').text)
    			rtn.loc[agent_index[id_institution], ('Decommissioning', 'Duration')] = duration
    			rtn.loc[agent_index[id_institution], ('Decommissioning', 'OvernightCost')] = overnight_cost
    			for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    				rtn.loc[agent_index[id_reactor], ('Decommissioning', 'Duration')] = duration
    				rtn.loc[agent_index[id_reactor], ('Decommissioning', 'OvernightCost')] = overnight_cost
    		operation_maintenance = institution.find('operation_maintenance')
    		if operation_maintenance is not None:
    			fixed = int(operation_maintenance.find('fixed').text)
    			variable = int(operation_maintenance.find('variable').text)
    			rtn.loc[agent_index[id_institution], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agent_index[id_institution], ('OperationMaintenance', 'VariableCost')] = variable
    			for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    				rtn.loc[agent_index[id_reactor], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agent_index[id_reactor], ('OperationMaintenance', 'VariableCost')] = variable
    		fuel = institution.find('fuel')
    		if fuel is not None:
    			cost = int(fuel.find('cost').text)
    			waste_fee = int(fuel.find('waste_fee').text)
    			rtn.loc[agent_index[id_institution], ('Fuel', 'Cost')] = cost
    			rtn.loc[agent_index[id_institution], ('Fuel', 'WasteFee')] = waste_fee
    			for id_reactor in f_entry[f_entry.ParentId==id_institution]['AgentId'].tolist():
    				rtn.loc[agent_index[id_reactor], ('Fuel', 'Cost')] = cost
    				rtn.loc[agent_index[id_reactor], ('Fuel', 'WasteFee')] = waste_fee
    		for reactor in institution.findall('reactor'):
    			id_reactor = int(reactor.find('id').text)
    			capital = reactor.find('capital')
    			if capital is not None:
    				rtn.loc[agent_index[id_reactor], ('Capital', 'Begin')] = int(capital.find('begin').text)
    				rtn.loc[agent_index[id_reactor], ('Capital', 'Duration')] = int(capital.find('duration').text)
    				rtn.loc[agent_index[id_reactor], ('Capital', 'Deviation')] = float(capital.find('deviation').text)
    				rtn.loc[agent_index[id_reactor], ('Capital', 'OvernightCost')] = int(capital.find('overnight_cost').text)
    			operation_maintenance = reactor.find('operation_maintenance')
    			if operation_maintenance is not None:
    				rtn.loc[agent_index[id_reactor], ('OperationMaintenance', 'FixedCost')] = int(operation_maintenance.find('fixed').text)
    				rtn.loc[agent_index[id_reactor], ('OperationMaintenance', 'VariableCost')] = int(operation_maintenance.find('variable').text)
    			fuel = reactor.find('fuel')
    			if fuel is not None:
    				rtn.loc[agent_index[id_reactor], ('Fuel', 'Cost')] = int(fuel.find('cost').text)
    				rtn.loc[agent_index[id_reactor], ('Fuel', 'WasteFee')] = int(fuel.find('waste_fee').text)
    			decommissioning = reactor.find('decommissioning')
    			if decommissioning is not None:
    				rtn.loc[agent_index[id_reactor], ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    				rtn.loc[agent_index[id_reactor], ('Decommissioning', 'OvernightCost')] = int(decommissioning.find('overnight_cost').text)
    return rtn
	
del _eideps, _eischema