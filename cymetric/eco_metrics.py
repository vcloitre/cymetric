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
    RANDOM ANALYSIS) related to the capital costs of a reactor. The overnightCost 
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
    		deviation = np.random.poisson(variance) - variance
    		begin = int(tmp.loc[('Capital', 'Begin')] + deviation)
    		duration = int(tmp.loc[('Capital', 'Duration')] + 2 * deviation)
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
        'Time'), ('EconomicInfo', (('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Fuel', 'Cost'), ('Fuel', 'WasteFee')), ('Finance','DiscountRate'))]

_fcschema = [('SimId', ts.UUID), ('TransactionId', ts.INT), ('AgentId', 
          ts.INT), ('Commodity', ts.STRING), ('Payment', ts.DOUBLE), ('Time', 
          ts.INT)]

@metric(name='FuelCost', depends=_fcdeps, schema=_fcschema)
def fuel_cost(series):
    """fuel_cost returns the cash flows related to the fuel costs for power 
    plants.
    """
    # fuel_price = 2360 # $/kg
    # see http://www.world-nuclear.org/info/Economic-Aspects/Economics-of-Nuclear-Power/
    # need to add a dictionnary with diff commodities and prices (uox, wast etc..)
    f_resources = series[0].reset_index().set_index(['ResourceId'])
    f_transactions = series[1].reset_index().set_index(['ResourceId'])
    f_ecoi = series[2].reset_index().set_index('AgentId')
    for agentId in f_ecoi.index:
    	tmp = f_ecoi.loc[agentId]
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
    tuples = [('Agent', 'Prototype'), ('Agent', 'AgentId'), ('Agent', 'ParentId'), ('Finance','ReturnOnDebt'), ('Finance','ReturnOnEquity'), ('Finance','TaxRate'), ('Finance','DiscountRate'), ('Capital', 'Begin'), ('Capital', 'Duration'), ('Capital', 'Deviation'), ('Capital', 'OvernightCost'), ('Decommissioning', 'Duration'), ('Decommissioning', 'OvernightCost'), ('OperationMaintenance', 'FixedCost'), ('OperationMaintenance', 'VariableCost'), ('Fuel', 'SupplyCost'), ('Fuel', 'WasteFee'), ('Truncation', 'Begin'), ('Truncation', 'End')]
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    rtn = pd.DataFrame(index=index)
    f_entry = series[0].reset_index()
    agentIndex = f_entry.reset_index().set_index('AgentId')['index']
    rtn = rtn.T
    rtn[('Agent', 'Prototype')] = f_entry['Prototype']
    rtn[('Agent', 'AgentId')] = f_entry['AgentId']
    rtn[('Agent', 'ParentId')] = f_entry['ParentId']
    parametersInput = 'parameters.xml'
    tree = ET.parse(parametersInput)
    root = tree.getroot()
    truncation = root.find('truncation')
    rtn[('Truncation', 'Begin')] = int(truncation.find('simulation_begin').text)
    rtn[('Truncation', 'End')] = int(truncation.find('simulation_end').text)
    finance = root.find('finance')
    if not finance == None:
    	rtn.loc[:, ('Finance', 'TaxRate')] = float(finance.find('tax_rate').text)
    	rtn.loc[:, ('Finance','ReturnOnDebt')] = float(finance.find('return_on_debt').text)
    	rtn.loc[:, ('Finance','ReturnOnEquity')] = float(finance.find('return_on_equity').text)
    	rtn.loc[:, ('Finance','DiscountRate')] = float(finance.find('discount_rate').text)
    capital = root.find('capital')
    if not capital == None:
    	rtn.loc[:, ('Capital', 'Begin')] = int(capital.find('begin').text)
    	rtn.loc[:, ('Capital', 'Duration')] = int(capital.find('duration').text)
    	rtn.loc[:, ('Capital', 'Deviation')] = float(capital.find('deviation').text)
    	rtn.loc[:, ('Capital', 'OvernightCost')] = int(capital.find('overnight_cost').text)
    decommissioning = root.find('decommissioning')
    if not decommissioning == None:
    	rtn.loc[:, ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    	rtn.loc[:, ('Decommissioning', 'OvernightCost')] = int(decommissioning.find('overnight_cost').text)
    operation_maintenance = root.find('operation_maintenance')
    if not operation_maintenance == None:
    	rtn.loc[:, ('OperationMaintenance', 'FixedCost')] = int(operation_maintenance.find('fixed').text)
    	rtn.loc[:, ('OperationMaintenance', 'VariableCost')] = int(operation_maintenance.find('variable').text)
    fuel = root.find('fuel')
    dfSupply = pd.DataFrame(index=rtn.index)
    dfSupply[('Fuel', 'SupplyCost')] = pd.Series()
    dfWaste = pd.DataFrame(index=rtn.index)
    dfWaste[('Fuel', 'WasteFee')] = pd.Series()
    if not fuel == None:
    	supply = {}
    	waste = {}
    	for type in fuel.findall('type'):
    		supply[type.find('name').text] = int(type.find('supply_cost').text)
    		waste[type.find('name').text] = int(type.find('waste_fee').text)
    	for j in rtn.index:
    		print(rtn)#test
    		print(rtn.loc[j, ('Fuel', 'SupplyCost')])#test
    		print(supply)#test
    		dfSupply.loc[j, ('Fuel', 'SupplyCost')] = supply
    		dfWaste.loc[j, ('Fuel', 'WasteFee')] = waste
    	print(rtn.loc[:, ('Fuel', 'SupplyCost')])#test
    	print(dfSupply.loc[:, ('Fuel', 'SupplyCost')])#test
    	rtn.loc[:, ('Fuel', 'SupplyCost')] = dfSupply.loc[:, ('Fuel', 'SupplyCost')]
    	rtn.loc[:, ('Fuel', 'WasteFee')] = dfWaste.loc[:, ('Fuel', 'WasteFee')]
    # discount rate is only possible at sim or reg level
    for region in root.findall('region'):
    	idRegion = int(region.find('id').text)
    	finance = region.find('finance')
    	if not finance == None:
    		returnOnDebt = float(finance.find('return_on_debt').text)
    		returnOnEquity = float(finance.find('return_on_equity').text)
    		taxRate = float(finance.find('tax_rate').text)
    		discountRate = float(finance.find('discount_rate').text)
    		rtn.loc[agentIndex[idRegion], ('Finance', 'TaxRate')] = taxRate
    		rtn.loc[agentIndex[idRegion], ('Finance','ReturnOnDebt')] = returnOnDebt
    		rtn.loc[agentIndex[idRegion],('Finance','ReturnOnEquity')] = returnOnEquity
    		rtn.loc[gent_index[idRegion], ('Finance','DiscountRate')] = discountRate
    		for idInstitution in f_entry[f_entry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('Finance', 'TaxRate')] = taxRate
    			rtn.loc[agentIndex[idInstitution], ('Finance','ReturnOnDebt')] = returnOnDebt
    			rtn.loc[agentIndex[idInstitution], ('Finance','ReturnOnEquity')] = returnOnEquity
    			rtn.loc[gent_index[idInstitution], ('Finance','DiscountRate')] = discountRate
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('Finance', 'TaxRate')] = taxRate
    				rtn.loc[agentIndex[idReactor], ('Finance','ReturnOnDebt')] = returnOnDebt
    				rtn.loc[agentIndex[idReactor], ('Finance','ReturnOnEquity')] = returnOnEquity
    				rtn.loc[gent_index[idReactor], ('Finance','DiscountRate')] = discountRate
    	capital = region.find('capital')
    	if capital is not None:
    		begin = int(capital.find('begin').text)
    		duration = int(capital.find('duration').text)
    		deviation = float(capital.find('deviation').text)
    		overnightCost = int(capital.find('overnight_cost').text)
    		rtn.loc[agentIndex[idRegion], ('Capital', 'Begin')] = begin
    		rtn.loc[agentIndex[idRegion], ('Capital', 'Duration')] = duration
    		rtn.loc[agentIndex[idRegion], ('Capital', 'Deviation')] = deviation
    		rtn.loc[agentIndex[idRegion], ('Capital', 'OvernightCost')] = overnightCost
    		for idInstitution in f_entry[f_entry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Begin')] = begin
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Deviation')] = deviation
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'OvernightCost')] = overnightCost
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Begin')] = begin
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Duration')] = duration
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Deviation')] = deviation
    				rtn.loc[agentIndex[idReactor], ('Capital', 'OvernightCost')] = overnightCost
    	decommissioning = region.find('decommissioning')
    	if decommissioning is not None:
    		duration = int(decommissioning.find('duration').text)
    		overnightCost = int(decommissioning.find('overnight_cost').text)
    		rtn.loc[agentIndex[idRegion], ('Decommissioning', 'Duration')] = duration
    		rtn.loc[agentIndex[idRegion], ('Decommissioning', 'OvernightCost')] = overnightCost
    		for idInstitution in f_entry[f_entry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'OvernightCost')] = overnightCost
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('Decommissioning', 'Duration')] = duration
    				rtn.loc[agentIndex[idReactor], ('Decommissioning', 'OvernightCost')] = overnightCost
    	operation_maintenance = region.find('operation_maintenance')
    	if operation_maintenance is not None:
    		fixed = int(operation_maintenance.find('fixed').text)
    		variable = int(operation_maintenance.find('variable').text)
    		rtn.loc[agentIndex[idRegion], ('OperationMaintenance', 'FixedCost')] = fixed
    		rtn.loc[agentIndex[idRegion], ('OperationMaintenance', 'VariableCost')] = variable
    		for idInstitution in f_entry[f_entry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'VariableCost')] = variable
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agentIndex[idReactor], ('OperationMaintenance', 'VariableCost')] = variable
    	fuel = region.find('fuel')
    	if fuel is not None:
    		supply = {}
    		waste = {}
    		for type in supply.findall('type'):
    			supply[type.find('name').text] = int(type.find('supply_cost').text)
    			waste[type.find('name').text] = int(type.find('waste_fee').text)
    		rtn.loc[agentIndex[idRegion], ('Fuel', 'SupplyCost')] = supply
    		rtn.loc[agentIndex[idRegion], ('Fuel', 'WasteFee')] = waste
    		for idInstitution in f_entry[f_entry.ParentId==idRegion]['AgentId'].tolist():
    			rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')] = supply
    			rtn.loc[agentIndex[idInstitution], ('Fuel', 'WasteFee')] = waste
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idReactor], ('Fuel', 'WasteFee')] = waste
    	for institution in region.findall('institution'):
    		idInstitution = int(institution.find('id').text)
    		finance = institution.find('finance')
    		if finance is not None:
    			returnOnDebt = float(finance.find('return_on_debt').text)
    			returnOnEquity = float(finance.find('return_on_equity').text)
    			taxRate = float(finance.find('tax_rate').text)
    			discountRate = float(finance.find('discount_rate').text)
    			rtn.loc[agentIndex[idInstitution], ('Finance', 'TaxRate')] = taxRate
    			rtn.loc[agentIndex[idInstitution], ('Finance','ReturnOnDebt')] = returnOnDebt
    			rtn.loc[agentIndex[idInstitution],('Finance','ReturnOnEquity')] = returnOnEquity
    			rtn.loc[gent_index[idInstitution], ('Finance','DiscountRate')] = discountRate
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('Finance', 'TaxRate')] = taxRate
    				rtn.loc[agentIndex[idReactor], ('Finance','ReturnOnDebt')] = returnOnDebt
    				rtn.loc[agentIndex[idReactor], ('Finance','ReturnOnEquity')] = returnOnEquity
    				rtn.loc[gent_index[idReactor], ('Finance','DiscountRate')] = discountRate
    		capital = institution.find('capital')
    		if capital is not None:
    			begin = int(capital.find('begin').text)
    			duration = int(capital.find('duration').text)
    			deviation = float(capital.find('deviation').text)
    			overnightCost = int(capital.find('overnight_cost').text)
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Begin')] = begin
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'Deviation')] = deviation
    			rtn.loc[agentIndex[idInstitution], ('Capital', 'OvernightCost')] = overnightCost
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Begin')] = begin
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Duration')] = duration
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Deviation')] = deviation
    				rtn.loc[agentIndex[idReactor], ('Capital', 'OvernightCost')] = overnightCost
    		decommissioning = institution.find('decommissioning')
    		if decommissioning is not None:
    			duration = int(decommissioning.find('duration').text)
    			overnightCost = int(decommissioning.find('overnight_cost').text)
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'Duration')] = duration
    			rtn.loc[agentIndex[idInstitution], ('Decommissioning', 'OvernightCost')] = overnightCost
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('Decommissioning', 'Duration')] = duration
    				rtn.loc[agentIndex[idReactor], ('Decommissioning', 'OvernightCost')] = overnightCost
    		operation_maintenance = institution.find('operation_maintenance')
    		if operation_maintenance is not None:
    			fixed = int(operation_maintenance.find('fixed').text)
    			variable = int(operation_maintenance.find('variable').text)
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'FixedCost')] = fixed
    			rtn.loc[agentIndex[idInstitution], ('OperationMaintenance', 'VariableCost')] = variable
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('OperationMaintenance', 'FixedCost')] = fixed
    				rtn.loc[agentIndex[idReactor], ('OperationMaintenance', 'VariableCost')] = variable
    		fuel = institution.find('fuel')
    		if fuel is not None:
    			supply = {}
    			waste = {}
    			for type in supply.findall('type'):
    				supply[type.find('name').text] = int(type.find('supply_cost').text)
    				waste[type.find('name').text] = int(type.find('waste_fee').text)
    			rtn.loc[agentIndex[idInstitution], ('Fuel', 'SupplyCost')] = supply
    			rtn.loc[agentIndex[idInstitution], ('Fuel', 'WasteFee')] = waste
    			for idReactor in f_entry[f_entry.ParentId==idInstitution]['AgentId'].tolist():
    				rtn.loc[agentIndex[idReactor], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idReactor], ('Fuel', 'WasteFee')] = waste
    		for reactor in institution.findall('reactor'):
    			idReactor = int(reactor.find('id').text)
    			capital = reactor.find('capital')
    			if capital is not None:
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Begin')] = int(capital.find('begin').text)
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Duration')] = int(capital.find('duration').text)
    				rtn.loc[agentIndex[idReactor], ('Capital', 'Deviation')] = float(capital.find('deviation').text)
    				rtn.loc[agentIndex[idReactor], ('Capital', 'OvernightCost')] = int(capital.find('overnight_cost').text)
    			operation_maintenance = reactor.find('operation_maintenance')
    			if operation_maintenance is not None:
    				rtn.loc[agentIndex[idReactor], ('OperationMaintenance', 'FixedCost')] = int(operation_maintenance.find('fixed').text)
    				rtn.loc[agentIndex[idReactor], ('OperationMaintenance', 'VariableCost')] = int(operation_maintenance.find('variable').text)
    			fuel = reactor.find('fuel')
    			if fuel is not None:
    				supply = {}
    				waste = {}
    				for type in supply.findall('type'):
    					supply[type.find('name').text] = int(type.find('supply_cost').text)
    					waste[type.find('name').text] = int(type.find('waste_fee').text)
    				rtn.loc[agentIndex[idReactor], ('Fuel', 'SupplyCost')] = supply
    				rtn.loc[agentIndex[idReactor], ('Fuel', 'WasteFee')] = waste
    			decommissioning = reactor.find('decommissioning')
    			if decommissioning is not None:
    				rtn.loc[agentIndex[idReactor], ('Decommissioning', 'Duration')] = int(decommissioning.find('duration').text)
    				rtn.loc[agentIndex[idReactor], ('Decommissioning', 'OvernightCost')] = int(decommissioning.find('overnight_cost').text)
    return rtn
	
del _eideps, _eischema