"""Functions dedicated to visualization of economic calculations. It thus
provides various functions to plot metrics calculated in eco_metrics.py
The goal is to offer many options in the plotting of cash flows. For instance, filtering by region, facility type enable to compare different region policies (i.e. to compare different the impact of the variation of financial parameters on the costs).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cymetric.tools import dbopen
from cymetric.evaluator import Evaluator
from cymetric.eco_inputs import default_cap_overnight, default_discount_rate, default_fuel_price, actualization_vector
import warnings
import os
import xml.etree.ElementTree as ET

xml_inputs = 'parameters.xml' # temporary solution : always store an xml file in your working directory that you will have to use. This file have to be known
        
####################################################################
# Calculation of average, annual and levalized cost of electricity #
####################################################################

# Reactor level

def annual_costs(output_db, reactor_id, capital=True):
    """Given a reactor's AgentId, calculate total costs per year over the 
    lifetime of the reactor.
    """
    db = dbopen(output_db)
    evaler = Evaluator(db)
    f_info = evaler.eval('Info').reset_index()
    duration = f_info.loc[0, 'Duration']
    initial_year = f_info.loc[0, 'InitialYear']
    initial_month = f_info.loc[0, 'InitialMonth']
    f_entry = evaler.eval('AgentEntry').reset_index()
    commissioning = f_entry[f_entry.AgentId==reactor_id]['EnterTime'].iloc[0]
    f_capital = evaler.eval('CapitalCost').reset_index()
    f_capital = f_capital[f_capital.AgentId==reactor_id]
    f_capital = f_capital.groupby('Time').sum()
    costs = pd.DataFrame({'Capital' : f_capital['Payment']}, index=list(range(0, duration)))
    f_decom = evaler.eval('DecommissioningCost').reset_index()
    if not f_decom.empty:
    	f_decom = f_decom[f_decom.AgentId==reactor_id]
    	f_decom = f_decom.groupby('Time').sum()
    	costs['Decommissioning'] = f_decom['Payment']
    f_OM = evaler.eval('OperationMaintenance').reset_index()
    f_OM = f_OM[f_OM.AgentId==reactor_id]
    f_OM = f_OM.groupby('Time').sum()
    costs['OandM'] = f_OM['Payment']
    waste_disposal = 1
    f_fuel = evaler.eval('FuelCost').reset_index()
    f_fuel = f_fuel[f_fuel.AgentId==reactor_id]
    f_fuel = f_fuel.groupby('Time').sum()
    costs['Fuel'] = f_fuel['Payment']
    costs = costs.fillna(0)
    costs['Year'] = (costs.index + initial_month - 1) // 12 + initial_year
    if not capital:
    	del costs['Capital']
    costs = costs.groupby('Year').sum()
    return costs
    
def annual_costs_present_value(output_db, reactor_id, capital=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = annual_costs(output_db, reactor_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
   
def average_cost(output_db, reactor_id, capital=True):
    """Given a reactor's AgentId, gather all annual costs and average it over
     the lifetime of the reactor.
    """
    db = dbopen(output_db)
    evaler = Evaluator(db)
    f_power = evaler.eval('TimeSeriesPower').reset_index()
    power_generated = sum(f_power[f_power.AgentId==reactor_id]['Value']) * 8760 / 12
    return annual_costs(output_db, reactor_id, capital).sum().sum() / power_generated
    
def accumulate_capital(output_db, reactor_id):
	"""-expenditures + income
	"""
	costs = - annual_costs(output_db, reactor_id).sum(axis=1)
	power_gen = power_generated(output_db, reactor_id) * lcoe(output_db, reactor_id)
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
    
def lcoe(output_db, reactor_id, capital=True):
	"""More efficient than lcoe (~15% faster) and easier to understand
	"""
	costs = annual_costs(output_db, reactor_id, capital)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	initial_month = f_info['InitialMonth'].iloc[0]
	initial_year = f_info['InitialYear'].iloc[0]
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power.AgentId==reactor_id]
	f_power['Date'] = pd.Series(f_power.loc[:, 'Time']).apply(lambda x: (x + initial_month - 1) // 12 + initial_year)
	del f_power['SimId']
	costs['Power'] = f_power.groupby('Date').sum()['Value'] * 8760 / 12
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated 
	
def period_costs(output_db, reactor_id, period=20, capital=True):
	"""Same as below but at a reactor level
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	initial_month = f_info['InitialMonth'].iloc[0]
	initial_year = f_info['InitialYear'].iloc[0]
	sim_duration = f_info['Duration'].iloc[0]
	f_entry = evaler.eval('AgentEntry').reset_index()
	f_entry = f_entry[f_entry.ParentId==reactor_id]
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power['AgentId']==reactor_id]
	f_power['Date'] = pd.Series(f_power.loc[:, 'Time']).apply(lambda x: (x + initial_month - 1) // 12 + initial_year)
	del f_power['SimId']
	f_power = f_power.groupby('Date').sum()
	f_capital = evaler.eval('CapitalCost').reset_index()
	f_capital = f_capital[f_capital['AgentId']==reactor_id].set_index('Time')
	f_capital = f_capital['Payment'] # other columns are useless
	f_decom = evaler.eval('DecommissioningCost').reset_index()
	f_decom = f_decom[f_decom['AgentId']==reactor_id].set_index('Time')
	f_decom = f_decom['Payment']
	f_OM = evaler.eval('OperationMaintenance').reset_index()
	f_OM = f_OM[f_OM['AgentId']==reactor_id].set_index('Time')
	f_OM = f_OM['Payment']
	f_fuel = evaler.eval('FuelCost').reset_index()
	f_fuel = f_fuel[f_fuel['AgentId']==reactor_id].set_index('Time')
	f_fuel = f_fuel['Payment']
	if capital:
		total = pd.concat([f_capital, f_decom, f_OM, f_fuel])
	else:
		total = pd.concat([f_decom, f_OM, f_fuel])
	total = total.reset_index()
	total['Date'] = pd.Series(total['Time']).apply(lambda x: (x + initial_month - 1) // 12 + initial_year)
	total = total.groupby('Date').sum()
	total['Power'] = f_power['Value']
	total['Power2'] = pd.Series()
	total['Payment2']= pd.Series()
	total = pd.concat([total, pd.DataFrame(index=list(range(initial_year, initial_year + sim_duration // 12 + 1)))],axis=1)
	total = total.fillna(0)
	for i in range(initial_year, initial_year + period):	
		total.loc[period // 2 + initial_year, 'Power2'] += total.loc[i, 'Power'] * 8760/12 / (1 + default_discount_rate) ** (i - period // 2 - initial_year)
		total.loc[period // 2 + initial_year, 'Payment2'] += total.loc[i, 'Payment'] / (1 + default_discount_rate) ** (i - period // 2 - initial_year)
	for j in range(period // 2 + initial_year + 1, sim_duration // 12 + initial_year - period // 2):
		total.loc[j, 'Power2'] = total.loc[j - 1, 'Power2'] * (1 + default_discount_rate) - total.loc[j - 1 - period // 2, 'Power'] * 8760/12 * (1 + default_discount_rate) ** (period // 2 + 1) + total.loc[j + period // 2, 'Power'] * 8760/12 / (1 + default_discount_rate) ** (period // 2)
		total.loc[j, 'Payment2'] = total.loc[j - 1, 'Payment2'] * (1 + default_discount_rate) - total.loc[j - 1 - period // 2, 'Payment'] * (1 + default_discount_rate) ** (period // 2 + 1) + total.loc[j + period // 2, 'Payment'] / (1 + default_discount_rate) ** (period // 2)
		#tmp['WasteManagement'][j] = pd.Series()
	rtn = pd.DataFrame({'Costs (billion $)' : total['Payment2'] / 10 ** 9,  'Power (MWh)' : total['Power2'], 'Ratio' : total['Payment2'] / total['Power2']})
	rtn.index.name = 'Time'
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = rtn['Ratio'] * actualization
	return rtn
   
def power_generated(output_db, reactor_id):
	"""
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_power = evaler.eval('TimeSeriesPower').reset_index()	
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	f_power = f_power[f_power['AgentId']==reactor_id]
	f_power['Year'] = (f_power['Time'] + initial_month - 1) // 12 + initial_year
	f_power = f_power.groupby('Year').sum()
	return f_power['Value'] * 8760 / 12
   
# Institution level
    
def institution_annual_costs(output_db, institution_id, capital=True, truncate=True):
	"""reactors annual costs for a given institution returned as a pandas DataFrame containing total annual costs for each reactor id
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	f_entry = f_entry[f_entry.ParentId==institution_id]
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_reactor = f_entry[f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_capital = evaler.eval('CapitalCost').reset_index()
	f_capital = f_capital[f_capital['AgentId'].apply(lambda x: x in id_reactor)]
	mini = min(f_capital['Time'])
	f_capital = f_capital.groupby('Time').sum()
	costs = pd.DataFrame({'Capital' : f_capital['Payment']}, index=list(range(mini, duration)))
	f_decom = evaler.eval('DecommissioningCost').reset_index()
	if not f_decom.empty:
		f_decom = f_decom[f_decom['AgentId'].apply(lambda x: x in id_reactor)]
		f_decom = f_decom.groupby('Time').sum()
		costs['Decommissioning'] = f_decom['Payment']
	f_OM = evaler.eval('OperationMaintenance').reset_index()
	f_OM = f_OM[f_OM['AgentId'].apply(lambda x: x in id_reactor)]
	f_OM = f_OM.groupby('Time').sum()
	costs['OandM'] = f_OM['Payment']
	waste_disposal = 1
	f_fuel = evaler.eval('FuelCost').reset_index()
	f_fuel = f_fuel[f_fuel['AgentId'].apply(lambda x: x in id_reactor)]
	f_fuel = f_fuel.groupby('Time').sum()
	costs['Fuel'] = f_fuel['Payment']
	costs = costs.fillna(0)
	costs['Year'] = (costs.index + initial_month - 1) // 12 + initial_year
	if truncate:
		end_year = (simulation_end + initial_month - 1) // 12 + initial_year
		costs = costs[costs['Year'].apply(lambda x : x <= end_year)]
		begin_year = (simulation_begin + initial_month - 1) // 12 + initial_year
		costs = costs[costs['Year'].apply(lambda x : x >= begin_year)]
	if not capital:
		del costs['Capital']
	costs = costs.groupby('Year').sum()
	return costs
	
def institution_annual_costs_present_value(output_db, reactor_id, capital=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = institution_annual_costs(output_db, institution_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
	
def institution_accumulate_capital(output_db, institution_id):
	"""-expenditures + income
	"""
	costs = - institution_annual_costs(output_db, institution_id).sum(axis=1)
	power_gen = institution_power_generated(output_db, institution_id) * institution_average_lcoe(output_db, institution_id)
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def institution_period_costs(output_db, institution_id, t0=0, period=20, capital=True):
	"""New manner to calculate price of electricity, maybe more accurate than lcoe : calculate all costs in a n years period and then determine how much the cost of electricity should be at an institutional level
	costs used to calculate period costs at date t are [t+t0, t+t0+period]
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = institution_annual_costs(output_db, institution_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = institution_power_generated(output_db, institution_id, truncate=False)
	df = pd.DataFrame(index=list(range(initial_year, initial_year + duration // 12 + 1)))
	df['Power'] = power
	df['Costs'] = costs
	df = df.fillna(0)
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end)))
	rtn['Power'] = pd.Series()
	rtn['Payment'] = pd.Series()
	rtn = rtn.fillna(0)
	for i in range(simulation_begin + t0, simulation_begin + t0 + period):	
		rtn.loc[simulation_begin, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - simulation_begin)
		rtn.loc[simulation_begin, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - simulation_begin)
	for j in range(simulation_begin + 1, simulation_end):
		rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * (1 + default_discount_rate) - df.loc[j -1 + t0, 'Power'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Power'] / (1 + default_discount_rate) ** (period + t0 - 1)
		rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Costs'] / (1 + default_discount_rate) ** (period + t0 - 1)
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power']
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = rtn['Ratio'] * actualization
	return rtn
	
def institution_period_costs2(output_db, institution_id, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = institution_annual_costs(output_db, institution_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = institution_power_generated(output_db, institution_id, truncate=False)
	df = pd.DataFrame(index=list(range(initial_year, initial_year + duration // 12 + 1)))
	df['Power'] = power
	df['Costs'] = costs
	df = df.fillna(0)
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end + 1)))
	rtn['Power'] = pd.Series()
	rtn['Payment'] = pd.Series()
	rtn = rtn.fillna(0)
	for j in range(simulation_begin, simulation_end + 1):
		for i in range(j + t0, j + t0 + period):
			rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - j)
			rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - j)
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power']
	return rtn
		
def institution_power_generated(output_db, institution_id, truncate=True):
	"""
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	f_entry = f_entry[f_entry.ParentId==institution_id]
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_reactor = f_entry[f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power['AgentId'].apply(lambda x: x in id_reactor)]
	f_power['Year'] = (f_power['Time'] + initial_month - 1) // 12 + initial_year
	f_power = f_power.groupby('Year').sum()
	rtn = f_power['Value'] * 8760 / 12
	rtn.name = 'Power in MWh'
	return rtn

def institution_lcoe(output_db, institution_id):
	"""
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = institution_annual_costs(output_db, institution_id, truncate=False)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	costs['Power'] = institution_power_generated(output_db, institution_id)
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

def institution_average_lcoe(output_db, institution_id):
	"""Time dependent lcoe
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	f_entry = f_entry[f_entry.ParentId==institution_id]
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_reactor = f_entry[f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	f_power = evaler.eval('TimeSeriesPower')
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end + 1)))
	rtn['Weighted sum'] = 0
	rtn['Power'] = 0
	rtn['Temp'] = pd.Series()
	rtn['Temp2'] = pd.Series()
	for id in id_reactor:
		tmp = lcoe(output_db, id)
		commissioning = f_entry[f_entry.AgentId==id]['EnterTime'].iloc[0]
		lifetime = f_entry[f_entry.AgentId==id]['Lifetime'].iloc[0]
		decommissioning = (commissioning + lifetime + initial_month - 1) // 12 + initial_year
		commissioning = (commissioning + initial_month - 1) // 12 + initial_year
		power = f_power[f_power.AgentId==id]['Value'].iloc[0]
		rtn['Temp'] = pd.Series(tmp, index=list(range(commissioning, decommissioning + 1))) * power
		rtn['Weighted sum'] += rtn['Temp'].fillna(0)
		rtn['Temp2'] = pd.Series(power, index=list(range(commissioning, decommissioning + 1))).fillna(0)
		rtn['Power'] += rtn['Temp2']
	rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
	return rtn
		
# Region level

def region_annual_costs(output_db, region_id, capital=True, truncate=True):
	"""reactors annual costs for a given institution returned as a pandas DataFrame containing total annual costs for each reactor id
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	tmp = f_entry[f_entry.ParentId==region_id]
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_inst = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
	id_reactor = []
	for id in id_inst:
		f_entry2 = f_entry[f_entry.ParentId==id]
		id_reactor += f_entry2[f_entry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_capital = evaler.eval('CapitalCost').reset_index()
	f_capital = f_capital[f_capital['AgentId'].apply(lambda x: x in id_reactor)]
	mini = min(f_capital['Time'])
	f_capital = f_capital.groupby('Time').sum()
	costs = pd.DataFrame({'Capital' : f_capital['Payment']}, index=list(range(mini, duration)))
	f_decom = evaler.eval('DecommissioningCost').reset_index()
	if not f_decom.empty:
		f_decom = f_decom[f_decom['AgentId'].apply(lambda x: x in id_reactor)]
		f_decom = f_decom.groupby('Time').sum()
		costs['Decommissioning'] = f_decom['Payment']
	f_OM = evaler.eval('OperationMaintenance').reset_index()
	f_OM = f_OM[f_OM['AgentId'].apply(lambda x: x in id_reactor)]
	f_OM = f_OM.groupby('Time').sum()
	costs['OandM'] = f_OM['Payment']
	waste_disposal = 1
	f_fuel = evaler.eval('FuelCost').reset_index()
	f_fuel = f_fuel[f_fuel['AgentId'].apply(lambda x: x in id_reactor)]
	f_fuel = f_fuel.groupby('Time').sum()
	costs['Fuel'] = f_fuel['Payment']
	costs = costs.fillna(0)
	costs['Year'] = (costs.index + initial_month - 1) // 12 + initial_year
	if truncate:
		end_year = (simulation_end + initial_month - 1) // 12 + initial_year
		costs = costs[costs['Year'].apply(lambda x : x <= end_year)]
		begin_year = (simulation_begin + initial_month - 1) // 12 + initial_year
		costs = costs[costs['Year'].apply(lambda x : x >= begin_year)]
	if not capital:
		del costs['Capital']
	costs = costs.groupby('Year').sum()
	return costs
		
def region_annual_costs_present_value(output_db, region_id, capital=True, truncate=True):
	"""Same as annual_cost except all values are actualized to the begin date of the SIMULATION
	"""
	costs = region_annual_costs(output_db, region_id, capital)
	actualization = actualization_vector(len(costs))
	actualization.index = costs.index
	return costs.apply(lambda x : x * actualization)
		
def region_accumulate_capital(output_db, region_id):
	"""-expenditures + income
	"""
	costs = - region_annual_costs(output_db, region_id).sum(axis=1)
	power_gen = region_power_generated(output_db, region_id) * region_average_lcoe(output_db, region_id)['Average LCOE']
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def region_period_costs(output_db, region_id, t0=0, period=20, capital=True):
	"""New manner to calculate price of electricity, maybe more accurate than lcoe : calculate all costs in a n years period and then determine how much the cost of electricity should be at an institutional level
	costs used to calculate period costs at date t are [t+t0, t+t0+period]
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = region_annual_costs(output_db, region_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = region_power_generated(output_db, region_id, truncate=False)
	df = pd.DataFrame(index=list(range(initial_year, initial_year + duration // 12 + 1)))
	df['Power'] = power
	df['Costs'] = costs
	df = df.fillna(0)
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end + 1)))
	rtn['Power'] = pd.Series()
	rtn['Payment'] = pd.Series()
	rtn = rtn.fillna(0)
	for i in range(simulation_begin + t0, simulation_begin + t0 + period):	
		rtn.loc[simulation_begin, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - simulation_begin)
		rtn.loc[simulation_begin, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - simulation_begin)
	for j in range(simulation_begin + 1, simulation_end + 1):
		rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * (1 + default_discount_rate) - df.loc[j -1 + t0, 'Power'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Power'] / (1 + default_discount_rate) ** (period + t0 - 1)
		rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Costs'] / (1 + default_discount_rate) ** (period + t0 - 1)
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power']
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = rtn['Ratio'] * actualization
	return rtn
	
def region_period_costs2(output_db, region_id, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = region_annual_costs(output_db, region_id, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = region_power_generated(output_db, region_id, truncate=False)
	df = pd.DataFrame(index=list(range(initial_year, initial_year + duration // 12 + 1)))
	df['Power'] = power
	df['Costs'] = costs
	df = df.fillna(0)
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end + 1)))
	rtn['Power'] = pd.Series()
	rtn['Payment'] = pd.Series()
	rtn = rtn.fillna(0)
	for j in range(simulation_begin, simulation_end + 1):
		for i in range(j + t0, j + t0 + period):
			rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - j)
			rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - j)
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power']
	return rtn
		
def region_power_generated(output_db, region_id, truncate=True):
	"""
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	tmp = f_entry[f_entry.ParentId==region_id]
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_inst = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
	id_reactor = []
	for id in id_inst:
		f_entry2 = f_entry[f_entry.ParentId==id]
		id_reactor += f_entry2[f_entry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power['AgentId'].apply(lambda x: x in id_reactor)]
	f_power['Year'] = (f_power['Time'] + initial_month - 1) // 12 + initial_year
	f_power = f_power.groupby('Year').sum()
	rtn = f_power['Value'] * 8760 / 12
	rtn.name = 'Power in MWh'
	return rtn

def region_lcoe(output_db, region_id):
	"""
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = region_annual_costs(output_db, region_id, truncate=False)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	costs['Power'] = region_power_generated(output_db, region_id)
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

def region_average_lcoe(output_db, region_id):
	"""Time dependent lcoe
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	tmp = f_entry[f_entry.ParentId==region_id]
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_inst = tmp[tmp.Kind=='Inst']['AgentId'].tolist()
	id_reactor = []
	f_power = evaler.eval('TimeSeriesPower')
	for id in id_inst:
		f_entry2 = f_entry[f_entry.ParentId==id]
		id_reactor += f_entry2[f_entry2['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end + 1)))
	rtn['Weighted sum'] = 0
	rtn['Power'] = 0
	rtn['Temp'] = pd.Series()
	rtn['Temp2'] = pd.Series()
	for id in id_reactor:
		tmp = lcoe(output_db, id)
		commissioning = f_entry[f_entry.AgentId==id]['EnterTime'].iloc[0]
		lifetime = f_entry[f_entry.AgentId==id]['Lifetime'].iloc[0]
		decommissioning = (commissioning + lifetime + initial_month - 1) // 12 + initial_year
		commissioning = (commissioning + initial_month - 1) // 12 + initial_year
		power = f_power[f_power.AgentId==id]['Value'].iloc[0]
		rtn['Temp'] = pd.Series(tmp, index=list(range(commissioning, decommissioning + 1))) * power
		rtn['Weighted sum'] += rtn['Temp'].fillna(0)
		rtn['Temp2'] = pd.Series(power, index=list(range(commissioning, decommissioning + 1))).fillna(0)
		rtn['Power'] += rtn['Temp2']
	rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
	return rtn
	
# Simulation level

def simulation_annual_costs(output_db, capital=True, truncate=True):
	"""reactors annual costs for a given institution returned as a pandas DataFrame containing total annual costs for each reactor id
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_reactor = f_entry[f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_capital = evaler.eval('CapitalCost').reset_index()
	f_capital = f_capital[f_capital['AgentId'].apply(lambda x: x in id_reactor)]
	mini = min(f_capital['Time'])
	f_capital = f_capital.groupby('Time').sum()
	costs = pd.DataFrame({'Capital' : f_capital['Payment']}, index=list(range(mini, duration)))
	f_decom = evaler.eval('DecommissioningCost').reset_index()
	if not f_decom.empty:
		f_decom = f_decom[f_decom['AgentId'].apply(lambda x: x in id_reactor)]
		f_decom = f_decom.groupby('Time').sum()
		costs['Decommissioning'] = f_decom['Payment']
	f_OM = evaler.eval('OperationMaintenance').reset_index()
	f_OM = f_OM[f_OM['AgentId'].apply(lambda x: x in id_reactor)]
	f_OM = f_OM.groupby('Time').sum()
	costs['OandM'] = f_OM['Payment']
	waste_disposal = 1
	f_fuel = evaler.eval('FuelCost').reset_index()
	f_fuel = f_fuel[f_fuel['AgentId'].apply(lambda x: x in id_reactor)]
	f_fuel = f_fuel.groupby('Time').sum()
	costs['Fuel'] = f_fuel['Payment']
	costs = costs.fillna(0)
	costs['Year'] = (costs.index + initial_month - 1) // 12 + initial_year
	if truncate:
		end_year = (simulation_end + initial_month - 1) // 12 + initial_year
		costs = costs[costs['Year'].apply(lambda x : x <= end_year)]
		begin_year = (simulation_begin + initial_month - 1) // 12 + initial_year
		costs = costs[costs['Year'].apply(lambda x : x >= begin_year)]
	if not capital:
		del costs['Capital']
	costs = costs.groupby('Year').sum()
	return costs
		
def simulation_annual_costs_present_value(output_db, capital=True, truncate=True):
	df = simulation_annual_costs(output_db, capital)
	for year in df.index:
		df.loc[year, :] = df.loc[year, :] / (1 + default_discount_rate) ** (year - df.index[0])
	return df
	
def simulation_accumulate_capital(output_db):
	"""-expenditures + income
	"""
	costs = - simulation_annual_costs(output_db).sum(axis=1)
	power_gen = simulation_power_generated(output_db) * simulation_average_lcoe(output_db)
	rtn = pd.concat([costs, power_gen], axis=1).fillna(0)
	rtn['Capital'] = (rtn[0] + rtn[1]).cumsum()
	actualization = actualization_vector(len(rtn))
	actualization.index = rtn.index
	rtn['Actualized'] = ((rtn[0] + rtn[1]) * actualization).cumsum()
	return rtn
		
def simulation_period_costs(output_db, t0=0, period=20, capital=True):
	"""New manner to calculate price of electricity, maybe more accurate than lcoe : calculate all costs in a n years period and then determine how much the cost of electricity should be at an institutional level
	costs used to calculate period costs at date t are [t+t0, t+t0+period]
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = simulation_annual_costs(output_db, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = simulation_power_generated(output_db, truncate=False)
	df = pd.DataFrame(index=list(range(initial_year, initial_year + duration // 12 + 1)))
	df['Power'] = power
	df['Costs'] = costs
	df = df.fillna(0)
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end)))
	rtn['Power'] = pd.Series()
	rtn['Payment'] = pd.Series()
	rtn = rtn.fillna(0)
	for i in range(simulation_begin + t0, simulation_begin + t0 + period):	
		rtn.loc[simulation_begin, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - simulation_begin)
		rtn.loc[simulation_begin, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - simulation_begin)
	for j in range(simulation_begin + 1, simulation_end + 1):
		rtn.loc[j, 'Power'] = rtn.loc[j - 1, 'Power'] * (1 + default_discount_rate) - df.loc[j -1 + t0, 'Power'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Power'] / (1 + default_discount_rate) ** (period + t0 - 1)
		rtn.loc[j, 'Payment'] = rtn.loc[j - 1, 'Payment'] * (1 + default_discount_rate) - df.loc[j - 1 + t0, 'Costs'] * (1 + default_discount_rate) ** (1 - t0) + df.loc[j - 1 + period + t0, 'Costs'] / (1 + default_discount_rate) ** (period + t0 - 1)
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power']
	return rtn
	
def simulation_period_costs2(output_db, t0=0, period=20, capital=True):
	"""Just for tests : slower but more secure
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = simulation_annual_costs(output_db, capital, truncate=False)
	costs = costs.sum(axis=1)
	power = simulation_power_generated(output_db, truncate=False)
	df = pd.DataFrame(index=list(range(initial_year, initial_year + duration // 12 + 1)))
	df['Power'] = power
	df['Costs'] = costs
	df = df.fillna(0)
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end + 1)))
	rtn['Power'] = pd.Series()
	rtn['Payment'] = pd.Series()
	rtn = rtn.fillna(0)
	for j in range(simulation_begin, simulation_end + 1):
		for i in range(j + t0, j + t0 + period):
			rtn.loc[j, 'Power'] += df.loc[i, 'Power'] / (1 + default_discount_rate) ** (i - j)
			rtn.loc[j, 'Payment'] += df.loc[i, 'Costs'] / (1 + default_discount_rate) ** (i - j)
			#tmp['WasteManagement'][j] = pd.Series()
	rtn['Ratio'] = rtn['Payment'] / rtn ['Power']
	return rtn
		
def simulation_power_generated(output_db, truncate=True):
	"""
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_reactor = f_entry[f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	f_power = evaler.eval('TimeSeriesPower').reset_index()
	f_power = f_power[f_power['AgentId'].apply(lambda x: x in id_reactor)]
	f_power['Year'] = (f_power['Time'] + initial_month - 1) // 12 + initial_year
	f_power = f_power.groupby('Year').sum()
	rtn = f_power['Value'] * 8760 / 12
	rtn.name = 'Power in MWh'
	return rtn

def simulation_lcoe(output_db):
	"""
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	costs = simulation_annual_costs(output_db, truncate=False)
	costs['TotalCosts'] = costs.sum(axis=1)
	commissioning = costs['Capital'].idxmax()
	costs['Power'] = simulation_power_generated(output_db)
	costs = costs.fillna(0)
	power_generated = 0
	total_costs = 0
	for i in costs.index:
		power_generated += costs['Power'][i] / ((1 + default_discount_rate) ** (i - commissioning))
		total_costs += costs['TotalCosts'][i] / ((1 + default_discount_rate) ** (i - commissioning))
	return total_costs / power_generated

def simulation_average_lcoe(output_db):
	"""Time dependent lcoe
	"""
	db = dbopen(output_db)
	evaler = Evaluator(db)
	f_info = evaler.eval('Info').reset_index()
	duration = f_info.loc[0, 'Duration']
	initial_year = f_info.loc[0, 'InitialYear']
	initial_month = f_info.loc[0, 'InitialMonth']
	if os.path.isfile(xml_inputs):
		tree = ET.parse(xml_inputs)
		root = tree.getroot()
		if root.find('truncation') is not None:
			truncation = root.find('truncation')
			if truncation.find('simulation_begin') is not None:
				simulation_begin = int(truncation.find('simulation_begin').text)
			else:
				simulation_begin = 0
			if truncation.find('simulation_end') is not None:
				simulation_end = int(truncation.find('simulation_end').text)
			else:
				simulation_end = duration
	f_entry = evaler.eval('AgentEntry').reset_index()
	f_entry = f_entry[f_entry['EnterTime'].apply(lambda x: x>simulation_begin and x<simulation_end)]
	id_reactor = f_entry[f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist()
	simulation_begin = (simulation_begin + initial_month - 1) // 12 + initial_year # year instead of months
	simulation_end = (simulation_end + initial_month - 1) // 12 + initial_year
	f_power = evaler.eval('TimeSeriesPower')
	rtn = pd.DataFrame(index=list(range(simulation_begin, simulation_end + 1)))
	rtn['Weighted sum'] = 0
	rtn['Power'] = 0
	rtn['Temp'] = pd.Series()
	rtn['Temp2'] = pd.Series()
	for id in id_reactor:
		tmp = lcoe(output_db, id)
		commissioning = f_entry[f_entry.AgentId==id]['EnterTime'].iloc[0]
		lifetime = f_entry[f_entry.AgentId==id]['Lifetime'].iloc[0]
		decommissioning = (commissioning + lifetime + initial_month - 1) // 12 + initial_year
		commissioning = (commissioning + initial_month - 1) // 12 + initial_year
		power = f_power[f_power.AgentId==id]['Value'].iloc[0]
		rtn['Temp'] = pd.Series(tmp, index=list(range(commissioning, decommissioning + 1))) * power
		rtn['Weighted sum'] += rtn['Temp'].fillna(0)
		rtn['Temp2'] = pd.Series(power, index=list(range(commissioning, decommissioning + 1))).fillna(0)
		rtn['Power'] += rtn['Temp2']
	rtn['Average LCOE'] = rtn['Weighted sum'] / rtn['Power']
	return rtn

###########################
# Plotting costs #
###########################

# Reactor level

def annual_costs_plot(output_db, reactor_id, capital=True):
    """Plot of total costs for one reactor per year
    """
    df = annual_costs(output_db, reactor_id, capital) / 10 ** 9
    df.plot(kind='area')
    plt.xlabel('Year')
    plt.ylabel('Cost (billion $2015)')
    plt.title('Annual costs for nuclear plants over their lifetime')
    plt.show()
    
def annual_costs_present_value_plot(output_db, reactor_id, capital=True):
    """Plot of total costs for one reactor per year
    """
    df = annual_costs_present_value(output_db, reactor_id, capital) / 10 ** 9
    df.plot(kind='area')
    plt.xlabel('Year')
    plt.ylabel('Cost (billion $2015)')
    plt.title('Annual costs for nuclear (present value)')
    plt.show()

def average_cost_plot(output_db, reactor_id, capital=True):
    """Plot of the average costs for one reactor over its lifetime
    """
    if not isinstance(reactor_id, list):
    	raise TypeError('Wrong input, reactor ids should be given in a list')
    db = dbopen(output_db)
    evaler = Evaluator(db)
    f_info = evaler.eval('Info').reset_index()
    duration = f_info['Duration'].iloc[0]
    df = pd.DataFrame(index=list(range(duration)))
    initial_year = f_info['InitialYear'].iloc[0]
    initial_month = f_info['InitialMonth'].iloc[0]
    f_entry = evaler.eval('AgentEntry').reset_index()
    for id in reactor_id:
    	f_entry2 = f_entry[f_entry.AgentId==id]
    	date_entry = f_entry2['EnterTime'].iloc[0]
    	lifetime = f_entry2['Lifetime'].iloc[0]
    	prototype = f_entry2['Prototype'].iloc[0]
    	ser = pd.Series(average_cost(output_db, id, capital), index=list(range(date_entry, date_entry + lifetime)))
    	df[prototype+' (AgentId : '+str(id)+')'] = ser
    df = df.fillna(0)
    df['Date'] = pd.Series(df.index.values).apply(lambda x: (x + initial_month - 1) // 12 + initial_year + (x % 12) / 12)
    df = df.set_index('Date')
    df.plot()
    plt.xlabel('Year')
    plt.ylabel('Cost ($2015/MWh)')
    plt.title('Average costs for nuclear plants over their lifetime')
    plt.show()
    	
def lcoe_plot(output_db, reactor_id, capital=True):
    """Plot of levelized cost of electricity obtained with one reactor (given a
    fuel cycle technology)
    """
    if not isinstance(reactor_id, list):
    	raise TypeError('Wrong input, reactor ids should be given in a list')
    db = dbopen(output_db)
    evaler = Evaluator(db)
    f_info = evaler.eval('Info').reset_index()
    duration = f_info['Duration'].iloc[0]
    initial_year = f_info['InitialYear'].iloc[0]
    initial_month = f_info['InitialMonth'].iloc[0]
    f_entry = evaler.eval('AgentEntry').reset_index()
    df = pd.DataFrame(index=list(range(duration)))
    for id in reactor_id:
    	f_entry2 = f_entry[f_entry.AgentId==id]
    	date_entry = f_entry2['EnterTime'].iloc[0]
    	lifetime = f_entry2['Lifetime'].iloc[0]
    	prototype = f_entry2['Prototype'].iloc[0]
    	ser = pd.Series(lcoe(output_db, id, capital), index=list(range(date_entry,lifetime)))
    	df[prototype+' (AgentId : '+str(id)+')'] = ser
    df = df.fillna(0)
    df['Date'] = pd.Series(df.index.values).apply(lambda x: (x + initial_month - 1) // 12 + initial_year + (x % 12) / 12)
    df = df.set_index('Date')
    df.plot()
    #plt.plot(df)#, label=prototype+' (AgentId : '+str(reactor_id)+')')
    #legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.xlabel('Year')
    plt.ylabel('Cost ($2015/MWh)')
    plt.title('Levelized cost of electricity for nuclar plants')
    plt.show()

# Institution level

def institution_annual_costs_plot(output_db, institution_id, capital=True):
	"""plot all reactors annual costs for a given institution
	"""
	total = institution_annual_costs(output_db, institution_id, capital) / 10 ** 9 # billion $2015
	total.plot(kind='area', colormap='Greens', linewidth = 3)
	plt.xlabel('Year')
	plt.ylabel('Cost (billion $2015)')
	plt.title('Annual costs related to its nuclear plants for institution ' + str(institution_id))
	plt.show()
	
def institution_annual_costs_present_value_plot(output_db, institution_id, capital=True):
	"""plot all reactors annual costs for a given institution
	"""
	total = institution_annual_costs_present_value(output_db, institution_id, capital) / 10 ** 9 # billion $2015
	total.plot(kind='area', colormap='Greens', linewidth = 3)
	plt.xlabel('Year')
	plt.ylabel('Cost (billion $2015)')
	plt.title('Annual costs related to its nuclear plants for institution ' + str(institution_id) + ' (present value)')
	plt.show()

def institution_period_costs_plot(output_db, institution_id, period=20, capital=True):
	"""New manner to calculate price of electricity, maybe more accurate than lcoe : calculate all costs in a n years period and then determine how much the cost of electricity should be at an institutional level
	"""
	institution_period_costs(output_db, institution_id, period, capital)['Ratio'].plot()
	plt.xlabel('Year')
	plt.ylabel('Cost ($2015/MWh)')
	plt.title('Reactor costs using a ' + str(period) + ' years time frame (institution n°' + str(institution_id)+ ')')
	plt.show()

# Region level

def region_annual_costs_plot(output_db, region_id, capital=True):
	"""plot all reactors annual costs for a given institution
	"""
	total = region_annual_costs(output_db, region_id, capital) / 10 ** 9
	total.plot(kind='area', colormap='Blues', linewidth = 3)
	plt.xlabel('Year')
	plt.ylabel('Cost (billion $2015)')
	plt.title('Annual costs related to its nuclear plants for region ' + str(region_id))
	plt.show()
	
def region_annual_costs_present_value_plot(output_db, region_id, capital=True):
	"""plot all reactors annual costs for a given institution
	"""
	total = region_annual_costs_present_value(output_db, region_id, capital) / 10 ** 9
	total.plot(kind='area', colormap='Blues', linewidth = 3)
	plt.xlabel('Year')
	plt.ylabel('Cost (billion $2015)')
	plt.title('Annual costs related to its nuclear plants for region ' + str(region_id) + ' (present value)')
	plt.show()

def region_period_costs_plot(output_db, region_id, period=20, capital=True):
	"""Same as instution_period_costs but at a regional level
	"""
	region_period_costs(output_db, region_id, period)['Ratio'].plot()
	plt.xlabel('Year')
	plt.ylabel('Cost ($2015/MWh)')
	plt.title('Reactor costs using a ' + str(period) + ' years time frame (region n°' + str(region_id)+ ')')
	plt.show()
	
#lcoe region, lcoe institution

########################################
# Iteration of simulation calculations #
########################################

def iter_metric(iteration, output_db, metric):
	""" metric is a string, output_db as well, iteration integer function works for the following metrics : CapitalCost,
	"""
	db = dbopen(output_db)
	frame = Evaluator(db).eval(metric)
	frame = frame.groupby(['AgentId', 'Time']).sum()
	for i in range(1, iteration):
		tmp = Evaluator(db).eval(metric)
		tmp = tmp.groupby(['AgentId', 'Time']).sum()
		frame = pd.concat([frame, tmp], axis=1)
	return frame

def iter_lcoe(output_db, reactor_id, iteration):
	"""iterate number of lcoe and then plot the distribution
	"""
	lst=[]
	for i in range(0,iteration):
		lst.append(lcoe(output_db, reactor_id))
	plt.hist(x=lst, bins=iteration/10)
	plt.show()

###############################
# Plotting other agents costs #
###############################

def agents_cash(output_db, agent_id):
    """For a given agent, plot the cash_flows (positive values = income,
    negative values = expenditures). Could be a region, institution or 
    facility.
    """ 