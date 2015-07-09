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


## The actual metrics ##


_ccdeps = [('TimeSeriesPower', ('SimId', 'AgentId', 'Value'), 'Time'), ('AgentEntry', ('AgentId', 'ParentId', 'Spec'), 'EnterTime'), ('Info', ('InitialYear', 'InitialMonth'), 'Duration')]

_ccschema = [('SimId', ts.UUID), ('AgentId', ts.INT),
             ('Time', ts.INT), ('Payment', ts.DOUBLE)]

def f(x): # test
    print(x)

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
    sim_duration = f_info['Duration'].iloc[0]
    f_entry = f_entry[f_entry.Spec == ":cycamore:Reactor"]
    id_reactors = f_entry["AgentId"].tolist()
    lst = 12 * np.random.randn(len(id_reactors))
    lst *= lst > 0 # we only consider delays (positive values), no head start (negative values)
    lst = list(map(int,lst))
    j = 0
    f_entry = pd.DataFrame([f_entry.EnterTime, f_entry.AgentId]).transpose()
    f_entry = f_entry.set_index(['AgentId'])
    f_entry['Capacity'] = pd.Series()
    rtn = pd.DataFrame()
    xml_inputs = 'parameters.xml' # temporary solution : always store an xml file in your working directory that you will have to use. This file have to be known
    begin = [default_cap_begin] * len(id_reactors)
    duration = [default_cap_duration] * len(id_reactors)
    shape = default_cap_shape
    overnight_cost = 0 # test default_cap_overnight
    default_cap_overnight = 0 # test
    discount_rate = default_discount_rate
    print(0) # test
    print(0.5) # test
    return f_power # test
    if os.path.isfile(xml_inputs):
    	print(1) # test
    	tree = ET.parse(xml_inputs)
    	root = tree.getroot()
    	# just in case no definition of parameters
    	if root.find('capital') == None:
    		print(2) # test
    		for region in root.findall('region'):
    			if region.find('capital') == None:
    				print(3) # test
    				for institution in region.findall('institution'):
    					if institution.find('capital') == None:
    						print(4) # test
    						for facility in institution.findall('facility'):
    							id = int(facility.find('id').text)
    							print(id) # test
    							capital = facility.find('capital')
    							if capital.find('pace') is not None: #random parameters
    								if capital.find('pace').text == "rapid":
    									begin = rapid_cap_begin + int(lst[j])
    									duration = rapid_cap_duration + 2 * int(lst[j])
    								elif capital.find('pace').text == "slow":
    									begin = slow_cap_begin + int(lst[j])
    									duration = slow_cap_duration + 2 * int(lst[j])
    								else: #normal
    									begin = default_cap_begin + int(lst[j])
    									duration = default_cap_duration + 2 * int(lst[j])
    								j += 1	
    							else: #own parameters
    								if capital.find('begin') is not None:
    									begin = int(capital.find('begin').text)
    								else:
    									begin = default_cap_begin
    								if capital.find('duration') is not None:
    									duration = int(capital.find('duration').text)
    								else:
    									duration = default_cap_duration
    								if capital.find('overnight_cost') is not None:
    									print(5) # test
    									overnight_cost = int(capital.find('overnight_cost').text)
    								else:
    									overnight_cost = default_cap_overnight					
    								if capital.find('shape') is not None:
    									shape = capital.find('costs_shape').text
    								else:
    									shape = default_cap_shape
    							print(overnight_cost) # test		
    							s_cost = capital_shape(begin, duration, shape)
    							tmp = f_power[f_power.AgentId==id]
    							f_entry.loc[id, 'Capacity'] = max(tmp['Value'])
    							s_cost2 = np.around(s_cost * overnight_cost * f_entry.loc[id, 'Capacity'], 3)
    							s_cost2 = s_cost2 * ((1 + discount_rate) ** math.ceil(duration / 12) - 1) / (discount_rate * math.ceil(duration / 12)) # to take into account interest (see ppt X "intérêts intercalaires" or "Appendix-A reactor-level analysis of busbar costs.pdf" in cost calculations folder)
    							reactor_i_df = pd.DataFrame({'AgentId': id, 'Time': pd.Series(list(range(duration + 1))) - begin + f_entry.EnterTime[id], 'Payment' : s_cost2})
    							rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)
    					else:
    						capital = institution.find('capital')
    						instId = int(institution.find('id').text)
    						id_reactors = f_entry[f_entry.ParentId==instId][f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist() # all reactor ids that belong to institution instId
    						if capital.find('pace') is not None: #random parameters
    							if capital.find('pace').text == "rapid":
    								begin = [rapid_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    								duration = [rapid_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    							elif capital.find('pace').text == "slow":
    								begin = [slow_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    								duration = [slow_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    							else: #normal
    								begin = [default_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    								duration = [default_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    						else: #own parameters
    							if capital.find('begin') is not None:
    								begin = [int(capital.find('begin').text)] * len(id_reactors)
    							else:
    								begin = [default_cap_begin] * len(id_reactors)
    							if capital.find('duration') is not None:
    								duration = [int(capital.find('duration').text)] * len(id_reactors)
    							else:
    								duration = [default_cap_duration] * len(id_reactors)
    						if capital.find('overnight_cost') is not None:
    							overnight_cost = int(capital.find('overnight_cost').text)
    						else:
    							overnight_cost = default_cap_overnight
    						if capital.find('shape') is not None:
    							shape = capital.find('costs_shape').text
    						else:
    							shape = default_cap_shape
    						j += len(id_reactors)
    						cpt = 0
    						for id in id_reactors:
    							s_cost = capital_shape(int(begin[cpt]), int(duration[cpt]), shape)
    							tmp = f_power[f_power.AgentId==id]
    							f_entry.loc[id, 'Capacity'] = max(tmp['Value'])
    							s_cost2 = np.around(s_cost * overnight_cost * f_entry.loc[id, 'Capacity'], 3)
    							s_cost2 = s_cost2 * ((1 + discount_rate) ** math.ceil(duration[cpt] / 12) - 1) / (discount_rate * math.ceil(duration[cpt] / 12)) # to take into account interest (see ppt X "intérêts intercalaires" or "Appendix-A reactor-level analysis of busbar costs.pdf" in cost calculations folder)
    							reactor_i_df = pd.DataFrame({'AgentId': id, 'Time': pd.Series(list(range(duration[cpt] + 1))) - begin[cpt] + f_entry.EnterTime[id], 'Payment' : s_cost2})
    							rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)
    							cpt += 1
    			else:
    				capital = region.find('capital')
    				regId = int(region.find('id').text)
    				id_inst = f_entry[f_entry.Kind=='Inst'][f_entry.ParentId==regId].tolist()
    				id_reactors = []
    				for instId in id_inst:
    					id_reactors +=  f_entry[f_entry.ParentId==instId][f_entry['Spec'].apply(lambda x: 'REACTOR' in x.upper())]['AgentId'].tolist() # all reactor ids that belong to institution n°instId	
    				if capital.find('pace') is not None: #random parameters
    					if capital.find('pace').text == "rapid":
    						begin = [rapid_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    						duration = [rapid_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    					elif capital.find('pace').text == "slow":
    						begin = [slow_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    						duration = [slow_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    					else: #normal
    						begin = [default_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    						duration = [default_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    				else: #own parameters
    					if capital.find('begin') is not None:
    						begin = [int(capital.find('begin').text)] * len(id_reactors)
    					else:
    						begin = [default_cap_begin] * len(id_reactors)
    					if capital.find('duration') is not None:
    						duration = [int(capital.find('duration').text)] * len(id_reactors)
    					else:
    						duration = [default_cap_duration] * len(id_reactors)
    				if capital.find('overnight_cost') is not None:
    					overnight_cost = int(capital.find('overnight_cost').text)
    				else:
    					overnight_cost = default_cap_overnight
    				if capital.find('shape') is not None:
    					shape = capital.find('costs_shape').text
    				else:
    					shape = default_cap_shape
    				j += len(id_reactors)
    				cpt = 0
    				for id in id_reactors:
    					s_cost = capital_shape(int(begin[cpt]), int(duration[cpt]), shape)
    					tmp = f_power[f_power.AgentId==id]
    					f_entry.loc[id, 'Capacity'] = max(tmp['Value'])
    					s_cost2 = np.around(s_cost * overnight_cost * f_entry.loc[id, 'Capacity'], 3)
    					s_cost2 = s_cost2 * ((1 + discount_rate) ** math.ceil(duration[cpt] / 12) - 1) / (discount_rate * math.ceil(duration[cpt] / 12)) # to take into account interest (see ppt X "intérêts intercalaires" or "Appendix-A reactor-level analysis of busbar costs.pdf" in cost calculations folder)
    					reactor_i_df = pd.DataFrame({'AgentId': id, 'Time': pd.Series(list(range(duration[cpt] + 1))) - begin[cpt] + f_entry.EnterTime[id], 'Payment' : s_cost2})
    					rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)
    					cpt += 1	
    	else:
    		capital = root.find('capital')
    		if capital.find('pace') is not None: #random parameters
    			if capital.find('pace').text == "rapid":
    				begin = [rapid_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    				duration = [rapid_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    			elif capital.find('pace').text == "slow":
    				begin = [slow_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    				duration = [slow_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    			else: #normal
    				begin = [default_cap_begin] * len(id_reactors) + np.array(lst[j:j+len(id_reactors)])
    				duration = [default_cap_duration] * len(id_reactors) + 2 * np.array(lst[j:j+len(id_reactors)])
    		else: #own parameters
    			if capital.find('begin') is not None:
    				begin = [int(capital.find('begin').text)] * len(id_reactors)
    			else:
    				begin = [default_cap_begin] * len(id_reactors)
    			if capital.find('duration') is not None:
    				duration = [int(capital.find('duration').text)] * len(id_reactors)
    			else:
    				duration = [default_cap_duration] * len(id_reactors)
    		if capital.find('overnight_cost') is not None:
    			overnight_cost = int(capital.find('overnight_cost').text)
    		else:
    			overnight_cost = default_cap_overnight
    		if capital.find('shape') is not None:
    			shape = capital.find('costs_shape').text
    		else:
    			shape = default_cap_shape
    		j += len(id_reactors)
    		cpt = 0
    		for id in id_reactors:
    			s_cost = capital_shape(int(begin[cpt]), int(duration[cpt]), shape)
    			tmp = f_power[f_power.AgentId==id]
    			f_entry.loc[id, 'Capacity'] = max(tmp['Value'])
    			s_cost2 = np.around(s_cost * overnight_cost * f_entry.loc[id, 'Capacity'], 3)
    			s_cost2 = s_cost2 * ((1 + discount_rate) ** math.ceil(duration[cpt] / 12) - 1) / (discount_rate * math.ceil(duration[cpt] / 12)) # to take into account interest (see ppt X "intérêts intercalaires" or "Appendix-A reactor-level analysis of busbar costs.pdf" in cost calculations folder)
    			reactor_i_df = pd.DataFrame({'AgentId': id, 'Time': pd.Series(list(range(duration[cpt] + 1))) - begin[cpt] + f_entry.EnterTime[id], 'Payment' : s_cost2})
    			rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)
    			cpt += 1
    else:
    	for id in id_reactors:
    		s_cost = capital_shape(begin, duration, shape)
    		tmp = f_power[f_power.AgentId==id]
    		f_entry.loc[id, 'Capacity'] = max(tmp['Value'])
    		s_cost2 = np.around(s_cost * overnight_cost * f_entry.loc[id, 'Capacity'], 3)
    		s_cost2 = s_cost2 * ((1 + discount_rate) ** math.ceil(duration / 12) - 1) / (discount_rate * math.ceil(duration / 12)) # to take into account interest (see ppt X "intérêts intercalaires" or "Appendix-A reactor-level analysis of busbar costs.pdf" in cost calculations folder)
    		reactor_i_df = pd.DataFrame({'AgentId': id, 'Time': pd.Series(list(range(duration + 1))) - begin + f_entry.EnterTime[id], 'Payment' : s_cost2})
    		rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)
    rtn['SimId'] = f_power['SimId'].iloc[0]
    subset = rtn.columns.tolist()
    subset = subset[3:] + subset[:1] + subset[2:3] + subset[1:2]
    rtn = rtn[subset]
    rtn = rtn[rtn['Time'].apply(lambda x: x >= 0 and x < sim_duration)]
    rtn = rtn.reset_index()
    del rtn['index']
    return rtn

del _ccdeps, _ccschema


_fcdeps = [('Resources', ('SimId', 'ResourceId'), 'Quantity'), ('Transactions',
        ('SimId', 'TransactionId', 'ReceiverId', 'ResourceId', 'Commodity'), 
        'Time')]

_fcschema = [('SimId', ts.UUID), ('TransactionId', ts.INT), ('ReceiverId', 
          ts.INT), ('Commodity', ts.STRING), ('Payment', ts.DOUBLE), ('Time', 
          ts.INT)]

@metric(name='FuelCost', depends=_fcdeps, schema=_fcschema)
def fuel_cost(series):
    """fuel_cost returns the cash flows related to the fuel costs for power 
    plants.
    """
    print(0) # test
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
    id_reactors = f_entry[f_entry.Spec==":cycamore:Reactor"]["AgentId"].tolist()
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


_eideps = [('Info', ('SimId'), 'InitialYear'), ('AgentEntry', ('AgentId', 'Spec'), 'EnterTime')]

_eischema = [('SimId', ts.UUID), ('DiscountRate', ts.DOUBLE), ('CapitalMode', ts.STRING), ('ConstructionBegin', ts.INT), ('ConstructionDuration', ts.INT), ('OvernightCost', ts.INT), ('FixedOM', ts.INT), ('VariableOM', ts.INT)]
		
@metric(name='EconomicInfo', depends=_eideps, schema=_eischema)
def economic_info(series):
    """Write the economic parameters in the database
    """
    f_entry = f_entry[f_entry.Spec == ":cycamore:Reactor"]
    id_reactors = f_entry["AgentId"].tolist()
    xml_inputs = 'parameters.xml'
    begin = default_cap_begin
    duration = default_cap_duration    
    overnight_cost = default_cap_overnight
    fixedOM = default_fixedOM
    variableOM = default_variableOM
    if root.find('capital') == None:
    	for region in root.findall('region'):
    		if region.find('capital') == None:
    			for institution in region.findall('institution'):
    				if institution.find('capital') == None:
    					for facility in instituion.findall('facility'):
    						if facility.find('capital') == None:
    							raise Exception("Missing capital costs parameters in " + xml_inputs)
    						else:
    							id = int(facility.find('id').text)
    							capital = facility.find('capital')
    							if capital.find('pace') is not None: #random parameters
    								if capital.find('pace').text == "rapid":
    									begin = rapid_cap_begin + int(lst[j])
    									duration = rapid_cap_duration + 2 * int(lst[j])
    								elif capital.find('pace').text == "slow":
    									begin = slow_cap_begin + int(lst[j])
    									duration = slow_cap_duration + 2 * int(lst[j])
    								else: #normal
    									begin = default_cap_begin + int(lst[j])
    									duration = default_cap_duration + 2 * int(lst[j])
    								j += 1	
    							else: #own parameters
    								if capital.find('begin') is not None:
    									begin = int(capital.find('begin').text)
    								else:
    									begin = default_cap_begin
    								if capital.find('duration') is not None:
    									duration = int(capital.find('duration').text)
    								else:
    									duration = default_cap_duration
    							if capital.find('overnight_cost') is not None:
    								overnight_cost = int(capital.find('overnight_cost').text)
    							else:
    								overnight_cost = default_cap_overnight			
    								
    							if capital.find('shape') is not None:
    								shape = capital.find('costs_shape').text
    							else:
    								shape = default_cap_shape
    							
    							rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)
    				else:
    					capital = institution.find('capital')
    					if capital.find('pace') is not None: #random parameters
    						if capital.find('pace').text == "rapid":
    							begin = rapid_cap_begin + int(lst[j])
    							duration = rapid_cap_duration + 2 * int(lst[j])
    						elif capital.find('pace').text == "slow":
    							begin = slow_cap_begin + int(lst[j])
    							duration = slow_cap_duration + 2 * int(lst[j])
    						else: #normal
    							begin = default_cap_begin + int(lst[j])
    							duration = default_cap_duration + 2 * int(lst[j])
    						j += 1	
    					else: #own parameters
    						if capital.find('begin') is not None:
    							begin = int(capital.find('begin').text)
    						else:
    							begin = default_cap_begin
    						if capital.find('duration') is not None:
    							duration = int(capital.find('duration').text)
    						else:
    							duration = default_cap_duration
    					if capital.find('overnight_cost') is not None:
    						overnight_cost = int(capital.find('overnight_cost').text)
    					else:
    						overnight_cost = default_cap_overnight
    					if capital.find('shape') is not None:
    						shape = capital.find('costs_shape').text
    					else:
    						shape = default_cap_shape
    					for facility in institution.find('facility'):
    						id = int(facility.find('id').text)
    						rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)
    		else:
    			capital = region.find('capital')
    			if capital.find('pace') is not None: #random parameters
    				if capital.find('pace').text == "rapid":
    					begin = rapid_cap_begin + int(lst[j])
    					duration = rapid_cap_duration + 2 * int(lst[j])
    				elif capital.find('pace').text == "slow":
    					begin = slow_cap_begin + int(lst[j])
    					duration = slow_cap_duration + 2 * int(lst[j])
    				else: #normal
    					begin = default_cap_begin + int(lst[j])
    					duration = default_cap_duration + 2 * int(lst[j])
    				j += 1
    			else: #own parameters
    				if capital.find('begin') is not None:
    					begin = int(capital.find('begin').text)
    				else:
    					begin = default_cap_begin
    				if capital.find('duration') is not None:
    					duration = int(capital.find('duration').text)
    				else:
    					duration = default_cap_duration
    			if capital.find('overnight_cost') is not None:
    				overnight_cost = int(capital.find('overnight_cost').text)
    			else:
    				overnight_cost = default_cap_overnight
    			if capital.find('shape') is not None:
    				shape = capital.find('costs_shape').text
    			else:
    				shape = default_cap_shape
    			for institution in region.find('instituion'):
    				for facility in institution.find('facility'):
    					id = int(facility.find('id').text)
    					rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)	
    else:
    	capital = root.find('capital')
    	if capital.find('pace') is not None: #random parameters
    		if capital.find('pace').text == "rapid":
    			begin = rapid_cap_begin + int(lst[j])
    			duration = rapid_cap_duration + 2 * int(lst[j])
    		elif capital.find('pace').text == "slow":
    			begin = slow_cap_begin + int(lst[j])
    			duration = slow_cap_duration + 2 * int(lst[j])
    		else: #normal
    			begin = default_cap_begin + int(lst[j])
    			duration = default_cap_duration + 2 * int(lst[j])
    		j += 1
    	else: #own parameters
    		if capital.find('begin') is not None:
    			begin = int(capital.find('begin').text)
    		else:
    			begin = default_cap_begin
    		if capital.find('duration') is not None:
    			duration = int(capital.find('duration').text)
    		else:
    			duration = default_cap_duration
    	if capital.find('overnight_cost') is not None:
    		overnight_cost = int(capital.find('overnight_cost').text)
    	else:
    		overnight_cost = default_cap_overnight
    	if capital.find('shape') is not None:
    		shape = capital.find('costs_shape').text
    	else:
    		shape = default_cap_shape
    	for id in id_reactors:
    		rtn = pd.concat([rtn, reactor_i_df], ignore_index=True)	
    return rtn
	
del _eideps, _eischema