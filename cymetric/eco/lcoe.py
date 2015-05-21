###########################################
# Calculate levalized cost of electricity #
###########################################
# idea : could model existing nuclear production of a region (e.g. US
# and compare it to the actual price. This Comparison would give rise
# to an analysis of the modeling of Cyclus transaction model (what
# could be criticized, improved, what is cyclus good/bad for economic
# modeling

import pandas as pd
import numpy as np
from eco_inputs import discount_rate, fixedOM, variableOM, heat_rate, 
fuel_cost, ovn_cap_cost
from metrics import capital_cost, fuel_cost, decommissioning_cost,
operation_maintenance

def s_lcoe(output_db, reactor_id):
    """Given an AgentId corresponding to a reactor, gather all costs and 
    calculate lcoe (need probably to import financial parameters
    http://www.nrel.gov/analysis/tech_lcoe.html
    """
    n = lifetime
    i = discount_rate
    crf = i * (1 + i)**n / (((1 + i)**n)-1)
    return  ((ovn_cap_cost * crf + fixedOM )/(8760 * capacity_factor)) + 
            (fuel_cost * heat_rate) + variableOM


def w_lcoe(output_db, reactor_id):
    """Given an AgentId corresponding to a reactor, gather all cost and
    calculate lcoe (need probably to import financial parameters  http://en.wikipedia.org/w/index.php?title=Levelised_energy_cost&oldid=365220846
    """
    r = discount_rate
    for t in range(1, lifetime):
        num += (invest[t] + om[t] + fuel_exp[i]) / ( 1 + r)**t
        den += gen_elec[t] / ( 1 + r)**t
    return num / den
# See calculating_the_levelized_cost_of_electricity.pdf to have a more complex
# method or d'Haeseleer provides a very similar method (page 40)

def annual_cost(output_db, reactor_id):
    """Given an AgentId corresponding to a reactor, calculate annual costs over
    the lifetime of the reactor
    """
    rtn = []
    for t in range(1, lifetime):
        rtn.append(#/!\/!\/!\/!\#sum all costs#/!\/!\/!\/!\#)
    return rtn

def average_cost(output_db, reactor_id):
    """Given an AgentId corresponding to a reactor, gather all annual 
    costs and average it over the lifetime of the reactor
    """
    return np.mean(annual_cost(output_db, reactor_id))
