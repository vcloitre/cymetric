###############################################################################
# Functions dedicated to visualization (basically plots) of economic calcula-
# tions
###############################################################################

###############################################################################
# We want to be able to :
# - compare region policies (i.e. impact of financial parameters on prices)
#
#
#
#
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lcoe import s_lcoe, w_lcoe, average_cost, annual_cost

############
# Reactors #
############

def lcoe_plot(output_db, reactor_id):
    """Levelized cost of electricity that we get with one reactor technology
    (given a fuel cycle technology)/the whole fuel cycle we built for the
    simulation
    """
    pd.Series(lifetime*[lcoe(output_db, reactor_id)]).plot()
    plt.show()

def annual_cost_plot(output_db, reactor_id):
    pd.Series(annual_cost(output_db, reactor_id)).plot()
    plt.show()

def average_cost_plot(output_db, reactor_id):
    pd.Series(lifetime*[average_cost(output_db, reactor_id)]).plot()
    plt.show()

    
####
def agents_cash(output_db, agent_id):
    """For a given agent, plot the cash_flows (positive values = income,
    negative values = expenditure). Could be a region, institution or facility.
    """
  
def reactor_costs(output_db, *args):
    """Plot cash flows for a reactor restricted to one component of the costs
    """

