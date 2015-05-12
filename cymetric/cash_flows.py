###################################################################################
# Functions dedicated to visualization (basically plots) of economic calculations #
###################################################################################

########################################################################################################
# We want to be able to :
# - compare region policies (i.e. impact of financial parameters on prices)
#
#
#
#
########################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def lcoe(output_db):
    """Levelized cost of electricity that we get with one reactor technology (given a fuel cycle technology)/the whole fuel cycle we built for the simulation
    """


def agents_cash(output_db, agent_id):
    """For a given agent, plot the cash_flows (positive values = income, negative values = expenditure)
    Could be a region, institution or facility.
    """
  
def reactor_costs(output_db, *args):
    """Plot cash flows for a reactor restricted to one component of the costs
    """

