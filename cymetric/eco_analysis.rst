Performing economic analysis with the economic tools provided by Cymetric
=========================================================================


Several functions are especially dedicated to the economic analysis of fuel cycles using Cymetric. The related files are the following :

| - eco_inputs.py
| - eco_metrics.py
| - lcoe.py
| - cash_flows.py
| - test_eco_metrics.py

Economic metrics calculation
----------------------------
eco_metrics.py has the same goal as metrics.py (containing metrics), only it is only containing the economic metrics.
Until now, these metrics are all related to reactor costs (capital cost, fixed O&M cost, decommissioning cost, variable O&M cost, fuel cost and waste fee). As for the front end and the back end of the fuel cycle, prices will be fixed at realistic values according to actual values. This is because prices should be  calculated by the Dynamic Resource Exchange.

Additional tools needed for analysis
------------------------------------
In order to make the analysis more complex and realistic, some tools are stored in eco_inputs.py.
First, some financial parameters are used in order to calculate realistic costs. These parameters are imported in eco_metrics.py.
Second, eco_inputs.py contains a few functions that mainly add complexity to the calculation of the metrics in eco_metrics.py.

Cash flows visualization
------------------------
Given all the fuel cycle costs, we are able to gather them all to calculate the monthly/annual total cash flows. We could also calculate important indicators such as the levelized cost of electricity. This features are stored in cash_flows/lcoe.py. 