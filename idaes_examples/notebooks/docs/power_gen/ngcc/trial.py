import os
import numpy as np
import pandas as pd
from IPython.core.display import SVG
import pyomo.environ as pyo
import idaes
from idaes.core.solvers import use_idaes_solver_configuration_defaults
import idaes.core.util.scaling as iscale
import idaes.core.util as iutil
from idaes_examples.mod.power_gen import ngcc
import pytest
import logging

logging.getLogger("pyomo").setLevel(logging.ERROR)

def make_directory(path):
    """Make a directory if it doesn't exist"""
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


make_directory("data")
make_directory("data_pfds")
make_directory("data_tabulated")

use_idaes_solver_configuration_defaults()
idaes.cfg.ipopt.options.nlp_scaling_method = "user-scaling"
idaes.cfg.ipopt.options.linear_solver = "ma57"
idaes.cfg.ipopt.options.OF_ma57_automatic_scaling = "yes"
idaes.cfg.ipopt.options.ma57_pivtol = 1e-5
idaes.cfg.ipopt.options.ma57_pivtolmax = 0.1
# idaes.cfg.ipopt.options.halt_on_ampl_error = "yes"
solver = pyo.SolverFactory("ipopt")

m = pyo.ConcreteModel()
m.fs = ngcc.NgccFlowsheet(dynamic=False)
iscale.calculate_scaling_factors(m)
m.fs.initialize(
    load_from='ngcc_init.json.gz',
    save_to='ngcc_init.json.gz',
)
res = solver.solve(m, tee=True)

def display_pfd():
    print("\n\nGas Turbine Section\n")
    display(SVG(m.fs.gt.write_pfd()))
    print("\n\nHRSG Section\n")
    display(SVG(m.fs.hrsg.write_pfd()))
    print("\n\nSteam Turbine Section\n")
    display(SVG(m.fs.st.write_pfd()))


display_pfd()

m.fs.gt.write_pfd(fname="data_pfds/gt_baseline.svg")
m.fs.hrsg.write_pfd(fname="data_pfds/hrsg_baseline.svg")
m.fs.st.write_pfd(fname="data_pfds/st_baseline.svg")

# Assert results approximately agree with baseline reoprt
assert pyo.value(m.fs.net_power_mw[0]) == pytest.approx(646)
assert pyo.value(m.fs.gross_power[0]) == pytest.approx(-690e6, rel=0.001)
assert pyo.value(100 * m.fs.lhv_efficiency[0]) == pytest.approx(52.8, abs=0.1)
assert pyo.value(
    m.fs.total_variable_cost_rate[0] / m.fs.net_power_mw[0]
) == pytest.approx(37.2799, rel=0.01)
assert pyo.value(m.fs.fuel_cost_rate[0] / m.fs.net_power_mw[0]) == pytest.approx(
    31.6462, rel=0.01
)
assert pyo.value(
    m.fs.other_variable_cost_rate[0] / m.fs.net_power_mw[0]
) == pytest.approx(5.63373, rel=0.01)
assert pyo.value(m.fs.gt.gt_power[0]) == pytest.approx(-477e6, rel=0.001)


from idaes.core.util import DiagnosticsToolbox

dt=DiagnosticsToolbox(m)
dt.report_numerical_issues()