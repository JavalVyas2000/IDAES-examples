import os
from collections import OrderedDict

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# IDAES Imports
from idaes.core import FlowsheetBlock
from idaes.core.util.initialization import propagate_state
from idaes.core.util import model_serializer as ms, ModelTag, ModelTagGroup
from idaes.core.util.tags import svg_tag
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.exceptions import InitializationError

import idaes.core.util.scaling as iscale

from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.base.generic_reaction import (
    GenericReactionParameterBlock,
)

from idaes.models.unit_models import (
    Mixer,
    Heater,
    HeatExchanger,
    PressureChanger,
    GibbsReactor,
    StoichiometricReactor,
    Separator,
    Translator,
)
from idaes.models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback,
)
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.separator import SplittingType
from idaes.models.unit_models.mixer import MomentumMixingType

from idaes.models_extra.power_generation.properties.natural_gas_PR import (
    get_prop,
    get_rxn,
)

from idaes_examples.mod.power_gen.SOFC_ROM import (
    build_SOFC_ROM,
    initialize_SOFC_ROM,
)

from idaes_examples.mod.power_gen import NGFC_flowsheet as NGFC

import logging

# # create model and flowsheet
m = pyo.ConcreteModel(name="NGFC no CCS")
m.fs = FlowsheetBlock(dynamic=False)

# create the solver
solver = pyo.SolverFactory("ipopt")
solver.options = {"bound_push": 1e-16}

NG_config = get_prop(
        components=[
            "H2",
            "CO",
            "H2O",
            "CO2",
            "CH4",
            "C2H6",
            "C3H8",
            "C4H10",
            "N2",
            "O2",
            "Ar",
        ]
    )
m.fs.NG_props = GenericParameterBlock(**NG_config)

syn_config = get_prop(
    components=["H2", "CO", "H2O", "CO2", "CH4", "N2", "O2", "Ar"]
)
m.fs.syn_props = GenericParameterBlock(**syn_config)

air_config = get_prop(components=["H2O", "CO2", "N2", "O2", "Ar"])
m.fs.air_props = GenericParameterBlock(**air_config)

m.fs.rxn_props = GenericReactionParameterBlock(
    **get_rxn(m.fs.syn_props, reactions=["h2_cmb", "co_cmb", "ch4_cmb"])
)
def reformer():
    NGFC.build_reformer(m)
    m.fs.NG_props.set_default_scaling("flow_mol", 1e-3)
    m.fs.NG_props.set_default_scaling("flow_mol_phase", 1e-3)
    m.fs.NG_props.set_default_scaling("temperature", 1e-2)
    m.fs.NG_props.set_default_scaling("pressure", 1e-5)

    m.fs.NG_props.set_default_scaling("mole_frac_comp", 1e2)
    m.fs.NG_props.set_default_scaling("mole_frac_comp", 1e2, index="C2H6")
    m.fs.NG_props.set_default_scaling("mole_frac_comp", 1e2, index="C3H8")
    m.fs.NG_props.set_default_scaling("mole_frac_comp", 1e2, index="C4H10")

    m.fs.NG_props.set_default_scaling("mole_frac_phase_comp", 1e2)
    m.fs.NG_props.set_default_scaling(
        "mole_frac_phase_comp", 1e2, index=("Vap", "C2H6")
    )
    m.fs.NG_props.set_default_scaling(
        "mole_frac_phase_comp", 1e2, index=("Vap", "C3H8")
    )
    m.fs.NG_props.set_default_scaling(
        "mole_frac_phase_comp", 1e2, index=("Vap", "C4H10")
    )

    m.fs.NG_props.set_default_scaling("enth_mol_phase", 1e-6)
    m.fs.NG_props.set_default_scaling("entr_mol_phase", 1e-4)

    # set syn_props default scaling
    m.fs.syn_props.set_default_scaling("flow_mol", 1e-3)
    m.fs.syn_props.set_default_scaling("flow_mol_phase", 1e-3)
    m.fs.syn_props.set_default_scaling("temperature", 1e-2)
    m.fs.syn_props.set_default_scaling("pressure", 1e-5)
    m.fs.syn_props.set_default_scaling("mole_frac_comp", 1e2)
    m.fs.syn_props.set_default_scaling("mole_frac_phase_comp", 1e2)
    m.fs.syn_props.set_default_scaling("enth_mol_phase", 1e-6)
    m.fs.syn_props.set_default_scaling("entr_mol_phase", 1e-4)

    # set air_props default scaling
    m.fs.air_props.set_default_scaling("flow_mol", 1e-3)
    m.fs.air_props.set_default_scaling("flow_mol_phase", 1e-3)
    m.fs.air_props.set_default_scaling("temperature", 1e-2)
    m.fs.air_props.set_default_scaling("pressure", 1e-5)
    m.fs.air_props.set_default_scaling("mole_frac_comp", 1e2)
    m.fs.air_props.set_default_scaling("mole_frac_phase_comp", 1e2)
    m.fs.air_props.set_default_scaling("enth_mol_phase", 1e-6)
    m.fs.air_props.set_default_scaling("entr_mol_phase", 1e-4)

    iscale.set_scaling_factor(m.fs.reformer.lagrange_mult, 1e-4)

    # overwrite mole_frac lower bound to remove warnings
    print('overwriting mole_frac lower bound, set to 0 to remove warnings')
    for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
        if '.mole_frac' in var.name: # don't catch log_mole_frac variables
            var.setlb(0)

    # some specific variable scaling

    # heat exchanger areas and overall heat transfer coefficiencts
    iscale.set_scaling_factor(m.fs.reformer_recuperator.area, 1e-4)
    iscale.set_scaling_factor(m.fs.reformer_recuperator.overall_heat_transfer_coefficient, 1)

    # control volume heats
    iscale.set_scaling_factor(m.fs.intercooler_s1.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.intercooler_s2.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.reformer.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.reformer_recuperator.shell.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.reformer_recuperator.tube.heat, 1e-6)

    # work
    iscale.set_scaling_factor(m.fs.air_compressor_s1.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.air_compressor_s2.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.NG_expander.control_volume.work, 1e-6)

    # reaction extents

    print('Scaling flowsheet constraints')

    list_units = ['reformer_recuperator',
                    'NG_expander', 'reformer_bypass', 'air_compressor_s1',
                    'intercooler_s1', 'air_compressor_s2', 'intercooler_s2',
                    'reformer_mix', 'reformer', 'bypass_rejoin']

    # set scaling for unit constraints
    for name in list_units:
        unit = getattr(m.fs, name)
        # mixer constraints
        if hasattr(unit, 'material_mixing_equations'):
            for (t, j), c in unit.material_mixing_equations.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
        if hasattr(unit, 'enthalpy_mixing_equations'):
            for t, c in unit.enthalpy_mixing_equations.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
        if hasattr(unit, 'minimum_pressure_constraint'):
            for (t, i), c in unit.minimum_pressure_constraint.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)
        if hasattr(unit, 'mixture_pressure'):
            for t, c in unit.mixture_pressure.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)

        # separator constraints
        if hasattr(unit, 'material_splitting_eqn'):
            for (t, o, j), c in unit.material_splitting_eqn.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)
        if hasattr(unit, 'temperature_equality_eqn'):
            for (t, o), c in unit.temperature_equality_eqn.items():
                iscale.constraint_scaling_transform(c, 1e-2, overwrite=False)
        if hasattr(unit, 'pressure_equality_eqn'):
            for (t, o), c in unit.pressure_equality_eqn.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)
        if hasattr(unit, 'sum_split_frac'):
            for t, c in unit.sum_split_frac.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)

        # pressurechanger constraints

        if hasattr(unit, "ratioP_calculation"):
            for t, c in unit.ratioP_calculation.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)

        if hasattr(unit, "actual_work"):
            for t, c in unit.actual_work.items():
                iscale.constraint_scaling_transform(c, 1e-6, overwrite=False)

        if hasattr(unit, "isentropic_pressure"):
            for t, c in unit.isentropic_pressure.items():
                iscale.constraint_scaling_transform(c, 1e-5, overwrite=False)

        if hasattr(unit, "isentropic"):
            for t, c in unit.isentropic.items():
                iscale.constraint_scaling_transform(c, 1e-1, overwrite=False)

        if hasattr(unit, "isentropic_energy_balance"):
            for t, c in unit.isentropic_energy_balance.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)

        if hasattr(unit, "state_material_balances"):
            for (t, j), c in unit.state_material_balances.items():
                iscale.constraint_scaling_transform(c, 1e-3, overwrite=False)

        # HeatExchanger non-CV constraints
        if hasattr(unit, "heat_transfer_equation"):
            for t, c in unit.heat_transfer_equation.items():
                iscale.constraint_scaling_transform(c, 1e-7, overwrite=False)

        if hasattr(unit, "unit_heat_balance"):
            for t, c in unit.unit_heat_balance.items():
                iscale.constraint_scaling_transform(c, 1e-7, overwrite=False)

        if hasattr(unit, "delta_temperature_in_equation"):
            for t, c in unit.delta_temperature_in_equation.items():
                iscale.constraint_scaling_transform(c, 1e-1, overwrite=False)

        if hasattr(unit, "delta_temperature_out_equation"):
            for t, c in unit.delta_temperature_out_equation.items():
                iscale.constraint_scaling_transform(c, 1e-1, overwrite=False)

        # Translator has no constraints to scale
        # Gibbs reactor minimization is scaled elsewhere, set by gibbs_scaling
        # adding scaling factors of unity here for completeness
        if hasattr(unit, "gibbs_minimization"):
            for (t, p, j), c in unit.gibbs_minimization.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)

        if hasattr(unit, "inert_species_balance"):
            for (t, p, j), c in unit.inert_species_balance.items():
                iscale.constraint_scaling_transform(c, 1, overwrite=False)

    print('Calculating scaling factors')
    iscale.calculate_scaling_factors(m)
    print()

    NGFC.set_reformer_inputs(m)
    NGFC.initialize_reformer(m)

    # for i in m.fs.reformer.control_volume.properties_in[0.0].component_data_objects(pyo.Constraint):
    #     print(i)
    # m.fs.reformer.control_volume.properties_out[0.0].mole_frac_comp.pprint()
    # m.fs.reformer_recuperator.tube_inlet.display()
    # m.fs.reformer.display()
    # m.fs.reformer.outlet.display()
    # m.fs.reformer_mix.inlet.display()
    print('reformer_recuperator')
    m.fs.reformer_recuperator.tube_inlet.display()
    m.fs.reformer_recuperator.shell_inlet.display()
    print('NG_expander')
    m.fs.NG_expander.inlet.display()
    print('reformer_bypass')
    m.fs.reformer_bypass.inlet.display()
    print('air_compressor_s1')
    m.fs.air_compressor_s1.inlet.display()
    print('intercooler_s1')
    m.fs.intercooler_s1.inlet.display()
    print('air_compressor_s2')
    m.fs.air_compressor_s2.inlet.display()
    print('intercooler_s2')
    m.fs.intercooler_s2.inlet.display()
    print('reformer_mix')
    m.fs.reformer_mix.steam_inlet.display()
    m.fs.reformer_mix.oxygen_inlet.display()
    m.fs.reformer_mix.gas_inlet.display()
    m.fs.reformer_mix.outlet.display()
    print('reformer')
    m.fs.reformer.inlet.display()
    print('bypass_rejoin')
    m.fs.bypass_rejoin.syngas_inlet.display()
    m.fs.bypass_rejoin.bypass_inlet.display()
    m.fs.reformer_recuperator.shell_inlet.display()
    m.fs.reformer.outlet.display()

reformer()