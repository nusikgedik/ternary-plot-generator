from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpltern.datasets import get_triangular_grid
import numpy as np
import mpltern


class UnitType(Enum):
    ATOMIC_PERCENT = 0
    MOLAR_PERCENT = 1
    WEIGHT_PERCENT = 2


class ScatterplotTypes(Enum):
    DEFINED = 0
    MEASURED = 1


class SystemType(Enum):
    BINARY = 0
    TERNARY = 1


def ternary_plot(system, unit_type=UnitType.ATOMIC_PERCENT):
    '''
    :param system: the ternary system represented as component and molar weights as a three item list of tuples
    :param unit_type: valid options are defined in UnitType
    :return: plots the ternary diagram
    '''

    plt.rcParams.update({'font.size': 16})
    metal_t, metal_per_molecule_t, component_t, molar_mass_t = system[0]
    metal_l, metal_per_molecule_l, component_l, molar_mass_l = system[1]
    metal_r, metal_per_molecule_r, component_r, molar_mass_r = system[2]

    # Define plot settings
    plot_setup = {
        UnitType.MOLAR_PERCENT: {
            "unit": "(mol %)",
            "ternary_sum": 100.0,
            "system_name": f"{component_t}-{component_l}-{component_r} system"
        },
        UnitType.ATOMIC_PERCENT: {
            "unit": "(at %)",
            "ternary_sum": 100.0,
            "system_name": f"{metal_t}-{metal_l}-{metal_r}-O system"
        },
        UnitType.WEIGHT_PERCENT: {
            "unit": "(wt %)",
            "ternary_sum": 100.0,
            "system_name": f"{metal_t}-{metal_l}-{metal_r}-O system"
        }
    }

    # Apply settings based on unit_type
    settings = plot_setup.get(unit_type)
    if settings is None:
        raise ValueError(f"Unsupported unit type: {unit_type}")

    unit = settings["unit"]
    system_name = settings["system_name"]
    fig, ax = plt.subplots(subplot_kw={"projection": "ternary", "ternary_sum": settings["ternary_sum"]})
    fig.canvas.manager.set_window_title(f"{system_name} {unit}")

    # Set label name, color and position
    if unit_type == UnitType.MOLAR_PERCENT:
        ax.set_tlabel(f"{component_t} {unit}")
        ax.set_llabel(f"{component_l} {unit}")
        ax.set_rlabel(f"{component_r} {unit}")
    if unit_type == UnitType.ATOMIC_PERCENT:
        ax.set_tlabel(f"{metal_t} {unit}")
        ax.set_llabel(f"{metal_l} {unit}")
        ax.set_rlabel(f"{metal_r} {unit}")

    # Set locators for all ternary axes relative to the ternary sum
    # Set label and tick colors for each axis
    # Set axis label position
    for axis, color in zip([ax.taxis, ax.laxis, ax.raxis], ["orange", "blue", "green"]):
        axis.set_major_locator(MultipleLocator(settings["ternary_sum"] / 5))
        axis.set_minor_locator(MultipleLocator(settings["ternary_sum"] / 10))
        axis.label.set_color(color)
        axis.set_tick_params(colors=color)
        axis.set_label_position('tick1')

    # Add grid pattern on plot
    ax.grid(which='both')

    # Draw grid lines and ticks below other plot elements
    ax.set_axisbelow(True)

    return ax


def binary_composition_diagram(system):
    '''
    :param system: the binary system represented as component and molar weights as a two item list of tuples
    :param total_mass: total mass that will end up in crucible (in grams)
    :param mass_uncertainty: (in grams) meaning +- uncertainty
    :return: plots the ternary diagram
    '''
    component_1, molar_mass_1 = system[0]
    component_2, molar_mass_2 = system[1]

    # X-axis from A to B (as %A)
    x = [0, 100]  # 0% A (100% B) to 100% A (0% B)
    y = [0, 0]  # Single line (flat)

    plt.figure(figsize=(8, 1))
    ax = plt.subplot()
    ax.plot(x, y, 'k-', linewidth=1)  # black line
    #ax.xaxis.set_major_locator(MultipleLocator(10))

    # Optional: mark ends
    plt.text(0, 0.05, f'100% {component_2}', ha='center')
    plt.text(100, 0.05, f'100% {component_1}', ha='center')

    # Remove y-axis and frame for clarity
    plt.yticks([])
    plt.xticks(np.linspace(0, 100, 11))
    plt.xlim(0, 100)
    plt.ylim(-0.05, 0.1)
    plt.title(f'Binary Composition: {component_1}–{component_2}')

    # Draw grid lines and ticks below other plot elements
    ax.set_axisbelow(True)

    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    return ax


def generate_equidistant_points(system_type, n=11):
    '''
    :param system_type: Whether the system is Binary or Ternary
    :return: data_set: Tuple containing t, l, r as numpy arrays if Ternary. Single np array if Binary.
    '''
    if system_type == SystemType.TERNARY:
        t, l, r = get_triangular_grid(n=n)

        is_not_zero = ~((t == 0) | (l == 0) | (r == 0))
        t = t[is_not_zero]
        l = l[is_not_zero]
        r = r[is_not_zero]

        data_set = (t, l, r)
    elif system_type == SystemType.BINARY:
        data_set = (np.linspace(0, 100, n + 2)[1:-1],)
    else:
        raise ValueError(f"Do not know how to generate points for system type {system_type}")
    return data_set


def plot_with_points(ax, data_set, scatter_type=ScatterplotTypes.DEFINED):
    '''
    :param ax: a matplotlib axis
    :param data_set: A tuple of three numpy arrays containing three coordinates of each point
    '''
    # If data_set only has one numpy array, it means it's datapoints for a binary system and we must add the
    # y-coordinates (all zeros)
    if len(data_set) == 1:
        data_set = (data_set[0], np.zeros(len(data_set[0])))

    if scatter_type == ScatterplotTypes.DEFINED:
        # Create scatter plot for the generated or defined points
        ax.scatter(*data_set, s=20, c="k", zorder=10)
    elif scatter_type == ScatterplotTypes.MEASURED:
        # Create scatter plot for additional points
        ax.scatter(*data_set, s=30, c="r", marker="x", zorder=10)

    else:
        raise ValueError(f"Do not know how to plot scatterplot type {scatter_type}")


def plot_error_bars(ax, system, data_set, total_mass, mass_uncertainty, unit_type=UnitType.ATOMIC_PERCENT):
    """
    Plots error bars on the plot 'ax'.

    Uses the length of the tuple 'data_set' and the list 'system' to decide if the system is binary or ternary,
    and switches between 'plot_binary_error_bars' and 'plot_ternary_error_bars' accordingly.

    :param ax: Matplotlib axes
    :param system: The ternary or binary system represented as component and molar weights as a list of tuples
    :param data_set:
    :param total_mass:
    :param mass_uncertainty:
    :param unit_type:
    :return:
    """

    system_type = None
    # First figure out if the system is binary or ternary
    if len(system) == 2:
        system_type = SystemType.BINARY
    if len(system) == 3:
        system_type = SystemType.TERNARY

    # Verify that the number of arrays in 'data_set' is as expected
    if system_type == SystemType.BINARY and len(data_set) != 1:
        raise ValueError(f"System is Binary, but 'data_set' contains {len(data_set)} arrays (expected 1)")
    if system_type == SystemType.TERNARY and len(data_set) != 3:
        raise ValueError(f"System is Binary, but 'data_set' contains {len(data_set)} arrays (expected 3)")

    if system_type == SystemType.BINARY:
        _plot_binary_error_bars(ax, system, data_set, total_mass, mass_uncertainty, unit_type)
    elif system_type == SystemType.TERNARY:
        _plot_ternary_error_bars(ax, system, data_set, total_mass, mass_uncertainty, unit_type)
    else:
        ValueError(f"Unknown system type: {system_type}")


def mol_percent_data_set_calculator(atomic_equidistant_data_set, system, system_type):
    if system_type == SystemType.TERNARY:
        metal_t, metal_per_molecule_t, component_t, molar_mass_t = system[0]
        metal_l, metal_per_molecule_l, component_l, molar_mass_l = system[1]
        metal_r, metal_per_molecule_r, component_r, molar_mass_r = system[2]
        t, l, r = atomic_equidistant_data_set
        mol_percent_data_set = [[], [], []]
        for i in range(len(t)):
            atomic_to_molar_percentage_conversion = t[i]/metal_per_molecule_t + l[i]/metal_per_molecule_l + r[i]/metal_per_molecule_r
            molecular_percentage_t = ((t[i]/metal_per_molecule_t) * 100) / atomic_to_molar_percentage_conversion
            molecular_percentage_l = ((l[i] / metal_per_molecule_l) * 100) / atomic_to_molar_percentage_conversion
            molecular_percentage_r = ((r[i] / metal_per_molecule_r) * 100) / atomic_to_molar_percentage_conversion
            mol_percent_data_set[0].append(molecular_percentage_t)
            mol_percent_data_set[1].append(molecular_percentage_l)
            mol_percent_data_set[2].append(molecular_percentage_r)
        return mol_percent_data_set
    elif system_type == SystemType.BINARY:
        component_a, metal_per_molecule_a, component_a, molar_mass_a = system[0]
        component_b, metal_per_molecule_b, component_b, molar_mass_b = system[1]
        points = atomic_equidistant_data_set[0]
        mol_percent_data_set = [[], []]
        for i in range(len(points)):
            atomic_to_molar_percentage_conversion = points[i]/metal_per_molecule_a + (100-points[i])/metal_per_molecule_b
            molecular_percentage_a = ((points[i]/metal_per_molecule_a) * 100) / atomic_to_molar_percentage_conversion
            molecular_percentage_b = 100-molecular_percentage_a
            mol_percent_data_set[0].append(molecular_percentage_a)
            mol_percent_data_set[1].append(molecular_percentage_b)
        return mol_percent_data_set
    else:
        raise ValueError(f"Do not know how to generate points for system type {system_type}")


def print_weighing_table(mol_percent_data_set, system, system_type, total_weight=0.5):
    if system_type == SystemType.TERNARY:
        metal_t, metal_per_molecule_t, component_t, molar_mass_t = system[0]
        metal_l, metal_per_molecule_l, component_l, molar_mass_l = system[1]
        metal_r, metal_per_molecule_r, component_r, molar_mass_r = system[2]
        t, l, r = mol_percent_data_set
        print(f"{component_t}, {component_l} , {component_r}")
        for i in range(len(t)):
            molar_sum = t[i] * molar_mass_t + l[i] * molar_mass_l + r[i] * molar_mass_r
            scale_factor = total_weight/molar_sum
            t_weight = t[i] * molar_mass_t * scale_factor
            l_weight = l[i] * molar_mass_l * scale_factor
            r_weight = r[i] * molar_mass_r * scale_factor
            print(f"{t_weight:.4f}g, {l_weight:.4f}g , {r_weight:.4f}g")

    elif system_type == SystemType.BINARY:
        component_a, metal_per_molecule_a, component_a, molar_mass_a = system[0]
        component_b, metal_per_molecule_b, component_b, molar_mass_b = system[1]
        points = mol_percent_data_set[0]
        print(f"{component_a}, {component_b}")
        for i in range(len(points)):
            molar_sum = points[i] * molar_mass_a + (100-points[i]) * molar_mass_b
            scale_factor = total_weight/molar_sum
            a_weight = points[i] * molar_mass_a * scale_factor
            b_weight = (100-points[i]) * molar_mass_b * scale_factor
            print(f"{a_weight:.4f}g , {b_weight:.4f}g")

    else:
        raise ValueError(f"Do not know how to generate points for system type {system_type}")


def weight_to_atomic_percent_dataset_calculator(system, system_type, measured_weight):
    experimental_mol_percent_data_list = []
    if system_type == SystemType.TERNARY:
        metal_t, metal_per_molecule_t, component_t, molar_mass_t = system[0]
        metal_l, metal_per_molecule_l, component_l, molar_mass_l = system[1]
        metal_r, metal_per_molecule_r, component_r, molar_mass_r = system[2]

        weight_t_list = measured_weight[0]
        weight_l_list = measured_weight[1]
        weight_r_list = measured_weight[2]

        experimental_mol_percent_t = []
        experimental_mol_percent_l = []
        experimental_mol_percent_r = []
        for weight_t, weight_l, weight_r in zip(weight_t_list, weight_l_list, weight_r_list):
            molar_sum = (weight_t / molar_mass_t) + (weight_l / molar_mass_l) + (weight_r / molar_mass_r)
            atomic_percent_t = (100 * (weight_t / molar_mass_t)) / molar_sum
            atomic_percent_l = (100 * (weight_l / molar_mass_l)) / molar_sum
            atomic_percent_r = (100 * (weight_r / molar_mass_r)) / molar_sum
            experimental_mol_percent_t.append(atomic_percent_t)
            experimental_mol_percent_l.append(atomic_percent_l)
            experimental_mol_percent_r.append(atomic_percent_r)

        return (np.array(experimental_mol_percent_t),
                np.array(experimental_mol_percent_l),
                np.array(experimental_mol_percent_r))

    elif system_type == SystemType.BINARY:
        component_a, molar_mass_a = system[0]
        component_b, molar_mass_b = system[1]

        weight_a_list = measured_weight[0]
        weight_b_list = measured_weight[1]

        experimental_mol_percent_a = []
        experimental_mol_percent_b = []

        for weight_a, weight_b in zip(weight_a_list, weight_b_list):
            molar_sum = (weight_a / molar_mass_a) + (weight_b / molar_mass_b)
            atomic_percent_a = (100 * (weight_a / molar_mass_a)) / molar_sum
            atomic_percent_b = (100 * (weight_b / molar_mass_b)) / molar_sum
            experimental_mol_percent_a.append(atomic_percent_a)
            experimental_mol_percent_b.append(atomic_percent_b)

        return (np.array(experimental_mol_percent_a), )

    else:
        raise ValueError(f"Do not know how to generate points for system type {system_type}")


def _plot_binary_error_bars(ax, system, data_set, total_mass, mass_uncertainty, unit_type=UnitType.ATOMIC_PERCENT):
    '''
    :param data_set: A tuple of one numpy array containing coordinates of each point
    :param total_mass: total mass that will end up in crucible (in grams)
    :param mass_uncertainty: (in grams) meaning +- uncertainty
    '''
    points = data_set[0]
    metal_1, metal_per_molecule_1, component_1, molar_mass_1 = system[0]
    metal_2, metal_per_molecule_2, component_2, molar_mass_2 = system[1]

    if unit_type == UnitType.ATOMIC_PERCENT or unit_type == UnitType.MOLAR_PERCENT:
        mole_uncertainty_1 = mass_uncertainty / molar_mass_1
        mole_uncertainty_2 = mass_uncertainty / molar_mass_2
    elif unit_type == UnitType.WEIGHT_PERCENT:
        # In case of wt% each system component has constant error
        wt_error = mass_uncertainty / total_mass
        # Combine the errors to get total possible error using standard error propagation techniques
        combined_error = np.sqrt(wt_error ** 2 + wt_error ** 2)
    else:
        raise ValueError(f"Do not know how to plot unit type {unit_type}")

    # Set error area color
    error_bars_linewidth = 0.5
    err_bars_color = "grey"

    for point in points:
        # Outer loop runs one for each point in plot

        # Calculate errors based on whether we're using mole percentages or not
        if unit_type == UnitType.ATOMIC_PERCENT or unit_type == UnitType.MOLAR_PERCENT:
            scale_factor = total_mass / (point * molar_mass_1 + (1 - point) * molar_mass_2)

            # Scale the uncertainties
            error_1 = mole_uncertainty_1 / scale_factor
            error_2 = mole_uncertainty_2 / scale_factor

            # Combine the errors using standard error propagation
            combined_error = np.sqrt(error_1 ** 2 + error_2 ** 2)

        half_error_width = combined_error / 2
        error_bar_x_coords = np.array([-half_error_width, half_error_width]) + point
        error_bar_y_coords = np.zeros(2)
        ax.plot(error_bar_x_coords, error_bar_y_coords, linewidth=2.5, c='c')


def _plot_ternary_error_bars(ax, system, data_set, total_mass, mass_uncertainty, unit_type=UnitType.ATOMIC_PERCENT):
    '''
    :param data_set: A tuple of three numpy arrays containing three coordinates of each point
    :param total_mass: total mass that will end up in crucible (in grams)
    :param mass_uncertainty: (in grams) meaning +- uncertainty
    '''
    t, l, r = data_set
    metal_t, metal_per_molecule_t, component_t, molar_mass_t = system[0]
    metal_l, metal_per_molecule_l, component_l, molar_mass_l = system[1]
    metal_r, metal_per_molecule_r, component_r, molar_mass_r = system[2]

    if unit_type == UnitType.ATOMIC_PERCENT or unit_type == UnitType.MOLAR_PERCENT:
        mole_uncertainty_t = mass_uncertainty / molar_mass_t
        mole_uncertainty_l = mass_uncertainty / molar_mass_l
        mole_uncertainty_r = mass_uncertainty / molar_mass_r
    elif unit_type == UnitType.WEIGHT_PERCENT:
        # In case of wt%, all axes have the same constant error
        error = mass_uncertainty / total_mass
        error_t = error_l = error_r = error
    else:
        raise ValueError(f"Do not know how to plot unit type {unit_type}")

    # Set error area color
    error_bars_linewidth = 0.5
    err_bars_color = "grey"

    for point_t, point_l, point_r in zip(t, l, r):
        # Outer loop runs one for each point in plot

        # Calculate errors based on whether we're using mole percentages or not
        if unit_type == UnitType.ATOMIC_PERCENT or unit_type == UnitType.MOLAR_PERCENT:
            scale_factor = total_mass / (point_t * molar_mass_t + point_l * molar_mass_l + point_r * molar_mass_r)

            # Scale the uncertainties
            error_t = mole_uncertainty_t / scale_factor
            error_l = mole_uncertainty_l / scale_factor
            error_r = mole_uncertainty_r / scale_factor

            _execute_hexagon_error_plotting(ax,
                                            (point_t, point_l, point_r),
                                            (error_t, error_l, error_r),
                                            err_bars_color)


def _execute_hexagon_error_plotting(ax, points, errors, err_bars_color):
    # Calculate half the size of the error of each axis
    point_t, point_l, point_r = points
    error_t, error_l, error_r = errors
    half_error_t = error_t / 2
    half_error_l = error_l / 2
    half_error_r = error_r / 2

    # Create arrays to store the hexagonal error bar coordinates
    # Each axis will use 2 positions in these arrays (6 total for 3 axes)
    hex_coords_t = np.full(shape=6, fill_value=np.nan)
    hex_coords_l = np.full(shape=6, fill_value=np.nan)
    hex_coords_r = np.full(shape=6, fill_value=np.nan)

    ####
    # Now we calculate the coordinates of the error bars.
    ###

    # Calculate tlr coordinates of the error bar on axis 0 (Top)
    hex_coords_t[:2] = [point_t - error_t, point_t + error_t]
    hex_coords_l[:2] = [point_l + half_error_t, point_l - half_error_t]
    hex_coords_r[:2] = [point_r + half_error_t, point_r - half_error_t]

    # Calculate tlr coordinates of the error bar on axis 1 (Left)
    hex_coords_t[2:4] = [point_t + half_error_l, point_t - half_error_l]
    hex_coords_l[2:4] = [point_l - error_l, point_l + error_l]
    hex_coords_r[2:4] = [point_r + half_error_l, point_r - half_error_l]

    # Calculate tlr coordinates of the error bar on axis 2 (Right)
    hex_coords_t[4:] = [point_t + half_error_r, point_t - half_error_r]
    hex_coords_l[4:] = [point_l + half_error_r, point_l - half_error_r]
    hex_coords_r[4:] = [point_r - error_r, point_r + error_r]

    hex_errs = True
    bar_errs = True
    if bar_errs:
        ax.plot(hex_coords_t[:2], hex_coords_l[:2], hex_coords_r[:2], linewidth=0.5, c='k')
        ax.plot(hex_coords_t[2:4], hex_coords_l[2:4], hex_coords_r[2:4], linewidth=0.5, c='k')
        ax.plot(hex_coords_t[4:], hex_coords_l[4:], hex_coords_r[4:], linewidth=0.5, c='k')
    if hex_errs:
        sorter = np.array([0, 5, 2, 1, 4, 3, 0])
        hex_coords_t = hex_coords_t[sorter]
        hex_coords_l = hex_coords_l[sorter]
        hex_coords_r = hex_coords_r[sorter]
        ax.fill(hex_coords_t, hex_coords_l, hex_coords_r,
                c=err_bars_color, alpha=0.2)




if __name__ == "__main__":
    # Ternary system example
    system_La_Al_Sn_O = [("La", 2, "La2O3", 325.81), ("Al", 2, "Al2O3", 101.96), ("Sn", 1, "SnO2", 150.71)]
    system = system_La_Al_Sn_O
    system_type = SystemType.TERNARY

    # First graph: Atomic percent, showing equidistant exploration points
    unit_type = UnitType.ATOMIC_PERCENT
    ax = ternary_plot(system, unit_type)
    generated_data_set = generate_equidistant_points(system_type, n=8)
    #print(f"{generated_data_set=}")
    plot_with_points(ax, generated_data_set, scatter_type=ScatterplotTypes.DEFINED)
    plt.show(block=False)
    plt.pause(1)

    # Second graph: Molar percent, showing how these equidistant exploration points points translate to
    # molar space considering the source chemicals
    unit_type = UnitType.MOLAR_PERCENT
    ax = ternary_plot(system, unit_type)
    mol_percent_data_set = mol_percent_data_set_calculator(generated_data_set, system, system_type)
    plot_error_bars(ax, system, mol_percent_data_set, total_mass=0.5, mass_uncertainty=0.025, unit_type=unit_type)

    #print(f"{mol_percent_data_set=}")
    plot_with_points(ax, mol_percent_data_set, scatter_type=ScatterplotTypes.DEFINED)

    print_weighing_table(mol_percent_data_set, system, system_type, total_weight=0.5)

    # Second graph modified: New points calculated from the experimental weighing results are added in red
    measured_weight = (np.array([0.40, 0.36, 0.32, 0.30, 0.27, 0.24, 0.23, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.09, 0.08]),
    np.array([0.02, 0.05, 0.02, 0.09, 0.05, 0.02, 0.14, 0.09, 0.05, 0.02, 0.22, 0.15, 0.09, 0.05, 0.02]),
    np.array([0.07, 0.08, 0.15, 0.09, 0.16, 0.22, 0.11, 0.19, 0.25, 0.30, 0.13, 0.22, 0.29, 0.34, 0.38]))

    experimental_mol_percent_data_list = weight_to_atomic_percent_dataset_calculator(system, system_type, measured_weight)
    plot_with_points(ax, experimental_mol_percent_data_list, scatter_type=ScatterplotTypes.MEASURED)


    plt.show()
    #plt.pause(1)


    # # Binary system example
    # system_La_Sn_O = [("La", 2, "La2O3", 325.81), ("Sn", 1, "SnO2", 150.71)]
    # system = system_La_Sn_O
    # unit_type = UnitType.ATOMIC_PERCENT
    # system_type = SystemType.BINARY
    # ax = binary_composition_diagram(system)





