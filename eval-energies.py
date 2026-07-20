# Copyright (c) 2025 Max Geier, Khachatur Nazaryan, Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import argv

def convert_moire_scales(me_eff_rel = 0.35, eps_inverse = 0.2, moire_a = 8.031, moire_potential_strength = 15):
    """ 
    Converts moire system parameters from SI to natural units used in the code.

    Default values as used in M Geier, K Nazaryan, T Zaklama, and L Fu, Phys. Rev. B 112, 045119

    Arguments:
        me_eff_rel = 0.35             # effective mass in units of bare electron mass
        moire_a = 8.031               # moire lattice constant, nm
        eps_inverse = 0.1             # inverse relative diel. constant of the surrounding dielectric
        moire_potential_strength = 15 # energy scale V of the hexagonal moire potential, meV 

    Returns:
        energy_scale: Conversion factor between energies returned from the code and SI units (meV)
        V: moire potential strength in natural units
        U: interaction energy scale in natural units
    """
    # Natural constants
    a_0  = 5.29177210544e-2      # Bohr radius, nm
    hbar = 6.582119569509066e-1  # meV * ps
    me   = 5.685630060215049e-3  # electron rest mass, meV/[(nm/ps)^2]

    # effective mass in SI units (meV/[(nm/ps)^2])
    me_eff = me_eff_rel * me      # effective mass, meV/[(nm/ps)^2]

    # scales converting dimensionless units from the code to SI units
    # Here we present units where distances are measured in terms of the moire lattice constant moire_a
    # and energies are measured in \hbar^2 / moire_a^2 / me_eff,
    energy_scale = hbar**2 / moire_a**2 / me_eff # meV
    # In these units, the kinetic energy term is
    # - 0.5 \sum_j \nabla_j^2 where j runs over all electrons.
    # In these units, the dimensionless moire potential strength is V, determining
    # - 2 V \sum_i \sum_{n = 1}^{3} \cos ( g_n \cdot r_i + \phi )
    V = moire_potential_strength / energy_scale 
    # The dimensionless Coulomb interaction energy scale is U, determining
    # 0.5 U \sum_{i \neq j} 1/{|r_j - r_i|}
    U = (moire_a / a_0) * eps_inverse * (me_eff / me)

    return energy_scale, V, U

def load_csv_data(folder_path, file_name):
    return pd.read_csv(
        Path(folder_path) / file_name,
        usecols=["step", "energy", "ewmean", "locstd"],
    )


def load_training_series(folder_paths, args):
    """Validate and load chronological training stages."""
    histories = []
    expected_architecture = None
    for stage_index, folder_path in enumerate(folder_paths):
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Training folder does not exist: {folder}")

        config_path = folder / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing result metadata: {config_path}")
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)

        system = config["system"]
        network = config["network"]
        network_type = network["network_type"]
        if (
            tuple(system["electrons"]) != args["nspins"]
            or system["make_local_energy_kwargs"]["potential_type"]
            != args["potential_type"]
            or network_type != args["network_type"]
        ):
            raise ValueError(
                "Series folders must match the requested potential type, spin "
                f"sector, and network type; mismatch found in {config_path}."
            )

        architecture = {
            "network_config": network[network_type],
            "determinants": network["determinants"],
            "complex": network["complex"],
            "bias_orbitals": network.get("bias_orbitals"),
            "jastrow": network["jastrow"],
            "feature_layer": network.get("make_feature_layer_fn"),
            "envelope": network.get("make_envelope_fn"),
        }
        if expected_architecture is None:
            expected_architecture = architecture
        elif architecture != expected_architecture:
            raise ValueError(
                "All series folders must use the same network architecture; "
                f"mismatch found in {config_path}."
            )

        archived_files = []
        for file_path in folder.glob("train_stats_*.csv"):
            match = re.fullmatch(r"train_stats_(\d+)\.csv", file_path.name)
            if match:
                archived_files.append((int(match.group(1)), file_path))
        log_files = [
            file_path
            for _, file_path in sorted(archived_files, key=lambda item: item[0])
        ]
        if (folder / "train_stats.csv").is_file():
            log_files.append(folder / "train_stats.csv")
        if not log_files:
            raise FileNotFoundError(f"No train_stats*.csv files found in: {folder}")

        stage_data = pd.concat(
            [load_csv_data(folder, file_path.name) for file_path in log_files],
            ignore_index=True,
        )
        if stage_data.empty:
            raise ValueError(f"Training logs are empty in: {folder}")
        stage_data[["step", "energy", "ewmean", "locstd"]] = stage_data[
            ["step", "energy", "ewmean", "locstd"]
        ].apply(pd.to_numeric, errors="raise")
        stage_data = stage_data.drop_duplicates("step", keep="last")
        stage_data["stage_index"] = stage_index
        stage_data["folder_name"] = folder.name
        histories.append(stage_data)

    training_series = pd.concat(histories, ignore_index=True)
    return (
        training_series.drop_duplicates("step", keep="last")
        .sort_values("step", kind="stable")
        .reset_index(drop=True)
    )


def convert_energy_columns(train_data, energy_scale, num_electrons):
    unit_conversion = energy_scale / num_electrons

    return {
        "energy": train_data["energy"] * unit_conversion,
        "ewmean": train_data["ewmean"] * unit_conversion,
        "locstd": train_data["locstd"] * unit_conversion,
    }

def parse_args(argv):
    def parse_nspins(nspins_arg):
        nspins = tuple(int(n) for n in nspins_arg.split('_'))
        if len(nspins) != 2:
            raise ValueError("nspins must have format nup_ndown, for example 6_0")
        return nspins

    def parse_folder_inputs(folder_inputs):
        if not folder_inputs:
            return "", []
        if len(folder_inputs) == 1 and (
            not folder_inputs[0] or folder_inputs[0].startswith("_")
        ):
            return folder_inputs[0], []
        if any(
            not folder_name or folder_name.startswith("_")
            for folder_name in folder_inputs
        ):
            raise ValueError(
                "A folder series requires full folder names as separate arguments; "
                "do not mix names with empty strings or suffixes."
            )
        if any(set(folder_name) & set("{},") for folder_name in folder_inputs):
            raise ValueError(
                "Pass folder names as separate Bash arguments without braces or commas."
            )
        return "", folder_inputs

    if len(argv) >= 2:
        potential_type = argv[1]

        if potential_type == "Coulomb":
            if len(argv) < 5:
                raise ValueError(
                    "Expected arguments for Coulomb: "
                    "Coulomb nspins r_s network_type"
                )

            folder_name_extension, folder_names = parse_folder_inputs(argv[5:])
            return {
                "potential_type": potential_type,
                "nspins": parse_nspins(argv[2]),
                "r_s": float(argv[3]),
                "network_type": argv[4],
                "supercell_shape": "tri",
                "folder_name_extension": folder_name_extension,
                "folder_names": folder_names,
            }

        if potential_type == "CoulombMoire":
            if len(argv) < 10:
                raise ValueError(
                    "Expected arguments for CoulombMoire: "
                    "CoulombMoire nspins num_unit_cells me_eff_rel eps_inverse "
                    "moire_lattice_constant_nm moire_potential_strength_meV "
                    "moire_potential_phi network_type"
                )

            folder_name_extension, folder_names = parse_folder_inputs(argv[10:])
            return {
                "potential_type": potential_type,
                "nspins": parse_nspins(argv[2]),
                "num_unit_cells": int(argv[3]),
                "me_eff_rel": float(argv[4]),
                "eps_inverse": float(argv[5]),
                "moire_lattice_constant_nm": float(argv[6]),
                "moire_potential_strength_meV": float(argv[7]),
                "moire_potential_phi": float(argv[8]),
                "network_type": argv[9],
                "folder_name_extension": folder_name_extension,
                "folder_names": folder_names,
            }

        raise ValueError(
            'potential_type must be "Coulomb" or "CoulombMoire"; '
            f"received {potential_type}"
        )

    return {
        "potential_type": "CoulombMoire",
        "nspins": (6, 0),
        "num_unit_cells": 9,
        "me_eff_rel": 0.35,
        "eps_inverse": 0.2,
        "moire_lattice_constant_nm": 8.031,
        "moire_potential_strength_meV": 15,
        "moire_potential_phi": 45,
        "network_type": "CustomPsiformer",
        "folder_name_extension": "",
        "folder_names": [],
    }

def get_folder_name(args, extension=""):
    if args["potential_type"] == "Coulomb":
        folder_name = (
            f"results/2deg-Coulomb/{args['network_type']}/"
            f"el{args['nspins'][0]}_{args['nspins'][1]}_rs{args['r_s']}_{args['supercell_shape']}"
        )
        return folder_name + extension

    _, moire_potential_strength, interaction_energy_scale = convert_moire_scales(
        args["me_eff_rel"],
        args["eps_inverse"],
        args["moire_lattice_constant_nm"],
        args["moire_potential_strength_meV"],
    )
    folder_name = (
        f"results/2deg-CoulombMoire/{args['network_type']}/"
        f"el{args['nspins'][0]}_{args['nspins'][1]}_N{args['num_unit_cells']}_"
        f"V{np.round(moire_potential_strength,8)}_{args['moire_potential_phi']}_"
        f"U{np.round(interaction_energy_scale,8)}"
    )
    return folder_name + extension


def plot_energy_history(train_data, energy_scale, num_electrons, energy_ylabel):
    energy_data = convert_energy_columns(train_data, energy_scale, num_electrons)
    if "stage_index" in train_data.columns:
        final_stage = train_data["stage_index"].max()
        final_row_index = train_data.index[
            train_data["stage_index"] == final_stage
        ][-1]
    else:
        final_row_index = train_data.index[-1]
    final_ewmean = float(energy_data["ewmean"].loc[final_row_index])
    final_locstd = float(energy_data["locstd"].loc[final_row_index])

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=300)
    ax.axhspan(
        final_ewmean - final_locstd,
        final_ewmean + final_locstd,
        color="tab:orange",
        alpha=0.16,
        linewidth=0,
        label=f"final EW mean +/- final local-energy std ({final_locstd:.3g})",
        zorder=0,
    )
    ax.axhline(
        final_ewmean,
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label=f"final EW mean ({final_ewmean:.6g})",
        zorder=1,
    )

    if "stage_index" in train_data.columns:
        for stage_index, stage_data in train_data.groupby(
            "stage_index", sort=True
        ):
            stage_label = stage_data["folder_name"].iloc[0]
            line, = ax.plot(
                stage_data["step"],
                energy_data["energy"].loc[stage_data.index],
                marker="o",
                linestyle="-",
                linewidth=0.4,
                markersize=1,
                alpha=0.4,
                label=stage_label,
                zorder=2,
            )
            if stage_index > 0:
                ax.axvline(
                    stage_data["step"].min(),
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.8,
                    zorder=1,
                )
    else:
        ax.plot(
            train_data["step"],
            energy_data["energy"],
            marker="o",
            linestyle="-",
            linewidth=0.4,
            markersize=1,
            alpha=0.4,
            label="batch mean energy",
            zorder=2,
        )

    ax.set_xlabel("step")
    ax.set_ylabel(energy_ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def main(command_line):
    print(f"Evaluating 2DEG energies with parameters: {command_line}")
    args = parse_args(command_line)
    potential_type = args["potential_type"]
    nspins = args["nspins"]

    if args["folder_names"]:
        result_parent = Path(get_folder_name(args)).parent
        folder_paths = []
        for folder_name in args["folder_names"]:
            folder_path = Path(folder_name)
            if not folder_path.is_absolute() and folder_path.parent == Path("."):
                folder_path = result_parent / folder_path
            folder_paths.append(folder_path)

        train_data = load_training_series(folder_paths, args)
        return plot_energy_history(
            train_data,
            energy_scale=1.0,
            num_electrons=sum(nspins),
            energy_ylabel="energy / electron (code units)",
        )

    if potential_type == "CoulombMoire":
        energy_scale, _, _ = convert_moire_scales(
            args["me_eff_rel"],
            args["eps_inverse"],
            args["moire_lattice_constant_nm"],
            args["moire_potential_strength_meV"],
        )
        energy_ylabel = "energy / electron (meV)"
    else:
        energy_scale = 1.0
        energy_ylabel = "energy / electron"

    folder_name = get_folder_name(args, args["folder_name_extension"])
    train_data = load_csv_data(folder_name, "train_stats.csv")
    return plot_energy_history(
        train_data,
        energy_scale=energy_scale,
        num_electrons=sum(nspins),
        energy_ylabel=energy_ylabel,
    )


if __name__ == "__main__":
    main(argv)
