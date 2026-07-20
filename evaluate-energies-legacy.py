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

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
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
    # Construct the full path to the CSV file
    file_path = os.path.join(folder_path, file_name)
    
    # Load the CSV file, only reading the first 5 columns
    data = pd.read_csv(file_path, usecols=["step", "energy", "ewmean", "ewvar", "pmove"])
    
    return data

def parse_args(argv):
    def parse_nspins(nspins_arg):
        nspins = tuple(int(n) for n in nspins_arg.split('_'))
        if len(nspins) != 2:
            raise ValueError("nspins must have format nup_ndown, for example 6_0")
        return nspins
    if len(argv) >= 2:
        potential_type = argv[1]

        if potential_type == "Coulomb":
            if len(argv) < 5:
                raise ValueError(
                    "Expected arguments for Coulomb: "
                    "Coulomb nspins r_s network_type"
                )

            return {
                "potential_type": potential_type,
                "nspins": parse_nspins(argv[2]),
                "r_s": float(argv[3]),
                "network_type": argv[4],
                "supercell_shape": "tri",
            }

        if potential_type == "CoulombMoire":
            if len(argv) < 10:
                raise ValueError(
                    "Expected arguments for CoulombMoire: "
                    "CoulombMoire nspins num_unit_cells me_eff_rel eps_inverse "
                    "moire_lattice_constant_nm moire_potential_strength_meV "
                    "moire_potential_phi network_type"
                )

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
    }

def get_folder_name(args):
    if args["potential_type"] == "Coulomb":
        return (
            f"results/2deg-Coulomb/{args['network_type']}/"
            f"el{args['nspins'][0]}_{args['nspins'][1]}_rs{args['r_s']}_{args['supercell_shape']}"
        )

    _, moire_potential_strength, interaction_energy_scale = convert_moire_scales(
        args["me_eff_rel"],
        args["eps_inverse"],
        args["moire_lattice_constant_nm"],
        args["moire_potential_strength_meV"],
    )
    return (
        f"results/2deg-CoulombMoire/{args['network_type']}/"
        f"el{args['nspins'][0]}_{args['nspins'][1]}_N{args['num_unit_cells']}_"
        f"V{np.round(moire_potential_strength,8)}_{args['moire_potential_phi']}_"
        f"U{np.round(interaction_energy_scale,8)}"
    )

ndim = 2 # spatial dimension of the system

# system parameters
print(f"Evaluating 2DEG energies with parameters: {argv}")
args = parse_args(argv)
potential_type = args["potential_type"]
network_type = args["network_type"]
nspins = args["nspins"]
num_electrons = sum(nspins)

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

# generate folder name
folder_name = get_folder_name(args)
train_data = load_csv_data(folder_name, "train_stats.csv")

fig, ax = plt.subplots(1,1, figsize = (4,3), dpi=300)
ax.plot(train_data['step'], train_data['energy'] * energy_scale / num_electrons, marker='o', linestyle='-', linewidth=0.4, markersize=1, alpha=0.4)
ax.set_xlabel("step")
ax.set_ylabel(energy_ylabel)
