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

import numpy as np
from matplotlib import pyplot as plt
import os
from sys import argv
from periodicwave.pbc import lattices
from periodicwave.utils import observables

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
                "folder_name_extension": argv[5] if len(argv) >= 6 else "",
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
                "folder_name_extension": argv[10] if len(argv) >= 11 else "",
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

def get_lattice(args):
    if args["potential_type"] == "Coulomb":
        num_electrons = sum(args["nspins"])
        if args["supercell_shape"] == "tri":
            supercell_a = np.sqrt(2 * np.pi / np.sqrt(3) * num_electrons)
            lat_vec, _ = lattices._triangular_lattice_vecs_periodic_potential(supercell_a, 1)
            return lat_vec
        if args["supercell_shape"] == "sq":
            supercell_a = np.sqrt(np.pi * num_electrons)
            return lattices._square_lattice_vecs(supercell_a)
        raise NotImplementedError(
            "Only supercell shapes 'tri' and 'sq' are implemented. "
            f"Received: {args['supercell_shape']}"
        )

    lat_vec, _, _ = lattices._triangular_lattice_vecs_periodic_potential(
        a=1.0, num_sites=args["num_unit_cells"], return_lattice_M=True
    )
    return lat_vec

def get_positions_from_latest_npz_files(folder_path, N):
    """
    Loads position data from N latest checkpoints of FermiNet ouput stored in
    a folder with path folder_path
    """
    # Find all files that match the "qmcjax_ckpt_XXXXXX.npz" pattern
    files = [
        f for f in os.listdir(folder_path) 
        if f.startswith("qmcjax_ckpt_") and f.endswith(".npz")
    ]
    
    # Extract the six-digit number from each filename and store it in a tuple (number, filename)
    file_numbers = []
    for file in files:
        try:
            number = int(file.split('_')[-1].split('.')[0])
            file_numbers.append((number, file))
        except ValueError:
            print(f"Skipping file {file}, could not extract valid six-digit number.")

    # Sort the files by the six-digit number in descending order
    sorted_files = sorted(file_numbers, key=lambda x: x[0], reverse=True)

    # Select the top N files with the largest numbers
    largest_files = sorted_files[:N]

    print("In folder " + folder_path + "/ loading checkpoints: ")
    print([el[0] for el in largest_files])

    # Load the contents of the selected files
    positions_ckpt = []
    spins_ckpt = []
    for number, filename in largest_files:
        file_path = os.path.join(folder_path, filename)
        try:
            ckpt_data = np.load(file_path, allow_pickle=True)
            data = ckpt_data['data'].item()
            positions_ckpt.append(data['positions'])
            spins_ckpt.append(data['spins'])
            # loaded_data[filename] = data
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    filenames = [el[1] for el in largest_files]

    return positions_ckpt, spins_ckpt, filenames

# system parameters
print(f"Evaluating 2DEG density with parameters: {argv}")
args = parse_args(argv)
potential_type = args["potential_type"]
ndim = 2

# generate folder name
folder_name = get_folder_name(args, args["folder_name_extension"])

lat_vec = get_lattice(args)

load_N_ckpts = 3 # number of latest checkpoints to load 
observable_bins = 80
structure_factor_max_index = 6

# load configurations from latest checkpoints
positions_ckpt, _, _ = get_positions_from_latest_npz_files(folder_name, load_N_ckpts)
positions_batch = observables.walker_positions_to_samples(positions_ckpt, ndim=ndim)

# Previous versions scattered raw MCMC coordinates and pair displacements.
# Use normalized Monte Carlo estimators for the observables instead.
density_obs = observables.estimate_density(
    positions_batch, lat_vec, bins=observable_bins
)
pair_obs = observables.estimate_pair_correlation(
    positions_batch, lat_vec, bins=observable_bins
)
k_indices = observables.reciprocal_index_grid(structure_factor_max_index)
structure_obs = observables.estimate_structure_factor(
    positions_batch, lat_vec, k_indices
)

position_xlabel = "x / a_M" if potential_type == "CoulombMoire" else "x"
position_ylabel = "y / a_M" if potential_type == "CoulombMoire" else "y"

# plot electron density
fig_n, ax_n = plt.subplots(1, 1, figsize = (7, 5))
im_n = ax_n.pcolormesh(
    density_obs["mesh_x"],
    density_obs["mesh_y"],
    density_obs["density"],
    shading="auto",
)
fig_n.colorbar(im_n, ax=ax_n, label=r"$\rho(\mathbf{r})$")
ax_n.set_aspect("equal", adjustable="box")
ax_n.set_xlabel(position_xlabel)
ax_n.set_ylabel(position_ylabel)
ax_n.set_title("Electron density")

# plot pair correlation
fig_nn, ax_nn = plt.subplots(1, 1, figsize = (7, 5))
im_nn = ax_nn.pcolormesh(
    pair_obs["mesh_x"],
    pair_obs["mesh_y"],
    pair_obs["pair_correlation"],
    shading="auto",
)
fig_nn.colorbar(im_nn, ax=ax_nn, label=r"$g(\mathbf{r})$")
ax_nn.set_aspect("equal", adjustable="box")
ax_nn.set_xlabel(position_xlabel)
ax_nn.set_ylabel(position_ylabel)
ax_nn.set_title("Pair correlation")

# plot static structure factor
fig_s, ax_s = plt.subplots(1, 1, figsize = (7, 5))
k_vectors = structure_obs["k_vectors"]
im_s = ax_s.scatter(
    k_vectors[:, 0],
    k_vectors[:, 1],
    c=structure_obs["structure_factor"],
    s=80,
)
fig_s.colorbar(im_s, ax=ax_s, label=r"$S(\mathbf{k})$")
ax_s.set_aspect("equal", adjustable="box")
ax_s.set_xlabel(r"$k_x$")
ax_s.set_ylabel(r"$k_y$")
ax_s.set_title("Static structure factor")
