# Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
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

"""
This code runs a calculation of a two-dimensional homogeneous electron gas with Coulomb interaction.

Optional positional arguments after NUP_NDOWN, R_S, and NETWORK_TYPE:
  FOLDER_EXTENSION
      Start an altered task in a suffixed result folder, using the default
      iteration target.
  FOLDER_EXTENSION TOTAL_ITERATIONS RESTORE_PATH
      Continue from RESTORE_PATH into the suffixed result folder, up to the
      specified total iteration target.
"""

import os
import sys
from sys import argv
import numpy as np

from absl import logging
from periodicwave import base_config
from periodicwave import checkpoint
from periodicwave import train

from periodicwave.utils import custom_logging
from periodicwave.utils import writers
from periodicwave.pbc import lattices

from typing import Tuple

from time import time

import jax
print("Jax Devices:", jax.devices())

# Always use float32 precision
print("Setting jax_default_matmul_precision to float32")
jax.config.update("jax_default_matmul_precision", "float32")
print(f"Matmul precision set to: {jax.default_matmul_precision}")

# Get parameters defining the physical system
print(f"Calculating 2DEG with parameters: {argv}\n=============================================")

def parse_run_arguments(command_line):
    usage = (
        f"Usage: {command_line[0]} [NUP_NDOWN R_S NETWORK_TYPE "
        "[FOLDER_EXTENSION [TOTAL_ITERATIONS RESTORE_PATH]]]"
    )

    if len(command_line) == 1:
        return [2, 0], 10, "CustomPsiformer", "", None, ""

    if len(command_line) not in (4, 5, 7):
        raise ValueError(
            usage
            + "\nA fresh altered run takes only FOLDER_EXTENSION. A "
            "continuation requires FOLDER_EXTENSION, TOTAL_ITERATIONS, and "
            "RESTORE_PATH."
        )

    # number of spin up, spin down electrons
    nspins = [int(n) for n in command_line[1].split('_')]
    # Ratio between interaction and kinetic energy scale
    r_s = float(command_line[2])
    network_type = command_line[3]
    folder_extension = command_line[4] if len(command_line) >= 5 else ""
    total_iterations = int(command_line[5]) if len(command_line) == 7 else None
    restore_path = command_line[6] if len(command_line) == 7 else ""

    if total_iterations is not None and total_iterations <= 0:
        raise ValueError("TOTAL_ITERATIONS must be positive.")

    return (
        nspins,
        r_s,
        network_type,
        folder_extension,
        total_iterations,
        restore_path,
    )


(
    nspins,
    r_s,
    network_type,
    folder_extension,
    total_iterations,
    restore_path,
) = parse_run_arguments(argv)

# By default, use square supercell for pbc
supercell_shape = 'tri'

# --------------------------- Set up config file ---------------------------
cfg = base_config.default()

# Set number of electrons
cfg.system.electrons = (nspins[0], nspins[1])  # (spin up, spin down) electrons

# Create the 2D lattice 
cfg.system.ndim = 2
if supercell_shape == "tri": # triangular supercell
    num_electrons = sum(nspins)
    supercell_a = np.sqrt(2 * np.pi / np.sqrt(3) * num_electrons)
    supercell_lattice, _ = lattices._triangular_lattice_vecs_periodic_potential(supercell_a, 1)
    area = np.linalg.det(supercell_lattice)
elif supercell_shape == "sq": # square supercell
    num_electrons = sum(nspins)
    supercell_a = np.sqrt(np.pi * num_electrons)
    supercell_lattice = lattices._square_lattice_vecs(supercell_a)
    area = np.linalg.det(supercell_lattice)
else:
    raise NotImplementedError("Only supercell shapes 'tri' are implemented. Received: " + supercell_shape)

# Set up the Hamiltonian parameters
cfg.system.make_local_energy_fn = "periodicwave.pbc.hamiltonian.local_energy"
cfg.system.make_local_energy_kwargs = {"lattice": supercell_lattice, 
    "potential_type": 'Coulomb',
    "potential_kwargs": {'interaction_energy_scale': r_s},
    "kinetic_kwargs": {'laplacian_method': 'folx'}, # use forward laplacian
    }
cfg.system.pbc_lattice = supercell_lattice

# feature layer for pbc
cfg.network.make_feature_layer_fn = ("periodicwave.pbc.feature_layer.make_pbc_feature_layer")
cfg.network.make_feature_layer_kwargs = { "lattice": supercell_lattice, "include_r_ae": False }

# envelopes
cfg.network.make_envelope_fn = ( "periodicwave.envelopes.make_null_envelope" ) # use no envelope with pbc

# Set batch size
cfg.batch_size = 1024

# learning rate parameters
cfg.optim.optimizer = 'kfac'
cfg.optim.objective = 'vmc'
cfg.optim.iterations = 10000
if total_iterations is not None:
    cfg.optim.iterations = total_iterations
cfg.optim.lr.rate  = 0.1
cfg.optim.lr.delay = 2000
cfg.optim.lr.decay = 1.0

# logging parameters
cfg.log.save_frequency = 2.0 # minutes

# mcmc parameters
cfg.mcmc.burn_in = 100
cfg.mcmc.init_width = 1.0
cfg.mcmc.move_width = 0.2
cfg.mcmc.steps = 10
cfg.mcmc.move_width_updater = 'adaptive'

# general network parameters
cfg.network.complex = True # complex wavefunction work better for pbc systems
cfg.network.determinants = 4
cfg.network.jastrow = "NONE"
cfg.network.jastrow_kwargs = {'ndim':cfg.system.ndim}

# network architecture parameters
# Code to run Psiformer
if network_type == 'CustomPsiformer':
    cfg.network.network_type = 'CustomPsiformer'
    cfg.network.CustomPsiformer.num_layers = 4
    cfg.network.CustomPsiformer.num_heads  = 4
    cfg.network.CustomPsiformer.attn_dim   = 16
    cfg.network.CustomPsiformer.value_dim  = 16
    cfg.network.CustomPsiformer.mlp_dim    = 64
    cfg.network.CustomPsiformer.num_perceptrons_per_layer = 2
    cfg.network.CustomPsiformer.use_layer_norm = True
    cfg.network.CustomPsiformer.mlp_activation_fct = "GELU"

# Code to run SlaterNet
elif network_type == 'SlaterNet':
    cfg.network.network_type = 'SlaterNet'
    cfg.network.SlaterNet.num_layers = 4
    cfg.network.SlaterNet.mlp_dim    = 64
    cfg.network.SlaterNet.num_perceptrons_per_layer = 2
    cfg.network.SlaterNet.use_layer_norm = True
    cfg.network.SlaterNet.mlp_activation_fct = "GELU"
    cfg.network.determinants = 1 # when using a single determinant, SlaterNet is equivalent to Hartree-Fock

# Get folder name to save the results. The extension is appended literally.
base_folder_name = f"PeriodicWave/results/2deg-Coulomb/{network_type}/el{nspins[0]}_{nspins[1]}_rs{r_s}_{supercell_shape}"
folder_name = base_folder_name + folder_extension

# Save new checkpoints in folder_name. For a branched continuation, restore
# from restore_path only when folder_name does not yet contain a checkpoint.
cfg.log.save_path = folder_name
cfg.log.restore_path = restore_path

if restore_path and os.path.abspath(restore_path) == os.path.abspath(folder_name):
    raise ValueError("RESTORE_PATH must differ from the destination folder.")

destination_checkpoint = checkpoint.find_last_checkpoint(folder_name)
source_checkpoint = (
    None
    if destination_checkpoint
    else checkpoint.find_last_checkpoint(restore_path)
)
selected_checkpoint = destination_checkpoint or source_checkpoint

if restore_path and selected_checkpoint is None:
    raise FileNotFoundError(
        "No checkpoint was found in either the destination folder "
        f"'{folder_name}' or RESTORE_PATH '{restore_path}'."
    )

if selected_checkpoint:
    with np.load(selected_checkpoint, allow_pickle=True) as checkpoint_data:
        next_iteration = int(checkpoint_data['t']) + 1
    if cfg.optim.iterations <= next_iteration:
        raise ValueError(
            f"TOTAL_ITERATIONS ({cfg.optim.iterations}) must be greater than "
            f"the checkpoint's next iteration ({next_iteration})."
        )
    checkpoint_role = "destination" if destination_checkpoint else "source"
    print(
        f"Continuing from {checkpoint_role} checkpoint: {selected_checkpoint}\n"
        f"Training iterations: {next_iteration} to {cfg.optim.iterations - 1}"
    )
elif folder_extension:
    print(f"Starting fresh altered run in: {folder_name}")

# log device info. 
# If same folder contains previous runs, rename old log file to keep information.
# The renaming follows the same convention as renaming train_stats.csv
writers.rename_file("device_info", folder_name, file_extension="log")
custom_logging.log_device_info(folder_name + "/device_info.log")
writers.rename_file("config", folder_name, file_extension="json")
custom_logging.save_config_dict_as_json(cfg, cfg.log.save_path + '/config.json')

t_init = time()
# --------------------------- train ---------------------------
train.train(cfg)
logging.info("Training completed after t [s] = " + str(int(time() - t_init)))
print("Training completed after t [s] = " + str(int(time() - t_init)) + "\n", flush=True)
