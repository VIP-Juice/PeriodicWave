# PeriodicWave Architecture and Usage Notes

This note explains the code architecture of `PeriodicWave` and how the code implements the neural-network variational Monte Carlo (NN-VMC) method described in Geier, Nazaryan, Zaklama, and Fu, Phys. Rev. B 112, 045119 (2025), "Self-attention neural network for solving correlated electron problems in solids."

The short version: this repository is a JAX implementation of continuous-space VMC for two-dimensional periodic electron systems. It builds an antisymmetric many-electron wavefunction from neural-network-generated orbitals, samples electron configurations from $|\Psi_\theta(R)|^2$, evaluates the local energy $E_{\mathrm{loc}}(R)=\Psi_\theta(R)^{-1}H\Psi_\theta(R)$, and updates the network parameters to lower the sampled energy.

## 1. What Problem the Code Solves

The target Hamiltonian is a continuum many-electron Hamiltonian in a finite periodic supercell,

$$
  H
= -\frac{1}{2}\sum_i \nabla_i^2
+ \sum_i V(\mathbf r_i)
+ \frac{1}{2}\sum_{i\ne j}\frac{U}{|\mathbf r_i-\mathbf r_j|_{\mathrm{PBC}}}.
$$

The code currently focuses on:

- Homogeneous two-dimensional electron gas with periodic Coulomb interaction.
- Two-dimensional electron gas in a triangular moire potential plus Coulomb interaction.
- Spin-polarized or fixed-spin sectors, specified by $(N_\uparrow,N_\downarrow)$.
- Complex wavefunctions, which are usually preferred for periodic boundary conditions.

The finite system is interpreted as one supercell periodically repeated in the plane. The wavefunction must be invariant when any electron coordinate is shifted by a supercell lattice vector. The Coulomb interaction is therefore not the bare pairwise finite-cell $1/r$; it is an Ewald-style periodic Coulomb sum with a neutralizing background and Madelung self-image contribution.

## 2. Repository Map

The main files are:

```text
periodicwave/
  base_config.py                         Default ConfigDict for VMC runs
  train.py                               Top-level training loop
  networks.py                            Shared network interfaces and orbital assembly
  CustomPsiformer.py                     Self-attention correlated-orbital ansatz
  SlaterNet.py                           Feed-forward Slater determinant ansatz
  hamiltonian.py                         Generic kinetic-energy helpers
  loss.py                                VMC loss and custom gradient/JVP
  mcmc.py                                Metropolis-Hastings sampling
  checkpoint.py                          NumPy checkpoint save/restore
  jastrows.py                            Optional electron-electron Jastrow factors
  network_blocks.py                      Linear layers, determinant utilities
  pbc/
    feature_layer.py                     Periodic sin/cos features and periodic norm
    hamiltonian.py                       2D Ewald Coulomb, triangular potential, local energy
    lattices.py                          Square and triangular supercell builders
  configs/
    run_2deg.py                          Homogeneous 2D electron gas example
    run_2deg_triangular_potential.py     Moire-potential example
  utils/
    writers.py                           CSV logging
    statistics.py                        Exponentially weighted energy statistics
    custom_logging.py                    Device/config logging helpers

evaluate-energies.py                     Plot logged training energy
evaluate-density.py                      Plot density and density-density correlator from checkpoints
README.md                               Installation and high-level usage
```

The code is derived from DeepMind's FermiNet implementation, but it has been reduced to the pieces needed for two-dimensional periodic materials problems and extended with `CustomPsiformer`, `SlaterNet`, periodic features, 2D Coulomb handling, moire potentials, and optional 2D Coulomb Jastrows.

## 3. Method in One Pass

The VMC loop is:

1. Choose a variational wavefunction $\Psi_\theta(R)$ represented by a neural network.
2. Initialize many walkers, where each walker is a flattened electron-position vector $R=(\mathbf r_1,\ldots,\mathbf r_N)$.
3. Use Metropolis-Hastings moves to sample walkers from $|\Psi_\theta(R)|^2$.
4. Evaluate the local energy at each walker:

   $$
   E_{\mathrm{loc}}(R)=\Psi_\theta(R)^{-1}H\Psi_\theta(R).
   $$

5. Average the local energies over the batch to estimate the variational energy.
6. Differentiate the VMC objective with respect to network parameters.
7. Update parameters with KFAC by default.
8. Save `train_stats.csv` and periodic `qmcjax_ckpt_*.npz` checkpoints.

This is implemented primarily by `train.train(cfg)`.

## 4. Data Model

The object passed through the sampler, network, and Hamiltonian is `networks.FermiNetData`:

```python
FermiNetData(
    positions,  # shape (..., nelectrons * ndim), flattened electron positions
    spins,      # shape (..., nelectrons), +1 for spin-up and -1 for spin-down
    atoms,      # compatibility placeholder, shape (..., natoms, ndim)
    charges,    # compatibility placeholder, shape (..., natoms)
)
```

For the current periodic electron-gas workflows, `atoms` and `charges` are mostly dummy compatibility data. The physical lattice, Coulomb strength, moire potential, and dimensionality are supplied through `cfg.system`.

## 5. Boundary Conditions and Input Features

Periodic boundary conditions enter in two places.

First, the feature layer maps real coordinates to periodic features. In `periodicwave/pbc/feature_layer.py`, electron positions and pair displacements are converted to fractional coordinates of the supercell and then represented by

$$
\sin(2\pi \mathbf s),\qquad \cos(2\pi \mathbf s),
$$

where $\mathbf s$ is the coordinate in reciprocal-lattice phase units. This ensures that positions differing by a supercell vector have the same feature representation.

Second, pair distances used by optional Jastrows and periodic feature distances use a smooth periodic norm. This norm agrees with the Euclidean norm for short separations but remains smooth and periodic across the cell boundary.

In the run configs, periodic features are selected by:

```python
cfg.network.make_feature_layer_fn = "periodicwave.pbc.feature_layer.make_pbc_feature_layer"
cfg.network.make_feature_layer_kwargs = {
    "lattice": supercell_lattice,
    "include_r_ae": False,
}
cfg.network.make_envelope_fn = "periodicwave.envelopes.make_null_envelope"
```

The null envelope is deliberate: for a periodic solid the orbitals should be periodic, not forced to decay at infinity.

## 6. Wavefunction Architectures

Both network architectures share the same final antisymmetrization step:

1. Build one vector stream per electron.
2. Project each stream to a set of complex orbitals.
3. Reshape the orbital values into one or more square matrices.
4. Return the log-domain sum of determinants using `network_blocks.logdet_matmul`.

The determinant is what enforces fermionic antisymmetry. The preceding neural network must be permutation equivariant: permuting electron labels should permute electron streams in the same way.

### 6.1 SlaterNet

`periodicwave/SlaterNet.py` implements the baseline neural Hartree-Fock ansatz.

Pipeline:

1. Periodic features of $\mathbf r_i$.
2. Concatenate the spin feature.
3. Linearly embed to `mlp_dim`.
4. Apply residual MLP blocks.
5. Project linearly to orbitals.
6. Evaluate the determinant.

Each electron is processed by the same feed-forward network. There is no attention mixing between electron streams. With one determinant and no Jastrow,

```python
cfg.network.network_type = "SlaterNet"
cfg.network.determinants = 1
cfg.network.jastrow = "NONE"
```

the ansatz is equivalent to an unrestricted Hartree-Fock variational calculation: it optimizes the best single Slater determinant of neural-network orbitals.

Important options:

- `num_layers`: number of residual MLP blocks.
- `mlp_dim`: stream dimension.
- `num_perceptrons_per_layer`: number of perceptrons inside each residual block.
- `use_layer_norm`: apply layer norm after each residual block.
- `mlp_activation_fct`: one of `TANH`, `ELU`, `GELU`.

### 6.2 CustomPsiformer

`periodicwave/CustomPsiformer.py` implements the self-attention ansatz used for correlated electron problems.

Pipeline:

1. Periodic features of $\mathbf r_i$.
2. Concatenate the spin feature.
3. Linearly embed to `mlp_dim`.
4. Apply repeated self-attention and MLP residual blocks.
5. Project linearly to complex orbitals.
6. Evaluate the sum of determinants.

The key difference from SlaterNet is the attention block. For each layer and attention head, each electron stream is projected to query, key, and value vectors. Dot-product attention lets each electron stream receive information from all other electron streams. The resulting orbitals have the form

$$
\phi_j(\mathbf r_i;\{\mathbf r_k:k\ne i\}),
$$

so the orbital evaluated at electron $i$ depends on the full electron configuration. These are "correlated orbitals" or generalized orbitals. The determinant still makes the total wavefunction antisymmetric, but the orbitals can encode correlations beyond a single-particle Hartree-Fock picture.

Important options:

- `num_layers`: number of attention layers.
- `num_heads`: attention heads per layer.
- `attn_dim`: query/key dimension per head.
- `value_dim`: value dimension per head.
- `mlp_dim`: stream dimension and MLP hidden dimension.
- `num_perceptrons_per_layer`: number of MLP perceptrons after attention.
- `use_layer_norm`: layer norm after attention and MLP residual updates.
- `mlp_activation_fct`: one of `TANH`, `ELU`, `GELU`.
- `determinants`: number of generalized Slater determinants in the final sum.

The paper finds that a few determinants and sufficiently many attention heads/layers are important. Increasing parameter count without enough attention structure may not improve the energy.

## 7. Orbital Assembly and Determinants

The shared determinant code is in `networks.make_orbitals`.

Notable implementation choices:

- Spin channels are merged into one determinant block using `nspins_merged = (n_total, 0)`.
- The orbital projection produces $N_{\mathrm{elec}}\times N_{\mathrm{det}}\times N_{\mathrm{states}}$ orbitals.
- If `complex_output=True`, the projection produces real and imaginary parts in alternating output channels.
- The final orbital tensor is reshaped to $(N_{\mathrm{det}}N_{\mathrm{states}},N_{\mathrm{elec}},N_{\mathrm{elec}})$.
- `network_blocks.logdet_matmul` evaluates determinants in the log domain for stability.
- Optional Jastrow factors multiply the orbitals before determinant evaluation.

The network apply function returns:

```python
phase_or_sign, log_abs_psi = network.apply(params, positions, spins, atoms, charges)
```

For complex wavefunctions, the first return value is the phase angle and the second is $\log|\Psi|$.

## 8. Hamiltonian and Local Energy

The local energy is built from:

```python
cfg.system.make_local_energy_fn = "periodicwave.pbc.hamiltonian.local_energy"
```

### 8.1 Kinetic Energy

The kinetic term is

$$
-\frac{1}{2}\sum_i \frac{\nabla_i^2\Psi(R)}{\Psi(R)}.
$$

The generic implementation in `periodicwave/hamiltonian.py` can compute the Laplacian by JAX linearization of gradients or by `folx.forward_laplacian`.

The periodic configs use:

```python
"kinetic_kwargs": {"laplacian_method": "folx"}
```

For complex wavefunctions, the code combines derivatives of the log magnitude and phase to return the real kinetic contribution.

### 8.2 Periodic Coulomb Potential

`periodicwave/pbc/hamiltonian.py` implements a two-dimensional Ewald sum:

- Real-space short-range part.
- Reciprocal-space long-range part.
- Madelung self-image term.
- Neutralizing-background subtraction.
- Optional interaction-energy prefactor `interaction_energy_scale`.

This is selected by:

```python
"potential_type": "Coulomb"
```

or as the Coulomb part of:

```python
"potential_type": "CoulombMoire"
```

The pair displacements are reduced into the first periodic cell before the Ewald potential is evaluated.

### 8.3 Triangular Moire Potential

For the moire calculation, `make_triangular_potential` adds

$$
V(\mathbf r)
= -2V_0\sum_{n=1}^3 \cos(\mathbf G_n\cdot\mathbf r+\phi).
$$

The config converts physical parameters such as effective mass, dielectric constant, moire lattice constant, and potential strength in meV into the dimensionless units used by the code. The conversion helper is `convert_moire_scales` in both the run script and analysis scripts.

## 9. Loss Function and Optimization

`periodicwave/loss.py` builds the VMC energy estimator:

```python
evaluate_loss = make_loss(log_network, local_energy, ...)
```

The loss function:

- vmaps local-energy evaluation over walkers.
- averages energy across local devices with `constants.pmean`.
- estimates the local-energy variance.
- clips local-energy outliers for the gradient if `cfg.optim.clip_local_energy > 0`.
- uses a custom JVP to implement the VMC energy gradient.
- registers the predictive distribution used by KFAC.

The default optimizer in the example configs is KFAC:

```python
cfg.optim.optimizer = "kfac"
```

Other accepted options are `adam`, `lamb`, and `none`. Use `none` for inference or measurement from a restored checkpoint without parameter updates.

The learning-rate schedule is:

$$
\mathrm{lr}(t)
= \mathrm{rate}\left(\frac{1}{1+t/\mathrm{delay}}\right)^{\mathrm{decay}}.
$$

## 10. MCMC Sampling

`periodicwave/mcmc.py` implements Metropolis-Hastings sampling.

The default move is an all-electron Gaussian proposal:

$$
R' = R + \mathrm{move\_width}\,\mathcal N(0,1).
$$

The accept/reject ratio uses:

$$
\log(\mathrm{ratio})
= 2\log|\Psi(R')| - 2\log|\Psi(R)|.
$$

The training loop adapts `move_width` every `cfg.mcmc.adapt_frequency` steps:

- If acceptance $p_{\mathrm{move}}>0.55$, increase width.
- If acceptance $p_{\mathrm{move}}<0.50$, decrease width.

Spin flips are not implemented. The spin configuration is fixed by `cfg.system.electrons`.

## 11. Training Flow in `train.train`

The high-level call graph is:

```text
config script
  -> base_config.default()
  -> set system, lattice, Hamiltonian, feature layer, network, optimizer, MCMC
  -> train.train(cfg)
       -> create dummy atoms/charges
       -> build feature layer
       -> build envelope
       -> build CustomPsiformer or SlaterNet
       -> initialize or restore params and walkers
       -> build MCMC step
       -> build local energy
       -> build VMC loss
       -> build optimizer
       -> burn in walkers
       -> repeat:
            MCMC updates
            local-energy loss
            optimizer update
            log train_stats.csv
            save qmcjax_ckpt_*.npz
```

The batch size must be divisible by the number of available JAX devices.

## 12. How to Run Calculations

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

The README gives pinned dependency sets for CPU and GPU runs. The package depends on JAX, `kfac-jax`, `folx`, `optax`, `ml-collections`, and standard scientific Python packages.

### 12.1 Homogeneous 2D Electron Gas

The example script is:

```bash
python periodicwave/configs/run_2deg.py
```

It sets up a 2D Coulomb gas with periodic boundary conditions. The save path has the form:

```text
results/2deg-Coulomb/{network_type}/el{n_up}_{n_down}_rs{r_s}_{supercell_shape}
```

As of this checkout, `run_2deg.py` contains a hard-coded line:

```python
network_type = "SlaterNet"
```

immediately before the architecture branch. This means the script will choose `SlaterNet` unless that line is changed, even though the earlier default says `CustomPsiformer`. Check this before using the script for attention-network production runs.

The command-line branch in the same file should also be checked before relying on arguments: it parses `argv[1]` both as `nspins` and as `r_s`. The no-argument default path is therefore the clearer starting point unless the argument parsing is corrected.

### 12.2 Moire Potential Calculation

The example script is:

```bash
python periodicwave/configs/run_2deg_triangular_potential.py
```

With explicit arguments, the ordering is:

```bash
python periodicwave/configs/run_2deg_triangular_potential.py \
  6_0 \
  9 \
  0.35 \
  0.2 \
  8.031 \
  15 \
  45 \
  CustomPsiformer
```

Argument meanings:

```text
6_0             n_up_n_down
9               number of moire unit cells in the supercell
0.35            effective mass in units of bare electron mass
0.2             inverse dielectric constant
8.031           moire lattice constant in nm
15              moire potential strength in meV
45              moire potential phase in degrees
CustomPsiformer network type, or SlaterNet
```

The save path has the form:

```text
results/2deg-CoulombMoire/{network_type}/el{n_up}_{n_down}_N{num_unit_cells}_V{V}_{phi}_U{U}
```

where $V$ and $U$ are dimensionless converted parameters.

## 13. Outputs

A training run writes:

```text
config.json
device_info.log
train_stats.csv
qmcjax_ckpt_000000.npz
qmcjax_ckpt_000001.npz
...
```

`train_stats.csv` columns are:

```text
step       training iteration
energy     current mean local energy
ewmean     exponentially weighted mean energy
ewvar      exponentially weighted energy variance
pmove      Metropolis-Hastings acceptance probability
locstd     standard deviation of local energy in the current batch
```

Checkpoints contain:

- iteration index `t`
- current walker data
- network parameters
- optimizer state
- current MCMC move width

`checkpoint.restore` validates that the checkpoint device count and requested batch size match the current run.

## 14. Postprocessing

`evaluate-energies.py` reads `train_stats.csv`, converts dimensionless energies to meV with the same moire unit conversion, and plots energy per electron.

`evaluate-density.py` loads the latest checkpoint files, extracts sampled electron positions, folds positions back into the first supercell, and plots:

- electron density from sampled positions
- pair/density-density relative-position scatter

Both analysis scripts are currently parameterized by literal variables near the bottom of the file. Edit those values to match the run folder you want to analyze.

## 15. How to Modify a Calculation

The usual edit points are in the config scripts:

```python
cfg.system.electrons
cfg.system.ndim
cfg.system.pbc_lattice
cfg.system.make_local_energy_kwargs
cfg.network.network_type
cfg.network.determinants
cfg.network.complex
cfg.network.CustomPsiformer.*
cfg.network.SlaterNet.*
cfg.mcmc.*
cfg.optim.*
cfg.log.save_path
```

For a new periodic 2D material model, the minimal path is:

1. Build a supercell lattice in `pbc/lattices.py` or directly in the config.
2. Set `cfg.system.pbc_lattice`.
3. Use `make_pbc_feature_layer` with that lattice.
4. Add or select a `potential_type` in `pbc/hamiltonian.py`.
5. Pass potential parameters through `cfg.system.make_local_energy_kwargs`.
6. Choose `CustomPsiformer` for correlated calculations or `SlaterNet` for a Hartree-Fock baseline.
7. Save into a folder name that encodes the physical parameters.

## 16. Practical Caveats

- The code is designed for two-dimensional periodic continuum systems, not generic 3D crystals or lattice Hilbert-space models.
- The atomic/nuclear fields are retained from FermiNet but are not central to the supplied electron-gas workflows.
- Complex output is recommended for periodic systems in the provided configs.
- The default examples use no envelope and no Jastrow. The paper reports that for the studied low-density moire systems, the simple Jastrow did not significantly improve energies and slowed training.
- `mcmc.py` does not implement spin updates; the spin sector is fixed.
- Large runs are expensive. The paper uses long optimization schedules and large batches; quick local runs are only sanity checks.
- `run_2deg.py` should be inspected before use because this checkout hard-codes `network_type = "SlaterNet"` near the architecture selection.
- `run_2deg.py` also has an inconsistent command-line parse path in this checkout; verify it before using positional arguments.
- The evaluation scripts are not general command-line tools; they are small analysis scripts whose hard-coded parameters should be synchronized with the chosen run folder.

## 17. Conceptual Summary

`SlaterNet` answers: how good is the best neural-network single Slater determinant? In the single-determinant, no-Jastrow setting, this is a neural unrestricted Hartree-Fock baseline.

`CustomPsiformer` answers: what happens when each electron orbital can depend on every other electron through self-attention? The attention layers communicate information between electron streams, producing correlated orbitals. Determinants then enforce fermionic antisymmetry. VMC optimization lowers the energy of this flexible ansatz under the periodic continuum Hamiltonian.

This is why the code can solve many-body periodic-boundary problems without choosing a finite band basis: it works directly in continuous electron coordinates, enforces PBC through the input features and Hamiltonian, handles fermionic antisymmetry through determinants, and learns correlation through attention-mediated generalized orbitals.
