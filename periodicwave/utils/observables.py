# Copyright (c) 2026, PeriodicWave contributors.
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

"""Monte Carlo estimators for diagonal real-space observables."""

import numpy as np


def walker_positions_to_samples(positions, ndim=2):
  """Reshapes checkpoint walker positions to (nsamples, nelectrons, ndim)."""
  positions = np.asarray(positions)
  if positions.shape[-1] % ndim != 0:
    raise ValueError(
        f"Last position dimension {positions.shape[-1]} is not divisible by ndim={ndim}."
    )
  return positions.reshape((-1, positions.shape[-1] // ndim, ndim))


def fractional_coordinates(positions, lattice):
  """Converts Cartesian positions to fractional coordinates of a column lattice."""
  lattice_inv = np.linalg.inv(np.asarray(lattice))
  return np.einsum("ij,...j->...i", lattice_inv, positions)


def cartesian_coordinates(fractional_positions, lattice):
  """Converts fractional coordinates to Cartesian coordinates of a column lattice."""
  return np.einsum("ij,...j->...i", np.asarray(lattice), fractional_positions)


def fold_fractional(fractional_positions):
  """Folds fractional coordinates into the centered unit cell [-1/2, 1/2)."""
  return (np.asarray(fractional_positions) + 0.5) % 1.0 - 0.5


def fold_positions(positions, lattice):
  """Folds Cartesian positions into the centered simulation cell."""
  return cartesian_coordinates(fold_fractional(fractional_coordinates(positions, lattice)), lattice)


def _uniform_fractional_edges(bins):
  if np.isscalar(bins):
    nbin = int(bins)
    if nbin <= 0:
      raise ValueError("bins must be positive.")
    return np.linspace(-0.5, 0.5, nbin + 1), np.linspace(-0.5, 0.5, nbin + 1)

  if len(bins) != 2:
    raise ValueError("bins must be an integer or a pair of integers.")
  nx, ny = int(bins[0]), int(bins[1])
  if nx <= 0 or ny <= 0:
    raise ValueError("bins must be positive.")
  return np.linspace(-0.5, 0.5, nx + 1), np.linspace(-0.5, 0.5, ny + 1)


def _bin_areas(lattice, u_edges, v_edges):
  area = abs(np.linalg.det(np.asarray(lattice)))
  return area * np.outer(np.diff(u_edges), np.diff(v_edges))


def _cartesian_bin_mesh(lattice, u_edges, v_edges):
  u_grid, v_grid = np.meshgrid(u_edges, v_edges, indexing="ij")
  frac_grid = np.stack((u_grid, v_grid), axis=-1)
  cart_grid = cartesian_coordinates(frac_grid, lattice)
  return cart_grid[..., 0], cart_grid[..., 1]


def estimate_density(positions, lattice, bins=80):
  """Estimates rho(r) from Monte Carlo walker positions.

  The returned density integrates to the number of electrons. The estimator is

    rho_B = counts_B / (nsamples * area_B),

  where B is a fractional-coordinate bin folded into the simulation cell.
  """
  positions = np.asarray(positions)
  if positions.ndim != 3:
    raise ValueError("positions must have shape (nsamples, nelectrons, ndim).")

  frac = fold_fractional(fractional_coordinates(positions, lattice))
  u_edges, v_edges = _uniform_fractional_edges(bins)
  counts, _, _ = np.histogram2d(
      frac[..., 0].reshape(-1),
      frac[..., 1].reshape(-1),
      bins=(u_edges, v_edges),
  )
  bin_areas = _bin_areas(lattice, u_edges, v_edges)
  density = counts / (positions.shape[0] * bin_areas)
  mesh_x, mesh_y = _cartesian_bin_mesh(lattice, u_edges, v_edges)

  return {
      "density": density,
      "counts": counts,
      "u_edges": u_edges,
      "v_edges": v_edges,
      "mesh_x": mesh_x,
      "mesh_y": mesh_y,
      "bin_areas": bin_areas,
  }


def estimate_pair_correlation(positions, lattice, bins=80, chunk_size=64):
  """Estimates the translationally averaged pair correlation g(r).

  Uses ordered pairs i != j and the normalization

    g_B = area * counts_B / (nsamples * nelectrons * (nelectrons - 1) * area_B).

  For an uncorrelated uniform distribution, g(r) is approximately 1 away from
  finite-size and self-correlation effects.
  """
  positions = np.asarray(positions)
  if positions.ndim != 3:
    raise ValueError("positions must have shape (nsamples, nelectrons, ndim).")
  if positions.shape[1] < 2:
    raise ValueError("At least two electrons are required for pair correlations.")

  frac = fold_fractional(fractional_coordinates(positions, lattice))
  nsamples, nelectrons, _ = frac.shape
  u_edges, v_edges = _uniform_fractional_edges(bins)
  counts = np.zeros((len(u_edges) - 1, len(v_edges) - 1), dtype=float)
  pair_mask = ~np.eye(nelectrons, dtype=bool)

  for start in range(0, nsamples, chunk_size):
    chunk = frac[start:start + chunk_size]
    displacements = chunk[:, :, None, :] - chunk[:, None, :, :]
    displacements = fold_fractional(displacements[:, pair_mask, :].reshape(-1, 2))
    chunk_counts, _, _ = np.histogram2d(
        displacements[:, 0],
        displacements[:, 1],
        bins=(u_edges, v_edges),
    )
    counts += chunk_counts

  cell_area = abs(np.linalg.det(np.asarray(lattice)))
  bin_areas = _bin_areas(lattice, u_edges, v_edges)
  pair_correlation = (
      cell_area * counts / (nsamples * nelectrons * (nelectrons - 1) * bin_areas)
  )
  mesh_x, mesh_y = _cartesian_bin_mesh(lattice, u_edges, v_edges)

  return {
      "pair_correlation": pair_correlation,
      "counts": counts,
      "u_edges": u_edges,
      "v_edges": v_edges,
      "mesh_x": mesh_x,
      "mesh_y": mesh_y,
      "bin_areas": bin_areas,
  }


def reciprocal_index_grid(max_index, include_zero=False):
  """Returns integer reciprocal-lattice indices in a square cutoff."""
  indices = []
  for i in range(-max_index, max_index + 1):
    for j in range(-max_index, max_index + 1):
      if include_zero or i != 0 or j != 0:
        indices.append((i, j))
  return np.asarray(indices, dtype=int)


def reciprocal_vectors(lattice, k_indices):
  """Converts integer reciprocal-lattice indices to Cartesian k vectors."""
  reciprocal_rows = 2 * np.pi * np.linalg.inv(np.asarray(lattice))
  return np.asarray(k_indices) @ reciprocal_rows


def estimate_structure_factor(positions, lattice, k_indices):
  """Estimates S(k) = <rho_k rho_-k> / N from Monte Carlo samples."""
  positions = np.asarray(positions)
  if positions.ndim != 3:
    raise ValueError("positions must have shape (nsamples, nelectrons, ndim).")

  frac = fractional_coordinates(positions, lattice)
  nelectrons = positions.shape[1]
  s_k = []
  for k_index in np.asarray(k_indices):
    phase = np.exp(-2j * np.pi * np.einsum("...i,i->...", frac, k_index))
    rho_k = np.sum(phase, axis=1)
    s_k.append(np.mean(np.abs(rho_k) ** 2) / nelectrons)

  return {
      "k_indices": np.asarray(k_indices, dtype=int),
      "k_vectors": reciprocal_vectors(lattice, k_indices),
      "structure_factor": np.asarray(s_k, dtype=float),
  }
