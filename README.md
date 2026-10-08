# Solar Irradiation and Thin-Shell Heat Solver

This project contains two Python scripts that generate direct-solar loading on a triangulated surface and then solve its transient temperature field:

1. `compute_view_factors.py` reads material STL files, calculates the sun position and clear-sky direct normal irradiance (DNI), ray-traces shadows, and writes the mesh and solar loading data.
2. `solve_heat.py` reads those files and solves a transient, material-dependent heat equation on the surface mesh.

Run both scripts from the directory containing the input STL files. The scripts use constants near the top of each file for their configuration.

## Requirements

- Python with NumPy, pandas, SciPy, mpi4py, trimesh, meshio, pvlib, and `rtree` available.
- An MPI implementation such as Open MPI or MPICH, including `mpirun`.
- Classic FEniCS (`dolfin`) for `solve_heat.py`. This code uses the `dolfin` API and is not written for FEniCSx.

The first script imports NumPy, pandas, mpi4py, trimesh, meshio, and pvlib. Trimesh's default ray-tracing path uses an `rtree` spatial index. The heat solver additionally uses SciPy's `cKDTree` and classic FEniCS/DOLFIN. `mpi4py` must be installed against the MPI implementation used to launch the scripts. DOLFIN is commonly installed through a system package or a dedicated FEniCS environment; installing the Python packages alone may not provide it.

For the packages available from PyPI, an install command is:

```bash
python -m pip install numpy pandas scipy mpi4py trimesh meshio pvlib rtree
```

Install a compatible MPI runtime and classic FEniCS/DOLFIN separately for your operating system or environment.

## Inputs and configuration

Place these files in the working directory before running the first script:

| File | Material label |
| --- | --- |
| `GLASS.stl` | `glass` |
| `SOLID_1.stl` | `solid_1` |
| `SOLID_2.stl` | `solid_2` |

Edit the `MATERIALS` list in `compute_view_factors.py` to change filenames or labels. The same labels must appear as keys in `MATERIAL_PROPERTIES` in `solve_heat.py`. The material ID is the position in the `MATERIALS` list, starting at zero.

The STL parts should form closed surfaces and share one coordinate system. Coordinates and thickness values are assumed to use metres. Mesh orientation matters: the script uses face normals for solar incidence and estimates outward directions from each part's centroid when offsetting shadow rays. The default `MESH_TO_ENU` matrix assumes mesh +X points East, +Y North, and +Z Up. Change it if the STL uses another orientation. `WELD_TOLERANCE` controls vertex merging where surfaces meet.

Set `LATITUDE`, `LONGITUDE`, and `ELEVATION_M` for the site. `TIMES` is a timezone-aware UTC date range; the defaults describe one day in Barcelona at hourly intervals. The solver's material densities (`RHO`), heat capacities (`CP`), conductivities (`K_COND`), thicknesses, convection coefficients (`H_FRONT`, `H_BACK`), and emissivity are set in `MATERIAL_PROPERTIES`. `T_AMB`, `dt`, and `START_OFFSET_SECONDS` configure the heat run.

## Run

From the project/input directory, first calculate the solar loading, then solve the heat equation. Use the same MPI size or a different size for either step:

```bash
mpirun -n 4 python compute_view_factors.py
mpirun -n 4 python solve_heat.py
```

The preprocessing script distributes time steps across MPI ranks and writes combined outputs on rank 0. The solver uses DOLFIN's MPI mesh and linear algebra support. Choose an MPI installation compatible with both `mpi4py` and DOLFIN.

## Model theory

### Solar loading and shadows

For each configured time, `pvlib` calculates apparent solar elevation and azimuth at the site and estimates clear-sky DNI with the Ineichen model. Night-time DNI is set to zero. A sun direction is constructed in East-North-Up coordinates and transformed into mesh coordinates using `MESH_TO_ENU`.

For each triangle, the script calculates a direct-beam exposure factor

\[
f_i(t) = \max(0, \mathbf{n}_i \cdot \mathbf{s}(t))\,V_i(t),
\]

where \(\mathbf{n}_i\) is the triangle face normal, \(\mathbf{s}\) points toward the sun, and \(V_i\) is one when the ray from that triangle is unobstructed and zero when another triangle blocks it. The incident solar flux is `DNI * f_i` in W/m².

Despite the output variable name `view_factors`, these are direct-sun cosine and visibility factors. The calculation does not include diffuse sky radiation, surface reflection, or radiative exchange between mesh faces. It uses one ray per triangle and the hourly default `TIMES`, so shadow and DNI changes between configured times are held piecewise constant by the heat solver.

### Transient surface heat equation

The solver uses a continuous, piecewise-linear temperature field on the triangulated surface and material-wise constant properties represented with discontinuous, piecewise-constant fields. For each material it applies density \(\rho\), specific heat \(c_p\), conductivity \(k\), and shell thickness \(d\). In surface notation, the implemented equation is

\[
\rho c_p d\,\frac{\partial T}{\partial t}
- \nabla_s \cdot (k d\,\nabla_s T)
+ (h_f + h_b)T
= q_{sun} + (h_f + h_b)T_{amb}
- \epsilon\sigma(T_{old}^4 - T_{amb}^4),
\]

where \(q_{sun}\) is the direct solar flux above, \(h_f\) and \(h_b\) are the two convection coefficients, and \(\sigma\) is the Stefan-Boltzmann constant. The initial temperature is `T_AMB`. The mass and conduction/convection terms are assembled in an implicit Euler step; the radiative term is evaluated from the previous time step (`u_n`), so it is explicit. The `emissivity` field contributes one radiative loss term as implemented in the script.

## Outputs

The preprocessing step writes:

- `mesh_surface.xdmf` and its mesh data file: combined triangle mesh.
- `material_markers.xdmf` and its mesh data file: triangle material IDs.
- `view_factors.npz`: `face_centers`, `solar_t`, `flux_per_solar_step`, `material_id`, and `material_names` arrays.
- `sun_position.csv`: UTC timestamps, solar elevation/azimuth, and DNI at each configured time.

The heat solver writes `shell_heat.xdmf` and its associated data file. It contains temperature and solar flux time series plus the material ID, density, specific heat, and thermal conductivity fields. Temperature and flux are written every 30 minutes; material fields are written at time zero. The solver maps face fluxes to finite-element degrees of freedom using the nearest triangle center. A final state is written only when the simulation time reaches a 30-minute save interval.

## Limitations

The solar model is clear-sky and direct-beam only. Solar forcing is sampled at the timestamps in `TIMES`, and the heat solve uses a fixed time step with a piecewise-constant lookup of those samples. The mesh is treated as a thin shell with a single temperature across its thickness; no through-thickness temperature gradient is resolved. Check that the chosen material properties, STL scale/orientation, and site/time settings match the intended physical case.
