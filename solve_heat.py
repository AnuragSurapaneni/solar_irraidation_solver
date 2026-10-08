"""
Stand-alone script that in parallel:

Solving the heat equation with different materials 

Version 2.0 5.Aug.2026

Anurag Surapaneni - anurag.surapaneni@bsc.es
Run with:
    mpirun -n N python solve_heat.py

"""

from dolfin import *
import numpy as np
from scipy.spatial import cKDTree
from mpi4py import MPI

# ------------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------------

XDMF_MESH_PATH = "mesh_surface.xdmf"
MATERIAL_XDMF_PATH = "material_markers.xdmf"
VIEW_FACTOR_IN = "view_factors.npz"

START_OFFSET_SECONDS = 0.0

# MATERIALS in compute_view_factors.py by keys
MATERIAL_PROPERTIES = {
    # Laminate glass
    "glass":        dict(RHO=2200.0, CP=840.0, K_COND=1.0,   THICKNESS=0.015, H_FRONT=5.0,  H_BACK=5.0,  EMISSIVITY=0.0),
    # Aluminum 2024-T3 
    "solid_1":  dict(RHO=2780.0, CP=875.0, K_COND=130.0, THICKNESS=0.002, H_FRONT=8.0,  H_BACK=8.0,  EMISSIVITY=0.85),
    "solid_2":  dict(RHO=2780.0, CP=875.0, K_COND=130.0, THICKNESS=0.002, H_FRONT=8.0,  H_BACK=8.0,  EMISSIVITY=0.85),
}

SIGMA_SB = 5.670374419e-8
# Ambient Temperature
T_AMB = 293.15
dt = 60.0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = np.load(VIEW_FACTOR_IN)
face_centers = data["face_centers"]
solar_t = data["solar_t"]
flux_per_solar_step = data["flux_per_solar_step"]
material_names = [str(n) for n in data["material_names"]]

missing = [n for n in material_names if n not in MATERIAL_PROPERTIES]
if missing:
    raise ValueError(
        f"Mismatch in material defintion."
    )

SIM_DURATION_SECONDS = solar_t[-1] - START_OFFSET_SECONDS
num_steps = int(np.ceil(SIM_DURATION_SECONDS / dt))

if rank == 0:
    print(f"Simulating {SIM_DURATION_SECONDS/3600:.2f} hours "
          f"({num_steps} steps of dt={dt}s)")
    print(f"Materials: {material_names}")

mesh = Mesh()
with XDMFFile(comm, XDMF_MESH_PATH) as infile:
    infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile(comm, MATERIAL_XDMF_PATH) as infile:
    infile.read(mvc, "material_id")
material_mf = MeshFunction("size_t", mesh, mvc)

V_flux = FunctionSpace(mesh, "CG", 1)
V = FunctionSpace(mesh, "P", 1)
tree = cKDTree(face_centers)
n_owned = V_flux.dofmap().ownership_range()[1] - V_flux.dofmap().ownership_range()[0]
dof_coords = V_flux.tabulate_dof_coordinates()[:n_owned]
_, dof_to_face_idx = tree.query(dof_coords)
flux = Function(V_flux, name="SolarFlux")


def flux_at_time(t_abs):
    if t_abs <= solar_t[0]:
        idx = 0
    else:
        idx = np.searchsorted(solar_t, t_abs, side="right") - 1
        idx = min(idx, len(solar_t) - 1)
    return flux_per_solar_step[idx][dof_to_face_idx]


Vdg = FunctionSpace(mesh, "DG", 0)
local_material_ids = material_mf.array()
local_material_names = [material_names[mid] for mid in local_material_ids]


def make_dg_field(prop_key):
    values = np.array(
        [MATERIAL_PROPERTIES[name][prop_key] for name in local_material_names],
        dtype=np.float64,
    )
    f = Function(Vdg)
    f.vector().set_local(values)
    f.vector().apply("insert")
    return f


rho = make_dg_field("RHO")
cp = make_dg_field("CP")
k = make_dg_field("K_COND")
thickness = make_dg_field("THICKNESS")
h_front = make_dg_field("H_FRONT")
h_back = make_dg_field("H_BACK")
emissivity = make_dg_field("EMISSIVITY")

T_amb = Constant(T_AMB)
sigma_sb = Constant(SIGMA_SB)

u = Function(V, name="Temperature")
u_n = Function(V)
u_trial = TrialFunction(V)
v = TestFunction(V)

# initial condition
u_n.interpolate(Constant(T_AMB))
u.assign(u_n)

# Fully consistent mass matrix 
a = (
    (rho * cp * thickness / dt) * u_trial * v * dx
    + k * thickness * dot(grad(u_trial), grad(v)) * dx
    + h_front * u_trial * v * dx
    + h_back * u_trial * v * dx
)
A = assemble(a)

xdmf = XDMFFile(comm, "shell_heat.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

# Write material ID and properties
material_out = Function(Vdg, name="MaterialID")
material_out.vector().set_local(local_material_ids.astype(np.float64))
material_out.vector().apply("insert")
xdmf.write(material_out, 0.0)

rho.rename("Density", "")
cp.rename("SpecificHeat", "")
k.rename("ThermalConductivity", "")

xdmf.write(rho, 0.0)
xdmf.write(cp, 0.0)
xdmf.write(k, 0.0)

# start time loop

t = 0.0
save_every = 30 * 60  # (Save every 30 mins)
next_save = save_every
for n in range(num_steps):
    t += dt
    t_abs = START_OFFSET_SECONDS + t
    flux.vector().set_local(flux_at_time(t_abs))
    flux.vector().apply("insert")
    L = (
        (rho * cp * thickness / dt) * u_n * v * dx
        + flux * v * dx
        + h_front * T_amb * v * dx
        + h_back * T_amb * v * dx
        - emissivity * sigma_sb * (u_n**4 - T_amb**4) * v * dx
    )
    b = assemble(L)
    solve(A, u.vector(), b, "cg", "amg")
    u_n.assign(u)
    if t >= next_save:
        xdmf.write(u, t)
        xdmf.write(flux, t)
        next_save += save_every

    flux_min, flux_max = flux.vector().min(), flux.vector().max()
    u_min, u_max = u.vector().min(), u.vector().max()
    if rank == 0:
        print(f"[solve] step {n}, t={t:.3f}s (t_abs={t_abs:.1f}s), "
              f"flux range [{flux_min:.1f}, {flux_max:.1f}] W/m^2, "
              f"T range [{u_min:.2f}, {u_max:.2f}] K")

if rank == 0:
    print("Done. Wrote shell_heat.xdmf (temperature + flux + material map, all materials combined).")
