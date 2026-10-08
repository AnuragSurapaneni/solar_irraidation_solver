"""
Stand-alone script that in parallel:

1. Load and cleans an STLs and assigns them material properties as indicies order of STL does not matter 
2. Stiches all STL (all of them need to from a closed surface)
2. Computes sun angle for a given location, and time frame
3. Computes view factors over the stiched surface and using pvlib
4. Saves view factors for the stiched surface

Version 2 05.Aug.2026

Anurag Surapaneni - anurag.surapaneni@bsc.es

Run with:
    mpirun -n N python compute_view_factors.py
"""

import numpy as np
import pandas as pd
from mpi4py import MPI
import trimesh
import meshio
import pvlib

# ------------------------------------------------------------------
MATERIALS = [
    {"name": "glass", "stl": "GLASS.stl"},
     {"name": "solid_1",  "stl": "SOLID_1.stl"},
     {"name": "solid_2",  "stl": "SOLID_2.stl"},
]

XDMF_MESH_PATH = "mesh_surface.xdmf"
MATERIAL_XDMF_PATH = "material_markers.xdmf"
VIEW_FACTOR_OUT = "view_factors.npz"
SUN_POSITION_CSV = "sun_position.csv"

# welding, if the surfaces dont overlap increase this untill they do
WELD_TOLERANCE = None  # None = trimesh default

# (example: Barcelona)
LATITUDE = 41.3874
LONGITUDE = 2.1686
ELEVATION_M = 20.0

TIMES = pd.date_range("2026-07-30 06:00", "2026-07-30 20:00", freq="1h", tz="UTC")

# STL orientation is important
MESH_TO_ENU = np.array([
    [1, 0, 0],   # mesh +X -> East
    [0, 1, 0],   # mesh +Y -> North
    [0, 0, 1],   # mesh +Z -> Up
], dtype=float)

# ------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

material_names = [m["name"] for m in MATERIALS]

submeshes = []
corrected_normals_chunks = []
material_id_chunks = []

for mat_idx, mat in enumerate(MATERIALS):
    sub = trimesh.load(mat["stl"], force='mesh', process=True)
    sub_center = sub.centroid
    vec_from_center = sub.triangles_center - sub_center
    dots = np.einsum('ij,ij->i', sub.face_normals, vec_from_center)
    sub_corrected_normals = sub.face_normals.copy()
    sub_corrected_normals[dots < 0] *= -1

    submeshes.append(sub)
    corrected_normals_chunks.append(sub_corrected_normals)
    material_id_chunks.append(np.full(len(sub.faces), mat_idx, dtype=np.int64))

mesh_tri = trimesh.util.concatenate(submeshes)
corrected_normals = np.concatenate(corrected_normals_chunks)
material_id = np.concatenate(material_id_chunks)

# Weld surfaces
if WELD_TOLERANCE is not None:
    mesh_tri.merge_vertices(digits_vertex=None, merge_tex=True, merge_norm=True)
else:
    mesh_tri.merge_vertices()

# Clean combined mesh 
mask = mesh_tri.nondegenerate_faces()
mesh_tri.update_faces(mask)
mesh_tri.remove_unreferenced_vertices()
corrected_normals = corrected_normals[mask]
material_id = material_id[mask]
face_centers = mesh_tri.triangles_center
origins = face_centers + (corrected_normals * 1e-4)

if rank == 0:
    tri_cells = mesh_tri.faces
    meshio.write(
        XDMF_MESH_PATH,
        meshio.Mesh(points=mesh_tri.vertices, cells={"triangle": tri_cells}),
    )
    meshio.write(
        MATERIAL_XDMF_PATH,
        meshio.Mesh(
            points=mesh_tri.vertices,
            cells={"triangle": tri_cells},
            cell_data={"material_id": [material_id.astype(np.int64)]},
        ),
    )

solpos = pvlib.solarposition.get_solarposition(TIMES, LATITUDE, LONGITUDE, altitude=ELEVATION_M)
altitude_deg = solpos["apparent_elevation"].to_numpy()
azimuth_deg = solpos["azimuth"].to_numpy()
solar_t = (TIMES - TIMES[0]).total_seconds().to_numpy().astype(np.float64)

apparent_zenith = 90.0 - altitude_deg
airmass_rel = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
pressure = pvlib.atmosphere.alt2pres(ELEVATION_M)
airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, pressure)
linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(TIMES, LATITUDE, LONGITUDE)
cs = pvlib.clearsky.ineichen(
    pd.Series(apparent_zenith, index=TIMES),
    pd.Series(airmass_abs, index=TIMES),
    linke_turbidity, altitude=ELEVATION_M,
)
dni_per_step = cs["dni"].to_numpy()
dni_per_step[altitude_deg <= 0] = 0.0


def sun_vector_enu(alt_deg, az_deg):
    alt_r, az_r = np.radians(alt_deg), np.radians(az_deg)
    return np.array([
        np.cos(alt_r) * np.sin(az_r),   # East
        np.cos(alt_r) * np.cos(az_r),   # North
        np.sin(alt_r),                  # Up
    ])


n_steps = len(TIMES)
n_faces = len(face_centers)

my_steps = np.array_split(np.arange(n_steps), size)[rank]
local_view_factors = np.zeros((len(my_steps), n_faces))

for local_i, i in enumerate(my_steps):
    if altitude_deg[i] <= 0:
        continue
    sun_enu = sun_vector_enu(altitude_deg[i], azimuth_deg[i])
    sun_vec = MESH_TO_ENU.T @ sun_enu
    sun_vec = sun_vec / np.linalg.norm(sun_vec)
    cos_theta = np.maximum(0, np.dot(mesh_tri.face_normals, sun_vec))
    directions = np.tile(sun_vec, (n_faces, 1))
    has_intersections = mesh_tri.ray.intersects_any(ray_origins=origins, ray_directions=directions)
    in_light = ~has_intersections
    local_view_factors[local_i, :] = cos_theta * in_light

    print(f"[rank {rank}] [view factors {i:03d}/{n_steps-1}] {TIMES[i]}  "
          f"alt={altitude_deg[i]:5.1f} az={azimuth_deg[i]:6.1f} "
          f"DNI={dni_per_step[i]:6.1f} W/m^2  lit_faces={int(in_light.sum())}/{n_faces}",
          flush=True)


all_view_factors = comm.gather(local_view_factors, root=0)
all_steps = comm.gather(my_steps, root=0)

comm.barrier()

if rank == 0:
    view_factor_per_step = np.zeros((n_steps, n_faces))
    for steps_chunk, vf_chunk in zip(all_steps, all_view_factors):
        view_factor_per_step[steps_chunk, :] = vf_chunk

    flux_per_solar_step = view_factor_per_step * dni_per_step[:, None]

    np.savez(
        VIEW_FACTOR_OUT,
        face_centers=face_centers,
        solar_t=solar_t,
        flux_per_solar_step=flux_per_solar_step,
        material_id=material_id,
        material_names=np.array(material_names),
    )

    pd.DataFrame({
        "step": np.arange(n_steps), "time_utc": TIMES,
        "altitude_deg": altitude_deg, "azimuth_deg": azimuth_deg, "dni": dni_per_step,
    }).to_csv(SUN_POSITION_CSV, index=False)

    print(f"Done. Wrote {VIEW_FACTOR_OUT}, {XDMF_MESH_PATH}, {MATERIAL_XDMF_PATH}, "
          f"and {SUN_POSITION_CSV}.")
    print(f"Materials: {material_names}")
