import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32) 
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:,1,1])
    rays[:,1,0] = 1
    return rays

"""
rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)
"""

def intersect_ray_1d(
    ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    A = t.stack([ray[1,0:2], segment[0,0:2] - segment[1,0:2]], dim=1)   # (2,2) tensor
    B = (segment[0,0:2] - ray[0,0:2]).unsqueeze(1)  # tensor of shape (2,1)
    
    try:
        X = t.linalg.solve(A, B)    # solve the system of linear equations A X = B
    except RuntimeError:   # in case the system can't be solved (ray and segment are parallel)
        return False

    if X[0] >= 0 and 0 <= X[1] <= 1:
        return True
    else:
        return False


# tests.test_intersect_ray_1d(intersect_ray_1d)
# tests.test_intersect_ray_1d_special_case(intersect_ray_1d)


def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """

    NR = rays.size(0)
    NS = segments.size(0)

    # remove z coordinate
    rays = rays[...,:2]
    segments = segments[...,:2]

    # we amplify the matrices to cover each ray-segment combination
    rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
    segments = einops.repeat(segments, "nsegments p d -> nrays nsegments p d", nrays=NR)

    O = rays[:,:,0]
    D = rays[:,:,1]
    L_1 = segments[:,:,0]
    L_2 = segments[:,:,1]

    A = t.stack([D, L_1 - L_2], dim=-1)
    B = L_1 - O

    determinants = t.linalg.det(A)
    is_singular = (determinants.abs() < 1e-8)
    A[is_singular] = t.eye(2)

    X = t.linalg.solve(A,B)
    U = X[...,0]
    V = X[...,1]

    return ((U >= 0) & (V >= 0) & (V <= 1) & ~is_singular).any(dim=-1)


# tests.test_intersect_rays_1d(intersect_rays_1d)
# tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float
) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    nrays = num_pixels_y * num_pixels_z
    O = t.tensor([0, 0, 0])     # define the origin; shape (3,)
    rays = t.zeros(nrays, 2, 3)     # each ray is defined by 2 3d tensors: the origin and direction

    # define the tensors of midpoints P of each pixel
    P_y = t.linspace(-y_limit, +y_limit, steps=num_pixels_y)
    P_z = t.linspace(-z_limit, +z_limit, steps=num_pixels_z)
    P_y = einops.repeat(P_y, "npixy -> (npixy npixz)", npixz=num_pixels_z)
    P_z = einops.repeat(P_z, "npixz -> (npixy npixz)", npixy=num_pixels_y)
    P_x = t.ones(nrays)   # all pixels have midpoints at x = 1
    P = t.stack([P_x, P_y, P_z], dim=1)   # midpoint of each pixel; shape (nrays, 3)

    D = P - O   # direction vector; D has shape (nrays, 3); O gets broadcasted from shape (3,) into (nrays,3)

    rays[:,1,:] = D     # assign D to every ray's second position

    return rays


# rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
# render_lines_with_plotly(rays_2d)

Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    mat = t.stack([-D, B - A, C - A], dim=1)
    vec = O - A
    s, u, v = t.linalg.solve(mat, vec)

    if (s.item() >= 0) & (u.item() >= 0) & (v.item() >= 0) & (u.item() + v.item() <= 1):
        return True
    else:
        return False


# tests.test_triangle_ray_intersects(triangle_ray_intersects)

# x = t.arange(6).view(2, 3)
# print(x)
# print("Shape:", x.shape)      # (2, 3)
# print("Stride:", x.stride())

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    nrays = rays.size(0)
    triangle = triangle.expand(nrays,-1,-1)
    A, B, C = triangle.unbind(dim=1)
    O, D = rays.unbind(dim=1)

    mat = t.stack([-D, B - A, C - A], dim=-1)
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A
    X = t.linalg.solve(mat, vec)
    s, u, v = X.unbind(dim=1)
    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
# render_lines_with_plotly(rays2d, triangle_lines)

"""
# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
"""

triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    NR = rays.size(0)
    NT = triangles.size(0)

    rays = einops.repeat(rays, "nrays p d -> nrays ntriangles p d", ntriangles=NT)
    triangles = einops.repeat(triangles, "ntriangles p d -> nrays ntriangles p d", nrays=NR)

    A, B, C = triangles.unbind(dim=2)    # A, B, C have shape (NR, NT, 3)
    O, D = rays.unbind(dim=2)   # O, D have shape (NR, NT, 3)

    mat = t.stack([-D, B - A, C - A], dim=-1)   # shape (NR, NT, 3, 3)
    dets = t.linalg.det(mat)    # shape (NR, NT)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A     # shape (NR, NT, 3)
    X = t.linalg.solve(mat, vec)    # shape (NR, NT, 3)
    s, u, v = X.unbind(dim=2)   # s, u, v have shape (NR, NT)
    
    distance = s * D[...,0]
    intersects = (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    distance[~intersects] = float('inf')

    return einops.reduce(distance, "NR NT -> NR", "min")


"""
num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
"""

def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    Ry = t.tensor([[t.cos(theta), 0.0, t.sin(theta)],
        [0.0, 1.0, 0.0],
        [-t.sin(theta), 0.0, t.cos(theta)]])

    return Ry

# tests.test_rotation_matrix(rotation_matrix)

def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is distance to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D
    grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(
        coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video"
    ).show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])

"""
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)
"""

"""
print("torch:", t.__version__, "cuda:", t.version.cuda)
print("is_available:", t.cuda.is_available())
print("device_count:", t.cuda.device_count())
if t.cuda.is_available(): print(t.cuda.get_device_name(0))
try: t.cuda.init(); print("init OK")
except Exception as e: print("cuda init error:", e)
"""

def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU.
    """
    NR = rays.size(0)
    NT = triangles.size(0)
    device = "cuda"
    triangles = triangles.to(device)
    rays = rays.to(device)

    rays = einops.repeat(rays, "nrays p d -> nrays ntriangles p d", ntriangles=NT)
    triangles = einops.repeat(triangles, "ntriangles p d -> nrays ntriangles p d", nrays=NR)

    A, B, C = triangles.unbind(dim=2)    # A, B, C have shape (NR, NT, 3)
    O, D = rays.unbind(dim=2)   # O, D have shape (NR, NT, 3)

    mat = t.stack([-D, B - A, C - A], dim=-1)   # shape (NR, NT, 3, 3)
    dets = t.linalg.det(mat)    # shape (NR, NT)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3).to(device)

    vec = O - A     # shape (NR, NT, 3)
    X = t.linalg.solve(mat, vec)    # shape (NR, NT, 3)
    s, u, v = X.unbind(dim=2)   # s, u, v have shape (NR, NT)
    
    distance = s * D[...,0]
    intersects = (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    distance[~intersects] = float('inf')

    return einops.reduce(distance, "NR NT -> NR", "min").cpu()

dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh_gpu, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)
