"""Conservative full-chassis swept clearance against bounded planar road patches.

This geometry certificate tests the *body*, not wheel-contact validity. It
supports ramps and overhangs by comparing 3D OBB vertices with each patch
plane only where swept XZ footprints may overlap that patch. Insufficient
resolution rejects the candidate. A parabola contributes g*dt**2/8 sagitta.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, floor, hypot, inf, isfinite, nextafter, sqrt, ulp
from types import MappingProxyType
from typing import Callable, Mapping

from .body_collision import VehicleBodyGeometry, obb_separation_lower_bound
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .independent_suspension_dynamics import SuspensionResolutionError
from .multi_surface_support import _body_corners, _intersection
from .road_footprint import _inflate, body_xz_polygon
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState


@dataclass(frozen=True, slots=True)
class _BoundedPatchIndex:
    source_patches: tuple
    obstacles: tuple[OrientedBoxVolume, ...]
    obstacle_axes: tuple[tuple[Vec3, Vec3, Vec3], ...]
    obstacle_half_extents: tuple[Vec3, ...]
    cells: Mapping[tuple[int, int, int], tuple[int, ...]]
    footprint_cells: Mapping[tuple[int, int], tuple[int, ...]]
    footprint_query_candidates: Mapping[tuple[int, int], tuple[int, ...]]
    coverage_edges: tuple[tuple[tuple[float, float, float, float, float], ...], ...]
    always_footprint_indices: tuple[int, ...]
    all_indices: tuple[int, ...]
    cell_size_m: float
    footprint_cell_size_m: float
    max_patch_radius_m: float

    def candidate_indices(
        self, body_box: OrientedBoxVolume, clearance_threshold_m: float,
    ) -> tuple[int, ...]:
        if (not isfinite(clearance_threshold_m) or clearance_threshold_m < 0.
                or not isfinite(self.cell_size_m) or self.cell_size_m <= 0.):
            return self.all_indices
        body_axes = tuple(body_box.orientation.rotate(axis) for axis in (
            Vec3(1., 0., 0.), Vec3(0., 1., 0.), Vec3(0., 0., 1.),
        ))
        return self._candidate_indices_with_axes(body_box, clearance_threshold_m, body_axes)

    def _candidate_indices_with_axes(
        self, body_box: OrientedBoxVolume, clearance_threshold_m: float,
        body_axes: tuple[Vec3, Vec3, Vec3],
    ) -> tuple[int, ...]:
        all_indices = self.all_indices
        if (not isfinite(clearance_threshold_m) or clearance_threshold_m < 0.
                or not isfinite(self.cell_size_m) or self.cell_size_m <= 0.):
            return all_indices
        body_half_extents = (body_box.half_extents.x, body_box.half_extents.y,
                             body_box.half_extents.z)
        limits = tuple(
            half_extent + self.max_patch_radius_m + clearance_threshold_m
            for half_extent in body_half_extents
        )
        radii = (
            sum(abs(axis.x) * limit for axis, limit in zip(body_axes, limits, strict=True)),
            sum(abs(axis.y) * limit for axis, limit in zip(body_axes, limits, strict=True)),
            sum(abs(axis.z) * limit for axis, limit in zip(body_axes, limits, strict=True)),
        )
        scale = max(1., *radii, abs(body_box.center.x), abs(body_box.center.y),
                    abs(body_box.center.z))
        if not all(isfinite(value) for value in (*limits, *radii, scale)):
            return all_indices
        margin = 64. * ulp(scale)
        center = body_box.center
        lower = tuple(nextafter(value - radius - margin, -inf)
                      for value, radius in zip((center.x, center.y, center.z), radii, strict=True))
        upper = tuple(nextafter(value + radius + margin, inf)
                      for value, radius in zip((center.x, center.y, center.z), radii, strict=True))
        if not all(isfinite(value) for value in (*lower, *upper)):
            return all_indices
        low = tuple(floor(nextafter(value, -inf) / self.cell_size_m) for value in lower)
        high = tuple(floor(nextafter(value, inf) / self.cell_size_m) for value in upper)
        cell_count = ((high[0] - low[0] + 1) * (high[1] - low[1] + 1)
                      * (high[2] - low[2] + 1))
        if cell_count > max(128, len(self.cells) * 8):
            return all_indices

        candidates = set()
        for x in range(low[0], high[0] + 1):
            for y in range(low[1], high[1] + 1):
                for z in range(low[2], high[2] + 1):
                    candidates.update(self.cells.get((x, y, z), ()))
        return tuple(sorted(candidates))

    def has_separation_at_most(
        self, body_box: OrientedBoxVolume, clearance_threshold_m: float,
    ) -> bool:
        if not isfinite(clearance_threshold_m):
            return True
        body_axes = tuple(body_box.orientation.rotate(axis) for axis in (
            Vec3(1., 0., 0.), Vec3(0., 1., 0.), Vec3(0., 0., 1.),
        ))
        body_half = (body_box.half_extents.x, body_box.half_extents.y,
                     body_box.half_extents.z)
        candidates = self._candidate_indices_with_axes(
            body_box, clearance_threshold_m, body_axes,
        )
        for index in candidates:
            obstacle = self.obstacles[index]
            obstacle_axes = self.obstacle_axes[index]
            delta = obstacle.center - body_box.center
            obstacle_half = self.obstacle_half_extents[index]
            body_axis_gaps = tuple(
                abs(delta.dot(axis))
                - sum(half * abs(axis.dot(unit)) for half, unit in zip(
                    body_half, body_axes, strict=True,
                ))
                - sum(half * abs(axis.dot(unit)) for half, unit in zip(
                    (obstacle_half.x, obstacle_half.y, obstacle_half.z),
                    obstacle_axes, strict=True,
                ))
                for axis in body_axes
            )
            if (all(isfinite(gap) for gap in body_axis_gaps)
                    and any(gap > clearance_threshold_m for gap in body_axis_gaps)):
                continue
            separation = obb_separation_lower_bound(body_box, obstacle)
            if separation != separation:
                return True
            if separation <= clearance_threshold_m:
                return True
        return False

    def candidate_indices_near_xz(
        self, x: float, z: float, footprint_radius_m: float,
    ) -> tuple[int, ...]:
        if (not isfinite(x) or not isfinite(z)
                or not isfinite(footprint_radius_m)
                or not 0. <= footprint_radius_m <= 0.2
                or not isfinite(self.footprint_cell_size_m)
                or self.footprint_cell_size_m <= 0.):
            return self.all_indices
        cell = (
            floor(x / self.footprint_cell_size_m),
            floor(z / self.footprint_cell_size_m),
        )
        return self.footprint_query_candidates.get(cell, self.always_footprint_indices)


def _build_bounded_patch_index(atlas: tuple, body: VehicleBodyGeometry) -> _BoundedPatchIndex:
    if not isinstance(body, VehicleBodyGeometry):
        raise ValueError("finite patch index requires typed body geometry")
    if (len(atlas) != 2 or not atlas[0] or len(atlas[0]) != len(atlas[1])
            or any(len(polygon) < 3 for polygon in atlas[1])):
        raise ValueError("bounded authored atlas required")

    obstacles = tuple(
        OrientedBoxVolume(
            patch.origin,
            Vec3(patch.width / 2., 1e-9, patch.length / 2.),
            Quaternion.from_basis(patch.right, patch.normal, patch.forward),
        )
        for patch in atlas[0]
    )
    obstacle_axes = tuple(tuple(obstacle.orientation.rotate(axis) for axis in (
        Vec3(1., 0., 0.), Vec3(0., 1., 0.), Vec3(0., 0., 1.),
    )) for obstacle in obstacles)
    obstacle_half_extents = tuple(obstacle.half_extents for obstacle in obstacles)
    patch_radii = tuple(
        hypot(patch.half_extents.x, patch.half_extents.y, patch.half_extents.z)
        for patch in obstacles
    )
    max_patch_radius = max(patch_radii)
    cell_size = max(1., min(4., 2. * max_patch_radius))
    cells_mutable: dict[tuple[int, int, int], list[int]] = {}
    for index, patch in enumerate(obstacles):
        key = tuple(floor(value / cell_size) for value in
                    (patch.center.x, patch.center.y, patch.center.z))
        cells_mutable.setdefault(key, []).append(index)
    cells = MappingProxyType({key: tuple(indices) for key, indices in cells_mutable.items()})
    footprint_cell_size = 4.
    footprint_cells_mutable: dict[tuple[int, int], list[int]] = {}
    coverage_edges = tuple(
        tuple((
            a[0], a[1], b[0] - a[0], b[1] - a[1],
            hypot(b[0] - a[0], b[1] - a[1]),
        ) for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True))
        for polygon in atlas[1]
    )
    always_footprint_indices = []
    max_footprint_cells = max(128, len(obstacles) * 8)
    for index, polygon in enumerate(atlas[1]):
        padding = 0.
        for vertex_index, vertex in enumerate(polygon):
            previous = polygon[vertex_index - 1]
            following = polygon[(vertex_index + 1) % len(polygon)]
            before = (previous[0] - vertex[0], previous[1] - vertex[1])
            after = (following[0] - vertex[0], following[1] - vertex[1])
            before_length = hypot(*before)
            after_length = hypot(*after)
            if before_length <= 1e-12 or after_length <= 1e-12:
                padding = inf
                break
            cosine = (before[0] * after[0] + before[1] * after[1]) / (
                before_length * after_length
            )
            sine_half_angle = sqrt(max(0., (1. - min(1., max(-1., cosine))) / 2.))
            if sine_half_angle <= 1e-12:
                padding = inf
                break
            padding = max(padding, 0.2 / sine_half_angle)
        if not isfinite(padding):
            always_footprint_indices.append(index)
            continue
        xs, zs = zip(*polygon, strict=True)
        scale = max(1., padding, *(abs(value) for value in (*xs, *zs)))
        margin = 64. * ulp(scale)
        lower_x = nextafter(min(xs) - padding - margin, -inf)
        upper_x = nextafter(max(xs) + padding + margin, inf)
        lower_z = nextafter(min(zs) - padding - margin, -inf)
        upper_z = nextafter(max(zs) + padding + margin, inf)
        if not all(isfinite(value) for value in (lower_x, upper_x, lower_z, upper_z)):
            always_footprint_indices.append(index)
            continue
        low_x = floor(lower_x / footprint_cell_size)
        high_x = floor(upper_x / footprint_cell_size)
        low_z = floor(lower_z / footprint_cell_size)
        high_z = floor(upper_z / footprint_cell_size)
        cell_count = (high_x - low_x + 1) * (high_z - low_z + 1)
        if cell_count > max_footprint_cells:
            always_footprint_indices.append(index)
            continue
        for x in range(low_x, high_x + 1):
            for z in range(low_z, high_z + 1):
                footprint_cells_mutable.setdefault((x, z), []).append(index)
    footprint_cells = MappingProxyType({key: tuple(indices)
                                        for key, indices in footprint_cells_mutable.items()})
    footprint_query_candidates = MappingProxyType({
        key: tuple(sorted(set(always_footprint_indices).union(indices)))
        for key, indices in footprint_cells.items()
    })
    return _BoundedPatchIndex(
        atlas[0], obstacles, obstacle_axes, obstacle_half_extents,
        cells, footprint_cells, footprint_query_candidates, coverage_edges,
        tuple(always_footprint_indices), tuple(range(len(obstacles))),
        cell_size, footprint_cell_size, max_patch_radius,
    )


def certify_swept_chassis_clearance(
    pose_at: Callable[[float], VehicleState],
    start_s: float,
    end_s: float,
    body: VehicleBodyGeometry,
    atlas: tuple,
    *,
    max_subdivisions: int = 12,
    gravity_mps2: float = 0.,
) -> int:
    """Certify over prescribed linear+SLERP or exact ballistic pose_at.

    For ballistic time sampling, gravitational curvature between endpoints is
    bounded by g*h²/8. Every movement bound is over-conservative, including
    projected XZ hull and plane-normal clearance. A witness or unresolved
    leaf raises SuspensionResolutionError (never marks collision as scored).
    """
    if not isinstance(body, VehicleBodyGeometry) or not callable(pose_at):
        raise ValueError("explicit body and pose sampler required")
    if (not isfinite(start_s) or not isfinite(end_s)
            or start_s < 0. or end_s < start_s):
        raise ValueError("invalid sweep time interval")
    if type(max_subdivisions) is not int or not 0 <= max_subdivisions <= 20:
        raise ValueError("invalid sweep subdivision budget")
    if (type(gravity_mps2) not in (float, int)
            or not isfinite(gravity_mps2) or gravity_mps2 < 0.):
        raise ValueError("invalid ballistic gravity")
    patches, polygons = atlas
    if len(patches) != len(polygons):
        raise ValueError("road atlas shape mismatch")
    visits = 0

    def witness(state: VehicleState):
        for patch, poly in zip(patches, polygons, strict=True):
            hull = body_xz_polygon(state, body)
            if not _intersection(hull, poly):
                continue
            clearance = min(
                (v - patch.origin).dot(patch.normal) for v in _body_corners(state, body)
            )
            if clearance < -1e-9:
                raise SuspensionResolutionError(
                    "swept chassis witnessed road penetration"
                )

    def visit(lo: float, hi: float, depth: int):
        nonlocal visits
        visits += 1
        a, b = pose_at(lo), pose_at(hi)
        witness(a)
        witness(b)
        half = (lo + hi) / 2.
        mid = pose_at(half)
        witness(mid)
        delta = b.position - a.position
        q0, q1 = a.orientation, b.orientation
        dot = abs(q0.w*q1.w + q0.x*q1.x + q0.y*q1.y + q0.z*q1.z)
        angle = 2.*acos(min(1., max(0., dot)))
        # Upper bound of distance from midpoint to any swept body point
        # under linear position, shortest-arc rotation and ballistic sagitta.
        bound = .5*(delta.magnitude + angle*body.motion_radius_m)
        bound += gravity_mps2*(hi-lo)**2/8.
        envelope = _inflate(body_xz_polygon(mid, body), bound)
        corners = _body_corners(mid, body)
        clear = True
        for patch, poly in zip(patches, polygons, strict=True):
            if not _intersection(envelope, poly):
                continue
            separation = min((v-patch.origin).dot(patch.normal) for v in corners)
            if separation <= bound+1e-9:
                clear = False
                break
        if clear:
            return
        if depth == 0 or half == lo or half == hi:
            raise SuspensionResolutionError(
                "swept chassis/road clearance unresolved at subdivision limit"
            )
        visit(lo, half, depth-1)
        visit(half, hi, depth-1)

    visit(start_s, end_s, max_subdivisions)
    return visits


def certify_swept_chassis_patch_intersection_free(
    pose_at: Callable[[float], VehicleState],
    start_s:float,end_s:float,body:VehicleBodyGeometry,atlas:tuple,
    *,max_subdivisions:int=14,gravity_mps2:float=0.,
    patch_index:_BoundedPatchIndex|None=None,
)->int:
    """Check exact finite oriented PATCH surfaces rather than extrapolated infinite planes.

    The old corner-plane witness was not sufficient at overhang: the lowest
    body vertex can be *outside* a narrow road patch. We instead separate
    OBB(body) and a bounded road rectangle by all 15 SAT axes. A thickness
    of 1e-9m is numerical conservatism, NOT penetration tolerance widening.
    Each midpoint separation also bounds every point in the interval using
    linear+SLERP and specified ballistic sagitta. Unresolved stays invalid.
    """
    if not callable(pose_at) or not isinstance(body,VehicleBodyGeometry):
        raise ValueError("finite patch clearance requires typed pose and body")
    if (type(max_subdivisions) is not int or not 0<=max_subdivisions<=20
        or any(type(v) not in (int,float) or not isfinite(v) for v in
               (start_s,end_s,gravity_mps2))
        or not 0<=start_s<=end_s or gravity_mps2<0.):
        raise ValueError("invalid time/acceleration/subdivision for bounded patches")
    if len(atlas)!=2 or len(atlas[0])!=len(atlas[1]):
        raise ValueError("bounded authored atlas required")
    if patch_index is None:
        patch_index=_build_bounded_patch_index(atlas,body)
    elif (not isinstance(patch_index,_BoundedPatchIndex)
          or patch_index.source_patches is not atlas[0]):
        raise ValueError("bounded patch index does not match atlas")
    samples=0
    def has_gap_at_most(state:VehicleState,threshold:float)->bool:
        nonlocal samples
        samples+=1
        box=body.box_at(state.position,state.orientation)
        return patch_index.has_separation_at_most(box,threshold)
    def visit(left:float,right:float,budget:int)->None:
        if right<=left:
            if has_gap_at_most(pose_at(left),1e-9):
                raise SuspensionResolutionError("bounded chassis/road patch overlap")
            return
        a=pose_at(left)
        b=pose_at(right)
        middle=(left+right)/2.
        m=pose_at(middle)
        # A witnessed bounded road/vehicle intersection is physical; an
        # overhanging body corner outside the patch is NOT an intersection.
        for state in (a,m,b):
            if has_gap_at_most(state,1e-9):
                raise SuspensionResolutionError("bounded chassis witnessed physical road contact")
        angle=2.*acos(min(1.,max(0.,abs(
            a.orientation.w*b.orientation.w+a.orientation.x*b.orientation.x+
            a.orientation.y*b.orientation.y+a.orientation.z*b.orientation.z))))
        bound=.5*((b.position-a.position).magnitude+angle*body.motion_radius_m)
        bound+=gravity_mps2*(right-left)*(right-left)/8.
        if not has_gap_at_most(m,bound+1e-9):
            return
        if budget==0 or middle==left or middle==right:
            raise SuspensionResolutionError("bounded patch/body sweep unresolved at refinement limit")
        visit(left,middle,budget-1)
        visit(middle,right,budget-1)
    visit(start_s,end_s,max_subdivisions)
    return samples


def certify_first_bounded_road_contact(
    before:VehicleState,after:VehicleState,body:VehicleBodyGeometry,atlas:tuple,
    *,duration_s:float,resolution_s:float=1e-6,
    ballistic_gravity_mps2:float=0.,
):
    """First finite road/body witness on linear or exact-gravity/SLERP source.

    Returns (patch_id, interval_lower_s, interval_upper_s, number_of_checks).
    Unknown or initially overlapping geometry raises, never becomes a scored
    crash. The caller must separately certify the dynamic segment itself.
    """
    from .linear_first_body_contact import certify_first_linear_body_contact
    from .geometry.rotation import Quaternion
    from .geometry.vector import Vec3
    from .track.volume import OrientedBoxVolume
    if len(atlas)!=2 or not atlas[0]:
        raise ValueError("nonempty authored road atlas required")
    collisions=[]
    for patch in atlas[0]:
        obstacle=OrientedBoxVolume(patch.origin,
            Vec3(patch.width/2.,1e-9,patch.length/2.),
            Quaternion.from_basis(patch.right,patch.normal,patch.forward))
        hit=certify_first_linear_body_contact(before,after,body,obstacle,
               duration_s=duration_s,resolution_s=resolution_s,max_refinements=24,
               ballistic_gravity_mps2=ballistic_gravity_mps2)
        if hit is not None:
            collisions.append((hit.last_certified_clear_s,patch.id,
                               hit.first_witnessed_overlap_s,hit.gap_checks))
    if not collisions:return None
    first=min(collisions)
    if any(other[0] < first[2] and other!=first
           for other in collisions):
        raise ValueError("multiple road patches have unresolved first-contact order")
    return first[1],first[0],first[2],first[3]
