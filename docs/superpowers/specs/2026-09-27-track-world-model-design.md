# Track World Model Design

## Goal

Build the deterministic world-model layer that tells the physics/runtime what exists at a vehicle position: drivable surface geometry, solid obstacles, trigger zones, and collectible objects.

This subsystem bridges authored track data and the existing 3D physics core without coupling sandbox logic to Three.js, FastAPI, Gymnasium, PPO, or RL rewards.

## Scope

### In scope

- Finite planar road patches for flat roads, slopes, ramps, gaps, and landing surfaces.
- Static solid obstacle descriptions and overlap queries.
- Trigger zones for boost, slow, checkpoints, and finish lines.
- Collectible score pickups with explicit collected/not-collected state.
- Deterministic aggregate world queries.
- Presentation-neutral identifiers and geometry suitable for a future editor/serializer.
### Out of scope

- Collision impulse/penetration resolution against obstacles.
- Lap timing, score aggregation, or RL reward calculation.
- Runtime event buses, networking, rendering, or editor UI.
- Triangle-mesh collision or arbitrary curved surfaces in V1.

## World categories

The track world contains four distinct categories:

1. **Surface geometry** â€” road, slope, ramp, landing surfaces.
2. **Solid geometry** â€” wall, barrier, static obstacle.
3. **Trigger zones** â€” boost, slow, checkpoint, finish.
4. **Collectibles** â€” score pickups and future pickup types.

These categories stay separate because they have different semantics: surfaces support vehicles, solids block them, triggers remain reusable, and collectibles have consumable state.
## Package layout

```text
f1rl/sandbox/track/
â”œâ”€â”€ __init__.py
â”œâ”€â”€ model.py
â”œâ”€â”€ patch.py
â”œâ”€â”€ query.py
â”œâ”€â”€ volume.py
â””â”€â”€ state.py

f1rl/sandbox/entities/
â”œâ”€â”€ obstacle.py
â”œâ”€â”€ boost_pad.py
â”œâ”€â”€ slow_zone.py
â”œâ”€â”€ checkpoint.py
â”œâ”€â”€ finish_line.py
â””â”€â”€ score_pickup.py
```

`track/` owns spatial primitives, aggregate queries, and collectible state. `entities/` owns the domain objects placed into the world. `rules/` remains responsible for collision response, timing, and game/demo scoring.
## Finite planar surface patches

`PlanarSurfacePatch` is the V1 drivable-surface primitive:

```python
PlanarSurfacePatch(
    id: str,
    origin: Vec3,
    forward: Vec3,
    normal: Vec3,
    width: float,
    length: float,
)
```

`forward` and `normal` must be finite, unit length, and orthogonal. The derived right axis is `normal.cross(forward)`. `width` and `length` must be positive.

A patch query projects a world position onto the patch plane. It returns `SurfaceContact(projected_point, normal)` only when the projected point lies within the finite width/length bounds and the absolute plane distance is within the caller-supplied `max_distance`; otherwise it returns `None`.
Multiple patches may overlap at seams. `TrackWorld.surface_at(...)` selects the valid candidate with the smallest absolute plane distance; equal-distance ties are broken by lexical patch `id` so repeated queries are deterministic.

A gap requires no special object: when no patch is within bounds and `max_distance`, the surface query returns `None`, allowing the existing physics core to transition to airborne motion.

## Shared spatial volume

Solid objects, trigger zones, and collectibles use a presentation-neutral oriented box:

```python
OrientedBoxVolume(
    center: Vec3,
    half_extents: Vec3,
    orientation: Quaternion,
)
```

All half-extents must be strictly positive. `contains(point: Vec3) -> bool` transforms the point into local box space and performs an inclusive bounds test. This gives the future editor one reusable volume primitive without introducing a collision engine.
## Solid geometry

V1 represents static solids as:

```python
Obstacle(id: str, volume: OrientedBoxVolume)
```

`TrackWorld.query_point(...)` reports obstacles whose volume contains the queried point. Actual vehicle-volume collision detection and collision response remain a later `rules/collision.py` concern; the world model only owns obstacle placement and spatial lookup.

## Trigger zones

Trigger zones are persistent world objects; entering or leaving them does not remove them:

```python
BoostPad(id: str, volume: OrientedBoxVolume, acceleration_bonus: float)
SlowZone(id: str, volume: OrientedBoxVolume, drag_multiplier: float)
Checkpoint(id: str, volume: OrientedBoxVolume, order: int)
FinishLine(id: str, volume: OrientedBoxVolume)
```

`acceleration_bonus` must be positive. `drag_multiplier` must be at least `1.0`. Checkpoint order must be non-negative. Zone membership is spatial only; this subsystem does not directly mutate vehicle velocity or race progress.
Continuous zones and one-shot zones have different downstream semantics:

- Boost and slow effects are active while the vehicle remains inside the zone.
- Checkpoint and finish events should later be emitted on **entry transitions**, not once per simulation frame.

The world model therefore reports current membership only. Runtime/rules code will compare previous and current query results when event semantics are introduced.

## Collectibles

The black score card is represented by domain behavior, not presentation color:

```python
ScorePickup(
    id: str,
    volume: OrientedBoxVolume,
    points: int,
)
```

`points` must be strictly positive. The renderer may display this object as a black card, coin, token, or another visual without changing sandbox behavior.

Unlike trigger zones, a pickup is consumable. Map definition remains immutable; consumption is tracked separately through runtime world state.
`f1rl/sandbox/track/state.py` owns:

```python
TrackState(collected_pickup_ids: frozenset[str])
ScorePickupCollected(pickup_id: str, points: int)
```

`TrackWorld.query_point(..., state=...)` excludes already-collected pickups. `collect_score_pickups(pickups, state)` returns a new `TrackState` plus one `ScorePickupCollected` event per newly overlapping pickup; inputs are never mutated.

## Game score versus RL reward

`ScorePickup.points` and `ScorePickupCollected.points` are **game/demo score data**, not an RL reward.

The intended dependency is:

```text
ScorePickupCollected
       â”œâ”€â”€> sandbox.rules.scoring -> game score
       â””â”€â”€> learning.rewards     -> optional experiment-specific reward
```

The track world must never import `f1rl.learning` or assign RL reward values. This keeps reward-shaping experiments independent from player-facing scoring.
## Aggregate query API

`f1rl/sandbox/track/query.py` owns the immutable query result:

```python
TrackQuery(
    contact: SurfaceContact | None,
    obstacles: tuple[Obstacle, ...],
    boost_pads: tuple[BoostPad, ...],
    slow_zones: tuple[SlowZone, ...],
    checkpoints: tuple[Checkpoint, ...],
    finish_lines: tuple[FinishLine, ...],
    score_pickups: tuple[ScorePickup, ...],
)
```

`TrackWorld.query_point(position, max_surface_distance, state)` performs all point-based lookups in one deterministic call. Every returned collection is ordered by entity `id`; definitions must reject duplicate IDs across the whole track world.

`max_surface_distance` must be non-negative. A `None` contact is valid and represents no supporting surface near the queried position.
## Validation and determinism

- IDs must be non-empty strings and unique across patches/entities.
- All geometry values must be finite.
- Surface axes must be unit-length and mutually orthogonal within numeric tolerance.
- Width, length, and volume half-extents must be strictly positive.
- Query results never depend on input tuple order when multiple matches are equivalent; public collections are sorted by ID.
- Track definitions and entity values are immutable.
- Query and collectible-state functions never mutate their inputs.

## Dependency boundary

Allowed dependency direction:

```text
geometry/ + surface.py
        â†“
track/volume.py + track/patch.py
        â†“
entities/
        â†“
track/model.py + track/query.py + track/state.py
        â†“
rules/ and runtime/ (later)
```

`track/` and `entities/` must not import learning, runtime, API, web, Three.js, or third-party physics/collision engines.
## Verification criteria

The Track World Model is ready for world stepping/collision work when tests prove:

1. Flat, sloped, and ramp patches return the correct projected contact point and unit normal.
2. Positions outside patch bounds or beyond `max_surface_distance` return no contact.
3. Overlapping patch queries choose the nearest surface deterministically.
4. A gap between patches produces `None`, allowing physics to enter airborne state.
5. Oriented box volumes correctly include/exclude points under translation and rotation.
6. Obstacles, boost pads, slow zones, checkpoints, and finish lines are reported by spatial membership.
7. Score pickups disappear from queries after collection without mutating the map definition.
8. Pickup collection emits the configured game-score points exactly once.
9. Duplicate world IDs and invalid geometry/zone parameters fail explicitly.
10. Query ordering and repeated identical queries are deterministic.
11. Track-world code remains independent from RL reward, API, web, and renderer dependencies.
