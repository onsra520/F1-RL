# `src.simulation.track`

## Purpose

Renderer-independent track/world geometry, spatial queries, and collectible runtime state.

## Responsibilities

- finite drivable surface patches
- reusable oriented volumes for solids, zones, and pickups
- deterministic aggregate world queries
- immutable collectible state transitions

## Boundary

This package reports world facts. Collision response, scoring, RL rewards, API, and rendering remain outside this package.
