# Documentation references and migration boundaries

The current documentation snapshot contains 164 Markdown documents in docs/.
A local relative Markdown link scan found 90 local links and 14 missing targets
in two dated research documents; the current docs/development, docs/operations,
docs/architecture and docs/README.md contain no missing relative links.

The 14 historical links point at the former f1rl/ Python source tree:
- docs/research/wp7-pv5-transition-domains-v1.md: 10 links
- docs/research/wp7-training-admission-cohort-review-2026-10-18.md: 4 links

These references are **original historical source context**, not current
developer instructions. Their original files were not silently edited.
New readers should use the source migration map at
../architecture/MIGRATION-MAP.md and CURRENT-STATE.md, not the old f1rl/ links.

Research/evidence documents, versioned manifests, source identifiers and dated
handoffs are retained in their historical locations. Documentation consolidation
means a single current index per workflow, not retroactively modifying recorded
research results. The counts are a snapshot and may change when docs are added.
