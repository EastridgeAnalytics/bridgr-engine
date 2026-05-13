# Bridgr Python Package — Dependency Licenses

All dependencies are permissively licensed (MIT, Apache 2.0, or BSD).
No GPL, AGPL, or BSL dependencies exist in the tree.

**Audited:** 2026-05-12

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| real_ladybug | 0.15.3 | MIT | LadybugDB embedded graph engine |
| pyarrow | 23.0.1 | Apache-2.0 | Arrow columnar format / interop |
| pandas | 2.3.3 | BSD-3-Clause | DataFrame output (optional) |
| PyYAML | 6.0.3 | MIT | YAML frontmatter parsing (migration) |

## Engine (C++ / upstream)

| Component | License | Notes |
|-----------|---------|-------|
| LadybugDB (Kuzu fork) | MIT | Core graph engine |
| Apache Arrow (C++) | Apache-2.0 | Columnar format |
| antlr4 | BSD-3-Clause | Cypher parser |

## Compliance

R7 requirement: every dependency must be MIT, Apache 2.0, or BSD.
Status: **COMPLIANT** — zero copyleft dependencies.
