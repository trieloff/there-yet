# There Yet — Feature Arrival Modeling (helix-cli)

## Research hypothesis

If a project uses semantic commits + semantic versioning, we can use the history of `feat:` commits between semver tags to model feature arrivals over time. If the product is approaching feature-complete, the arrival rate should decay and cumulative features should asymptotically saturate.

## Data & method

- Signal: count conventional `feat:` commits between semver tags (`vX.Y.Z`).
- Timeline: full project history.
- Models fit to cumulative features over time:
  - Gompertz (saturating S-curve)
  - NHPP Goel-Okumoto (exponential approach to asymptote)
  - NHPP Weibull (flexible decay; includes GO when shape ≈ 1)
- Fit diagnostics per model:
  - SSE (sum of squared errors)
  - RMSE (root mean squared error)
  - R2 (coefficient of determination)

## Findings (full history run)

- Date range: **2018-07-18 → 2026-01-06**
- Feature commits counted: **126**
- Model asymptotes:
  - Gompertz: **K ≈ 129.1**
  - NHPP GO: **a ≈ 138.6**
  - NHPP Weibull: **a ≈ 132.3**, **c ≈ 1.00** (essentially GO)

Interpretation: Gompertz suggests the project is very close to saturation in feature arrivals; NHPP GO leaves more remaining headroom. Weibull collapses toward GO for this data.

## How to run

Single local repo (feature arrivals from `feat:` commits):

```bash
./model_features.py --repo /Users/trieloff/Developer/helix-cli --out /tmp/helix-cli-feature-models.png
```

Multiple local repos (direct children of a root directory):

```bash
./model_features.py --repos-root ~/Developer --out-dir /tmp/feature-models
```

Multiple local repos (recursive discovery):

```bash
./model_features.py --repos-root ~/Developer --recursive --out-dir /tmp/feature-models
```

GH-only release-rate mode (no local checkout needed):

```bash
./model_features.py --remote adobe/helix-cli --out /tmp/helix-cli-release-models.png
```

GH-only mode across an owner/org:

```bash
./model_features.py --gh-owner adobe --gh-limit 200 --gh-bin /opt/homebrew/bin/gh --out-dir /tmp/release-models
```

The script writes one PNG per analyzed repo and prints model parameters plus fit diagnostics.

## Caveats

- Assumes strict conventional commits: `feat:` reliably represents a shipped feature.
- GH-only mode models release arrivals from GitHub Releases (`published_at`/`created_at`) with semver tag names (`vX.Y.Z`).
- In GH owner mode, repos you cannot access (404) are skipped.
- Releases are frequent (per `feat` or `fix`), so tag frequency is high but not the main driver of the feature signal.
- Features folded into non-`feat:` commits will be missed.
