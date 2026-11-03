# Plan: 3-D extension and Earth-tide forcing of the Burridge-Knopoff model

Goals:

1. A **3-D model**: the fault becomes a dipping surface in a 3-D elastic half-space, with depth, free-surface effects, vertical deformation and realistic co-seismic gravity changes.
2. **Earth tides** (Moon and Sun) in both the 2-D and the 3-D model. Each block feels the tidal **force** (shear driving it along its slip direction, plus the direct body force) and the tidal **distension** (extension normal to the fault, which unclamps it and lowers its friction). The interseismic integration then advances in **steps of about 1 hour**.
3. A **comparison with and without tides**, for both models, of the number of events and of the co-seismic gravity changes.

The work follows the rules of the existing `IMPROVEMENT_PLAN.md`: test first (red, green, refactor), one cycle per commit, and benchmarks before any performance work. With tides off, every existing result stays bit-identical; the golden catalogue test guards this.

---

## 0. Starting point (done)

The default fault patch is now **500 km × 500 km**: 250 × 250 blocks, 2 km apart, with 25 × 25 sensors (one every 20 km, as before). The block spacing is unchanged on purpose. The springs follow from the geometry (k_c = μH, k_p = μL²/D), so changing L would change k_c/k_p = HD/L² and with it the correlation length and the magnitude statistics.

| 100 years, seed 0 | 100 × 100 (200 km), 10 × 10 sensors | 250 × 250 (500 km), 25 × 25 sensors |
|---|---|---|
| wall time | 14 s | 193 s |
| peak memory | 39 MB | 128 MB |
| events | 20,766 | 124,577 |
| largest event | Mw 8.16 | Mw 8.70 |
| b above Mc = 4.1 | 0.66 | 0.66 |

The cost grows faster than the number of blocks because every event does O(N) work: `time_to_failure` takes an argmax over all blocks, and the plate shift updates all forces. This matters for the hourly tidal stepping below (§2.3).

## 1. What the current (2-D) model is, and what limits it

| Aspect | 2-D model today | Consequence for this plan |
|---|---|---|
| geometry | blocks on the horizontal plane z = 0, slip along x only; sensors 5 km above | a horizontal plane at the free surface carries **no tidal traction** (σ_xz = σ_yz = σ_zz = 0 there), so the tide must be resolved on a nominal dipping fault orientation (§2.2) |
| loading | each block has a leaf spring to the plate and all blocks load at the same rate k_p·v_p | between events every force grows linearly, so the time to failure is exact; with tides it is not (§2.3) |
| interactions | nearest-neighbour springs, or a truncated k_c (L/r)^p kernel | the 3-D model replaces these with elastic half-space Green's functions (§3) |
| gravity | point masses shifted horizontally; potential change ΔΦ only | Δg_z (what a gravimeter measures) must be added for the comparison (§2.5) |
| time | seconds since an arbitrary t = 0 | tides need an absolute epoch (UTC/TT) |

---

## 2. Earth tides (shared module, then the 2-D model)

### 2.1 Tidal forcing: `burridge_knopoff/tides.py` (new, shared by 2-D and 3-D)

| Step | Method | Choice |
|---|---|---|
| Moon and Sun positions | Meeus (1998), *Astronomical Algorithms*: ch. 47 (Moon, truncated ELP-2000/82, ~10″) and ch. 25 (Sun, ~0.01°). Earth-fixed frame via GMST (Meeus ch. 12), with constant ΔT = TT − UT | analytic numpy code with no new dependency; vectorised over time (one year of hourly epochs per call) |
| tidal potential | degree 2 for Moon and Sun, W₂ = (GM/d)(r/d)² P₂(cos ψ); degree 3 for the Moon (~1.7 % of degree 2) | Moon and Sun are kept as separate terms so that each can be switched off |
| deformation ("distension") | spherical-Earth Love/Shida numbers, h₂ = 0.6078, l₂ = 0.0847 (IERS 2010). Surface strains (Agnew, 2007): ε_θθ = [hW + l ∂²W/∂θ²]/(ga); ε_λλ = [hW + l(cot θ ∂W/∂θ + sin⁻²θ ∂²W/∂λ²)]/(ga); ε_θλ = l[∂²W/∂θ∂λ − cot θ ∂W/∂λ]/(ga sin θ). Free surface: ε_rr = −ν/(1−ν)(ε_θθ + ε_λλ) | stress from Hooke's law with the model's μ and ν = 0.25, so σ_rr = σ_rθ = σ_rλ = 0. At 5–90 km depth the surface stress is used; the error is O(depth/R_E) < 1.5 % |
| body force ("force") | tidal acceleration ∇W on each block mass, horizontal amplitude ≈ 8·10⁻⁷ m/s² (Moon) and 4·10⁻⁷ m/s² (Sun) | switchable on its own. It is about 20 Pa equivalent, ~1 % of the stress term, and is kept so that its effect can be measured, not assumed |
| pore pressure (optional) | undrained response Δp = −B K Δε_kk, with Skempton coefficient B (default 0: drained) | partly cancels normal-stress changes when B > 0 |
| sensors | tidal potential (1 + k₂ − h₂)W and tidal gravity δ·∂W/∂r with δ = 1 + h₂ − 3k₂/2 ≈ 1.16 | evaluated analytically when needed, never stored at the 0.01-year sampling, where it would alias |

**Resolving onto the fault.** The fault has a strike φ, dip δ and rake λ (defaults: the 2011 Tohoku-oki mechanism, about 203°/10°/88°, centred at about 38.3° N, 143° E, close to the Japan Trench plate rate the model already uses). From the normal n and the slip vector s:

- shear driving slip: τ = s·σ·n, positive when it pushes slip in the tectonic sense (the same sign as k_p·v_p);
- normal: σ_n = n·σ·n, tension positive (unclamping);
- body force along slip: m (∇W · s).

The tidal Coulomb force on block i is then C_i(t) = A[τ_i(t) + μ_s σ_n,i(t)] + m ∇W·s. Strength and dynamic friction both scale with normal stress, so a static coefficient μ_s = 0.6 defines an effective normal stress σ_n,eff = F_s/(μ_s A) ≈ 16.7 MPa. The static strength then shifts by μ_s σ_n A and the dynamic friction by μ_d σ_n A, with μ_d = μ_s F_d/F_s = 0.42.

**Spatial representation.** The tide varies smoothly across a 500 km region (≈ 4.5° of arc, with ~5–10 % amplitude and phase changes). Each hour, the stress tensor is evaluated at a 5 × 5 grid of control points and least-squares fitted per component as c₀(t) + c_x(t)·x + c_y(t)·y. Mode `uniform` keeps only c₀ (the domain centre); mode `linear` is the default. A test bounds the quadratic residual at < 2 % of the amplitude.

**New config fields** (all off by default; tides off ⇒ today's code path):

| Field | Default | Meaning |
|---|---|---|
| `tides` | `False` | turn on tidal forcing (and hourly stepping) |
| `tide_bodies` | `('moon', 'sun')` | bodies included |
| `tide_stress`, `tide_body_force` | `True`, `True` | the "distension"/stress and direct-force terms, switchable separately |
| `tide_scale` | `1.0` | amplification factor for sensitivity runs (0 = hourly stepping with no tide) |
| `tide_step_hours` | `1.0` | interseismic step |
| `tide_spatial` | `'linear'` | `'uniform'` or `'linear'` |
| `epoch_utc` | `'2000-01-01T12:00:00'` | absolute time of t = 0 |
| `centre_latitude_deg`, `centre_longitude_deg` | 38.3, 143.0 | domain centre |
| `fault_strike_deg`, `fault_dip_deg`, `fault_rake_deg` | 203, 10, 88 | used to resolve the tide (2-D) and as the real geometry (3-D) |
| `friction_coefficient` | 0.6 | μ_s, giving σ_n,eff and the normal-stress coupling |
| `love_h2`, `love_l2`, `love_k2`, `poisson_ratio`, `skempton_b` | 0.6078, 0.0847, 0.30, 0.25, 0.0 | Earth response |

The 2-D block grid maps to geography in a local tangent plane: x points along the down-dip azimuth (strike + 90°, ≈ 293°: the Pacific plate direction) and y along the strike.

### 2.2 Tests for the tidal module (red first)

| Test | Check |
|---|---|
| ephemerides | Meeus worked examples 25.a (Sun) and 47.a (Moon) reproduced to their stated precision; optional cross-check against `skyfield` when installed (skipped otherwise) |
| potential | degree-2 term harmonic (finite-difference Laplacian ≈ 0); Moon/Sun ratio of maximum W₂ ≈ 2.2 at mean distances |
| spectrum | harmonic least-squares fit of a 1-year hourly series at one site recovers the main constituent ratios: M2/S2 ≈ 2.15, K1/O1 ≈ 1.41 (Cartwright & Edden, 1973), within 3 % |
| strain | areal strain equals (2h₂ − 6l₂)W/(ga); σ_rr = 0; strain amplitude 10⁻⁸–10⁻⁷ at mid-latitude |
| stress | Coulomb amplitude on the default fault in the kPa range; resolved tractions invariant to adding a pressure when μ_s = 0 |
| units/limits | `tide_scale = 0` gives exactly zero forcing; Moon-only plus Sun-only equals both (superposition) |

### 2.3 Hourly interseismic integration (2-D)

Without tides the loop jumps exactly to the next failure. With tides each block's driving force is F_i + k_p v_p t + C_i(t), and its strength is S_i − μ_s A σ_n,i(t). This is no longer linear in t, so the loop steps the interseismic period in **Δt = 1 h** (`tide_step_hours`):

1. At the start of each hour compute c(t) (cached per year chunk: 9 coefficients × 8,766 hours).
2. Check failure: max_i [D_i + C_i(t)] > 0, where D_i = F_i − S_i is maintained as today. In `uniform` mode this is O(1) per hour, because the argmax is fixed between events. In `linear` mode the check runs only over a *candidate set* (blocks within the largest spatial tidal variation of the current maximum), rebuilt after each event in O(N).
3. When a step crosses zero, locate the failure time inside the hour by bisection to 1 s. The tide is interpolated by cubic Hermite from the hourly values and their analytic derivatives (error ~(ωΔt)⁴/384 ≈ 0.02 % for M2).
4. During an event (seconds), freeze the tide at its start value. The slip kernel gets a per-block bias C_i and shifted thresholds; this changes the kernel's arguments, not its algorithm.
5. Optional exact skip-ahead: if the gap exceeds the largest possible tidal swing (2 max|C|), jump straight to the time when it no longer does. A test checks that results are identical to plain hourly stepping. At scale 1 the swing equals ~260 h of tectonic loading, so most of the time is still stepped hourly.

Cost: 100 years is 876,600 hourly steps. Expected overhead: under 1 min for `uniform`, a few minutes for `linear` on 62,500 blocks; to be measured in `benchmarks/run.py`.

**Strength redraw with tides.** The floor "new strength ≥ force held" must use the *tide-free* quantities: the base strength S_i must hold F_i + C_i − μ_s A σ_n,i at stop time. Otherwise a block could fail immediately with no extra load.

### 2.4 Tests for the 2-D tidal integration

| Test | Check |
|---|---|
| bit-identity | `tides=False` reproduces the golden catalogue exactly |
| stepping control | `tides=True, tide_scale=0`: same events as the exact-jump run, with start times within the 1 s bisection tolerance |
| single block, analytic | sinusoidal forcing A sin ωt (by injecting a test tide): the first failure time solves k_p v_p t + A sin ωt = gap (root found independently) |
| energy | the event energy balance includes the tide's work C·Σslip (frozen tide) and still closes to 10⁻³ |
| phase locking (sanity) | single block with A ω ≫ k_p v_p: failures occur while the tidal Coulomb force is increasing |
| redraw floor | with tides, no block fails at zero added load right after it stops |

### 2.5 Gravity additions (2-D)

- **Δg_z kernel**: the vertical gravity change from the slipped blocks, in a cancellation-free form like the potential. With a = |d − u_new x̂| and b = |d − u_old x̂|: 1/a³ − 1/b³ = (b − a)(a² + ab + b²)/(a³b³), and b − a = Δu(2d_x − u_new − u_old)/(a + b). Tested against a 50-digit reference, as the potential is.
- **Per-event co-seismic step** at every sensor: ΔΦ and Δg_z (after minus before). The catalogue gets summary columns (`dg_max_abs_ugal`, `dg_rms_ugal`, `dg_max_sensor`, `dphi_max_abs`). Full sensor maps are kept only for events above `gravity_map_min_magnitude` (default Mw 6.5), which bounds memory.
- **Per-event tide columns**: `tide_shear_pa`, `tide_normal_pa`, `tide_coulomb_pa`, `tide_coulomb_rate_pa_h` at the hypocentre at origin time, and `tide_phase_rad` (phase of the Coulomb stress within its semidiurnal cycle, from the Hilbert transform of the hourly series). The hourly Coulomb series at the domain centre is saved (7 MB per 100 years).

---

## 3. The 3-D model

### 3.1 Design choice

**The fault is a dipping planar surface in a 3-D elastic half-space.** Blocks (patches) on the fault keep the Burridge-Knopoff ingredients: mass, heterogeneous static/dynamic (optionally velocity-weakening) friction, inertial slip with velocity Verlet, and event-driven loading. Their elastic interactions and the deformation everywhere come from the closed-form half-space dislocation solutions of Okada (1992). Gravity and potential changes come from Okubo (1992). This is the standard 3-D generalisation of spring-block models (Ben-Zion & Rice, 1993; Ward, 2000). The 2-D model's long-range 1/r³ coupling is an approximation of exactly this.

*Alternative considered:* a volumetric lattice of nx × ny × nz blocks with springs. It was rejected because the non-frictional bulk blocks would need an equilibrium solve after each of ~10⁵ events, or their own inertial integration, and its stiffness would only approximate elasticity. The half-space formulation gives exact elasticity, a free surface and vertical motion with no bulk unknowns.

### 3.2 Geometry and loading

| Item | Default | Notes |
|---|---|---|
| footprint | 500 km (x, down-dip azimuth) × 500 km (y, along strike) | the 500 km requirement also holds in 3-D |
| free surface | z = 0 | sensors on it: 25 × 25 grid, every 20 km (ocean-bottom gravimeters offshore) |
| fault | top edge at x = 0 at 5 km depth (the 2-D sensor height), dip δ = 10° | at x = 500 km the fault is at 93 km depth |
| patches | 2 km along strike × 2 km in horizontal projection (2.03 km down-dip) | same cell size as 2-D |
| frictional (locked) zone | depth 5–50 km: 250 × 128 = **32,000 dynamic patches** | `locking_depth_m` configurable |
| creeping zone | below 50 km: 250 × 122 patches slipping steadily at v_p | not dynamic unknowns; they only load the locked zone |
| loading | **back-slip** (Savage, 1983): the stressing rate of each locked patch, τ̇_i = Σ_j K_ij v_p over creeping patches j, is constant in time | event-driven exact jumps still work without tides. A `loading='leaf'` option keeps 2-D-style leaf springs for a like-for-like test |
| slip direction | fixed rake (pure thrust, 88°) | scalar slip per patch, so the existing kernel structure carries over; variable rake is a later option |
| mass | ρ L² H, as in 2-D | sets event durations; optional radiation damping η = μ/(2c_s) |
| strength | uniform mean plus heterogeneity, as in 2-D; optional depth profile | keeps the comparison with 2-D clean |

### 3.3 Interaction kernel

- **Okada (1992) DC3D**, ported to numba and validated against the paper's check-list values (Table 2). From it: shear stress along the rake and normal stress on patch i per unit slip on patch j, K_ij (stiffness = K_ij·A).
- **Storage.** The geometry is invariant along strike, so K depends only on (Δ along-strike index, down-dip row i, down-dip row j): 499 × 128 × 128 entries per component, ≈ 65 MB in float64 for each of the shear and normal components. It is cached on disk under a hash of the geometry parameters and takes seconds to minutes to compute once.
- **Stability check**: the self-stiffness is positive and the locked-zone stiffness matrix positive definite; the test uses eigenvalues on a small grid.
- **During an event**: interactions are truncated to `interaction_range_patches` (default 5, i.e. 80 partners, with a stencil per down-dip row), using the incremental force updates the 2-D long-range mode already has.
- **At the end of each event**: an exact far-field correction applies the full kernel to the event's slip (O(N × slipped), FFT along strike if needed). If this pushes any patch over its strength, the *same* event continues from those patches, so a truncated kernel never splits one rupture into two catalogue entries.
- **Normal stress**: changes from slip on other patches enter the strength in the same way as the tide (μ_s Δσ_n A). They are off in the first version, which keeps the kernel scalar, and are added as an option once the comparison runs.

Time step during slip (estimate, with self-stiffness K_ii ≈ μ/L): ω² ≈ K_ii A/m ≈ 0.56 s⁻² gives Δt ≈ 0.08 s at 100 steps per period, about 4 times the 2-D step.

### 3.4 Gravity and surface deformation (3-D)

- **Okubo (1992)** closed-form potential and gravity changes for a rectangular shear fault in a half-space. These include surface uplift and the density change of the dilatated medium. Two quantities are stored:
  - gravity at a point fixed in space;
  - gravity on the moving ground, which differs by the free-air term −β u_z with β ≈ 0.309 mGal/m. This is what a gravimeter on the ground records.
- **Green's matrices** (sensors × locked patches) for ΔΦ, Δg and u_z: 625 × 32,000, float32, ≈ 80 MB each, computed once and cached. The co-seismic step of an event is G[:, slipped] · slip, costing sensors × slipped as in 2-D.
- **Tests**:
  - Okubo's published examples;
  - the far-field limit matches the point-source solution (Okubo, 1991);
  - uniform-uplift limit: the Bouguer-plate value (2πGρ − β) u_z;
  - superposition over patches;
  - the free-air term sign.

### 3.5 Tides in 3-D

The shared `tides.py` supplies the same hourly stress coefficients. The fault is planar, so n and s are the same for all patches and only positions (x, y, depth) differ. Resolution, the Coulomb force, the hourly stepping, the candidate set and the redraw floor are the 2-D code applied to a patch array. The depth dependence of the tidal stress at 5–50 km is below 1 % and is neglected, with a comment and a test that bounds it.

### 3.6 Code layout

A folder whose name starts with a digit cannot be a Python package, so `3d_model/` holds the plan, a separate package `bk3d` and its experiments:

```
burridge-knopoff/
├── burridge_knopoff/
│   ├── tides.py            # NEW, shared: ephemerides, potential, strain/stress, body force, resolution
│   ├── tide_analysis.py    # NEW, shared: tidal phase at events, Schuster test, rate vs stress
│   ├── model.py            # tidal bias and shifted thresholds in the slip kernel; hourly loading
│   ├── simulation.py       # hourly loop when config.tides
│   ├── gravity.py          # + vertical gravity change kernel and per-event steps
│   └── events.py           # + tide and gravity catalogue columns
└── 3d_model/
    ├── PLAN.md             # this file
    ├── pyproject.toml      # package "burridge-knopoff-3d" (bk3d), depends on burridge_knopoff
    ├── bk3d/
    │   ├── config.py       # FaultConfig3D (geometry, locking depth, loading mode, range)
    │   ├── geometry.py     # patches, local tangent plane <-> lat/lon, sensors on the surface
    │   ├── okada.py        # DC3D (numba) + check-list tests
    │   ├── okubo.py        # potential/gravity Green's functions
    │   ├── kernel.py       # strike-invariant stiffness, cache, truncated stencils, far-field correction
    │   ├── model.py        # Model3D: same interface as Model (time_to_failure, load, slip)
    │   └── simulation.py   # reuses the 2-D loop through that interface
    ├── tests/
    └── experiments/
        ├── tides_comparison.py   # the ensembles of §4
        └── results/              # CSV summaries and figures (not committed)
```

`Model3D` exposes the same interface as `Model`, so `simulation.run`, `events`, `analysis`, `output` and the ensemble runner are reused. The 2-D loop's two assumptions about the model (grid indexing and `SensorArray`) move behind small interfaces in phase 3.

---

## 4. Comparison with and without tides (2-D and 3-D)

### 4.1 Experimental design

The models are chaotic: two runs with the same seed diverge after the first event whose timing differs. Comparisons are therefore made over **ensembles of seeds**, paired by seed, so that the initial heterogeneity is shared but the trajectories are not.

| Condition | Purpose | 2-D seeds | 3-D seeds |
|---|---|---|---|
| A. no tides, exact jumps | reference (today's model) | 20 | 10 |
| B. tides on, `tide_scale = 0` | control: the hourly stepping alone must not change the statistics | 20 | 10 |
| C. Moon + Sun, scale 1 | **the main comparison** (vs A) | 20 | 10 |
| D. Moon only, Sun only, scale 1 | attribution | 10 + 10 | 5 + 5 |
| E. Moon + Sun, scale 10, 100, 1000 | dose-response: at what amplitude does each metric respond? | 10 each | 5 each |

All runs: default 500 km grid, 120 years, of which 20 are spin-up (excluded), Coulomb friction. A repeat with `--weakening 0.1 0.3` (b ≈ 1) checks that conclusions do not depend on the friction law. `run_ensemble(config, seeds, n_jobs)` already parallelises over seeds.

Budget estimates:

| Model | Per run | Total |
|---|---|---|
| 2-D | ~4 min | ~180 runs ≈ 12 CPU-hours |
| 3-D | ~10–20 min (to be measured in phase 3, initially on a 100 km × 100 km development grid) | ~85 runs ≈ 20 CPU-hours |

### 4.2 Metrics

**Number of events** (per run, then ensemble mean ± 95 % bootstrap CI, and paired differences C − A):

1. total events after the spin-up; events above Mc; events per 0.5-magnitude bin; events with Mw ≥ 7;
2. b-value (Aki/Page, as in `analysis`), Mc and local b;
3. moment release rate (N·m/yr). It is fixed by the plate rate over long windows, so it should not change: a consistency check;
4. inter-event time distribution and coefficient of variation (clustering);
5. tidal correlation of origin times:
   - Schuster test p-value on `tide_phase_rad`, overall and per magnitude class;
   - event rate vs tidal Coulomb stress level (observed / expected per stress bin, Cochran et al., 2004);
   - b-value vs tidal stress at origin (Ide et al., 2016).

   The same statistics are also computed on the no-tide catalogues against the same tidal time series. This is the null control: they must show no correlation.

**Co-seismic gravity changes** (2-D: point-mass ΔΦ and Δg_z at fixed points 5 km above the fault; 3-D: Okubo ΔΦ and Δg, fixed-point and on-ground):

1. distribution (CCDF) of the per-event maximum |Δg| over sensors, and of its RMS over sensors;
2. number of events whose co-seismic Δg exceeds detection thresholds of 0.1, 1 and 10 μGal at at least one sensor;
3. |Δg|max vs Mw scaling: slope and scatter, with vs without tides;
4. cumulative co-seismic Δg per sensor over the run (maps), and the difference map C − A;
5. co-seismic Δg relative to the tidal gravity signal at the same sensors (±~150 μGal): the fraction of steps that would stand out after an ideal tidal correction, given a stated instrument noise floor.

Absolute gravity values are **not** comparable between 2-D and 3-D, which have different physics: horizontal point-mass shifts vs half-space deformation with uplift. The with/without-tide comparison is made within each model, and the 2-D vs 3-D comparison is of *relative* changes.

### 4.3 Statistical tests

| Comparison | Test |
|---|---|
| paired event counts | Wilcoxon signed-rank, plus the bootstrap CI of the mean difference |
| magnitude and Δg distributions (pooled) | two-sample Anderson-Darling |
| b-values | the Aki σ, propagated |
| phase locking | Schuster p-value, with the no-tide catalogue as null |

Results are reported as effect sizes with CIs, not only p-values. With ~10⁵ events per run, tiny effects will be "significant".

### 4.4 Expected outcomes (hypotheses to test, not results)

- **H1, counts:** at scale 1 the tidal stress has zero mean and is ~10³ times smaller than stress drops. The long-term number of events should change by at most a few per cent and the moment rate not at all. The scale sweep (E) shows where counts start to respond.
- **H2, timing:** strong phase locking is expected even at scale 1. In a pure threshold (Coulomb) model the tidal stressing rate (~1 kPa/h) is ~65 times the tectonic rate (~15 Pa/h), so failures occur almost only while the tidal Coulomb stress is rising. Natural catalogues show only weak correlation (Cochran et al., 2004; Ide et al., 2016), because nucleation is delayed (Dieterich, 1987). This would be a known limitation of BK-type models, to be reported as such; rate-and-state nucleation is a possible follow-up.
- **H3, gravity:** per-event co-seismic Δg is set by the slip distribution. Tides change it only through the event population, so the per-event Δg distributions should differ little at scale 1. The 3-D model gives larger, more realistic steps, dominated by uplift and the free-air term.
- **2-D vs 3-D:** in 3-D the tidal shear on a 10° thrust is reduced (≈ sin δ cos δ of the horizontal stress). The 3-D normal-stress (unclamping) term should matter relatively more than in 2-D.

### 4.5 Deliverables

- `3d_model/experiments/tides_comparison.py MODEL CONDITION --seeds ... --n-jobs ...` writes per-run summaries (CSV) and catalogues (NPZ).
- `tides_comparison.py report` writes a summary table plus figures:
  - event counts with CIs per condition;
  - magnitude-frequency overlays;
  - Schuster p-value vs tide scale;
  - Δg CCDFs;
  - difference maps;
  - rate vs tidal stress.
- A results section appended to this plan (or a short report) stating H1–H3 as confirmed or not, with the numbers.

---

## 5. Phases

Each phase ends with the full fast suite green. The slow budgets are updated with ~3× margin over the measured values.

| Phase | Content | Acceptance |
|---|---|---|
| 1 | `tides.py`: ephemerides, potential, strain/stress, body force, resolution, spatial fit; config fields | tests of §2.2 |
| 2 | 2-D hourly integration, tidal bias in the slip kernel, redraw floor, Δg kernel, catalogue columns, `tide_analysis.py`, CLI `--tides [--tide-scale S] [--tide-bodies moon sun]` | tests of §2.4 and §2.5; golden catalogue unchanged; benchmark of the tidal overhead |
| 3 | `bk3d` geometry, Okada DC3D (check-list), strike-invariant kernel and cache, back-slip loading, truncated slip kernel + far-field correction, `Model3D` behind the shared interface | Okada check-list; stability; energy balance; single-patch analytic cycle (isolated patch with back-slip loading); event merging after far-field correction |
| 4 | Okubo gravity Green's functions and sensors in 3-D | tests of §3.4 |
| 5 | tides in 3-D (reuse) | tests of §2.4 rerun on `Model3D` |
| 6 | experiments of §4: development grids first (100 km), then full size | report with tables and figures |
| 7 | README sections (tides, 3-D), benchmarks, budgets | docs match measured numbers |

## 6. Risks and open points

| Risk | Mitigation |
|---|---|
| ocean-tide loading is ignored. Near trenches it can exceed the solid-Earth tidal stress several times | stated as a limitation; a later phase can add loading from a tide model (e.g. NAO.99b or FES) through Farrell (1972) Green's functions, via the same per-hour stress coefficients |
| threshold friction overstates tidal phase locking (H2) | reported explicitly; the scale sweep and the velocity-weakening repeat show how robust it is |
| 3-D cost: dense interactions, ~32,000 patches | strike-invariant storage, truncated in-event kernel with exact far-field correction, development on small grids, and a per-phase benchmark before any optimisation |
| Okubo (1992) formulas are long and error-prone | port from the paper with term-by-term tests; point-source far-field and Bouguer-limit checks |
| the 2-D tide resolution uses a nominal fault orientation for a flat model | documented; the same orientation as the 3-D fault keeps the two comparable |
| chaos makes single-run comparisons meaningless | ensembles of paired seeds, effect sizes with CIs (§4.3) |

## References

- Agnew, D. C. (2007). Earth tides. *Treatise on Geophysics*, vol. 3, 163–195.
- Ben-Zion, Y., & Rice, J. R. (1993). Earthquake failure sequences along a cellular fault zone in a three-dimensional elastic solid containing asperity and nonasperity regions. *JGR*, 98(B8), 14109–14131.
- Cartwright, D. E., & Edden, A. C. (1973). Corrected tables of tidal harmonics. *Geophys. J. R. Astr. Soc.*, 33, 253–264.
- Cochran, E. S., Vidale, J. E., & Tanaka, S. (2004). Earth tides can trigger shallow thrust fault earthquakes. *Science*, 306, 1164–1166.
- Dieterich, J. H. (1987). Nucleation and triggering of earthquake slip: effect of periodic stresses. *Tectonophysics*, 144, 127–139.
- Farrell, W. E. (1972). Deformation of the Earth by surface loads. *Rev. Geophys.*, 10(3), 761–797.
- Ide, S., Yabe, S., & Tanaka, Y. (2016). Earthquake potential revealed by tidal influence on earthquake size–frequency statistics. *Nature Geoscience*, 9, 834–837.
- Meeus, J. (1998). *Astronomical Algorithms* (2nd ed.). Willmann-Bell.
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. *BSSA*, 82(2), 1018–1040.
- Okubo, S. (1991). Potential and gravity changes raised by point dislocations. *Geophys. J. Int.*, 105, 573–586.
- Okubo, S. (1992). Gravity and potential changes due to shear and tensile faults in a half-space. *JGR*, 97(B5), 7137–7144.
- Petit, G., & Luzum, B. (2010). IERS Conventions (2010). IERS Technical Note 36.
- Savage, J. C. (1983). A dislocation model of strain accumulation and release at a subduction zone. *JGR*, 88(B6), 4984–4996.
- Schuster, A. (1897). On lunar and solar periodicities of earthquakes. *Proc. R. Soc. Lond.*, 61, 455–465.
- Tanaka, S., Ohtake, M., & Sato, H. (2002). Evidence for tidal triggering of earthquakes as revealed from statistical analysis of global data. *JGR*, 107(B10), 2211.
- Ward, S. N. (2000). San Francisco Bay Area earthquake simulations: a step toward a standard physical earthquake model. *BSSA*, 90(2), 370–386.
