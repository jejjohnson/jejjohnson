<h2 align="center">J. Emmanuel Johnson</h2>
<p align="center">
Research scientist and scientific software engineer · differentiable Earth observation in JAX<br>
UNEP IMEO · Methane Alert and Response System (MARS) · Valencia, Spain
</p>

<p align="center">
<a href="https://jejjohnson.github.io/geostack/"><b>GeoStack docs</b></a> ·
<a href="https://jejjohnson.github.io/research_notebook/">Research notebook</a> ·
<a href="https://linkedin.com/in/jejjohnson">LinkedIn</a> ·
<a href="https://twitter.com/jejjohnson">X / Twitter</a>
</p>

---

I build the software that turns satellite and in-situ observations into numbers someone can act on, each with an honest error bar.
The physics is differentiable, the inference is Bayesian, and every step is checked against a trusted oracle.
All of it lives in one open-source JAX stack: **GeoStack**.

<p align="center">
<img src="assets/hero.svg" width="100%" alt="From observations to decisions: sensors feed GeoStack, which feeds differentiable physics (GeoModels) and inference (GeoML) on top of GeoNumerics; outputs are decisions with uncertainty, and gradients flow back end to end. Methane and sea surface height are two case studies.">
</p>

## The stack

Earth observation code usually means gluing together a reader, a regridder, a physics model, an inverse method and an ML model that were never designed to meet.
GeoStack is the opposite: small libraries with one contract each, laid out along the data-maturity axis from raw instrument output (L0) to decisions (L5).
The **science axis** says what each level means.
The **compute** between two levels is where physics and inference live.
The **engineering axis** underneath says how the data is stored, carried and composed at every level.

<p align="center">
<img src="assets/geostack_hero.svg" width="100%" alt="GeoStack along the data-maturity axis: the science axis from L0 unstructured observations to L5 derived products, with methane and sea-surface-height case studies running along it; the compute step between each pair of levels, all following the same oracle-first ladder on GeoNumerics; and the engineering axis of store, index, read, carry, patch, transform and compose, with the carrier switching from geoarray to xarray at L3.">
</p>

<details>
<summary><b>How the packages layer</b>: imports only point down, and CI proves the graph has no cycles</summary>
<p align="center">
<img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/overview/tier_stack.svg" width="100%" alt="GeoStack's four layers: GeoModels on top, GeoML below it, and GeoNumerics and GeoStack side by side at the base, all on JAX, Equinox, lineax, NumPyro and xarray.">
</p>
</details>

| Layer | Library | What it does |
| --- | --- | --- |
| **GeoStack** | [geotoolz](https://github.com/jejjohnson/geotoolz) | Raster operators, STAC catalog and patching; `geoarray`, one geo-carrier for NumPy **and** JAX, with a differentiable affine transform |
| | [xrtoolz](https://github.com/jejjohnson/xrtoolz) | The same operator model on xarray: indices, temporal statistics, CF output |
| | [pipekit](https://github.com/jejjohnson/pipekit) | Composable `Operator` and pipeline protocols shared by every package |
| **GeoNumerics** | [gaussx](https://github.com/jejjohnson/gaussx) | Structured linear algebra: Kalman filters, GP primitives, block operators on lineax |
| | [finitevolX](https://github.com/jejjohnson/finitevolX) | Finite-volume operators on staggered grids |
| | [spectraldiffx](https://github.com/jejjohnson/spectraldiffx) | Spectral derivatives and solvers |
| **GeoML** | [vardax](https://github.com/jejjohnson/vardax) | Variational data assimilation: 3D-Var, 4D-Var, optimal estimation |
| | [filterax](https://github.com/jejjohnson/filterax) | Ensemble and Kalman filtering |
| | [pyrox](https://github.com/jejjohnson/pyrox) · [kernellib](https://github.com/jejjohnson/kernellib) | Gaussian processes and probabilistic NNs for NumPyro; kernels |
| | [geonnax](https://github.com/jejjohnson/geonnax) | Neural fields and basis functions for geoscience |
| | [gauss_flows](https://github.com/jejjohnson/gauss_flows) | Normalizing flows and SurVAE |
| | [optax_bayes](https://github.com/jejjohnson/optax_bayes) | The Bayesian learning rule as Optax optimizers |
| | [xtremax](https://github.com/jejjohnson/xtremax) | Extreme values with JAX and NumPyro |
| **GeoModels** | [radtrax](https://github.com/jejjohnson/radtrax) | Differentiable radiative transfer, matched filters, retrievals |
| | [plumax](https://github.com/jejjohnson/plumax) | Plume physics: masks, integrated mass enhancement, flux, emission rates |
| | [somax](https://github.com/jejjohnson/somax) | Simple ocean models: quasi-geostrophic, shallow water |

## The method

Every problem in the stack follows the same ladder.
Step 2, a slow but trusted physics-based inversion, is the oracle that every faster learned method must match before it ships.

<p align="center">
<img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/overview/six_steps.svg" width="100%" alt="Six steps: a simple generative model; model-based inference as the trusted oracle; an emulator; emulator-based inference; amortized inference; and an outer loop that improves each step against the one before it.">
</p>

## Case studies

### Methane: from radiance to emission rate

Detecting and quantifying methane plumes from EMIT, Sentinel-2 and Landsat 8/9.
A layered Beer–Lambert forward model in [radtrax](https://github.com/jejjohnson/radtrax) gives the matched-filter target and every Jacobian by autodiff.
The pipeline then runs matched filter, plume vetting and optimal estimation through [vardax](https://github.com/jejjohnson/vardax).
[plumax](https://github.com/jejjohnson/plumax) turns the column enhancement into an emission rate in kg/h.

<details>
<summary><b>The full methane walkthrough</b>, from scene catalog to served model</summary>
<p align="center">
<img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/overview/methane_walkthrough.svg" width="100%" alt="Eight steps: catalog EMIT scenes, retrieve methane, match with TCCON, grid and gap-fill with a GP, assimilate hourly with an ensemble Kalman filter and a plume model, then index, evaluate and serve.">
</p>
</details>

### Sea surface height: from altimetry tracks to gap-free maps

Mapping the ocean's mesoscale from sparse along-track altimetry.
**[OceanBench: The Sea Surface Height Edition](https://arxiv.org/abs/2309.15599)** (NeurIPS 2023, Datasets and Benchmarks) is a benchmark and toolkit for SSH interpolation ([docs](https://jejjohnson.github.io/oceanbench), [code](https://github.com/jejjohnson/oceanbench)).
The methods range from optimal interpolation and neural fields ([ml4ssh](https://github.com/jejjohnson/ml4ssh), [knerf](https://github.com/jejjohnson/knerf), [functa4eo](https://github.com/jejjohnson/functa4eo)) to quasi-geostrophic models in [somax](https://github.com/jejjohnson/somax) inside 4D-Var.

## How it came together

GeoStack is not a fresh start; it is ten years of research code, rebuilt so the pieces fit.

<p align="center">
<img src="assets/lineage.svg" width="100%" alt="Swimlane timeline 2017–2026: manipy, kernellib, rbig, ml4ssh and jbayesevt became GeoML; the 2023 ocean solvers became GeoModels and GeoNumerics; OceanBench became the benchmarks; plumax and radtrax carry the methane work.">
</p>

Older work still worth a look: [rbig](https://github.com/jejjohnson/rbig) (Rotation-Based Iterative Gaussianization and information measures), [jbayesevt](https://github.com/jejjohnson/jbayesevt) (Bayesian extreme value models), [manipy](https://github.com/jejjohnson/manipy) (semi-supervised manifold learning), [keras-fairkl](https://github.com/jejjohnson/keras-fairkl) (fair kernel learning), and my [research notebook](https://jejjohnson.github.io/research_notebook/) of notes and tutorials.

## How I work

- **JAX all the way down.** Equinox modules, with `jit`, `vmap` and `grad` through the physics, and float64 where the maths needs it.
- **Tested against the truth.** Every refactor merges behind numeric gates: parity with the code it replaces, autodiff against finite differences, closed-form oracles.
- **Small packages, explicit seams.** Heavy dependencies sit behind optional extras, and everything is MIT-licensed.

**Interests:** atmospheric retrievals · data assimilation · Gaussian processes and uncertainty quantification · neural fields · extremes · ML for dynamical systems.

## Built on

GeoStack stands on the shoulders of these open-source projects, and the packages above are designed to compose with them rather than replace them.

**JAX numerics** · [JAX](https://github.com/jax-ml/jax) ·
[Equinox](https://github.com/patrick-kidger/equinox) ·
[jaxtyping](https://github.com/patrick-kidger/jaxtyping) ·
[lineax](https://github.com/patrick-kidger/lineax) ·
[diffrax](https://github.com/patrick-kidger/diffrax) ·
[optimistix](https://github.com/patrick-kidger/optimistix) ·
[matfree](https://github.com/pnkraemer/matfree) ·
[finitediffx](https://github.com/ASEM000/finitediffx) ·
[interpax](https://github.com/f0uriest/interpax) ·
[einx](https://github.com/fferflo/einx) ·
[einops](https://github.com/arogozhnikov/einops)

**Probabilistic ML and training** · [NumPyro](https://github.com/pyro-ppl/numpyro) ·
[Optax](https://github.com/google-deepmind/optax) ·
[FlowJAX](https://github.com/danielward27/flowjax) ·
[BlackJAX](https://github.com/blackjax-devs/blackjax) ·
[Grain](https://github.com/google/grain) ·
[Orbax](https://github.com/google/orbax)

**Geospatial data** · [xarray](https://github.com/pydata/xarray) ·
[georeader](https://github.com/spaceml-org/georeader) (GeoTensor) ·
[rasterio](https://github.com/rasterio/rasterio) ·
[rioxarray](https://github.com/corteva/rioxarray) ·
[GeoPandas](https://github.com/geopandas/geopandas) ·
[Shapely](https://github.com/shapely/shapely) ·
[pyproj](https://github.com/pyproj4/pyproj) ·
[xvec](https://github.com/xarray-contrib/xvec) ·
[regionmask](https://github.com/regionmask/regionmask) ·
[xrft](https://github.com/xgcm/xrft)

**Cloud-native storage and catalogs** · [obstore](https://github.com/developmentseed/obstore) ·
[async-geotiff](https://github.com/developmentseed/async-geotiff) ·
[Zarr](https://github.com/zarr-developers/zarr-python) ·
[fsspec](https://github.com/fsspec/filesystem_spec) ·
[Dask](https://github.com/dask/dask) ·
[DuckDB](https://github.com/duckdb/duckdb) ·
[PyArrow](https://github.com/apache/arrow) ·
[pystac](https://github.com/stac-utils/pystac) ·
[pystac-client](https://github.com/stac-utils/pystac-client)

**Data access** · [earthaccess](https://github.com/nsidc/earthaccess) ·
[Planetary Computer](https://github.com/microsoft/planetary-computer-sdk-for-python) ·
[Earth Engine](https://github.com/google/earthengine-api) ·
[Copernicus Marine](https://github.com/mercator-ocean/copernicus-marine-toolbox) ·
[CDS API](https://github.com/ecmwf/cdsapi) ·
[HITRAN](https://hitran.org/hapi/)

**The scientific Python base** · [NumPy](https://github.com/numpy/numpy) ·
[SciPy](https://github.com/scipy/scipy) ·
[scikit-learn](https://github.com/scikit-learn/scikit-learn) ·
[scikit-image](https://github.com/scikit-image/scikit-image) ·
[pandas](https://github.com/pandas-dev/pandas)

<sub>A special thanks to Patrick Kidger's JAX ecosystem: Equinox, lineax, diffrax, optimistix and jaxtyping are the backbone of nearly every package here.</sub>

<p align="center"><sub>Figures generated with GeoStack's own figure code · Python · JAX · Equinox · NumPyro · xarray</sub></p>
