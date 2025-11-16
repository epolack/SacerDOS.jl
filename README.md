# SacerDOS.jl: Semiclassical Aperiodic Density of States

Code companion for [Numerical computation of the density of states of aperiodic multiscale
Schrödinger operators](https://arxiv.org/abs/2510.15369) which computes the density of
states of the semiclassical derivation.
The momentum-space version can be found [here](https://github.com/xuequan818/TBG_DFT.jl).

## Quick run

```bash
julia --project -p 6 -e 'include("test/run.jl"); run_params(:huge; n_bands_diag=1); run_params(:huge; n_bands_diag=20)'
```

## Article

Authors: Éric Cancès, Daniel Massat, Long Meng, Étienne Polack and Xue Quann

### Abstract

_Computing the electronic structure of incommensurate materials is a central challenge in condensed matter physics, requiring efficient ways to approximate spectral quantities such as the density of states (DoS). In this paper, we numerically investigate two distinct approaches for approximating the DoS of incommensurate Hamiltonians for small values of the incommensurability parameters ε (e.g., small twist angle, or small lattice mismatch): the first employs a momentum-space decomposition, and the second exploits a semiclassical expansion with respect to ε In particular, we compare these two methods using a 1D toy model. We check their consistency by comparing the asymptotic expansion terms of the DoS, and it is shown that, for full DoS, the two methods exhibit good agreement in the small ε limit, while discrepancies arise for less small ε, which indicates the importance of higher-order corrections in the semiclassical method for such regimes. We find these discrepancies to be caused by oscillations in the DoS at the semiclassical analogues of Van Hove singularities, which can be explained qualitatively, and quantitatively for ε small enough, by a semiclassical approach._

### Citation

```bibtex
@misc{cances_numerical_2025,
	title = {Numerical computation of the density of states of aperiodic multiscale {Schrödinger} operators},
	url = {http://arxiv.org/abs/2510.15369},
	doi = {10.48550/arXiv.2510.15369},
	abstract = {Computing the electronic structure of incommensurate materials is a central challenge in condensed matter physics, requiring efficient ways to approximate spectral quantities such as the density of states (DoS). In this paper, we numerically investigate two distinct approaches for approximating the DoS of incommensurate Hamiltonians for small values of the incommensurability parameters \$ε\$ (e.g., small twist angle, or small lattice mismatch): the first employs a momentum-space decomposition, and the second exploits a semiclassical expansion with respect to \$ε\$. In particular, we compare these two methods using a 1D toy model. We check their consistency by comparing the asymptotic expansion terms of the DoS, and it is shown that, for full DoS, the two methods exhibit good agreement in the small \$ε\$ limit, while discrepancies arise for less small \$ε\$, which indicates the importance of higher-order corrections in the semiclassical method for such regimes. We find these discrepancies to be caused by oscillations in the DoS at the semiclassical analogues of Van Hove singularities, which can be explained qualitatively, and quantitatively for \$ε\$ small enough, by a semiclassical approach.},
	urldate = {2025-11-16},
	publisher = {arXiv},
	author = {Cancès, Eric and Massatt, Daniel and Meng, Long and Polack, Étienne and Quan, Xue},
	month = oct,
	year = {2025},
	note = {arXiv:2510.15369},
	keywords = {Mathematical Physics},
}
```
