# ptvmc-systematic-study
Repository for material accompanying the ptvmc systematic study paper

- `ProductExamples/` directory containing the material discussing Section 2 of the paper, showing the accuracy of the various integration schemes.
	- `ProductExamples/mathematica_scripts` Contains the mathematica scripts used to generate the (arbitrary precision) coefficients for the various integrators discussed in the paper.
- `netket_ptvmc` directory containing a python package built on top of [NetKet](https://github.com/netket/netket) implementing the Infidelity optimisation strategies discussed in the manuscript as well as the ptvmc integration schemes.
	- `netket_ptvmc/examples` contains some examples using the code, both for bare infidelity optimisation (with and without the autotuning logic) and for ptvmc.