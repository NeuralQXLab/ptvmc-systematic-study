# ptvmc-systematic-study
This is a package for simulating the real-time dynamics of large-scale quantum spin systems using the projected time-dependent Variational Monte Carlo (p-tVMC) method. The package is based on the results of the [Neural Projected Quantum Dynamics: a systematic study](https://arxiv.org/abs/2410.10720) paper.

<!-- $h=2h_c$\ -->
<!-- ![3x3 TFIM quench $h=2h_c$](./docs/movies/animation_2hc.gif) -->

<!-- $h=h_c/10$\ -->
![3x3 TFIM quench $h=h_c/10$](./docs/movies/animation_0.1hc.gif)

## Content of the repository

This repository contains the following things:
 - `Analyics` : a set of Mathematica notebooks that can be used to re-derive the coefficients for the discretization schemes discussed in the manuscript;
 - `data` : A folder containing the data obtained from major simulations of the manuscript, divided in several per-figure folders. In particular:
    - Figure 2: The data to compare the SNR of different fidelity estimators;
    - Figure 3: The data to compare the SNR of different gradient estimators;
    - Figure 7: The states obtained by p-tVMC calculations on the 10x10 lattice. Those can be loaded by running ``nqxpack.load("path/to/file.nk")``;
 - `packages` : The codes used for our simulations, discussed below. You can install the package `ptvmc` as well as some other internal requirements by following the instructions below.
 - `examples` : Some example code to run some fast (few minutes) calculations on a 3x3 lattice, some longer-running calculations on a 6x6 lattice (few hours) and which have the same hyperparameters we used for our large 10x10 calculations.

## Installation
This package is not registered on PyPi, so you must install it directly from GitHub. To do so, you can run the following command:
```bash
pip install git+https://github.com/NeuralQXLab/ptvmc-systematic-study
```
You can also clone the repository and install the package locally by running
```bash
git clone https://github.com/NeuralQXLab/ptvmc-systematic-study
cd ptvmc-systematic-study
pip install -e .
```

## List of examples
We provide three main examples for the moment. The most complete one is the ['TFIM dynamics'](./examples/tfim_dynamics_3x3.py) example, which is used to produce the data in the animations above. 
We also provide an ['example'](./examples/state_compression.py) showing a simple code snippet to perform a single state compression. Finally we provide an ['example'](./examples/tfim_dynamics_6x6.py) very similar to the first one, but with a larger system size and with parameters more closely resembling the ones used in the paper.


## Explanation of the method
The dynamics of a closed quantum system is described by the Schrödinger equation $\ket{\psi(t+\text{d}t)} = e^{-i  H \text{d}t} \ket{\psi(t)}$, where $H$ is the Hamiltonian of the system. As $\ket{\psi(t)}$ is exponentially costly to store and manipulate, a parameterized ansatz $\ket{\psi_{\theta(t)}} \approx \ket{\psi(t)}$ with a polynomial number of parameters and a tractable query complexity is used to approximate the state at all times. The McLachlan variational principle is then used to recast the Schrödinger equation into the optimization problem
```math
\theta(t+\text{d} t) = \underset{\theta}{\text{argmin}}\,\, \mathcal{L}\left(\ket{\psi_\theta}, e^{-i  H \text{d} t} \ket{\psi_{\theta(t)}}\right),
```
where $\mathcal{L}$ is a suitable loss function quantifying the discrepancy between two quantum states. This is the starting point for the p-tVMC method. 

A practical implementation of the method requires a careful analysis of two aspects: (i) an efficient approximation of the evolutor $e^{-i  H \text{d} t}$, and (ii) a reliable way of driving the optimizations to convergence. The paper [Neural Projected Quantum Dynamics: a systematic study](https://arxiv.org/abs/2410.10720) addresses these two aspects by proposing a systematic study of the discretization schemes and optimization strategies for the p-tVMC method.

### Integration schemes
Efficient, scalable, and high-order approximations of the evolutor can be obtained in the form of a product series
```math
     e^{-i  H \text{d} t} = \prod_{k=1}^{s} \left( V_{k}^{-1}  U_{k}\right) \cdot \mathcal{D}_k + \mathcal{O}\left({\text{d} t^{o(s) + 1}}\right),
```
where the number of elements $s$ in the series is related to the order of the expansion $o = o(s)$.
The operators $U_k$ and $V_k$ are linear functions of the Hamiltonian $H$ chosen to approximate the evolutor to a desired order of accuracy.
The evolution of the parameters $\theta(t) \to \theta(t+\text{d} t)$ is thus found by solving a sequence of $s$ subsequent optimization problems, with the output of each substep serving as the input for the next. Specifically, setting $\theta(t) \equiv \theta^{(0)}$, and $\theta(t+\text{d} t) \equiv \theta^{(s)}$, we can decompose the evolution of the parameters as 
```math
    \theta^{(k)} = \underset{\theta}{\text{argmin}}\,\,\mathcal{L}\left( V_k \ket{\psi_\theta},  U_k \ket{\psi_{\theta^{(k-1)}}}\right),
```
with $0<k<s$. 

Each optimization is preceded by the application of $\mathcal{D}_k$ to the state $\ket{\psi_{\theta^{(k-1)}}}$.
$\mathcal{D}_k$ includes all those operations that can be applied exactly to the state via an analytical change of the parameters and, as such, require no optimization. For the purpose of this package, $\mathcal{D}_k$ is used to apply a diagonal operator to the state in an exact fashion and at no additional cost.

With this package, a single optimization of the above form can be easily performed as 

```python
import netket as nk
from netket.optimizer.solver import cholesky

from advanced_drivers.driver import InfidelityOptimizerNG

import optax

vs = nk.vqs.MCState(sampler, model, n_samples=1024, n_discard_per_chain=5)
ts = nk.vqs.MCState(sampler, model, n_samples=1024, n_discard_per_chain=5)

driver = InfidelityOptimizerNG(
    target_state=ts,
    optimizer=optax.sgd(1e-2),
    diag_shift=1e-6,
    variational_state=vs,
    linear_solver_fn=cholesky,
    U=U,
	V=V,
    sample_Uphi=True,
)
driver.run(n_iter=100, out=logger)
```
The `InfidelityOptimizerNG` driver is an instance of the `AdvancedVariationalDriver` class. This is an extension of the `VariationalDriver` class in NetKet, which allows for a more controlled optimization loop and more flexible callback functions. 

`U` and `V` are NetKet `DiscreteOperator`s to be defined by the user. Choosing a suitable set of $\{ V_k,  U_k\}$ is instrumental to correctly capture the dynamics of the system. The package provides a set of pre-defined integration schemes, whose properties are extensively discussed in the paper. The following table summarizes the available schemes:

<!-- <p align="center">

| Scheme | Order | Substeps | Complexity | $U_k$ | $V_k$ | $D_k$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `LPE-o` | o | o | $\mathcal{O}(N)$ | $1 + a_k \Lambda$ | $\mathbb{1}$ | $\mathbb{1}$ |
| `PPE-o` | o | o/2 | $\mathcal{O}(2N)$ | $1 + a_k \Lambda$ | $1 + b_k \Lambda$ | $\mathbb{1}$ |
| `S-LPE-o` | o | $\dagger$ | $\mathcal{O}(N)$ | $1 + a_k \Lambda_x$ | $\mathbb{1}$ | $\text{exp}(\alpha_k \Lambda_z)$ |
| `S-PPE-o` | o | $\dagger$ | $\mathcal{O}(2N)$ | $1 + a_k \Lambda_x$ | $1 + b_k \Lambda$ | $\text{exp}(\alpha_k \Lambda_z)$ |

</p> -->

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./docs/figures/table_white.png">
        <source media="(prefers-color-scheme: light)" srcset="./docs/figures/table.png">
        <img src="./docs/figures/table.png" alt="Schemes Table" width="700">
    </picture>
</p>


Here, $N$ is the number of particles in the system, $\Lambda = -i  H \text{d}t$, and $\Lambda_x$ and $\Lambda_z$ are the diagonal and off-diagonal parts of $\Lambda$, respectively. The symbol $\dagger$ indicates that the number of substeps does not have a clear simple relation to the order of the expansion. 
Semi-analytically we could determine that for `S-LPE-o` the first few substeps and orders are $(s,o) = (1,1), (2,2), (4,3)$ and for `S-PPE-o` they are $(s,o) = (1,2), (2,3), (3,4)$.

The proposed schemes are explicitly constructed to take advantage of the structure of the optimization problems, minimizing the number of optimizations required to achieve a given order of accuracy and the computational cost of each optimization. The numerical values of the expansion coefficients $a_k$, $b_k$, and $\alpha_k$ were found with the Mathematica scripts provided in the [`mathematica_scripts`](./ProductExpansions/mathematica_scripts/) directory.
In this directory we also provide python implementations of the product expansions and an easy [example](./ProductExpansions/example.py) comparing the performance of the product expansion discretization scheme to the exact evolution operator when applied to state vector simulations (no MC sampling and full description of the state).

### Optimization process
The optimization process is a crucial aspect of the p-tVMC method. As we show in the paper, to drive optimizations to convergence, the use of Natural Gradient Descent (NGD) is essential. Within this framework, the vanilla gradient descent update
```math
\theta \to \theta - \alpha \nabla \mathcal{L}(\theta)
```
is replaced by the natural gradient descent update
```math
\theta \to \theta - \alpha S^{-1} \nabla \mathcal{L}(\theta),
```
where $S$ is the Quantum Fisher Information Matrix (QFIM) or Quantum Geometric Tensor (QGT).
This preconditioning by the QGT is hardcoded in the [`InfidelityOptimizerNG`](./packages/advanced_drivers/_src/driver/ngd/driver_infidelity_ngd.py) driver. 
The most basic keyword arguments that can be used to customize the preconditioning are:
- `optimizer`: an optax optimizer object. This should always be `optax.sgd` when performing natural gradient descent.
- `diag_shift`: It is often the case that the QFIM is ill-conditioned, leading to numerical instabilities in the optimization process. To mitigate this issue, we use Tikhnov regularization, adding a small positive shift to the diagonal of the QGT as $S \to S + \lambda \mathbb{1}$. The `diag_shift` parameter controls the value of $\lambda$.
- `linear_solver_fn`: a function that takes as input a matrix and a vector and returns the solution of the linear system. This is used to solve the linear system in the natural gradient descent step. The `cholesky` solver from NetKet is one of the most efficient choices for this.

#### Nerual tangent kernel
The QGT is a square matrix of size $N_p \times N_p$, where $N_p$ is the number of parameters in the ansatz. 
The main challenge with NGD is the high computational cost of inverting the QGT in large-scale models where the number of parameters largely surpasses the number of samples $N_s$ ($N_p\gg N_s$). 
At the moment, the only method enabling the use of NGD in deep architectures without approximating the curvature matrix is the tangent kernel method. This formulation requires only the inversion of the Neural Tangent Kernel (NTK) matrix, an $N_s \times N_s$ matrix, thereby shifting the computational bottleneck from $N_p$ to $N_s$.

The [`InfidelityOptimizerNG`](./packages/advanced_drivers/_src/driver/ngd/driver_infidelity_ngd.py) driver provides an implementation of the tangent kernel method. The main keyword arguments that can be used to activate/deactivate the tangent kernel method are:
- `use_ntk`: a boolean flag to enable the use of the tangent kernel method. We recommend using this method when the number of parameters largely surpasses the number of samples.
- `on_the_fly`: a boolean flag to enable the on-the-fly computation of the NTK. This is useful when the number of samples and parameters is too large to store the Jacobian matrix in memory.


#### Autotuning (or autonomous damping)
Choosing an appropriate value for the diagonal-shift $\lambda$ of the QGT or NTK is crucial for successful optimization.  If $\lambda$ is too large, the update resembles standard gradient descent with a very small learning rate. Conversely, if $\lambda$ is too small, updates can become excessively large, particularly in low-curvature directions, possibly leading to increasing the loss rather than decreasing it.

Identifying an optimal value for $\lambda$ is a nontrivial problem and a standard approach with strong theoretical guarantees is not available. In the paper, we propose an extension of the Levenberg-Marquardt heuristic incorporating proportional control to smoothly and automatically adjust the value of $\lambda$ during the optimization process. 

While the details of the [autotuning logic](./packages/advanced_drivers/_src/callbacks/autodiagshift.py) can be found in the paper, its use in the [`InfidelityOptimizerNG`](./packages/advanced_drivers/_src/driver/ngd/driver_infidelity_ngd.py) driver is straightforward. It is implemented as a callback function that can be passed to the driver as follows:

```python
from advanced_drivers.callbacks import PI_controller_diagshift
from advanced_drivers.driver import InfidelityOptimizerNG

autotune_cb = PI_controller_diagshift(
	target=0.9,
	safety_fac=1.,
	clip_min=0.5,
	clip_max=2,
	diag_shift_min=1e-9,
	diag_shift_max=0.1,
	order=1,
	beta_1=0.9,
	beta_2=0.1,
)

driver = InfidelityOptimizerNG(
	...
)
driver.run(n_iter=100, out=logger, callback=autotune_cb)
```

## Performing the time evolution
As we have seen above, a single timestep ($t\to t + \text{d}t$) consists of a sequence of optimizations with a properly chosen set of $U_k$ and $V_k$.
In the sections above we showed how to easily perform a single infidelity optimization for a given choice of $U$ and 
$V$. 
In this package we provide a modular structure allowing to perform a single optimization, a sequence of optimizations, and the full time evolution (sequence of sequences). The structure of the packager is pictorially represented in the following figure:

<p align="center">
    <img src="./docs/figures/ptvmc_driver_schematic/ptvmc_driver.png" width="1000">

</p>

The main classes are:
- [`AbstractStateCompression`](./packages/ptvmc/_src/compression/abstract_compression.py): an abstract class defining the interface for a single state compression. This class is used to define the compression algorithm to be used in the optimization process.

    For the case of infidelity optimization a concretization of the abstract class is already provided in [`InfidelityCompression`](./packages/ptvmc/_src/compression/infidelity.py).\
To use it, one simply does
    ```python
    compression_alg = ptvmc.compression.InfidelityCompression(
        driver_class=advd.driver.InfidelityOptimizerNG,
        build_parameters={
            "diag_shift": 1e-6,
            "optimizer": optax.inject_hyperparams(optax.sgd)(learning_rate=0.05),
            "linear_solver_fn": cholesky,
            "proj_reg": None,
            "momentum": None,
            "chunk_size_bwd": None,
            "collect_quadratic_model": True,
            "use_ntk": False,
            "cv_coeff": -0.5,
            "resample_fraction": None,
            "estimator": "cmc",
        },
        run_parameters={
            "n_iter": 100,
            "callback": [
                PI_controller_diagshift(
                    target=0.9,
                    safety_fac=1.0,
                    clip_min=0.5,
                    clip_max=2,
                    diag_shift_min=1e-9,
                    diag_shift_max=0.1,
                    order=1,
                    beta_1=0.9,
                    beta_2=0.1,
                )
            ],
        },
    )
    ```

- [`AbstractDiscretization`](./packages/ptvmc/_src/solver/base.py): an abstract class defining the interface for a single timestep. This class is used to define the sequence of compressions making up a physical timestep.

    We provide a ready-to-use concretization of the general [`AbstractDiscretization`](./packages/ptvmc/_src/solver/base.py) class implementing all product expansions schemes detailed above. They can be simply called as
    ```python
    solver = ptvmc.solver.SPPE3()
    ```
    These solvers allow looping over the sequence of compressions defining a physical timestep. 
    While this class defines the structure of the discretization scheme, it does not execute the compressions.
    This can be performed by calling the `step` method of the [`Integrator`](./packages/ptvmc/_src/integrator/integrator.py) class which takes as input the compression algorithm, and the solver. 

- [`Integrator`](./packages/ptvmc/_src/integrator/integrator.py): a class that takes care of a single step of time evolution. It takes as input the compression algorithm and the solver and performs a cycle over the different stages of the discretization algorithms.


- [`PTVMCDriver`](./packages/ptvmc/_src/driver/ptvmc_driver.py): a class that takes care of the full time evolution. It takes as input the generator of the dynamics, the initial time, the solver, the compression algorithm, and the initial variational state. It then performs the full time evolution of the system looping over the single timesteps (loops over the [`Integrator`](./packages/ptvmc/_src/integrator/integrator.py) class). 

    The full time evolution can be performed as follows:
    ```python
    integration_params = ptvmc.IntegrationParameters(dt=dt)
    
    generator = -1j * H
    
    te_driver = ptvmc.PTVMCDriver(
        generator,
        0.0,
        solver=solver,
        integration_params=integration_params,
        compression_algorithm=compression_alg,
        variational_state=vs,
    )
    te_driver.run(T)
    ```
    where `H` is the Hamiltonian of the system, `dt` is the time step, `T` is the final time of the physical simulation, and `vs` is initial variational state.


## How to cite
If you use this package in your research, please cite this repository as
```
@software{
    netket_fidelity,
    author = {Gravina, Luca and Vicentini, Filippo},
    title = {ptvmc-systematic-study package},
    url = {https://github.com/NeuralQXLab/ptvmc-systematic-study},
    year = {2025}
}
```
<!-- doi = {10.5281/zenodo.8344170}, -->
<!-- version = {0.0.2}, -->
and the paper as
```
@article{
    gravina_neural_2024,
    title={Neural Projected Quantum Dynamics: a systematic study},
    author={Gravina, Luca and Savona, Vincenzo and Vicentini, Filippo},
    journal={arXiv preprint arXiv:2410.10720},
    year={2024},
    url = {https://arxiv.org/abs/2410.10720}
}
```