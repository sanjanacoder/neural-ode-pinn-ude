# NeuralODE, PINN, and UDE: A Beginner's Guide

An introductory blog series explaining the three main approaches to combining neural networks with differential equations, with working Julia code for every example.

## What's covered

| Method | Core idea | When to use |
|---|---|---|
| **Neural ODE** | Neural network *replaces* the ODE right-hand side | No physics known; abundant data |
| **PINN** | Neural network *is* the solution; ODE is the loss | Equation known; little or no data |
| **UDE** | Known physics + neural network *fills the gap* | Partial physics known; some data |

## Blog posts

| File | Topic |
|---|---|
| `neural_ode_pinn_ude_blog.md` | Full combined blog post |
| `part1_intro_de_ml.md` | What are differential equations? |
| `part2_neural_ode.md` | Neural ODEs explained |
| `part3_pinn.md` | Physics-Informed Neural Networks |
| `part4_ude.md` | Universal Differential Equations |
| `part5_comparison.md` | Side-by-side comparison and when to use each |

## Running the code

### Install Julia packages (once)

```julia
using Pkg
Pkg.add([
    "DifferentialEquations", "Lux", "ComponentArrays",
    "SciMLSensitivity", "Optimization", "OptimizationOptimisers",
    "Zygote", "Plots", "Random", "Statistics", "LinearAlgebra"
])
```

### Run the full example

```bash
julia blog_examples.jl
```

This generates five plots:

| File | Description |
|---|---|
| `01_classical_ode.png` | Lotka-Volterra (predator-prey) solved with Tsit5 |
| `02_neural_ode.png` | Neural ODE fit to noisy observations |
| `03_pinn.png` | PINN solving logistic growth without data |
| `04_ude.png` | UDE with known growth/death rates, learned interactions |
| `00_summary_comparison.png` | All four methods side by side |

### UDE development script

`ude_test.jl` is a standalone script for iterating on the UDE section. It uses a two-stage training strategy:
1. **Pre-train** the NN to cancel the known linear terms (fast, no ODE solving)
2. **Fine-tune** on the full ODE data from the well-behaved warm start

## Key implementation notes

- **Neural ODE**: uses `QuadratureAdjoint` / `InterpolatingAdjoint` from SciMLSensitivity for efficient gradient computation through the ODE solver
- **PINN**: hard-encodes the initial condition via the ansatz `u(t) = u₀ + t·net(t)` and enforces the ODE with discrete Euler step consistency — avoids nested AD issues entirely
- **UDE**: requires a two-stage warm-start to prevent the known physics from causing exponential divergence during early training

## Dependencies

- Julia 1.9+
- [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
- [Lux.jl](https://lux.csail.mit.edu/)
- [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/)
- [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)

## References

- Chen et al. (2018) — *Neural Ordinary Differential Equations*
- Raissi et al. (2019) — *Physics-Informed Neural Networks*
- Rackauckas et al. (2020) — *Universal Differential Equations for Scientific Machine Learning*
