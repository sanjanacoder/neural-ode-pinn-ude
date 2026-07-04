# Part 5: Comparing Neural ODEs, PINNs, and UDEs — Which One Do You Need?

*Part 5 of 5 in the series: **NeuralODE, PINN, and UDE — A Beginner's Guide to AI-Augmented Science***

**Series Navigation**
| Part | Topic |
|------|-------|
| [Part 1](part1_intro_de_ml.md) | Intro: DEs and the case for ML |
| [Part 2](part2_neural_ode.md) | Neural ODEs |
| [Part 3](part3_pinn.md) | Physics-Informed Neural Networks (PINNs) |
| [Part 4](part4_ude.md) | Universal Differential Equations (UDEs) |
| **Part 5 (this post)** | **Comparison and when to use each** |

---

## Introduction

Over the course of this series, we have built up three distinct ways to combine machine learning with differential equations. Each started from a different premise, used a different mathematical trick, and produced a different kind of model.

Now it is time to bring them together. This final post answers the question practitioners actually face:

> *I have a scientific modelling problem. Which tool do I reach for?*

We will compare the three methods head-to-head on the same predator-prey problem, walk through the trade-offs, and give concrete guidance for different scientific scenarios.

---

## All Four Approaches on the Same Problem

Across the series, we applied four techniques to the rabbit-and-fox system. Here is a summary of what each one produced:

![Summary of all four approaches on the same problem](00_summary_comparison.png)

- **Classical ODE**: exact oscillating trajectory when parameters are known
- **Neural ODE**: recovered similar dynamics from noisy observations alone, with no equation
- **PINN**: solved the logistic growth equation without any labelled data, using only the equation and initial condition
- **UDE**: matched the full predator-prey trajectory using known birth/death rates plus a learned interaction term

Each succeeded — but under very different conditions. The choice between them depends entirely on what you have and what you want.

---

## Side-by-Side Comparison

| | Classical ODE | Neural ODE | PINN | UDE |
|---|---|---|---|---|
| **What you need** | Full equations + parameters | Only data | Full equation (no data) | Partial equations + data |
| **What you get** | Exact trajectory | Black-box dynamics | Smooth function u(t) | Interpretable hybrid model |
| **Physics** | 100% explicit | None | Enforced via loss | Partially explicit |
| **Data** | None needed | Required | Not needed | Required |
| **Extrapolation** | Excellent | Poor (black box) | Good within domain | Good (guided by physics) |
| **Interpretability** | Full | Low | Medium | High |
| **Best for** | Known, well-validated physics | Completely unknown dynamics | Known equation, want a solver | Partial knowledge situations |

---

## Decision Guide: Which Method to Use

### Start Here

```
Do you know the governing equations?
│
├─ YES, fully known
│   └─ Do you know the parameters?
│       ├─ YES → Classical ODE. No neural network needed.
│       └─ NO  → Classical ODE + parameter fitting, or PINN (inverse problem)
│
├─ PARTIALLY known
│   └─ Do you have time-series data?
│       ├─ YES → UDE. Use known structure + NN for unknown terms.
│       └─ NO  → Rethink your setup; consider collecting data
│
└─ NO equations known
    └─ Do you have dense time-series data?
        ├─ YES → Neural ODE.
        └─ NO  → You need more data or domain knowledge before ML can help.
```

---

## When to Use a Classical ODE

Use a classical ODE when you know and trust your equations. There is no reason to add a neural network if you already have the full model.

**Typical scenarios:**
- Well-established physics: Newtonian mechanics, classical thermodynamics, electrical circuits
- Validated ecological or pharmacokinetic models with measured parameters
- Any case where the model has been extensively tested and the parameters are known

Adding machine learning to a well-understood classical ODE does not improve it — it only adds complexity and reduces interpretability.

---

## When to Use a Neural ODE

Use a Neural ODE when you have **dense time-series data but no physical model** to start from.

**Typical scenarios:**
- Learning latent dynamics from sensor data when the underlying process is unknown
- Building a continuous-time recurrent model (an alternative to LSTMs for irregularly-sampled sequences)
- Modelling complex systems where the governing equations are genuinely unknown and data is plentiful

**Watch out for:**
- Sparse or noisy data — the model will overfit or fail to converge
- Extrapolation — Neural ODEs are black boxes and often diverge outside the training regime
- Interpretability requirements — if you need to explain the model to domain experts, a Neural ODE gives you nothing to show them

---

## When to Use a PINN

Use a PINN when you know the governing PDE but want a **mesh-free solution**, especially for complex geometries or when you want to incorporate sparse measurements directly into the solve.

**Typical scenarios:**
- Solving PDEs over irregular 3D geometries where mesh generation is expensive or infeasible
- **Inverse problems**: you know the equation structure and want to recover unknown parameters from sparse observations
- Situations where you want the solution as a smooth, continuously differentiable function rather than a discrete table
- Incorporating scattered experimental measurements directly into the PDE solve

**Watch out for:**
- Stiff equations — training often fails or requires specialised techniques
- High-dimensional state spaces — collocation cost scales with dimension
- Simple, regular geometries — traditional solvers will be faster and more accurate

PINNs are particularly powerful for inverse problems. If you observe that a system follows a known equation with unknown parameters, you can add a data-fitting term to the PINN loss and simultaneously solve the equation and recover the parameters.

---

## When to Use a UDE

Use a UDE when you know **part of the physics** and have data. The UDE framework is the most flexible for scientific machine learning.

**Typical scenarios:**
- You have measured some parameters (e.g., birth rates, decay constants) but not others
- You want to learn a missing constitutive law — e.g., a turbulence closure, a reaction rate, or a contact force
- Your goal is **equation discovery**: train the UDE, then apply symbolic regression to recover a formula for the unknown part
- You want better extrapolation than a Neural ODE by keeping the known physics in the model

**Watch out for:**
- You need scientific judgment about which terms to keep and which to replace
- If you get the known structure wrong (e.g., use additive when multiplicative is correct), the neural network will compensate in ways that are hard to interpret
- Still requires data — if you have no observations, a classical ODE or PINN is more appropriate

**UDEs and equation discovery**: one of the most exciting workflows in scientific ML is to train a UDE, sample the learned NN over a grid, and pass those samples into a symbolic regression tool (`DataDrivenDiffEq.jl`, SINDy, or similar). If the true unknown term is simple — e.g., `-β·x·y` — symbolic regression often recovers it exactly. This pipeline turns a data-driven model back into an interpretable scientific equation.

---

## Comparing Interpretability

Interpretability is often the deciding factor in scientific contexts, where you need to communicate results, write papers, or gain regulatory approval.

| Method | What you can explain |
|---|---|
| Classical ODE | Every parameter has a physical meaning; the equation is fully readable |
| Neural ODE | "A neural network learned the dynamics." That is all. |
| PINN | The equation is known and explicit; the NN is just a solver |
| UDE | The known part is explicit; the NN part can potentially be symbolically regressed |

For scientific publications, a UDE with subsequent symbolic regression gives you the best of both worlds: data-driven discovery with a human-readable result.

---

## Comparing Extrapolation

How far can you trust predictions beyond the training data?

| Method | Extrapolation quality |
|---|---|
| Classical ODE | Excellent — governed by physics |
| Neural ODE | Poor — no physics to constrain behaviour at unseen inputs |
| PINN | Moderate — physics-constrained, but neural networks can diverge far from training |
| UDE | Good — the known physics terms constrain the model; only the NN part is unconstrained |

This is a major practical consideration. If your model will be used to predict conditions that differ significantly from your training data — different initial conditions, longer time horizons, different parameter regimes — the Neural ODE is the riskiest choice.

---

## The SciML Ecosystem

All the Julia code in this series runs on the **SciML ecosystem** — a collection of open-source Julia packages that make scientific machine learning practical:

- `DifferentialEquations.jl` — state-of-the-art ODE/PDE solvers
- `Lux.jl` — neural network library designed for composability with differential equations
- `SciMLSensitivity.jl` — adjoint methods for backpropagating through ODE solvers
- `Optimization.jl` — unified interface for gradient-based optimisation
- `DataDrivenDiffEq.jl` — symbolic regression from UDE-learned components

The SciML ecosystem is unique in offering all these tools in a unified, composable framework. As the UDE paper demonstrates, it supports stiff ODEs, SDEs, DDEs, and is over 100× faster than equivalent PyTorch implementations on scientific models.

The key design principle is that all components — solvers, sensitivity methods, optimisers, neural networks — speak the same language and compose together without glue code. That is what makes the UDE workflow (embed physics, train, symbolically regress) possible in a few hundred lines of Julia.

---

## Conclusion

Differential equations are one of science's most powerful tools for describing how the world changes. The three methods explored in this series extend that power to situations where knowledge is incomplete:

- **Neural ODEs** say: *"I have data. Let a neural network be the equation."*
- **PINNs** say: *"I have an equation. Let a neural network be the solution."*
- **UDEs** say: *"I have some knowledge and some data. Let me combine them."*

None of these replaces classical differential equation modelling when the equations are known and trusted. But in the increasingly common situation where physics is partially known, experimental data is expensive, and interpretability matters, UDEs in particular offer a principled framework for merging the rigour of scientific modelling with the flexibility of machine learning.

The next step after mastering these tools is **equation discovery**: training a UDE, then using symbolic regression on the learned neural network component to recover a fully interpretable mathematical expression for the missing physics. That is where the promise of "scientific machine learning" truly begins to shine.

---

## Running the Code

All code in this series is in [blog_examples.jl](blog_examples.jl). To run it:

```bash
julia blog_examples.jl
```

You will need the following Julia packages (install once in the Julia REPL):

```julia
using Pkg
Pkg.add([
    "DifferentialEquations", "Lux", "ComponentArrays",
    "SciMLSensitivity", "Optimization", "OptimizationOptimisers",
    "ForwardDiff", "Zygote", "Plots", "Random", "Statistics"
])
```

The script saves four PNG plots to the working directory.

---

## Further Reading

- **Neural ODEs**: Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018
- **PINNs**: Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks*, Journal of Computational Physics, 2019
- **UDEs**: Rackauckas et al., *Universal Differential Equations for Scientific Machine Learning*, arXiv 2001.04385
- **SciML Docs**: [docs.sciml.ai](https://docs.sciml.ai)

---

**Series Navigation**
| Part | Topic |
|------|-------|
| [Part 1](part1_intro_de_ml.md) | Intro: DEs and the case for ML |
| [Part 2](part2_neural_ode.md) | Neural ODEs |
| [Part 3](part3_pinn.md) | Physics-Informed Neural Networks (PINNs) |
| [Part 4](part4_ude.md) | Universal Differential Equations (UDEs) |
| **Part 5 (this post)** | **Comparison and when to use each** |
