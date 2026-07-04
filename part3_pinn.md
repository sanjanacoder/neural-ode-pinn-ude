# Part 3: Physics-Informed Neural Networks — The Equation Is the Teacher

*Part 3 of 5 in the series: **NeuralODE, PINN, and UDE — A Beginner's Guide to AI-Augmented Science***

**Series Navigation**
| Part | Topic |
|------|-------|
| [Part 1](part1_intro_de_ml.md) | Intro: DEs and the case for ML |
| [Part 2](part2_neural_ode.md) | Neural ODEs |
| **Part 3 (this post)** | **Physics-Informed Neural Networks (PINNs)** |
| [Part 4](part4_ude.md) | Universal Differential Equations (UDEs) |
| [Part 5](part5_comparison.md) | Comparison and when to use each |

---

## Introduction

In [Part 2](part2_neural_ode.md) we saw how a neural network can *learn* a differential equation from data, treating the ODE's right-hand side as something to be discovered. But what if you are in the opposite situation?

You have the equation. You just need a good way to solve it.

Traditional numerical solvers for PDEs require you to discretise space into a mesh — a process that becomes expensive and geometrically complicated for irregular domains. **Physics-Informed Neural Networks (PINNs)** offer an alternative: encode the equation directly into a neural network's loss function and let optimisation find the solution.

---

## Recap: The Setup

We are still working with our running predator-prey theme, but PINNs are more naturally demonstrated on a single equation. We will use **logistic growth** — a population that grows fast when small, then saturates as it approaches a carrying capacity `K`:

```
du/dt = r · u · (1 - u/K),    u(0) = u₀
```

This equation has a known analytical solution, which makes it easy to verify accuracy:

```
u(t) = K / (1 + (K/u₀ - 1)·e^{-r·t})
```

The parameters we will use: `r = 1.0`, `K = 10.0`, `u₀ = 1.0`.

---

## What Is a PINN?

### The Core Idea

A **Physics-Informed Neural Network** flips the Neural ODE's premise. Instead of data-first, it is **physics-first**.

The scenario: you know the governing equation but you want a neural network to *solve* it — giving you a smooth, differentiable function `u(t)` as the answer, rather than a table of numerical values.

The clever trick is to build the physics directly into the **loss function**. The neural network `u_NN(t; θ)` is trained so that:

1. **Physics residual**: The ODE is approximately satisfied at many collocation points `t_1, t_2, ..., t_N`
2. **Boundary/initial condition**: The network value at `t=0` matches the known starting condition

```
Loss = mean[ (du_NN/dt - f(u_NN, t))² ]    ← physics residual
     + (u_NN(0) - u₀)²                      ← boundary condition
```

Because neural networks are differentiable, `du_NN/dt` can be computed exactly using **automatic differentiation** — no finite differences needed.

### Intuition: Teaching by Constraints, Not Examples

Normally, you teach a student by showing them examples ("here is the question, here is the answer"). A PINN instead teaches the network by giving it **rules** ("any function you output must satisfy this equation"). The network then figures out a function that obeys the rules everywhere.

This means **you need almost no training data** — just the equation and boundary conditions.

### What Are Collocation Points?

Collocation points are simply a set of values of `t` scattered across the domain — say, 50 evenly spaced points between `t=0` and `t=5`. At each point, the network predicts `u(t)`, automatic differentiation computes `du/dt`, and the physics residual checks whether `du/dt = r·u·(1 - u/K)` holds.

The network is optimised until the residual is small everywhere, which forces it to converge on the true solution.

---

## Julia Code: PINN

```julia
# Neural network represents the SOLUTION u(t) — not the derivative
pinn_net = Lux.Chain(
    Lux.Dense(1, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 1)
)

ps_pinn, st_pinn = Lux.setup(rng, pinn_net)
ps_pinn = ComponentArray(ps_pinn)

# Helper: evaluate u_NN at scalar time t
function u_pinn(t_val::Number, ps)
    out, _ = pinn_net([t_val], ps, st_pinn)
    only(out)
end

# PINN loss = physics residual + boundary condition
function pinn_loss(ps, _)
    t_col = LinRange(0.0, 5.0, 50)   # collocation points

    physics_loss = mean(t_col) do t
        u_val = u_pinn(t, ps)
        # Automatic differentiation gives us du/dt exactly
        du_dt = ForwardDiff.derivative(τ -> u_pinn(τ, ps), t)
        # The ODE should hold: residual should be zero
        residual = du_dt - r_log * u_val * (1 - u_val / K_log)
        residual^2
    end

    bc_loss = (u_pinn(0.0, ps) - u0_log)^2   # initial condition

    return physics_loss + bc_loss
end
```

The remarkable thing: **no labelled `(t, u)` training pairs are used**. The only information fed to the network is the equation form and the initial condition `u(0) = 1`.

![PINN solution closely matching the exact logistic growth curve](03_pinn.png)

### What's Happening Step by Step

1. A set of collocation times `t_1, ..., t_50` is sampled (no labels attached — just time values)
2. The network predicts `u(tᵢ)` for each point
3. Automatic differentiation computes `du/dt` at each point exactly
4. The physics residual measures how far the derivative deviates from what the ODE demands
5. The boundary loss penalises any deviation from the known initial condition `u(0) = 1`
6. Gradient descent on the combined loss pushes the network toward the true solution

---

## Strengths and Weaknesses

| Feature | PINN |
|---|---|
| Equations needed? | **Yes** — the full governing equation |
| Data needed? | **No** (or very little) |
| Output | A smooth, differentiable function u(t) |
| Strengths | No mesh/grid required; handles complex PDEs; works with scattered data |
| Weaknesses | Hard to train for stiff or high-dimensional problems; slower than traditional solvers |

### When PINNs Shine

- You know the governing PDE and want a **mesh-free solution** — especially over irregular 3D geometries where meshing is expensive
- You want to **incorporate sparse measurements** directly into the solve (add a data term to the loss)
- You are solving an **inverse problem**: you know the equation structure but want to recover unknown parameters from observations
- You want the solution as a **smooth, callable function** rather than a table of values

### When PINNs Struggle

- **Stiff equations**: when the dynamics have very different time scales, standard PINN training often fails to converge
- **High dimensions**: the collocation grid grows exponentially with the number of state variables
- **Training speed**: for a well-understood PDE on a regular geometry, a classical solver will be faster and more reliable
- **Spectral bias**: neural networks tend to learn low-frequency components first, which can cause PINNs to miss sharp features in the solution

---

## PINNs vs. Traditional Solvers: A Brief Comparison

| | Classical Numerical Solver | PINN |
|---|---|---|
| Requires mesh? | Yes | No |
| Handles irregular geometry? | With difficulty | Naturally |
| Solution form | Discrete table | Smooth function |
| Can incorporate sparse data? | With extra work | Yes, directly in the loss |
| Speed (simple problems) | Much faster | Slower |
| Good for inverse problems? | Sometimes | Often yes |

---

## Summary

A PINN treats the differential equation as a *teacher* rather than something to be solved numerically. The network learns a function that satisfies the equation at many collocation points, making meshes unnecessary. The trade-off is slow and sometimes unstable training, especially for stiff or high-dimensional problems.

---

## What's Next

In [Part 4](part4_ude.md), we combine the approaches of Parts 2 and 3. What if you know *some* of the physics — but not all of it? **Universal Differential Equations (UDEs)** let you embed a neural network inside a known equation structure to fill in exactly the parts you don't know, while keeping the structure you do.

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

---

## Further Reading

- **PINNs**: Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks*, Journal of Computational Physics, 2019
- **Neural ODEs**: Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018
- **UDEs**: Rackauckas et al., *Universal Differential Equations for Scientific Machine Learning*, arXiv 2001.04385
- **SciML Docs**: [docs.sciml.ai](https://docs.sciml.ai)

---

**Series Navigation**
| Part | Topic |
|------|-------|
| [Part 1](part1_intro_de_ml.md) | Intro: DEs and the case for ML |
| [Part 2](part2_neural_ode.md) | Neural ODEs |
| **Part 3 (this post)** | **Physics-Informed Neural Networks (PINNs)** |
| [Part 4](part4_ude.md) | Universal Differential Equations (UDEs) |
| [Part 5](part5_comparison.md) | Comparison and when to use each |
