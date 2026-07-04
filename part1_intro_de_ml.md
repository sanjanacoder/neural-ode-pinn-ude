# Part 1: Solving Scientific Problems with Differential Equations and Machine Learning

*Part 1 of 5 in the series: **NeuralODE, PINN, and UDE — A Beginner's Guide to AI-Augmented Science***

**Series Navigation**
| Part | Topic |
|------|-------|
| **Part 1 (this post)** | **Intro: DEs and the case for ML** |
| [Part 2](part2_neural_ode.md) | Neural ODEs |
| [Part 3](part3_pinn.md) | Physics-Informed Neural Networks (PINNs) |
| [Part 4](part4_ude.md) | Universal Differential Equations (UDEs) |
| [Part 5](part5_comparison.md) | Comparison and when to use each |

---

*"A model is worth a thousand datasets."*
— Rackauckas et al., Universal Differential Equations for Scientific Machine Learning (2021)

---

## Introduction

Imagine you are a wildlife biologist tracking rabbit and fox populations in a forest. You take measurements every few months. You want to predict future populations, understand what drives their cycles, and perhaps figure out some hidden interaction you haven't modelled yet.

For centuries, scientists have used **differential equations** to describe how things change over time — from population dynamics and fluid flow to planetary orbits and electrical circuits. These equations are powerful, but they require you to already *know* the governing laws. What happens when you only know part of the physics, or none at all?

This is where three modern techniques — **Neural ODEs**, **PINNs**, and **UDEs** — enter the picture. They all sit at the intersection of machine learning and differential equations, but they answer different versions of the same question:

> *How do we model a dynamical system when we don't have complete knowledge?*

In this five-part series we will:
1. Build up intuition for differential equations and why ML becomes necessary (this post)
2. Explore **Neural ODEs** — letting a neural network be the equation
3. Explore **PINNs** — letting a neural network be the solution
4. Explore **UDEs** — combining partial physics knowledge with data
5. Compare all three and give practical guidance on choosing between them

All examples use the same running scenario — **rabbits and foxes** — so you can see exactly how each technique approaches the same problem from a different angle.

---

## What Are Differential Equations?

Before diving into the AI-augmented methods, let's build up the foundation.

A **differential equation** describes how a quantity *changes* over time (or space), rather than describing the quantity itself directly.

### Intuitive Example: A Cooling Cup of Coffee

Suppose you make a coffee at 90°C and leave it in a 20°C room. The coffee cools faster when it is much hotter than the room, and more slowly as it approaches room temperature. Newton's Law of Cooling captures this:

```
dT/dt = -k * (T - T_room)
```

This reads: *"the rate of change of temperature equals some constant times the difference between the coffee's temperature and the room."*

You don't need to know the temperature at every future instant — you just need to know *how fast it changes*, and a solver figures out the rest.

### A Richer Example: Rabbits and Foxes

The **Lotka-Volterra** equations describe how a predator and prey population interact:

```
dx/dt =  α·x - β·x·y     (rabbits: births - predation losses)
dy/dt =  γ·x·y - δ·y     (foxes:   gains from hunting - natural deaths)
```

Here `x` is the rabbit population, `y` is the fox population, and α, β, γ, δ are constants describing birth rates, death rates, and how effectively foxes hunt rabbits.

The four terms have simple biological meanings:
- `α·x` — rabbits reproduce at rate α
- `β·x·y` — rabbits get eaten (more encounters when both populations are large)
- `γ·x·y` — foxes gain from eating rabbits
- `δ·y` — foxes die at rate δ

**Solving** this system means finding the trajectories `x(t)` and `y(t)` that satisfy both equations simultaneously, starting from some initial populations.

### Solving ODEs in Julia

Julia's `DifferentialEquations.jl` makes this straightforward:

```julia
using DifferentialEquations, Plots

function lotka_volterra!(du, u, p, _)
    x, y = u
    α, β, γ, δ = p
    du[1] = α * x - β * x * y   # rabbit dynamics
    du[2] = γ * x * y - δ * y   # fox dynamics
end

true_p = [1.3, 0.9, 0.8, 1.8]
u0     = [0.44, 0.47]
tspan  = (0.0, 10.0)

prob = ODEProblem(lotka_volterra!, u0, tspan, true_p)
sol  = solve(prob, Tsit5(), saveat=0.0:0.5:10.0)

plot(sol, labels=["Rabbits" "Foxes"], lw=2)
```

This gives you the oscillating predator-prey cycle you might expect: rabbit populations rise, which feeds more foxes, which then drive the rabbits down, causing fox populations to decline, and the cycle repeats.

![Classical ODE result showing oscillating rabbit and fox populations](01_classical_ode.png)

This works perfectly when you know **all** the parameters α, β, γ, δ. But what if you don't?

---

## When Physics Alone Isn't Enough

The classical ODE approach breaks down in three common real-world scenarios:

### 1. You don't know the equations at all

Perhaps you are studying a complex biological process — say, the interaction of five competing species, or how a new drug distributes through the body. The underlying mechanisms are too complicated, or simply unknown. You have sensor data but no governing equations.

**Enter Neural ODEs**: let a neural network learn the equations directly from data.

### 2. You know the equations but finding the solution is hard

Solving a partial differential equation (PDE) over a complex 3D geometry traditionally requires discretising space into a mesh — a computationally expensive and geometry-sensitive process. You know exactly what the physics says, but you need a smarter solver.

**Enter PINNs**: embed the equation into a neural network's loss function and let it find the solution directly, without a mesh.

### 3. You know some of the physics, but not all of it

You are a biologist who has measured rabbit birth rates in the lab. You know rabbits reproduce at rate α = 1.3. But the predation interaction is harder to measure directly. You have field data, and you want to use it alongside your known physics.

**Enter UDEs**: embed your known physics into the ODE and use a neural network to fill in the gaps.

---

## The Landscape at a Glance

Here is how the four approaches (including classical ODEs) compare before we go deeper:

| | Classical ODE | Neural ODE | PINN | UDE |
|---|---|---|---|---|
| **What you need** | Full equations + parameters | Only data | Full equation (no data) | Partial equations + data |
| **What you get** | Exact trajectory | Black-box dynamics | Smooth function u(t) | Interpretable hybrid model |
| **Physics** | 100% explicit | None | Enforced via loss | Partially explicit |
| **Data** | None needed | Required | Not needed | Required |

We'll unpack each column in the next three posts. For now, the key takeaway is this:

> Classical ODEs are the gold standard — use them when you can. The ML-augmented methods exist for when your knowledge or data is incomplete.

---

## What's Next

In [Part 2](part2_neural_ode.md), we'll dive into **Neural ODEs**: how replacing the right-hand side of a differential equation with a neural network lets you learn dynamics purely from data, and how gradients can be propagated through an ODE solver using the adjoint method.

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

- **Neural ODEs**: Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018
- **PINNs**: Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks*, Journal of Computational Physics, 2019
- **UDEs**: Rackauckas et al., *Universal Differential Equations for Scientific Machine Learning*, arXiv 2001.04385
- **SciML Docs**: [docs.sciml.ai](https://docs.sciml.ai)

---

**Series Navigation**
| Part | Topic |
|------|-------|
| **Part 1 (this post)** | **Intro: DEs and the case for ML** |
| [Part 2](part2_neural_ode.md) | Neural ODEs |
| [Part 3](part3_pinn.md) | Physics-Informed Neural Networks (PINNs) |
| [Part 4](part4_ude.md) | Universal Differential Equations (UDEs) |
| [Part 5](part5_comparison.md) | Comparison and when to use each |
