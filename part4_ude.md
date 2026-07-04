# Part 4: Universal Differential Equations — Filling in the Blanks with Data

*Part 4 of 5 in the series: **NeuralODE, PINN, and UDE — A Beginner's Guide to AI-Augmented Science***

**Series Navigation**
| Part | Topic |
|------|-------|
| [Part 1](part1_intro_de_ml.md) | Intro: DEs and the case for ML |
| [Part 2](part2_neural_ode.md) | Neural ODEs |
| [Part 3](part3_pinn.md) | Physics-Informed Neural Networks (PINNs) |
| **Part 4 (this post)** | **Universal Differential Equations (UDEs)** |
| [Part 5](part5_comparison.md) | Comparison and when to use each |

---

## Introduction

So far in this series we've seen two extremes:

- [Part 2: Neural ODEs](part2_neural_ode.md) — **data only**, no physics. The neural network learns everything from observations.
- [Part 3: PINNs](part3_pinn.md) — **physics only**, no data. The equation is embedded in the loss and the network finds the solution.

But in scientific practice, you rarely sit at either extreme. More often, you know *some* of the physics — measured in the lab, validated in prior literature — and you have *some* data. You want to use both.

**Universal Differential Equations (UDEs)**, introduced by Rackauckas et al. (2021), are designed exactly for this sweet spot.

---

## Recap: The Setup

We return to rabbits and foxes. Suppose you are a field biologist with the following situation:

- You have measured **rabbit birth rates** in the lab: α = 1.3 (you trust this)
- You have measured **fox natural death rates**: δ = 1.8 (you trust this)
- You have **field observations** of both populations over time
- But you don't know the exact form of the **predation interaction** — how many rabbits a fox catches, and how much that translates into fox reproduction

The classical Lotka-Volterra model assumes those terms are proportional to `x·y`, but maybe your ecosystem is different. You want the data to tell you.

---

## What Is a UDE?

### The Core Idea

**Universal Differential Equations** occupy the sweet spot between Neural ODEs and PINNs. Rather than discarding your knowledge (Neural ODE) or requiring the full equations (PINN), a UDE lets you **embed a neural network inside a known equation structure** to fill in the gaps.

Mathematically:

```
du/dt = known_physics(u, t) + NN_θ(u, t)
```

The neural network only needs to learn the *missing or uncertain parts* of the dynamics. Everything else is handled by your domain knowledge.

### Intuition: Filling in the Blanks

Think of writing an exam where you know most of the answer but one term is missing:

```
dx/dt = [α·x]  - [  ???  ]    ← we know rabbit births, not predation
dy/dt = [???]  - [  δ·y  ]    ← we know fox deaths, not hunting gains
```

A UDE fills in the `???` with a neural network trained on data. The result is:
- **More interpretable** than a pure Neural ODE — you kept the known structure
- **Less data-hungry** than a pure Neural ODE — the network only needs to learn the unknown part
- **More flexible** than a pure ODE — you don't need to know everything upfront

### A Biologist's Perspective

From a domain expert's standpoint, this is a very natural workflow:

1. Write down what you know from first principles or prior experiments
2. Identify the terms you are uncertain about
3. Replace those terms with a neural network
4. Train on field data
5. Optionally, pass the learned neural network through symbolic regression to recover a formula

Step 5 — equation discovery — is where UDEs truly shine. You start with partial physics, fill in the gaps with data, and then potentially end up with a *fully interpretable* equation that you can publish, test, and validate.

---

## Julia Code: UDE

```julia
α_known = 1.3   # rabbit birth rate (measured, certain)
δ_known = 1.8   # fox death rate    (measured, certain)

# Neural network learns ONLY the unknown interaction terms
nn_ude = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)
ps_ude, st_ude = Lux.setup(rng, nn_ude)
ps_ude = ComponentArray(ps_ude)

# UDE: known physics + neural network for missing terms
function ude_dynamics!(du, u, p, _)
    x, y = u
    nn_out, _ = nn_ude(u, p, st_ude)
    du[1] =  α_known * x + nn_out[1]   # known birth  + learned interaction
    du[2] = -δ_known * y + nn_out[2]   # known death  + learned interaction
end

prob_ude = ODEProblem(ude_dynamics!, u0, tspan, ps_ude)

# Same training loop as Neural ODE — just the ODE function changed
function loss_ude(ps, _)
    pred = solve(prob_ude, Tsit5(), p=ps, saveat=t_save,
                 sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
    !SciMLBase.successful_retcode(pred.retcode) && return Inf
    sum(abs2, Array(pred) .- ode_data)
end
```

Notice how similar the training loop is to the Neural ODE. The only change is that the ODE function now combines your known physics with the neural network output, rather than delegating everything to the network.

![UDE reconstructing population dynamics using only partial physics knowledge](04_ude.png)

### What's Happening Inside the UDE

- `α_known * x` — the network is never asked to learn this; it's baked in
- `nn_out[1]` — the network learns to reproduce `-β·x·y` (the predation term) purely from data
- `-δ_known * y` — again, the network doesn't touch this
- `nn_out[2]` — the network learns to reproduce `+γ·x·y` (the hunting gain term) from data

After training, if you pass `nn_out[1]` and `nn_out[2]` through a tool like `DataDrivenDiffEq.jl` (symbolic regression), you can potentially recover the symbolic formulas `-β·x·y` and `+γ·x·y` — ending up with the complete Lotka-Volterra equations from data, despite only knowing half of them upfront.

---

## Strengths and Weaknesses

| Feature | UDE |
|---|---|
| Equations needed? | **Partial** — only what you know |
| Data needed? | Yes — time-series observations |
| Output | A hybrid: known structure + learned component |
| Strengths | Best of both worlds; interpretable; data-efficient; enables equation discovery |
| Weaknesses | Requires some domain knowledge to set up the structure |

### When UDEs Shine

- You know **part of the physics** and have measured some parameters independently
- You want the **learned component to be interpretable** — perhaps via symbolic regression
- You have **limited data**: because the neural network only learns the unknown part, it needs less data than a pure Neural ODE
- You care about **extrapolation**: the known physics constrains the model's behaviour outside the training regime
- Your goal is **equation discovery**: learning a symbolic formula for a process that has never been written down

### When UDEs Struggle

- You have **no prior physics** at all — in that case, a Neural ODE is more appropriate
- You know **all the physics** — then a classical ODE is better and simpler
- Setting up the **partial structure** requires scientific judgment about which terms to keep and which to replace

---

## The Path to Equation Discovery

One of the most exciting aspects of the UDE framework is what comes next. After training:

1. Sample the learned neural network `NN_θ(x, y)` over a grid of `(x, y)` values
2. Feed those samples into a symbolic regression tool (e.g., `DataDrivenDiffEq.jl` or SINDy)
3. The tool searches for simple mathematical expressions that fit the samples
4. If the data was generated by something simple like `-β·x·y`, symbolic regression will likely recover it

This pipeline turns a data-driven model back into an interpretable scientific equation — closing the loop between machine learning and classical science.

---

## Summary

UDEs combine the structure of classical differential equations with the flexibility of neural networks. By embedding known physics into the ODE and using a neural network only for the unknown parts, you get a model that is more interpretable, more data-efficient, and better at extrapolation than a pure Neural ODE — while being more flexible than a classical ODE that requires full knowledge upfront.

---

## What's Next

In [Part 5](part5_comparison.md), we bring all three methods together. We'll compare Neural ODEs, PINNs, and UDEs side by side, discuss the trade-offs in depth, and give practical guidance for choosing the right tool for your problem — including a look at the broader SciML ecosystem that makes all of this possible in Julia.

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

- **UDEs**: Rackauckas et al., *Universal Differential Equations for Scientific Machine Learning*, arXiv 2001.04385
- **Neural ODEs**: Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018
- **PINNs**: Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks*, Journal of Computational Physics, 2019
- **SciML Docs**: [docs.sciml.ai](https://docs.sciml.ai)

---

**Series Navigation**
| Part | Topic |
|------|-------|
| [Part 1](part1_intro_de_ml.md) | Intro: DEs and the case for ML |
| [Part 2](part2_neural_ode.md) | Neural ODEs |
| [Part 3](part3_pinn.md) | Physics-Informed Neural Networks (PINNs) |
| **Part 4 (this post)** | **Universal Differential Equations (UDEs)** |
| [Part 5](part5_comparison.md) | Comparison and when to use each |
