# Part 2: Neural ODEs — Learning Dynamics Purely from Data

*Part 2 of 5 in the series: **NeuralODE, PINN, and UDE — A Beginner's Guide to AI-Augmented Science***

**Series Navigation**
| Part | Topic |
|------|-------|
| [Part 1](part1_intro_de_ml.md) | Intro: DEs and the case for ML |
| **Part 2 (this post)** | **Neural ODEs** |
| [Part 3](part3_pinn.md) | Physics-Informed Neural Networks (PINNs) |
| [Part 4](part4_ude.md) | Universal Differential Equations (UDEs) |
| [Part 5](part5_comparison.md) | Comparison and when to use each |

---

## Introduction

In [Part 1](part1_intro_de_ml.md) we built up the classical picture: you write down a differential equation that captures the physics of your system, hand it to a solver, and out comes the trajectory. That works beautifully — as long as you know the equations.

But what if you don't? What if you have dense time-series observations of a dynamical system, but no physical model to write down?

That is exactly the problem a **Neural ODE** is designed to solve.

---

## Recap: The Setup

We are tracking rabbit (`x`) and fox (`y`) populations over time. In [Part 1](part1_intro_de_ml.md) we used the Lotka-Volterra equations with known parameters. Now suppose we have only **observations** — noisy measurements of `x(t)` and `y(t)` at discrete times — and we have no idea what the governing equations look like.

Can a machine learning model learn the dynamics?

---

## What Is a Neural ODE?

### The Core Idea

A Neural ODE asks: *what if we replace the right-hand side of the ODE with a neural network?*

In a classical ODE, you write the governing equations by hand:

```
du/dt = f(u, t)       ← f is hand-crafted from domain knowledge
```

In a **Neural ODE**, you have data (time-series observations) but no known equations. So you replace `f` with a neural network:

```
du/dt = NN_θ(u, t)    ← NN_θ is a neural network with parameters θ
```

You then **train** the neural network by:
1. Solving the ODE numerically (using a standard ODE solver)
2. Comparing the solution to your observed data
3. Backpropagating the error through the ODE solver to update the network weights

The key insight (from Chen et al., NeurIPS 2018) is that you can treat the ODE solver as a differentiable computation and propagate gradients through it using a technique called the **adjoint method**.

### Intuition: ResNets as Discrete ODEs

Think of a deep residual neural network (ResNet). Each residual layer computes:

```
h_{n+1} = h_n + f(h_n)
```

This looks exactly like Euler's method for integrating an ODE:

```
y(t + Δt) = y(t) + Δt · f(y(t), t)
```

A Neural ODE takes this analogy to its logical conclusion: instead of a fixed number of discrete layers, use a *continuous* flow of hidden states governed by a neural network ODE. This gives you:
- **Fewer parameters** — depth becomes continuous, not a fixed count
- **Adaptive computation** — the solver takes more steps where the dynamics are complex
- **Natural handling of irregular time series** — observations don't need to be evenly spaced

### The Adjoint Method: How Gradients Flow Through a Solver

Training requires computing `d(Loss)/dθ`, the gradient of the loss with respect to the neural network weights. The challenge is that the loss depends on the ODE solution, which is computed by a black-box numerical solver.

Storing the full computation graph through a long ODE solve would use prohibitive amounts of memory. The adjoint method solves this by running a *second ODE backwards in time* to recover the gradients — using only the solver output, not its internal trajectory. This gives O(1) memory cost regardless of the number of solver steps.

---

## Julia Code: Neural ODE

```julia
using Lux, ComponentArrays, SciMLSensitivity
using Optimization, OptimizationOptimisers

# Neural network that replaces the unknown ODE right-hand side
nn_node = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

rng = Random.default_rng()
ps_node, st_node = Lux.setup(rng, nn_node)
ps_node = ComponentArray(ps_node)

# The ODE is now: du/dt = NN_θ(u)
function neural_ode_rhs!(du, u, p, _)
    pred, _ = nn_node(u, p, st_node)
    du[1] = pred[1]
    du[2] = pred[2]
end

prob_node = ODEProblem(neural_ode_rhs!, u0, tspan, ps_node)

# Loss: solve the Neural ODE and compare to observations
function loss_neural_ode(ps, _)
    pred = solve(prob_node, Tsit5(), p=ps, saveat=t_save,
                 sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
    !SciMLBase.successful_retcode(pred.retcode) && return Inf
    sum(abs2, Array(pred) .- ode_data)
end

# Train with the Adam optimiser
optf    = Optimization.OptimizationFunction(loss_neural_ode, Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, ps_node)
result  = Optimization.solve(optprob, Adam(0.01), maxiters=500)
```

After training, the neural network has learned to mimic the Lotka-Volterra dynamics purely from the noisy observations — without ever being told the equations.

![Neural ODE matching the true predator-prey dynamics from data alone](02_neural_ode.png)

### What's Happening Layer by Layer

- The **input** to the network is the current state `[x, y]` (rabbit and fox populations)
- The **output** is the predicted derivative `[dx/dt, dy/dt]`
- The network has no explicit knowledge that these represent populations — it just maps states to rates of change
- The ODE solver integrates those rates forward in time to produce a trajectory

The loss compares this trajectory to the observed data, and gradient descent adjusts the network weights until the trajectory matches.

---

## Strengths and Weaknesses

| Feature | Neural ODE |
|---|---|
| Equations needed? | **No** — learned entirely from data |
| Data needed? | Yes — time-series observations |
| Output | A black-box continuous dynamical system |
| Strengths | Works when physics is completely unknown |
| Weaknesses | May not extrapolate well; no interpretability |

### When Neural ODEs Shine

- You have **dense, accurate time-series data** from sensors or experiments
- You want a **continuous-time model** — e.g., for interpolating between observations or for handling irregular sampling
- You are building a **latent dynamics model** in a deep learning pipeline

### When Neural ODEs Struggle

- **Extrapolation**: the learned dynamics are data-driven and can diverge wildly outside the training distribution
- **Interpretability**: you get a neural network — there is no equation to read, interpret, or share with domain experts
- **Data efficiency**: if observations are sparse or noisy, the network has little to go on and will likely fail

If you have some physical knowledge you could incorporate, a UDE (Part 4) will almost always outperform a pure Neural ODE in those settings.

---

## Summary

A Neural ODE replaces the right-hand side of a differential equation with a neural network, then trains the network by differentiating through an ODE solver. It is the right tool when you have time-series data and no physical model. The main trade-off is a black-box model that may not generalise beyond the training regime.

---

## What's Next

In [Part 3](part3_pinn.md), we flip the premise entirely. Instead of having data but no equations, we'll consider the case where you know the governing equations but want a smarter way to solve them — without a computational mesh. That's the domain of **Physics-Informed Neural Networks (PINNs)**.

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
- **SciMLSensitivity.jl** — adjoint methods for Julia: [docs.sciml.ai/SciMLSensitivity](https://docs.sciml.ai/SciMLSensitivity)
- **UDEs**: Rackauckas et al., *Universal Differential Equations for Scientific Machine Learning*, arXiv 2001.04385
- **SciML Docs**: [docs.sciml.ai](https://docs.sciml.ai)

---

**Series Navigation**
| Part | Topic |
|------|-------|
| [Part 1](part1_intro_de_ml.md) | Intro: DEs and the case for ML |
| **Part 2 (this post)** | **Neural ODEs** |
| [Part 3](part3_pinn.md) | Physics-Informed Neural Networks (PINNs) |
| [Part 4](part4_ude.md) | Universal Differential Equations (UDEs) |
| [Part 5](part5_comparison.md) | Comparison and when to use each |
