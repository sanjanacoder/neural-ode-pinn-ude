##############################################################
# blog_examples.jl
# Complete executable code for the blog post:
# "NeuralODE, PINN, and UDE: A Beginner's Guide"
#
# Run with: julia blog_examples.jl
#
# Required packages (install once):
#   julia> using Pkg
#   julia> Pkg.add(["DifferentialEquations", "Lux", "ComponentArrays",
#                   "SciMLSensitivity", "Optimization",
#                   "OptimizationOptimisers", "ForwardDiff",
#                   "Zygote", "Plots", "Random", "Statistics"])
##############################################################

using DifferentialEquations
using Lux, ComponentArrays
using SciMLSensitivity
using Optimization, OptimizationOptimisers
using Zygote
using Plots, Random, Statistics

Random.seed!(42)

##############################################################
# SECTION 1: Classical ODE – The Predator-Prey System
#
# The Lotka-Volterra equations describe rabbits (x) and foxes (y):
#   dx/dt = α*x - β*x*y   (rabbits: births - predation)
#   dy/dt = γ*x*y - δ*y   (foxes: gains from hunting - deaths)
#
# We KNOW all four parameters α, β, γ, δ.
# A classical ODE solver gives us the solution directly.
##############################################################
println("\n========================================")
println("SECTION 1: Classical ODE Solver")
println("========================================")

function lotka_volterra!(du, u, p, _)
    x, y = u
    α, β, γ, δ = p
    du[1] = α * x - β * x * y   # rabbit dynamics
    du[2] = γ * x * y - δ * y   # fox dynamics
end

# True parameters (pretend we know these)
true_p = [1.3, 0.9, 0.8, 1.8]
u0     = [0.44, 0.47]
tspan  = (0.0, 10.0)
t_save = 0.0:0.5:10.0

prob = ODEProblem(lotka_volterra!, u0, tspan, true_p)
sol  = solve(prob, Tsit5(), saveat=t_save)

# Simulate noisy observations (what we'd measure in a real experiment)
ode_data = Array(sol) .+ 0.02 .* randn(size(Array(sol)))

p1 = plot(sol, labels=["Rabbits" "Foxes"], lw=2, linestyle=:dash,
          title="Section 1: Classical ODE (True Solution)",
          xlabel="Time", ylabel="Population")
scatter!(p1, collect(t_save), ode_data[1, :], label="Observed Rabbits", alpha=0.7, ms=4)
scatter!(p1, collect(t_save), ode_data[2, :], label="Observed Foxes",   alpha=0.7, ms=4)
savefig(p1, "01_classical_ode.png")
println("Saved: 01_classical_ode.png")

##############################################################
# SECTION 2: Neural ODE
#
# Scenario: We have the noisy observations but DO NOT know
# the governing equations at all.
#
# Key idea: Replace the unknown ODE right-hand side f(u, t)
# with a neural network NN_θ(u, t).
#
#   du/dt = NN_θ(u, t)
#
# We then train NN_θ by solving this ODE and comparing
# the solution to our observations.
##############################################################
println("\n========================================")
println("SECTION 2: Neural ODE")
println("========================================")

rng = Random.default_rng()

# Neural network replaces the unknown ODE right-hand side
nn_node = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)
ps_node, st_node = Lux.setup(rng, nn_node)
ps_node = ComponentArray(ps_node)

# The ODE is now: du/dt = NN_θ(u)
function neural_ode_rhs!(du, u, p, _)
    pred, _ = nn_node(u, p, st_node)
    du[1] = pred[1]
    du[2] = pred[2]
end

u0_f32   = Float32.(u0)   # Float32 initial condition matches network weight type
prob_node = ODEProblem(neural_ode_rhs!, u0_f32, tspan, ps_node)

# Loss: solve the Neural ODE and compare solution to data
function loss_neural_ode(ps, _)
    pred = solve(prob_node, Tsit5(), p=ps, saveat=t_save)
    !SciMLBase.successful_retcode(pred.retcode) && return Inf
    sum(abs2, Array(pred) .- ode_data)
end

iter_node = Ref(0)
cb_node = function (_, l)
    iter_node[] += 1
    iter_node[] % 500 == 0 && println("  Iter $(iter_node[])  loss = $(round(l, digits=4))")
    return false
end

optf_node    = Optimization.OptimizationFunction(loss_neural_ode, Optimization.AutoZygote())
optprob_node = Optimization.OptimizationProblem(optf_node, ps_node)
res_node     = Optimization.solve(optprob_node, Adam(0.005), maxiters=2000, callback=cb_node)

# Plot result
fine_t        = 0.0:0.1:10.0
pred_node_sol = solve(prob_node, Tsit5(), p=res_node.u, saveat=fine_t)

p2 = plot(sol.t, Array(sol)[1, :], label="True Rabbits", lw=2, ls=:dash, color=:blue)
plot!(p2, sol.t, Array(sol)[2, :], label="True Foxes",   lw=2, ls=:dash, color=:red)
plot!(p2, pred_node_sol.t, Array(pred_node_sol)[1, :], label="NeuralODE Rabbits", lw=2, color=:blue)
plot!(p2, pred_node_sol.t, Array(pred_node_sol)[2, :], label="NeuralODE Foxes",   lw=2, color=:red)
scatter!(p2, collect(t_save), ode_data[1, :], label="Observed Rabbits", ms=3, alpha=0.5, color=:blue)
scatter!(p2, collect(t_save), ode_data[2, :], label="Observed Foxes",   ms=3, alpha=0.5, color=:red)
title!(p2, "Section 2: Neural ODE – Fully Data-Driven")
xlabel!(p2, "Time"); ylabel!(p2, "Population")
savefig(p2, "02_neural_ode.png")
println("Saved: 02_neural_ode.png")

##############################################################
# SECTION 3: PINN – Physics-Informed Neural Network
#
# Scenario: We KNOW the governing equation (logistic growth)
# but want to find the solution u(t) using a neural network
# instead of a traditional ODE solver.
#
# Logistic growth: du/dt = r*u*(1 - u/K),  u(0) = u0
#
# Key idea: The ODE residual IS the training loss.
# No labelled (t, u) pairs are ever needed — the equation is enough.
#
# Implementation tricks used here:
#
#   1. Hard-encode the initial condition to avoid competing objectives:
#        u_NN(t; θ) = u0 + t * base_net(t; θ)
#      This guarantees u_NN(0) = u0 exactly.
#
#   2. Enforce the ODE via Euler step consistency on a dense grid:
#        u(t+dt) ≈ u(t) + dt * f(u(t))   for all t in [0, 5]
#      Continuous PINNs use exact du/dt from automatic differentiation;
#      this discrete variant enforces the same physics constraint.
#
#   3. All operations are vectorised (1×N matrix batches) so that
#      Zygote can compute gradients without any mutation errors.
##############################################################
println("\n========================================")
println("SECTION 3: PINN (Logistic Growth)")
println("========================================")

r_log  = 1.0f0
K_log  = 10.0f0
u0_log = 1.0f0

logistic_exact(t) = K_log / (1 + (K_log / u0_log - 1) * exp(-r_log * t))

# Base network — maps t → a correction term.
# The PINN output is: u_NN(t; θ) = u0_log + t * base_net(t; θ)
# This automatically satisfies u_NN(0) = u0_log for any θ.
pinn_base = Lux.Chain(
    Lux.Dense(1, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 1)
)
ps_pinn, st_pinn = Lux.setup(rng, pinn_base)
ps_pinn = ComponentArray(ps_pinn)

# Collocation grid — defined outside the loss so Zygote never sees a collect()
const N_pinn  = 200
const t_pinn  = reshape(Float32.(range(0.0, 5.0, N_pinn + 1)), 1, :)  # 1×201 matrix
const dt_pinn = t_pinn[2] - t_pinn[1]

# Evaluate the PINN at any 1×N time matrix
function u_pinn_eval(t_mat, ps)
    base_out, _ = pinn_base(t_mat, ps, st_pinn)
    t_mat .* base_out .+ u0_log         # hard-encodes u(0) = u0_log
end

# PINN loss: penalise ODE violations at every collocation point.
# No bc term needed — the initial condition is built into the ansatz.
function pinn_loss(ps, _)
    u       = vec(u_pinn_eval(t_pinn, ps))
    u_curr  = u[1:end-1]
    u_next  = u[2:end]
    f_u     = r_log .* u_curr .* (1 .- u_curr ./ K_log)
    euler_residuals = u_next .- u_curr .- dt_pinn .* f_u
    mean(abs2, euler_residuals)
end

iter_pinn = Ref(0)
cb_pinn = function (_, l)
    iter_pinn[] += 1
    iter_pinn[] % 1000 == 0 && println("  Iter $(iter_pinn[])  loss = $(round(l, digits=9))")
    return false
end

optf_pinn    = Optimization.OptimizationFunction(pinn_loss, Optimization.AutoZygote())
optprob_pinn = Optimization.OptimizationProblem(optf_pinn, ps_pinn)
res_pinn     = Optimization.solve(optprob_pinn, Adam(0.003), maxiters=5000, callback=cb_pinn)

t_test_pinn = reshape(Float32.(range(0.0, 5.0, 100)), 1, :)
pinn_pred   = vec(u_pinn_eval(t_test_pinn, res_pinn.u))
true_pred   = logistic_exact.(vec(t_test_pinn))

p3 = plot(vec(t_test_pinn), true_pred, label="Exact Solution", lw=2, ls=:dash, color=:black)
plot!(p3, vec(t_test_pinn), pinn_pred, label="PINN Solution",  lw=2, color=:green)
title!(p3, "Section 3: PINN – Physics as Loss (No Data Needed!)")
xlabel!(p3, "Time"); ylabel!(p3, "Population u(t)")
savefig(p3, "03_pinn.png")
println("Saved: 03_pinn.png")
println("  Max error vs exact: $(round(maximum(abs.(pinn_pred .- true_pred)), digits=4))")

##############################################################
# SECTION 4: UDE – Universal Differential Equation
#
# Scenario: We partially know the equations.
#   - We know rabbits grow at rate α and foxes die at rate δ.
#   - We do NOT know the interaction terms (predation).
#
# True system:
#   dx/dt =  α*x - β*x*y    ← interaction "-β*x*y" is UNKNOWN
#   dy/dt = -δ*y + γ*x*y    ← interaction "+γ*x*y" is UNKNOWN
#
# UDE:
#   dx/dt =  α*x + NN_θ(x, y)[1]   ← NN learns -β*x*y
#   dy/dt = -δ*y + NN_θ(x, y)[2]   ← NN learns +γ*x*y
#
# We train NN_θ to fill in the gap between what we know
# and what we observe.
##############################################################
println("\n========================================")
println("SECTION 4: UDE (Universal Differential Equation)")
println("========================================")

α_known = 1.3   # rabbit birth rate   (we know this)
δ_known = 1.8   # fox death rate      (we know this)
# β and γ (interaction strengths) are UNKNOWN → learned by NN

nn_ude = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)
ps_ude, st_ude = Lux.setup(rng, nn_ude)
ps_ude = ComponentArray(ps_ude)

# UDE: known terms + neural network for unknown interaction terms
function ude_dynamics!(du, u, p, _)
    x, y = u
    nn_out, _ = nn_ude(u, p, st_ude)
    du[1] =  α_known * x + nn_out[1]   # known birth  + learned interaction
    du[2] = -δ_known * y + nn_out[2]   # known death  + learned interaction
end

prob_ude = ODEProblem(ude_dynamics!, u0_f32, tspan, ps_ude)

function loss_ude(ps, _)
    pred = solve(prob_ude, Tsit5(), p=ps, saveat=t_save)
    !SciMLBase.successful_retcode(pred.retcode) && return Inf
    sum(abs2, Array(pred) .- ode_data)
end

iter_ude = Ref(0)
cb_ude = function (_, l)
    iter_ude[] += 1
    iter_ude[] % 500 == 0 && println("  Iter $(iter_ude[])  loss = $(round(l, digits=4))")
    return false
end

optf_ude    = Optimization.OptimizationFunction(loss_ude, Optimization.AutoZygote())
optprob_ude = Optimization.OptimizationProblem(optf_ude, ps_ude)
res_ude     = Optimization.solve(optprob_ude, Adam(0.01), maxiters=3000, callback=cb_ude)

pred_ude_sol = solve(prob_ude, Tsit5(), p=res_ude.u, saveat=fine_t)

p4 = plot(sol.t, Array(sol)[1, :], label="True Rabbits", lw=2, ls=:dash, color=:blue)
plot!(p4, sol.t, Array(sol)[2, :], label="True Foxes",   lw=2, ls=:dash, color=:red)
plot!(p4, pred_ude_sol.t, Array(pred_ude_sol)[1, :], label="UDE Rabbits", lw=2, color=:blue)
plot!(p4, pred_ude_sol.t, Array(pred_ude_sol)[2, :], label="UDE Foxes",   lw=2, color=:red)
scatter!(p4, collect(t_save), ode_data[1, :], label="Observed Rabbits", ms=3, alpha=0.5, color=:blue)
scatter!(p4, collect(t_save), ode_data[2, :], label="Observed Foxes",   ms=3, alpha=0.5, color=:red)
title!(p4, "Section 4: UDE – Known Structure + Learned Interactions")
xlabel!(p4, "Time"); ylabel!(p4, "Population")
savefig(p4, "04_ude.png")
println("Saved: 04_ude.png")

##############################################################
# SUMMARY COMPARISON PLOT
##############################################################
println("\n========================================")
println("Generating summary comparison plot...")
println("========================================")

p_summary = plot(layout=(2,2), size=(900, 700))

# Panel 1: Classical ODE
plot!(p_summary[1], sol.t, Array(sol)[1, :], label="Rabbits", lw=2, color=:blue)
plot!(p_summary[1], sol.t, Array(sol)[2, :], label="Foxes",   lw=2, color=:red)
title!(p_summary[1], "Classical ODE\n(full equations known)")

# Panel 2: Neural ODE
plot!(p_summary[2], pred_node_sol.t, Array(pred_node_sol)[1, :], label="Rabbits (NeuralODE)", lw=2, color=:blue)
plot!(p_summary[2], pred_node_sol.t, Array(pred_node_sol)[2, :], label="Foxes (NeuralODE)",   lw=2, color=:red)
plot!(p_summary[2], sol.t, Array(sol)[1, :], label="True", lw=1, color=:black, ls=:dash)
plot!(p_summary[2], sol.t, Array(sol)[2, :], label="",     lw=1, color=:black, ls=:dash)
title!(p_summary[2], "Neural ODE\n(no equations known)")

# Panel 3: PINN
plot!(p_summary[3], vec(t_test_pinn), true_pred, label="Exact", lw=2, color=:black, ls=:dash)
plot!(p_summary[3], vec(t_test_pinn), pinn_pred, label="PINN",  lw=2, color=:green)
title!(p_summary[3], "PINN\n(full equation, NN as solver)")

# Panel 4: UDE
plot!(p_summary[4], pred_ude_sol.t, Array(pred_ude_sol)[1, :], label="Rabbits (UDE)", lw=2, color=:blue)
plot!(p_summary[4], pred_ude_sol.t, Array(pred_ude_sol)[2, :], label="Foxes (UDE)",   lw=2, color=:red)
plot!(p_summary[4], sol.t, Array(sol)[1, :], label="True", lw=1, color=:black, ls=:dash)
plot!(p_summary[4], sol.t, Array(sol)[2, :], label="",     lw=1, color=:black, ls=:dash)
title!(p_summary[4], "UDE\n(partial equations + NN)")

savefig(p_summary, "00_summary_comparison.png")
println("Saved: 00_summary_comparison.png")

println("\n========================================")
println("All done! Plots saved to current directory.")
println("========================================")
