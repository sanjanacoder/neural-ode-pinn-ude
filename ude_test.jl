##############################################################
# ude_test.jl  — standalone UDE iteration
#
# Strategy to avoid divergence in early training:
#
# STEP 1 – Pre-train NN so that the UDE starts with near-zero
#           net dynamics. We train NN to approximately cancel
#           the known linear terms:
#             NN[1](x,y) ≈ -α*x   (neutralises rabbit growth)
#             NN[2](x,y) ≈  δ*y   (neutralises fox death)
#           After pre-training: du/dt ≈ 0 → flat trajectory.
#           Loss is finite, ODE bounded.
#
# STEP 2 – Fine-tune on the full data. The NN now needs to
#           additionally learn the interaction terms so that
#           the trajectory oscillates like the true system.
##############################################################

using DifferentialEquations
using Lux, ComponentArrays
using SciMLSensitivity
using Optimization, OptimizationOptimisers
using Zygote
using Plots, Random, Statistics, LinearAlgebra

rng = Random.default_rng()
Random.seed!(rng, 42)

# ── True Lotka-Volterra data ─────────────────────────────────
function lotka_volterra!(du, u, p, _)
    x, y = u
    α, β, γ, δ = p
    du[1] = α * x - β * x * y
    du[2] = γ * x * y - δ * y
end

true_p = [1.3, 0.9, 0.8, 1.8]
u0     = [0.44, 0.47]
tspan  = (0.0, 10.0)
t_save = 0.0:0.5:10.0

prob_true = ODEProblem(lotka_volterra!, u0, tspan, true_p)
sol_true  = solve(prob_true, Tsit5(), saveat=t_save)
ode_data  = Array(sol_true) .+ 0.02 .* randn(size(Array(sol_true)))

println("Data shape: $(size(ode_data))")

# ── UDE setup ────────────────────────────────────────────────
α_known = 1.3f0
δ_known = 1.8f0
u0_f32  = Float32.(u0)

nn_ude = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)
ps_ude, st_ude = Lux.setup(rng, nn_ude)
ps_ude = ComponentArray(ps_ude)

function ude_dynamics!(du, u, p, _)
    x, y = u
    nn_out, _ = nn_ude(u, p, st_ude)
    du[1] =  α_known * x + nn_out[1]
    du[2] = -δ_known * y + nn_out[2]
end

prob_ude = ODEProblem(ude_dynamics!, u0_f32, tspan, ps_ude)

# Safe loss: avoids `nothing` gradient when ODE fails by keeping
# a tiny ps-dependent term so Zygote never returns `nothing`.
function loss_ude(ps, _)
    pred = solve(prob_ude, Tsit5(), p=ps, saveat=t_save,
                 sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    if !SciMLBase.successful_retcode(pred.retcode)
        return sum(abs2, ps) * 1f-8 + 1f6
    end
    sum(abs2, Array(pred) .- ode_data)
end

# ── STEP 1: Pre-train NN to cancel known linear terms ────────
# Target: NN(x,y) ≈ [-α*x,  +δ*y] so that du/dt ≈ 0 initially.
# This is pure supervised regression — no ODE solving needed.
function pretrain_loss(ps, _)
    total = zero(eltype(ps))
    for i in axes(ode_data, 2)
        u_i    = Float32.(ode_data[:, i])
        x, y   = u_i
        nn_out, _ = nn_ude(u_i, ps, st_ude)
        total += (nn_out[1] - (-α_known * x))^2 +
                 (nn_out[2] -  (δ_known * y))^2
    end
    total
end

println("\n── Step 1: pre-training NN to cancel linear terms ──")
optf_pre    = Optimization.OptimizationFunction(pretrain_loss, Optimization.AutoZygote())
optprob_pre = Optimization.OptimizationProblem(optf_pre, ps_ude)
res_pre     = Optimization.solve(optprob_pre, Adam(0.01), maxiters=500)
ps_warm     = res_pre.u
println("Pre-train loss: $(round(pretrain_loss(ps_warm, nothing), digits=6))")

sol_warm = solve(prob_ude, Tsit5(), p=ps_warm, saveat=t_save)
if SciMLBase.successful_retcode(sol_warm.retcode)
    println("UDE loss after pre-train: $(round(sum(abs2, Array(sol_warm) .- ode_data), digits=2))")
else
    println("WARNING: ODE still fails after pre-train")
end

# ── STEP 2: Full UDE training from warm start ────────────────
println("\n── Step 2: full UDE training ──")
iter_ude = Ref(0)
cb_ude = function (_, l)
    iter_ude[] += 1
    iter_ude[] % 500 == 0 && println("  Iter $(iter_ude[])  loss = $(round(l, digits=4))")
    return false
end

optf    = Optimization.OptimizationFunction(loss_ude, Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, ps_warm)
res_ude = Optimization.solve(optprob, Adam(0.005), maxiters=3000, callback=cb_ude)

println("Final loss: $(round(loss_ude(res_ude.u, nothing), digits=4))")

# ── Plot ─────────────────────────────────────────────────────
fine_t   = 0.0:0.1:10.0
pred_sol = solve(prob_ude, Tsit5(), p=res_ude.u, saveat=fine_t)

p = plot(sol_true.t, Array(sol_true)[1, :], label="True Rabbits",  lw=2, ls=:dash, color=:blue)
plot!(p, sol_true.t, Array(sol_true)[2, :], label="True Foxes",    lw=2, ls=:dash, color=:red)
if SciMLBase.successful_retcode(pred_sol.retcode)
    plot!(p, pred_sol.t, Array(pred_sol)[1, :], label="UDE Rabbits", lw=2, color=:blue)
    plot!(p, pred_sol.t, Array(pred_sol)[2, :], label="UDE Foxes",   lw=2, color=:red)
end
scatter!(p, collect(t_save), ode_data[1, :], label="Obs Rabbits", ms=3, alpha=0.5, color=:blue)
scatter!(p, collect(t_save), ode_data[2, :], label="Obs Foxes",   ms=3, alpha=0.5, color=:red)
title!(p, "UDE: known α,δ  +  NN learns interaction terms")
savefig(p, "ude_test.png")
println("Saved: ude_test.png")
