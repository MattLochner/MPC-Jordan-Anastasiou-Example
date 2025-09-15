using DifferentialEquations
using DataInterpolations
using Plots
using Random
using Optim
using Optimization
using OptimizationOptimJL
using BenchmarkTools

# Original code setup (keeping your existing code structure)
t_start = 0.0
t_end = 100.0

Random.seed!(42)
mu = 0.5
phi = 0.999
lambda = 0.004

N = 1500
F_in = zeros(N)
F_in[1] = mu

for i = 2:N
    F_in[i] = phi * F_in[i-1] + (1 - phi) * mu + lambda * randn()
end

t = range(t_start, stop = t_end, length = (N))
F_in = LinearInterpolation(F_in, t)

# Optimized parameter struct with pre-allocated vectors
mutable struct OptimizedParams3{F}
    A::Float64
    cv::Float64
    t_step::Float64
    H::Int
    alpha::Float64
    fx::F
    F_in::Float64
    # Pre-allocated vectors to avoid repeated allocations
    predictions_cache::Vector{Float64}
    uv_diff_cache::Vector{Float64}
end

# Optimized MPC struct with pre-allocated time vector and cached values
mutable struct OptimizedMpc_p3
    p::OptimizedParams3
    xSP::Function
    x_est::Matrix{Float64}
    t::Vector{Float64}
    i::Int
    uv_ext::Vector{Float64}
    template::ODEProblem
    # Pre-allocated time vector for horizon
    t_horizon_cache::Vector{Float64}
    # Pre-allocated xSP values to avoid repeated function calls
    xSP_cache::Vector{Float64}
end

function xSP(t)
    return 2 + 0.5 * (t > 23) - 1.2 * (t > 60)
end

# More efficient fx function using pre-computed indices
function make_optimized_fx(uv_ext, tgrid)
    dt = tgrid[2] - tgrid[1]
    t_start = tgrid[1]
    n = length(uv_ext)
    
    return t -> begin
        idx = floor(Int, (t - t_start) / dt) + 1
        return uv_ext[clamp(idx, 1, n)]
    end
end

function MPC_ODE!(dx, x, p, t)
    p.F_in = x[2]
    dx[1] = (p.F_in - p.fx(t) * p.cv * sqrt(x[1])) / p.A
    dx[2] = 0.0
end

# Highly optimized MPC_OF function
function optimized_MPC_OF(uv, p)
    # Avoid repeated array operations by using views and pre-allocated arrays
    @views p.uv_ext[1:end-1] .= uv
    p.uv_ext[end] = uv[end]
    
    # Use cached time vector instead of creating new range
    t_start_val = p.t[p.i]
    
    # Pre-compute time points (already cached in struct)
    for j in 1:(p.p.H + 1)
        p.t_horizon_cache[j] = t_start_val + (j-1) * p.p.t_step
    end
    
    # Update fx function
    p.p.fx = make_optimized_fx(p.uv_ext, p.t_horizon_cache)
    
    # Use pre-allocated initial conditions
    x0 = [p.x_est[p.i, 2], p.x_est[p.i, 3]]
    
    # Solve ODE with optimized parameters
    t_horizon_val = p.t[p.i + p.p.H]
    problem = remake(p.template; u0 = x0, p = p.p, tspan = (t_start_val, t_horizon_val))
    
    # Use more efficient solver settings
    sol = solve(problem, Tsit5(), saveat = p.t_horizon_cache, 
               abstol=1e-6, reltol=1e-6, dense=false)
    
    # Extract predictions efficiently using pre-allocated cache
    for j in 1:length(sol.u)
        p.p.predictions_cache[j] = sol.u[j][1]
    end
    
    # Pre-compute setpoint values to avoid repeated function calls
    for j in 1:length(p.t_horizon_cache)
        p.p.xSP_cache[j] = p.xSP(p.t_horizon_cache[j])
    end
    
    # Compute objective function components efficiently
    tracking_error = 0.0
    for j in 1:length(p.p.predictions_cache)
        diff = p.p.xSP_cache[j] - p.p.predictions_cache[j]
        tracking_error += diff * diff
    end
    
    # Compute control effort penalty efficiently using pre-allocated cache
    control_penalty = 0.0
    if length(uv) > 1
        for j in 1:(length(uv)-1)
            p.p.uv_diff_cache[j] = uv[j+1] - uv[j]
            control_penalty += p.p.uv_diff_cache[j] * p.p.uv_diff_cache[j]
        end
    end
    
    return tracking_error + p.p.alpha * control_penalty
end

# Setup optimized version
t_step = 1.0
H = 5
t = t_start:t_step:t_end

initial_level = 1.8
initial_F_in = F_in(t_start)

x_est_headings = ("time", "level", "F_in", "valve position")
x_est = fill(NaN, length(t), 4)
x_est[1, 2] = initial_level

# Create optimized parameter struct with pre-allocated caches
timeframe = 0.0:t_step:5.0
uv_init = fill(0.4, H)
uv_ext = similar(uv_init, length(uv_init)+1)
eg_fx = make_optimized_fx(uv_ext, timeframe)

# Pre-allocate cache vectors
predictions_cache = Vector{Float64}(undef, H + 1)
uv_diff_cache = Vector{Float64}(undef, H)

p3_opt = OptimizedParams3(10.0, 0.5, 1.0, 5, 0.1, eg_fx, F_in(t[1]),
                         predictions_cache, uv_diff_cache)

# Create optimized MPC struct
i = 1
x_est[i, 1] = t[i]
x_est[i, 3] = F_in(t[i])

initial_level = x_est[i, 2]
initial_F_in = x_est[i, 3]
x0 = [initial_level, initial_F_in]

problem_template = ODEProblem(MPC_ODE!, x0, [0.0, 5.0], p3_opt)

# Pre-allocated caches for MPC struct
t_horizon_cache = Vector{Float64}(undef, H + 1)
xSP_cache = Vector{Float64}(undef, H + 1)

mpc_p3_opt = OptimizedMpc_p3(p3_opt, xSP, x_est, t, i, uv_ext, 
                            problem_template, t_horizon_cache, xSP_cache)

# Test the optimized function
println("Testing optimized MPC_OF function...")
checkJ_opt = optimized_MPC_OF(uv_init, mpc_p3_opt)
println("Objective value: ", checkJ_opt)

# Benchmark comparison (you can run this with your original function)
println("\nBenchmarking optimized version:")
@benchmark optimized_MPC_OF($uv_init, $mpc_p3_opt)

#https://claude.ai/chat/d3d29f92-cdb2-4845-b7da-afca1e92580d