using DifferentialEquations
using DataInterpolations
using Plots
using Random
using Optim
using Optimization
using OptimizationOptimJL
using BenchmarkTools

#Specify time for which the process will start and end:
t_start = 0.0
t_end = 100.0

#######
#Create randomized F_in 
######

Random.seed!(42)
mu  = 0.4
phi = 0.999
lambda = 0.004

N = 1500 #this is just the number of points, not the time

F_in = zeros(N); 
F_in[1] = mu;

for i = 2:N
    F_in[i] = phi * F_in[i-1] + (1 - phi) * mu + lambda * randn()
end

t = range(t_start, stop = t_end, length = (N))
#display(t)
#display(F_in)
F_in = LinearInterpolation(F_in, t)
plot(F_in, xlab = "time (min)", ylab = "In Flow (m\$^3\$/min)")

mutable struct Params3{F}
    A::Float64           #m^2, buffer tank cross-sectional area
    cv::Float64          #m^2.5/min, valve constant
    t_step::Float64      #min, time steps between controller actions
    H::Int               #number of time steps in controller prediction horizon
    alpha::Float64       #weighting parameter controlling the manipulated variable movemement
    fx::F
    F_in::Float64
end

t_step = 1.0
H = 5

function xSP(t)
    return 2 + 0.5 * (t > 23) - 1.2 * (t > 60)
end

#This is my reference point for time, t_step is the time at which controller actions are taken. It is assumed measurements are only taken at the time points that control is implemented.
t = t_start:t_step:t_end

#Initialize ground truth, measurements, and state estimates:

initial_level = 1.8 #m
initial_F_in = F_in(t_start)

#x_est should actually be called stored_values, it is where all process inputs and states are stored for each time step.

x_est_headings = ("time", "level", "F_in", "valve position")
x_est = fill(NaN, length(t), 4)
x_est[1, 2] = initial_level

mutable struct Mpc_p3
    p::Params3
    xSP::Function
    x_est::Matrix{Float64}
    t::Vector{Float64}
    i::Int
    uv_ext::Vector{Float64}
    template::ODEProblem
end

function MPC_ODE!(dx, x, p, t)
    p.F_in = x[2]
    dx[1] = (p.F_in - p.fx(t) * p.cv * sqrt(x[1])) / p.A
    dx[2] = 0.0
end

function make_fx(uv_ext, tgrid)
    return t -> begin
        idx = floor(Int,(t-tgrid[1]) / (tgrid[2] - tgrid[1])) + 1
        return uv_ext[clamp(idx, 1, length(uv_ext))]
    end
end

function MPC_OF(uv, p)
    
    #necessary to add extra uv in order to be able to interpolate uv at the final time step of the prediction horizon.

    p.uv_ext[1:end-1] .= uv                 #dunno what effect doing something like this would have on the optimisation solver...
    p.uv_ext[end] = uv[end]
    t_start = p.t[p.i]
    t_horizon = p.t[p.i+p.p.H]
    t = t_start:p.p.t_step:t_horizon
    p.p.fx = make_fx(p.uv_ext, t)
 
    # initial_level = p.x_est[p.i, 2]
    # initial_F_in = p.x_est[p.i,3]
    # x0 = [initial_level, initial_F_in]

    x0 = [p.x_est[p.i, 2], p.x_est[p.i,3]]

   #problem = ODEProblem(MPC_ODE!, x0, [t_start, t_horizon], p.p) #End integration just before end of horizon to avoid issues with no interpolation at t= t_start + horizon
    problem = remake(p.template; u0 = x0, p = p.p, tspan = (t_start, t_horizon))
    sol = solve(problem, Tsit5(), saveat = t, abstol=1e-6, reltol=1e-6)

    predictions = first.(sol.u)

    J = sum((p.xSP.(t) .- predictions).^2) + p.p.alpha * sum((uv[2:(end-1)] .- uv[1:(end-2)]).^2)

    return J
end

#initial uv guess

uv_init = fill(0.4, H)
uv_ext = similar(uv_init, length(uv_init)+1)

i =  1

x_est[i, 1] = t[i]
x_est[i, 3] = F_in(t[i])                # Current input flow

#Specify ODE parameters

#must generate a fx output as a placeholder...
timeframe = 0.0:t_step:5.0
eg_fx = make_fx(uv_ext, timeframe)

p3 = Params3(10.0, 0.5, 1.0, 5, 0.1, eg_fx, x_est[i,3])

#create ODEProblem

initial_level = x_est[i, 2]
initial_F_in = x_est[i,3]
x0 = [initial_level, initial_F_in]

problem_template = ODEProblem(MPC_ODE!, x0, [0.0, 5.0], p3)

# Create MPC parameter struct for this iteration
display(t)
mpc_p3 = Mpc_p3(p3, xSP, x_est, t, i, uv_ext, problem_template)

checkJ = MPC_OF(uv_init, mpc_p3)

@benchmark MPC_OF($uv_init, $mpc_p3)

#https://chatgpt.com/c/68c4201f-a4f0-8332-b4bd-234f23232c2b

#JUST TRY CACHE THINGS AND BRING DOWN THE NUMBER OF ALLOCATIONS...