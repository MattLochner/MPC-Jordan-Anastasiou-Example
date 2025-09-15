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
mu  = 0.5
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

########

mutable struct Params
    A::Float64           #m^2, buffer tank cross-sectional area
    cv::Float64          #m^2.5/min, valve constant
    t_step::Int      #min, time steps between controller actions
    H::Int           #number of time steps in controller prediction horizon
    alpha::Float64       #weighting parameter controlling the manipulated variable movemement
end

struct Params3
    A::Float64           #m^2, buffer tank cross-sectional area
    cv::Float64          #m^2.5/min, valve constant
    t_step::Int      #min, time steps between controller actions
    H::Int           #number of time steps in controller prediction horizon
    alpha::Float64       #weighting parameter controlling the manipulated variable movemement
end

mutable struct Params2
    A           #m^2, buffer tank cross-sectional area
    cv          #m^2.5/min, valve constant
    t_step      #min, time steps between controller actions
    H           #number of time steps in controller prediction horizon
    alpha       #weighting parameter controlling the manipulated variable movemement
end

# specify parameters for tank ODE and MPC together 

p = Params(10.0, 0.5, 1, 5, 0.1)
p2 = Params2(10.0, 0.5, 1, 5, 0.1)
p3 = Params3(10.0, 0.5, 1, 5, 0.1)

function xSP(t)
    return 2 + 0.5 * (t > 23) - 1.2 * (t > 60)
end

#This is my reference point for time, t_step is the time at which controller actions are taken. It is assumed measurements are only taken at the time points that control is implemented.

t = t_start:p.t_step:t_end

#Initialize ground truth, measurements, and state estimates:

initial_level = 1.8 #m
initial_F_in = F_in(t_start)

#x_est should actually be called stored_values, it is where all process inputs and states are stored for each time step.

x_est_headings = ("time", "level", "F_in", "valve position")
x_est = fill(NaN, length(t), 4)
x_est[1, 2] = initial_level

mutable struct Mpc_p
    p
    xSP
    x_est
    t
    i
end

mutable struct Mpc_p3
    p::Params
    xSP::Function
    x_est::Matrix{Float64}
    t::Vector{Float64}
    i::Int
end

mutable struct Mpc_p4
    p::Params3
    xSP::Function
    x_est::Matrix{Float64}
    t::Vector{Float64}
    i::Int
end

mutable struct Mpc_p5
    p::Params3
    xSP::Function
    x_est::Matrix{Float64}
    t::Vector{Float64}
    i::Int
end

function MPC_OF(uv, p)
    
    #necessary to add extra uv in order to be able to interpolate uv at the final time step of the prediction horizon.

    uv = [uv..., uv[end]]                 #dunno what effect doing something like this would have on the optimisation solver...
    t_start = p.t[p.i]
    t_horizon = p.t[p.i+p.p.H]
    t = t_start:p.p.t_step:t_horizon
    fx = ConstantInterpolation(uv, t)
 
    function MPC_ODE!(dx, x, p, t)
        F_in = x[2]
        dx[1] = (F_in - fx(t) * p.cv * sqrt.(x[1])) / p.A
        dx[2] = zero(x[2]) 
    end

    initial_level = p.x_est[p.i, 2]
    initial_F_in = p.x_est[p.i,3]
    x0 = [initial_level, initial_F_in]

    problem = ODEProblem(MPC_ODE!, x0, [t_start, t_horizon], p.p) #End integration just before end of horizon to avoid issues with no interpolation at t= t_start + horizon
    sol = solve(problem, Tsit5(), saveat = t)

    predictions = first.(sol.u)

    J = sum((p.xSP.(t) .- predictions).^2) + p.p.alpha * sum((uv[2:(end-1)] .- uv[1:(end-2)]).^2)

    return J
end

#initial uv guess

uv_init = fill(0.4, p.H)
uv_ext = similar(uv, length(uv)+1)

i =  1

x_est[i, 1] = t[i]
x_est[i, 3] = F_in(t[i])                # Current input flow

# Create MPC parameter struct for this iteration
mpc_p = Mpc_p(p, xSP, x_est, t, i)
mpc_p2 = Mpc_p(p2, xSP, x_est, t, i)
mpc_p3 = Mpc_p3(p, xSP, x_est, t, i)
mpc_p4 = Mpc_p4(p3, xSP, x_est, t, i)

checkJ = MPC_OF(uv_init, mpc_p)

@benchmark MPC_OF($uv_init, $mpc_p4)

