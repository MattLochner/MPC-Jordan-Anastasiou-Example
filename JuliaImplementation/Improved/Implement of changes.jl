using DifferentialEquations
using DataInterpolations
using Plots
using Random
using Optim
using Optimization
using OptimizationOptimJL

#
#see below for what I'm trying to implement here...
#https://chatgpt.com/c/68b55ec6-8fc4-832e-a5ed-600d4d872757
#

#Specify time for which the process will start and end:
t_start = 0.0
t_end = 8.0

#######
#Create randomized F_in 
######

#Select parameters for autoregressive function
mu  = 0.4
phi = 0.999
lambda = 0.004

#Select number of points in linear interpolation
N = 1500 #this is just the number of points, not the time

function Generate_F_in(t_start, t_end, mu, phi, lambda, N)
    Random.seed!(44)

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
    return F_in
end

F_inflow = Generate_F_in(t_start, t_end, mu, phi, lambda, N)
########

#Specify Parameters for ODE model
########################

mutable struct Params{F}
    A::Float64           #m^2, buffer tank cross-sectional area
    cv::Float64          #m^2.5/min, valve constant
    t_step::Float64      #min, time steps between controller actions
    H::Int               #number of time steps in controller prediction horizon
    alpha::Float64       #weighting parameter controlling the manipulated variable movemement
    fx::F
    F_in::Float64
end

#some of the parameters need to be calculated i order to fix types to make objective function faster.

A = 10.0                #m^2, buffer tank cross-sectional area
cv = 0.5                #m^2.5/min, valve constant
t_step = 1.0            #choose time steps between controller actions
H = 5                   #number of time steps between controller actions
alpha = 0.1             #weighting parameter controlling the manipulated variable movemement

#calculate example fx -> necessary to set type to make OF function faster
function make_fx(uv_ext, tgrid)
    return t -> begin
        idx = floor(Int,(t-tgrid[1]) / (tgrid[2] - tgrid[1])) + 1
        return uv_ext[clamp(idx, 1, length(uv_ext))]
    end
end

#generate time reference step range for MPC horizon
timeframe = 0:t_step:t_step*H                                   #this zero might be an issue later, it should actually be the time at the beginning of the MPC execution...

#specify initial uv estimate for time frame
uv_init = fill(0.4, H)
uv_ext = vcat(uv_init, uv_init[end])

eg_fx = ConstantInterpolation(uv_ext, timeframe)

eg_F_in = 1.0 #any float, will be updated

p = Params(A, cv, t_step, H, alpha, eg_fx, eg_F_in)

######################

#define set-point as a function of time

function xSP(t)
    return 2 + 0.5 * (t > 12) - 1.2 * (t > 60)
end

#This is my reference point for time for the full execution, t_step is the time steps at which controller actions are taken. It is assumed measurements are only taken at the time points that control is implemented.

t = t_start:p.t_step:t_end

#Initialize ground truth, measurements, and state estimates:

initial_level = 1.8 #m
initial_F_in = F_inflow(t_start)

#x_est should actually be called stored_values, it is where all process inputs and states are stored for each time step.

x_est_headings = ("time", "level", "F_in", "valve position")
x_est = fill(NaN, length(t), 4)
x_est[1, 2] = initial_level

#Define parameters for objective function

mutable struct Mpc_p
    p::Params
    x_setpoint::Vector{Float64}
    x_est::Matrix{Float64}
    t::Vector{Float64}
    i::Int
    uv_ext::Vector{Float64}
    template::ODEProblem
    timeframe
end


function MPC_OF(uv, mpc_p)
    
    #necessary to add extra uv in order to be able to interpolate uv at the final time step of the prediction horizon.

    #mpc_p.uv_ext[1:end-1] .= uv                 #dunno what effect doing something like this would have on the optimisation solver...
    #mpc_p.uv_ext[end] = uv[end]

    uv_ext = [uv..., uv[end]]
    t_start = mpc_p.t[mpc_p.i]
    t_horizon = mpc_p.t[mpc_p.i+mpc_p.p.H]
    t2 = t_start:mpc_p.p.t_step:t_horizon
    #mpc_p.p.fx = make_fx(uv_ext, mpc_p.timeframe)

    fx = ConstantInterpolation(uv_ext, t2)

    function MPC_ODE!(dx, x, p, t)
        p.F_in = x[2]
        dx[1] = (p.F_in - fx(t) * p.cv * sqrt(x[1])) / p.A
        dx[2] = 0.0
    end
 
    initial_level = mpc_p.x_est[mpc_p.i, 2]
    initial_F_in = mpc_p.x_est[mpc_p.i,3]
    x0 = [initial_level, initial_F_in]

    #x0 = [mpc_p.x_est[mpc_p.i, 2], mpc_p.x_est[mpc_p.i,3]]

   #problem = ODEProblem(MPC_ODE!, x0, [t_start, t_horizon], p.p) #End integration just before end of horizon to avoid issues with no interpolation at t= t_start + horizon
    problem = remake(mpc_p.template; u0 = x0, p = mpc_p.p, tspan = (mpc_p.timeframe[1], mpc_p.timeframe[end]))
    sol = solve(problem, Tsit5(), saveat = mpc_p.timeframe, abstol=1e-6, reltol=1e-6)

    predictions = first.(sol.u)

    J = sum((mpc_p.x_setpoint .- predictions).^2) + mpc_p.p.alpha * sum((uv[2:(end-1)] .- uv[1:(end-2)]).^2)

    return J
end

#specify initial uv guess

uv_init = fill(0.4, H)
uv_ext = vcat(uv_init, uv_init[end])

i = 1
x_est[i, 1] = t[i]
x_est[i, 3] = F_inflow(t[i])        #current in flow

#create template for ode problem

initial_level = x_est[i, 2]
initial_F_in = x_est[i,3]
x0 = [initial_level, initial_F_in]

problem_template = ODEProblem(MPC_ODE!, x0, [0.0, 5.0], p)

#create MPC parameter struct

# Create MPC parameter struct for testing this iteration
display(t)

timeframe = timeframe #MPC time frame
display(xSP.(timeframe))

x_setpoint = xSP.(timeframe)

mpc_p = Mpc_p(p, x_setpoint, x_est, t, i, uv_ext, problem_template, timeframe)

checkJ = MPC_OF(uv_init, mpc_p) #test objective function

#@benchmark MPC_OF($uv_init, $mpc_p) #test objective function speed

function mpc_step!(x_est, mpc_p, t, F_inflow, xSP, i, uv_prev)
    "
    Execute on MPC iteration in each step

    Arguments:
    - x_est: results matrix (modefied in-place)
    - p: ODE parameters structure
    - F_inflow: input flow interpolation function
    - xSP: setpoint function
    - i: current iteration index
    - uv_prev: check this but its probabily the uv from the previous timestep.!!!!

    Returns:
    - odesol: ODE solution for this step (to pass to next iteration)
    - optimal_uv: optimised solution for future inputs
    "
    # Update time and flow in measurements 
    x_est[i, 1] = t[i]
    x_est[i, 3] = F_inflow(t[i])                # Current input flow

    # Create timestep horizon for MPC step

    t_start = t[i]
    t_horizon = t[i + mpc_p.p.H]
    mpc_p.timeframe = t_start:mpc_p.p.t_step:t_horizon

    #Calculate set points over time horizon

    mpc_p.x_setpoint = xSP.(mpc_p.timeframe)

    #Optimization Method 1
    
    #Set optimization bounds
    lower = fill(0.01, mpc_p.p.H)
    upper = fill(1.0, mpc_p.p.H)

    options = Optim.Options(g_tol = 1e-4, f_reltol = 1e-6, iterations = 100)
    display("uv_prev")
    display(uv_prev)
    display("mpc_.x_setpoint")
    display(mpc_p.x_setpoint)
    display("mpc_t")
    display(mpc_p.timeframe)
    display("J before optimization")
    display(MPC_OF(uv_prev, mpc_p))
    # Solve MPC optimization problem using Optim
    res = optimize(uv -> MPC_OF(uv, mpc_p), lower, upper, uv_prev, Fminbox(BFGS()), options; autodiff = :finite)
    best_uv = Optim.minimizer(res)

    display("checkJ")
    display(MPC_OF(best_uv, mpc_p)) #test objective function


    # #Optimization method 2
    
    # # Solve MPC optimization problem using OptimizationFunction

    # f = OptimizationFunction(MPC_OF, Optimization.AutoFiniteDiff()); 
    # prob = OptimizationProblem(f, uv, mpc_p, lb = fill(0.01, p.H), ub = fill(1, p.H)); # optimization problem with initial guess, p not needed here
    # sol = solve(prob, LBFGS())
    # best_uv = sol.u

    # Store optimal valve position
    x_est[i, 4] = best_uv[1]

    println("Iteration $i: best_uv = $(best_uv[1])")

    fxval = x_est[i, 4]

    function fx_current(t)
        return fxval
    end

    # Define ground truth ODE
    function GroundTruth_ODE!(dx, x, p, t)
        dx[1] = (F_inflow(t) - fx_current(t) * p.cv * sqrt(x[1])) / p.A
    end
    # Simulate one step forward
    odeproblem = ODEProblem(GroundTruth_ODE!, [x_est[i, 2]], (t[i], t[i+1]), mpc_p.p)
    odesol = solve(odeproblem, Tsit5())
    
    return odesol, best_uv
end

function run_mpc(x_est, mpc_p, t, F_inflow, xSP, uv_init)
    #display("t")
    #display(t)
    #display("p")
    #display(mpc_p)
     for i = 1:Int(t[end]-p.H + 1)
        display("i in loop")
        display(i)
        odesol, best_uv = mpc_step!(x_est, mpc_p, t, F_inflow, xSP, i, uv_init)
        x_est[i+1, 2] = odesol.u[end][1]                      # Final Level from ode problem solved to time i+1
        uv_init = best_uv
     end
     return x_est
end
#initial uv guess

uv_init = fill(0.4, p.H)

x_est = run_mpc(x_est, mpc_p, t, F_inflow, xSP, uv_init)

plot(x_est[:,1], x_est[:,2], label = "h")
plot!(x_est[:,1], x_est[:,3], label = "F_in")
plot!(x_est[:,1], x_est[:,4], label = "fx")
plot!(xSP)


