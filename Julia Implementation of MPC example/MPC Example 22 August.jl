using DifferentialEquations
using DataInterpolations
using Plots
using Random
using Optim

#Specify time for which the process will start and end:
t_start = 0.0
t_end = 100.0

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

mutable struct Params
    A           #m^2, buffer tank cross-sectional area
    cv          #m^2.5/min, valve constant
    t_step       #min, time steps between controller actions
    H           #number of time steps in controller prediction horizon
    alpha       #weighting parameter controlling the manipulated variable movemement
end

p = Params(10, 0.5, 1, 5, 0.1)

function xSP(t)
    return 2 - 0.5 * (t > 20) + 1.2 * (t > 50)
end
;

t = t_start:p.t_step:t_end #This is my reference point for time, t_step is the time at which controller actions are taken. It is assumed measurements are only taken at the time points that control is implemented.

#Initialize ground truth, measurements, and state estimates:

initial_level = 1.8 #m
initial_F_in = F_in(t_start)

#x_est should actually be called stored_values, it is where all process inputs and states are stored for each time step.

x_est_headings = ("time", "level", "F_in", "valve position")
x_est = fill(NaN, length(t), 4)
x_est[1, 1] = t_start
x_est[1, 2] = initial_level
x_est[1, 3] = initial_F_in

x_est;

#set initial estimate for valve position set point horizon

uv = fill(0.5, p.H) #Final UV value is required for end time step in order for ConstantInterpolation to work. (e.g. for running process from 0 -> 5, a uv value at 5 must exist)
#uv = [0.9, 0.9, 0.9, 0.7, 0.8]

mutable struct Mpc_p
    p
    xSP
    x_est
    t
    i
end

i = 1 #First Iteration

mpc_p = Mpc_p(p, xSP, x_est, t, i)

function MPC_OF(uv, p)
    
    #define time horizon
    uv = [uv..., uv[end]]                                       #dunno what effect doing something like this would have on the optimisation solver...
    t_start = p.t[p.i]
    t_horizon = p.t[p.i+p.p.H]
    t = t_start:p.p.t_step:t_horizon
    fx = ConstantInterpolation(uv, t)
 
    function MPC_ODE!(dx, x, p, t)
        F_in = x[2]
        dx[1] = (F_in - fx(t) * p.cv * sqrt(x[1])) / p.A
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

check = MPC_OF(uv, mpc_p)

lower = zeros(p.H) #set lower limits on uv values
upper = ones(p.H) #set upper limits on uv values

#Optimizer using finite difference and not forward diferentiation, much slower but doesn't require me to rejig all the types in my objective function.

res = optimize(u -> MPC_OF(u, mpc_p), lower, upper, uv, Fminbox(BFGS()); autodiff = :finite) #This is a lot like fmincon in Matlab
best_uv = Optim.minimizer(res)

x_est[i, 4] = best_uv[1]

fxval = x_est[i, 4]

function fx_current(t)
    return fxval
end

check = fx_current(2)

function GroundTruth_ODE!(dx, x, p, t)
    dx[1] = (F_in(t) - fx_current(t) * p.cv * sqrt(x[1])) / p.A
end

odeproblem = ODEProblem(GroundTruth_ODE!, [x_est[i, 2]], (t[i], t[i+1]), p)
odesol = solve(odeproblem, Tsit5())
odesol.u[end] #final level

display(x_est)

display("begin loop")

function mpc_step!(x_est, p, t, F_in, xSP, i, odesol_prev)
    """
    Execute one MPC iteration step
    
    Arguments:
    - x_est: Results matrix (modified in-place)
    - p: Parameters struct
    - t: Time vector  
    - F_in: Input flow interpolation function
    - xSP: Setpoint function
    - i: Current iteration index
    - odesol_prev: Previous ODE solution for initial condition
    
    Returns:
    - odesol: ODE solution for this step (to pass to next iteration)
    """
    
    # Update state estimates
    x_est[i, 1] = t[i]
    x_est[i, 2] = odesol_prev.u[end][1]     # Level from previous simulation
    x_est[i, 3] = F_in(t[i])                # Current input flow
    
    # Initialize control horizon
    uv = fill(0.4, p.H)
    
    # Create MPC parameter struct for this iteration
    mpc_p = Mpc_p(p, xSP, x_est, t, i)
    
    # Set optimization bounds
    lower = zeros(p.H)
    upper = ones(p.H)

    options = Optim.Options(g_tol = 1e-4, f_reltol = 1e-6, iterations = 100)
    
    # Solve MPC optimization problem
    res = optimize(u -> MPC_OF(u, mpc_p), lower, upper, uv, Fminbox(LBFGS()), options; autodiff = :finite)
    best_uv = Optim.minimizer(res)
    
    # Store optimal valve position
    x_est[i, 4] = best_uv[1]
    
    println("Iteration $i: best_uv = $(best_uv[1])")
    
    # Create valve position function for this step
    fxval = x_est[i, 4]
    function fx_current(t)
        return fxval
    end
    
    # Define ground truth ODE
    function GroundTruth_ODE!(dx, x, p, t)
        dx[1] = (F_in(t) - fx_current(t) * p.cv * sqrt(x[1])) / p.A
    end
    
    # Simulate one step forward
    odeproblem = ODEProblem(GroundTruth_ODE!, [x_est[i, 2]], (t[i], t[i+1]), p)
    odesol = solve(odeproblem, Tsit5())
    
    return odesol
end

# OPTION 1

# Initialize first step (you already have this) from your existing first step

function run_mpc(x_est, p, t, F_in, xSP, odesol)
     for i = 2:100
        display("i in loop")
        display(i)
        odesol = mpc_step!(x_est, p, t, F_in, xSP, i, odesol) #!!! I think its calling the previous ODEsol
     end
     return x_est
end

x_est = run_mpc(x_est, p, t, F_in, xSP, odesol)

# x_est

plot(x_est[:,1], x_est[:,2], label = "h")
plot!(x_est[:,1], x_est[:,3], label = "F_in")
plot!(x_est[:,1], x_est[:,4], label = "fx")
plot!(xSP)

# # # #OPTION 2

# i = 2
# odesol = mpc_step!(x_est, p, t, F_in, xSP, i, odesol)

# i = 3
# odesol = mpc_step!(x_est, p, t, F_in, xSP, i, odesol)

# i = 4
# odesol = mpc_step!(x_est, p, t, F_in, xSP, i, odesol)

# i = 5
# odesol = mpc_step!(x_est, p, t, F_in, xSP, i, odesol)


# x_est