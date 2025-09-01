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
t_end = 150.0


#######
#Create randomized F_in 
######

Random.seed!(44)
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

########

mutable struct Params
    A           #m^2, buffer tank cross-sectional area
    cv          #m^2.5/min, valve constant
    t_step      #min, time steps between controller actions
    H           #number of time steps in controller prediction horizon
    alpha       #weighting parameter controlling the manipulated variable movemement
end

# specify parameters for tank ODE and MPC together 

p = Params(10, 0.5, 1, 5, 0.1)

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
    t #i don't actually need this, i only need to the horizon time scale not the entire timescale
    i
    h_t
    sp
end

function MPC_OF(uv, p)
    
    #necessary to add extra uv in order to be able to interpolate uv at the final time step of the prediction horizon.

    uv = [uv..., uv[end]]                 #dunno what effect doing something like this would have on the optimisation solver...

    fx = ConstantInterpolation(uv, p.h_t)  #everything is gonna have to become h_t
 
    function MPC_ODE!(dx, x, p, t)
        F_in = x[2]
        dx[1] = (F_in - fx(t) * p.cv * sqrt(x[1])) / p.A
        dx[2] = zero(x[2]) 
    end

    initial_level = p.x_est[p.i, 2]
    initial_F_in = p.x_est[p.i,3]
    x0 = [initial_level, initial_F_in]

    problem = ODEProblem(MPC_ODE!, x0, [p.h_t[1], t[p.i + p.p.H]], p.p) #End integration just before end of horizon to avoid issues with no interpolation at t= t_start + horizon
    sol = solve(problem, Tsit5(), saveat = t) #t or p.h_t

    predictions = first.(sol.u)

    #This was supposed to be faster but it wasnt
    # J1 = sum(@. (p.xSP(t) - predictions)^2)
    # du = @. uv[2:end-1] - uv[1:end-2]
    # J2 = sum(@. du^2)
    # J = sum(J1 + p.p.alpha * J2)

    J = sum((p.sp .- predictions).^2) + p.p.alpha * sum((uv[2:(end-1)] .- uv[1:(end-2)]).^2)

    return J
end

function mpc_step!(x_est, p, t, F_in, xSP, i, uv)
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
    
    # Update time and flow in measurements 
    x_est[i, 1] = t[i]
    x_est[i, 3] = F_in(t[i])                # Current input flow

    # Create timestep horizon for MPC step

    t_start = t[i]
    t_horizon = t[i + p.H]
    mpc_t = t_start:p.t_step:t_horizon

    #Calculate set points over time horizon

    sp = xSP.(mpc_t)

    # Create MPC parameter struct for this iteration
    mpc_p = Mpc_p(p, xSP, x_est, t, i, mpc_t, sp)

    #Optimization Method 1
    
    # Set optimization bounds
    lower = fill(0.01, p.H)
    upper = fill(0.99, p.H)

    options = Optim.Options(g_tol = 1e-4, f_reltol = 1e-6, iterations = 100)
    
    # Solve MPC optimization problem using Optim
    res = optimize(u -> MPC_OF(u, mpc_p), lower, upper, uv, Fminbox(BFGS()), options; autodiff = :finite)
    best_uv = Optim.minimizer(res)

    # #Optimization method 2
    
    # # Solve MPC optimization problem using OptimizationFunction

    # f = OptimizationFunction(MPC_OF, Optimization.AutoFiniteDiff()); 
    # prob = OptimizationProblem(f, uv, mpc_p, lb = fill(0.01, p.H), ub = fill(1, p.H)); # optimization problem with initial guess, p not needed here
    # sol = solve(prob, LBFGS())
    # best_uv = sol.u

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
    
    return odesol, best_uv
end

function run_mpc(x_est, p, t, F_in, xSP, uv_init)
    display("t")
    display(t)
    display("p")
    display(p)
     for i = 1:Int(t[end]-p.H + 1)
        display("i in loop")
        display(i)
        odesol, best_uv = mpc_step!(x_est, p, t, F_in, xSP, i, uv_init)
        x_est[i+1, 2] = odesol.u[end][1]                      # Final Level from ode problem solved to time i+1
        uv_init = best_uv
     end
     return x_est
end

#initial uv guess

uv_init = fill(0.4, p.H)

x_est = run_mpc(x_est, p, t, F_in, xSP, uv_init)

plot(x_est[:,1], x_est[:,2], label = "h")
plot!(x_est[:,1], x_est[:,3], label = "F_in")
plot!(x_est[:,1], x_est[:,4], label = "fx")
plot!(xSP)


