using DifferentialEquations
using Plots

mutable struct NewParams
    A           #m^2, buffer tank cross-sectional area
    cv          #m^2.5/min, valve constant
    t_step       #min, time steps between controller actions
    H           #number of time steps in controller prediction horizon
    alpha       #weighting parameter controlling the manipulated variable movemement
    fx
end

p = NewParams(10, 0.5, 1, 5, 0.1, 1)


function MPC_ODE!(dx, x, p, t)
    F_in = x[2]
    dx[1] = (F_in - 1 * p.cv * sqrt(x[1])) / p.A
    dx[2] = zero(x[2]) 
end

t_start = 55
t_horizon = 100
x0 = [2.5, 0.5]

problem = ODEProblem(MPC_ODE!, x0, [t_start, t_horizon], p) #End integration just before end of horizon to avoid issues with no interpolation at t= t_start + horizon
sol = solve(problem, Tsit5())

plot(sol)