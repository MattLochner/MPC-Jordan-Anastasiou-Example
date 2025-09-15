t_start = 0.0
t_horizon = 5
t_step = 1.0

t_grid = t_start:t_step:t_horizon*t_step

display(t_grid)

uv_ext = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

display(uv_ext)

function make_fx(uv_ext, tgrid)
    return t -> begin
        idx = floor(Int,(t-tgrid[1]) / (tgrid[2] - tgrid[1])) + 1
        return uv_ext[clamp(idx, 1, length(uv_ext))]
    end
end

fx = make_fx(uv_ext, t_grid)

check = fx(3.0)