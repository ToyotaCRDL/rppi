module ESN

export reservoir_generate_weight, reservoir_update, reservoir_output, update_Wout_rls, calc_cost_esn, select_activation_function, train_Wout_batch

using LinearAlgebra, Graphs, Distributions

function reservoir_generate_weight(esn_params, num_reservoir_nodes, num_input_nodes, num_output_nodes, esn_rng)
  """generate weight matrix for reservoir"""
  RESERVOIR_SPECTRAL_RADIUS = esn_params["SPEC_RAD"]
  WIN_SCALE = esn_params["WIN_SCALE"]
  DENSITY = esn_params["RES_DENS"]
  Wout_scale = esn_params["WOUT_SCALE"]

  m = round(Int, num_reservoir_nodes * (num_reservoir_nodes - 1) * DENSITY / 2)
  G = SimpleGraph(num_reservoir_nodes)
  for _ in 1:m
    add_edge!(G, rand(esn_rng, 1:num_reservoir_nodes), rand(esn_rng, 1:num_reservoir_nodes))
  end
  connection = adjacency_matrix(G)
  weight_reservoir_onezero = Array(connection)

  weight_reservoir = weight_reservoir_onezero .* (2.0 .* rand(esn_rng, num_reservoir_nodes, num_reservoir_nodes) .- 1.0)

  spectral_radius = maximum(abs.(eigvals(weight_reservoir)))
  weight_reservoir .= RESERVOIR_SPECTRAL_RADIUS .* weight_reservoir ./ spectral_radius

  weight_input = WIN_SCALE .* (2.0 .* rand(esn_rng, num_reservoir_nodes, num_input_nodes) .- 1)
  weight_output = Wout_scale .* (2.0 .* rand(esn_rng, num_output_nodes, num_reservoir_nodes) .- 1)

  return weight_reservoir, weight_input, weight_output
end


function select_activation_function(ACT::Int)
  return ACT == 0 ? tanh :
         ACT == 1 ? x -> max.(0, x) :
         ACT == 2 ? x -> x .- tanh.(x) :
         ACT == 3 ? x -> x :
         ACT == 4 ? x -> 1 ./ (1 .+ exp.(-x)) :
         tanh
end

function reservoir_update(x, u, W, Win, ACT, leakrate)
  """update reservoir state"""
  z = (W * x) .+ (Win * u)
  activation_function = select_activation_function(ACT)

  x_next = (1 - leakrate) * x .+ leakrate * activation_function.(z)
  return x_next
end

function reservoir_output(x, Wout)
  """calculate reservoir output"""
  return Wout * x
end

function update_Wout_rls(x, P, Wout, current_output)
  """update Wout using RLS"""
  lambda_ = 1.0
  P = (P .- (P * x * x' * P) ./ (lambda_ .+ x' * P * x)) ./ lambda_
  e = current_output - reservoir_output(x, Wout)
  delta = (P * x ./ (lambda_ .+ x' * P * x)) * e'
  Wout = Wout .+ delta'
  return Wout, P
end

function calc_cost_esn(esn_y_ar, plant_y_ar)
  """calculate cost for ESN"""
  return mean((esn_y_ar .- plant_y_ar)' * (esn_y_ar .- plant_y_ar))
end

function train_Wout_batch(X, Y; alpha=1e-6)
  # X: (N_res, T), Y: (N_out, T)
  N_res = size(X, 1)
  reg = alpha * I(N_res)
  Wout = Y * X' * inv(X * X' + reg)
  return Wout
end

end # module