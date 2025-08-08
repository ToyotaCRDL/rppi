##
ENV["QT_QPA_PLATFORM"] = "offscreen"
ENV["GKS_WSTYPE"] = "png"  # GRバックエンドの場合

using ArgParse
using YAML
using LinearAlgebra
using Random
using Plots
using Dates
using ProgressMeter
using JLD2

include("plant-fourtank.jl")
include("esn.jl")
# include("mppi-esn.jl")
include("mppi-esn-fast.jl")
include("utils.jl")
using .Plant
using .ESN
using .MPPI
using .Utils


num_threads = Threads.nthreads()
BLAS.set_num_threads(1)
println("Total threads available: ", num_threads)
println("BLAS threads set to: ", BLAS.get_num_threads())


## Parse command-line arguments
arg_definitions = Dict(
  "--plant_config" => ("Path to the plant configuration file", String, "config/plant-fourtank.yaml"),
  "--simulation_config" => ("Path to the simulation configuration file", String, "config/simulation-fourtank.yaml"),
  "--esn_config" => ("Path to the ESN configuration file", String, "config/esn-fourtank.yaml"),
  "--mppi_config" => ("Path to the MPPI configuration file", String, "config/mppi-fourtank.yaml"),
  "--out_dir" => ("Root directory for all results", String, "results"),
  "--key" => ("Random seed key for reproducibility", Int, 1),
  "--wout_scale" => ("Scale factor for Wout", Float64, 0.0),
  "--sigma2" => ("Sigma2 for MPPI", Float64, 0.1),
  "--sample" => ("Sample size for MPPI", Int, 10),
  "--sample2" => ("Sample size for UMPPI", Int, 1),
)

args = Utils.parse_commandline(arg_definitions)

# Load hyperparameters from YAML files
plant_params = Utils.load_hyperparameters(args["plant_config"])
sim_params = Utils.load_hyperparameters(args["simulation_config"])
esn_params = Utils.load_hyperparameters(args["esn_config"])
mppi_params = Utils.load_hyperparameters(args["mppi_config"])


# ハイパーパラメータ上書き
esn_params["WOUT_SCALE"] = args["wout_scale"]
mppi_params["SIGMA2"] = args["sigma2"]
mppi_params["SAMPLE"] = args["sample"]
mppi_params["SAMPLE2"] = args["sample2"]

# Initialize directories
base_out = args["out_dir"]
key = args["key"]
script_with_ext = basename(PROGRAM_FILE)
script_name = splitext(script_with_ext)[1]
prefix = string(script_name, "_key", key)


results_dir, data_dir, fig_dir, logs_dir, config_dir = generate_result_dirs(base_out, prefix)
# results_dir, data_dir, fig_dir, logs_dir, config_dir = generate_result_dirs(base_out, "smd_lesn_mppi")

YAML.write_file(joinpath(config_dir, "plant-fourtank.yaml"), plant_params)
YAML.write_file(joinpath(config_dir, "simulation-fourtank.yaml"), sim_params)
YAML.write_file(joinpath(config_dir, "esn-fourtank.yaml"), esn_params)
YAML.write_file(joinpath(config_dir, "mppi-fourtank.yaml"), mppi_params)


esn_rng = Xoshiro(args["key"])
mppi_rng = Xoshiro(args["key"] + 1)
mppi_covar_rng = Xoshiro(args["key"] + 2)
sim_rng = Xoshiro(args["key"] + 3)


## Initialize plant, ESN, and MPPI
dt = plant_params["dt"]
# plant_x0 = zeros(plant_params["N"], 1)
plant_x0 = [12.4; 12.7; 1.8; 1.4]

NUM_INPUT_NODES = plant_params["M"]
NUM_OUTPUT_NODES = plant_params["L"]
NUM_RESERVOIR_NODES = esn_params["DIM_RES"]
W, Win, Wout = ESN.reservoir_generate_weight(esn_params, NUM_RESERVOIR_NODES, NUM_INPUT_NODES, NUM_OUTPUT_NODES, esn_rng)
ACT = esn_params["ACT"]
leakrate = esn_params["LEAK"]

esn_x0 = zeros(NUM_RESERVOIR_NODES, 1)
esn_P0 = I(NUM_RESERVOIR_NODES)

esn_x = esn_x0
esn_P = esn_P0
plant_x = plant_x0


## ESN Training
if sim_params["PRETRAIN"]
  global plant_x, esn_x
  println("Start ESN Training.")

  train_step = sim_params["SIM_STEP"]
  train_input = 0.5 .* mppi_params["INPUT_LIM"] .* (2 .* rand(sim_rng, train_step, plant_params["M"]) .- 1)

  plant_y_log = zeros(train_step, plant_params["L"])
  esn_x_log = zeros(NUM_RESERVOIR_NODES, train_step)
  Y_batch = zeros(NUM_OUTPUT_NODES, train_step)

  esn_x = esn_x0

  for i in 1:esn_params["washout_len"]
    global esn_x
    input_ = train_input[i, :]
    esn_x = ESN.reservoir_update(esn_x, input_, W, Win, ACT, leakrate)
    # この間は X_batch, Y_batch には記録しない
  end

  plant_x = plant_x0

  for ind_ in 1:train_step
    global plant_x, esn_x
    input_ = train_input[ind_, :]
    plant_x = Plant.plant_update(plant_params, plant_x, input_)
    plant_y = Plant.plant_output(plant_x)
    esn_x = ESN.reservoir_update(esn_x, input_, W, Win, ACT, leakrate)
    esn_x_log[:, ind_] = esn_x[:, 1]
    Y_batch[:, ind_] = plant_y[:, 1]
    plant_y_log[ind_, :] = plant_y
  end
  # バッチ最小二乗でWoutを一括学習（L2正則化付き）
  Wout = ESN.train_Wout_batch(esn_x_log, Y_batch; alpha=esn_params["regularization"])
  esn_y_log = (Wout * esn_x_log)'

  if sim_params["plot"]
    println("Plotting ESN training results.")
    plot(plant_y_log, label="Plant output", xlabel="Time Step", ylabel="Output", title="Output Response", grid=true)
    plot!(esn_y_log, label="ESN output", xlabel="Time Step", ylabel="Output", title="Output Response", grid=true)
    savefig(joinpath(fig_dir, "plant_esn_output.png"))
    plot(esn_x_log', alpha=0.1, label="", xlabel="Time Step", ylabel="Reservoir State", title="Reservoir State Response", grid=true)
    savefig(joinpath(fig_dir, "esn_state.png"))
    plot(abs.(esn_y_log .- plant_y_log), label="", xlabel="Time Step", ylabel="Error", title="Identification Error", grid=true)
    savefig(joinpath(fig_dir, "identification_error.png"))
    println("Plotting ESN training results done.")
  end

  cost_esn = ESN.calc_cost_esn(esn_y_log, plant_y_log)
  println("ESN Cost: ", cost_esn)
  println("End ESN Training.")

end


## MPPI Simulation
sim_step = sim_params["SIM_STEP"]
plant_output_log = zeros(sim_step, plant_params["L"])
esn_output_log = zeros(sim_step, plant_params["L"])
esn_x_log = zeros(sim_step, NUM_RESERVOIR_NODES)
esn_Wout_log = zeros(sim_step, plant_params["L"], NUM_RESERVOIR_NODES)
esn_P_log = zeros(sim_step, NUM_RESERVOIR_NODES, NUM_RESERVOIR_NODES)
mppi_input_log = zeros(sim_step, plant_params["M"])
mppi_weight_log = zeros(sim_step, mppi_params["SAMPLE"])
computation_time_log = zeros(sim_step)



# esn_x = esn_x0
# esn_P = esn_P0

plant_x = plant_x0
prev_input = zeros(plant_params["M"], mppi_params["HORIZON"])
# ref_out = mppi_params["REF_OUT"]
ref_input = mppi_params["REF_INPUT"]


reference_outputs = zeros(sim_step)
for t in 1:sim_step
  # 100ステップごとに符号を切り替え
  sign = (-1)^div(t - 1, 150)
  reference_outputs[t] = mppi_params["REF_OUT"] + 1.0 * sign
end


## MPPI Simulation Loop
println("Start MPPI Simulation.")
@showprogress for ind_ in 1:sim_step
  global plant_x, esn_x, prev_input, esn_P, Wout

  ref_out = reference_outputs[ind_]
  observed_output = Plant.plant_output(plant_x)
  epsilon = mppi_params["SIGMA"] * randn(mppi_rng, plant_params["M"], mppi_params["SAMPLE"], mppi_params["HORIZON"])

  # --- MPPI.controller_calc_inputの計算時間を測定 ---
  t_start = time()
  # optimal_input, optimal_input_sequence, prev_input, mppi_w = MPPI.controller_calc_input(
  optimal_input, prev_input, _, mppi_w = MPPI.controller_calc_input(
    mppi_params, prev_input, esn_x, ref_out, epsilon, Wout, esn_P, plant_params, W, Win, ACT, leakrate, reference_input=ref_input, mppi_covar_rng=mppi_covar_rng
  )
  t_end = time()
  computation_time_log[ind_] = t_end - t_start
  # -----------------------------------------------

  plant_x = Plant.plant_update(plant_params, plant_x, optimal_input)

  if !sim_params["PRETRAIN"]
    Wout, esn_P = ESN.update_Wout_rls(esn_x, esn_P, Wout, observed_output)
  end

  esn_output = ESN.reservoir_output(esn_x, Wout)
  esn_x = ESN.reservoir_update(esn_x, optimal_input, W, Win, ACT, leakrate)

  # ログ記録（この後でもOK）
  mppi_input_log[ind_, :] = optimal_input
  plant_output_log[ind_, :] = observed_output
  esn_output_log[ind_, :] = esn_output
  esn_x_log[ind_, :] = esn_x
  esn_Wout_log[ind_, :, :] = Wout
  esn_P_log[ind_, :, :] = esn_P
  mppi_weight_log[ind_, :] = mppi_w
end
println("End MPPI Simulation.")


## Plot MPPI results
if sim_params["plot"]
  println("Plotting MPPI results.")

  plot(plant_output_log, label="Plant output", xlabel="Time Step", ylabel="Output", title="Output Response", grid=true)
  plot!(esn_output_log, label="ESN output", xlabel="Time Step", ylabel="Output", title="Output Response", grid=true)
  plot!(reference_outputs, label="Reference output", xlabel="Time Step", ylabel="Output", title="Output Response", grid=true)
  savefig(joinpath(fig_dir, "mppi_output.png"))


  plot()
  for i in 1:NUM_RESERVOIR_NODES
    plot!(esn_x_log[:, i], label="", xlabel="Time Step", ylabel="Reservoir State", title="Reservoir State Response", grid=true)
  end
  savefig(joinpath(fig_dir, "esn_state.png"))


  plot(mppi_input_log, label="", xlabel="Time Step", ylabel="Control Input", title="MPPI Input Response", grid=true)
  savefig(joinpath(fig_dir, "mppi_input.png"))


  plot()
  for i in 1:mppi_params["SAMPLE"]
    plot!(mppi_weight_log[:, i], label="", xlabel="Time Step", title="MPPI Weight Response", grid=true)
  end
  savefig(joinpath(fig_dir, "mppi_weight.png"))

  plot()
  for i in 1:NUM_RESERVOIR_NODES
    plot!(esn_Wout_log[:, :, i], label="", xlabel="Time Step", title="Wout Response", grid=true)
  end
  savefig(joinpath(fig_dir, "esn_Wout.png"))

  plot()
  for i in 1:NUM_RESERVOIR_NODES
    plot!(esn_P_log[:, i, i], label="", xlabel="Time Step", title="P Response", grid=true)
  end
  save
  savefig(joinpath(fig_dir, "esn_P.png"))

  plot(abs.(esn_output_log .- plant_output_log), label="", xlabel="Time Step", ylabel="Error", title="Identification Error", grid=true)
  savefig(joinpath(fig_dir, "identification_error.png"))

  plot(abs.(plant_output_log .- reference_outputs), label="", xlabel="Time Step", ylabel="Error", title="Control Error", grid=true)
  savefig(joinpath(fig_dir, "control_error.png"))

  println("Plotting MPPI results done.")
end


## Calculate costs
cost_output = MPPI.calc_cost(mppi_params, plant_output_log; reference_outputs=reference_outputs)
cost_total = MPPI.calc_cost_tot(mppi_params, plant_output_log, mppi_input_log, reference_input=ref_input, reference_outputs=reference_outputs)

println("mppi cost: ", cost_total)

## Save data

println("saving data: data.jld2")
save(joinpath(data_dir, "data.jld2"),
  "plant_output_log", plant_output_log,
  "esn_output_log", esn_output_log,
  "mppi_input_log", mppi_input_log,
  "mppi_weight_log", mppi_weight_log,
  "esn_x_log", esn_x_log,
  "esn_Wout_log", esn_Wout_log,
  "esn_P_log", esn_P_log,
  "cost_output", cost_output,
  "cost_total", cost_total,
  "computation_time_log", computation_time_log
)
println("saving data done.")