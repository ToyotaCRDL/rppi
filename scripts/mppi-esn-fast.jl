module MPPI

export controller_calc_input, calc_cost, calc_cost_tot

using LinearAlgebra, Distributions, Statistics, Random
using Base.Threads

include("esn.jl")
using .ESN

include("utils.jl")
using .Utils


function controller_g(params, v; reference_input=0.0)
  INPUT_LIM = params["INPUT_LIM"]
  v = clamp.(v, -INPUT_LIM + reference_input, INPUT_LIM + reference_input)
  return v
end

@inline function controller_c(params, y, reference_output)
  """calculate stage cost (inlined with manual dot)"""
  Q = params["Q"]
  cost = 0.0
  @inbounds for i in eachindex(y)
    diff = reference_output - y[i]
    cost += diff * diff
  end
  return Q * cost
end

@inline function controller_phi(params, y, reference_output)
  """calculate terminal cost (inlined with manual dot)"""
  Q = params["Q"]
  cost = 0.0
  @inbounds for i in eachindex(y)
    diff = reference_output - y[i]
    cost += diff * diff
  end
  return Q * cost
end

# スレッドローカルバッファをモジュールレベルで事前確保
const MAX_HORIZON = 20  # 余裕を持ったサイズ
const MAX_RES_SIZE = 500
const MAX_INPUT_DIM = 5

# スレッドローカルバッファ構造
struct ThreadBuffer
  v::Matrix{Float64}
  x::Vector{Float64}
  state_trajectory::Matrix{Float64}
  input_costs::Vector{Float64}
end

# スレッドローカルバッファの初期化
function create_thread_buffers(num_threads::Int)
  return [ThreadBuffer(
    zeros(Float64, MAX_INPUT_DIM, MAX_HORIZON),
    zeros(Float64, MAX_RES_SIZE),
    zeros(Float64, MAX_RES_SIZE, MAX_HORIZON),
    zeros(Float64, MAX_HORIZON)
  ) for _ in 1:num_threads]
end

# グローバルなスレッドバッファ（遅延初期化）
const THREAD_BUFFERS = Ref{Vector{ThreadBuffer}}()
const BUFFER_LOCK = ReentrantLock()

function get_thread_buffers()
  if !isassigned(THREAD_BUFFERS)
    lock(BUFFER_LOCK) do
      if !isassigned(THREAD_BUFFERS)  # Double-checked locking
        THREAD_BUFFERS[] = create_thread_buffers(Threads.nthreads())
      end
    end
  end
  return THREAD_BUFFERS[]
end


function controller_S_optimized(params, x0, u, epsilon, ref_output, K, T, K2, Wrand, S, obs_noise, W, Win, ACT, leakrate; reference_input=0.0)
  """
  Optimized version following ESN uncertainty evaluation approach:
  - Compute state trajectory once per k (control sample)
  - Evaluate uncertainty only through output weight samples (k2)
  """
  param_gamma = params["LAMBDA"] * (1.0 - params["ALPHA"])
  Sigma_inv = 1.0 / params["SIGMA"]
  
  @threads for k in 1:K
    # スレッドローカルバッファを取得
    buffers = get_thread_buffers()
    tid = Threads.threadid()
    buffer = buffers[tid]
    
    input_dim = size(u, 1)
    res_size = size(x0, 1)
    v = view(buffer.v, 1:input_dim, 1:T)
    x = view(buffer.x, 1:res_size)
    state_trajectory = view(buffer.state_trajectory, 1:res_size, 1:T)
    input_costs = view(buffer.input_costs, 1:T)
    
    # バッファをクリア
    fill!(v, 0.0)
    fill!(x, 0.0)
    fill!(state_trajectory, 0.0)
    fill!(input_costs, 0.0)
    
    # 1. 状態軌道を一度だけ計算 (O(N*T) operations)
    x .= view(x0, :, 1)
    
    for t in 1:T
      # 制御入力の計算
      @inbounds for i in 1:input_dim
        v[i, t] = u[i, t] + epsilon[i, k, t]
      end
      
      # 状態更新（一度だけ）
      input_limited = controller_g(params, view(v, :, t), reference_input=reference_input)
      x .= ESN.reservoir_update(x, input_limited, W, Win, ACT, leakrate)
      state_trajectory[:, t] .= x
      
      # 入力コストを事前計算
      input_cost = 0.0
      @inbounds for i in 1:input_dim
        u_diff = u[i, t] - reference_input
        v_diff = v[i, t] - reference_input
        input_cost += u_diff * Sigma_inv * v_diff
      end
      input_costs[t] = input_cost
    end
    
    # 2. 出力重み行列のサンプルのみで不確実性評価 (O(N*L*K2) operations)
    for k2 in 1:K2
      S_k2 = 0.0
      
      # ステージコスト計算（状態は固定、出力重みのみ変化）
      for t in 1:T
        current_x = view(state_trajectory, :, t)
        wout_slice = view(Wrand, k2, t, :, :)
        y = ESN.reservoir_output(current_x, wout_slice)
        S_k2 += controller_c(params, y, ref_output) + param_gamma * input_costs[t]
      end
      
      # 終端コスト
      final_x = view(state_trajectory, :, T)
      final_wout_slice = view(Wrand, k2, T, :, :)
      y_final = ESN.reservoir_output(final_x, final_wout_slice)
      S_k2 += controller_phi(params, y_final, ref_output)
      
      S[k, k2] = S_k2
    end
  end
  return S
end


function controller_compute_weights(params, S, K2)
  param_lambda = params["LAMBDA"]
  rho = minimum(S)
  eta = sum(exp.((-1.0 / param_lambda) .* (S .- rho)))
  w = (1.0 / eta) .* exp.((-1.0 / param_lambda) .* (S .- rho))
  return sum(w, dims=2)
end

function moving_average_filter(X, filter_size)
  """moving average filter (cumsum optimized)"""
  m, len = size(X)
  Y = similar(X, Float64)

  if filter_size <= 1
    return X
  end

  integral_image = hcat(zeros(m), cumsum(X, dims=2))

  back_delta = div(filter_size, 2)
  forward_delta = isodd(filter_size) ? div(filter_size, 2) : div(filter_size, 2) - 1

  for n = 1:len
    lo = max(1, n - back_delta)
    hi = min(len, n + forward_delta)

    sum_val = integral_image[:, hi+1] .- integral_image[:, lo]
    Y[:, n] = sum_val ./ (hi - lo + 1)
  end
  return Y
end


function controller_calc_W(params, K2, T, Wout, P, W_buf, plant_params, rng=Random.GLOBAL_RNG)
  esn_sigma = params["SIGMA2"]
  cov_matrix = esn_sigma * Matrix(P)
  for i in 1:plant_params["L"]
    try
      mvn = MvNormal(Wout[i, :], cov_matrix)
      for t in 1:T
        for k in 1:K2
          W_buf[k, t, i, :] = rand(rng, mvn)
        end
      end
    catch
      for t in 1:T
        for k in 1:K2
          W_buf[k, t, i, :] = Wout[i, :]
        end
      end
    end
  end
  return W_buf
end

function controller_calc_obs_noise(params, K2, T, plant_params)
  return zeros(K2, T, plant_params["L"])
end

function controller_calc_input(params, u_prev, x0, ref_output, epsilon, Wout, P, plant_params, W, Win, ACT, leakrate; reference_input=0.0, mppi_covar_rng=Random.GLOBAL_RNG)
  """calculate optimal control input (CPU Final Optimized Version)"""
  K = params["SAMPLE"]
  K2 = params["SAMPLE2"]
  T = params["HORIZON"]
  Filter_size = params["FIL_SIZE"]
  u = u_prev

  Wrand = zeros(K2, T, size(Wout)[1], size(Wout)[2])
  Wrand = controller_calc_W(params, K2, T, Wout, P, Wrand, plant_params, mppi_covar_rng)

  obs_noise = controller_calc_obs_noise(params, K2, T, plant_params)

  S = zeros(K, K2)
  S = controller_S_optimized(params, x0, u, epsilon, ref_output, K, T, K2, Wrand, S, obs_noise, W, Win, ACT, leakrate, reference_input=reference_input)

  w = controller_compute_weights(params, S, K2)

  # メモリ効率的な重み付きepsilon計算（一時配列を使わず直接計算）
  w_epsilon = zeros(Float64, size(epsilon, 1), size(epsilon, 3))
  for j in 1:size(epsilon, 2)
    weight = w[j]
    for t in 1:size(epsilon, 3)
      @inbounds for i in 1:size(epsilon, 1)
        w_epsilon[i, t] += epsilon[i, j, t] * weight
      end
    end
  end

  w_epsilon = moving_average_filter(w_epsilon, Filter_size)
  u .+= w_epsilon

  u = controller_g(params, u, reference_input=reference_input)

  u_prev[:, 1:end-1] .= u[:, 2:end]
  u_prev[:, end] = u[:, end]

  return u[:, 1], u, u_prev, w
end


function calc_cost(params, plant_output_log; reference_outputs=nothing)
  Q = params["Q"]
  default_ref = params["REF_OUT"]
  cost = 0.0
  T = size(plant_output_log, 1)
  for t in 1:T
    ref = reference_outputs === nothing ? default_ref : (ndims(reference_outputs) == 1 ? fill(reference_outputs[t], size(plant_output_log, 2)) : reference_outputs[t, :])
    tmp_diff = ref .- plant_output_log[t, :]
    cost += Q * dot(tmp_diff, tmp_diff)
  end
  return cost
end

function calc_cost_input(params, plant_input_log; reference_input=0.0)
  """calculate input cost"""
  lambda = params["LAMBDA"]
  Sigma_inv = 1.0 / params["SIGMA"]
  cost = 0.0
  for t in 1:length(plant_input_log)
    cost_t = dot(plant_input_log[t][1] .- reference_input, Sigma_inv * (plant_input_log[t][1] .- reference_input))
    cost += cost_t
  end
  return lambda * cost / 2
end

function calc_cost_tot(params, plant_output_log, plant_input_log; reference_input=0.0, reference_outputs=nothing)
  """calculate total cost"""
  cost_output = calc_cost(params, plant_output_log; reference_outputs=reference_outputs)
  cost_input = calc_cost_input(params, plant_input_log, reference_input=reference_input)
  cost = cost_output + cost_input
  return cost
end

end # module