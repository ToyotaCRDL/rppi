module Plant

export plant_update, plant_output

function plant_update(plant_params, x, u)
  dt = plant_params["dt"]
  g = plant_params["g"]

  # 状態 (水位)
  h1, h2, h3, h4 = x
  # 入力
  u1, u2 = u

  # 各タンクのパラメータ
  A1 = plant_params["A1"]
  A2 = plant_params["A2"]
  A3 = plant_params["A3"]
  A4 = plant_params["A4"]
  a1 = plant_params["a1"]
  a2 = plant_params["a2"]
  a3 = plant_params["a3"]
  a4 = plant_params["a4"]
  gamma1 = plant_params["gamma1"]
  gamma2 = plant_params["gamma2"]

  # 微分方程式（各タンクの水位変化）
  dh1 = -(a1 / A1) * sqrt(2 * g * h1) + (a3 / A1) * sqrt(2 * g * h3) + (gamma1 / A1) * u1
  dh2 = -(a2 / A2) * sqrt(2 * g * h2) + (a4 / A2) * sqrt(2 * g * h4) + (gamma2 / A2) * u2
  dh3 = -(a3 / A3) * sqrt(2 * g * h3) + ((1 - gamma2) / A3) * u2
  dh4 = -(a4 / A4) * sqrt(2 * g * h4) + ((1 - gamma1) / A4) * u1

  # Euler 積分による状態更新
  dx = [dh1, dh2, dh3, dh4]
  x_next = x .+ dt .* dx
  return x_next
end

# プラント出力関数（例：下部タンクの水位 h3, h4 を出力とする）
function plant_output(x)
  return [x[1], x[2]]
end


end # module