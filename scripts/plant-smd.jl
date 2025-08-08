module Plant

export plant_update, plant_output

function plant_update(pp, x, u)
  u0 = isa(u, AbstractVector) ? u[1] : u

  c = pp["c"]
  k = pp["k"]
  k_nl = pp["k_nl"]
  m = pp["m"]
  x1, x2 = x

  F_spring = -k * x1 - k_nl * x1^3
  F_damp = -c * x2
  a = (F_spring + F_damp + u0) / m

  x1_next = x1 + pp["dt"] * x2
  x2_next = x2 + pp["dt"] * a
  return [x1_next; x2_next]
end

function plant_output(x)
  return [x[1]]
end

end # module