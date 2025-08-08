module Utils

using Dates

export load_hyperparameters, save_data, generate_result_dirs
export parse_commandline, select_activation_function, check_nan!
export init_logger, log_print

using YAML
using JLD2
using FilePathsBase
using ArgParse
using Logging


# --- AutoFlushLogger: SimpleLoggerをラップしてflushを保証 ---
struct AutoFlushLogger <: AbstractLogger
  wrapped::SimpleLogger
end

# 1.1) 最低出力レベルを委譲
Logging.min_enabled_level(logger::AutoFlushLogger) = Logging.min_enabled_level(logger.wrapped)

# 1.2) ログ出力すべきかどうかを委譲
Logging.shouldlog(logger::AutoFlushLogger, level, _module, group, id) =
  Logging.shouldlog(logger.wrapped, level, _module, group, id)

# 1.3) 実際にメッセージを書き込んだ後、ファイルストリームを flush する
function Logging.handle_message(
  logger::AutoFlushLogger,
  level, message, _module, group, id, file, line;
  kwargs...
)
  # (1) SimpleLogger に書き込みを任せる
  Logging.handle_message(logger.wrapped, level, message, _module, group, id, file, line; kwargs...)
  # (2) その後すぐに flush してバッファを飛ばす
  flush(logger.wrapped.stream)
end


# --- init_logger: ログファイルを開きAutoFlushLoggerをセット ---
"""
    init_logger(path::AbstractString; min_level=Logging.Info) -> IO

指定した `path` に対して追記モードでログファイルを開き、
AutoFlushLogger をグローバルロガーに設定します。

- `min_level` で出力する最低ログレベルを指定（デフォルトは `Logging.Info`）。
- 開いたファイルストリームを返すので、プログラム終了時などに `close(logfile)` してください。
"""
function init_logger(path::AbstractString; min_level=Logging.Info)
  logfile = open(path, "a")                        # ① 追記モードでファイルを開く
  base_logger = SimpleLogger(logfile, min_level)   # ② SimpleLogger を作成
  flush_logger = AutoFlushLogger(base_logger)      # ③ AutoFlushLogger でラップ
  global_logger(flush_logger)                      # ④ グローバルロガーに設定
  return logfile                                   # ⑤ ファイルストリームを返す
end


# --- log_print: ログと標準出力に即時出力 ---
"""
    log_print(label::AbstractString, value)

ラベル付きでログと標準出力に即時出力する。
"""
function log_print(label::AbstractString, value)
  @info "$label: $value"

  current = current_logger()
  if current isa SimpleLogger
    flush(current.stream)
  elseif current isa AutoFlushLogger
    # AutoFlushLogger.handle_message ですでに flush 済みなので何もしない
  end

  println("$label: ", value)
  flush(stdout)
end


# --- NaNチェック ---
function check_nan!(name::String, value)
  if any(isnan, value)
    error("NaN detected in $name")
  end
end


# --- JLD2でデータ保存 ---
# Function to save data to a JLD2 file
function save_data(data_dir::String, filename::String, data::Dict)
  mkpath(data_dir)
  filepath = joinpath(data_dir, filename * ".jld2")
  @save filepath data
  println("Data saved to: ", filepath)
end

function generate_result_dirs(base_dir::String, prefix::String)
  # base_dir がなければ作る
  mkpath(base_dir)

  # 現在時刻を文字列化 (例: "2025-06-03_14-55")
  current_time = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM")

  # 「タイムスタンプ_プレフィクス」 を結合 (例: "2025-06-03_14-55_key3" 
  # prefix が "" の場合は "2025-06-03_14-55_" となるので、呼び出し側で prefix="" なら "_" を含めない運用でも可)
  folder_name = isempty(prefix) ? current_time : string(current_time, "_", prefix)

  # results_dir: base_dir/ folder_name
  results_dir = joinpath(base_dir, folder_name)
  data_dir = joinpath(results_dir, "data")
  fig_dir = joinpath(results_dir, "figures")
  logs_dir = joinpath(results_dir, "logs")
  config_dir = joinpath(results_dir, "config")

  # 下層ディレクトリをすべて作成
  mkpath(data_dir)
  mkpath(fig_dir)
  mkpath(logs_dir)
  mkpath(config_dir)

  return results_dir, data_dir, fig_dir, logs_dir, config_dir
end

# --- YAMLからハイパーパラメータ読込 ---
# Function to load hyperparameters from a YAML file
function load_hyperparameters(filepath::String)
  return YAML.load_file(filepath)
end

# --- コマンドライン引数パース ---
function parse_commandline(arg_definitions)
  s = ArgParseSettings()
  for (arg, (help_text, arg_type, default)) in arg_definitions
    add_arg_table!(s, arg, Dict(
      :help => help_text,
      :arg_type => arg_type,
      :default => default
    ))
  end
  return parse_args(s)
end

# --- cost_totalを集める ---
function collect_cost_array(base_dir::AbstractString)
  cost_array = Float64[]
  for subfolder in readdir(base_dir)
    subdir_path = joinpath(base_dir, subfolder)
    if isdir(subdir_path)
      datafile = joinpath(subdir_path, "data", "data.jld2")
      if isfile(datafile)
        println("Loading: ", datafile)
        contents = JLD2.load(datafile)
        @show keys(contents)
        push!(cost_array, contents["cost_total"])
      else
        @warn "ファイルが見つかりませんでした: $datafile"
      end
    end
  end
  return cost_array
end


"""
collect_key_array(base_dir, key)

指定したディレクトリ配下の各サブフォルダ内にある
`data/data.jld2` ファイルから、
引数 `key` で指定したキーの値を配列としてまとめて返します。

# 引数
- `base_dir::AbstractString`: 探索を開始するベースディレクトリのパス
- `key::AbstractString`: 取得したいデータのキー名

# 戻り値
- 各ファイルから取得した値を格納した `Vector{Any}`
"""
function collect_key(base_dir::AbstractString, key::AbstractString)
  values = Any[]
  for subfolder in readdir(base_dir)
    subdir = joinpath(base_dir, subfolder)
    if isdir(subdir)
      # println("Processing subdirectory: ", subdir)
      datafile = joinpath(subdir, "data", "data.jld2")
      if isfile(datafile)
        println("Loading: ", datafile)
        try
          # println("Attempting to load key '$key' from: $datafile")
          contents = JLD2.load(datafile)
          # println("Keys in $datafile: ", keys(contents))
          if haskey(contents, key)
            push!(values, contents[key])
            # println("Key '$key' found in: $datafile")
            # @show contents[key]
          else
            # println("Key '$key' not found in: $datafile")
            @warn "キー '$key' が見つかりませんでした: $datafile"
          end
        catch e
          # println("Error loading $key from $datafile: $e")
          @warn "ファイルの読み込みに失敗しました: $datafile" exception = (e, catch_backtrace())
          continue
        end
      else
        # println("Skipping: $datafile (not a file)")
        @warn "ファイルが見つかりませんでした: $datafile"
      end
    end
  end
  return values
end


function safe_collect_key(base_dir, key)
  try
    return collect_key(base_dir, key)
  catch e
    println("Error collecting key '$key' from base directory '$base_dir': $e")
    @warn "Failed to load $key from $base_dir: $e"
    return []
  end
end


function calc_identification_error_time(plant_output_log, esn_output_log)
  error_ar = zeros(length(plant_output_log[:, 1]))
  for t in 1:length(plant_output_log[:, 1])
    error_ar[t] = (plant_output_log[t] .- esn_output_log[t])' * (plant_output_log[t] .- esn_output_log[t])
  end
  return error_ar
end

function calc_identification_error(plant_output_log, esn_output_log, t_length, dt)
  return dt .* sum(calc_identification_error_time(plant_output_log, esn_output_log)[1:t_length])
end

# function calc_control_cost_time(plant_output_log, plant_input_log, Q, R, reference_output, reference_input)
#   cost_ar = zeros(length(plant_output_log[:, 1]))
#   for t in 1:length(plant_output_log[:, 1])
#     cost_out = Q * (reference_output .- plant_output_log[t])' * (reference_output .- plant_output_log[t])
#     cost_in = R * (plant_input_log[t] .- reference_input)' * (plant_input_log[t] .- reference_input) ./ 2
#     cost_ar[t] = cost_out + cost_in
#   end
#   return cost_ar
# end

function calc_control_cost_time(plant_output_log, plant_input_log, Q, R, reference_output, reference_input)
  T = size(plant_output_log, 1)
  cost_ar = zeros(T)
  for t in 1:T
    # reference_outputの型・shapeに応じて参照値を決定
    if isa(reference_output, Number)
      ref_out = reference_output
    else
      ref_out = reference_output[t]
    end
    # reference_inputの型・shapeに応じて参照値を決定
    if isa(reference_input, Number)
      ref_in = reference_input
    else
      ref_in = reference_input[t]
    end
    cost_out = Q * (ref_out .- plant_output_log[t])' * (ref_out .- plant_output_log[t])
    cost_in = R * (plant_input_log[t] .- ref_in)' * (plant_input_log[t] .- ref_in) / 2
    cost_ar[t] = cost_out + cost_in
  end
  return cost_ar
end


function calc_control_cost(plant_output_log, plant_input_log, Q, R, reference_output, reference_input, t_length, dt)
  return dt .* sum(calc_control_cost_time(plant_output_log, plant_input_log, Q, R, reference_output, reference_input)[1:t_length])
end


end # module