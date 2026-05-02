import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import json

# ======================== 工具函数 ========================
def str2dt(s):
    """'YYYYMMDDHHmmss' -> datetime"""
    return datetime.strptime(s, "%Y%m%d%H%M%S")

def dt2str(dt):
    """datetime -> 'YYYYMMDDHHmmss'"""
    return dt.strftime("%Y%m%d%H%M%S")

def compute_weights(lead_times_min, method="mean"):
    n = len(lead_times_min)
    if method == "mean" or n == 1:
        return np.ones(n) / n

    eps = 1.0  # 避免除零; 单位与 lead_time 一致 (分钟)
    leads = np.array(lead_times_min, dtype=float)
    w = 1.0 / (leads + eps)
    return w / w.sum()

def _load_json_records(path, fc_start_dt):
    """读取 geojson, 返回 [(tgt_dt, data_dict), ...]"""
    _ = fc_start_dt  # 保留参数仅为兼容调用
    with open(path, "r", encoding="utf-8") as f:
        pre_data = json.load(f)
    records = []
    for feature in pre_data.get("features", []):
        properties = feature.get("properties", {})
        time_str = properties.get("t")
        if time_str is None:
            continue
        tgt_dt = pd.to_datetime(time_str, utc=True).tz_localize(None)
        data_dict = {}
        # 兼容旧文件：无 atten_single 时回退到 Atten
        if "atten_single" in properties:
            data_dict["Atten"] = properties["atten_single"]
        elif "Atten" in properties:
            data_dict["Atten"] = properties["Atten"]
        if "Atten_1" in properties:
            data_dict["Atten_1"] = properties["Atten_1"]
        records.append((tgt_dt, data_dict))
    return records


def _load_csv_records(path, fc_start_dt):
    """读取 csv, 返回 [(tgt_dt, data_dict), ...]"""
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        tgt_dt = pd.to_datetime(row["Time"])
        records.append((tgt_dt, row.drop("Time").to_dict()))
    return records

# ======================== CSV 集成 (衰减) ========================
def ensemble_atten_csv(results_dir, save_dir, csv_prefix, current_dt, init_times,
                       sta_name, prefix, freq_ghz, interval_min, lookback_steps,
                       forecast_steps, method):
    """
    对衰减 CSV 做集成平均, 累积式更新.

    三类时刻:
      - stable:    n_members >= forecast_steps(20步) → 冻结
      - finalized: 目标时刻 < 当前 fc_start, 不会再有新成员 → 冻结
      - active:    目标时刻 >= 当前 fc_start, 可能有新成员 → 重新计算
    """

    # ============================================================
    # Step 1: 确定文件路径, 读取已有集成结果
    # ============================================================
    # 以“当前起报对应的预报窗口起点(fc_start)”分桶，避免跨日时 23:54 落到前一天目录导致次日成员缺失
    fc_start_current_dt = current_dt + timedelta(minutes=lookback_steps * interval_min)
    date_tag = dt2str(fc_start_current_dt)[:8]
    legacy_date_tag = dt2str(current_dt)[:8]
    latest_dir = os.path.join(save_dir, "latest", date_tag)
    os.makedirs(latest_dir, exist_ok=True)

    atten_ens_file = f"{csv_prefix}_{date_tag}_atten_ens.csv"
    latest_path = os.path.join(latest_dir, atten_ens_file)
    legacy_latest_path = os.path.join(
        save_dir, "latest", legacy_date_tag, f"{csv_prefix}_{legacy_date_tag}_atten_ens.csv"
    )

    old_df = pd.DataFrame()
    if os.path.exists(latest_path):
        try:
            old_df = pd.read_csv(latest_path).set_index("Time")
            print(f"  读取已有集成: {len(old_df)} 个时刻")
        except Exception as e:
            print(f"  [WARN] 读取已有文件失败: {e}")
            old_df = pd.DataFrame()
    # 兼容旧分桶：若旧路径有数据则并入，避免切换分桶后跨天首轮丢成员
    if legacy_date_tag != date_tag and os.path.exists(legacy_latest_path):
        try:
            legacy_df = pd.read_csv(legacy_latest_path).set_index("Time")
            if old_df.empty:
                old_df = legacy_df
            else:
                old_df = pd.concat([old_df, legacy_df])
                old_df = old_df[~old_df.index.duplicated(keep="last")]
            print(f"  合并旧分桶集成: {legacy_date_tag} +{len(legacy_df)} 个时刻")
        except Exception as e:
            print(f"  [WARN] 读取旧分桶失败: {e}")

    # ============================================================
    # Step 2: 确定冻结边界, 对已有时刻分类
    #
    # 冻结边界 = 当前起报的 fc_start = current_dt + lookback * interval
    # 目标时刻 T < 冻结边界 → 未来不会有新起报覆盖 T → 冻结
    # 目标时刻 T >= 冻结边界 → 当前或未来起报还会覆盖 T → active
    # ============================================================
    freeze_boundary_dt = fc_start_current_dt

    # 2a. stable: 已达最大成员数
    frozen_stable = set()
    if not old_df.empty and "status" in old_df.columns:
        frozen_stable = set(old_df[old_df["status"] == "stable"].index)

    # 2b. finalized: growing 但目标时刻 < 冻结边界
    frozen_finalized = set()
    if not old_df.empty:
        for t_str in old_df.index:
            if t_str in frozen_stable:
                continue
            t_dt = pd.to_datetime(t_str)
            if t_dt < freeze_boundary_dt:
                frozen_finalized.add(t_str)

    # 合并所有需要跳过的时刻
    skip_times = frozen_stable | frozen_finalized

    if frozen_stable or frozen_finalized:
        print(f"  冻结: {len(frozen_stable)} stable + "
              f"{len(frozen_finalized)} finalized "
              f"(边界={freeze_boundary_dt.strftime('%H:%M')})")

    # ============================================================
    # Step 3: 遍历滑动窗口内的起报, 只收集 active 时刻 (适配JSON和CSV)
    # ============================================================
    target_collection = defaultdict(list)
    found_files = 0

    FC_DURATION = timedelta(minutes=interval_min*(forecast_steps-1))  # 预报时长, 可按需改为由 interval_min 算出

    for init_dt in init_times:
        fc_start_dt = init_dt + timedelta(minutes=lookback_steps * interval_min)
        fc_start_str = dt2str(fc_start_dt)

        use_json = True
        # 构造文件路径
        if use_json:
            fc_start_fmt = fc_start_dt.strftime("%Y%m%dT%H%M%SZ")
            fc_end_fmt = (fc_start_dt + FC_DURATION).strftime("%Y%m%dT%H%M%SZ")

            # 确定好高/低轨json文件的命名规则，当前仅适用于高轨卫星json文件
            # 文件名示例: LA_JMS_AN0101_G02_20260403T080000Z_20260403T095400Z.geojson
            file_path = os.path.join(
                results_dir,
                f"LA_{prefix}_{fc_start_fmt}_{fc_end_fmt}.geojson"
            )
            # print("file_path", file_path)
            loader = _load_json_records
        else:
            file_path = os.path.join(
                results_dir,
                f"Atten_PRE_{fc_start_str}_{sta_name}_{freq_ghz}GHz.csv"
            )
            loader = _load_csv_records

        if not os.path.exists(file_path):
            continue

        try:
            records = loader(file_path, fc_start_dt)
        except Exception as e:
            print(f"  [WARN] 读取失败 {file_path}: {e}")
            continue

        found_files += 1

        # 统一的过滤与收集逻辑
        for tgt_dt, data_dict in records:
            tgt_str = tgt_dt.strftime("%Y-%m-%dT%H:%M:%S")

            if tgt_str in skip_times:
                continue

            lead_min = (tgt_dt - fc_start_dt).total_seconds() / 60.0
            if lead_min < 0:
                continue

            target_collection[tgt_str].append((lead_min, data_dict))

    print(f"  扫描 {found_files} 个起报文件, "
          f"收集到 {len(target_collection)} 个 active 时刻")

    if not target_collection and old_df.empty:
        print(f"  [WARN] {freq_ghz} GHz: 无任何有效数据")
        return

    # ============================================================
    # Step 4: 对 active 时刻做加权集成
    # ============================================================
    max_possible_members = forecast_steps

    new_rows = []
    for tgt_str in sorted(target_collection.keys()):
        items = target_collection[tgt_str]
        if not items:
            continue

        lead_arr = np.array([lt for lt, _ in items])

        # 权重
        if method == "mean" or len(items) == 1:
            weights = np.ones(len(items)) / len(items)
        else:
            eps = 1.0
            raw_w = 1.0 / (lead_arr + eps)
            weights = raw_w / raw_w.sum()

        # 数值列名
        all_keys = set()
        for _, d in items:
            all_keys.update(d.keys())

        result = {
            "Time":         tgt_str,
            "n_members":    len(items),
            "status":       "stable" if len(items) >= max_possible_members
                            else "growing",
        }

        for key in sorted(all_keys):
            vals, ws = [], []
            for i, (_, d) in enumerate(items):
                v = d.get(key, np.nan)
                try:
                    fv = float(v)
                except (ValueError, TypeError):
                    continue
                if not np.isnan(fv):
                    vals.append(fv)
                    ws.append(weights[i])
            if vals:
                ws = np.array(ws)
                ws /= ws.sum()
                result[key] = np.average(vals, weights=ws)

        new_rows.append(result)

    if new_rows:
        new_df = pd.DataFrame(new_rows).set_index("Time")
    else:
        new_df = pd.DataFrame()

    # ============================================================
    # Step 5: 三路合并
    #   A. old_df 中未被重算的行 (stable + finalized) → 原样保留
    #   B. new_df 中 old_df 已有的行 → 用新值覆盖旧 growing
    #   C. new_df 中 old_df 没有的行 → 新增追加
    # ============================================================
    if not old_df.empty:
        if not new_df.empty:
            # A: old_df 中不在 new_df 中的行 → 全部保留
            kept_idx = old_df.index.difference(new_df.index)
            kept_df = old_df.loc[kept_idx].copy()

            # 拼接 A + (B∪C)
            combined_df = pd.concat([kept_df, new_df])
            combined_df = combined_df[
                ~combined_df.index.duplicated(keep='last')]
        else:
            combined_df = old_df.copy()
    else:
        if not new_df.empty:
            combined_df = new_df.copy()
        else:
            print(f"  [WARN] {freq_ghz} GHz: 合并后无数据")
            return

    combined_df = combined_df.sort_index()

    # ============================================================
    # Step 6: 保存 & 统计
    # ============================================================
    combined_df.to_csv(latest_path)
    print(f"  [OK] {latest_path}")


# ======================== 主程序 ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",    type=str, required=True)
    parser.add_argument("--save_dir",       type=str, required=True)
    parser.add_argument("--current_init",   type=str, default='20260418234200', help="本次(刚完成的)起报时刻")
    parser.add_argument("--interval_min",   type=int, default=6)
    parser.add_argument("--lookback_steps", type=int, default=10)
    parser.add_argument("--forecast_steps", type=int, default=20)
    parser.add_argument("--cycle_min",      type=int, default=6)
    parser.add_argument("--sta_name",       type=str, required=True)
    parser.add_argument("--freq",           type=str, default='38.0')
    parser.add_argument("--method",         type=str, default="weighted", choices=["mean", "weighted"])
    parser.add_argument("--prefix",         type=str, default='XA_AN0501_G02')
    # parser.add_argument("--results_dir",    type=str, required=True)
    # parser.add_argument("--save_dir",       type=str, required=True)
    # parser.add_argument("--current_init",   type=str, required=True, help="本次(刚完成的)起报时刻")
    # parser.add_argument("--interval_min",   type=int, required=True)
    # parser.add_argument("--lookback_steps", type=int, required=True)
    # parser.add_argument("--forecast_steps", type=int, required=True)
    # parser.add_argument("--cycle_min",      type=int, required=True)
    # parser.add_argument("--sta_name",       type=str, required=True)
    # parser.add_argument("--freq",           type=str, required=True)
    # parser.add_argument("--method",         type=str, default="weighted", choices=["mean", "weighted"])
    args = parser.parse_args()

    current_dt = str2dt(args.current_init)

    # ================================================================
    # 滑动窗口: 只回溯能覆盖当前预报窗口的起报
    #
    # 当前起报的预报窗口: [current + lookback, current + lookback + forecast]
    # 最早有贡献的起报:   其预报尾部 >= current 的预报起点
    #   即 old_init + lookback + forecast >= current + lookback
    #   => old_init >= current - forecast * interval
    #
    # forecast_steps=20, interval=6min => 回溯 120 分钟 = 20 个起报
    # ================================================================
    max_lookback_min = args.forecast_steps * args.interval_min  # 120 min
    earliest_dt = current_dt - timedelta(minutes=max_lookback_min)

    init_times = []
    cur = earliest_dt
    while cur < current_dt:
        cur += timedelta(minutes=args.cycle_min)
        init_times.append(cur)
        # print(cur)

    # print("max_lookback_min:", max_lookback_min)
    # print("init_times", init_times)
    freq_band = 38.0
    ensemble_csv_prefix = 'Atten_ENS'
    ensemble_atten_csv(
        results_dir=args.results_dir,  # json文件路径
        save_dir=args.save_dir,
        csv_prefix=ensemble_csv_prefix,
        current_dt=current_dt,
        init_times=init_times,
        sta_name=args.sta_name,
        prefix=args.prefix,
        freq_ghz=freq_band,
        interval_min=args.interval_min,
        lookback_steps=args.lookback_steps,
        forecast_steps=args.forecast_steps,
        method=args.method,
    )



# if __name__ == "__main__":
#     main()
