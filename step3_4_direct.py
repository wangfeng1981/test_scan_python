#!/usr/bin/env python3
"""
Step3+4 合并：不写 CREF/Atten CSV，在内存中算完直接填 GeoJSON。
每站 NC 与模型各加载一次，减少 I/O 与进程启动。
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 保证可导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xarray as xr
from step3_1_nc2csv_dz_ae import (
    compute_station_cref_parallel,
    _parse_feature_time,
    sat_pos_to_azel,
)
from step3_2_infer import (
    load_atten_model,
    predict_atten_from_cref_arrays,
    predict_atten_batch,
    scale_rain_attenuation,
    FREQ_GHZ,
)

# ======================= 新增 ======================= #
from step4_ensemble_avg import ensemble_atten_csv

from step4_fill_geojson import fill_geojson_from_arrays
import torch


def _satellite_name_from_basename(base):
    """文件名 L_站点_天线_卫星名_开始时间_终止时间，卫星名可能含下划线，取 parts[3:-2]。"""
    parts = base.split("_")
    if len(parts) < 5:
        return ""
    if len(parts) == 5:
        return parts[3]
    return "_".join(parts[3:-2])


def _parse_t_to_comparable(t_str):
    """将 GeoJSON 的 t（如 2026-03-07T00:00:00Z 或 20260307T000000Z）转为可比较字符串 20260307T000000Z。"""
    if not t_str or not isinstance(t_str, str):
        return None
    t_str = t_str.strip().replace("Z", "").replace("z", "").strip()
    # 含毫秒时截断
    if "." in t_str:
        t_str = t_str.split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y%m%dT%H%M%S"):
        try:
            s = t_str[:19] if fmt == "%Y-%m-%dT%H:%M:%S" and len(t_str) >= 19 else t_str
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y%m%dT%H%M%SZ")
        except ValueError:
            continue
    return None


def _filter_geojson_candidates(json_paths, result_time_iso, result_time_plus2h, max_files):
    """
    仅通过读 GeoJSON 筛选：GEO 全部保留；LOW 仅当首尾 feature 的 t 都在 [result_time_iso, result_time_plus2h] 内保留。
    返回保留的路径列表，最多 max_files 条（GEO 在前，LOW 在后）。max_files<=0 表示不限制。
    """
    if not json_paths:
        return []
    geo_list = []
    low_list = []
    for path in json_paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                geo_data = json.load(f)
        except Exception:
            continue
        sat_type = (geo_data.get("satelliteType") or "").strip().upper()
        if sat_type != "GEO" and sat_type != "LOW":
            sat_type = "GEO"
        feats = geo_data.get("features") or []
        if sat_type == "GEO":
            geo_list.append(path)
            continue
        if sat_type == "LOW" and result_time_iso and result_time_plus2h and len(feats) >= 1:
            t_first = (feats[0].get("properties") or {}).get("t")
            t_last = (feats[-1].get("properties") or {}).get("t")
            first_ok = _parse_t_to_comparable(t_first)
            last_ok = _parse_t_to_comparable(t_last)
            if first_ok and last_ok and first_ok >= result_time_iso and first_ok <= result_time_plus2h and last_ok >= result_time_iso and last_ok <= result_time_plus2h:
                low_list.append(path)
        elif sat_type == "LOW" and (not result_time_iso or not result_time_plus2h):
            low_list.append(path)
    out = geo_list + low_list
    if max_files and max_files > 0:
        out = out[:max_files]
    return out


MODEL_FREQ_GHZ = 38.0  # 模型训练频率 (Q 频段)


def _elevation_deg_from_geojson(geo_data):
    """与链路计算相同：sat_pos_to_azel（站址 + 第一个 feature 卫星点）的俯仰角，供 ITU 换算。"""
    try:
        lat_sta = float(geo_data.get("lat", 0))
        lon_sta = float(geo_data.get("lon", 0))
        h_sta = float(geo_data.get("alt", 0))
        feats = geo_data.get("features", [])
        if not feats or "coordinates" not in feats[0].get("geometry", {}):
            return 30.0
        coords = feats[0]["geometry"]["coordinates"]
        lon_sat = float(coords[0])
        lat_sat = float(coords[1])
        h_sat = float(coords[2])
        _az_deg, el_deg = sat_pos_to_azel(lon_sta, lat_sta, h_sta, lon_sat, lat_sat, h_sat)
        return float(el_deg)
    except Exception:
        return 30.0


def _build_elevation_azimuth_lists(geo_data):
    """与 features 顺序一致；与 step3_1 sat_pos_to_azel 同源，不重复实现 ENU。"""
    lat_sta = float(geo_data.get("lat", 0))
    lon_sta = float(geo_data.get("lon", 0))
    h_sta = float(geo_data.get("alt", 0))
    el_out, az_out = [], []
    for feat in geo_data.get("features") or []:
        coords = (feat.get("geometry") or {}).get("coordinates")
        if not coords or len(coords) < 3:
            el_out.append(None)
            az_out.append(None)
            continue
        lon_sat, lat_sat, h_sat = float(coords[0]), float(coords[1]), float(coords[2])
        try:
            az_deg, el_deg = sat_pos_to_azel(lon_sta, lat_sta, h_sta, lon_sat, lat_sat, h_sat)
            az_f = float(az_deg)
            if az_f < 0:
                az_f += 360.0
            el_out.append(float(el_deg))
            az_out.append(az_f)
        except Exception:
            el_out.append(None)
            az_out.append(None)
    return el_out, az_out


def _atten_primary_and_extra(atten_38, freq_band, elevation_deg):
    """
    根据频段得到主衰减与配对频段衰减。
    atten_38: 模型输出 (38 GHz) 的数组。
    freq_band: GeoJSON 的 "freq"，如 "Q", "Ka"（会统一转大写再判断）。
    返回 (atten_primary_list, atten_extra_dict or None)。
    - Q: 主衰减=38 GHz，配对 Atten_1 (50 GHz)
    - Ka: 主衰减=30 GHz，配对 Atten_1 (14 GHz)
    - V:  主衰减=50 GHz，配对 Atten_1 (38 GHz)
    - Ku: 主衰减=14 GHz，配对 Atten_1 (30 GHz)
    - 其他: 主衰减按 band_ghz 缩放或保持 38，无配对
    输出数值范围限制在 [0, 17.8]。
    """
    ATTEN_MIN = 0.0
    ATTEN_MAX = 17.8

    # 允许缺失值（None/NaN）；LOW 最终在 GeoJSON 中丢弃无有效 CREF/Atten 的 feature
    atten_38 = np.asarray(
        [np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in atten_38],
        dtype=np.float64,
    )
    band_upper = (freq_band or "Q").strip().upper()
    band_ghz = {"Q": 38.0, "KA": 30.0, "KU": 14.0, "V": 50.0}
    primary_ghz = band_ghz.get(band_upper, MODEL_FREQ_GHZ)
    atten_primary = atten_38
    if abs(primary_ghz - MODEL_FREQ_GHZ) > 0.1:
        atten_primary = scale_rain_attenuation(
            atten_38, MODEL_FREQ_GHZ, primary_ghz, elevation_deg
        )

    atten_primary = np.clip(atten_primary, ATTEN_MIN, ATTEN_MAX)

    atten_extra = None
    if band_upper == "Q":
        atten_v = scale_rain_attenuation(atten_38, MODEL_FREQ_GHZ, FREQ_GHZ["V"], elevation_deg)
        atten_v = np.clip(atten_v, ATTEN_MIN, ATTEN_MAX)
        atten_extra = {"Atten_1": atten_v.tolist() if hasattr(atten_v, "tolist") else list(atten_v)}
    elif band_upper == "KA":
        atten_ku = scale_rain_attenuation(atten_38, MODEL_FREQ_GHZ, FREQ_GHZ["Ku"], elevation_deg)
        atten_ku = np.clip(atten_ku, ATTEN_MIN, ATTEN_MAX)
        atten_extra = {"Atten_1": atten_ku.tolist() if hasattr(atten_ku, "tolist") else list(atten_ku)}
    elif band_upper == "V":
        atten_q = np.clip(atten_38, ATTEN_MIN, ATTEN_MAX)
        atten_extra = {"Atten_1": atten_q.tolist() if hasattr(atten_q, "tolist") else list(atten_q)}
    elif band_upper == "KU":
        atten_ka = scale_rain_attenuation(atten_38, MODEL_FREQ_GHZ, FREQ_GHZ["Ka"], elevation_deg)
        atten_ka = np.clip(atten_ka, ATTEN_MIN, ATTEN_MAX)
        atten_extra = {"Atten_1": atten_ka.tolist() if hasattr(atten_ka, "tolist") else list(atten_ka)}

    primary_list = atten_primary.tolist() if hasattr(atten_primary, "tolist") else list(atten_primary)
    primary_list = [None if (x is None or not np.isfinite(float(x))) else float(x) for x in primary_list]
    if atten_extra:
        for k, vlist in list(atten_extra.items()):
            atten_extra[k] = [None if (x is None or not np.isfinite(float(x))) else float(x) for x in vlist]
    return primary_list, atten_extra


def _load_ensemble_att_map(latest_csv_path):
    """读取 latest 集成 CSV，返回 {time_key: attenuation} 映射。"""
    if not os.path.isfile(latest_csv_path):
        return {}
    try:
        df = pd.read_csv(latest_csv_path)
    except Exception:
        return {}
    if "Time" not in df.columns:
        return {}
    val_col = "Atten" if "Atten" in df.columns else ("Attenuation" if "Attenuation" in df.columns else None)
    if val_col is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        try:
            key = pd.to_datetime(row["Time"]).strftime("%Y-%m-%dT%H:%M:%S")
            out[key] = float(row[val_col])
        except Exception:
            continue
    return out


def _to_time_key(x):
    """任意时间类型 -> 20260331T010203Z，用于严格时间匹配。"""
    try:
        return pd.to_datetime(x).strftime("%Y%m%dT%H%M%SZ")
    except Exception:
        return None


def _series_to_time_map(times, values):
    """把并行数组转为 {time_key: value}。重复时间以最后一个为准。"""
    out = {}
    n = min(len(times), len(values))
    for i in range(n):
        k = _to_time_key(times[i])
        if not k:
            continue
        try:
            v = float(values[i])
            if np.isfinite(v):
                out[k] = v
        except Exception:
            continue
    return out


def build_params_from_geojson(geo_data, sta_name, start_time, n_pos=20):
    """从 GeoJSON 内容构建 compute_station_cref_parallel 所需参数。"""
    lat_sta = geo_data["lat"]
    lon_sta = geo_data["lon"]
    h_sta = geo_data["alt"]
    features = geo_data["features"]
    time_list = []
    coord_sat_list = []
    for feature in features:
        props = feature["properties"]
        geom = feature["geometry"]
        t_val = props.get("t", "")
        time_list.append(t_val)
        coords = geom["coordinates"]
        lon_sat, lat_sat, h_sat = float(coords[0]), float(coords[1]), float(coords[2])
        # 直接用站–星经纬度，不转方位角/俯仰角
        coord_sat_list.append((lon_sat, lat_sat, h_sat))
    is_low = geo_data.get("satelliteType") == "LOW"
    if is_low:
        nsteps = len(coord_sat_list)
        feature_times = [_parse_feature_time(t) for t in time_list]
        for i in range(nsteps):
            if feature_times[i] is None:
                feature_times[i] = start_time + timedelta(minutes=6 * i)
    else:
        nsteps = min(n_pos, len(coord_sat_list))
        feature_times = None
    return {
        "lon_sta": lon_sta,
        "lat_sta": lat_sta,
        "hgt_sta": h_sta,
        "coord_sat_list": coord_sat_list,
        "nsteps": nsteps,
        "feature_times": feature_times,
        "is_low": is_low,
    }


def _process_one_json(
    json_path,
    cref_da,
    start,
    sta_name,
    n_pos,
    target_h,
    inner_max_workers,
    allowed_codes,
    nc_time_keys=None,
):
    """
    单条链路：始终从 GeoJSON 读取站址、卫星位置、satelliteType 等，算 CREF 后返回 (geo_data, json_path, df, params)。
    allowed_codes: 允许的 stationId/file_codes 元组，由调用方传入（如 run_ae 从 JSON 的 file_codes 传入）。
    """
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            geo_data = json.load(f)
    except Exception:
        return None
    json_station_code = geo_data.get("stationId") or geo_data.get("station_id", "")
    if json_station_code not in allowed_codes:
        return None
    if not geo_data.get("features"):
        return None
    params = build_params_from_geojson(geo_data, sta_name, start, n_pos)

    # LOW：仅保留与 NC 时间轴精确对上的 feature，再逐点计算 CREF
    if params.get("is_low", False) and nc_time_keys:
        coord_sat_list = params.get("coord_sat_list", []) or []
        feature_times = params.get("feature_times", []) or []
        keep_coord = []
        keep_times = []
        for c, t in zip(coord_sat_list, feature_times):
            if _to_time_key(t) in nc_time_keys:
                keep_coord.append(c)
                keep_times.append(t)
        params["coord_sat_list"] = keep_coord
        params["feature_times"] = keep_times
        params["nsteps"] = len(keep_coord)

    # 若 LOW 没有任何可对齐时刻，返回空序列
    if params.get("is_low", False) and params.get("nsteps", 0) <= 0:
        df = pd.DataFrame({"Time": [], "CREF": []})
    else:
        df = compute_station_cref_parallel(
            use_ae=False,
            lon_sta=params["lon_sta"],
            lat_sta=params["lat_sta"],
            hgt_sta=params["hgt_sta"],
            coord_sat_list=params["coord_sat_list"],
            h_target_m=target_h,
            start_time=start,
            nc_root=None,
            max_workers=inner_max_workers,
            save_path=None,
            nsteps=params["nsteps"],
            feature_times=params["feature_times"],
            sta_name=sta_name,
            cref_da=cref_da,
        )
    return (geo_data, json_path, df, params)


def main():
    parser = argparse.ArgumentParser(description="Step3+4 direct: CREF+Atten in memory, fill GeoJSON without CSV")
    parser.add_argument("--nc_root", type=str, required=True, help="Directory containing CREF_PRE_*_Station.nc")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, nargs="+", required=True, help="GeoJSON file path(s)")
    parser.add_argument("--start_str", type=str, required=True, help="Start time YYYYmmddHHMMSS")
    parser.add_argument("--sta_name", type=str, required=True)
    parser.add_argument("--json_output_dir", type=str, required=True, help="Output directory for filled GeoJSON")
    parser.add_argument("--result_time_iso", type=str, default="", help="预报结果时刻 ISO（如 20260307T000000Z），GEO 输出文件名与内容用此时刻")
    parser.add_argument("--result_time_end_iso", type=str, default="", help="预报结束时刻 ISO（如 20260307T020000Z），GEO 输出文件名用")
    parser.add_argument("--result_time_plus2h", type=str, default="", help="预报窗口上界 ISO，LOW 链路按 GeoJSON 内 t 筛选时用")
    parser.add_argument("--max_json_per_station", type=int, default=0, help="每站最多处理 GeoJSON 数量，0 表示不限制")
    parser.add_argument("--file_codes", type=str, required=True, help="空格分隔的站点 file_codes（由 run_ae 从 sta.json 传入，用于校验 GeoJSON stationId）")
    parser.add_argument("--cpu", type=int, default=8, help="Workers for CREF parallel")
    parser.add_argument("--n_pos", type=int, default=20)
    parser.add_argument("--target_h", type=float, default=8000)

    # ======================= 新增 ======================= #
    # BUGFIX: 保留该参数仅为兼容 run_ae 传参；0326 原版无此开关（默认执行集成）
    parser.add_argument("--enable_ensemble", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--current_init", type=str, default="", help="当前起报时间 YYYYmmddHHMMSS（不传默认 start_str）")
    parser.add_argument("--interval_min", type=int, default=6)
    parser.add_argument("--lookback_steps", type=int, default=10)
    parser.add_argument("--forecast_steps", type=int, default=20)
    parser.add_argument("--cycle_min", type=int, default=6)
    parser.add_argument("--method", type=str, default="weighted", choices=["mean", "weighted"])
    parser.add_argument("--save_ens_dir", type=str, default="", help="集成 CSV 输出目录，默认 json_output_dir/csv")
    args = parser.parse_args()
    # BUGFIX: 0326 原版在 parse_args 前使用 args，运行会报错；改为 parse 后设默认路径
    if not args.save_ens_dir:
        args.save_ens_dir = os.path.join(args.json_output_dir, "csv")
    os.makedirs(args.save_ens_dir, exist_ok=True)
    # 与 0426 参数习惯保持一致：调用集成函数时沿用 save_dir 命名
    args.save_dir = args.save_ens_dir
    # BUGFIX: 0326 原版使用 args.current_init 但未定义参数；这里支持不传时回落到 start_str
    current_init = args.current_init or args.start_str
    current_dt = datetime.strptime(current_init, "%Y%m%d%H%M%S")

    t0 = time.perf_counter()
    start = datetime.strptime(args.start_str, "%Y%m%d%H%M%S") + timedelta(hours=1)
    start_str_pre = start.strftime("%Y%m%d%H%M%S")
    nc_file = os.path.join(args.nc_root, f"CREF_PRE_{start_str_pre}_{args.sta_name}.nc")
    if not os.path.isfile(nc_file):
        print(f"[step3_4_direct] NC not found: {nc_file}, skip")
        return
    ds = xr.open_dataset(nc_file)
    cref_da = ds["CREF"]
    nc_time_keys = set()
    try:
        nc_time_keys = set(
            k for k in (_to_time_key(t) for t in cref_da["time"].values) if k
        )
    except Exception:
        nc_time_keys = set()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 雨衰模型：model_dir 由调用方传入；权重文件名在 step3_2_infer.load_atten_model 中，换模型改 step3_2_infer.py
    model, norm_data = load_atten_model(args.model_dir, device)
    t1 = time.perf_counter()

    allowed_codes = tuple(s.strip() for s in (args.file_codes or "").split() if s.strip())

    json_paths = args.json_dir if isinstance(args.json_dir, list) else [args.json_dir]
    # 仅通过读 GeoJSON 筛选：GEO 全保留，LOW 按首尾 t 在 [result_time_iso, result_time_plus2h] 内保留，并限制数量
    if args.result_time_iso or args.result_time_plus2h or (args.max_json_per_station and args.max_json_per_station > 0):
        json_paths = _filter_geojson_candidates(
            json_paths,
            args.result_time_iso or "",
            args.result_time_plus2h or "",
            args.max_json_per_station or 0,
        )
    print(f"[step3_4_direct] [{args.sta_name}] 筛后实际处理 {len(json_paths)} 条")
    # 第一轮：多链路并行算 CREF（线程池，共享 cref_da 只读），加快整站耗时
    n_outer = max(1, min(8, len(json_paths)))
    inner_workers = max(1, args.cpu // n_outer)
    items = []
    if n_outer <= 1:
        for json_path in json_paths:
            one = _process_one_json(
                json_path, cref_da, start, args.sta_name, args.n_pos, args.target_h, args.cpu, allowed_codes, nc_time_keys
            )
            if one is not None:
                items.append(one)
    else:
        with ThreadPoolExecutor(max_workers=n_outer) as exe:
            futures = {
                exe.submit(
                    _process_one_json,
                    jp, cref_da, start, args.sta_name, args.n_pos, args.target_h, inner_workers, allowed_codes, nc_time_keys,
                ): jp
                for jp in json_paths
            }
            for fut in as_completed(futures):
                one = fut.result()
                if one is not None:
                    items.append(one)
    t2 = time.perf_counter()
    # 批量 Atten 推理，每批最多 4 条链路（20 站并行时多进程共抢一块 GPU，减小批大小避免 CUBLAS_STATUS_ALLOC_FAILED）
    if not items:
        ds.close()
        return
    atten_batch_size = 4
    for i in range(0, len(items), atten_batch_size):
        chunk = items[i : i + atten_batch_size]
        # LOW 的精确时间筛选已在 _process_one_json 前置完成，这里直接组批
        cref_arrays_list = []
        time_arrays_list = []
        low_satellite_list = []
        batch_pos = []
        for pos, (_, _, df, params) in enumerate(chunk):
            is_low = params.get("is_low", False)
            cref_vals = df["CREF"].values
            time_vals = df["Time"].values
            # 若一条 LOW 前置筛后无可计算时刻，跳过该条（无 CREF 可算）
            if len(time_vals) == 0:
                continue
            batch_pos.append(pos)
            cref_arrays_list.append(cref_vals)
            time_arrays_list.append(time_vals)
            low_satellite_list.append(is_low)

        atten_results = []
        if cref_arrays_list:
            atten_results = predict_atten_batch(
                cref_arrays_list, time_arrays_list, norm_data, model, device, low_satellite_list
            )
        atten_map = {pos: res for pos, res in zip(batch_pos, atten_results)}

        ######################
        for pos, (geo_data, json_path, df, params) in enumerate(chunk):
            time_arr, atten_arr = atten_map.get(pos, (np.array([]), np.array([])))
            is_low = params.get("is_low", False)
            freq_band = (geo_data.get("freq") or "Q").strip().upper()
            base = os.path.basename(json_path).replace(".geojson", "").replace(".json", "")
            if base.startswith("L_"):
                if is_low:
                    # 低轨：按源文件名输出
                    out_basename = "LA_" + base[2:] + ".geojson"
                    ensemble_csv_prefix = os.path.splitext(out_basename)[0]
                elif len(base.split("_")) >= 5 and args.result_time_iso and args.result_time_end_iso:
                    # GEO：输出文件名改为正确预报时间
                    parts = base.split("_")
                    sat = _satellite_name_from_basename(base)
                    prefix = f"{parts[1]}_{parts[2]}_{sat}" if sat else "_".join(parts[1:-2])
                    out_basename = f"LA_{prefix}_{args.result_time_iso}_{args.result_time_end_iso}.geojson"
                    # 集成 CSV 前缀含卫星名，与 LA_{站点}_{天线}_{卫星}_... 一致
                    ensemble_csv_prefix = f"LA_{prefix}"
                else:
                    out_basename = "LA_" + base[2:] + ".geojson"
                    ensemble_csv_prefix = os.path.splitext(out_basename)[0]
            else:
                out_basename = base + "_out.geojson"
                ensemble_csv_prefix = os.path.splitext(out_basename)[0]

            # 先用“单次跑(未集成)”结果写当前 GeoJSON，确保本条链路文件已落盘可被集成扫描到
            atten_38_raw = np.array(atten_arr, dtype=np.float64)
            if is_low:
                cref_map = _series_to_time_map(df["Time"].values, df["CREF"].values)
                atten38_raw_map = _series_to_time_map(time_arr, atten_38_raw)
                cref_list = []
                atten_38_raw_aligned = []
                for feat in (geo_data.get("features") or []):
                    t_raw = (feat.get("properties") or {}).get("t")
                    t_key = _parse_t_to_comparable(t_raw)
                    cref_list.append(cref_map.get(t_key))
                    atten_38_raw_aligned.append(atten38_raw_map.get(t_key))
            else:
                cref_list = df["CREF"].tolist()
                atten_38_raw_aligned = (
                    atten_38_raw.tolist() if hasattr(atten_38_raw, "tolist") else list(atten_38_raw)
                )

            elevation_deg = _elevation_deg_from_geojson(geo_data)
            elev_list, azim_list = _build_elevation_azimuth_lists(geo_data)
            output_file = os.path.join(args.json_output_dir, out_basename)
            atten_list_raw, atten_extra_raw = _atten_primary_and_extra(
                atten_38_raw_aligned, freq_band, elevation_deg
            )
            fill_geojson_from_arrays(
                geo_data, cref_list, atten_list_raw, output_file,
                result_time_iso=(args.result_time_iso or None) if not is_low else None,
                atten_extra=atten_extra_raw,
                write_null_for_missing=False,
                drop_features_without_values=is_low,
                elevation_list=elev_list,
                azimuth_list=azim_list,
                atten_single_list=atten_38_raw_aligned,
            )

            # 再做集成，并用集成结果覆盖回填同名 GeoJSON
            if args.enable_ensemble:
                max_lookback_min = args.forecast_steps * args.interval_min  # 120 min
                earliest_dt = current_dt - timedelta(minutes=max_lookback_min)
                init_times = []
                cur = earliest_dt
                while cur < current_dt:
                    cur += timedelta(minutes=args.cycle_min)
                    init_times.append(cur)

                parts = out_basename.split("_")
                prefix = "_".join(parts[1:4]) if len(parts) >= 4 else os.path.splitext(out_basename)[0]
                ensemble_atten_csv(
                    results_dir=args.json_output_dir,
                    save_dir=args.save_dir,
                    csv_prefix=ensemble_csv_prefix,
                    current_dt=current_dt,
                    init_times=init_times,
                    sta_name=args.sta_name,
                    prefix=prefix,
                    freq_ghz=freq_band,
                    interval_min=args.interval_min,
                    lookback_steps=args.lookback_steps,
                    forecast_steps=args.forecast_steps,
                    method=args.method,
                )
                date_tag = (current_dt + timedelta(minutes=args.lookback_steps * args.interval_min)).strftime("%Y%m%d")
                latest_path = os.path.join(
                    args.save_ens_dir, "latest", date_tag, f"{ensemble_csv_prefix}_{date_tag}_atten_ens.csv"
                )
                ens_map = _load_ensemble_att_map(latest_path)
                if ens_map:
                    atten_38_for_fill = np.array(atten_38_raw, dtype=np.float64)
                    for idx_t, t_val in enumerate(time_arr):
                        try:
                            t_key = pd.to_datetime(t_val).strftime("%Y-%m-%dT%H:%M:%S")
                        except Exception:
                            continue
                        v = ens_map.get(t_key)
                        if v is not None and np.isfinite(v):
                            atten_38_for_fill[idx_t] = float(v)

                    if is_low:
                        atten38_map = _series_to_time_map(time_arr, atten_38_for_fill)
                        atten_38_aligned = []
                        for feat in (geo_data.get("features") or []):
                            t_raw = (feat.get("properties") or {}).get("t")
                            t_key = _parse_t_to_comparable(t_raw)
                            atten_38_aligned.append(atten38_map.get(t_key))
                    else:
                        atten_38_aligned = (
                            atten_38_for_fill.tolist() if hasattr(atten_38_for_fill, "tolist")
                            else list(atten_38_for_fill)
                        )
                    atten_list, atten_extra = _atten_primary_and_extra(
                        atten_38_aligned, freq_band, elevation_deg
                    )
                    fill_geojson_from_arrays(
                        geo_data, cref_list, atten_list, output_file,
                        result_time_iso=(args.result_time_iso or None) if not is_low else None,
                        atten_extra=atten_extra,
                        write_null_for_missing=False,
                        drop_features_without_values=is_low,
                        elevation_list=elev_list,
                        azimuth_list=azim_list,
                        atten_single_list=atten_38_raw_aligned,
                    )
                print("集成更新完成.\n")


    t3 = time.perf_counter()
    ds.close()
    t4 = time.perf_counter()
    load_s = round(t1 - t0, 2)
    cref_s = round(t2 - t1, 2)
    atten_s = round(t3 - t2, 2)
    fill_s = round(t4 - t3, 2)
    total_s = round(t4 - t0, 2)
    print(f"[step3_4_direct] [{args.sta_name}] 加载NC+模型: {load_s}s, CREF: {cref_s}s, Atten: {atten_s}s, 填GeoJSON: {fill_s}s, 合计: {total_s}s")


if __name__ == "__main__":
    main()
