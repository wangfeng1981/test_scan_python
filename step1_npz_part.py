import argparse
import glob
import gzip
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

# 站点中心由 run_ae 通过 --lat --lon 传入（来自 sta.json），不再依赖 stations_config

RAW_NLAT = 4200
RAW_NLON = 6200
LAT_FULL = np.linspace(54.2, 12.2, RAW_NLAT)  # 递减
LON_FULL = np.linspace(73.0, 135.0, RAW_NLON)  # 递增
ROW_BYTES = RAW_NLON * 2  # int16
EXPECTED_BIN_SIZE = ROW_BYTES * RAW_NLAT
_DECOMPRESS_LOCKS = {}
_DECOMPRESS_LOCKS_GUARD = threading.Lock()


def _lock_for_path(path: str) -> threading.Lock:
    with _DECOMPRESS_LOCKS_GUARD:
        lock = _DECOMPRESS_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _DECOMPRESS_LOCKS[path] = lock
        return lock


def build_file_index(data_dir):
    """
    一次扫描 data_dir 下所有 CREF 相关文件，建立 (date_str, time_str) -> (path, needs_unzip)。
    已解压的优先于 .npz/.gz。避免对每个 need_time 重复 glob。
    """
    pattern = os.path.join(data_dir, "YYYH_Z_RADA_C_BABJ_*_P_DOR_ACHN_CREF_*")
    index = {}
    for p in glob.glob(pattern):
        name = os.path.basename(p)
        if "CREF_" not in name:
            continue
        suffix = name.split("CREF_")[-1]
        date_time = suffix.replace(".npz", "").replace(".gz", "").strip()
        if "_" not in date_time or len(date_time) < 15:
            continue
        date_str, time_str = date_time.split("_", 1)
        needs_unzip = p.endswith(".npz") or p.endswith(".gz")
        key = (date_str, time_str)
        if key not in index or (not needs_unzip and index[key][1]):
            index[key] = (p, needs_unzip)
    return index


def find_file_for_target(file_index, data_dir, target_dt, max_gap_min):
    """
    按索引查找某时刻对应的雷达文件，支持时间容差。不重复 glob。
    """
    for offset in range(0, max_gap_min + 1):
        if offset == 0:
            candidates = [target_dt]
        else:
            candidates = [target_dt + timedelta(minutes=offset), target_dt - timedelta(minutes=offset)]
        for dt in candidates:
            date_str = dt.strftime("%Y%m%d")
            time_str = dt.strftime("%H%M%S")
            key = (date_str, time_str)
            if key in file_index:
                return file_index[key][0], file_index[key][1]
    return None, False


def decompress_to_bin(compressed_path):
    """将 .npz 或 .gz（内容为 gzip 流）解压到同路径无后缀的 bin 文件。落盘一次，后续复用。"""
    if compressed_path.endswith(".npz"):
        out_path = compressed_path[:-4]
    elif compressed_path.endswith(".gz"):
        out_path = compressed_path[:-3]
    else:
        return compressed_path
    lock = _lock_for_path(out_path)
    with lock:
        if os.path.exists(out_path) and os.path.getsize(out_path) == EXPECTED_BIN_SIZE:
            return out_path

        tmp_path = f"{out_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        print(f"  [解压] {os.path.basename(compressed_path)} -> {os.path.basename(out_path)}", flush=True)
        with gzip.open(compressed_path, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=2**20)
        got = os.path.getsize(tmp_path)
        if got != EXPECTED_BIN_SIZE:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise ValueError(
                f"decompress size mismatch: {os.path.basename(compressed_path)} -> "
                f"{os.path.basename(out_path)} got={got} expect={EXPECTED_BIN_SIZE}"
            )
        os.replace(tmp_path, out_path)
        print(f"  [解压完成] {os.path.basename(out_path)}", flush=True)
    return out_path


rain = [0, 1, 3, 5, 7, 10, 12, 15, 18, 22, 25, 29, 32, 36, 40, 44, 49, 53, 57, 62, 66, 71, 76, 81, 86, 91, 96, 101, 106, 112, 117, 122, 128, 134, 139, 145, 151, 157, 163, 169, 175, 181, 187, 194, 200, 206, 213, 219, 226, 232, 239, 246, 253, 259, 266, 273, 280, 287, 294, 301, 309, 316, 323, 330, 338, 345, 353, 360, 368, 375, 383, 391, 398, 406, 414, 422, 430, 438, 446, 454, 462, 470, 478, 486, 494, 503, 511, 519, 528, 536, 544, 553, 561, 570, 579, 587, 596, 605, 613, 622, 631, 640, 649, 658, 667, 676, 685, 694, 703, 712, 721, 730, 739, 749, 758, 767, 777, 786, 795, 805, 814, 824, 833, 843, 853, 862, 872, 882, 891, 901, 911, 921, 931, 941, 950, 960, 970, 980, 990, 1001, 1011, 1021, 1031, 1041, 1051, 1061, 1072, 1082, 1092, 1103, 1113, 1123, 1134, 1144, 1155, 1165, 1176, 1186, 1197, 1208, 1218, 1229, 1240, 1250, 1261, 1272, 1283, 1294, 1304, 1315, 1326, 1337, 1348, 1359, 1370, 1381, 1392, 1403, 1414, 1426, 1437, 1448, 1459, 1470, 1482, 1493, 1504, 1516, 1527, 1538, 1550, 1561, 1573, 1584, 1596, 1607, 1619, 1630, 1642, 1653, 1665, 1677, 1688, 1700, 1712, 1724, 1735, 1747, 1759, 1771, 1783, 1795, 1807, 1819, 1831, 1843, 1855, 1867, 1879, 1891, 1903, 1915, 1927, 1939, 1951, 1964, 1976, 1988, 2000, 2013, 2025, 2037, 2050, 2062, 2074, 2087, 2099, 2112, 2124, 2137, 2149, 2162, 2174, 2187, 2200, 2212, 2225, 2238, 2250, 2263, 2276, 2288, 2301, 2314, 2327, 2340, 2353, 2365, 2378, 2391, 2404, 2417, 2430, 2443, 2456, 2469, 2482, 2495, 2508, 2521, 2535, 2548, 2561, 2574, 2587, 2601, 2614, 2627, 2640, 2654, 2667, 2680, 2694, 2707, 2720, 2734, 2747, 2761, 2774, 2788, 2801, 2815, 2828, 2842, 2856, 2869, 2883, 2896, 2910, 2924, 2937, 2951, 2965, 2979, 2992, 3006, 3020, 3034, 3048, 3062, 3075, 3089, 3103, 3117, 3131, 3145, 3159, 3173, 3187, 3201, 3215, 3229, 3243, 3257, 3272, 3286, 3300, 3314, 3328, 3343, 3357, 3371, 3385, 3400, 3414, 3428, 3443, 3457, 3471, 3486, 3500, 3514, 3529, 3543, 3558, 3572, 3587, 3601, 3616, 3630, 3645, 3660, 3674, 3689, 3703, 3718, 3733, 3747, 3762, 3777, 3792, 3806, 3821, 3836, 3851, 3866, 3880, 3895, 3910, 3925, 3940, 3955, 3970, 3985, 4000, 4015, 4030, 4045, 4060, 4075, 4090, 4105, 4120, 4135, 4150, 4165, 4180, 4196, 4211, 4226, 4241, 4256, 4272, 4287, 4302, 4318, 4333, 4348, 4364, 4379, 4394, 4410, 4425, 4440, 4456, 4471, 4487, 4502, 4518, 4533, 4549, 4564, 4580, 4595, 4611, 4627, 4642, 4658, 4674, 4689, 4705, 4721, 4736, 4752, 4768, 4783, 4799, 4815, 4831, 4847, 4862, 4878, 4894, 4910, 4926, 4942, 4958, 4974, 4990, 5006, 5022, 5037, 5053, 5070, 5086, 5102, 5118, 5134, 5150, 5166, 5182, 5198, 5214, 5230, 5247, 5263, 5279, 5295, 5311, 5328, 5344, 5360, 5376, 5393, 5409, 5425, 5442, 5458, 5474, 5491, 5507, 5524, 5540, 5557, 5573, 5589, 5606, 5622, 5639, 5655, 5672, 5689, 5705, 5722, 5738, 5755, 5772, 5788, 5805, 5821, 5838, 5855, 5872, 5888, 5905, 5922, 5938, 5955, 5972, 5989, 6006, 6022, 6039, 6056, 6073, 6090, 6107, 6124, 6141, 6158, 6174, 6191, 6208, 6225, 6242, 6259, 6276, 6293, 6310, 6328, 6345, 6362, 6379, 6396, 6413, 6430, 6447, 6464, 6482, 6499, 6516, 6533, 6551, 6568, 6585, 6602, 6620, 6637, 6654, 6672, 6689, 6706, 6724, 6741, 6758, 6776, 6793, 6811, 6828, 6845, 6863, 6880, 6898, 6915, 6933, 6950, 6968, 6985, 7003, 7021, 7038, 7056, 7073, 7091, 7109, 7126, 7144, 7162, 7179, 7197, 7215, 7233, 7250, 7268, 7286, 7304, 7321, 7339, 7357, 7375, 7393, 7410, 7428, 7446, 7464, 7482, 7500, 7518, 7536, 7554, 7572, 7590, 7608, 7626, 7644, 7662, 7680, 7698, 7716, 7734, 7752, 7770, 7788, 7806, 7824, 7843, 7861, 7879, 7897, 7915, 7933, 7952, 7970, 7988, 8006, 8025, 8043, 8061, 8079, 8098, 8116, 8134, 8153, 8171, 8190, 8208, 8226, 8245, 8263, 8282, 8300, 8318, 8337, 8355, 8374, 8392, 8411, 8429, 8448, 8466, 8485, 8504, 8522, 8541, 8559, 8578, 8597, 8615, 8634, 8653, 8671, 8690, 8709, 8727, 8746, 8765, 8783, 8802, 8821, 8840, 8859, 8877, 8896, 8915, 8934, 8953, 8971, 8990, 9009, 9028, 9047, 9066, 9085, 9104, 9123, 9142, 9161, 9180, 9199, 9218, 9237, 9256, 9275, 9294, 9313, 9332, 9351, 9370, 9389, 9408, 9427, 9446, 9466, 9485, 9504, 9523, 9542, 9561, 9581, 9600, 9619, 9638, 9658, 9677, 9696, 9715, 9735, 9754, 9773, 9793, 9812, 9831, 9851, 9870, 9890, 9909, 9928, 9948, 9967, 9987, 10006, 10026, 10045, 10065, 10084, 10104, 10123, 10143, 10162, 10182, 10201, 10221, 10240, 10260, 10280, 10299, 10319, 10338, 10358, 10378, 10397, 10417, 10437, 10456, 10476, 10496, 10516, 10535, 10555, 10575, 10595, 10614, 10634, 10654, 10674, 10694, 10713, 10733, 10753, 10773, 10793, 10813, 10833, 10853, 10873, 10892, 10912, 10932, 10952, 10972, 10992, 11012, 11032, 11052, 11072, 11092, 11112, 11132, 11152, 11173, 11193, 11213, 11233, 11253, 11273, 11293, 11313, 11334, 11354, 11374, 11394, 11414, 11434, 11455, 11475, 11495, 11515, 11536, 11556, 11576, 11596, 11617, 11637, 11657, 11678, 11698, 11718, 11739, 11759, 11780, 11800, 11820, 11841, 11861, 11882, 11902, 11922, 11943, 11963, 11984, 12004, 12025, 12045, 12066, 12086, 12107, 12128, 12148, 12169, 12189, 12210, 12230, 12251, 12272, 12292, 12313, 12334, 12354, 12375, 12396, 12416, 12437, 12458, 12478, 12499, 12520, 12541, 12561, 12582, 12603, 12624, 12644, 12665, 12686, 12707, 12728, 12749, 12769, 12790, 12811, 12832, 12853, 12874, 12895, 12916, 12937, 12958, 12979, 13000, 13021, 13041, 13062, 13083, 13104, 13126, 13147, 13168, 13189, 13210, 13231, 13252, 13273, 13294, 13315, 13336, 13357, 13378, 13400, 13421, 13442, 13463, 13484, 13506, 13527, 13548, 13569, 13590, 13612, 13633, 13654, 13675, 13697, 13718, 13739, 13761, 13782, 13803, 13825, 13846, 13867, 13889, 13910, 13931, 13953, 13974, 13996, 14017, 14038, 14060, 14081, 14103, 14124, 14146, 14167, 14189, 14210, 14232, 14253, 14275, 14296, 14318, 14339, 14361, 14382, 14404, 14426, 14447, 14469, 14491, 14512, 14534, 14555, 14577, 14599, 14620, 14642, 14664, 14686, 14707, 14729, 14751, 14772, 14794, 14816, 14838, 14860, 14881, 14903, 14925, 14947, 14969, 14990, 15012, 15034, 15056, 15078, 15100, 15122, 15143, 15165, 15187, 15209, 15231, 15253, 15275, 15297, 15319, 15341, 15363, 15385, 15407, 15429, 15451, 15473, 15495, 15517, 15539, 15561, 15583, 15605, 15627, 15650, 15672, 15694, 15716, 15738, 15760, 15782, 15805, 15827, 15849, 15871, 15893, 15916, 15938, 15960, 19999]

RAIN_ARR = np.asarray(rain, dtype=np.int32)
RAIN_MAX_IDX = len(RAIN_ARR) - 2


def rain_to_rada(arr_2d):
    flat = arr_2d.ravel()
    indices = np.searchsorted(RAIN_ARR, flat, side="right") - 1
    indices = np.clip(indices, 0, RAIN_MAX_IDX)
    return indices.reshape(arr_2d.shape).astype(np.int16)


def compute_window(center_lat, center_lon, half_size):
    lat_ds = LAT_FULL[::2]
    lon_ds = LON_FULL[::2]

    i_c = int(np.argmin(np.abs(lat_ds - center_lat)))
    j_c = int(np.argmin(np.abs(lon_ds - center_lon)))

    i0 = max(i_c - half_size, 0)
    i1 = i0 + 2 * half_size
    j0 = max(j_c - half_size, 0)
    j1 = j0 + 2 * half_size

    i1 = min(i1, len(lat_ds))
    j1 = min(j1, len(lon_ds))
    i0 = i1 - 2 * half_size
    j0 = j1 - 2 * half_size

    sub_lat = lat_ds[i0:i1]
    sub_lon = lon_ds[j0:j1]
    return 2 * i0, 2 * i1, 2 * j0, 2 * j1, sub_lat, sub_lon


def read_window(path, i0_raw, i1_raw, j0_raw, j1_raw):
    n_rows = i1_raw - i0_raw
    raw_flat = np.fromfile(
        path,
        dtype=np.int16,
        count=n_rows * RAW_NLON,
        offset=i0_raw * ROW_BYTES,
    )
    expected = n_rows * RAW_NLON
    if raw_flat.size != expected:
        raise ValueError(
            f"incomplete read from {path}: got {raw_flat.size}, expected {expected}, "
            f"offset={i0_raw * ROW_BYTES}, rows={n_rows}"
        )
    raw = raw_flat.reshape(n_rows, RAW_NLON)

    sub_raw = raw[::2, j0_raw:j1_raw:2]

    cref = rain_to_rada(sub_raw).astype(np.float32)
    cref *= 0.1
    np.clip(cref, 0.0, 60.0, out=cref)
    return cref


def read_any_path(path, needs_unzip, i0_raw, i1_raw, j0_raw, j1_raw):
    if needs_unzip:
        path = decompress_to_bin(path)
    return read_window(path, i0_raw, i1_raw, j0_raw, j1_raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--start_str", type=str, required=True, help="start time YYYYmmddHHMMSS")
    parser.add_argument("--end_str", type=str, required=True, help="end time YYYYmmddHHMMSS")
    parser.add_argument("--cpu_count", type=int, required=True)
    parser.add_argument("--sta_name", type=str, required=True)
    parser.add_argument("--lat", type=float, required=True, help="站点纬度（由 run_ae 从 sta.json 传入）")
    parser.add_argument("--lon", type=float, required=True, help="站点经度（由 run_ae 从 sta.json 传入）")
    parser.add_argument("--half_size", type=int, required=True)
    parser.add_argument("--max_gap_min", type=int, default=6, help="时间容差（分钟），找不到精确时刻时在±max_gap_min 内查找")
    parser.add_argument("--print_input_files", action="store_true", help="打印 Step1 输入文件明细（通常只需一个站点打印一次）")
    parser.add_argument("--min_valid_frames", type=int, default=1, help="原始读取成功的最少时次，低于该值则失败")
    args = parser.parse_args()

    center_lat, center_lon = args.lat, args.lon
    start = datetime.strptime(args.start_str, "%Y%m%d%H%M%S")
    end = datetime.strptime(args.end_str, "%Y%m%d%H%M%S")

    i0_raw, i1_raw, j0_raw, j1_raw, sub_lat, sub_lon = compute_window(center_lat, center_lon, args.half_size)
    print(
        f"[step1] 按需解压: 共 {int((end - start).total_seconds() // 360) + 1} 个时刻(约), "
        f"data_dir={os.path.abspath(args.data_dir)} save_dir={os.path.abspath(args.save_dir)} sta_name={args.sta_name}",
        flush=True,
    )
    print(
        f"[window] raw rows {i0_raw}:{i1_raw}, cols {j0_raw}:{j1_raw} -> sub ({len(sub_lat)}, {len(sub_lon)})",
        flush=True,
    )

    need_times = []
    t = start
    while t <= end:
        need_times.append(t)
        t += timedelta(minutes=6)

    print(f"[step1] 建索引 / 按需定位: 共 {len(need_times)} 个时刻", flush=True)
    file_index = build_file_index(args.data_dir)

    seen_path = {}
    source_jobs = []  # [(path, needs_unzip)]
    time_to_job_idx = []  # 与 need_times 对齐，值为 source_jobs 下标或 None
    per_target_sources = []
    for target in need_times:
        found, needs_unzip = find_file_for_target(file_index, args.data_dir, target, args.max_gap_min)
        if found is None:
            per_target_sources.append((target, None, None))
            time_to_job_idx.append(None)
            continue
        per_target_sources.append((target, os.path.abspath(found), needs_unzip))
        dedup_key = found.replace(".npz", "").replace(".gz", "")
        if dedup_key not in seen_path:
            seen_path[dedup_key] = len(source_jobs)
            source_jobs.append((found, needs_unzip))
        time_to_job_idx.append(seen_path[dedup_key])

    if not source_jobs:
        raise SystemExit(
            f"No radar files found for {args.start_str}..{args.end_str} in {args.data_dir} (max_gap_min={args.max_gap_min})"
        )

    timestamps = [t.strftime("%Y%m%d%H%M%S") for t in need_times]

    print(f"[step1] 共 {len(source_jobs)} 个去重雷达源（并行窗口读）", flush=True)
    if args.print_input_files:
        print("[step1] 各时刻在 data_dir 中命中的源文件（未命中则显示 未找到）:", flush=True)
        for t, fp, nu in per_target_sources:
            ts = t.strftime("%Y%m%d%H%M%S")
            if fp is None:
                print(f"  {ts}  — 未找到 (max_gap_min={args.max_gap_min})", flush=True)
            else:
                tag = "需解压(.npz/.gz)" if nu else "直接读取(bin)"
                print(f"  {ts}  [{tag}] {fp}", flush=True)
        print("[step1] 去重后参与读取的 bin/源路径（np.fromfile 窗口读，压缩包将按需解压）:", flush=True)
        for i, (path, nu) in enumerate(source_jobs, 1):
            if nu:
                if path.endswith(".npz"):
                    read_path = path[:-4]
                elif path.endswith(".gz"):
                    read_path = path[:-3]
                else:
                    read_path = path
            else:
                read_path = path
            print(f"  [{i}] {os.path.abspath(read_path)}", flush=True)

    source_results = [None] * len(source_jobs)
    read_errors = []
    with ThreadPoolExecutor(max_workers=max(1, args.cpu_count)) as exe:
        future_to_idx = {
            exe.submit(read_any_path, path, needs_unzip, i0_raw, i1_raw, j0_raw, j1_raw): k
            for k, (path, needs_unzip) in enumerate(source_jobs)
        }
        done = 0
        total = len(future_to_idx)
        for fut in as_completed(future_to_idx):
            k = future_to_idx[fut]
            try:
                source_results[k] = fut.result()
            except Exception as e:
                source_results[k] = None
                read_errors.append((k, str(e)))
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [read] {done}/{total}", flush=True)

    if read_errors:
        print(f"[step1][WARN] 读取失败 {len(read_errors)} 个去重源文件（将按邻近时次补齐）", flush=True)
        for idx, err in read_errors[:5]:
            print(f"  [read_err#{idx}] {err}", flush=True)

    # 对齐到 need_times：命中且读取成功则用源帧，否则记为缺失
    aligned_frames = []
    for src_idx in time_to_job_idx:
        if src_idx is None:
            aligned_frames.append(None)
        else:
            aligned_frames.append(source_results[src_idx])

    valid_idx = [i for i, x in enumerate(aligned_frames) if x is not None]
    if len(valid_idx) < max(1, args.min_valid_frames):
        raise SystemExit(
            f"[step1] valid frames too few: {len(valid_idx)} < min_valid_frames={args.min_valid_frames}"
        )

    # 缺失时刻用最近邻有效时次补齐，保证输出时间维与 need_times 一致
    miss_cnt = 0
    if len(valid_idx) < len(aligned_frames):
        for i, frame in enumerate(aligned_frames):
            if frame is not None:
                continue
            miss_cnt += 1
            near = min(valid_idx, key=lambda j: abs(j - i))
            aligned_frames[i] = aligned_frames[near].copy()
        print(f"[step1][WARN] 缺失/坏文件时次 {miss_cnt} 个，已按最近邻有效时次补齐", flush=True)

    cref_sub = np.stack(aligned_frames, axis=0)

    sub_ds = xr.Dataset(
        data_vars={"CREF": (("time", "lat", "lon"), cref_sub)},
        coords={
            "time": timestamps,
            "lat": sub_lat,
            "lon": sub_lon,
        },
        attrs={
            "description": "Composite reflectivity",
            "units": "dBZ",
        },
    )

    os.makedirs(args.save_dir, exist_ok=True)
    out_nc = f"{args.save_dir}/CREF_{args.start_str}_{args.sta_name}.nc"
    out_nc_abs = os.path.abspath(out_nc)
    print(f"[step1] 输出文件: {out_nc_abs}", flush=True)
    sub_ds.to_netcdf(out_nc_abs)

    print("Saved:", out_nc_abs)
