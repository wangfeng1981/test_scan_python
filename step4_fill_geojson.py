#!/usr/bin/env python3
"""
将CREF和Atten预测数据填入GeoJSON文件

功能：
1. 读取对应的CSV文件（CREF和Atten）
2. 读取对应的GeoJSON文件
3. 将数据填入GeoJSON的features中
4. 保存到输出目录
"""

import argparse
import json
import math
import pandas as pd
import os
from datetime import datetime, timedelta


def _is_finite_number(x):
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def parse_time(time_str):
    """解析时间字符串为datetime对象"""
    try:
        if time_str.endswith('Z'):
            time_str_clean = time_str.replace('Z', '+00:00')
        elif '+' in time_str or (len(time_str) > 6 and time_str[-6] in '+-'):
            time_str_clean = time_str
        else:
            time_str_clean = time_str + '+00:00'
        
        time_obj = datetime.fromisoformat(time_str_clean)
        if time_obj.tzinfo:
            time_obj = time_obj.replace(tzinfo=None)
        return time_obj
    except Exception as e:
        print(f"Error parsing time '{time_str}': {e}")
        return None


def update_geojson_with_predictions(geojson_file, cref_csv_file, atten_csv_file, output_file, low_satellite=False):
    """
    将CREF和Atten预测值填入GeoJSON文件
    
    Args:
        geojson_file: 原始GeoJSON文件路径
        cref_csv_file: CREF预测CSV文件路径
        atten_csv_file: Atten预测CSV文件路径
        output_file: 输出GeoJSON文件路径
        
    Returns:
        bool: 是否成功
    """
    # 读取原始GeoJSON文件
    with open(geojson_file, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    # 读取CREF CSV文件（保留行序，用于按索引回退）
    cref_df = pd.read_csv(cref_csv_file)
    cref_dict = {}
    cref_list = []
    for _, row in cref_df.iterrows():
        time_obj = parse_time(row['Time'])
        if time_obj is not None:
            cref_dict[time_obj] = float(row['CREF'])
        try:
            cref_list.append(float(row['CREF']) if pd.notna(row.get('CREF')) else None)
        except (TypeError, ValueError):
            cref_list.append(None)
    
    # 读取Atten CSV文件
    atten_df = pd.read_csv(atten_csv_file)
    atten_dict = {}
    atten_list = []
    for _, row in atten_df.iterrows():
        time_obj = parse_time(row['Time'])
        if time_obj is not None:
            atten_dict[time_obj] = float(row['Atten'])
        try:
            atten_list.append(float(row['Atten']) if pd.notna(row.get('Atten')) else None)
        except (TypeError, ValueError):
            atten_list.append(None)
    
    # 更新features中的properties
    # LOW: 严格按时间键匹配；无值写 null（不做最临近、不按索引回退）
    tolerance_seconds = 360  # 6分钟容差（360秒）
    strict_low_time_match = bool(low_satellite)
    updated_count = 0
    features = geojson_data.get('features', [])
    for i, feature in enumerate(features):
        properties = feature.get('properties', {})
        time_str = properties.get('t', '')
        
        cref_value = None
        atten_value = None
        
        if time_str:
            time_obj = parse_time(time_str)
            if time_obj is not None:
                if time_obj in cref_dict:
                    cref_value = cref_dict[time_obj]
                elif not strict_low_time_match:
                    closest_cref_time = None
                    min_cref_diff = float('inf')
                    for t in cref_dict.keys():
                        diff = abs((t - time_obj).total_seconds())
                        if diff < min_cref_diff:
                            min_cref_diff = diff
                            closest_cref_time = t
                    if closest_cref_time is not None and min_cref_diff <= tolerance_seconds:
                        cref_value = cref_dict[closest_cref_time]
                
                if time_obj in atten_dict:
                    atten_value = atten_dict[time_obj]
                elif not strict_low_time_match:
                    closest_atten_time = None
                    min_atten_diff = float('inf')
                    for t in atten_dict.keys():
                        diff = abs((t - time_obj).total_seconds())
                        if diff < min_atten_diff:
                            min_atten_diff = diff
                            closest_atten_time = t
                    if closest_atten_time is not None and min_atten_diff <= tolerance_seconds:
                        atten_value = atten_dict[closest_atten_time]
        
        # 非 LOW 回退：GeoJSON 与 CSV 常为“结果时间”vs“过境时间”，时间未匹配且数量一致时按索引对应
        if (not strict_low_time_match) and (cref_value is None or atten_value is None) and i < len(cref_list) and i < len(atten_list):
            if cref_value is None and cref_list[i] is not None:
                cref_value = cref_list[i]
            if atten_value is None and atten_list[i] is not None:
                atten_value = atten_list[i]
        
        properties['CREF'] = cref_value
        properties['Atten'] = atten_value
        feature['properties'] = properties
        updated_count += 1
    
    # 修改文件名前缀：L_ -> LA_
    original_name = geojson_data.get('name', '')
    if original_name.startswith('L_'):
        geojson_data['name'] = 'LA_' + original_name[2:]
    
    # 保存更新后的GeoJSON文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated {updated_count} features")
    print(f"Saved to: {output_file}")
    
    return True


def _parse_iso_time(s):
    """解析 20260307T000000Z 或 2026-03-07T00:00:00Z 为 datetime（无时区）。"""
    if not s:
        return None
    try:
        s = s.strip().replace("Z", "").replace("+00:00", "")
        if "T" in s and "-" in s[:10]:
            return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        if "T" in s and len(s) >= 15:
            return datetime.strptime(s[:15], "%Y%m%dT%H%M%S")
        return None
    except Exception:
        return None


def fill_geojson_from_arrays(
    geojson_data,
    cref_list,
    atten_list,
    output_file,
    result_time_iso=None,
    atten_extra=None,
    write_null_for_missing=False,
    drop_features_without_values=False,
    elevation_list=None,
    azimuth_list=None,
    atten_single_list=None,
):
    """
    用内存中的 CREF/Atten 列表按 feature 索引填入 GeoJSON 并写文件（不读 CSV）。
    cref_list、atten_list 与 features 顺序一致，长度需一致。
    result_time_iso: 若提供（如 20260307T000000Z），则把每个 feature 的 properties["t"] 改为预报时刻（result_time + i*6min）。
    atten_extra: 可选，成对频段衰减，统一键为 Atten_1，如 {"Atten_1": [v0,v1,...]}，与 features 等长。
    write_null_for_missing: 为 True 时，缺失值显式写入 null（与 drop_features_without_values 互斥）。
    drop_features_without_values: 为 True 时，CREF/Atten 均无有效数值的 feature 不写入输出（LOW：只保留有预报的点）。
    elevation_list / azimuth_list: 可选，与 features 等长；写入 properties「elevation」「azimuth」（度，方位角自正北顺时针 0~360）。
    atten_single_list: 可选，单次预报原始衰减（未集成），写入 properties["atten_single"]。
    """
    features = geojson_data.get("features", [])
    n = min(len(features), len(cref_list), len(atten_list))
    result_time = _parse_iso_time(result_time_iso) if result_time_iso else None

    if drop_features_without_values:
        out_features = []
        out_idx = 0
        for i in range(n):
            cref_i = cref_list[i]
            atten_i = atten_list[i]
            if not (_is_finite_number(cref_i) and _is_finite_number(atten_i)):
                continue
            feat = json.loads(json.dumps(features[i]))
            props = dict(feat.get("properties") or {})
            props["CREF"] = float(cref_i)
            props["Atten"] = float(atten_i)
            if atten_single_list is not None and i < len(atten_single_list) and _is_finite_number(atten_single_list[i]):
                props["atten_single"] = float(atten_single_list[i])
            elif write_null_for_missing and atten_single_list is not None:
                props["atten_single"] = None
            if atten_extra:
                for key, values in atten_extra.items():
                    if isinstance(values, (list, tuple)) and i < len(values) and _is_finite_number(values[i]):
                        props[key] = float(values[i])
            if elevation_list is not None and i < len(elevation_list) and _is_finite_number(elevation_list[i]):
                props["elevation"] = float(elevation_list[i])
            if azimuth_list is not None and i < len(azimuth_list) and _is_finite_number(azimuth_list[i]):
                props["azimuth"] = float(azimuth_list[i])
            if result_time is not None:
                t_i = result_time + timedelta(minutes=6 * out_idx)
                props["t"] = t_i.strftime("%Y-%m-%dT%H:%M:%SZ")
                out_idx += 1
            feat["properties"] = props
            out_features.append(feat)
        geojson_data["features"] = out_features
        kept = len(out_features)
    else:
        for i in range(n):
            props = features[i].get("properties", {})
            if cref_list[i] is not None:
                props["CREF"] = float(cref_list[i])
            elif write_null_for_missing:
                props["CREF"] = None
            if atten_list[i] is not None:
                props["Atten"] = float(atten_list[i])
            elif write_null_for_missing:
                props["Atten"] = None
            if atten_single_list is not None and i < len(atten_single_list):
                if _is_finite_number(atten_single_list[i]):
                    props["atten_single"] = float(atten_single_list[i])
                elif write_null_for_missing:
                    props["atten_single"] = None
            if atten_extra:
                for key, values in atten_extra.items():
                    if isinstance(values, (list, tuple)) and i < len(values) and values[i] is not None:
                        props[key] = float(values[i])
                    elif write_null_for_missing:
                        props[key] = None
            if elevation_list is not None and i < len(elevation_list):
                if _is_finite_number(elevation_list[i]):
                    props["elevation"] = float(elevation_list[i])
                elif write_null_for_missing:
                    props["elevation"] = None
            if azimuth_list is not None and i < len(azimuth_list):
                if _is_finite_number(azimuth_list[i]):
                    props["azimuth"] = float(azimuth_list[i])
                elif write_null_for_missing:
                    props["azimuth"] = None
            if result_time is not None:
                t_i = result_time + timedelta(minutes=6 * i)
                props["t"] = t_i.strftime("%Y-%m-%dT%H:%M:%SZ")
            features[i]["properties"] = props
        kept = n

    if geojson_data.get("name", "").startswith("L_"):
        geojson_data["name"] = "LA_" + geojson_data["name"][2:]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)
    print(f"Updated {kept} features (from arrays), saved to: {output_file}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fill CREF and Atten predictions into GeoJSON file')
    parser.add_argument('--geojson_file', type=str, required=True, help='Input GeoJSON file path')
    parser.add_argument('--cref_csv', type=str, required=True, help='CREF prediction CSV file path')
    parser.add_argument('--atten_csv', type=str, required=True, help='Atten prediction CSV file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output GeoJSON file path')
    parser.add_argument('--low_satellite', action='store_true', help='Use nearest time only for CREF/Atten match (no strict tolerance)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.geojson_file):
        print(f"Error: GeoJSON file not found: {args.geojson_file}")
        exit(1)
    if not os.path.exists(args.cref_csv):
        print(f"Error: CREF CSV file not found: {args.cref_csv}")
        exit(1)
    if not os.path.exists(args.atten_csv):
        print(f"Error: Atten CSV file not found: {args.atten_csv}")
        exit(1)
    
    success = update_geojson_with_predictions(
        args.geojson_file,
        args.cref_csv,
        args.atten_csv,
        args.output_file,
        low_satellite=getattr(args, 'low_satellite', False)
    )
    
    if success:
        print("Successfully filled GeoJSON file")
    else:
        print("Failed to fill GeoJSON file")
        exit(1)
