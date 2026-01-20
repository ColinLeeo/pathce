#!/usr/bin/env python
import sys
import json
import argparse
import csv
import pathlib
import os

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description=
    "Make vertex ids of a dataset unique (by building a global vertex map)")
parser.add_argument("-d",
                    "--dataset",
                    help="Specify the dataset dir",
                    required=True)
parser.add_argument("-s",
                    "--schema",
                    help="Specify the gCard schema path",
                    required=True)
args = parser.parse_args()

with open(args.schema) as f:
    schema = json.load(f)

next_global_id = 0
global_vertex_map = {}
dir = pathlib.Path(args.dataset)
assert dir.is_dir()

for vertex_label, vertex_label_id in schema["vertex_labels"].items():
    local_vertex_map = {}
    path = dir.joinpath(f"{vertex_label}.csv")
    
    # 读取所有记录，建立映射，同时保存完整记录（包括所有属性）
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames  # 保存所有列名
        for record in reader:
            local_id = int(record["id"])
            local_vertex_map[local_id] = next_global_id
            next_global_id += 1
            # 保存完整记录（包括所有属性）
            records.append((local_id, record))
    
    global_vertex_map[vertex_label_id] = local_vertex_map
    
    # 写入更新后的文件，保留所有列
    new_path = dir.joinpath(f"{vertex_label}.csv.tmp")
    with open(new_path, "w+") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # 按全局ID排序，但保持原始记录的所有属性
        for local_id, record in sorted(records, key=lambda x: local_vertex_map[x[0]]):
            record["id"] = local_vertex_map[local_id]  # 只更新 id 列
            writer.writerow(record)

for edge_label, edge_label_id in schema["edge_labels"].items():
    src_vertex_map = None
    dst_vertex_map = None
    for edge in schema["edges"]:
        if edge["label"] == edge_label_id:
            src_label_id = edge["from"]
            dst_label_id = edge["to"]
            src_vertex_map = global_vertex_map[src_label_id]
            dst_vertex_map = global_vertex_map[dst_label_id]
            break
    path = dir.joinpath(f"{edge_label}.csv")
    edges = []
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames  # 保存所有列名
        for record in reader:
            src = int(record["src"])
            dst = int(record["dst"])
            src_global_id = src_vertex_map[src]
            dst_global_id = dst_vertex_map[dst]
            # 更新 src 和 dst，保留所有其他属性
            record["src"] = src_global_id
            record["dst"] = dst_global_id
            edges.append(record)
    
    new_path = dir.joinpath(f"{edge_label}.csv.tmp")
    with open(new_path, "w+") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)

for vertex_label in schema["vertex_labels"]:
    path = dir.joinpath(f"{vertex_label}.csv")
    new_path = dir.joinpath(f"{vertex_label}.csv.tmp")
    os.rename(new_path, path)

for edge_label in schema["edge_labels"]:
    path = dir.joinpath(f"{edge_label}.csv")
    new_path = dir.joinpath(f"{edge_label}.csv.tmp")
    os.rename(new_path, path)
