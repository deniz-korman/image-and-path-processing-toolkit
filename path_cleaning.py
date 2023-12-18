import path_util
import yaml
import argparse
import os
import numpy as np
from datetime import datetime
import pandas as pd

req_link_funcs = [
    "jump_filter",
    "from_fixed_filter",
    "conf_filter",
]


def gather_data_paths(folder):
    tracked_vids = [os.path.join(folder, f) for f in os.listdir(folder) if "_filtered_labeled" in f][3:5]
    raw_vids = [os.path.join(folder, f) for f in os.listdir(folder) if "labeled" not in f and ".mp4" in f][3:5]
    data_files = [os.path.join(folder, f) for f in os.listdir(folder) if "_filtered.h5" in f][3:5]

    return data_files, raw_vids, tracked_vids


def extract_points(data_files, points):
    subs = [[] for _ in points]

    print(subs)
    
    for dfile in data_files:
        # Load the data
        df = pd.read_hdf(dfile)
        npdf = df.to_numpy()

        for i, pt_name in enumerate(points):
            sub = df.iloc[:, df.columns.get_level_values(1)==pt_name]
            subs[i].append(sub)

    subs = np.array(subs)
    subs = subs.reshape(subs.shape[0], -1, 3)

    print(subs.shape)

    return subs

def run_filter_pipeline(pipeline, path_data, point_names):
    path = np.copy(path_data)
    for stage in pipeline:
        print("=====================")
        print(stage)
        print()
        new_path = np.copy(path)

        if "targets" in stage:
            targs = stage['targets']
            indices = [point_names.index(point) for point in point_names if point in targs]
            new_path = new_path[indices]
            
            
        if stage['name'] in req_link_funcs:
            print("Running Linked Stage: ", stage['name'])
            func = getattr(path_util, stage['name'])
            mask = func(new_path, *stage["params"], **stage["kwargs"])

            regions = path_util.get_fix_regions(mask, 1)
            
            if stage['linker']:
                params = []
                kwargs = {}
                if "linker_params" in stage.keys():
                    params = ["linker_params"]
                if "linker_kwargs" in stage.keys():
                    params = ["linker_kwargs"]
                if stage['linker'] == "spline":
                    new_path = path_util.link_regions_spline(new_path, regions, *params, **kwargs)
                    
                elif stage['linker'] == "lerp":
                    new_path = path_util.link_regions_lerp(new_path, regions, *params, **kwargs)
        else:
            print("Running stage: ", stage['name'])
            func = getattr(path_util, stage['name'])
            new_path = func(new_path, *stage["params"], **stage["kwargs"])
            
        if "targets" in stage:
            for i, idx in enumerate(indices):
                path[idx] = new_path[i]
        else:
            path = new_path
        
    return path

def main(conf):
    src_folder = conf["base_video_folder"]

    data_files, raw_vids, tracked_vids = gather_data_paths(src_folder)

    points = extract_points(data_files, conf["points"])

    points = run_filter_pipeline(conf["pipeline"], points, conf["points"])

    path_util.make_video_set(raw_vids, points, conf["output_name"])

    print(f"Process complete and video saved at: {conf['output_name']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file")
    parser.add_argument("-i", "--input",        help="path to image folder")
    parser.add_argument("-o", "--output",       help="output name")

    args = parser.parse_args()
    with open(args.config, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)        
    # data = apply_cmd_args(args, data)

    if args.input:
        data['base_video_folder'] = args.input
    if args.output:
        data['output_name'] = args.output
    
    main(data)
