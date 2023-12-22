import path_util
import yaml
import argparse
import os
import numpy as np
from datetime import datetime
import pandas as pd

# List of functions that will require a linker
req_link_funcs = [
    "jump_filter",
    "from_fixed_filter",
    "conf_filter",
]


# Gather all data from the deeplabcut folder, you want to pass the DLC_Project/videos path
def gather_data_paths(folder):
    # This gets all of the video files with tracks
    tracked_vids = [os.path.join(folder, f) for f in os.listdir(folder) if "_filtered_labeled" in f]
    # list of all the raw unprocessed videos
    raw_vids = [os.path.join(folder, f) for f in os.listdir(folder) if "labeled" not in f and ".mp4" in f]
    #list of all the data files from tracking results
    data_files = [os.path.join(folder, f) for f in os.listdir(folder) if "_filtered.h5" in f]
    return data_files, raw_vids, tracked_vids

# extracting the data points themselves from each datafile
def extract_points(data_files, points):
    # make an empty list per data point
    subs = [[] for _ in points]
    # iterate over each file
    for dfile in data_files:
        # Load the data
        df = pd.read_hdf(dfile)
        npdf = df.to_numpy()
        # extract every point and save them based on the key. This is why it is critical that they data points in config match the ones in DLC
        for i, pt_name in enumerate(points):
            sub = df.iloc[:, df.columns.get_level_values(1)==pt_name]
            subs[i].append(sub)

    subs = np.array(subs)
    subs = subs.reshape(subs.shape[0], -1, 3)
    return subs


# This is the main file that actually runs the pipeline from config
def run_filter_pipeline(pipeline, path_data, point_names):
    # copy to not modify data
    path = np.copy(path_data)
    # iterate over each stage
    for stage in pipeline:
        print("=====================")
        print(stage)
        print()
        new_path = np.copy(path)

        # this is to make sure you only run a stage on the targetted points, can be set in config
        if "targets" in stage:
            targs = stage['targets']
            indices = [point_names.index(point) for point in point_names if point in targs]
            new_path = new_path[indices]
            
        # If we require a linker we need to do some extra steps
        if stage['name'] in req_link_funcs:
            print("Running Linked Stage: ", stage['name'])
            # get the function from utils based on the name
            func = getattr(path_util, stage['name'])
            # the functions requiring linkers all generate a boolean mask of valid and invalid points, this is said mask
            mask = func(new_path, *stage["params"], **stage["kwargs"])
            # from them mask we get the first and last valid index for each region of invalidity. With a minimum size as well if you want to allow single points to jump or otherwise be an outlier
            regions = path_util.get_fix_regions(mask, 1)
            
            # checking if they set a linker, technically you could pass no linker and it would not function
            if stage['linker']:
                params = []
                kwargs = {}
                # access and store keywords if they are passed
                if "linker_params" in stage.keys():
                    params = ["linker_params"]
                if "linker_kwargs" in stage.keys():
                    params = ["linker_kwargs"]
                if stage['linker'] == "spline":
                    new_path = path_util.link_regions_spline(new_path, regions, *params, **kwargs)
                
                # this is to default to lerp if you do not set
                elif stage['linker'] == "lerp":
                    new_path = path_util.link_regions_lerp(new_path, regions, *params, **kwargs)
        # for stages like shape filter and kalman that do not need a linke
        else:
            print("Running stage: ", stage['name'])
            func = getattr(path_util, stage['name'])
            new_path = func(new_path, *stage["params"], **stage["kwargs"])
        # save results to targetted points
        if "targets" in stage:
            for i, idx in enumerate(indices):
                path[idx] = new_path[i]
        else:
            path = new_path
        
    return path

# this is the main function that handles most of everything
def main(conf):
    # get video source folder
    src_folder = conf["base_video_folder"]
    # get all data
    data_files, raw_vids, tracked_vids = gather_data_paths(src_folder)
    # get points from datafiles
    points = extract_points(data_files, conf["points"])
    # process tracking points 
    points = run_filter_pipeline(conf["pipeline"], points, conf["points"])
    # make one large video from all the sub videos
    path_util.make_video_set(raw_vids, points, conf["output_name"])

    print(f"Process complete and video saved at: {conf['output_name']}")



if __name__ == "__main__":
    # handle command line args, only 3 are needed for this toolkit
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
