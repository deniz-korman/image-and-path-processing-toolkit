import video_util
import yaml
import argparse
import os
import numpy as np
from datetime import datetime

single_funcs = [
    "cv_denoise",
    "sobel_2d",
    "cv_sobel",
    "blur",
    "sharpen",
    "median_filter",
    "detect_edge",
    "threshold_img",
    "modify_contrast",
    "get_hog",
    "fourier_masker_low",
    "fourier_masker_vert",
    "fourier_masker_hor",
    "fourier_masker",
    "fourier_masker_center",
    "block_match"
]

def run_pipeline(pipeline, video):
    vid=np.copy(video)
    for stage in pipeline:
        print("Running stage: ", stage['name'])
        func = getattr(video_util, stage['name'])

        if stage['name'] in single_funcs:
            vid = video_util.process_video(vid, func, *stage["params"], **stage["kwargs"])
        else:
            vid = func(vid, *stage["params"], **stage["kwargs"])
    return vid

def write_vid(vids, name, conf):
    name = name.replace(" ", "")
    print("Writing video to: videos/", name)

    files = os.listdir("videos")
    if name in files:
        i = 0
        new_name = name + "_" + str(i)
        while new_name in files:
            i += 1
            new_name = name + "_" + str(i)
        name = new_name

    save_folder = os.path.join("videos", name)
    os.makedirs(save_folder)
    i = 0
    vids = np.array(vids)
    for vid in vids:
        vid_path = str(i) + "_" + name+".mp4"
        video_util.write_video(vid,vid_path , save_folder)
        i+=1
    yaml_file = open(os.path.join(save_folder, "config.yaml"), 'w')
    yaml.dump(conf, yaml_file)

    video_util.plt.imsave(os.path.join(save_folder, "thumbnail.png"), vid[len(vid)//2])

def get_and_process_vid(path, start_index, num_images, stride, crop, conf, flat_path):
    vid = video_util.get_vid(path, start_index, num_images, stride, crop)
    vid = vid.astype(float) / 255.0

    if flat_path != None:
        print("Loading Flat")
        flat = video_util.get_flat_ave(flat_path)
        print("Applying flat to video")
        vid = vid / flat[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]

    if conf["convert_to_gray"]:
        vid = video_util.process_video(vid, video_util.np.min, axis=2)


    if "trials" in conf:
        i = 0
        vids = [run_pipeline(trial, vid) for trial in conf["trials"]]
        write_vid(vids, conf["output_name"], conf)
    else:
        vid = run_pipeline(conf["pipeline"], vid)
        write_vid([vid], conf["output_name"], conf)
    del vid

def main(conf):
    src = conf["image_folder"]
    if conf["flat_folder"] != "None":
        flat_path = conf["flat_folder"]
    else:
        flat_path = None
    print(f"Number of files in designated path: {len(video_util.os.listdir(src))}")

    if ("crop" in conf.keys() and conf["crop"] is not None):
        crop = conf["crop"]
    else:
        crop = ((0,-1), (0,-1))

    if conf['all_batches']:
        start = conf["vid_start_index"]
        num = conf["num_images"]
        base_of = conf['output_name']

        n_images = len(os.listdir(src))
        n_steps = n_images // num
        cur_step = 0
        while start < n_images:
            conf['output_name'] = base_of + "_" + str(cur_step)
            print(f"Running step {cur_step} of {n_steps}")
            cur_step += 1
            get_and_process_vid(src, start, num, conf["video_stride"], crop, conf, flat_path)
            start += num
            if (start + num > n_images):
                num = n_images - start

    else:
        get_and_process_vid(src, conf["vid_start_index"], conf["num_images"], conf["video_stride"], crop, conf, flat_path)

def apply_cmd_args(args, data):
    if (args.input):
        data["image_folder"] = args.input
    if (args.flat):
        data["flat_folder"] = args.flat
    if (args.output):
        data["output_name"] = args.output
    if (args.begin):
        data["vid_start_index"] = int(args.begin)
    if (args.num_images):
        data["num_images"] = int(args.num_images)
    if (args.stride):
        data["video_stride"] = int(args.stride)
    if (args.crop == "0"):
        data["crop"] = None
    if (args.no_flat):
        data["flat_folder"] = None
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file")

    parser.add_argument("-i", "--input",        help="path to image folder")
    parser.add_argument("-f", "--flat",         help="path to flat folder")
    parser.add_argument("-o", "--output",       help="output name")
    parser.add_argument("-b", "--begin",        help="image index to begin at")
    parser.add_argument("-n", "--num_images",   help="number of images to read")
    parser.add_argument("-s", "--stride",       help="stride for images")
    parser.add_argument("-cr", "--crop",        help="if cropping should be used, 0 for no 1 for yes (defaults to yes)")
    parser.add_argument("-r", "--recursive",    help="add this flag for the pipeline to be run on all subfolders of image_folder (for instance if you have multiple videos to run at once)", action='store_true')
    parser.add_argument("-nf", "--no_flat",     help="add this flag to remove the flat file usage, useful for running on various videos of different spiders", action='store_true')
    parser.add_argument("-a", "--all_batches",        help="pass this flag if you want it to process all images in the folder one batch at a time", action='store_true')
    args = parser.parse_args()
    print(args)
    videos = []


    if (args.recursive and not args.config):
        print("You cannot use recursive mode with multiple configs, please either specify an input video or config file")

    if (not args.recursive and args.config):
        with open(args.config, "r") as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
        data = apply_cmd_args(args, data)
        main(data)

    elif (args.recursive):
        with open(args.config, "r") as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
        data = apply_cmd_args(args, data)
        folders = [path for path in os.listdir(data["image_folder"]) if os.path.isdir(os.path.join(data["image_folder"], path))]
        data_root = data["image_folder"]
        folder_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
        output_folder = os.path.join("videos", folder_name)

        os.mkdir(output_folder)
        base_name = data["output_name"]
        for folder in folders:
            data["image_folder"] = os.path.join(data_root, folder)
            data["output_name"] = f'{base_name}_{folder}_output'
            print(folder_name)
            print(data["output_name"])
            print(output_folder)
            main(data)


    elif(not args.recursive and not args.config):
        answer = input("No config file provided, would you like to run all config files on your input (y/n): ")
        if "y" in answer.lower():
            configs = os.listdir("configs")
            n = len(configs)
            i = 0
            for config in configs:
                i += 1
                print(f'Running config {i}/{n} named: {config}')
                with open(os.path.join("configs", config), "r") as yamlfile:
                    data = yaml.load(yamlfile, Loader=yaml.FullLoader)
                    data = apply_cmd_args(args, data)
                    data["output_name"] = config[:-5]
                main(data)
    else:
        print("Please provide either a config file with -c or an input folder with -i")