import util
import yaml
import argparse
import os
import numpy as np

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
        func = getattr(util, stage['name'])

        if stage['name'] in single_funcs:
            vid = util.process_video(vid, func, *stage["params"], **stage["kwargs"])
        else:
            vid = func(vid, *stage["params"], **stage["kwargs"])
    return vid

def write_vid(vids, name, conf):
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
    os.mkdir(save_folder)
    i = 0
    for vid in vids:
        util.write_video(vid, str(i) + "_" + name+".mp4", save_folder)
        i+=1
    yaml_file = open(os.path.join(save_folder, "config.yaml"), 'w')
    yaml.dump(conf, yaml_file)

    util.plt.imsave(os.path.join(save_folder, "thumbnail.png"), vid[len(vid)//2])

def main(conf):
    src = conf["image_folder"]
    if conf["flat_folder"] != "None":
        flat_path = conf["flat_folder"]
    else:
        flat_path = None
    print(f"Number of files in designated path: {len(util.os.listdir(src))}")

    if ("crop" in conf.keys()):
        crop = conf["crop"]
    else:
        crop = ((0,-1), (0,-1))
    vid = util.get_vid(src, conf["vid_start_index"], conf["num_images"], conf["video_stride"], crop)
    vid = vid.astype(float) / 255.0

    if flat_path != None:
        print("Loading Flat")
        flat = util.get_flat_ave(flat_path)
        print("Applying flat to video")
        vid = vid / flat[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]

    if conf["convert_to_gray"]:
        vid = util.process_video(vid, util.np.min, axis=2)


    if "trials" in conf:
        i = 0
        vids = [run_pipeline(trial, vid) for trial in conf["trials"]]
        write_vid(vids, conf["output_name"], conf)
    else:
        vid = run_pipeline(conf["pipeline"], vid)
        write_vid([vid], conf["output_name"], conf)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file")

    parser.add_argument("-i", "--input",        help="path to image folder")
    parser.add_argument("-f", "--flat",         help="path to flat folder")
    parser.add_argument("-o", "--output",       help="output name")
    parser.add_argument("-b", "--begin",        help="image index to begin at")
    parser.add_argument("-n", "--num_images",   help="number of images to read")
    parser.add_argument("-s", "--stride",       help="stride for images")
    args = parser.parse_args()

    with open(args.config, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)


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

    main(data)