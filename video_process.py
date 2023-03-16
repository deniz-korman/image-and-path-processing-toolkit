import util
import yaml
import argparse


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
    "fourier_masker_center"
]

def main(conf):
    src = conf["image_folder"]
    if conf["flat_folder"] is not None:
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


    for stage in conf['pipeline']:
        print("Running stage: ", stage['name'])
        func = getattr(util, stage['name'])

        if stage['name'] in single_funcs:
            vid = util.process_video(vid, func, *stage["params"], **stage["kwargs"])
        else:
            vid = func(vid, *stage["params"], **stage["kwargs"])

    print("Writing video to: videos/", conf['output_name'])
    util.write_video(vid, conf['output_name'])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file")    
                    
    args = parser.parse_args()

    with open(args.config, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    main(data["trial"])