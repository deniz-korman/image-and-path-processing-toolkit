import os
import argparse
import shutil
from tqdm import tqdm
def main(prefix, output_name = None, copy = False):
    if output_name is None:
        output_name = "all_" + prefix

    output_folder = os.path.join("videos", output_name) 
    os.makedirs(output_folder, exist_ok=True)

    folders = [os.path.join("videos", f) for f in os.listdir("videos") if f.startswith(prefix)]

    videos = []

    for f in tqdm(folders):
        files = os.listdir(f)
        vid = [file for file in files if ".mp4" in file][0]
        new_name = "".join(vid.split(prefix + "_")[1:])
        vid_path = os.path.join(f, vid)
        new_path = os.path.join(output_folder, new_name)
        videos.append(vid_path)
        if copy:
            shutil.copy(vid_path, new_path)
        else:
            shutil.move(vid_path, new_path)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", help="prefix to search for")
    parser.add_argument("-o", "--output", help="name of folder to put all videos in")
    parser.add_argument("-c", "--copy", help="if passed it copy all videos over instead of simply moving them", action="store_true")

    args = parser.parse_args()
    
    main(args.prefix, args.output, args.copy)
