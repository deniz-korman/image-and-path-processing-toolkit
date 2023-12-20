# This is to cover the very specific steps I took in completing a full test of these 2 pipelines

1. download images from oneDrive.
    - you will need to organize them as follows: One parent folder containing a number of folders (on for each spider). In each of these spider specific folders there should be all of the .tif images. So in short `Parent_folder(N spider_folders(M images))`.
  

2. Once this is done you can process all of the images with a single call, this is the exact call I used: `python .\video_process.py -c .\configs\video_processing\temp_bilateral.yaml -i "F:\spiders\onedrive\extracted" -o "large" -b 0 -n 400 -r` It is important you include -r for recursive otherwise it will not search subfolders for images. Another imporant note is that this config contains the `all_batches: True` which is why it will process all images in each folder, normally it will only process n images.
    - once this is complete you will have a large number of folders with the prefix `large` or whatever you set with `-o`. The middle of the file name will be the spider these videos came from, and the end will be an integer to show its place in the order. For example `large_Spider27_Habro16_gap45_gain20_AAAnorest_output_3` is from the `large` run with images from `Spider27_Habro16_gap45_gain20_AAAnorest` and this is the 4th video (the videos index starting with 0).
  
3. Now for DeepLabCut (DLC). You have 2 options, A) you can manually add each video to the project one at a time. or B) call `python gather_videos.py -p large` which will gather all videos with the `large` prefix into a single folder named `all_large`.
    - now comes the hard part of hand annotation and this portion is better handled by DLC documentation so I will skip to the end.
  
4. Once you have tracks you can process them fairly easily by calling `python .\path_cleaning.py -c .\configs\path_filtering\kalman_shape_moving_pit.yaml -o new_kalman_pit.mp4` so long as the config file has the correct input location this will sort and filter the tracks into a single large video. there should now be a file called `track_data.csv` which includes all the filtered datapoints.