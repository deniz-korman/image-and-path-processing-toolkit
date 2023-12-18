#Imports needed for code to work
import numpy as np
from matplotlib import pyplot as plt
import skvideo
import skvideo.io
from skimage import color, data, filters, restoration
from skimage.segmentation import slic, mark_boundaries, felzenszwalb
from skimage.restoration import denoise_wavelet
from skimage.color import label2rgb
from skimage.feature import hog
from scipy.fftpack import fft2, ifft2
import os
import cv2 as cv2
from scipy import ndimage
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import bm3d

#Utility Methods
def get_vid(src, init, nframes, step=1, crop=((0,-1), (0,-1))):
    files = os.listdir(src)
    files.sort()
    vid = []
    print(src)
    print(len(files))
    for i in tqdm(range(init, init + nframes, step)):
        img = plt.imread(os.path.join(src, files[i]))
        img = img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
        img = img.astype(np.float32, order='C')
        vid.append(img)
    vid = np.array(vid)
    return vid

def get_flat_ave(flats_folder, init=None, nave=None):
    flat_names = os.listdir(flats_folder)
    if (init == None):
        init = 0
    if (nave == None):
        nave = len(flat_names)
    proc_flats = []
    for i in tqdm(range(init, init + nave)):
        proc_flats.append(plt.imread(os.path.join(flats_folder, flat_names[i])))
    proc_flats = np.array(proc_flats)
#     proc_flats =  np.array([plt.imread(os.path.join(flats_folder, flat_names[i])) for i in range(init, init + nave)])
    im1 = proc_flats[0]
    imave = np.zeros_like(im1)
    for i in range(nave):
        imave = imave + proc_flats[i]/nave
    return imave

def norm_vid(images):
    for i in range(len(images)):
        img = images[i]
        top = np.max(img)
        images[i] = img/top
    return images

def rename_images(path):
    files = os.listdir(path)
    renamed = 0
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            name, ext = file.split(".")
            prefix, idx = name.split("_")
            if (len(idx) < 4):
                new_idx = idx.zfill(4)
                new_file = f"{prefix}_{new_idx}.{ext}"
                os.rename(os.path.join(path, file), os.path.join(path, new_file))         
                renamed += 1
    print(f"Renamed {renamed} Files")
    
    
def get_all_files(path):
    files = [os.path.join(path, p) for p in os.listdir(path) if os.path.isfile(os.path.join(path, p))]
    folders = [os.path.join(path, p) for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
    for folder in folders:
        files+=get_all_files(folder)
    return files
    
def merge_folders(path, target=None):
    if (target == None):
        target = path
    files = get_all_files(path)
    for file in files:
        head, tail = os.path.split(file)
        os.rename(file, os.path.join(target, tail))
        
def get_mean(vid, n_0, n):
    frames = vid[n_0:n_0 + n]
    return np.mean(frames, axis=0, dtype=np.float64)

def rgb2gray(rgb, kernel=[0.2989, 0.5870, 0.1140]):
    return np.dot(rgb[...,:3], kernel)

#pass threshold as tuple [min, max]
def show(img, threshold=None):
    if (threshold != None):
        plt.imshow(img, cmap="gray", vmin=threshold[0], vmax=threshold[1])
    else:
        plt.imshow(img, cmap="gray")
        
def frame_norm(vid):
    for img in vid:
        img -= img.min()
        img /= img.max()
    return vid

def std_normalize(vid, stds=1):
    v_std = np.std(vid)
    v_mean = np.mean(vid)

    v_max = v_mean + v_std * stds
    v_min = v_mean - v_std * stds

    vid[vid<v_min] = v_min
    vid[vid>v_max] = v_max

    vid = (vid-np.min(vid))/(np.max(vid) - np.min(vid))
    return vid

def normalize(vid, max_val=1):
    vid = (vid-np.min(vid))/(np.max(vid) - np.min(vid))
    return vid * max_val

def mul(vid, coef):
    return vid * coef

def exp(vid, factor):
    return vid ** factor

def write_video(vid, name, folder="videos"):
    skvideo.io.vwrite(os.path.join(folder, name), vid)

def process_video(vid, func, *args, **kwargs):
    frame_shape = np.array(func(vid[0], *args, **kwargs)).shape
    ret_vid = np.zeros(shape=(len(vid), *frame_shape))
    for i in tqdm(range(vid.shape[0])):
        to_add =  np.expand_dims(func(vid[i],*args, **kwargs), axis=0)
        ret_vid[i] = to_add
        # ret_vid = np.append(ret_vid, to_add, axis=0)
    return ret_vid

def block_match(img, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING):
    ret_img = bm3d.bm3d(img, sigma_psd=sigma_psd, stage_arg=stage_arg)
    return ret_img

def gauss_diff(img, blur_1, blur_2, mix_coef):
    b1 = np.copy(img)
    b2 = np.copy(img)

    b1 = blur(b1, blur_1)
    b2 = blur(b2, blur_2)

    return (1 - mix_coef) * b1 - mix_coef * b2


#Processing Methods for individual images
def cv_denoise(img, strength=2):
    cv_img = np.uint8(img*10)
    # cv_img = ndimage.median_filter(cv_img, 3)
    cv_denoise = cv2.fastNlMeansDenoising(cv_img,None,10, 7, 2)
    return cv_denoise

def sobel_2d(img):
    sx = ndimage.sobel(img, axis=0, mode='constant')
    sy = ndimage.sobel(img, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob

def cv_sobel(img, axis=0):
    sobel_x = np.array([[ -1, 0, 1],
                        [ -2, 0, 2],
                        [ -1, 2, 1]])
    sobel_y = np.array([[ -1, -2, -1],
                        [ 0, 0, 0],
                        [ 1, 2, 1]])
    sobels = [sobel_x, sobel_y]
    output_img = cv2.filter2D(img, -1, sobels[axis])
    return output_img

def blur(img, size=3):
    return ndimage.gaussian_filter(img, size)

def sharpen(img, size=3):
    blurred_f = ndimage.gaussian_filter(img, size)
    
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    alpha = 1000
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened

def median_filter(img, size=3):
    im_med = ndimage.median_filter(img, size)
    return im_med

def detect_edge(img, image_mul=1):
    image = Image.fromarray(np.uint8(img*image_mul))
    image = image.filter(ImageFilter.FIND_EDGES)
    return image

def threshold_img(img, threshold=[0.1,0.8]):
    thresh_min = threshold[0]
    thresh_max = threshold[1]
    sxbinary = np.zeros_like(img)
    sxbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
    return sxbinary

def modify_contrast(img, factor=1.5, img_mul=100):
    #read the image
    im = Image.fromarray(np.uint8(img * img_mul))

    #image brightness enhancer
    enhancer = ImageEnhance.Contrast(im)

    im_output = enhancer.enhance(factor)
    return im_output

def get_hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
    fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                        cells_per_block=cells_per_block, visualize=True)
    return hog_image

def fourier_masker_low(image, i, show=False):
    threshold = 25
    gap = 375
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    center = dark_image_grey_fourier.shape[1] // 2

    dark_image_grey_fourier[:gap, 0:center-threshold] = i
    dark_image_grey_fourier[:gap, center+threshold:] = i

    
    dark_image_grey_fourier[-gap:, 0:center-threshold] = i
    dark_image_grey_fourier[-gap:, center+threshold:] = i

    output_img = abs(np.fft.ifft2(dark_image_grey_fourier))
    if (show):
        fig, ax = plt.subplots(1,3,figsize=(15,15))
        ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
        ax[0].set_title('Masked Fourier', fontsize = f_size)
        ax[1].imshow(image, cmap = 'gray')
        ax[1].set_title('Greyscale Image', fontsize = f_size);
        ax[2].imshow(output_img, 
                        cmap='gray')
        ax[2].set_title('Transformed Greyscale Image', 
                        fontsize = f_size);
    return output_img

def fourier_masker_vert(image, i, show=False,threshold=25,gap=375,f_size=15):
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    center = dark_image_grey_fourier.shape[1] // 2

    dark_image_grey_fourier[:gap, center-threshold:center+threshold] = i
    dark_image_grey_fourier[-gap:,center-threshold:center+threshold] = i
    output_img = abs(np.fft.ifft2(dark_image_grey_fourier))
    if (show):
        fig, ax = plt.subplots(1,3,figsize=(15,15))
        ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
        ax[0].set_title('Masked Fourier', fontsize = f_size)
        ax[1].imshow(image, cmap = 'gray')
        ax[1].set_title('Greyscale Image', fontsize = f_size);
        ax[2].imshow(output_img, 
                          cmap='gray')
        ax[2].set_title('Transformed Greyscale Image', 
                          fontsize = f_size);
    return output_img

def fourier_masker_hor(image, i, show=False):
    threshold = 25
    gap = 450
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    center = dark_image_grey_fourier.shape[0] // 2
    dark_image_grey_fourier[center-threshold:center+threshold, :gap] = i
    dark_image_grey_fourier[center-threshold:center+threshold,-gap:] = i
    output_img = abs(np.fft.ifft2(dark_image_grey_fourier))
    if (show):
        fig, ax = plt.subplots(1,3,figsize=(15,15))
        ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
        ax[0].set_title('Masked Fourier', fontsize = f_size)
        ax[1].imshow(image, cmap = 'gray')
        ax[1].set_title('Greyscale Image', fontsize = f_size);
        ax[2].imshow(output_img, 
                        cmap='gray')
        ax[2].set_title('Transformed Greyscale Image', 
                        fontsize = f_size);
    return output_img


def fourier_masker(image, i=1):
    vert = fourier_masker_vert(image, i)
    final = fourier_masker_hor(vert, i)
    return final


def fourier_masker_center(image, size=5, i=1, show=False):
    threshold = 25
    patch_size = size
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    center_y = dark_image_grey_fourier.shape[1] // 2
    center_x = dark_image_grey_fourier.shape[0] // 2

    dark_image_grey_fourier[center_x-patch_size:center_x+patch_size, center_y-patch_size:center_y+patch_size] = i

    output_img = abs(np.fft.ifft2(dark_image_grey_fourier))
    if (show):
        fig, ax = plt.subplots(1,3,figsize=(15,15))
        ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
        ax[0].set_title('Masked Fourier', fontsize = f_size)
        ax[1].imshow(image, cmap = 'gray')
        ax[1].set_title('Greyscale Image', fontsize = f_size)
        ax[2].imshow(output_img, 
                        cmap='gray')
        ax[2].set_title('Transformed Greyscale Image', 
                        fontsize = f_size)
    return output_img


def laplacian(vid, type=cv2.CV_64F):
     return process_video(vid, cv2.Laplacian, type)

#functions for video (array of images)

def mean_divide_single_frame(vid, image, index, n_frames):
    mean = get_mean(vid, index, n_frames).astype(image.dtype)
    return image / mean

def std_frame(vid, idx, n, upper=0.05, lower=0.031):
    new_img = vid[idx] * (np.std(vid[idx:idx+n], axis=0))
    new_img[new_img>upper] = upper
    new_img[new_img<lower] = 0
    return new_img

def std_divide_video(vid, n_frames):
    return np.array([std_frame(vid, i, n_frames) for i in range(len(vid) - n_frames)])

def mean_divide_video(vid, n_frames):
    return np.array([mean_divide_single_frame(vid, vid[i], i, n_frames) for i in range(len(vid) - n_frames)])

def mean_divide_video_bidir(vid, n_frames_f, n_frames_b):
    ret_vid = []
    dtype = vid[0].dtype
    for i in range(n_frames_b-1, len(vid) - n_frames_b):
        frame_f = get_mean(vid, i, n_frames_f).astype(dtype)
        frame_b = get_mean(vid, i - n_frames_b + 1, n_frames_b).astype(dtype)
        ret_vid.append(frame_f / frame_b)
    return np.array(ret_vid)


def mix_videos(vid_a, vid_b, mix_coef):
    print(vid_b.shape)
    print(vid_a.shape)
    vid_b = vid_b[:len(vid_a)]
    vid_scale = vid_a.mean() / vid_b.mean()
    vid_b = vid_b * vid_scale
    final = vid_a * mix_coef + vid_b * (1 - mix_coef)
    return final


# Full pipeline methods:

def temporal_bilateral_filter(vid):
    lap_vid = process_video(vid, cv2.Laplacian, cv2.CV_64F)
    lap_vid += vid
    lap_vid += process_video(vid, modify_contrast, 3)
    out_vid = lap_vid / 10
    out_vid = out_vid ** 2
    out_vid = process_video(out_vid, blur, 3)

    b_vid = process_video(out_vid.astype(np.uint8).T, cv2.bilateralFilter, 15, 5, 5).T.astype(np.float32)**2/100
    out_vid2 = np.swapaxes(out_vid, 1,2)
    b_vid2 = process_video(out_vid2.astype(np.uint8).T, cv2.bilateralFilter, 15, 5, 5).T.astype(np.float32)**2/100
    b_vid2 = np.swapaxes(b_vid2, 1,2)
    b_vid = b_vid + b_vid2
    b_vid /= 2

    return b_vid
