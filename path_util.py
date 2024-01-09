# import deeplabcut
import numpy as np
import cv2
from pykalman import KalmanFilter
import pickle
from scipy.interpolate import CubicHermiteSpline
import os
from tqdm import tqdm
import math
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean

# colors to be used in visualizing
colors = [
    np.array((255, 0, 0)),      # Red
    np.array((0, 255, 0)),      # Green
    np.array((0, 0, 255)),      # Blue
    np.array((255, 255, 0)),    # Yellow
    np.array((0, 255, 255)),    # Cyan
    np.array((255, 0, 255)),    # Magenta
    np.array((255, 165, 0)),    # Orange
    np.array((128, 0, 128)),    # Purple
    np.array((0, 128, 0)),      # DarkGreen
    np.array((128, 0, 0)),      # Maroon
    np.array((0, 128, 128)),    # Teal
    np.array((255, 192, 203)),  # Pink
    np.array((128, 128, 0)),    # Olive
    np.array((75, 0, 130)),     # Indigo
    np.array((127, 255, 212))   # Aquamarine
]

# function to get the euclidean distance between each point and the next, very useful for finding large jumps
def get_deltas(track):
    new = [track[i + 1] - track[i] for i in range(len(track) - 1)]
    return np.array(new)

# class to manage a kalman fiilter for more stable stracks.
class KalmanFilter2D:
    def __init__(self, a=1000, b=10):
        # Initial state (position and velocity)
        self.x = np.array([0, 0, 0, 0], dtype=float)
        
        # State transition matrix (2D constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1]
        ])
        
        # Initial uncertainty
        self.P = np.identity(4) * a
        
        # Process noise (tune as needed)
        self.Q = np.identity(4) * b
        
        # Measurement noise (tune based on sensor accuracy)
        self.R = np.array([
            [10, 0],
            [0, 10]
        ])
        
        # Identity matrix
        self.I = np.identity(4)
    
    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x)
        # Update uncertainty
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0:2]
    
    def update(self, measurement):
        # Measurement update
        y = measurement - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        return self.x[0:2]

# function to create and handle a kaldman_2d object
def kalman_filter_2d(data, a=1000, b=10):
    new_tracks = np.copy(data)
    for i, part in enumerate(data):
        kf = KalmanFilter2D(a, b)
        filtered_points = []
    
        for point in part[:, :2]:
            kf.predict()
            filtered_point = kf.update(point)
            filtered_points.append(filtered_point)
        new_tracks[i, :, :2] = np.array(filtered_points)
    return new_tracks

# This is an important function to generate the final videos
def make_video_set(videos, tracks, name = "video.mp4"):
    # Open the video file
    cap = cv2.VideoCapture(videos[0])
    # get target FPS, width, and heights (from source video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    #output video writier to generate final video
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))
    cap.release()
    i = 0
    # iterate over each video to be combined
    for vid_path in tqdm(videos):
        # read in ground truth video without tracking
        cap = cv2.VideoCapture(vid_path)
        while(cap.isOpened()):
            # get next frame
            ret, frame = cap.read()
            if ret:
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # current frame number
                # for each bodypart we get its position
                for j, bodypart in enumerate(tracks):
                    pt = bodypart[i]
                    x = int(pt[0])
                    y = int(pt[1])

                    color_mul = 1
                    if len(pt) == 3:
                        color_mul = pt[2]
                    color = colors[j] * color_mul
                    # draw a point in the image for each bodypart based on the colors above
                    radius = 15
                    cv2.circle(frame, (x, y), radius, color.tolist(), -1)  # -1 to fill the circle
                i += 1
                out.write(frame)  # Save the frame to the output video
            else:
                cap.release()
    
    # Release everything
    cap.release()
    out.release()

# one of the two linking functions, this one is for a spline which helps to conserve velocity coming in and out of an invalid region but can create artifacts if that velocity is hard to calculate
def link_regions_spline(track, lost_regions, search_size = 4):
    new_track = np.copy(track)
    
    # iterate over the regions
    for region in lost_regions:
        # get the part of the track we want to fix with buffer regions to get velocity
        focus = track[region[0] - search_size : region[1] + search_size + 1]
        # get the pre-invalid region
        f_start = max(0, region[0] - search_size)
        # get the post invalid region
        f_end = min(len(track), region[1] + search_size)

        pre_jump = track[f_start:region[0]]
        post_jump = track[region[1]:f_end]

        # for this type of spline you need start and end position and start and end velocity
        p1 = track[region[0]]
        p2 = track[region[1]]
        v1 = p1 - pre_jump.mean(axis=0)
        v2 = p2 - post_jump.mean(axis=0)

        v1 = p2 - p1
        v1 = v1 / np.linalg.norm(v1)
        v2 = v1

        # get the spline that we will be sampling from
        spline = create_spline(p1, p2, v1, v2)
        # get the number of samples we will need
        n_samples = region[1] - region[0] + 1
        samples = np.linspace(0, 1, n_samples)
        # get the samples from spline
        region_curve = np.array([spline(s) for s in samples])
        # save the new values to the new track
        new_track[region[0]:region[1] + 1] = region_curve
    return new_track

# linker for lerping, more stable but less physically accurate.
def link_regions_lerp(tracks, regions, search_size = 4, max=None):
    new_tracks = []
    # iterate over tracks and regions
    for track, lost_regions in zip(tracks, regions):
        new_track = np.copy(track)
        for region in lost_regions:
            p1 = track[region[0]]
            p2 = track[region[1]]

            if region[0] >= 0 and region[1] < track.shape[0]:
        
                n_samples = region[1] - region[0] + 1
                samples = np.linspace(0, 1, n_samples)
                
                # instead of a spline we simply interpolate between p1 and p2
                region_lerp = [( 1 - s ) * p1 + s * p2 for s in samples]
                if (len(region_lerp) >= 1):
                    new_track[region[0]:region[1] + 1] = region_lerp
        new_tracks.append(new_track)
    return np.array(new_tracks)

# method for filtering points based on their distance to a fixed point, very useful for making sure the pit stays within a reaonsable range from the lens
def filter_from_fixed(track, dmax, dmin, center):
    mask = track - center
    mask = np.linalg.norm(mask, axis=1)
    mask = (mask > dmin) & (mask < dmax)
    return mask

# this converts from the boolean mask to a set of regions
def get_fix_regions(masks, min_size=5):
    n_masks = masks.shape[0]
    n_pts = masks.shape[1]

    all_regions = []
    
    # iterate over the mask
    for mask in masks:
        regions = []
        i = 0
        while i < n_pts:
            val = mask[i]
            # if we found the start of an invalid area
            if val == False:
                top = None
                # we then want to search forward until we find another valid point
                for j in range(n_pts - i):
                    if mask[i + j]:
                        top = i + j
                        break
                # if we managed to find another valid point and it is within our min size add it to a list
                if top:
                    if ( top - i >= min_size):
                        regions.append([i - 1, top])
                    # move to the end of this region and continue
                    i = top
            i += 1
        all_regions.append(regions)
    return all_regions

def calculate_edge_lengths(points):
    """
    Calculate the edge lengths of a polygon formed by connecting the points in order.
    Includes the edge connecting the last point back to the first.
    """
    n_points = len(points)
    edge_lengths = []
    for i in range(n_points):
        # Calculate distance from current point to the next, wrapping around at the end
        edge_length = euclidean(points[i], points[(i + 1) % n_points])
        edge_lengths.append(edge_length)
    return np.array(edge_lengths)
    
def calculate_angles(reference_points):
    """
    Calculate all the angles formed by each triplet of points in the reference points array.

    Parameters:
    reference_points (np.array): An array of reference points of shape (n_points, 2).

    Returns:
    np.array: An array of angles in radians.
    """
    angles = []
    n_points = len(reference_points)
    
    for i in range(n_points):
        # Get the current point and the next two points in the sequence
        p1 = reference_points[i]
        p2 = reference_points[(i + 1) % n_points]  # Wrap around using modulo
        p3 = reference_points[(i - 1 ) % n_points]  # Wrap around using modulo
        
        # Calculate the vectors from p2 to p1 and from p2 to p3
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Compute the angle using the dot product formula
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    
    return np.array(angles)

    
# objective function for matching shape
def shape_objective_function(x, expected_distances, expected_angles, dist_coef = 1, angle_coef = 1, side_weights = None):
    coords = x.reshape((-1, 2))
    n_points = coords.shape[0]
    
    # set default side weights
    if side_weights == None:
        side_weights = np.ones(n_points)
    else:
        side_weights /= np.sum(side_weights) 
        side_weights *= side_weights.shape[0]

    # set default angle weights
    if angle_weights == None:
        angle_weights = np.ones(n_points)
    else:
        angle_weights /= np.sum(angle_weights) 
        angle_weights *= angle_weights.shape[0]

    # get distances for each edge points
    distances = calculate_edge_lengths(coords)
    # get difference between real and ideal distances
    distance_penalties = distances - expected_distances
    # adjust for edge weights
    distance_penalties *= side_weights
    # sum edge penalties
    distance_penalties = np.sum(distance_penalties**2)
    
    angle_penalties = 0
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % n_points]  # Wrap around using modulo
        p3 = coords[(i - 1 ) % n_points]  # Wrap around using modulo
        # Calculate the vectors from p2 to p1 and from p2 to p3
        v1 = p2 - p1
        v2 = p3 - p1

        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))        
        # Calculate angle penalty
        angle_dif = angle - expected_angles[i]
        angle_dif *= angle_weights[i]
        angle_penalties += ((angle_dif)**2)
        
    # Sum of squared penalties
    penalties = distance_penalties * dist_coef + angle_penalties * angle_coef
    return penalties

# very similar to above but with some extra steps
def linear_pit_objective_function(x, expected_distances, expected_angles, dist_coef = 1, angle_coef = 1, side_weights = None, angle_weights = None):
    coords = x.reshape((-1, 2))
    n_points = coords.shape[0]
    if side_weights == None:
        side_weights = np.ones(n_points)
    else:
        side_weights /= np.sum(side_weights) 
        side_weights *= side_weights.shape[0]

    if angle_weights == None:
        angle_weights = np.ones(n_points)
    else:
        angle_weights /= np.sum(angle_weights) 
        angle_weights *= angle_weights.shape[0]

    distances = calculate_edge_lengths(coords)    
    distance_penalties = distances - expected_distances
    distance_penalties[0] = distance_penalties[-1] = 0

    distance_penalties *= side_weights
    distance_penalties = np.sum(distance_penalties**2)
    angle_penalties = 0

    # we first need to calculate the midpoints of the waist and shoulders to establish a line for the eye tube
    waist_midpoint = (coords[2] + coords[3]) / 2
    shoulder_midpoint = (coords[1] + coords[4]) / 2

    # we next use some linear algebra to find the distance of the pit from that line
    d = np.linalg.norm(np.cross(waist_midpoint-shoulder_midpoint, shoulder_midpoint-coords[0]))/np.linalg.norm(waist_midpoint-shoulder_midpoint)
    # and add said distance to the penalties.
    distance_penalties += d
    for i in range(1, len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % n_points]  # Wrap around using modulo
        p3 = coords[(i - 1 ) % n_points]  # Wrap around using modulo
        # Calculate the vectors from p2 to p1 and from p2 to p3
        v1 = p2 - p1
        v2 = p3 - p1

        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))        
        # Calculate angle penalty
        angle_dif = angle - expected_angles[i]
        angle_dif *= angle_weights[i]
        angle_penalties += ((angle_dif)**2)
    # Sum of squared penalties
    penalties = distance_penalties * dist_coef + angle_penalties * angle_coef
    return penalties

    
# Define an optimization function for each frame
def optimize_frame(objective_func, frame_points, expected_distances, expected_angles, coefs, side_weights = None, angle_weights = None):
    # Flatten the array of points
    # x0 = frame_points.flatten()
    conf = frame_points[:, 2]
    x0 = frame_points[:, :2]
    # Minimize the objective function
    result = minimize(
        objective_func, 
        x0, 
        args=(
            expected_distances, 
            expected_angles, 
            coefs[0], 
            coefs[1], 
            side_weights, 
            angle_weights),
        method='SLSQP')

    optimized_points = result.x.reshape(x0.shape)

    # Merging the points and confidences into one array
    optimized_points = np.column_stack((optimized_points, conf))
    
    # Reshape the result to the original shape
    return optimized_points

# front facing function to actually conduct the shape matching, this is what is called form the config files
def shape_filter(path, dist_coef = 1, angle_coef = 1, base_pt = None, side_weights = None, angle_weights = None, objective = None):
    # I included a base point from my testing, it should be close to ideal across videos but you may find better results with including your own in the config
    if base_pt is None:
        base_pt = np.array(
            [[190.25768647, 453.522568],
            [325.70858745, 574.28230165],
            [631.20260825, 537.41828323],
            [599.03480258, 275.56768098],
            [300.85975782, 288.11255928]]
        )
    else:
        base_pt = np.array(base_pt)
    if objective is None:
        objective_func = shape_objective_function
    else:
        objective_func = globals()[objective]

    # get ground truth values fro mbase point
    expected_distances = calculate_edge_lengths(base_pt)
    expected_angles = calculate_angles(base_pt)
    
    optimized_points_list = []
    j = 0
    # for each frame optimize it based on the ground truth
    for frame_points in tqdm(path.transpose(1,0,2)):
        optimized_points = optimize_frame(
            objective_func,
            frame_points, 
            expected_distances, 
            expected_angles, 
            coefs = (dist_coef, angle_coef),
            side_weights = side_weights,
            angle_weights = angle_weights,
        )
        optimized_points_list.append(optimized_points)
        j += 1
    return_list = np.array(optimized_points_list).transpose((1,0,2))
    
    return return_list 

# this is another masking function, it finds points where there has been a very large jump and invalidates them.
def jump_filter(all_tracks, threshold=5):
    all_masks = []
    for track in all_tracks:
        lens = np.array([track[i + 1] - track[i] for i in range(len(track) - 1)])
        lens = np.linalg.norm(lens, axis=1)
        jumps = [length > threshold for length in lens]
        all_masks.append(jumps)
    return np.array(all_masks)

# this is a similar method to above but a bit more detailed. Instead of just making large jumps invalid it will wait until the point returns to where it started. This can be very helpful becomes at times the track will jump to a different spot and stick. The above method will not catch those but this would. However this method can have issues with valid jumping if tracking lost then found after the pit has moved across the image.
def jump_filter_sophistic(all_tracks, threshold=5):
    all_masks = []
    for track in all_tracks:
        lens = np.array([track[i + 1] - track[i] for i in range(len(track) - 1)])
        lens = np.linalg.norm(lens, axis=1)
        jumps = []
        i = 0
        while i < len(lens):
            dist = lens[i]
            if dist > threshold and i > 1:
                anchor = i
                top = None
                from_anchor = np.array([track[anchor] - track[anchor + j] for j in range(1, len(track) - anchor)])
                from_anchor = np.linalg.norm(from_anchor, axis=1)
                for j, delta in enumerate(from_anchor):
                    if delta < threshold:
                        top = j + anchor + 1
                        i = top
                        break
                if top is not None:
                    jumps.append([anchor, top])
            i += 1
        print(jumps.shape)
        all_masks.append(jumps)
    return np.array(all_masks)
    
    
# simple making method to find points outside a min-max range from a fixed poitn
def from_fixed_filter(track, dmax=675, dmin=525, center=None):
    if center is None:
        center = np.array([800, 350])
    else:
        center = np.array(center)
    mask = track[:, :, :2] - center
    mask = np.linalg.norm(mask, axis=1)
    mask = (mask > dmin) & (mask < dmax)
    return mask

# more simple masking method to mask based on tracking confidence
def conf_filter(track, conf_threshold=0.9):
    conf_mask = np.array([pt[2] > conf_threshold for pt in track])
    return conf_mask
