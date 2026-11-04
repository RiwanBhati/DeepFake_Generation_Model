import sys
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import torch
from tqdm import tqdm
import face_alignment

# initialize the face detection and landmark model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

# marker if the bbox is not sufficient 
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    average_range_minus = []
    average_range_plus = []
    for frame in tqdm(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(frame_rgb)
        if preds is None or len(preds) == 0:
            continue
        face_land_mark = preds[0].astype(np.int32)
        range_minus = (face_land_mark[30] - face_land_mark[29])[1]
        range_plus = (face_land_mark[29] - face_land_mark[28])[1]
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)

    minus_avg = int(sum(average_range_minus) / max(len(average_range_minus), 1))
    plus_avg = int(sum(average_range_plus) / max(len(average_range_plus), 1))
    text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{minus_avg}~{plus_avg} ] , the current value: {upperbondrange}"
    return text_range

def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    coords_list = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for frame in tqdm(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(frame_rgb)
        if preds is None or len(preds) == 0:
            coords_list += [coord_placeholder]
            continue
        face_land_mark = preds[0].astype(np.int32)
        
        detected_faces = fa.face_detector.detect_from_image(frame_rgb.copy())
        f = detected_faces[0][:4] if len(detected_faces) > 0 else None
        
        half_face_coord = face_land_mark[29].copy()
        range_minus = (face_land_mark[30] - face_land_mark[29])[1]
        range_plus = (face_land_mark[29] - face_land_mark[28])[1]
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)
        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        min_upper_bond = 0
        upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
        
        f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
        x1, y1, x2, y2 = f_landmark
        
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            if f is not None:
                coords_list += [(int(f[0]), int(f[1]), int(f[2]), int(f[3]))]
            else:
                coords_list += [coord_placeholder]
        else:
            coords_list += [f_landmark]

    minus_avg = int(sum(average_range_minus) / max(len(average_range_minus), 1))
    plus_avg = int(sum(average_range_plus) / max(len(average_range_plus), 1))
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{minus_avg}~{plus_avg} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list, frames

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png", "./results/lyria/00001.png", "./results/lyria/00002.png", "./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
