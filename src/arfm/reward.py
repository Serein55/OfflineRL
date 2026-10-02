"""Table-8 reconstruction, NOT an official ARFM reward implementation.

Assumptions: reference=next within-demo heuristic keypoint; raw uint8 RGB;
position MSE scale .01; finite differences per control step; terminal-only success;
progress=number of keypoints already reached / total. See README.md for reconstruction assumptions.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity

def keypoints(joints, actions):
    velocity = np.diff(joints, axis=0, prepend=joints[:1])
    grip = actions[:, -1] > 0
    points, cooldown = [], 0
    for i in range(1, len(joints)):
        stable = i >= 2 and i < len(joints)-2 and np.all(grip[i-2:i+2] == grip[i])
        stopped = cooldown <= 0 and stable and np.all(np.abs(velocity[i]) <= .1)
        cooldown = 4 if stopped else cooldown - 1
        if grip[i] != grip[i-1] or stopped or i == len(joints)-1:
            points.append(i)
    if not points:
        points = [len(joints)-1]
    if len(points)>1 and points[-2] == points[-1]-1:
        points.pop(-2)
    return np.asarray(points)

def table8_reward(images, wrist, joints, actions, success=True):
    T=len(actions)
    goals=keypoints(joints,actions)
    refs=goals[np.minimum(np.searchsorted(goals,np.arange(T),side='left'),len(goals)-1)]
    parts=np.zeros((T,13),dtype=np.float64)
    orb=cv2.ORB_create(edgeThreshold=0,fastThreshold=40)
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    for camera, frames in enumerate((images,wrist)):
        feats=[orb.detectAndCompute(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),None) for f in frames]
        for t,g in enumerate(refs):
            a,b=frames[t],frames[g]
            mse=np.mean((a.astype(np.float64)-b)**2)
            ss=structural_similarity(a,b,channel_axis=-1,data_range=255)
            ka,da=feats[t]; kb,db=feats[g]
            match=0. if da is None or db is None else len(matcher.match(da,db))/max(1,min(len(ka),len(kb)))
            parts[t,camera*3:camera*3+3]=[np.exp(-.01*mse),np.exp(ss-1),np.exp(match-1)]
    parts[:,6]=np.exp(-.01*np.mean((joints-joints[refs])**2,axis=-1))
    parts[:,7]=np.searchsorted(goals,np.arange(T),side='right')/len(goals)
    for values,idx in ((joints,8),(actions,10)):
        velocity=np.diff(values,axis=0,prepend=values[:1])
        accel=np.diff(velocity,axis=0,prepend=velocity[:1])
        parts[:,idx]=-np.sum(velocity**2,axis=-1)
        parts[:,idx+1]=-np.sum(accel**2,axis=-1)
    parts[-1,12]=float(success)
    weights=np.full(13,.1/13); weights[10:12]=.01/13
    return parts@weights, parts, goals
