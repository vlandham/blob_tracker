#!/usr/bin/env python


import numpy as np
import cv2
import video
import time
import sys, os, random, hashlib
from math import *

import pdb

blob_params = cv2.SimpleBlobDetector_Params()
blob_params.filterByInertia = False
blob_params.filterByConvexity = False
blob_params.filterByColor = False
blob_params.filterByCircularity = False
blob_params.filterByArea = True
blob_params.minArea = 100.0
blob_params.maxArea = 500.0

blob_detctor = cv2.SimpleBlobDetector(blob_params)


cv2.namedWindow("preview")
cv2.namedWindow("contours")
video_src = "/Users/vlandham/code/python/opencv/data/test.avi"
vc = cv2.VideoCapture(video_src)
# if( !vc.isOpened() ):
#   throw "Error when reading image file";
framecount = vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
print framecount;
#video.create_capture(fn)
# vc = video.create_capture(video_src)

fgbg = cv2.BackgroundSubtractorMOG()
rval, frame = vc.read()

# An tracked is list: [ name, color, last_time_seen, last_known_coords ]
last_frame_tracking = []
this_frame_tracking = []
frame_count = 0
frame_t0 = time.time()


while True:
  ret, frame = vc.read()
  frame_count += 1
  frame_t0 = time.time()
  this_frame_tracking = []

  if frame is not None:
    crop_frame = frame[100:800, 200:800]
    frame = crop_frame
    h, w = frame.shape[:2]
    fgmask = fgbg.apply(frame)
    blobs = blob_detctor.detect(fgmask)
    # view = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    view = frame.copy()
    # view = fgmask

    contours0, hierarchy = cv2.findContours(fgmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1]
    contours = filtered_contours
    # pdb.set_trace()
    vis = np.zeros((h, w, 3), np.uint8)
    levels = 1 
    # cv2.drawContours( vis, contours, (-1, 3)[levels <= 0], (128,255,255), 3, cv2.CV_AA, hierarchy, abs(levels) )
    cv2.drawContours( vis, contours, -1, (128,255,255))
    cv2.imshow('contours', vis)

    boundingRects = [cv2.boundingRect(cnt) for cnt in contours]


    if blobs is not None:
      for blob in blobs:
        blob_found = False
        blob_distances = {}
        point = blob.pt
        # get distance from all tracked blobs

        for tracked in last_frame_tracking:
          tracked_point = tracked[3]
          delta_x = tracked_point[0] - point[0]
          delta_y = tracked_point[1] - point[1]
          distance = sqrt( pow(delta_x,2) + pow( delta_y,2) )
          blob_distances[distance] = tracked

        distance_list = blob_distances.keys()
        distance_list.sort()
        for distance in distance_list:
          # Yes. see if we can claim the nearest one:
          nearest_possible_tracked = blob_distances[distance]
          # Don't consider entities that are already claimed:
          if nearest_possible_tracked in this_frame_tracking:
            continue
          blob_found = True
          nearest_possible_tracked[2] = frame_t0  # Update last_time_seen
          nearest_possible_tracked[3] = point  # Update the new location
          this_frame_tracking.append(nearest_possible_tracked)
          break
        if blob_found == False:
          color = ( random.randint(0,255), random.randint(0,255), random.randint(0,255) )
          name = hashlib.md5( str(frame_t0) + str(color) ).hexdigest()[:6]
          last_time_seen = frame_t0
          new_tracking = [ name, color, last_time_seen, point ]
          this_frame_tracking.append( new_tracking )
      # Now "delete" any not-found tracking which have expired:
      tracking_ttl = 1.0  # 1 sec.
      for tracking in last_frame_tracking:
        last_time_seen = tracking[2]
        if frame_t0 - last_time_seen > tracking_ttl:
          pass
        else:
          this_frame_tracking.append( tracking )
      last_frame_tracking = this_frame_tracking 
      for tracking in this_frame_tracking:
        point = tracking[3]
        color = tracking[1]
        cv2.circle(view, (int(point[0]), int(point[1])), 5, color, -1)
      cv2.imshow('preview', view)
      print len(this_frame_tracking)

      # pdb.set_trace()
      # for x, y in np.float32(blobs).reshape(-1, 2):
        # print x
        # blobs.append([(x, y)])
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
