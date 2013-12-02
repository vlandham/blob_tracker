#!/usr/bin/env python


import numpy as np
import cv2
import video
import time
import sys, os, random, hashlib
from math import *

import pdb

#
# BBoxes must be in the format:
# ( (topleft_x), (topleft_y) ), ( (bottomright_x), (bottomright_y) ) )
top = 0
bottom = 1
left = 0
right = 1

def merge_collided_bboxes( bbox_list ):
  # For every bbox...
  for this_bbox in bbox_list:

    # Collision detect every other bbox:
    for other_bbox in bbox_list:
      if this_bbox is other_bbox: continue  # Skip self

      # Assume a collision to start out with:
      has_collision = True

      # These coords are in screen coords, so > means 
      # "lower than" and "further right than".  And < 
      # means "higher than" and "further left than".

      # We also inflate the box size by 10% to deal with
      # fuzziness in the data.  (Without this, there are many times a bbox
      # is short of overlap by just one or two pixels.)
      if (this_bbox[bottom][0]*1.1 < other_bbox[top][0]*0.9): has_collision = False
      if (this_bbox[top][0]*.9 > other_bbox[bottom][0]*1.1): has_collision = False

      if (this_bbox[right][1]*1.1 < other_bbox[left][1]*0.9): has_collision = False
      if (this_bbox[left][1]*0.9 > other_bbox[right][1]*1.1): has_collision = False

      if has_collision:
        # merge these two bboxes into one, then start over:
        top_left_x = min( this_bbox[left][0], other_bbox[left][0] )
        top_left_y = min( this_bbox[left][1], other_bbox[left][1] )
        bottom_right_x = max( this_bbox[right][0], other_bbox[right][0] )
        bottom_right_y = max( this_bbox[right][1], other_bbox[right][1] )

        new_bbox = ( (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) )

        bbox_list.remove( this_bbox )
        bbox_list.remove( other_bbox )
        bbox_list.append( new_bbox )

        # Start over with the new list:
        return merge_collided_bboxes( bbox_list )

  # When there are no collions between boxes, return that list:
  return bbox_list


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
cv2.moveWindow("contours", 600, 0)
video_src = "/Users/vlandham/code/python/opencv/data/test.avi"
vc = cv2.VideoCapture(video_src)
# vc = cv2.VideoCapture(0)
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
t0 = time.time()
frame_t0 = time.time()


while True:
  ret, frame = vc.read()
  frame_count += 1
  frame_t0 = time.time()
  this_frame_tracking = []

  if frame is not None:
    # TODO: This is just a quick hack to crop the test video
    # REMOVE
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
    contour_vis = np.zeros((h, w, 3), np.uint8)
    levels = 1 
    # cv2.drawContours( vis, contours, (-1, 3)[levels <= 0], (128,255,255), 3, cv2.CV_AA, hierarchy, abs(levels) )
    cv2.drawContours( contour_vis, contours, -1, (128,255,255))

    bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
    bounding_box_list = []

    for bounding_rect in bounding_rects:
      point1 = ( bounding_rect[0], bounding_rect[1] )
      point2 = ( bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3] )
      bounding_box_list.append( ( point1, point2 ) )

    box_areas = []
    for box in bounding_box_list:
      box_width = box[right][0] - box[left][0]
      box_height = box[bottom][0] - box[top][0]
      box_areas.append( box_width * box_height )

    average_box_area = 0.0
    if len(box_areas): average_box_area = float( sum(box_areas) ) / len(box_areas)
    trimmed_box_list = []
    for box in bounding_box_list:
      box_width = box[right][0] - box[left][0]
      box_height = box[bottom][0] - box[top][0]
      # Only keep the box if it's not a tiny noise box:
      if (box_width * box_height) > average_box_area*0.1: trimmed_box_list.append( box )

    # bounding_box_list = merge_collided_bboxes( trimmed_box_list )
    bounding_box_list = trimmed_box_list

    for box in bounding_box_list:
      cv2.rectangle(contour_vis, box[0], box[1], (255,0,0), 1)

    centers = []
    for box in bounding_box_list: 
      box_width = box[right][0] - box[left][0]
      box_height = box[bottom][0] - box[top][0]
      box_center_x = box[top][0] + (box_width / 2.0)
      box_center_y = box[top][1] - (box_height / 2.0)
      center = dict(pt = (box_center_x, box_center_y))
      centers.append(center)


    cv2.imshow('contours', contour_vis)

    if centers is not None:
      for blob in centers:
        blob_found = False
        blob_distances = {}
        point = blob['pt']
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
        # create new tracking
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
        cv2.circle(view, (int(point[0]), int(point[1])), 15, color, 2)
        cv2.circle(view, (int(point[0]), int(point[1])), 10, color, 1)
        cv2.circle(view, (int(point[0]), int(point[1])), 5, color, 3)
      cv2.imshow('preview', view)
      print len(this_frame_tracking)
      t1 = time.time()
      time_delta = t1 - t0
      processed_fps = float( frame_count ) / time_delta
      print "Got %d frames. %.1f s. %f fps." % ( frame_count, time_delta, processed_fps )

      # pdb.set_trace()
      # for x, y in np.float32(blobs).reshape(-1, 2):
        # print x
        # blobs.append([(x, y)])
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
