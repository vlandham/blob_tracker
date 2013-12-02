#!/usr/bin/env python

import numpy as np
import cv2
import video
import time
from scipy import *
from scipy.cluster import vq
from math import *
import sys, os, random, hashlib

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


video_src = "/Users/vlandham/code/python/opencv/test.avi"
vc = cv2.VideoCapture(0)
# vc = video.create_capture(video_src)


class Target:
  def __init__(self):
    fps = 15
    is_color = True

    self.capture = cv2.VideoCapture(0)
    #cv.SetCaptureProperty( self.capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640 );
    #cv.SetCaptureProperty( self.capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480 );
    # cv.SetCaptureProperty( self.capture, cv.CV_CAP_PROP_FRAME_WIDTH, 320 );
    # cv.SetCaptureProperty( self.capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 240 );
    self.writer = None

    cv2.namedWindow("Target")

  def run(self):
    # An tracked is list: [ name, color, last_time_seen, last_known_coords ]
    last_frame_tracking = []
    this_frame_tracking = []
    frame_count = 0
    frame_t0 = time.time()
    background = None


    while True:
      ret, frame = self.capture.read()
      frame_count += 1
      frame_t0 = time.time()

      if frame is not None:
        this_frame_tracking = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        view = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        #TODO: experiment with other blurs
        frame_blur = cv2.medianBlur(frame, 3)
        if background is None:
          background = frame_blur.copy()
        # a = 0.020 leaves artifacts lingering way too long.
        # a = 0.320 works well at 320x240, 15fps.  (1/a is roughly num frames.)
        # background = frame_blur.copy()
        cv2.accumulateWeighted(cv2.cvtColor(frame_blur, cv2.CV_32F), cv2.cvtColor(background, cv2.CV_32F), 0.320)
        foreground = cv2.absdiff(frame_blur, background)
        cv2.imshow('preview', foreground)


        # blobs = blob_detctor.detect(fgmask)
        # view = fgmask
        blobs = None

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

      # pdb.set_trace()
      # for x, y in np.float32(blobs).reshape(-1, 2):
        # print x
        # blobs.append([(x, y)])
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.destroyAllWindows()

if __name__=="__main__":
	t = Target()
#	import cProfile
#	cProfile.run( 't.run()' )
	t.run()

