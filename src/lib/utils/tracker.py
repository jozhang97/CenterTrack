# borrowed from https://raw.githubusercontent.com/xingyizhou/CenterTrack/master/src/lib/utils/tracker.py
# How to evaluate? save_results then os.subprocess call
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy

class Tracker(object):
  # This gives us the tracks at one timestep, combined in a dict indexed by time
  def __init__(self, opt):
    self.opt = opt  # keys: hungarian, public_det, new_thresh, max_age
    self.reset()

  def init_track(self, results):
    '''
    :param results first set of detections
    initialize tracks for all detections with high scores
    '''
    for item in results:
      if item['score'] > self.opt.new_thresh:
        self.id_count += 1
        # active and age are never used in the paper
        item['active'] = 1
        item['age'] = 1
        item['tracking_id'] = self.id_count
        if not ('ct' in item):
          bbox = item['bbox']
          item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.tracks.append(item)

  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step(self, results, public_det=None):
    '''
    :param results list of detection (dict)
          [{score, class, ct, tracking, bbox}]
    :param public_det using known detections 
    :return ret list of detections with ID. Does not include track history
          [{score, class, ct, tracking, bbox, tracking_id, age, active}]

    Sketch
    - Find the size, class, center for each object for all existing objects and all new detections
    - Compute pairwise distances between existing and new objects
    - Perform matching (hungarian or greedy)
    - Update tracks with new matched detections, create new tracks for unmatched detections, 
      remove old tracks with no new detections
    '''
    N = len(results)
    M = len(self.tracks)

    # for all tracks, get sizes and classes and cts
    track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * (track['bbox'][3] - track['bbox'][1])) \
      for track in self.tracks], np.float32) # M
    track_cat = np.array([track['class'] for track in self.tracks], np.int32) # M
    tracks = np.array([pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    # for all results, get sizes and classes, apply offset
    dets = np.array([det['ct'] + det['tracking'] for det in results], np.float32) # N x 2
    item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * (item['bbox'][3] - item['bbox'][1])) \
      for item in results], np.float32) # N
    
    item_cat = np.array([item['class'] for item in results], np.int32) # N

    # compute pairwise image distances, apply large dist if invalid (not same class, not close enough) 
    dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M
    invalid = ((dist > track_size.reshape(1, M)) + (dist > item_size.reshape(N, 1)) + \
              (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
    dist = dist + invalid * 1e18
    
    # match bipartite graph
    if self.opt.hungarian:
      item_score = np.array([item['score'] for item in results], np.float32) # N
      dist[dist > 1e18] = 1e18
      matched_indices = linear_assignment(dist)
    else:
      matched_indices = greedy_assignment(copy.deepcopy(dist))

    # tracks that disappeared or just started
    unmatched_dets   = [d for d in range(dets.shape[0])   if not (d in matched_indices[:, 0])]
    unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]
    
    # postprocess hungarian 
    if self.opt.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
          unmatched_tracks.append(m[1])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    # update existing tracks that found matching detections
    ret = []
    for m in matches:
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)

    # handle new objects
    if self.opt.public_det and len(unmatched_dets) > 0:
      # Public detection: only create tracks from provided detections
      pub_dets = np.array([d['ct'] for d in public_det], np.float32)
      dist3 = ((dets.reshape(-1, 1, 2) - pub_dets.reshape(1, -1, 2)) ** 2).sum(axis=2)
      matched_dets = [d for d in range(dets.shape[0]) if not (d in unmatched_dets)]
      dist3[matched_dets] = 1e18
      for j in range(len(pub_dets)):
        i = dist3[:, j].argmin()    # pick the closest object
        if dist3[i, j] < item_size[i]:
          dist3[i, :] = 1e18
          track = results[i]
          if track['score'] > self.opt.new_thresh:
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
            ret.append(track)
    else:
      # Private detection: create tracks for all un-matched detections
      for i in unmatched_dets:
        track = results[i]
        if track['score'] > self.opt.new_thresh:
          self.id_count += 1
          track['tracking_id'] = self.id_count
          track['age'] = 1
          track['active'] =  1
          ret.append(track)
    
    # handle stale objects (objects that have not been seen for a while)
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.opt.max_age:
        track['age'] += 1
        track['active'] = 0
        bbox = track['bbox']
        ct = track['ct']
        v = [0, 0]
        track['bbox'] = [
          bbox[0] + v[0], bbox[1] + v[1],
          bbox[2] + v[0], bbox[3] + v[1]]
        track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
        ret.append(track)
    self.tracks = ret
    return ret

def greedy_assignment(dist):
  '''
  :param dist N, M or M, N distances
  :return K, 2 matched indices
  '''
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    # Pick the closest j
    j = dist[i].argmin()

    # If the closest is closer than 1e16, match and remove j from consideration
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)
