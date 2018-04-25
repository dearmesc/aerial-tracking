import time
from sort import sort
import numpy as np

# create instance of SORT
mot_tracker = sort.Sort()

total_frames = 0
total_time = 0

# get detections
# dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
seq_dets = np.loadtxt('det_sort.csv', delimiter=',')  # load detections
with open('out_sort.csv', 'w') as out_file:
    print("Processing detections...")
    for frame in range(int(seq_dets[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                  file=out_file)
# update SORT
# track_bbs_ids = mot_tracker.update(detections)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
# ...

# TODO: track associations

print('Cycle time = {:0.3f} sec'.format(total_time))
print('Frames: {}'.format(total_frames))
