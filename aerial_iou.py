import time
from ioutracker import util, iou_tracker


# sigma_l (float): low detection threshold.
# sigma_h (float): high detection threshold.
# sigma_iou (float): IOU threshold.
# t_min (float): minimum track length in frames.
sigma_l = 0
sigma_h = 1
sigma_iou = 0  # TODO: tune
t_min = 1         # TODO: tune


detections = util.load_mot('det_iou.csv')
start = time.time()
tracks = iou_tracker.track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min)
end = time.time()

num_frames = len(detections)
print("finished at " + str(int(num_frames / (end - start))) + " fps!")

util.save_to_csv('out_iou.csv', tracks)