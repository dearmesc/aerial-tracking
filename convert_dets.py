import re
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def dirsig2sort(det_file, outfile):
    # Input format: <frame path> <x> <y>
    #     (e.g.: hsi_truth-t0000-c0000.img 1395.32 898.903)
    # Output format: <frame>,-1,<x>,<y>,<w>,<h>,<score>,-1,-1,-1
    #     (assume score constant for simplicity)
    id = re.search(r'(\d+)_track.txt', det_file).group(1)
    with open(det_file, 'r') as f, open(outfile, 'a') as out, open('gt_track.csv', 'a') as gt:
        for line in f:
            line = line.strip().split()
            frame = int(re.search(r'c(\d{4})', line[0]).group(1))
            x = float(line[1])
            y = float(line[2])
            out.write('{},-1,{},{},{},{},0.98,-1,-1,-1\n'.format(
                frame, x, y, 10, 10
            ))
            gt.write('{},{},{},{},{},{}\n'.format(
                frame, id, x, y, 10, 10
            ))


def dirsig2iou(det_file, outfile):
    # Input format: <frame path> <x> <y>
    #     (e.g.: hsi_truth-t0000-c0000.img 1395.32 898.903)
    # Output format: 'frame', -1, 'x', 'y', 'w', 'h', 'score'
    #     (assume score constant for simplicity)
    with open(det_file, 'r') as f, open(outfile, 'a') as out:
        for line in f:
            line = line.strip().split()
            frame = int(re.search(r'c(\d{4})', line[0]).group(1))
            x = float(line[1])
            y = float(line[2])
            out.write('{},-1,{},{},{},{},0.98\n'.format(
                frame, x, y, 10, 10
            ))


def avg_overlap(c1, c2, w=10, h=10):
    dist = np.abs(c1 - c2)
    cover_x = w - dist[:, 0]
    cover_y = h - dist[:, 1]
    cover_x[cover_x < 0] = 0
    cover_y[cover_y < 0] = 0
    return np.mean(cover_x * cover_y / (w * h))


def ruler(sdt, gt_tracks):
    sdt = np.loadtxt(sdt, delimiter=',')
    gt_tracks = np.loadtxt(gt_tracks, delimiter=',')
    # thresh = 0.5

    # lbl_sdt = np.unique(sdt[:, 1])
    # lbl_gt = np.unique(gt_tracks[:, 1])
    n = int(gt_tracks[:, 0].max()) + 1  # number of frames
    fp = np.zeros(n - 1, dtype=np.uint16)
    fn = np.zeros_like(fp)
    idsw = np.zeros_like(fp)
    n_matches = np.zeros_like(fp)
    overlap = np.zeros_like(fp, dtype=np.float64)

    # First frame
    gt = gt_tracks[gt_tracks[:, 0] == 1, :]
    sdt_now = sdt[sdt[:, 0] == 1, :]
    cost_matrix = cdist(gt[:, 2:4], sdt_now[:, 2:4])  # x,y coordinates
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind]
    n_matches[0] = len(cost)
    if len(cost) > 0:
        overlap[0] = avg_overlap(gt[row_ind, 2:4], sdt_now[col_ind, 2:4], w=gt[0, 4], h=gt[0, 5])
    # fp[0] = np.sum(cost > thresh)
    fn[0] = gt.shape[0] - len(cost)  # gt - fp - tp = fn
    prev_lbl = dict()
    for i, j in zip(gt[row_ind, 1], sdt_now[col_ind, 1]):
        prev_lbl[i] = j
    # Subsequent frames
    for frame in range(2, n):
        gt = gt_tracks[gt_tracks[:, 0] == frame, :]
        sdt_now = sdt[sdt[:, 0] == frame, :]
        cost_matrix = cdist(gt[:, 2:4], sdt_now[:, 2:4])  # x,y coordinates
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Compute the cost of this assignment
        cost = cost_matrix[row_ind, col_ind]

        # Precision components
        n_matches[frame - 1] = len(cost)
        if len(cost) > 0:
            overlap[frame - 1] = avg_overlap(gt[row_ind, 2:4], sdt_now[col_ind, 2:4], w=gt[0, 4], h=gt[0, 5])

        # False positives and false negatives
        # fp[frame - 1] = np.sum(cost > thresh)  # Using ground truth as detections, there should be no FPs
        fn[frame - 1] = gt.shape[0] - len(cost)  # gt - fp - tp = fn

        # ID switches
        lbl = dict()
        for i, j in zip(gt[row_ind, 1], sdt_now[col_ind, 1]):
            lbl[i] = j
            if i in prev_lbl:
                # print('\tPrev: {:d} -> {:d}\tCurr: {:d} -> {:d}'.format(int(i), int(prev_lbl[i]), int(i), int(j)))
                idsw[frame - 1] += (j != prev_lbl[i])
        prev_lbl = dict(lbl)

        print(row_ind, col_ind, len(gt))

    # Compute MOTA
    MOTA = 1 - np.sum(fn + fp + idsw) / gt_tracks.shape[0]

    # Compute MOTP
    MOTP = np.mean(overlap)
    print('MOTA = {:0.4f}'.format(MOTA))
    print('MOTP = {:0.4f}'.format(MOTP))
    return MOTA, MOTP


# home = os.path.expanduser('~')
# det_file = '5007_track.txt'
# obj_id = re.match(r'\d+_track\.txt', det_file)

# files = os.listdir('/home/scd1442/data/dirsig/Ground_Truth_Files')
# for det_file in files:
#     det_file = os.path.join('/home/scd1442/data/dirsig/Ground_Truth_Files', det_file)
#     dirsig2sort(det_file, 'det_sort.csv')
#     dirsig2iou(det_file, 'det_iou.csv')

MOTA, MOTP = ruler('out_sort.csv', 'gt_track.csv')
