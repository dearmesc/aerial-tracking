import numpy as np
import cv2

GT = np.genfromtxt(
    "/Users/sophie/Downloads/gt_track.csv",
    delimiter=',',
    dtype=[int, int, float, float, float, float],
    names=['frame', 'id', 'row', 'col', 'width', 'height']
)

ST = np.genfromtxt(
    "/Users/sophie/Downloads/out_sort.csv",
    delimiter=',',
    dtype=[int, int, float, float, float, float],
    names=['frame', 'id', 'row', 'col', 'width', 'height']
)

# BGR (not RGB)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

radius = 25
thickness = 2


for frame in range(1, 157):

    img = cv2.imread('/Users/sophie/Downloads/data/image%03d.jpg' % frame)

    # ground truth
    for row in np.flatnonzero(GT['frame'] == frame):
        y, x = GT[row][['row', 'col']]
        x = int(round(x))
        y = int(round(y))
        img = cv2.circle(img, (x, y), radius, RED, thickness)

    # ground truth
    for row in np.flatnonzero(ST['frame'] == frame):
        y, x = ST[row][['row', 'col']]
        x = int(round(x))
        y = int(round(y))
        img = cv2.circle(img, (x, y), radius, GREEN, thickness)

    cv2.imwrite('/Users/sophie/Downloads/hitMiss/image%03d.jpg' % frame, img)
