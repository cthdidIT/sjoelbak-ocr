# Standard imports
import cv2
import numpy as np
import sys


def find_lines(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    minLineLength = 250
    maxLineGap = 30
    lines = cv2.HoughLinesP(edges, 2, np.pi / 240, 80,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)

    if lines is not None:
        print "found %s lines" % len(lines)

        return lines
    else:
        raise BaseException("found no lines")


def process_image(path):

    circles = find_circles(path)
    lines = find_lines(path)
    img = cv2.imread(path, 0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for x, y, radius in circles[0]:
        cv2.circle(cimg, (x, y), radius, (0, 0, 255), 2)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(cimg, (x1, y1),
                 (x2, y2), (0, 255, 0), 2)

    new_path = path.split("/")[-1][:-4]

    cv2.imwrite("processed/" + new_path + "-keypoints.jpg", cimg)


def find_circles(path):
    img = cv2.imread(path, 0)
    img = cv2.medianBlur(img, 5)

    radius = 30

    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 2, radius * 2, minRadius=radius, maxRadius=radius + 20)

    # cv2.HoughCircles(image, method, dp, minDist, [circles, param1, param2, minRadius, maxRadius])
    # [x, y, radius]
    if circles is not None:
        # circles = np.uint16(np.around(circles))
        # circles = circles[0, :]
        print "found %s circles" % len(circles[0])

        return circles
    else:
        raise BaseException("found no circles")


for arg in sys.argv[1:]:
    try:
        process_image(arg)
    except BaseException as e:
        print "%s for %s" % (e, arg)
