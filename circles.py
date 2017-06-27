# Standard imports
import cv2
import numpy as np
import sys
import random


def find_lines(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    minLineLength = 250
    maxLineGap = 30
    lines = cv2.HoughLinesP(edges, 2, np.pi / 240, 80,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)


    if lines is not None:
        print "found %s lines" % len(lines)

        return lines
    else:
        raise BaseException("found no lines")

def bagify_lines(lines):
    y_threshold = 180
    x_threshold = 200
    table = dict()

    max_k = -1
    min_k = 99999
    for l in lines:
        x1, y1, x2, y2 = l[0]
        k = abs((y2-y1)/(float(x2)-float(x1)+0.0000001))
        max_k = max(max_k, k)
        min_k = min(min_k, k)

    avg_k = float(max_k + min_k)/2

    for l in lines:
        x1, y1, x2, y2 = l[0]

        k = abs((y2-y1)/(float(x2)-float(x1)+0.0000001))

        key = 'unknown'
        if(k<avg_k):
            key = 'horizontal'
        else:
            key = 'vertical'

        bag = table.get(key, set())
        bag.add((x1,y1,x2,y2))
        table[key] = bag

    h_lines = table['horizontal']
    h_table = dict()
    for l in h_lines:
        x1, y1, x2, y2 = l
        norm_y = ((y1+y2)/2)/y_threshold
        bag = h_table.get(norm_y, set())
        bag.add(l)
        h_table[norm_y] = bag

    v_lines = table['vertical']
    v_table = dict()
    for l in v_lines:
        x1, y1, x2, y2 = l
        norm_x = ((x1+x2)/2)/x_threshold
        bag = v_table.get(norm_x, set())
        bag.add(l)
        v_table[norm_x] = bag


    ret_table = dict()
    it = 0

    for k,v in v_table.iteritems():
        total = 0
        nbrs = 0
        max_y = max(map(lambda (x1, y1, x2, y2): max(y1,y2), v))
        min_y = min(map(lambda (x1, y1, x2, y2): min(y1,y2), v))

        for (x1, y1, x2, y2) in v:
            total += x1 + x2
            nbrs += 2

        avg = (total)/nbrs

        normalized = (avg, min_y, avg, max_y)
        ret_table[it] = set([normalized])
        it += 1


    for k,v in h_table.iteritems():
        total = 0
        nbrs = 0
        max_x = max(map(lambda (x1, y1, x2, y2): max(x1,x2), v))
        min_x = min(map(lambda (x1, y1, x2, y2): min(x1,x2), v))

        for (x1, y1, x2, y2) in v:
            total += y1 + y2
            nbrs += 2

        avg = (total)/nbrs

        normalized = (min_x, avg, max_x, avg)
        ret_table[it] = set([normalized])
        it += 1

    return ret_table






def process_image(path):

    circles = find_circles(path)
    lines = find_lines(path)

    img = cv2.imread(path, 0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for x, y, radius in circles[0]:
        cv2.circle(cimg, (x, y), radius, (0, 0, 255), 2)

    """
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(cimg, (x1, y1),
                 (x2, y2), (0, 255, 0), 2)
    """

    table = bagify_lines(lines)

    for key, value in table.iteritems():
        c1 = random.random() * 255
        c2 = random.random() * 255
        c3 = random.random() * 255
        for l in value:
            x1, y1, x2, y2 = l
            cv2.line(cimg, (x1, y1),
                     (x2, y2), (0, 255, 0), 2)

    new_path = path.split("/")[-1][:-4]

    cv2.imwrite("processed/" + new_path+"-keypoints.jpg", cimg)
    cv2.waitKey(0)


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
