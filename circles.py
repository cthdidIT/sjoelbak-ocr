# Standard imports
import cv2
import numpy as np
import sys
import random


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

    avg_k = float(max_k + min_k)/3

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

    vertical = list()
    for k,v in v_table.iteritems():
        nbrs = 0
        m_total = 0
        k_total = 0

        for (x1, y1, x2, y2) in v:
            k = ((y2-y1)/(float(x2)-float(x1)+0.0000001))
            k_total += k
            m_total += y1 - (k*x1)
            nbrs += 1

        avg_k = k_total/nbrs
        avg_m = m_total/nbrs

        line = (avg_k, avg_m)
        vertical.append(line)
        it += 1

    horizontal = list()
    for k,v in h_table.iteritems():
        nbrs = 0
        m_total = 0
        k_total = 0

        for (x1, y1, x2, y2) in v:
            left_x = min(x1,x2)
            right_x = max(x1,x2)

            k = ((y2-y1)/(float(x2)-float(x1)+0.0000001))
            k_total += k
            m_total += y1 - (k*x1)
            nbrs += 1

        avg_k = k_total/nbrs
        avg_m = (m_total/nbrs)

        line = (avg_k, avg_m)
        horizontal.append(line)
        it += 1

    sorted(horizontal, key=lambda (k, m) : m, reverse=True)

    return_table = dict(vertical=vertical, horizontal=horizontal)

    return return_table

def get_abc(l):
    (k, m) = l
    x1 = 1
    x2 = 2
    y1 = k*x1 + m
    y2 = k*x2 + m

    A = y2-y1
    B = x1-x2
    C = A*x1+B*y1

    return (A,B,C)

def intersection(h,v):
    (hk, hm) = h
    (vk, vm) = v

    (A1,B1,C1) = get_abc(h)
    (A2,B2,C2) = get_abc(v)

    det = A1*B2 - A2*B1
    x = (B2*C1 - B1*C2)/det
    y = (A1*C2 - A2*C1)/det

    return (x,y)

def get_intersections(hs, vs):
    points = list()
    for h in hs:
        for v in vs:
            points.append(intersection(h,v))
    return points

def draw_line(cimg, l):
    (k, m) = l
    height, width = cimg.shape[:2]
    print k, m

    c1 = random.random() * 255
    c2 = random.random() * 255
    c3 = random.random() * 255
    cv2.line(cimg, (0, int(m)), (width, int(k*width+m)), (c1, c2, c3), 2)



def draw_box(cimg, b):
    c1 = random.random() * 255
    c2 = random.random() * 255
    c3 = random.random() * 255

    (p1,p2,p3,p4) = b

    color = (c1, c2, c3)
    cv2.line(cimg, p1, p2, color, 2)
    cv2.line(cimg, p2, p3, color, 2)
    cv2.line(cimg, p3, p4, color, 2)
    cv2.line(cimg, p4, p1, color, 2)

def box_intersection(b, c):
    (cx, cy, r) = c
    (p1,p2,p3,p4) = b

    if(cx > p1[0] and cx < p2[0] and cx < p3[0] and cx > p4[0]):
        if(cy < p1[1] and cy < p2[1] and cy > p3[1] and cy > p4[1]):
            return True

    return False

def calculate_score(scores):
    ones = scores[1]
    twos = scores[2]
    threes = scores[3]
    fours = scores[4]

    min_amount = min(ones, twos, threes, fours)
    total = min_amount * 20
    total += (ones-min_amount) * 1
    total += (twos-min_amount) * 2
    total += (threes-min_amount) * 3
    total += (fours-min_amount) * 4

    return total

def process_image(path):

    circles = find_circles(path)
    lines = find_lines(path)

    img = cv2.imread(path, 0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for x, y, radius in circles[0]:
        cv2.circle(cimg, (x, y), radius, (0, 0, 255), 2)

    for line in lines:
        x1, y1, x2, y2 = line[0]

    table = bagify_lines(lines)
    horizontal = list(table['horizontal'])

    vs = table['vertical']

    for e in horizontal:
        draw_line(cimg, e)

    for e in vs:
        draw_line(cimg, e)

    intersections = get_intersections(horizontal, vs)

    for (x,y) in intersections:
        cv2.circle(cimg, (int(x), int(y)), 2, (255,255,0), 2)

    iss = map(lambda (x,y): (int(x), int(y)), intersections)

    height, width = cimg.shape[:2]

    middle = width / 2

    lefties = list()
    righties = list()
    for p in iss:
        if(p[0] < middle ):
            lefties.append(p)
        else:
            righties.append(p)

    pair1 = [lefties.pop(), righties.pop()]
    boxes = list()
    while(len(lefties) > 0 and len(righties) > 0):
        pair2 = [righties.pop(), lefties.pop()]
        p1 = pair1[0]
        p2 = pair1[1]
        p3 = pair2[0]
        p4 = pair2[1]
        b = (p1,p2,p3,p4)
        boxes.append(b)
        next_shit = list(pair2)
        next_shit.reverse()
        pair1 = next_shit

    for b in boxes:
        draw_box(cimg, b)


    font = cv2.FONT_HERSHEY_SIMPLEX

    box_points = [1,4,3,2]
    scores = {1:0, 2:0, 3:0, 4:0 }
    for b in boxes:
        points = box_points.pop()
        for c in circles[0]:
            if(box_intersection(b,c)):
                print "INTERSECT"
                scores[points] = scores[points] + 1
                (x,y,r) = c
                hr = round(r/2)
                cv2.putText(cimg, str(points), (int(x-hr),int(y+hr)), font, 2, (255, 255, 255), 2)

    score = calculate_score(scores)

    cv2.putText(cimg, str(score), (int(width/2),int((height-(height/10)))), font, 2, (255, 255, 255), 4)

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
