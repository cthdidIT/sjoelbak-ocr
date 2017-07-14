import base64
import uuid
import cv2
import numpy as np
import sys
import random
import json


def lambda_handler(event, context):
    img_base64 = event.get('base64Image')
    if img_base64 is None:
        return respond(True, dict(message="No base64Image key"))

    img = base64.decodestring(img_base64)

    name = uuid.uuid4()
    path = '/tmp/{}.png'.format(name)

    image_result = open(path, 'wb')
    image_result.write(img)
    image_result.close()

    process_image(path)

    image_processed_path = '/tmp/{}-processed.png'.format(name)
    image_processed = open(image_processed_path, 'rb')
    image_processed_data = image_processed.read()
    image_processed.close()
    image_64_encode = base64.urlsafe_b64encode(image_processed_data)

    return respond(None, image_64_encode)


def respond(err, res):
    return {
        'statusCode': '400' if err else '200',
        'body': json.dumps(res),
        'headers': {
            'Content-Type': 'application/json',
        },
    }


def find_lines(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    min_line_length = 250
    max_line_gap = 30
    lines = cv2.HoughLinesP(edges, 2, np.pi / 240, 80,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        print("found %s lines" % len(lines))

        return lines
    else:
        raise BaseException("found no lines")


def bagify_lines(lines):
    y_threshold = 180
    x_threshold = 200
    avg_k = calculate_avg_k(lines)
    table = split_lines_on_avg_k(avg_k, lines)

    h_lines = table['horizontal']
    v_lines = table['vertical']

    h_table = group_lines(h_lines, lambda (x1, y1, x2, y2): ((y1 + y2) / 2) / y_threshold)
    v_table = group_lines(v_lines, lambda (x1, y1, x2, y2): ((x1 + x2) / 2) / x_threshold)

    vertical = average_lines(v_table)
    horizontal = average_lines(h_table)

    sorted(horizontal, key=lambda (k, m): m, reverse=True)

    return dict(vertical=vertical, horizontal=horizontal)


def average_lines(v_table):
    avg_lines = list()
    for k, v in v_table.iteritems():
        total = 0
        m_total = 0
        k_total = 0

        for x1, y1, x2, y2 in v:
            k = (y2 - y1) / (float(x2) - float(x1) + 0.0000001)
            k_total += k
            m_total += y1 - (k * x1)
            total += 1

        avg_k = k_total / total
        avg_m = m_total / total

        line = (avg_k, avg_m)
        avg_lines.append(line)
    return avg_lines


def group_lines(lines, normalize_function):
    table = dict()
    for l in lines:
        norm_x = normalize_function(l)
        bag = table.get(norm_x, set())
        bag.add(l)
        table[norm_x] = bag
    return table


def split_lines_on_avg_k(avg_k, lines):
    table = dict()
    for l in lines:
        x1, y1, x2, y2 = l[0]

        k = (y2 - y1) / (float(x2) - float(x1) + 0.0000001)

        if k < avg_k:
            key = 'horizontal'
        else:
            key = 'vertical'

        bag = table.get(key, set())
        bag.add((x1, y1, x2, y2))
        table[key] = bag

    return table


def calculate_avg_k(lines):
    max_k = -1
    min_k = 99999
    for l in lines:
        x1, y1, x2, y2 = l[0]
        k = (y2 - y1) / (float(x2) - float(x1) + 0.0000001)
        max_k = max(max_k, k)
        min_k = min(min_k, k)
    avg_k = float(max_k + min_k) / 3
    return avg_k


def get_abc(l):
    (k, m) = l
    x1 = 1
    x2 = 2
    y1 = k * x1 + m
    y2 = k * x2 + m

    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1

    return A, B, C


def calculate_intersection_point_between_lines(h, v):
    (A1, B1, C1) = get_abc(h)
    (A2, B2, C2) = get_abc(v)

    det = A1 * B2 - A2 * B1
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det

    return x, y


def get_intersections(hs, vs):
    points = list()
    for h in hs:
        for v in vs:
            points.append(calculate_intersection_point_between_lines(h, v))
    return points


def draw_line(img, l):
    (k, m) = l
    height, width = img.shape[:2]

    c1 = random.random() * 255
    c2 = random.random() * 255
    c3 = random.random() * 255
    cv2.line(img, (0, int(m)), (width, int(k * width + m)), (c1, c2, c3), 2)


def draw_box(img, b):
    c1 = random.random() * 255
    c2 = random.random() * 255
    c3 = random.random() * 255

    (p1, p2, p3, p4) = b

    color = (c1, c2, c3)
    cv2.line(img, p1, p2, color, 2)
    cv2.line(img, p2, p3, color, 2)
    cv2.line(img, p3, p4, color, 2)
    cv2.line(img, p4, p1, color, 2)


def box_intersection(b, c):
    (cx, cy, r) = c
    (p1, p2, p3, p4) = b

    if p1[0] < cx < p2[0] and p3[0] > cx > p4[0] and cy > p3[1] < cy < p1[1] > p4[1]:
        return True

    return False


def calculate_score(scores):
    ones = scores[1]
    twos = scores[2]
    threes = scores[3]
    fours = scores[4]

    min_amount = min(ones, twos, threes, fours)
    total = min_amount * 20
    total += (ones - min_amount) * 1
    total += (twos - min_amount) * 2
    total += (threes - min_amount) * 3
    total += (fours - min_amount) * 4

    return total


def process_image(path):
    circles = find_circles(path)
    lines = find_lines(path)

    img = cv2.imread(path, 0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    draw_circles(cimg, circles)

    table = bagify_lines(lines)
    horizontal = table['horizontal']
    vertical = table['vertical']
    draw_lines(cimg, horizontal)
    draw_lines(cimg, vertical)

    intersections = get_intersections(horizontal, vertical)
    draw_intersection_points(cimg, intersections)

    iss = map(lambda (x, y): (int(x), int(y)), intersections)

    height, width = cimg.shape[:2]
    middle = width / 2

    lefties, righties = split_lines_by_x(iss, middle)
    boxes = get_boxes(lefties, righties)

    draw_boxes(boxes, cimg)

    font = cv2.FONT_HERSHEY_SIMPLEX

    points = determine_points(boxes, cimg, circles, font)

    score = calculate_score(points)
    ppg = float(score) / 30

    score_text_y = int(height - height / 10)
    cv2.putText(cimg, str(score), (int(width / 2), score_text_y), font, 2, (255, 255, 255), 4)
    cv2.putText(cimg, str("PPG: %.2f" % ppg), (int(width / 2), score_text_y + 50), font, 1, (255, 255, 255), 3)

    new_path = path[:-4] + "-processed.png"
    cv2.imwrite(new_path, cimg)

    return new_path


def determine_points(boxes, cimg, circles, font):
    box_points = [1, 4, 3, 2]
    scores = {1: 0, 2: 0, 3: 0, 4: 0}
    for b in boxes:
        points = box_points.pop()
        for c in circles[0]:
            if box_intersection(b, c):
                scores[points] = scores[points] + 1
                (x, y, r) = c
                hr = round(r / 2)
                cv2.putText(cimg, str(points), (int(x - hr), int(y + hr)), font, 2, (255, 255, 255), 2)
    return scores


def draw_boxes(boxes, cimg):
    for b in boxes:
        draw_box(cimg, b)


def get_boxes(lefties, righties):
    pair1 = [lefties.pop(), righties.pop()]
    boxes = list()
    while len(lefties) > 0 and len(righties) > 0:
        pair2 = [righties.pop(), lefties.pop()]
        p1 = pair1[0]
        p2 = pair1[1]
        p3 = pair2[0]
        p4 = pair2[1]
        b = (p1, p2, p3, p4)
        boxes.append(b)
        next_shit = list(pair2)
        next_shit.reverse()
        pair1 = next_shit
    return boxes


def split_lines_by_x(iss, middle):
    lefties = list()
    righties = list()
    for p in iss:
        if p[0] < middle:
            lefties.append(p)
        else:
            righties.append(p)
    return lefties, righties


def draw_intersection_points(cimg, intersections):
    for x, y in intersections:
        cv2.circle(cimg, (int(x), int(y)), 2, (255, 255, 0), 2)


def draw_lines(img, lines):
    for l in lines:
        draw_line(img, l)


def draw_circles(img, circles):
    for x, y, radius in circles[0]:
        cv2.circle(img, (x, y), radius, (0, 0, 255), 2)


def find_circles(path):
    img = cv2.imread(path, 0)
    img = cv2.medianBlur(img, 5)

    radius = 30
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 2, radius * 2, minRadius=radius, maxRadius=radius + 20)

    if circles is not None:
        print("found %s circles" % len(circles[0]))

        return circles
    else:
        raise BaseException("found no circles")


for arg in sys.argv[1:]:
    try:
        process_image(arg)
    except BaseException as e:
        print("%s for %s" % (e, arg))
