"""
Lanes detection project
"""

import math
import numpy as np
import glob
import os
import logging

import cv2
import vlogging
import moviepy.editor
import matplotlib.image as mpimg


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to log.html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("faces")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def grayscale(img):
    """
    Applies the Grayscale transform
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def get_grayscale(image):

    changed = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    return changed[:,:,2]


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def get_image_contours(image):

    return cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=5)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    for line in lines:

        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]),
        minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., lambda_parameter=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lambda_parameter)


def get_simple_contours_image(binary_image):
    """
    Given a binary image return image that contains only simple contours
    :param binary_image:
    :return: image
    """

    _, contours, _ = cv2.findContours(binary_image.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    simple_contours = []

    for contour in contours:

        simple_contour = cv2.approxPolyDP(contour, epsilon=10, closed=True)

        if len(simple_contour) <= 8:

            simple_contours.append(contour)

    simple_image = np.zeros_like(binary_image)
    cv2.drawContours(simple_image, np.array(simple_contours), contourIdx=-1, color=255)

    return simple_image


def get_lane_line(lines, image_shape):

    try:

        xs = []
        ys = []

        for line in lines:

            x1, y1, x2, y2 = line[0]

            xs.extend([x1, x2])
            ys.extend([y1, y2])

        lane_equation = np.polyfit(xs, ys, deg=1)

        min_y = min(ys)
        min_x = round((min_y - lane_equation[1]) / lane_equation[0])

        max_y = image_shape[0]
        max_x = round((max_y - lane_equation[1]) / lane_equation[0])

        lane = np.array([
            min_x, min_y, max_x, max_y
        ]).astype(int)

        return np.array([lane])

    except TypeError:

        # If no lines were detected, np.polyfit will get empty data
        # For simplicity just return a 0 line then
        return np.array([[0, 0, 0, 0]])


def get_lane_line_dev(lines, image_shape):

    try:

        xs = []
        ys = []

        for line in lines:

            x1, y1, x2, y2 = line[0]

            xs.extend([x1, x2])
            ys.extend([y1, y2])

        lane_equation = np.polyfit(xs, ys, deg=1)

        min_y = min(ys)
        min_x = round((min_y - lane_equation[1]) / lane_equation[0])

        max_y = image_shape[0]
        max_x = round((max_y - lane_equation[1]) / lane_equation[0])

        lane = np.array([
            min_x, min_y, max_x, max_y
        ]).astype(int)

        print(lane)

        return lane

    except TypeError:

        # If no lines were detected, np.polyfit will get empty data
        # For simplicity just return a 0 line then
        lane = np.array([0, 0, 0, 0])

        print(lane)
        return lane


def is_line_left_lane_candidate(line, image_shape):

    x1, y1, x2, y2 = line[0]

    slope_degrees = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    is_left_lane_slope = -45 < slope_degrees < -30

    is_line_left_of_vehicle = x1 < image_shape[1] // 2 and x2 < image_shape[1] // 2

    return is_left_lane_slope and is_line_left_of_vehicle


def is_line_right_lane_candidate(line, image_shape):

    x1, y1, x2, y2 = line[0]

    slope_degrees = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    is_right_lane_slope = 30 < slope_degrees < 45

    is_line_right_of_vehicle = x1 > image_shape[1] // 2 and x2 > image_shape[1] // 2

    return is_right_lane_slope and is_line_right_of_vehicle


def get_left_road_lane_candidates(lines, image_shape):

    return [line for line in lines if is_line_left_lane_candidate(line, image_shape)]


def get_right_road_lane_candidates(lines, image_shape):

    return [line for line in lines if is_line_right_lane_candidate(line, image_shape)]


def get_road_lanes_lines(lines):
    """
    Given an array of lines, compute likely lanes lines and return them
    :param lines:
    :return: lines
    """

    left_lines = [line for line in lines if is_line_left_lane_candidate(line)]
    right_lines = [line for line in lines if is_line_right_lane_candidate(line)]


    return left_lines + right_lines


def process_image(image):

    grayscale_image = grayscale(image)
    # blurred_image = gaussian_blur(grayscale_image, 3)
    contours_image = get_image_contours(grayscale_image)

    mask_vertices = np.array([[
        (400, 300), (50, image.shape[0]), (image.shape[1] - 50, image.shape[0]), (image.shape[1] - 400, 300)
    ]])

    masked_image = region_of_interest(contours_image, mask_vertices)

    # Lines can't be complex geometrical shapes, so we can remove any contour that isn't sufficiently simple
    simple_contours_image = get_simple_contours_image(masked_image)

    lines = cv2.HoughLinesP(
        simple_contours_image, rho=4, theta=math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    lane_lanes = get_road_lanes_lines(lines)

    lanes_image = np.zeros_like(image)
    draw_lines(lanes_image, lane_lanes, thickness=12, color=[255, 0, 0])

    lanes_overlay_image = weighted_img(lanes_image, image, alpha=1)
    return lanes_overlay_image


def get_contours(image):

    grayscale_image = grayscale(image)
    blurred_image = gaussian_blur(grayscale_image, 3)
    contours_image = get_image_contours(blurred_image)

    return contours_image


def get_masked_contours(image):

    mask_vertices = np.array([[
        (400, 300), (50, image.shape[0]), (image.shape[1] - 50, image.shape[0]), (image.shape[1] - 400, 300)
    ]])

    masked_image = region_of_interest(image, mask_vertices)

    return masked_image


def detect_images_lines(directory, logger):

    paths = glob.glob(os.path.join(directory, "*.jpg"))

    for path in paths:

        image = mpimg.imread(path)
        lanes_image = process_image(image)
        lanes_image = cv2.cvtColor(lanes_image, cv2.COLOR_RGB2BGR)

        logger.info(vlogging.VisualRecord("Detections", lanes_image))


def detect_movies_lines():

    paths = ["solidWhiteRight.mp4", "solidYellowLeft.mp4", "challenge.mp4"]
    # paths = ["challenge.mp4"]
    # paths = ["solidWhiteRight.mp4"]

    for path in paths:

        clip = moviepy.editor.VideoFileClip(path)

        simplified_contours_clip = clip.fl_image(get_simplified_contours_movie)
        lines_clip = simplified_contours_clip.fl_image(get_lines_movie)
        road_lanes_candidates_clip = simplified_contours_clip.fl_image(get_road_lane_lines_candidates_movie)
        road_lanes_clip = clip.fl_image(get_road_lanes_movie)

        final_clip = moviepy.editor.clips_array([[clip, lines_clip], [road_lanes_candidates_clip, road_lanes_clip]])

        output_name = path.split(".")[0] + "_output.mp4"
        final_clip.write_videofile(output_name, audio=False, fps=12)


def get_single_channel_movie(image):

    graycale_image = get_grayscale(image)
    return np.dstack([graycale_image, graycale_image, graycale_image])


def get_simplified_contours_movie(image):

    grayscale_image = get_grayscale(image)

    blurred_image = gaussian_blur(grayscale_image, 7)
    contours_image = get_image_contours(blurred_image)

    mask_vertices = np.array([[
        (400, 300), (50, image.shape[0]), (image.shape[1] - 50, image.shape[0]), (image.shape[1] - 400, 300)
    ]])

    masked_image = region_of_interest(contours_image, mask_vertices)

    simple_contours_image = get_simple_contours_image(masked_image)
    return np.dstack([simple_contours_image, simple_contours_image, simple_contours_image])


def get_lines_movie(simple_image):

    lines = cv2.HoughLinesP(
        simple_image[:, :, 0], rho=4, theta=math.pi / 180, threshold=80, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    line_img = np.zeros((simple_image.shape[0], simple_image.shape[1], 3), dtype=np.uint8)

    if lines is not None:

        draw_lines(line_img, lines, color=[255, 255, 255])

    return line_img


def get_road_lane_lines_candidates_movie(simple_image):

    lines = cv2.HoughLinesP(
        simple_image[:, :, 0], rho=4, theta=math.pi / 180, threshold=80, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    line_img = np.zeros((simple_image.shape[0], simple_image.shape[1], 3), dtype=np.uint8)

    lane_lanes = get_left_road_lane_candidates(lines, line_img.shape) + \
                 get_right_road_lane_candidates(lines, line_img.shape)

    draw_lines(line_img, lane_lanes, color=[255, 255, 255])

    return line_img


def get_road_lanes_movie(image):

    grayscale_image = get_grayscale(image)

    blurred_image = gaussian_blur(grayscale_image, 7)
    contours_image = get_image_contours(blurred_image)

    mask_vertices = np.array([[
        (400, 300), (50, image.shape[0]), (image.shape[1] - 50, image.shape[0]), (image.shape[1] - 400, 300)
    ]])

    masked_image = region_of_interest(contours_image, mask_vertices)

    simple_contours_image = get_simple_contours_image(masked_image)

    lines = cv2.HoughLinesP(
        simple_contours_image, rho=4, theta=math.pi / 180, threshold=80, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    left_lines_candidates = get_left_road_lane_candidates(lines, image.shape)
    right_lines_candidates = get_right_road_lane_candidates(lines, image.shape)

    left_lane_line = get_lane_line(left_lines_candidates, image.shape)
    right_lane_line = get_lane_line(right_lines_candidates, image.shape)

    # lane_lines = left_lines_candidates + right_lines_candidates
    lane_lines = [left_lane_line, right_lane_line]

    lanes_image = np.zeros_like(image)
    draw_lines(lanes_image, lane_lines, thickness=12, color=[255, 0, 0])

    lanes_overlay_image = weighted_img(lanes_image, image, alpha=1)
    return lanes_overlay_image


def main():
    #
    # logger = get_logger("/tmp/lanes_detection.html")
    # images_directory = "./test_images"
    # detect_images_lines(images_directory, logger)

    detect_movies_lines()


if __name__ == "__main__":

    main()