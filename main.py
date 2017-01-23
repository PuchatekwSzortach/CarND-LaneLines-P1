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
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def get_image_contours(image):

    return cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=5)


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
    Given a binary image return image with only simple contours
    :param binary_image:
    :return: image
    """

    _, contours, _ = cv2.findContours(binary_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    simple_contours = []

    for contour in contours:

        simple_contour = cv2.approxPolyDP(contour, epsilon=5, closed=False)

        if len(simple_contour) <= 10:

            simple_contours.append(simple_contour)

    simple_image = np.zeros_like(binary_image)
    cv2.drawContours(simple_image, np.array(simple_contours), contourIdx=-1, color=255)

    return simple_image


def get_lane_line(lines, slope_condition):

    try:

        xs = []
        ys = []

        for line in lines:

            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)

            if slope_condition(slope):

                xs.extend([x1, x2])
                ys.extend([y1, y2])

        lane_equation = np.polyfit(xs, ys, deg=1)

        min_x = min(xs)
        max_x = max(xs)

        # Compute line that fits our equation and spans our xs
        lane = np.array([
            min_x, (min_x * lane_equation[0]) + lane_equation[1],
            max_x, (max_x * lane_equation[0]) + lane_equation[1]
        ]).astype(int)

        return np.array([lane])

    except TypeError:

        # If no lines were detected, np.polyfit will get empty data
        # For simplicity just return a 0 line then
        return np.array([[0, 0, 0, 0]])


def get_lanes_lines(lines):
    """
    Given an array of lines, compute likely lanes lines and return them
    :param lines:
    :return: lines
    """

    left_line_slope_condition = lambda x: x < 0
    left_line = get_lane_line(lines, left_line_slope_condition)

    right_line_slope_condition = lambda x: x > 0
    right_line = get_lane_line(lines, right_line_slope_condition)

    return [left_line, right_line]


def process_image(image):

    grayscale_image = grayscale(image)
    blurred_image = gaussian_blur(grayscale_image, 3)
    contours_image = get_image_contours(blurred_image)

    mask_vertices = np.array([[
        (400, 300), (50, image.shape[0]), (image.shape[1] - 50, image.shape[0]), (image.shape[1] - 400, 300)
    ]])

    masked_image = region_of_interest(contours_image, mask_vertices)

    # Lines can't be complex geometrical shapes, so we can remove any contour that isn't sufficiently simple
    simple_contours_image = get_simple_contours_image(masked_image)

    lines = cv2.HoughLinesP(
        simple_contours_image, rho=4, theta=math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=1)

    lane_lanes = get_lanes_lines(lines)

    lanes_image = np.zeros_like(image)
    draw_lines(lanes_image, lane_lanes, thickness=8, color=[0, 0, 255])

    lanes_overlay_image = weighted_img(lanes_image, image, alpha=1)
    return lanes_overlay_image


def process_image_experimental(image):

    grayscale_image = grayscale(image)
    blurred_image = gaussian_blur(grayscale_image, 3)
    contours_image = get_image_contours(blurred_image)

    mask_vertices = np.array([[
        (400, 300), (50, image.shape[0]), (image.shape[1] - 50, image.shape[0]), (image.shape[1] - 400, 300)
    ]])

    masked_image = region_of_interest(contours_image, mask_vertices)

    # Lines can't be complex geometrical shapes, so we can remove any contour that isn't sufficiently simple
    simple_contours_image = get_simple_contours_image(masked_image)

    lines = cv2.HoughLinesP(
        simple_contours_image, rho=4, theta=math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=1)

    lane_lanes = get_lanes_lines(lines)

    lanes_image = np.zeros_like(image)
    draw_lines(lanes_image, lane_lanes, thickness=4, color=[255, 0, 0])

    lanes_overlay_image = weighted_img(lanes_image, image, alpha=1)

    return lanes_overlay_image


def detect_images_lines(directory, logger):

    paths = glob.glob(os.path.join(directory, "*.jpg"))

    for path in paths:

        image = cv2.imread(path)
        lanes_image = process_image(image)
        # lanes_image_experimental = process_image_experimental(image)

        images = [image, lanes_image]
        # images = [image, lanes_image, lanes_image_experimental]
        logger.info(vlogging.VisualRecord("Detections", images))


def detect_movies_lines():

    paths = ["solidWhiteRight.mp4", "solidYellowLeft.mp4", "challenge.mp4"]

    for path in paths:

        output_name = path.split(".")[0] + "_output.mp4"
        clip = moviepy.editor.VideoFileClip(path)
        white_clip = clip.fl_image(process_image)
        white_clip.write_videofile(output_name, audio=False)


def main():

    # logger = get_logger("/tmp/lanes_detection.html")
    # images_directory = "./test_images"
    # detect_images_lines(images_directory, logger)

    detect_movies_lines()


if __name__ == "__main__":

    main()