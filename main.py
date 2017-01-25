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


def get_simple_mask_vertices(image_shape):

    return np.array([[
        (400, 300), (50, image_shape[0]), (image_shape[1] - 50, image_shape[0]), (image_shape[1] - 400, 300)
    ]])


def get_challenge_mask_vertices(image_shape):

    return np.array([[
        (500, 450), (100, image_shape[0] - 50), (image_shape[1] - 100, image_shape[0] - 50), (image_shape[1] - 500, 450)
    ]])


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to log.html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("lanes")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def get_xyz_grayscale(image):

    changed = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    return changed[:,:,2]


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def get_image_contours(image):

    return cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 thresholdType=cv2.THRESH_BINARY_INV, blockSize=7, C=5)


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


def get_line_coordinates(lines):
    """
    Given a list of lines, returns a tuple (xs, ys), where xs are x coordinates and ys are y coordinates
    :param lines: list of lines, each line has (x1, y1, x2, y2) format
    :return: tuple (xs, ys)
    """

    xs = []
    ys = []

    for line in lines:

        x1, y1, x2, y2 = line

        xs.extend([x1, x2])
        ys.extend([y1, y2])

    return xs, ys


def get_lane_line(lines, image_shape):

    try:

        xs, ys = get_line_coordinates(lines)
        lane_equation = np.polyfit(xs, ys, deg=1)

        min_y = 320
        min_x = (min_y - lane_equation[1]) / lane_equation[0]

        max_y = image_shape[0]
        max_x = (max_y - lane_equation[1]) / lane_equation[0]

        lane = [int(coordinate) for coordinate in [min_x, min_y, max_x, max_y]]
        return lane

    except TypeError:

        # If no lines were detected, np.polyfit will get empty data
        # For simplicity just return a 0 line then
        return [0, 0, 0, 0]


def get_lane_line_challenge(lines, image_shape):

    try:

        xs, ys = get_line_coordinates(lines)
        lane_equation = np.polyfit(xs, ys, deg=1)

        min_y = 450
        min_x = (min_y - lane_equation[1]) / lane_equation[0]

        max_y = image_shape[0]
        max_x = (max_y - lane_equation[1]) / lane_equation[0]

        lane = [int(coordinate) for coordinate in [min_x, min_y, max_x, max_y]]
        return lane

    except TypeError:

        # If no lines were detected, np.polyfit will get empty data
        # For simplicity just return a 0 line then
        return [0, 0, 0, 0]


def get_line_length(line):

    x1, y1, x2, y2 = line
    return np.sqrt((y2 - y1)**2 + (x2 - x1)**2)


def are_lines_collinear(first, second):
    """
    Given two lines check if they are approximately colinear
    :param first: line
    :param second: line
    :return: boolean
    """

    first_equation = np.polyfit([first[0], first[2]], [first[1], first[3]], deg=1)
    second_equation = np.polyfit([second[0], second[2]], [second[1], second[3]], deg=1)

    slope_difference = np.abs(first_equation[0] - second_equation[0])
    angular_difference = np.rad2deg(np.arctan(slope_difference))

    offset_distance = np.abs(first_equation[1] - second_equation[1])

    return angular_difference < 1 and offset_distance < 0.5


def merge_lines(first, second):

    xs = [first[0], first[2], second[0], second[2]]
    ys = [first[1], first[3], second[1], second[3]]

    min_index = np.argmin(xs)
    max_index = np.argmax(xs)

    return [xs[min_index], ys[min_index], xs[max_index], ys[max_index]]


def get_lines_in_descending_length_order(lines):

    line_lengths = [get_line_length(line) for line in lines]
    lines_lengths_tuple = zip(lines, line_lengths)

    # Sort lines by length in descending order
    sorted_lines_lengths_tuples = sorted(lines_lengths_tuple, key=lambda x: x[1], reverse=True)

    # Extract lines, now they are in descending length order
    return [line_length_tuple[0] for line_length_tuple in sorted_lines_lengths_tuples]


def get_lane_line_dev(lines, image_shape):

    if len(lines) == 0:

        # If no lines are available, just return an empty line for simplicity
        return [0, 0, 0, 0]

    sorted_lines = get_lines_in_descending_length_order(lines)

    lane_candidates = []
    lane_candidates_lengths = []

    for line in sorted_lines:

        is_collinear_line_found = False

        for index in range(len(lane_candidates)):

            if are_lines_collinear(lane_candidates[index], line):

                is_collinear_line_found = True
                lane_candidates[index] = merge_lines(lane_candidates[index], line)
                lane_candidates_lengths[index] += get_line_length(line)

        if not is_collinear_line_found:

            lane_candidates.append(line)
            lane_candidates_lengths.append(get_line_length(line))

    sorted_lane_candidates = get_lines_in_descending_length_order(lane_candidates)
    longest_lane = sorted_lane_candidates[0]

    # Get lane equation
    x1, y1, x2, y2 = longest_lane
    lane_equation = np.polyfit([x1, x2], [y1, y2], deg=1)

    # Extrapolate lane from bottom of screen to its furthest reach
    min_y = min(y1, y2)
    min_x = round((min_y - lane_equation[1]) / lane_equation[0])

    max_y = image_shape[0]
    max_x = round((max_y - lane_equation[1]) / lane_equation[0])

    lane = [int(coordinate) for coordinate in [min_x, min_y, max_x, max_y]]
    return lane


def is_line_left_lane_candidate(line, image_shape):

    x1, y1, x2, y2 = line

    slope_degrees = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    is_left_lane_slope = -45 < slope_degrees < -15

    max_x = 0.5 * image_shape[1]
    is_line_left_of_vehicle = x1 < max_x and max_x

    return is_left_lane_slope and is_line_left_of_vehicle


def is_line_right_lane_candidate(line, image_shape):

    x1, y1, x2, y2 = line

    slope_degrees = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    is_right_lane_slope = 15 < slope_degrees < 45

    min_x = 0.5 * image_shape[1]
    is_line_right_of_vehicle = x1 > min_x and x2 > min_x

    return is_right_lane_slope and is_line_right_of_vehicle


def get_left_road_lane_candidates(lines, image_shape):

    return [line for line in lines if is_line_left_lane_candidate(line, image_shape)]


def get_right_road_lane_candidates(lines, image_shape):

    return [line for line in lines if is_line_right_lane_candidate_challenge(line, image_shape)]


def is_line_left_lane_candidate_challenge(line, image_shape):
    x1, y1, x2, y2 = line

    slope_degrees = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    is_left_lane_slope = -45 < slope_degrees < -10

    max_x = 0.5 * image_shape[1]
    is_line_left_of_vehicle = x1 < max_x and max_x

    return is_left_lane_slope and is_line_left_of_vehicle


def is_line_right_lane_candidate_challenge(line, image_shape):
    x1, y1, x2, y2 = line

    slope_degrees = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    is_right_lane_slope = 10 < slope_degrees < 45

    min_x = 0.5 * image_shape[1]
    is_line_right_of_vehicle = x1 > min_x and x2 > min_x

    return is_right_lane_slope and is_line_right_of_vehicle


def get_left_road_lane_candidates_challenge(lines, image_shape):

    return [line for line in lines if is_line_left_lane_candidate(line, image_shape)]


def get_right_road_lane_candidates_challenge(lines, image_shape):

    return [line for line in lines if is_line_right_lane_candidate_challenge(line, image_shape)]


def get_filtered_lines(lines):
    """
    Filter out lines that are too dissimilar to the mean
    :param lines: list of lines
    :return: list of lines
    """

    data = np.array(lines)
    xs = data[:, 0] + data[:, 2]
    ys = data[:, 1] + data[:, 3]

    mean_equation = np.polyfit(xs, ys, deg=1)

    filtered_lines = []

    for line in lines:

        equation = np.polyfit([line[0], line[2]], [line[1], line[3]], deg=1)

        slope_difference = np.abs(mean_equation[0] - equation[0])
        angle_difference = np.rad2deg(np.arctan(slope_difference))

        if angle_difference < 5:

            filtered_lines.append(line)

    return filtered_lines


def pipeline(image):

    contours_image = get_xyz_space_contours(image)

    mask_vertices = get_simple_mask_vertices(image.shape)
    masked_image = region_of_interest(contours_image, mask_vertices)

    lines = cv2.HoughLinesP(
        masked_image, rho=4, theta=4 * math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    # Hough lines are wrapped in unnecessary list, extract them for easier processing
    lines = [line[0] for line in lines]

    left_lines_candidates = get_left_road_lane_candidates(lines, image.shape)
    right_lines_candidates = get_right_road_lane_candidates(lines, image.shape)

    filtered_left_lines_candidates = get_filtered_lines(left_lines_candidates)
    filtered_right_lines_candidates = get_filtered_lines(right_lines_candidates)

    left_lane_line = get_lane_line(filtered_left_lines_candidates, image.shape)
    right_lane_line = get_lane_line(filtered_right_lines_candidates, image.shape)

    lane_lines = [left_lane_line, right_lane_line]

    # Wrap up lines to format expected by draw_lines
    lane_lines = [[line] for line in lane_lines]

    lanes_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(lanes_image, lane_lines, thickness=10, color=[255, 0, 0])
    lanes_overlay_image = weighted_img(lanes_image, image)

    return lanes_overlay_image


def process_image(image):

    return pipeline(image)


def get_contours(image):

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = gaussian_blur(grayscale_image, 5)
    contours_image = get_image_contours(blurred_image)

    return contours_image


def get_xyz_space_contours(image):

    grayscale_image = get_xyz_grayscale(image)
    blurred_image = gaussian_blur(grayscale_image, 5)
    contours_image = get_image_contours(blurred_image)

    return contours_image


def detect_images_lines(directory, logger):

    paths = glob.glob(os.path.join(directory, "*.jpg"))

    for path in paths:

        image = mpimg.imread(path)
        lanes_image = process_image(image)
        lanes_image = cv2.cvtColor(lanes_image, cv2.COLOR_RGB2BGR)

        logger.info(vlogging.VisualRecord("Detections", lanes_image))


def get_image_stack(image_processor):

    def stacked_processor(image):

        processed_image = image_processor(image)
        return np.dstack([processed_image, processed_image, processed_image])

    return stacked_processor


def get_masked_image(image):

    contours_image = get_contours(image)
    mask_vertices = get_simple_mask_vertices(image.shape)

    return region_of_interest(contours_image, mask_vertices)


def get_masked_image_challenge(image):

    contours_image = get_xyz_space_contours(image)
    mask_vertices = get_challenge_mask_vertices(image.shape)

    return region_of_interest(contours_image, mask_vertices)


def get_lines_image(image):

    contours_image = get_contours(image)

    mask_vertices = get_simple_mask_vertices(image.shape)
    masked_image = region_of_interest(contours_image, mask_vertices)

    lines = cv2.HoughLinesP(
        masked_image, rho=4, theta=4 * math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if lines is not None:

        draw_lines(line_img, lines, color=[255, 255, 255], thickness=1)

    return line_img


def get_lines_image_challenge(image):

    contours_image = get_xyz_space_contours(image)

    mask_vertices = get_challenge_mask_vertices(image.shape)
    masked_image = region_of_interest(contours_image, mask_vertices)

    lines = cv2.HoughLinesP(
        masked_image, rho=4, theta=4 * math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if lines is not None:

        draw_lines(line_img, lines, color=[255, 255, 255], thickness=1)

    return line_img


def get_road_lane_lines_candidates_image(image):

    contours_image = get_xyz_space_contours(image)

    mask_vertices = get_simple_mask_vertices(image.shape)
    masked_image = region_of_interest(contours_image, mask_vertices)

    lines = cv2.HoughLinesP(
        masked_image, rho=4, theta=4 * math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    # Hough lines are wrapped in unnecessary list, extract them for easier processing
    lines = [line[0] for line in lines]

    left_lines_candidates = get_left_road_lane_candidates(lines, image.shape)
    right_lines_candidates = get_right_road_lane_candidates(lines, image.shape)

    lane_candidates = left_lines_candidates + right_lines_candidates
    lane_candidates = [[line] for line in lane_candidates]

    lanes_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(lanes_image, lane_candidates, color=[0, 0, 255])

    return lanes_image


def get_road_lane_lines_candidates_image_challenge(image):

    contours_image = get_xyz_space_contours(image)

    mask_vertices = get_challenge_mask_vertices(image.shape)
    masked_image = region_of_interest(contours_image, mask_vertices)

    lines = cv2.HoughLinesP(
        masked_image, rho=4, theta=4 * math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    # Hough lines are wrapped in unnecessary list, extract them for easier processing
    lines = [line[0] for line in lines]

    left_lines_candidates = get_left_road_lane_candidates_challenge(lines, image.shape)
    right_lines_candidates = get_right_road_lane_candidates_challenge(lines, image.shape)

    lane_candidates = left_lines_candidates + right_lines_candidates
    lane_candidates = [[line] for line in lane_candidates]

    lanes_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(lanes_image, lane_candidates, color=[0, 0, 255])

    return lanes_image


def process_image_challenge(image):

    contours_image = get_xyz_space_contours(image)

    mask_vertices = get_challenge_mask_vertices(image.shape)
    masked_image = region_of_interest(contours_image, mask_vertices)

    lines = cv2.HoughLinesP(
        masked_image, rho=4, theta=4 * math.pi / 180, threshold=100, lines=np.array([]),
        minLineLength=10, maxLineGap=10)

    # Hough lines are wrapped in unnecessary list, extract them for easier processing
    lines = [line[0] for line in lines]

    left_lines_candidates = get_left_road_lane_candidates_challenge(lines, image.shape)
    right_lines_candidates = get_right_road_lane_candidates_challenge(lines, image.shape)

    filtered_left_lines_candidates = get_filtered_lines(left_lines_candidates)
    filtered_right_lines_candidates = get_filtered_lines(right_lines_candidates)

    left_lane_line = get_lane_line_challenge(filtered_left_lines_candidates, image.shape)
    right_lane_line = get_lane_line_challenge(filtered_right_lines_candidates, image.shape)

    lane_lines = [left_lane_line, right_lane_line]

    # Wrap up lines to format expected by draw_lines
    lane_lines = [[line] for line in lane_lines]

    lanes_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(lanes_image, lane_lines, thickness=10, color=[255, 0, 0])
    lanes_overlay_image = weighted_img(lanes_image, image)

    return lanes_overlay_image


def detect_movies_lines_simple():

    paths = ["solidWhiteRight.mp4", "solidYellowLeft.mp4"]
    # paths = ["solidWhiteRight.mp4"]
    # paths = ["solidYellowLeft.mp4"]
    #
    for path in paths:

        clip = moviepy.editor.VideoFileClip(path)

        masked_image_stack = get_image_stack(get_masked_image)
        masked_image_clip = clip.fl_image(masked_image_stack)

        all_lines_clip = clip.fl_image(get_lines_image)

        road_lanes_candidates_clip = clip.fl_image(get_road_lane_lines_candidates_image)
        road_lanes_clip = clip.fl_image(process_image)

        final_clip = moviepy.editor.clips_array(
            [[masked_image_clip, all_lines_clip], [road_lanes_candidates_clip, road_lanes_clip]])

        output_name = path.split(".")[0] + "_output.mp4"
        final_clip.write_videofile(output_name, audio=False, fps=12)


def detect_movies_lines_challenge():

    path = "challenge.mp4"

    clip = moviepy.editor.VideoFileClip(path)

    masked_image_stacker = get_image_stack(get_masked_image_challenge)
    masked_image_clip = clip.fl_image(masked_image_stacker)

    all_lines_clip = clip.fl_image(get_lines_image_challenge)

    road_lanes_candidates_clip = clip.fl_image(get_road_lane_lines_candidates_image_challenge)
    road_lanes_clip = clip.fl_image(process_image_challenge)

    final_clip = moviepy.editor.clips_array(
        [[masked_image_clip, all_lines_clip], [road_lanes_candidates_clip, road_lanes_clip]])

    output_name = path.split(".")[0] + "_output.mp4"
    final_clip.write_videofile(output_name, audio=False, fps=12)


def main():

    logger = get_logger("/tmp/lanes_detection.html")
    images_directory = "./test_images"
    # detect_images_lines(images_directory, logger)
    #
    # detect_movies_lines_simple()
    detect_movies_lines_challenge()


if __name__ == "__main__":

    main()