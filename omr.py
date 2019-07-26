import argparse
import cv2
# import math
import numpy as np
from matplotlib import pyplot as plt
import operator
from PIL import Image


# Load an color image in grayscale
img = cv2.imread('img/answered-sheet-photo-result.png',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('sheet/sheet.png',img)
    cv2.destroyAllWindows()









GRAY = [255,0,0]

img1 = cv2.imread('sheet/sheet.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=GRAY)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()




img1 = cv2.imread('sheet/sheet.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)


# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#.....................................
#....................................
#.......................................

def draw_circles(img, circles):
    # img = cv2.imread(img,0)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv2.putText(cimg,str(i[0])+str(',')+str(i[1]), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    return cimg

def detect_circles(image):
    gray = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gray_blur = cv2.medianBlur(gray, 13)  # Remove noise before laplacian
    gray_lap = cv2.Laplacian(gray_blur, cv2.CV_8UC1, ksize=5)
    dilate_lap = cv2.dilate(gray_lap, (3, 3))  # Fill in gaps from blurring. This helps to detect circles with broken edges.
    # Furture remove noise introduced by laplacian. This removes false pos in space between the two groups of circles.
    lap_blur = cv2.bilateralFilter(dilate_lap, 5, 9, 9)
    # Fix the resolution to 16. This helps it find more circles. Also, set distance between circles to 55 by measuring dist in image.
    # Minimum radius and max radius are also set by examining the image.
    circles = cv2.HoughCircles(lap_blur, cv2.cv.CV_HOUGH_GRADIENT, 16, 55, param2=450, minRadius=20, maxRadius=40)
    cimg = draw_circles(gray, circles)
    print("{} circles detected.".format(circles[0].shape[0]))
    # There are some false positives left in the regions containing the numbers.
    # They can be filtered out based on their y-coordinates if your images are aligned to a canonical axis.
    # I'll leave that to you.
    return cimg

















# CORNER_FEATS = (
#     0.322965313273202,
#     0.19188334690998524,
#     1.1514327482234812,
#     0.998754685666376,
# )

# TRANSF_SIZE = 512


# def normalize(im):
#     return cv2.normalize(im, np.zeros(im.shape), 0, 255, norm_type=cv2.NORM_MINMAX)

# def get_approx_contour(contour, tol=.01):
#     """Get rid of 'useless' points in the contour"""
#     epsilon = tol * cv2.arcLength(contour, True)
#     return cv2.approxPolyDP(contour, epsilon, True)

# def get_contours(image_gray):
#     im2, contours, hierarchy = cv2.findContours(
#         image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     return map(get_approx_contour, contours)

# def get_corners(contours):
#     return sorted(
#         contours,
#         key=lambda c: features_distance(CORNER_FEATS, get_features(c)))[:4]

# def get_bounding_rect(contour):
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     return np.int0(box)

# def get_convex_hull(contour):
#     return cv2.convexHull(contour)

# def get_contour_area_by_hull_area(contour):
#     return (cv2.contourArea(contour) /
#             cv2.contourArea(get_convex_hull(contour)))

# def get_contour_area_by_bounding_box_area(contour):
#     return (cv2.contourArea(contour) /
#             cv2.contourArea(get_bounding_rect(contour)))

# def get_contour_perim_by_hull_perim(contour):
#     return (cv2.arcLength(contour, True) /
#             cv2.arcLength(get_convex_hull(contour), True))

# def get_contour_perim_by_bounding_box_perim(contour):
#     return (cv2.arcLength(contour, True) /
#             cv2.arcLength(get_bounding_rect(contour), True))

# def get_features(contour):
#     try:
#         return (
#             get_contour_area_by_hull_area(contour),
#             get_contour_area_by_bounding_box_area(contour),
#             get_contour_perim_by_hull_perim(contour),
#             get_contour_perim_by_bounding_box_perim(contour),
#         )
#     except ZeroDivisionError:
#         return 4*[np.inf]
# #..................................................................................ken
# #......................................................................................
# #......................................................................................
# def features_distance(f1, f2):
#     return np.linalg.norm(np.array(f1) - np.array(f2))

# # Default mutable arguments should be harmless here
# def draw_point(point, img, radius=5, color=(0, 0, 255)):
#     cv2.circle(img, tuple(point), radius, color, -1)

# def get_centroid(contour):
#     m = cv2.moments(contour)
#     x = int(m["m10"] / m["m00"])
#     y = int(m["m01"] / m["m00"])
#     return (x, y)

# def order_points(points):
#     """Order points counter-clockwise-ly."""
#     origin = np.mean(points, axis=0)

#     def positive_angle(p):
#         x, y = p - origin
#         ang = np.arctan2(y, x)
#         return 2 * np.pi + ang if ang < 0 else ang

#     return sorted(points, key=positive_angle)

# def get_outmost_points(contours):
#     all_points = np.concatenate(contours)
#     return get_bounding_rect(all_points)

# def perspective_transform(img, points):
#     """Transform img so that points are the new corners"""

#     source = np.array(
#         points,
#         dtype="float32")

#     dest = np.array([
#         [TRANSF_SIZE, TRANSF_SIZE],
#         [0, TRANSF_SIZE],
#         [0, 0],
#         [TRANSF_SIZE, 0]],
#         dtype="float32")

#     img_dest = img.copy()
#     transf = cv2.getPerspectiveTransform(source, dest)
#     warped = cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
#     return warped

# def sheet_coord_to_transf_coord(x, y):
#     return list(map(lambda n: int(np.round(n)), (
#         TRANSF_SIZE * x/744.055,
#         TRANSF_SIZE * (1 - y/1052.362)
#     )))

# def get_question_patch(transf, q_number):
#     # Top left
#     tl = sheet_coord_to_transf_coord(
#         200,
#         850 - 80 * (q_number - 1)
#     )

#     # Bottom right
#     br = sheet_coord_to_transf_coord(
#         650,
#         800 - 80 * (q_number - 1)
#     )
#     return transf[tl[1]:br[1], tl[0]:br[0]]

# def get_question_patches(transf):
#     for i in range(1, 11):
#         yield get_question_patch(transf, i)

# def get_alternative_patches(question_patch):
#     for i in range(5):
#         x0, _ = sheet_coord_to_transf_coord(100 * i, 0)
#         x1, _ = sheet_coord_to_transf_coord(50 + 100 * i, 0)
#         yield question_patch[:, x0:x1]

# def draw_marked_alternative(question_patch, index):
#     cx, cy = sheet_coord_to_transf_coord(
#         50 * (2 * index + .5),
#         50/2)
#     draw_point((cx, TRANSF_SIZE - cy), question_patch, radius=5, color=(255, 0, 0))

# def get_marked_alternative(alternative_patches):
#     means = list(map(np.mean, alternative_patches))
#     sorted_means = sorted(means)

#     # Simple heuristic
#     if sorted_means[0]/sorted_means[1] > .7:
#         return None

#     return np.argmin(means)

# def get_letter(alt_index):
#     return ["", "", "", "", ""][alt_index] if alt_index is not None else "N/A"

# def get_answers(source_file):
    # """Run the full pipeline:

    #     - Load image
    #     - Convert to grayscale
    #     - Filter out high frequencies with a Gaussian kernel
    #     - Apply threshold
    #     - Find contours
    #     - Find corners among all contours
    #     - Find 'outmost' points of all corners
    #     - Apply perpsective transform to get a bird's eye view
    #     - Scan each line for the marked answer
    # """

#     im_orig = cv2.imread(source_file)

#     blurred = cv2.GaussianBlur(im_orig, (11, 11), 10)

#     im = normalize(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

#     ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

#     contours = get_contours(im)
#     corners = get_corners(contours)

#     cv2.drawContours(im_orig, corners, -1, (0, 255, 0), 3)

#     outmost = order_points(get_outmost_points(corners))

#     transf = perspective_transform(im_orig, outmost)

#     answers = []
#     for i, q_patch in enumerate(get_question_patches(transf)):
#         alt_index = get_marked_alternative(get_alternative_patches(q_patch))

#         if alt_index is not None:
#             draw_marked_alternative(q_patch, alt_index)

#         answers.append(get_letter(alt_index))

#     #cv2.imshow('orig', im_orig)
#     #cv2.imshow('blurred', blurred)
#     #cv2.imshow('bw', im)

#     return answers, transf

# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--input",
#         help="Input image filename",
#         required=True,
#         type=str)

#     parser.add_argument(
#         "--output",
#         help="Output image filename",
#         type=str)

#     parser.add_argument(
#         "--show",
#         action="store_true",
#         help="Displays annotated image")

#     args = parser.parse_args()

#     answers, im = get_answers(args.input)

#     for i, answer in enumerate(answers):
#         print("Q{}: {}".format(i + 1, answer))

#     if args.output:
#         cv2.imwrite(args.output, im)
#         print("Wrote image to {}".format(args.output))

#     if args.show:
#         cv2.imshow('trans', im)

#         print("Close image window and hit ^C to quit.")
#         while True:
#             cv2.waitKey()

if __name__ == '__main__':
    main()
