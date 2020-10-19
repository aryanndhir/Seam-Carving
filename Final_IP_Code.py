import cv2
import numpy as np

global max_int
max_int = 1e6   # 1x10^6


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0, 0)


def sobel_xaxis(img):
    return cv2.Sobel(
        img,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
        scale=1,
        delta=0,
        borderType=cv2.BORDER_DEFAULT,
    )


def sobel_yaxis(img):
    return cv2.Sobel(
        img,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
        scale=1,
        delta=0,
        borderType=cv2.BORDER_DEFAULT,
    )


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def energy(img):
    gaus_blurr = gaussian_blur(img)
    img_gray = grayscale(gaus_blurr)
    dx = sobel_xaxis(img_gray)
    dy = sobel_yaxis(img_gray)

    return cv2.add(np.absolute(dx), np.absolute(dy))


# In this function we are creating an energy map wrt vertical seams
def cumulative_energies_vertical(energy):

    (height, width) = energy.shape[:2]
    energy_map = np.zeros((height, width))

    for i in range(1, height):
        for j in range(width):
            left = (energy_map[i - 1, j - 1] if j - 1 >= 0 else max_int)
            middle = energy_map[i - 1, j]
            right = (energy_map[i - 1, j + 1] if j + 1 < width else max_int)

            energy_map[i, j] = energy[i, j] + min(left, middle, right)

    return energy_map


# In this function we are creating an energy map wrt horizontal seams
def cumulative_energies_horizontal(energy):
    (height, width) = energy.shape[:2]
    energy_map = np.zeros((height, width))

    for j in range(1, width):
        for i in range(height):
            top = (energy_map[i - 1, j - 1] if i - 1 >= 0 else max_int)
            middle = energy_map[i, j - 1]
            bottom = (energy_map[i + 1, j - 1] if i + 1 < height else max_int)

            energy_map[i, j] = energy[i, j] + min(top, middle, bottom)

    return energy_map


def horizontal_seam(energy_map):
    height, width = energy_map.shape[0], energy_map.shape[1]
    prev, seam = 0, []

    # In this for loop we are adding horizontal seams to the list
    # Since we are finding horizontal seams we are iterating through all the columns (col)
    for i in range(width - 1, -1, -1):
        col = energy_map[:, i]

        if i == width - 1:
            prev = np.argmin(col)   # returns the indices of the minimum value
        else:
            top = (col[prev - 1] if prev - 1 >= 0 else max_int)
            middle = col[prev]
            bottom = (col[prev + 1] if prev + 1 < height else max_int)

            prev += np.argmin([top, middle, bottom]) - 1

        seam.append([i, prev])

    return seam


def vertical_seam(energy_map):
    height, width = energy_map.shape[0], energy_map.shape[1]
    prev, seam = 0, []

    # In this for loop we are adding vertical seams to the list
    # Since we are finding vertical seams we are iterating through all the rows (row)
    for i in range(height - 1, -1, -1):
        row = energy_map[i, :]

        if i == height - 1:
            prev = np.argmin(row)   # returns the indices of the minimum values
        else:
            left = (row[prev - 1] if prev - 1 >= 0 else max_int)
            middle = row[prev]
            right = (row[prev + 1] if prev + 1 < width else max_int)

            prev += np.argmin([left, middle, right]) - 1

        seam.append([prev, i])

    return seam


# This function helps us draw a line on the image to show the seam being removed
def draw_seam(img, seam):
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 0))
    cv2.imshow('seam', img)
    cv2.waitKey(1)


# We are simply removing the horizontal seams by iterating through every index in the "seam" list
# We add the values of the original image as it is but skip the index present in the list
def remove_horizontal_seam(img, seam):
    (height, width, depth) = img.shape
    removed = np.zeros((height - 1, width, depth), np.uint8)

    for (x, y) in reversed(seam):
        removed[0:y, x] = img[0:y, x]
        removed[y:height - 1, x] = img[y + 1:height, x]

    return removed


# We are simply removing the vertical seams by iterating through every index in the "seam" list
# We add the values of the original image as it is but skip the index present in the list
def remove_vertical_seam(img, seam):
    (height, width, depth) = img.shape
    removed = np.zeros((height, width - 1, depth), np.uint8)

    for (x, y) in reversed(seam):
        removed[y, 0:x] = img[y, 0:x]
        removed[y, x:width - 1] = img[y, x + 1:width]

    return removed


def resize(img):
    result = img

    img_height, img_width = img.shape[0], img.shape[1]

    width = int(input('\nEnter Width of the Resized Image: '))

    ratio = img_width / width
    height = int(img_height / ratio)

    print("\n Note: Height is not taken as user input in order to maintain the aspect ratio of the image.")

    cv2.namedWindow('seam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('seam', 300, 300)
    cv2.imwrite("Initial.jpg", result)
    # dx and dy are number of seams we have to remove from our image
    dx = img_width - width if img_width > width else 0
    dy = img_height - height if img_height > height else 0

    print("\n Removing Horizontal Seams...")
    for i in range(dy):
        energy_map = cumulative_energies_horizontal(energy(result))
        seam = horizontal_seam(energy_map)
        draw_seam(result, seam)
        result = remove_horizontal_seam(result, seam)

    print(" Removing Vertical Seams...")
    for i in range(dx):
        energy_map = cumulative_energies_vertical(energy(result))
        seam = vertical_seam(energy_map)
        draw_seam(result, seam)
        result = remove_vertical_seam(result, seam)

    cv2.imwrite('Result.jpg', result)

    print(f"\nFinal Size: {result.shape[1]} x {result.shape[0]} x {result.shape[2]}")

    cv2.imshow('seam', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("1.jpg")

# To maintain the aspect ratio of image
x, y = img.shape[:2]
ratio = y / 500
new_height = x / ratio

img = cv2.resize(img, (500, int(new_height)))     # Resize the image only when the size is too large

print(f"Initial Size: {img.shape[1]} x {img.shape[0]} x {img.shape[2]}")

resize(img)
