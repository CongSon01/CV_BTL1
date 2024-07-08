import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Contour:
    def __init__(self, contour_data):
        self.x, self.y, self.width, self.height = cv.boundingRect(contour_data) #  hình chữ nhật bao quanh contour 
        return

    def get_precedence(self, row_and_column_classifiers):
        y_values_of_rows, horizontal_one_x = row_and_column_classifiers
        x, y, width, height = self.x, self.y, self.width, self.height

        for row_y in y_values_of_rows:
            if row_y - height / 3 < y < row_y + height / 3:
                row_num = y_values_of_rows.index(row_y)   # Chỉ số của contour hợp lệ
                break
            
        if x < horizontal_one_x or horizontal_one_x * 2 < x < horizontal_one_x * 3:
            self.x = x + 40     # trừ STT
            self.width = width - 40

        first_column = x < horizontal_one_x * 2
        if first_column:
            column = 0
        else:
            column = 1
        # Nếu là cột đầu tiên sẽ + 10.000
        # Không phải + 10.000.000
        # Xét mức độ ưu tiên của cột 1 trước sau đến cột 2
        return column * 10000000 + row_num * 10000 + x

def show_image(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def show_matrix(matrix):
    plt.imshow(matrix)
    plt.colorbar()
    plt.show()

def visualize_contours_red(image, contours):
    image_copy = image.copy()
    cv.drawContours(image_copy, contours, -1, (0, 0, 255), 2)  # Sử dụng (0, 0, 255) cho màu đỏ
    plt.imshow(cv.cvtColor(image_copy, cv.COLOR_BGR2RGB))
    plt.show()

def cut_image(img):
    height, width = img.shape[:2]

    y_start = int(height / 50)
    y_end = int(height * 49 / 50)

    x_start = int(width / 50)
    x_end = int(width * 49 / 50)

    img = img[y_start: y_end, x_start: x_end]
    return img

def box_extraction(img):
    img = cut_image(img)

    # Nhị phân ảnh dựa trên Adaptive mean
    extraction_template = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 2)
    extraction_template = 255 - extraction_template  # Invert the image (White => Black)

    kernel_length = np.array(img).shape[1] // 30 #35
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, np.array(img).shape[0] // 45)) # [35x1]
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (np.array(img).shape[1] // 55, 1)) # [1x35]

    # Tăng cường độ dày của các đường dọc
    # Áp dụng phép mở
    vertical_template = cv.erode(extraction_template, vertical_kernel, iterations=3)
    vertical_lines = cv.dilate(vertical_template, vertical_kernel, iterations=3)

    # Tăng cường độ dày của các đường ngang
    # Áp dụng phép mở
    horizontal_template = cv.erode(extraction_template, horizontal_kernel, iterations=3)
    horizontal_lines = cv.dilate(horizontal_template, horizontal_kernel, iterations=3)

    alpha = 0.5  # Weighting parameters for adding two images.
    beta = 1.0 - alpha
    kernel_3x3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    template_sum = cv.addWeighted(vertical_lines, alpha, horizontal_lines, beta, 0.0)
    template_sum = cv.erode(~template_sum, kernel_3x3, iterations=2)
    (thresh, template_sum) = cv.threshold(template_sum, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    extraction_template = cv.morphologyEx(template_sum, cv.MORPH_OPEN, vertical_kernel)
    extraction_template = cv.morphologyEx(extraction_template, cv.MORPH_OPEN, horizontal_kernel)
    breakpoint()

    contours, hierarchy = cv.findContours(extraction_template, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    breakpoint()

    contours = [Contour(contour) for contour in contours]

    contours = filter_contours(contours, img)
    breakpoint()

    y_values_of_rows = get_row_y_values(contours)
    horizontal_one_x = round(extraction_template.shape[1] / 4) # Weight của 1 ô
    row_and_column_classifiers = (y_values_of_rows, horizontal_one_x)

    # Sort theo mức độ ưu tiên (Cột 1 sẽ được hiển thị trước => Tới cột 2)
    contours.sort(key=lambda contour: Contour.get_precedence(contour, row_and_column_classifiers))

    extracted_boxes = []
    for contour in contours:
        x, y, width, height = contour.x, contour.y, contour.width, contour.height
        extracted_boxes.append(img[y:y + height, x:x + width])
    return extracted_boxes


def filter_contours(contours, img):
    filtered_contours = []
    for contour in contours:
        x, y, width, height = contour.x, contour.y, contour.width, contour.height
        # breakpoint()
        # img[y:y + hecight, x:x + width]
        if (img.shape[1]/4 - 100) < width:  # Input image is 1632x1056 px.
            filtered_contours.append(contour)
    return filtered_contours


# Trích xuất các giá trị y của các hàng chứa contours
def get_row_y_values(contours):
    y_values = []

    for contour in contours:
        x, y, width, height = contour.x, contour.y, contour.width, contour.height
        new_row = True

        for row_y in y_values:
            same_row = y - height / 3 < row_y < y + height / 3
            if same_row:
                new_row = False
        if new_row:
            y_values.append(y)

    y_values.sort(key=lambda y_value: y_value)
    return y_values
