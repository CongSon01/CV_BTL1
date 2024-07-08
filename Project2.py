import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import BoxExtract

processed_images = []
im = Image.open('./data/012_0.png')
numpy = np.array(im)[:, :, 0]

resized = cv2.resize(numpy, (1100, 1700))
cut_images = BoxExtract.box_extraction(resized)
# Số lượng box cần vẽ
num_boxes = len(cut_images)

# Tính toán số lượng hàng và cột cho subplots
cols = 5  # Ví dụ, bạn có thể thay đổi số cột này tuỳ ý
rows = (num_boxes + cols - 1) // cols

# Tạo figure và subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
axes = axes.flatten()

for i, box in enumerate(cut_images):
    axes[i].imshow(cv2.cvtColor(box, cv2.COLOR_BGR2RGB))
    # axes[i].set_title(f'Box {i+1}')
    axes[i].axis('off')

# Ẩn các subplot thừa
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Four-corner-marking

# # Hàm để vẽ bảng
# def draw_grid(image, rows, cols):
#     # Lấy kích thước của hình ảnh
#     height, width = image.shape

#     # Tính toán khoảng cách giữa các đường ngang và dọc
#     row_height = height // rows
#     col_width = width // cols

#     # Vẽ các đường ngang
#     for i in range(1, rows):
#         y = i * row_height
#         cv2.line(image, (0, y), (width, y), (0, 255, 0), 2)

#     # Vẽ các đường dọc
#     for i in range(1, cols):
#         x = i * col_width
#         cv2.line(image, (x, 0), (x, height), (0,255, 0), 2)

#     return image

# def show_image(img):
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

# im = Image.open('001_0.png')
# numpy = np.array(im)[:, :, 0]

# resized = cv2.resize(numpy, (1100, 1700))

# cut_images = CutUp.cut_image(resized)

# # Gọi hàm để vẽ bảng lên hình ảnh
# grid_img = draw_grid(cut_images, 30, 4)

# # Hiển thị hình ảnh đã vẽ bảng
# show_image(grid_img)