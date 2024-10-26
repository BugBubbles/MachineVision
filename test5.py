import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace


def harris_corner_detection(
    image_path, block_size=2, aperture_size=3, k=0.04, threshold=0.01
):
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path)
    cv2.imwrite("input.png", image)
    image = gaussian_filter(image, sigma=1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Harris角点检测器
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    image_dst = image[:, :, :]
    # dst = cv2.dilate(dst,None)#https://cloud.tencent.com/developer/article/1123720
    image_dst[dst > 0.01 * dst.max()] = [0, 0, 255]

    """
    harris_response = cv2.cornerHarris(gray, block_size, aperture_size, k)

    # 阈值处理
    threshold_value = threshold * harris_response.max()
    corners = np.argwhere(harris_response > threshold_value)
    """
    corners = np.argwhere(dst > 0.01 * dst.max())
    return image, gray, corners


"""
def compute_hessian_matrix(gray, x, y):
    # 计算灰度图像在x和y方向上的梯度
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算各个二阶导数
    Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
    Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)

    # 计算Hessian矩阵
    H = np.array([[Ixx[y, x], Ixy[y, x]], [Ixy[y, x], Iyy[y, x]]])

    return H
"""


def refine_subpixel_location(gray, corners):
    refined_corners = []

    for corner in corners:
        y, x = corner

        # 计算一阶偏导数
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
        Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
        Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)

        # 计算Hessian矩阵
        H = np.array([[Ixx[y, x], Ixy[y, x]], [Ixy[y, x], Iyy[y, x]]])

        # 计算梯度向量
        gradient = np.array([Ix[y, x], Iy[y, x]])

        # 计算亚像素位置增量: Δp = -H^-1 * gradient
        H_inv = np.linalg.inv(H)
        delta_p = -H_inv @ gradient

        # 计算亚像素坐标
        subpixel_x = x + delta_p[0]
        subpixel_y = y + delta_p[1]

        refined_corners.append((subpixel_x, subpixel_y))

    return np.mean(refined_corners, axis=0)


def main(image_path):
    image, gray, corners = harris_corner_detection(image_path)

    # 使用Hessian矩阵和二阶泰勒展开式计算亚像素级别的角点位置
    refined_corners = refine_subpixel_location(gray, corners)

    # 显示初始角点
    for corner in corners:
        y, x = corner
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # 显示亚像素级别角点
    for refined_corner in refined_corners:
        x, y = refined_corner
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

    # 显示结果图像
    # cv2.imshow('Refined Corners', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("output.png", image)

    # 输出亚像素级别的角点坐标
    print("Refined Corners:")
    for refined_corner in refined_corners:
        x, y = refined_corner
        print(f"({x:.3f}, {y:.3f})")


# 调用main函数
image_path = "corner.bmp"
main(image_path)
