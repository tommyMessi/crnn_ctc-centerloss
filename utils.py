# -*- coding: utf-8 -*-
import os
import io
import cv2
import numpy as np

from PIL import Image


class Rect(object):
    """基本矩形框类"""
    x = None
    y = None
    w = None
    h = None
    angle = 0
    rotated_rect = None

    def __init__(self, rect):
        if len(rect) == 3:
            self.angle = rect[2]
            self.rx, self.ry = rect[0]
            self.rw, self.rh = rect[1]
            self.rotated_rect = tuple(rect)
            points = cv2.boxPoints(rect)
            self.x, self.y, self.w, self.h = cv2.boundingRect(points)
        else:
            self.x, self.y, self.w, self.h = rect
            self.angle = 0
            self.rotated_rect = (
                (self.center_x, self.center_y), (self.w, self.h), 0)

    @property
    def left(self):
        """
        矩形框的左边界
        :return: 矩形框的左边界
        """
        return self.x

    @property
    def top(self):
        """
        矩形框的上边界
        :return: 矩形框的上边界
        """
        return self.y

    @property
    def right(self):
        """
        矩形框的右边界
        :return: 矩形框的右边界
        """
        return self.x + self.w

    @property
    def bottom(self):
        """
        矩形的下边界
        :return: 矩形的下边界
        """
        return self.y + self.h

    @property
    def center_x(self):
        """
        矩形框的中心x坐标
        :return: 矩形框的中心x坐标
        """
        return self.x + self.w / 2

    @property
    def center_y(self):
        """
        矩形框的中心y坐标
        :return: 矩形框的中心y坐标
        """
        return self.y + self.h / 2

    def union(self, rect2):
        """合并矩形"""
        self.x = min(self.x, rect2.x)
        self.y = min(self.y, rect2.y)
        self.w = max(self.right, rect2.right) - self.x
        self.h = max(self.bottom, rect2.bottom) - self.y

    def extend(self, dw, dh):
        """扩大矩形"""
        self.w += dw
        self.h += dh
        return self

    def move(self, dx, dy):
        """移动矩形"""
        self.x += dx
        self.y += dy
        return self

    def intersects(self, rect2):
        """判断矩形是否相交"""
        return rect2.right > self.x and \
               rect2.bottom > self.y and \
               rect2.x < self.right and \
               rect2.y < self.bottom

    def copy(self):
        return Rect((self.x, self.y, self.w, self.h))

    @property
    def area(self):
        return self.w * self.h

    @property
    def rect_3d(self):
        """
        返回类似于opencv RotatedRect结构的矩形框tuple
        :return:
        """
        return [self.x, self.y], [self.w, self.h], 0

    def __str__(self):
        return "x: %d, y: %d, w: %d, h:%d" % (self.x, self.y, self.w, self.h)


def hsv_diff(hsv1, hsv2):
    if (hsv_is_black(hsv1) and hsv_is_black(hsv2)) or (hsv_is_white(hsv1) and hsv_is_white(hsv2)):
        return np.array([0, 0, 0])
    diff = abs(hsv1 - hsv2)
    if diff[0] > 90:
        diff[0] = 180 - diff[0]
    return diff


def hsv_is_white(hsv):
    return hsv[1] < 20 and hsv[2] > 200


def hsv_is_black(hsv):
    return hsv[2] < 30


def hsv_is_gray(hsv):
    return hsv[1] < 30 or hsv[2] < 30


def is_pil_image(image):
    return 'PIL' in str(type(image))


def is_np_image(image):
    return 'numpy' in str(type(image))


def pil_image_to_np(image):
    np_img = np.array(image.convert('RGB'))
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


def np_image_to_pil(image):
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise Exception("Unsupported image channel: %d" % image.shape[2])
    return Image.fromarray(rgb)


def img_get_binary_data(image):
    image = image_to_pil(image)
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def image_to_np(image):
    if is_pil_image(image):
        return pil_image_to_np(image)
    else:
        return image


def image_to_pil(image):
    if is_np_image(image):
        return np_image_to_pil(image)
    else:
        return image


def hsv_to_rgb(hsv):
    one_pixel_image = np.expand_dims(np.expand_dims(hsv, 0), 0).astype(np.uint8)
    one_pixel_image = cv2.cvtColor(one_pixel_image, cv2.COLOR_HSV2RGB)
    return one_pixel_image[0, 0, :]


def list_dir(path):
    """
    列出文件夹中所有的文件夹
    :param path:
    :return:
    """
    paths = []
    for file_and_dir in os.listdir(path):
        _path = os.path.join(path, file_and_dir)
        if os.path.isdir(_path):
            paths.append(_path)
    return paths


def bgr_2_web(color, flag='opencv'):
    """

    :param color:
    :param flag:
    :return:
    """
    if flag == 'opencv':
        return '#%02x%02x%02x' % (color[2], color[1], color[0])
    else:
        return '#%02x%02x%02x' % color


def web_2_rgb(web_color_str):
    """Return (red, green, blue) for the color given as #rrggbb."""
    web_color_str = web_color_str.lstrip('#')
    lv = len(web_color_str)
    return tuple(
        int(web_color_str[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
