import random
import cv2
import numpy as np


def crop(img, offset):
    h1, w1, h2, w2 = offset
    return img[h1:h2, w1:w2, ...]


def random_crop(img, target_size):
    h, w = img.shape[0:2]
    th, tw = target_size
    h1 = random.randint(0, max(0, h - th))
    w1 = random.randint(0, max(0, w - tw))
    h2 = min(h1 + th, h)
    w2 = min(w1 + tw, w)
    return crop(img, [h1, w1, h2, w2])


def center_crop(img, target_size):
    h, w = img.shape[0:2]
    th, tw = target_size
    h1 = max(0, int((h - th) / 2))
    w1 = max(0, int((w - tw) / 2))
    h2 = min(h1 + th, h)
    w2 = min(w1 + tw, w)
    return crop(img, [h1, w1, h2, w2])

def pad(img, offeset, value=0):
    h1, w1, h2, w2 = offeset
    img = cv2.copyMakeBorder(
        img, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=value)
    return img


def rescale(img, dsize, interpolation=cv2.INTER_LINEAR):
    img = cv2.resize(
        img,
        dsize=tuple(dsize),
        interpolation = interpolation)
    return img


def rotation(img, degrees, interpolation=cv2.INTER_LINEAR, value=0):
    if isinstance(degrees, list):
        if len(degrees) == 2:
            degree = random.uniform(degrees[0], degrees[1])
        else:
            degree = random.choice(degrees)
    else:
        degree = degrees

    h, w = img.shape[0:2]
    center = (w / 2, h / 2)
    map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

    img = cv2.warpAffine(
        img,
        map_matrix, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=value)

    return img

def flip(img):
    return np.fliplr(img)

def random_flip(img):
    if random.random() < 0.5:
        return flip(img)
    else:
        return img
