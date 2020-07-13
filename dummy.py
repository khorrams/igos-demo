import urllib.request
from urllib.parse import urlparse
from os.path import splitext
import os
import time
import hashlib


class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) " \
              "Chrome/47.0.2526.69 Safari/537.36"


def hash_image(url):  # todo for non type file MIMETYPE
    extension = splitext(urlparse(url).path)[1]
    h = hashlib.new('ripemd160')
    h.update(bytes(url, 'utf8'))
    return h.hexdigest() + extension


def download_image(url, dest_img):
    img_path = dest_img + hash_image(url)
    if os.path.exists(img_path):
        return img_path
    else:
        urllib._urlopener = AppURLopener()
        urllib._urlopener.retrieve(url, img_path)
        return img_path


def video_image_name(url):
    # path to the downloaded image
    taregt = 'static/images/downloaded/'
    if not os.path.exists(taregt):
        os.makedirs(taregt)

    image_path = download_image(url, taregt)

    # path of the video to be saved
    # video_name = image_path.split('/')[-1].split('.')[0] + '.mp4'

    return image_path


def carousel_name(url, carousel_path='./static/carousel/images/'):
    img_num = url.split('$%#carousel')[1]
    image_path = carousel_path + img_num + '.jpg'
    # saliency_path = carousel_path + 'images/' + img_num + '_saliency.png'
    # video_name = carousel_path + 'videos/' + img_num + '.mp4'

    return image_path #, saliency_path, video_name


if __name__ == '__main__':
    download_image('https://files.slack.com/files-pri/TDT9UN4LE-FFQNUN7SM/image.png', './')
