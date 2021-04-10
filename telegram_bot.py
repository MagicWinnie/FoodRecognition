from logging import exception
import numpy as np
import io
import cv2
from PIL import Image
from mrcnn.model import log
import mrcnn.model as modellib
# from mrcnn.visualize import display_images
from mrcnn import visualize
from mrcnn import utils
from config import CustomConfig, FoodDataset

from skimage.measure import find_contours
from matplotlib.patches import Polygon
from matplotlib import patches,  lines
import matplotlib.pyplot as plt
import telebot
import json
import os
import colorsys
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data = json.load(open('telegram.json', 'r', encoding='utf-8'))

# BOT PROPERTIES
admins = data['admins']
tg_token = data['token']
file_name = data['file']

# COMMAND REPLIES
add_text = 'Привет, это бот для проекта по распознаванию еды и определению количества калорий.\nОтправь фото и получи результат!'
about_text = 'Бот создан для МНСК-2021.\nАвторы: Оконешников Дмитрий, Паньков Максим.\nGithub: https://github.com/MagicWinnie/FoodRecognition'
help_text = """
            Доступные команды:
            /start - начало работы с ботом
            /help  - команды бота
            /about - информация о боте
            """

# MODEL


config = CustomConfig()


class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 14
    DETECTION_MIN_CONFIDENCE = 0
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256


config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference',
                          config=config,
                          model_dir='logs/')
model.load_weights(file_name, by_name=True)
model.keras_model._make_predict_function()


def get_ax(rows=1, cols=1, size=16):
    """"Return a Matplotlib Axes array to be used in all visualizations in the notebook. Provide a central point to control graph sizes. Adjust the size attribute to control how big to render images"""
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = list(
        map(lambda x: (255*int(x[-1]), 255*int(x[1]), 255*int(x[0])), colors))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None,
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    for i in range(N):
        color = colors[i][::-1]

        # Bounding box
        if (not np.any(boxes[i])) or scores[i] < 0.4:
            continue

        y1, x1, y2, x2 = boxes[i]

        image = cv2.rectangle(image, (int(x1), int(y1)),
                              (int(x2), int(y2)), color=color, thickness=2)

        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label
        image = cv2.rectangle(image, (int(x1), int(y1) - 40),
                              (int(x2), int(y1) - 5), color=(0, 0, 0), thickness=-1)
        image = cv2.putText(image, caption, (int(x1), int(
            y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # Mask
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            for i in range(len(verts)):
                verts[i][0] = int(verts[i][0])
                verts[i][1] = int(verts[i][1])
            verts = np.array(verts, np.int32)
            verts = verts.reshape((-1, 1, 2))
            image = cv2.polylines(image, [verts], True, color, 3)

    return cv2.imencode('.jpg', image)[1]


classes = ['BG', 'banana', 'apple', 'juice-orange', 'mandarine', 'kiwi', 'fish', 'rice',
           'bread-black', 'milk', 'tea-green', 'chocolate', 'tea-black', 'orange', 'bread']

bot = telebot.TeleBot(tg_token)
print("BOT STARTED")


@bot.message_handler(commands=['start'])
def add_id(message):
    bot.send_message(message.chat.id, add_text)
    bot.send_message(message.chat.id, help_text)
    bot.send_message(message.chat.id, about_text)


@bot.message_handler(commands=['help'])
def get_help(message):
    bot.send_message(message.chat.id, help_text)


@bot.message_handler(commands=['about'])
def get_about(message):
    bot.send_message(message.chat.id, about_text)


def send_photo(img, message):
    bot.send_photo(message.chat.id, img,
                   reply_to_message_id=message.message_id)


@bot.message_handler(content_types=['photo'])
def photo(message):
    print("---RECEIVED IMAGE---")
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    print("---STARTING DETECTION---")
    img = np.array(Image.open(io.BytesIO(downloaded_file)), dtype='uint8')

    bot.send_message(message.chat.id, "Please wait...")
    r1 = model.detect([img], verbose=1)[0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    try:
        send_photo(display_instances(img, r1['rois'], r1['masks'], r1['class_ids'],
                                     classes, r1['scores'], ax=get_ax(1)), message)
    except:
        bot.send_message(message.chat.id, "Nothing found!")


bot.polling()
