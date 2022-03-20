import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import io
import cv2
from PIL import Image
import mrcnn.model as modellib

from config import CustomConfig
from calories import calories

from skimage.measure import find_contours
import matplotlib.pyplot as plt
import telebot
import json
import os
import colorsys
import random
from PIL import Image, ImageDraw, ImageFont

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


data = json.load(open("telegram.json", "r", encoding="utf-8"))

# BOT PROPERTIES
admins = data["admins"]
tg_token = data["token"]
file_name = data["file"]

# COMMAND REPLIES
add_text = "Привет, это бот для проекта по распознаванию еды и определению количества калорий.\nОтправь фото и получи результат!"
about_text = "Бот создан для МНСК-2021.\nАвторы: Оконешников Дмитрий, Паньков Максим.\nGithub: https://github.com/MagicWinnie/FoodRecognition"
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
    DETECTION_MIN_CONFIDENCE = 0.2
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs/")
model.load_weights(file_name, by_name=True)
model.keras_model._make_predict_function()


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = list(
        map(lambda x: (255 * int(x[-1]), 255 * int(x[1]), 255 * int(x[0])), colors)
    )
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c]
        )
    return image


def display_instances(
    image,
    boxes,
    masks,
    class_ids,
    class_names,
    scores=None,
    figsize=(16, 16),
    ax=None,
    show_mask=True,
    show_bbox=True,
    colors=None,
    captions=None,
):

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = colors or random_colors(N)

    for i in range(N):
        color = colors[i][::-1]
        if class_names[class_ids[i]].lower() == "orange":
            continue
        if (not np.any(boxes[i])) or scores[i] < 0.2:
            continue

        y1, x1, y2, x2 = boxes[i]

        image = cv2.rectangle(
            image, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2
        )

        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]

        caption = "{}".format(to_russian.get(label, label)) if score else label
        image = cv2.rectangle(
            image,
            (int(x1), int(y1) - 40),
            (int(x2) + 100, int(y1) - 5),
            color=(0, 0, 0),
            thickness=-1,
        )

        font = ImageFont.truetype("arial.ttf", 24)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text(
            (int(x1) + 5, int(y1) - 37), caption, font=font, fill=(255, 255, 255, 0)
        )
        image = np.array(img_pil)
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for verts in contours:
            verts = np.fliplr(verts) - 1
            for i in range(len(verts)):
                verts[i][0] = int(verts[i][0])
                verts[i][1] = int(verts[i][1])
            verts = np.array(verts, np.int32)
            verts = verts.reshape((-1, 1, 2))
            image = cv2.polylines(image, [verts], True, color, 3)

    return image  # cv2.imencode(".jpg", image)[1]


classes = [
    "BG",
    "banana",
    "apple",
    "juice-orange",
    "mandarine",
    "kiwi",
    "fish",
    "rice",
    "bread-black",
    "milk",
    "tea-green",
    "chocolate",
    "tea-black",
    "orange",
    "bread",
]

to_russian = {
    "BG": "Задний фон",
    "banana": "Банан",
    "apple": "Яблоко",
    "juice-orange": "Апельсиновый сок",
    "mandarine": "Мандарин",
    "kiwi": "Киви",
    "fish": "Рыба",
    "rice": "Рис",
    "bread-black": "Хлеб",
    "bread": "Хлеб",
    "milk": "Молоко",
    "tea-green": "Чай",
    "tea-black": "Чай",
    "chocolate": "Шоколад",
    "orange": "Апельсин",
}

bot = telebot.TeleBot(tg_token)
print("BOT STARTED")


@bot.message_handler(commands=["start"])
def add_id(message):
    bot.send_message(message.chat.id, add_text)
    bot.send_message(message.chat.id, help_text)
    # bot.send_message(message.chat.id, about_text)


@bot.message_handler(commands=["help"])
def get_help(message):
    bot.send_message(message.chat.id, help_text)


@bot.message_handler(commands=["about"])
def get_about(message):
    bot.send_message(message.chat.id, about_text)


def send_photo(img, message):
    bot.send_photo(message.chat.id, img, reply_to_message_id=message.message_id)


@bot.message_handler(content_types=["photo"])
def photo(message):
    print("---RECEIVED IMAGE---")
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    print("---STARTING DETECTION---")
    img = np.array(Image.open(io.BytesIO(downloaded_file)), dtype="uint8")

    # center = img.shape
    # w = img.shape[1]
    # h = img.shape[1]
    # x = center[1] / 2 - w / 2
    # y = center[0] / 2 - h / 2
    # img = img[int(y) : int(y + h), int(x) : int(x + w)]
    # img = cv2.resize(img, (256, 256))

    desired_size = img.shape[0]
    old_size = img.shape[:2]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    img = cv2.resize(img, (256, 256))
    

    bot.send_message(
        message.chat.id, "Пожалуйста, подождите... Это займет около минуты."
    )
    r1 = model.detect([img], verbose=0)[0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_orig = img.copy()

    instance = display_instances(
        img,
        r1["rois"],
        r1["masks"],
        r1["class_ids"],
        classes,
        r1["scores"],
        ax=get_ax(1),
    )

    send_photo(
        cv2.imencode(".jpg", instance)[1],
        message,
    )
    try:
        # send_photo(display_instances(img, r1['rois'], r1['masks'], r1['class_ids'],
        # classes, r1['scores'], ax=get_ax(1)), message)
        pass
    except Exception as e:
        bot.send_message(
            message.chat.id, "Извините, случилась ошибка. Ничего не найдено."
        )
        print("[ERROR]", e)
    else:
        # cv2.imwrite("img.jpg", img_orig)
        c = calories(img_orig, r1, classes)
        dict_cal = c.process()
        out_message = "Калории:\n"
        for key in dict_cal:
            out_message += f"\t{to_russian.get(key, key)}: {dict_cal[key]}\n"
        if out_message == "":
            bot.send_message(message.chat.id, "Ничего не найдено.")
        else:
            bot.send_message(message.chat.id, out_message)
    print("---DONE---")


bot.polling()
