import os
import json
import telebot

data = json.load(open('telegram.json', 'r', encoding='utf-8'))

# BOT PROPERTIES
admins = data['admins']
tg_token = data['token']

# COMMAND REPLIES
add_text   = 'Привет, это бот для проекта по распознаванию еды и определению количества калорий.\nОтправь фото и получи результат!'
about_text = 'Бот создан для МНСК-2021.\nАвторы: Оконешников Дмитрий, Паньков Максим.\nGithub: https://github.com/MagicWinnie/FoodRecognition'
help_text  = """
            Доступные команды:
            /start - начало работы с ботом
            /help  - команды бота
            /about - информация о боте
            """

bot = telebot.TeleBot(token)

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

bot.polling()
