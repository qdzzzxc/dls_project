from aiogram.types import InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder

import logging

def not_this_book():
    kb_builder = InlineKeyboardBuilder()
    button_names = ['Найдена не та книга']
    back_data = ['FindTop10books']
    buttons = [InlineKeyboardButton(text=text, callback_data=data) for text, data in zip(button_names, back_data)]
    kb_builder.row(*buttons, width=1)
    return kb_builder.as_markup(resize_keyboard=True)

def not_this_book_again(list_of_books):
    kb_builder = InlineKeyboardBuilder()
    button_names = list(list_of_books.name.values) + ['❌ Нужной книги нет в представленном списке ❌']
    back_data = [f'book_{x}' for x in range(len(list_of_books))] + ['FindByUserRequest']
    logging.info(button_names)
    logging.info(back_data)
    buttons = [InlineKeyboardButton(text=text, callback_data=data) for text, data in zip(button_names, back_data)]
    kb_builder.row(*buttons, width=1)
    return kb_builder.as_markup(resize_keyboard=True)

def not_this_book_again_again():
    kb_builder = InlineKeyboardBuilder()
    button_names = ['Нужной книги нет в списке']
    back_data = ['HaveNotThisBook']
    buttons = [InlineKeyboardButton(text=text, callback_data=data) for text, data in zip(button_names, back_data)]
    kb_builder.row(*buttons, width=1)
    return kb_builder.as_markup(resize_keyboard=True)
