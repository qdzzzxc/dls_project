from aiogram.types import InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder


def not_this_book():
    kb_builder = InlineKeyboardBuilder()
    button_names = ["Найдена не та книга"]
    back_data = ["FindTop10books"]
    buttons = [
        InlineKeyboardButton(text=text, callback_data=data)
        for text, data in zip(button_names, back_data)
    ]
    kb_builder.row(*buttons, width=1)
    return kb_builder.as_markup(resize_keyboard=True)


def try_text_search():
    kb_builder = InlineKeyboardBuilder()
    button_names = ["Сделать текстовый запрос"]
    back_data = ["TextSearch"]
    buttons = [
        InlineKeyboardButton(text=text, callback_data=data)
        for text, data in zip(button_names, back_data)
    ]
    kb_builder.row(*buttons, width=1)
    return kb_builder.as_markup(resize_keyboard=True)


def not_this_book_again(list_of_books, typed=False):
    kb_builder = InlineKeyboardBuilder()
    button_names = list(list_of_books.name.values) + [
        "❌ Нужной книги нет в представленном списке ❌"
    ]
    back_data = list(map(str, range(len(button_names) - 1)))
    if typed:
        back_data.append("SorryHaveNot")
    else:
        back_data.append("FindByUserRequest")
    buttons = [
        InlineKeyboardButton(text=text, callback_data=data)
        for text, data in zip(button_names, back_data)
    ]
    kb_builder.row(*buttons, width=1)
    return kb_builder.as_markup(resize_keyboard=True)
