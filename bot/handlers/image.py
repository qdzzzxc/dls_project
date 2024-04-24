from io import BytesIO
from aiogram import Bot, F, Router
from aiogram.types import Message, CallbackQuery
import logging

from keyboards import not_this_book, not_this_book_again

from scripts.bb_text_extractor import get_masked_image
from scripts.yandex_ocr import get_yandex_ocr
from scripts.llm import Mixtral, llm
from scripts.tf_idf_search import cosine_search, find_top_ten_books

from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

from PIL import Image

import re

router = Router()

mixtral = Mixtral.from_llm(llm, verbose=False)


class FindText(StatesGroup):
    is_correct = State()
    choose_from_10 = State()
    wait_for_type_request = State()
    choose_from_10_typed = State()


def llm_from_series(row_data):
    description = (
        row_data["description"] if row_data["description"] else "Описание отсутствует"
    )

    reviews = row_data["reviews"].split(";\n") if row_data["reviews"] else ""

    header_template = f"""
        Название: {row_data['name']}
        Оценка: {row_data['rate']}
        Ссылка на книгу: {row_data['link']}
        """

    return header_template, mixtral.reviews_summary(
        header_template, description=description, reviews=reviews
    ).content


def remove_urls(text):
    pattern = r"https?://\S+|www\.\S+"
    return re.sub(pattern, "", text)


@router.message(F.photo)
async def get_llm_description(message: Message, state: FSMContext, bot: Bot):
    photo = message.photo[-1]

    msg = await message.reply("Первичная обработка фотографии\n|----")

    file = BytesIO()

    await bot.download(photo, destination=file)

    image = Image.open(file).convert("RGB")

    try:
        masked_image = get_masked_image(image, mask_padding=25)

        masked_image.save("test_pic.png")
        logging.info("The masked text was received and saved")
        await msg.edit_text("Считывание текста с фото\n#|---")

        text = get_yandex_ocr("test_pic.png").strip().lower().replace("\n", " ")

        logging.info("Yandex OCR has worked out")
        logging.info(text)
        await msg.edit_text("Обработка полученного текста\n##|--")

        res = mixtral.text_normalization(text)
        text_mixtral = res.content.split("\n")[0].strip()

        logging.info("error correction with the mixtral worked out")
        logging.info(text_mixtral)
        await msg.edit_text("Поиск книги в базе\n###|-")

        row_data = cosine_search(text + ";" + text_mixtral)
        logging.info("Cosine search worked out")
        await msg.edit_text("Суммаризация найденной информации\n####|")

        header_template, text = llm_from_series(row_data)

        logging.info("Mixtral answer generation worked out")
        await message.answer(
            header_template + remove_urls(text), reply_markup=not_this_book()
        )
        await state.set_state(FindText.is_correct)
        await state.update_data(text=text)

        await msg.delete()
    except Exception as e:
        logging.error(e)
        await msg.edit_text(
            f"Произошла ошибка {e.__class__.__name__}, попробуйте снова"
        )
        return


@router.callback_query(F.data == "FindTop10books", FindText.is_correct)
async def get_top_10_books(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    text = data["text"]
    await callback.message.edit_text(f"Идёт поиск всех похожих книг по тексту {text}")
    top_10_books = find_top_ten_books(text)
    typed = data.get("typed", False)
    await callback.message.edit_text(
        "Выберите, какой текст вам нужен",
        reply_markup=not_this_book_again(top_10_books, typed=typed),
    )
    await state.update_data(ten_books=top_10_books)

    await state.set_state(FindText.choose_from_10)


@router.callback_query(F.data == "SorryHaveNot")
async def have_not_this_book(callback: CallbackQuery):
    await callback.message.edit_text(
        "К сожалению, моя база данных не содержит информацию о нужной вам книге"
    )


@router.callback_query(FindText.choose_from_10)
async def choose_from_10_books(callback: CallbackQuery, state: FSMContext):
    if callback.data in ["FindByUserRequest"]:
        await callback.message.edit_text(
            "Отправьте сообщением название нужной вам книги"
        )
        await state.set_state(FindText.wait_for_type_request)
    else:
        data = await state.get_data()
        row_data = data["ten_books"].iloc[int(callback.data)]

        await callback.message.edit_text("Происходит генерация ответа")

        header_template, text = llm_from_series(row_data)
        logging.info("Mixtral answer generation worked out")

        await callback.message.answer(
            header_template + remove_urls(text), reply_markup=not_this_book()
        )
        await callback.message.delete()
        await state.clear()
        await state.set_state(FindText.is_correct)
        await state.update_data(text=text)


@router.message(FindText.wait_for_type_request)
async def type_your_book(message: Message, state: FSMContext):
    text = message.text

    msg = await message.answer("Поиск книги в базе\n|-")

    row_data = cosine_search(text)
    logging.info("Cosine search worked out")
    await msg.edit_text("Суммаризация найденной информации\n#|")

    header_template, text = llm_from_series(row_data)

    logging.info("Mixtral answer generation worked out")

    await message.answer(header_template + text, reply_markup=not_this_book())
    await state.set_state(FindText.is_correct)
    await state.update_data(typed=True)
    await state.update_data(text=text)

    await msg.delete()


@router.message()
async def no_filter_answer(message: Message):
    await message.reply("Чтобы начать работу бота, просто отправьте любую фотографию")
