from io import BytesIO
from aiogram import Bot, F, Router
from aiogram.types import Message, BufferedInputFile
import logging

from scripts.bb_text_extractor import get_masked_image
from scripts.yandex_ocr import get_yandex_ocr
from scripts.llm import Mixtral, llm
from scripts.tf_idf_search import cosine_search

from PIL import Image

import datetime as dt

router = Router()

mixtral = Mixtral.from_llm(llm, verbose=False)

@router.message(F.photo)
async def get_photo(message: Message, bot: Bot):
    photo = message.photo[-1]

    msg = await message.reply("Первичная обработка фотографии\n|----")

    file = BytesIO()

    await bot.download(photo, destination=file)

    image = Image.open(file).convert("RGB")

    masked_image = get_masked_image(image, mask_padding=25)
    
    masked_image.save("test_pic.png")
    logging.info('The masked text was received and saved')
    await msg.edit_text('Считывание текста с фото\n#|---')

    text = get_yandex_ocr("test_pic.png").strip().lower().replace("\n", " ")

    logging.info('Yandex OCR has worked out')
    await msg.edit_text('Обработка полученного текста\n##|--')

    res = mixtral.text_normalization(text)
    text = res.content.strip().replace("\n", " ")

    logging.info('error correction with the mixtral worked out')
    await msg.edit_text('Поиск книги в базе\n###|-')

    row_data = cosine_search(text)
    logging.info('Cosine search worked out')
    await msg.edit_text('Суммаризация найденной информации\n####|')

    description = (
    row_data["description"] if row_data["description"] else "Описание отсутствует"
    )

    reviews = row_data["reviews"].split(";\n") if row_data["reviews"] else ""

    # await message.answer(f'Описание: {description}')
    # if reviews:
    #     await message.answer(f'Отзывы: {" ".join(reviews)}')
    # else:
    #     await message.answer(f'Нет отзывов')

    header_template = f'''
Название: {row_data['name']}
Оценка: {row_data['rate']}
Ссылка на книгу: {row_data['link']}

    '''

    text = mixtral.reviews_summary(header_template, description=description, reviews=reviews).content

    logging.info('Mixtral answer generation worked out')

    await message.answer(header_template + text)

    await msg.delete()


@router.message()
async def no_filter_answer(message: Message):
    await message.reply("Чтобы начать работу бота, просто отправьте любую фотографию")
