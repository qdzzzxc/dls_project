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

router = Router()

mixtral = Mixtral.from_llm(llm, verbose=False)

class FindText(StatesGroup):
    first_time_completed = State()

def llm_from_series(row_data):
    description = (
    row_data["description"] if row_data["description"] else "Описание отсутствует"
    )

    reviews = row_data["reviews"].split(";\n") if row_data["reviews"] else ""

    header_template = f'''
        Название: {row_data['name']}
        Оценка: {row_data['rate']}
        Ссылка на книгу: {row_data['link']}
        '''

    return header_template, mixtral.reviews_summary(header_template, description=description, reviews=reviews).content

@router.message(F.photo)
async def get_llm_description(message: Message, state: FSMContext, bot: Bot):

    # await message.answer('Привет', reply_markup=not_this_book())

    # await state.update_data(text='достоевский')
    # await state.set_state(FindText.first_time_completed)

    photo = message.photo[-1]

    msg = await message.reply("Первичная обработка фотографии\n|----")

    file = BytesIO()

    await bot.download(photo, destination=file)

    image = Image.open(file).convert("RGB")

    try:
        masked_image = get_masked_image(image, mask_padding=25)
    
        masked_image.save("test_pic.png")
        logging.info('The masked text was received and saved')
        await msg.edit_text('Считывание текста с фото\n#|---')

        text = get_yandex_ocr("test_pic.png").strip().lower().replace("\n", " ")

        logging.info('Yandex OCR has worked out')
        logging.info(text)
        await msg.edit_text('Обработка полученного текста\n##|--')

        res = mixtral.text_normalization(text)
        text_mixtral = res.content.split("\n")[0].strip()

        logging.info('error correction with the mixtral worked out')
        logging.info(text_mixtral)
        await msg.edit_text('Поиск книги в базе\n###|-')

        row_data = cosine_search(text + ';' + text_mixtral)
        logging.info('Cosine search worked out')
        logging.info(row_data)
        await msg.edit_text('Суммаризация найденной информации\n####|')

        
        header_template, text = llm_from_series(row_data)

        logging.info('Mixtral answer generation worked out')

        await message.answer(header_template + text, reply_markup=not_this_book())

        await msg.delete()
    except Exception as e:
        logging.error(e)
        await msg.edit_text(f'Произошла ошибка {e.__class__.__name__}, попробуйте снова')
        return

@router.callback_query(F.data == 'FindTop10books', FindText.first_time_completed)
async def get_top_10_books(callback: CallbackQuery, state: FSMContext):
    text = await state.get_data()
    text = text['text']
    logging.info(text)
    top_10_books = find_top_ten_books(text)
    await callback.message.edit_text(f'Идёт поиск всех похожих книг по тексту {text}', reply_markup=not_this_book_again(top_10_books))

    await state.clear()

@router.callback_query(F.data == 'FindByUserRequest')
async def try_find_by_text(callback: CallbackQuery):
    await callback.message.edit_text('К сожалению, моя база данных не содержит информацию о нужной вам книге')
    
@router.callback_query(F.data == 'HaveNotThisBook')
async def have_not_this_book(callback: CallbackQuery):
    await callback.message.edit_text('К сожалению, моя база данных не содержит информацию о нужной вам книге')

@router.message()
async def no_filter_answer(message: Message):
    await message.reply("Чтобы начать работу бота, просто отправьте любую фотографию")
