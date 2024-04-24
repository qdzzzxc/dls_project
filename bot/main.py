import asyncio
import datetime as dt
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from config import *  # noqa: F403

from handlers import image


def setup_routers(dp):
    dp.include_router(image.router)


async def on_startup(dispatcher):
    logging.info(f"Started in {dt.datetime.now() - STARTUP_TIME}")  # noqa: F405


async def on_shutdown(dispatcher):
    logging.info(
        f"Stopped, worked for {dt.datetime.now() - STARTUP_TIME}")  # noqa: F405


async def main():
    bot = Bot(
        token=TOKEN,  # noqa: F405
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())

    setup_routers(dp)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s]: %(message)s",
        datefmt="%d/%b/%y %H:%M:%S",
        filename="bot.log",
        filemode="w",
        force=True,
    )
    asyncio.run(main())
