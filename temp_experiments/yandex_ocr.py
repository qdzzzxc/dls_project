import os
import time

from pynput.keyboard import Controller, Key
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


def get_yandex_ocr(file_name):
    driver = webdriver.Chrome()

    link = "https://translate.yandex.ru/ocr"

    # cant run headless cuz of input
    driver.get(link)

    original_window = driver.current_window_handle

    driver.find_element(By.XPATH, "//a[@data-action='pickFile']").click()

    time.sleep(1)

    keyboard = Controller()

    keyboard.type(os.path.abspath(file_name))

    time.sleep(2)

    keyboard.press(Key.enter)
    keyboard.release(Key.enter)

    WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable(
            (By.XPATH, "//button[@class='button button_view_ghost state-show-wide']")
        )
    ).click()

    wait = WebDriverWait(driver, timeout=5)

    wait.until(EC.number_of_windows_to_be(2))

    for window_handle in driver.window_handles:
        if window_handle != original_window:
            driver.switch_to.window(window_handle)
            break

    result = driver.find_element(By.XPATH, "//div[@id='fakeArea']").text

    for handle in driver.window_handles:
        driver.switch_to.window(handle)
        driver.close()

    return result
