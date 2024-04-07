from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
import re
import time
import requests

def auto_auth(url):
    edge_driver_path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedgedriver.exe"
    service = Service(executable_path=edge_driver_path)
    driver = webdriver.Edge(service=service)
    #driver = webdriver.Edge()
    driver.get(str(url))

    username_field = driver.find_element(By.NAME, "username")
    password_field = driver.find_element(By.NAME, "password")
    submit_button = driver.find_element(By.NAME, "connect")

    username_field.send_keys("yiny21")
    password_field.send_keys("948SHISIa#")

    submit_button.click()
    time.sleep(5)
    driver.quit()
    return None


def check_internet():
    url = "http://www.baidu.com"
    timeout = 10
    try:
        response = requests.get(url, timeout=timeout)
        url_pattern = r'location\.href="(http[s]?://[^"]+)"'
        match = re.search(url_pattern, response.text)

        if match:
            url = match.group(1)
        else:
            url=None
        return True, url
    except requests.ConnectionError:
        return False, None


if __name__ == "__main__":

    while True:
        online, url = check_internet()
        if not online:
            print("Offline")
        else:
            if url is not None:
                print("Online, but need Authentication.")
                try:
                    auto_auth(url)
                except:
                    None
                print("Finished Authentication!")
            else:
                None
                print("Online!")
        time.sleep(600)
