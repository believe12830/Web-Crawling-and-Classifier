#!/usr/bin/env python
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient
import pprint
from time import sleep
import re

client = MongoClient()
db = client.AAneu_congresses2017


all_links=[]
def mainlinkextractor(response):
    driver.get(url)
    sleep(5)
    
    alllinks=driver.find_elements_by_xpath('//*[@id="body"]/div/div[2]/div[2]/div/div/div/div/div/div[2]/ul/li/a')
    for link in alllinks:
        all_links.append(link.get_attribute('href'))



if __name__ == '__main__':
    url = 'http://www.abstractsonline.com/pp8/#!/4031'
    response = requests.get(url, stream=True)

    driver = webdriver.Chrome()
    
    mainlinkextractor(url)

    for i in all_links:
        print i
        print '\n'

