#!/usr/bin/env python
import requests
from bs4 import BeautifulSoup
import os
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
db = client.clinic
main="https://www.cas.ca"
collection_name="table"
req_link=[]

if __name__=='__main__':
    
            url='https://www.cas.ca/English/2012-AM-Abstracts'
          
            driver = webdriver.Chrome()
            
            for i in range(1,8):
                 driver.get(url)
                 if i!=1:
                        linklis=driver.find_element_by_xpath('//*[@id="ctl00_PageContentID_divText"]/table[2]/tbody/tr/td[1]/p[3]/strong'+'['+ str(i) + ']' +'/a')
                 else:
                        linklis=driver.find_element_by_xpath('//*[@id="ctl00_PageContentID_divText"]/table[2]/tbody/tr/td[1]/p[3]/strong[1]/strong/a')
                 print (linklis.get_attribute('href'))
                 
                    
                 
                 driver.get(linklis.get_attribute('href'))

                 for j in range (2,6):
                            alllinks=driver.find_elements_by_xpath('//*[@id="ctl00_PageContentID_divText"]/table[2]/tbody/tr/td[1]/p'+'['+ str(j) + ']' +'/a')
                            for t in alllinks:
                               print (t.get_attribute('href'))
                               st=t.get_attribute('href')
                               os.system('wget %s'%(st))
                 alllinks3=driver.find_elements_by_xpath('//*[@id="ctl00_PageContentID_divText"]/table[2]/tbody/tr/td[1]/a')
                 for t in alllinks3:
                            print (t.get_attribute('href'))
                            st=t.get_attribute('href')
                            os.system('wget %s'%(st))
               
                                
                
                
