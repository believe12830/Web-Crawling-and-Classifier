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
import urllib2

client = MongoClient()
db = client.clinic
main="https://www.escmid.org"
collection_name="table"
req_link=[]

if __name__=='__main__':
    
            url='https://www.escmid.org/escmid_publications/escmid_elibrary/?tx_solr%5Bfilter%5D%5B0%5D=main_filter_eccmid%253Atrue&tx_solr%5Bfilter%5D%5B1%5D=pub_date%253A201501010000-201512312359&tx_solr%5Bpage%5D'
            response = requests.get(url, stream=True)
            soup = BeautifulSoup(response.content,'html.parser')
            div_tags=soup.find_all('div' ,{'class':'result-files'})
            print len(div_tags)
            c=1
            
            for i in div_tags:
                st=i.find('a').get('href')
                temp='wget %s --output-document=a' %(st) + str(c) +'.pdf'
                c=c+1
                os.system(temp)
                # os.system('wget %s'%(st))


            
            
            # os.system(pdf_links[0])
            raw_input()

            for k in range(1,351):
                url=url+'='+ str(k)
                response = requests.get(url, stream=True)
                soup2= BeautifulSoup(response.content,'html.parser')
                div_tags2=soup2.find_all('div' ,{'class':'result-files'})
                print k
                for i in div_tags2:
                    st=i.find('a').get('href')
                    temp='wget %s --output-document=a_2016_' %(st) + str(c) +'.pdf'
                    c=c+1
                    os.system(temp)
            # os.system('wget %s'%(st))



            
            # for i in range(1,380):
                
            #     link=url + str(i)
            #     driver.get(link)
            #     sleep(2)
            #     a_tags=driver.find_elements_by_xpath('//*[@id="search-container"]/ol/li/article/ul/li[2]/a')
            #     print len(a_tags)

            #     for t in a_tags:


            #         print (t.get_attribute('href'))
            #         temp='wget %s --output-document=a' %(t.get_attribute('href')) + str(c) +'.pdf'
            #         c=c+1
            #         os.system(temp)

