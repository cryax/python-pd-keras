# -*- coding: utf-8 -*-
#!/bin/sh
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
driver = webdriver.Chrome()
driver.get("http://www.google.com")
#assert "Python" in driver.title
search = driver.find_element_by_name('q')
search.send_keys("dhbk location")
search.send_keys(Keys.RETURN)

while(1):
    aa = driver.find_element_by_class_name("_XWk")
    print aa.text
#driver.close()
