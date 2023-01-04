# Module imports

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import os, sys
import time

# Local imports
from local_paths import *


# Creates options for chromedriver

def undetectable_options(chrome_profile_num = 0
                        ,incognito = False
                        ):
    
    options = webdriver.ChromeOptions()
    
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"]) # > no me funciona en undetectable chromedriver
    options.add_experimental_option('useAutomationExtension', False) # > no me funciona en undetectable chromedriver
    options.add_argument('--disable-notifications')
    options.add_argument("--mute-audio")
    options.add_argument("--disable-popup-blocking");

    if incognito:
        options.add_argument("--incognito") ## chrome incognito mode
    
    
    # add vpn extension
    #options.add_extension('extension_vpn.crx')


    # Options to avoid bot detection
    options.add_argument('--disable-blink-features=AutomationControlled') # el undetectable chromedriver recomienda sacarlo
    options.add_argument("--window-size=1282,814")
    options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36")
    
    # profile
    if chrome_profile_num > 0:
        options.add_argument(f"user-data-dir=/Users/matias/Library/Application Support/Google/Chrome/Profile {chrome_profile_num}") #Path to your chrome profile

    return options


# Initializes the driver

def chromedriver(options = None):

    # change the mode of path to the numeric mode
    os.chmod(CHROMEDRIVER_PATH, 755)
    
    try:
        driver.quit()
    except:
        pass
    
    driver = webdriver.Chrome(CHROMEDRIVER_PATH, options=options)

    actions = ActionChains(driver)
    
    time.sleep(10)
    
    # detect all open tabs, close VeePN tab
    
    while True:
        
        try:
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(1)
            driver.close()
        except:
            break

    return driver, actions

    
def open_link(link_str
             ,mode = 'equal'
             ,print_logs = False
             ):
    
    # mode = equal: function will iterate until it opens that exact same link
    # mode = in: opens that link and is ok with receving a link that contains extra text to the right
    
    if mode not in ('equal','in'): sys.exit('Bad mode entered for open_link')
    
    open_link_try = 0
    
    link_str = link_str.rstrip('/')
    
    time.sleep(2)
    
    while True:
        
        try:
            
            open_link_try += 1
            
            current_url_raw = driver.current_url
            
            current_url_strict = current_url_raw.rstrip('/')
            
            if current_url_strict.find('?') > -1: current_url = current_url_strict[:current_url_strict.find('?')]
            else: current_url = current_url_strict
            
            if print_logs: print(f'current url: {current_url} // instructed_url = {link_str} // got_url = {link_str == current_url}')
        
            if (mode == 'equal') & (link_str == current_url): return None
        
            elif (mode == 'in') & (link_str in current_url): return None
        
                
            elif open_link_try > 5:
                
                sys.exit(f'Error: open_link() has failed {open_link_try} times. System exit')

            else:

                driver.get(link_str)

                time.sleep(5)
                
        except:

            pass
        