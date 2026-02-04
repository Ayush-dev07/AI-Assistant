import os
import time 
import json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
import re
from datetime import datetime
import subprocess

class WhatsappAssistant:
    def __init__(self, profile_dir=None, download_dir=None):
        self.home_dir = str(Path.home())
        self.profile_dir = profile_dir or os.path.join(self.home_dir, "whatsapp_assistant")
        self.download_dir = download_dir or os.path.join(self.home_dir, "Downloads", "WhatsApp")

        os.makedirs(self.profile_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

        self.driver = None
        self.wait = None
        self.last_message = {}

    def find_chrome_binary(self):
        possible_paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/local/bin/google-chrome",
            "/opt/google/chrome/google-chrome",
            "/snap/bin/chromium"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
            
        try:
            result = subprocess.run(['which', 'google-chrome'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        try:
            result = subprocess.run(['which', 'google-chrome-stable'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        raise Exception("Google chrome not found. Please install it to use the assistant.")

    def setup_driver(self):
        print("Setting up Google Chrome")

        chrome_binary = self.find_chrome_binary()
        print(f"Using Chrome browser at: {chrome_binary}")  
        options = Options()
        options.binary_location = chrome_binary

        options.add_argument(f"--user-data-dir={self.profile_dir}")
        options.add_argument("--profile-directory=Default")

        prefs = {
            "download.default_directory" : self.download_dir,
            "download.prompt_for_download" : False,
            "download.directory_upgrade" : True,
            "safebrowsing.enabled" : False,
            "profile.default_content_setting_values.notifications" : 2,
            "profile.default_content_settings.popups" : 0,
        }
        options.add_experimental_option("prefs", prefs)

        options.add_argument("--disable-blink-features=AutomationControlled") 
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        options.add_argument("--no-sandbox")

        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        try:
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, 30)

            print("Browser Ready!")
            return True
        except Exception as e:
            print(f"Error setting up browser: {e}")
            return False
        
    def start(self):
        if not self.setup_driver():
            return False
        
        print("Opening WhatsApp Web...")
        self.driver.get("https://web.whatsapp.com")

        try:
            for i in range(60):
                try:
                    qr_code = self.driver.find_element(By.CSS_SELECTOR, "canvas[aria-label='Scan this code']")
                    print(f"\n{'='*50}")
                    print("QR Code detected!")
                    print(f"{'='*50}\n")
                    time.sleep(2)
                except NoSuchElementException:
                    try:
                        search_box = self.driver.find_element(By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='3']")
                        print("Already Logged in!")
                        time.sleep(3)
                        return True
                    except NoSuchElementException:
                        pass
                time.sleep(1)

            try:
                search_box = self.driver.find_element(By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='3']")
                print("\n Login successful!")
                time.sleep(3)
                return True
            except NoSuchElementException:
                print("Login timeout. Please try again.")
                return False
        except Exception as e:
            print(f"Error during login: {e}")
            return False
        
def main():
    print("="*60)
    print("WhatsApp assistant is ready.")
    print("="*60)
    
    assistant = WhatsappAssistant()

    if not assistant.start():
        print("failed to start Whatsapp Assistant.")
        return
    
if __name__ == "__main__":
    main()
