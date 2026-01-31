#!/usr/bin/env python3
"""
WhatsApp Assistant - Complete automation for WhatsApp Web (Chrome Version)
Features:
- Send messages to contacts
- Check new messages and notify
- Download files from WhatsApp
- Search and open files from specific contacts
"""

import os
import time
import json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
from datetime import datetime
import subprocess


class WhatsAppAssistant:
    def __init__(self, profile_dir=None, download_dir=None):
        """Initialize WhatsApp Assistant"""
        self.home_dir = str(Path.home())
        self.profile_dir = profile_dir or os.path.join(self.home_dir, ".whatsapp_assistant")
        self.download_dir = download_dir or os.path.join(self.home_dir, "Downloads", "WhatsApp")
        
        # Create directories if they don't exist
        os.makedirs(self.profile_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)
        
        self.driver = None
        self.wait = None
        self.last_messages = {}
        
    def find_chrome_binary(self):
        """Find Google Chrome binary location"""
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
        
        # Try to find using which command
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
            
        raise Exception("Google Chrome not found. Please install Google Chrome first.")
    
    def setup_driver(self):
        """Setup Chrome browser with persistent session"""
        print("Setting up Google Chrome...")
        
        chrome_binary = self.find_chrome_binary()
        print(f"Using Chrome at: {chrome_binary}")
        
        options = Options()
        options.binary_location = chrome_binary
        
        # Use persistent user data directory for session persistence
        options.add_argument(f"--user-data-dir={self.profile_dir}")
        options.add_argument("--profile-directory=Default")
        
        # Download preferences
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False,
            "profile.default_content_setting_values.notifications": 2,  # Disable notifications
            "profile.default_content_settings.popups": 0,
        }
        options.add_experimental_option("prefs", prefs)
        
        # Additional options for stability
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        
        # Exclude automation flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            # Initialize driver
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, 30)
            
            print("Browser setup complete!")
            return True
        except Exception as e:
            print(f"Error setting up browser: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Google Chrome is installed")
            print("2. Make sure ChromeDriver is installed: run ./install_chrome_143.sh")
            print("3. Verify versions match: google-chrome --version and chromedriver --version")
            return False
    
    def start(self):
        """Start WhatsApp Web and handle QR code login"""
        if not self.setup_driver():
            return False
        
        print("Opening WhatsApp Web...")
        self.driver.get("https://web.whatsapp.com")
        
        # Check if already logged in
        try:
            # Wait for either QR code or chat list
            for i in range(60):
                try:
                    # Check for QR code
                    qr_code = self.driver.find_element(By.CSS_SELECTOR, "canvas[aria-label='Scan this QR code to link a device!']")
                    print(f"\n{'='*50}")
                    print("QR CODE DETECTED - Please scan with your phone")
                    print(f"{'='*50}\n")
                    time.sleep(2)
                except NoSuchElementException:
                    # Check if logged in (search box or chat list present)
                    try:
                        search_box = self.driver.find_element(By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='3']")
                        print("\n✓ Already logged in!")
                        time.sleep(3)
                        return True
                    except NoSuchElementException:
                        pass
                
                time.sleep(1)
            
            # Final check after timeout
            try:
                search_box = self.driver.find_element(By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='3']")
                print("\n✓ Login successful!")
                time.sleep(3)
                return True
            except NoSuchElementException:
                print("Login timeout. Please try again.")
                return False
                
        except Exception as e:
            print(f"Error during login: {e}")
            return False
    
    def find_contact(self, contact_name):
        """Search and click on a contact"""
        try:
            # Click on search box
            search_box = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='3']"))
            )
            search_box.click()
            time.sleep(0.5)
            
            # Clear and type contact name
            search_box.clear()
            search_box.send_keys(contact_name)
            time.sleep(1.5)
            
            # Click on the first result
            contact = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='listitem']"))
            )
            contact.click()
            time.sleep(1)
            
            return True
        except Exception as e:
            print(f"Error finding contact '{contact_name}': {e}")
            return False
    
    def send_message(self, contact_name, message):
        """Send a message to a contact"""
        try:
            print(f"Sending message to {contact_name}...")
            
            if not self.find_contact(contact_name):
                return False
            
            # Find message input box
            message_box = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='10']"))
            )
            
            # Type and send message
            message_box.click()
            time.sleep(0.3)
            message_box.send_keys(message)
            time.sleep(0.5)
            message_box.send_keys(Keys.ENTER)
            
            print(f"✓ Message sent to {contact_name}!")
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def check_new_messages(self):
        """Check for new unread messages and return them"""
        try:
            new_messages = []
            
            # Find all chats with unread badge
            unread_chats = self.driver.find_elements(By.CSS_SELECTOR, "span[data-icon='unread-count']")
            
            if not unread_chats:
                return new_messages
            
            print(f"\n{'='*50}")
            print(f"Found {len(unread_chats)} chat(s) with unread messages")
            print(f"{'='*50}\n")
            
            # Go through each unread chat
            for i in range(min(len(unread_chats), 10)):  # Limit to 10 chats
                try:
                    # Re-find unread chats as DOM changes
                    unread_chats = self.driver.find_elements(By.CSS_SELECTOR, "span[data-icon='unread-count']")
                    if i >= len(unread_chats):
                        break
                    
                    # Find the parent chat element
                    chat_element = unread_chats[i].find_element(By.XPATH, "./ancestor::div[@role='listitem']")
                    
                    # Get contact name
                    try:
                        contact_name = chat_element.find_element(By.CSS_SELECTOR, "span[dir='auto'][title]").get_attribute("title")
                    except:
                        contact_name = "Unknown"
                    
                    # Get unread count
                    try:
                        count = unread_chats[i].text
                    except:
                        count = "1"
                    
                    # Click on chat
                    chat_element.click()
                    time.sleep(1.5)
                    
                    # Get recent messages
                    messages = self.driver.find_elements(By.CSS_SELECTOR, "div.message-in span.selectable-text")
                    
                    recent_msgs = []
                    for msg in messages[-int(count) if count.isdigit() else -5:]:
                        text = msg.text.strip()
                        if text:
                            recent_msgs.append(text)
                    
                    if recent_msgs:
                        new_messages.append({
                            'contact': contact_name,
                            'count': count,
                            'messages': recent_msgs
                        })
                        
                        print(f"📱 From: {contact_name} ({count} new)")
                        for msg in recent_msgs:
                            print(f"   💬 {msg}")
                        print()
                    
                except Exception as e:
                    print(f"Error reading chat: {e}")
                    continue
            
            return new_messages
            
        except Exception as e:
            print(f"Error checking messages: {e}")
            return []
    
    def download_files_from_chat(self, contact_name=None, file_types=None):
        """Download files from current chat or specific contact"""
        try:
            if contact_name:
                if not self.find_contact(contact_name):
                    return []
            
            downloaded_files = []
            
            # Find all media/document elements
            media_elements = self.driver.find_elements(By.CSS_SELECTOR, "div[data-icon='audio'], div[data-icon='document'], div[data-icon='image'], div[data-icon='video']")
            
            print(f"Found {len(media_elements)} file(s) in chat")
            
            for idx, element in enumerate(media_elements[:20]):  # Limit to 20 files
                try:
                    # Click on the file to open it
                    element.click()
                    time.sleep(1)
                    
                    # Try to find and click download button
                    try:
                        download_btn = self.wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-icon='download']"))
                        )
                        download_btn.click()
                        print(f"✓ Downloading file {idx + 1}...")
                        downloaded_files.append(f"file_{idx + 1}")
                        time.sleep(2)
                    except:
                        pass
                    
                    # Close the media viewer
                    try:
                        close_btn = self.driver.find_element(By.CSS_SELECTOR, "div[data-icon='x-viewer']")
                        close_btn.click()
                        time.sleep(0.5)
                    except:
                        self.driver.send_keys(Keys.ESCAPE)
                        time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error downloading file {idx + 1}: {e}")
                    continue
            
            print(f"\n✓ Downloaded {len(downloaded_files)} file(s) to {self.download_dir}")
            return downloaded_files
            
        except Exception as e:
            print(f"Error downloading files: {e}")
            return []
    
    def search_file_in_chat(self, contact_name, search_term):
        """Search for a specific file in a contact's chat"""
        try:
            if not self.find_contact(contact_name):
                return None
            
            # Click on menu (three dots)
            menu_button = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-icon='menu']"))
            )
            menu_button.click()
            time.sleep(0.5)
            
            # Click on "Search"
            try:
                search_option = self.wait.until(
                    EC.presence_of_element_located((By.XPATH, "//div[@role='button']//span[contains(text(), 'Search')]"))
                )
                search_option.click()
                time.sleep(0.5)
            except:
                print("Search option not found in menu")
                return None
            
            # Type in search box
            search_box = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='6']"))
            )
            search_box.send_keys(search_term)
            time.sleep(1.5)
            
            # Find matching messages/files
            results = self.driver.find_elements(By.CSS_SELECTOR, "div[data-icon='audio'], div[data-icon='document'], div[data-icon='image'], div[data-icon='video']")
            
            if results:
                print(f"✓ Found {len(results)} file(s) matching '{search_term}'")
                # Click on first result
                results[0].click()
                time.sleep(1)
                
                # Try to download
                try:
                    download_btn = self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-icon='download']"))
                    )
                    download_btn.click()
                    print(f"✓ Downloading file...")
                    time.sleep(2)
                    return True
                except:
                    print("File opened but couldn't download automatically")
                    return True
            else:
                print(f"No files found matching '{search_term}'")
                return False
            
        except Exception as e:
            print(f"Error searching file: {e}")
            return None
    
    def parse_command(self, command):
        """Parse natural language command"""
        command = command.lower().strip()
        
        # Message command: "message ani that i'll be late"
        if command.startswith("message "):
            match = re.match(r"message\s+(\w+)\s+(?:that\s+)?(.+)", command)
            if match:
                contact = match.group(1).capitalize()
                message = match.group(2)
                return ('send', contact, message)
        
        # Check messages
        if "check messages" in command or "new messages" in command:
            return ('check', None, None)
        
        # Download files: "download files from ani"
        if "download" in command and "from" in command:
            match = re.search(r"from\s+(\w+)", command)
            if match:
                contact = match.group(1).capitalize()
                return ('download', contact, None)
        
        # Search file: "search report in ani chat"
        if "search" in command and ("in" in command or "from" in command):
            match = re.search(r"search\s+(.+?)\s+(?:in|from)\s+(\w+)", command)
            if match:
                search_term = match.group(1).strip()
                contact = match.group(2).capitalize()
                return ('search', contact, search_term)
        
        return ('unknown', None, None)
    
    def execute_command(self, command):
        """Execute a parsed command"""
        action, param1, param2 = self.parse_command(command)
        
        if action == 'send':
            return self.send_message(param1, param2)
        elif action == 'check':
            return self.check_new_messages()
        elif action == 'download':
            return self.download_files_from_chat(param1)
        elif action == 'search':
            return self.search_file_in_chat(param1, param2)
        else:
            print(f"Unknown command. Try:")
            print("  - 'message ani that I'll be late'")
            print("  - 'check messages'")
            print("  - 'download files from ani'")
            print("  - 'search report in ani chat'")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            print("\nClosing browser...")
            self.driver.quit()


def main():
    """Main function to run the assistant"""
    print("="*60)
    print("WhatsApp Assistant - Chrome Version")
    print("="*60)
    
    assistant = WhatsAppAssistant()
    
    if not assistant.start():
        print("Failed to start WhatsApp Web. Exiting.")
        return
    
    print("\n" + "="*60)
    print("Assistant is ready! You can now use commands.")
    print("="*60)
    print("\nExample commands:")
    print("  • message ani that I'll be late")
    print("  • check messages")
    print("  • download files from john")
    print("  • search invoice in ani chat")
    print("  • quit")
    print()
    
    try:
        while True:
            command = input("Command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            if not command:
                continue
            
            assistant.execute_command(command)
            print()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        assistant.close()
        print("Goodbye!")


if __name__ == "__main__":
    main()