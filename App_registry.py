import re
from word2number import w2n

APPS = {
    "brave": ["brave", "brave browser"],
    "chrome": ["chrome", "google chrome"],
    "terminal": ["terminal", "console", "cmd"],
    "files": ["files", "file manager", "explorer"],
    "spotify": ["spotify","music"]
}
def get_app_name(command):
    if isinstance(command, list):
        command = " ".join(command)
    if not isinstance(command, str):
        return None
    command = command.lower()
    command = re.sub(r"[^\w\s]"," ", command)
    tokens = set(command.split())

    for app, keywords in APPS.items():
        for keyword in keywords:
            if keyword in tokens:
                return app
    return None

def extract_number(text):
    text = text.lower()

    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    
    words = text.split()
    for i in range(len(words)):
        try:
            return w2n.word_to_num(words[i])
        except:
            pass

    return None

def detect_unit(text):
    t = text.lower()
    if "percent" in t or "%" in t:
        return "percent"
    return None

