import os
import configparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

embedder = SentenceTransformer("all-MiniLM-L6-v2")


DESKTOP_DIRS = [
    "/usr/share/applications",
    "/usr/local/share/applications",
    os.path.expanduser("~/.local/share/applications"),
]
def clean_exec(exec_line):
    parts = exec_line.split()
    cleaned = [p for p in parts if not p.startswith("%")]
    return cleaned[0] if cleaned else None

def scan_desktop_files():
    apps = []

    for directory in DESKTOP_DIRS:
        if not os.path.exists(directory):
            continue
        for file in os.listdir(directory):
            if not file.endswith(".desktop"):
                continue

            path = os.path.join(directory, file)
            parser = configparser.ConfigParser(interpolation=None)

            try:
                parser.read(path)
                entry = parser["Desktop Entry"]
            except Exception:
                continue

            if "Desktop Entry" not in parser:
                continue
            entry = parser["Desktop Entry"]

            if entry.get("NoDisplay") == "true":
                continue

            apps.append({
                "name" : entry.get("Name", ""),
                "generic" : entry.get("GenericName", ""),
                "comment" : entry.get("Comment", ""),
                "keywords" : entry.get("Keywords", ""),
                "categories" :entry.get("Categories", ""),
                "exec" : entry.get("Exec", ""),
                "desktop_file" : file
            })

    return apps

SEMANTIC_ALIASES ={
    "brave": ["brave", "brave browser", "browser"],
    "chrome": ["chrome", "google chrome"],
    "terminal": ["terminal", "console", "cmd", "command", "bash"],
    "files": ["files", "file manager", "explorer"],
    "spotify": ["spotify", "music"]
}
def enrich_apps(apps):
    for canonical, aliases in SEMANTIC_ALIASES.items():
        for app, meta in apps.items():
            if app == canonical or canonical in meta["aliases"]:
                meta["aliases"].extend(aliases)
    return apps

def build_aliases(app):
    alias_text = " ".join([
        app.get("name", ""),
        app.get("generic", ""),
        app.get("comment", ""),
        app.get("keywords", ""),
        app.get("categories", "")
    ]).lower()

    base_aliases = set(alias_text.split())
    expanded = set(base_aliases)

    for word in base_aliases:
        if word in SEMANTIC_ALIASES:
            expanded.update(SEMANTIC_ALIASES[word])

    return list(expanded)

def embed_app(apps):
    for app in apps:
        aliases = build_aliases(app)
        app["aliases"] = aliases
        text = " ".join(aliases)
        app["embedding"] = embedder.encode(text)
    return apps

def match_linux_app(command, apps, threshold = 0.70):
    cmd_emb = embedder.encode(command)

    best_app = None
    best_score = 0

    for app in apps:
        score = cosine_similarity([cmd_emb], [app["embedding"]])[0][0]
        if score > best_score:
            best_score = score
            best_app = app

    return best_app, best_score

def launch_linux_app(app):
    if not app or not app.get("exec"):
        return False
    cmd = app["exec"].split("%")[0].strip()

    try:
        subprocess.Popen(cmd, shell=True)
        return True
    except:
        return False
    
def resolve_and_launch(command, threshold = 0.70):
    apps = embed_app(scan_desktop_files())

    app, score = match_linux_app(command, apps, threshold)

    if not app:
        return{
            "status" : "no_match",
            "confidence" : score
        }
    success = launch_linux_app(app)
    if success:
        print("Launching app.")
    return{
        "status" : "launched" if success else "failed",
        "app" : app["name"],
        "exec" : app["exec"],
        "confidence" : score
    }