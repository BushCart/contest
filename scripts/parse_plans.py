import os
import requests
import subprocess

URLS = {
    "ai":      "https://api.itmo.su/constructor-ep/api/v1/static/programs/10033/plan/abit/pdf",
    "ai_prod": "https://api.itmo.su/constructor-ep/api/v1/static/programs/10130/plan/abit/pdf",
}


RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_plan_file(name: str, url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    ext = os.path.splitext(url)[1] or ".pdf"
    local_path = os.path.join(RAW_DIR, f"{name}_plan{ext}")
    print(f"[+] Скачиваем план {name}: {url} -> {local_path}")
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return local_path

def main():
    downloaded = []
    for name, url in URLS.items():
        downloaded.append(fetch_plan_file(name, url))

    for path in downloaded:
        subprocess.run([
            "python", "-m", "scripts.parse_docs",
            "--input", path,
            "--output-dir", "parsed/plans"
        ], check=True)

if __name__ == "__main__":
    main()