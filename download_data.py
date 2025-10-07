import aiohttp
import asyncio
import aiofiles
from pathlib import Path
import csv

SAVE_DIR = Path("images")
SAVE_DIR.mkdir(exist_ok=True)

async def download_image(session, url, file_path):
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                async with aiofiles.open(file_path, mode='wb') as f:
                    await f.write(await resp.read())
                return True
            else:
                print(f"Failed {url} -> Status {resp.status}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

async def download_all(csv_file, max_connections=10):
    tasks = []
    sem = asyncio.Semaphore(max_connections)

    async with aiohttp.ClientSession() as session:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["ImageID"]
                url = row["OriginalURL"]
                file_path = SAVE_DIR / f"{image_id}.jpg"

                async def bound_download(url=url, file_path=file_path):
                    async with sem:
                        await download_image(session, url, file_path)

                tasks.append(bound_download())

        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(download_all("first-100k.csv"))

