# Как использовать 
---

1. Клонировать репозиторий
```bash
git clone https://github.com/Timokrut/metadata_analysis.git
cd metadata_analysis
```

2. Скачать [docker](https://www.docker.com/products/docker-desktop/) (если не установлен)

3. Сбилдить и поднять образ
```bash
docker build -t metadata .
docker run -p 8888:8888 metadata:latest
```

4. Для того чтобы отправить запрос на сервер через *curl* 
```bash
curl -X POST -F "file=@/home/user/dataset/7.JPG" http://localhost:8888/metadata/
```

5. Чтобы получить \_metadata
```bash
curl http://localhost:8888/get-metadata
```

