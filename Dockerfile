FROM python:3.12-slim

RUN apt-get update -y
RUN apt-get install git -y

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install "fastapi[standard]"
RUN pip install requests
RUN git clone https://github.com/exiftool/exiftool

COPY ./download_data.py ./server.py ./tags.db ./

RUN mkdir -p templates 
COPY ./templates/index.html ./templates

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]
