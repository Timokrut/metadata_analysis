.PHONY: build bash dl discover extract train report interp thr errors export all api-build api-up api-down api-logs api-test clean

# Общие цели для всего проекта
build:
    docker compose build

bash:
    docker compose run --rm app bash

dl:
    docker compose run --rm app ffpp-download --kaggle-id "xdxd003/ff-c23" --dst data/ffpp

discover:
    docker compose run --rm app ffpp-discover --config configs/train.yaml --out data/ffpp_manifest.csv

extract:
    docker compose run --rm app ffpp-extract-faces --config configs/train.yaml --manifest data/ffpp_manifest.csv

train:
    docker compose run --rm app train-both --config configs/train.yaml

report:
    docker compose run --rm app build-report --config configs/train.yaml

interp:
    docker compose run --rm app interpret-occlusion --config configs/train.yaml

thr:
    docker compose run --rm app threshold-curves --config configs/train.yaml

errors:
    docker compose run --rm app error-grids --config configs/train.yaml

export:
    docker compose run --rm app export-models --config configs/train.yaml

all: dl discover extract train report export

# API цели
api-build:
    docker compose build api

api-up:
    docker compose up -d api

api-down:
    docker compose down --remove-orphans

api-logs:
    docker compose logs -f api

api-test:
    curl -X POST -F "file=@test.jpg" http://localhost:8888/analyze || true

# Тесты и утилиты
test:
    docker compose run --rm app pytest -q

smoke:
    docker compose run --rm app bash -lc "python scripts/generate_tiny_dataset.py data/tiny && python -m dfdetect.cli train-both --data-dir data/tiny --out runs/smoke --epochs 1 --batch-size 4 --num-workers 0"

clean:
    rm -rf runs exports data/tiny

# Дополнительно: переменная PYTHONPATH
export PYTHONPATH := src
