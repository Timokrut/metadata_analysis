import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from importlib import import_module
import librosa
import io
import soundfile as sf
from torch import Tensor
import torch.nn.functional as F
import datetime
import csv

device = None
model_weights = None
model_class = None
model_config = None


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.audio_files = []
        self.labels = []

        real_path = os.path.join(root_dir, 'real')
        fake_path = os.path.join(root_dir, 'fake')

        if os.path.exists(real_path):
            for file in os.listdir(real_path):
                if file.endswith(('.wav', '.WAW', '.mp3', '.flac')):
                    self.audio_files.append(os.path.join(real_path, file))
                    self.labels.append(0)

        if os.path.exists(fake_path):
            for file in os.listdir(fake_path):
                if file.endswith(('.wav', '.WAW', '.mp3', '.flac')):
                    self.audio_files.append(os.path.join(fake_path, file))
                    self.labels.append(1)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)

            flac_buffer = io.BytesIO()
            sf.write(flac_buffer, audio_data, sample_rate, format='FLAC')
            flac_buffer.seek(0)

            X, sample_rate = sf.read(flac_buffer)

            X_pad = pad(X, 64600)
            x_inp = Tensor(X_pad).ravel()

            if x_inp.size() == torch.Size([129200]):
                batch_x = x_inp.reshape((2, 64600))
            else:
                batch_x = x_inp.reshape((1, 64600))

            return batch_x, label, audio_path

        except Exception as e:
            print(f"Ошибка загрузки файла {audio_path}: {str(e)}")
            # Возвращаем нулевой тензор в случае ошибки
            batch_x = torch.zeros(1, 64600)
            return batch_x, label, audio_path


def pad(x, max_len=64600):
    x_len = len(x)
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(np.ceil(max_len / x_len))
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def load_model_weights(checkpoint_path, device):
    print(f"Загружаем веса из: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Ключи в checkpoint: {list(checkpoint.keys())}")

    # Пробуем разные возможные ключи
    possible_keys = ['model', 'state_dict', 'model_state_dict', 'weights']

    for key in possible_keys:
        if key in checkpoint:
            print(f"Найден ключ: {key}")
            return checkpoint[key]

    # Если нет стандартных ключей, возможно checkpoint уже содержит state_dict
    # Проверяем, есть ли в checkpoint параметры модели
    has_parameters = any('weight' in k or 'bias' in k for k in checkpoint.keys())
    if has_parameters:
        print("Checkpoint содержит параметры модели напрямую")
        return checkpoint

    # Если ничего не нашли, покажем структуру checkpoint для отладки
    print("Структура checkpoint:")
    for key, value in checkpoint.items():
        print(f"  {key}: {type(value)}")

    raise KeyError(f"Не найден ключ с весами модели в checkpoint. Доступные ключи: {list(checkpoint.keys())}")


def init_model_state(config, device, model_weights_path=None):
    global model_weights, model_class, model_config

    model_config = config["model_config"]

    # Определяем путь к весам модели
    weights_path = model_weights_path or config.get('model_path')
    if not weights_path:
        raise ValueError(
            "Не указан путь к весам модели. Используйте --eval_model_weights или укажите model_path в конфиге")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл с весами не найден: {weights_path}")

    # Создаем модель
    module = import_module(f"models.{model_config['architecture']}")
    model_class = getattr(module, "Model")
    model = model_class(model_config).to(device)

    # Загружаем веса модели
    model_weights = load_model_weights(weights_path, device)

    # Загружаем веса в модель
    try:
        model.load_state_dict(model_weights)
        print("✅ Веса успешно загружены в модель")
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке весов: {e}")
        print("Пробуем загрузить с strict=False...")
        model.load_state_dict(model_weights, strict=False)

    # Сохраняем веса для будущего использования
    model_weights = model.state_dict().copy()

    # Проверяем, что веса загружены
    print(f"✅ Модель инициализирована. Сохранено параметров: {len(model_weights)}")
    return True


def create_fresh_model(device):
    global model_weights, model_class, model_config

    if model_class is None or model_weights is None:
        raise ValueError("Модель не инициализирована. Вызовите init_model_state()")

    model = model_class(model_config).to(device)
    model.load_state_dict(model_weights)
    model.eval()

    return model


def get_model_predictions(data_loader, device):
    predictions = []
    labels = []
    file_paths = []

    for batch_idx, (batch_x, label, audio_path) in enumerate(data_loader):
        print(f"\n--- Обработка батча {batch_idx} ---")
        print(f"Файл: {audio_path}")

        # Создаем свежую модель
        model = create_fresh_model(device)

        # Подготавливаем входные данные
        if batch_x.dim() > 2:
            batch_x = batch_x.squeeze(0)

        print(f"Размер входных данных: {batch_x.shape}")
        print(f"Статистика - mean: {batch_x.mean():.6f}, std: {batch_x.std():.6f}")

        model.eval()
        with torch.no_grad():
            batch_x = batch_x.to(device)
            outputs = model(batch_x)

            # Обрабатываем разные форматы выходов модели
            if isinstance(outputs, tuple):
                batch_coef, batch_out = outputs
            else:
                batch_out = outputs
                batch_coef = None

            print(f"Выход модели: {batch_out.cpu().numpy()}")

            probabilities = F.softmax(batch_out, dim=-1)
            print(f"Вероятности: {probabilities.cpu().numpy()}")

            # Получаем score
            if batch_out.shape[1] == 2:  # Два класса
                batch_score = probabilities[0, 1].item()
            else:
                batch_score = probabilities[0].item()

            print(f"Score: {batch_score:.6f}")

        # Обрабатываем коэффициенты если они есть
        if batch_coef is not None:
            coef = batch_coef.to('cpu')
            print(f"Коэффициенты shape: {coef.shape}")

            T_max = str(coef[:, 0:32].mean().item())
            T_avg = str(coef[:, 32:64].mean().item())
            S_max = str(coef[:, 64:96].mean().item())
            S_avg = str(coef[:, 96:128].mean().item())
        else:
            T_max = T_avg = S_max = S_avg = "0.0"

        prediction = [batch_score, T_max, T_avg, S_max, S_avg]
        predictions.append(prediction)

        # Обрабатываем label
        if hasattr(label, 'numpy'):
            label_val = label.numpy()[0] if label.numel() > 1 else label.item()
        else:
            label_val = label

        labels.append(label_val)
        file_paths.append(audio_path[0] if isinstance(audio_path, list) else audio_path)

        # Очистка
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return predictions, labels, file_paths


def save_results_to_csv(predictions, labels, file_paths, csv_filename=None):
    if csv_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"model_predictions_{timestamp}.csv"

    # Определяем заголовки в зависимости от наличия коэффициентов
    has_coefficients = len(predictions[0]) > 1

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        if has_coefficients:
            fieldnames = ['filename', 'true_label', 'predicted_score', 'predicted_class',
                          'T_max', 'T_avg', 'S_max', 'S_avg', 'correct']
        else:
            fieldnames = ['filename', 'true_label', 'predicted_score', 'predicted_class', 'correct']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        correct_predictions = 0
        total_predictions = len(predictions)

        for i, (pred, true_label, file_path) in enumerate(zip(predictions, labels, file_paths)):
            score = pred[0]
            pred_class = "fake" if score > 0.5 else "real"
            true_class = "fake" if true_label == 1 else "real"
            correct = pred_class == true_class

            if correct:
                correct_predictions += 1

            row_data = {
                'filename': file_path,
                'true_label': true_class,
                'predicted_score': f"{score:.6f}",
                'predicted_class': pred_class,
                'correct': 'YES' if correct else 'NO'
            }

            # Добавляем коэффициенты если они есть
            if has_coefficients:
                row_data.update({
                    'T_max': pred[1],
                    'T_avg': pred[2],
                    'S_max': pred[3],
                    'S_avg': pred[4]
                })

            writer.writerow(row_data)

        # Добавляем итоговую статистику
        accuracy = correct_predictions / total_predictions * 100
        writer.writerow({})
        writer.writerow({'filename': 'SUMMARY', 'true_label': f'Accuracy: {accuracy:.2f}%',
                         'predicted_score': f'Correct: {correct_predictions}/{total_predictions}'})

    print(f"✅ Результаты сохранены в: {csv_filename}")
    print(f"📊 Точность: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

    return csv_filename, accuracy

def main(args: argparse.Namespace):
    global device

    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    # Настройки
    testing_dir = args.test_dir
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    # Инициализируем модель
    try:
        init_model_state(config, device, args.eval_model_weights)
    except Exception as e:
        print(f"❌ Ошибка инициализации модели: {e}")
        return

    # Создаем dataset и dataloader
    dataset = AudioDataset(testing_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Получаем предсказания
    try:
        predictions, labels, file_paths = get_model_predictions(dataloader, device)
    except Exception as e:
        print(f"❌ Ошибка при получении предсказаний: {e}")
        return

    # Сохраняем результаты
    results = {
        'predictions': predictions,
        'true_labels': labels,
        'file_paths': file_paths
    }

    np.save('model_predictions.npy', results)

    csv_filename, accuracy = save_results_to_csv(predictions, labels, file_paths)

    # Выводим статистику
    print(f"\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
    print(f"Обработано файлов: {len(predictions)}")
    print(f"Real файлов: {np.sum(np.array(labels) == 0)}")
    print(f"Fake файлов: {np.sum(np.array(labels) == 1)}")
    print(f"📊 Точность: {accuracy:.2f}%")

    print("\nДетальные результаты:")
    for i, (pred, true, path) in enumerate(zip(predictions, labels, file_paths)):
        pred_class = "fake" if pred[0] > 0.5 else "real"
        true_class = "fake" if true == 1 else "real"
        correct = "✓" if pred_class == true_class else "✗"
        print(
            f"{i + 1:2d}. {correct} Score: {pred[0]:.4f} | True: {true_class:4s} | Pred: {pred_class:4s} | {str(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=False,
                        default="./config/AASIST.conf")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file")
    parser.add_argument("--test_dir",
                        type=str,
                        default="testing",
                        required=False
    )

    args = parser.parse_args()
    main(args)