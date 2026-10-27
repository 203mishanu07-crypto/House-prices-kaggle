from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

def predict_torch(X_test, checkpoint, model, device, batch_size=256):
    cat_features = checkpoint["cat_cols"]
    num_features = checkpoint["num_cols"]
    encoder = checkpoint["encoder"]
    scaler = checkpoint["scaler"]

    missing = set(cat_features + num_features) - set(X_test.columns)
    if missing:
        raise ValueError(f"Отсутствуют признаки для DL: {sorted(missing)}")

    X_cat = encoder.transform(X_test[cat_features]).astype("int64") + 1
    X_num = scaler.transform(X_test[num_features]).astype("float32")

    if not np.isfinite(X_num).all():
        raise ValueError("В числовых признаках после StandardScaler есть NaN или inf.")
    if (X_cat < 0).any():
        raise ValueError(
            "В категориальных индексах есть отрицательные значения. Проверьте кодирование."
        )

    # OrdinalEncoder у модели должен обучаться с unknown_value=-1,
    # чтобы после +1 неизвестные категории получили индекс 0.
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(X_test), batch_size):
            end = start + batch_size
            x_cat = torch.as_tensor(X_cat[start:end], dtype=torch.long, device=device)
            x_num = torch.as_tensor(
                X_num[start:end], dtype=torch.float32, device=device
            )
            preds = model(x_num, x_cat)
            predictions.append(preds.cpu().numpy())

    return np.concatenate(predictions)