"""Fine-tune DistilBERT (cased) trên CoNLL-2003 rồi lưu dự đoán trên test.

Chạy:  python train_ner.py 2          (CPU; có cơ chế resume nếu tiến trình bị kill)
Xuất:  ner_checkpoint/ (epoch có val F1 tốt nhất), training_log.json, transformer_predictions.json
"""

import json
import os
import sys

import torch

from ner_model import RESUME_PATH, encode_split, load_model, predict, train_model
from ner_utils import load_conll, tags_to_labels

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, "threads", torch.get_num_threads(), flush=True)
    ds = load_conll()
    train_model(ds, epochs=epochs, device=device)
    model, tok = load_model(device=device)
    enc_te = encode_split(tok, ds["test"])
    pred, conf = predict(model, tok, enc_te, [len(e["tokens"]) for e in ds["test"]], device=device)
    with open("transformer_predictions.json", "w") as f:
        json.dump({"pred": pred, "conf": conf}, f)
    print("saved transformer_predictions.json")
    if os.path.exists(RESUME_PATH):
        os.remove(RESUME_PATH)  # trạng thái resume không còn cần
