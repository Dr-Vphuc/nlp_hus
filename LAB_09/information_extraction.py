"""Mini Information Extraction (LAB 09, mục 25-27) — Option A (News): PERSON / ORGANIZATION / LOCATION.

Pipeline:  raw text -> tách từ -> NER (DistilBERT đã fine-tune) -> gom span -> JSON có cấu trúc.
Đánh giá: 30 câu tin tức tự viết (KHÔNG nằm trong CoNLL) + gold annotation tự gán ở dưới, tính P/R/F1 entity-level (exact match).
MISC bị bỏ khỏi kết quả IE (Option A chỉ lấy PERSON/ORGANIZATION/LOCATION); gold cũng chỉ gán 3 loại này.

Chạy:  python information_extraction.py     (cần ner_checkpoint/ do train_ner.py tạo ra)
Xuất:  ie_examples.json (JSON có cấu trúc cho 30 câu), ie_results.csv (P/R/F1), ie_errors.csv
"""

import json
import re
import sys

import pandas as pd
import torch

from ner_model import encode_sentence, load_model, predict
from ner_utils import entity_metrics, spans_from_tags, words_to_bio

TYPE_NAME = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"}
TOKEN_RE = re.compile(r"(?:[A-Za-z]\.){2,}|\w+(?:[-'’]\w+)*|[^\w\s]")  # U.S. | từ (có - ') | dấu câu


def split_words(text):
    return TOKEN_RE.findall(text)


class NERExtractor:
    def __init__(self, ckpt_dir="ner_checkpoint", device="cpu"):
        self.device = device
        self.model, self.tok = load_model(ckpt_dir, device)

    def tag(self, words):
        enc = [encode_sentence(self.tok, words)]
        pred, conf = predict(self.model, self.tok, enc, [len(words)], device=self.device)
        return pred[0], conf[0]

    def extract(self, text):
        """Raw text -> (dict {PERSON: [...], ORGANIZATION: [...], LOCATION: [...]}, list span chi tiết)."""
        words = split_words(text)
        tags, conf = self.tag(words)
        out = {v: [] for v in TYPE_NAME.values()}
        spans = []
        for etype, s, e in sorted(spans_from_tags(tags), key=lambda x: x[1]):
            if etype not in TYPE_NAME:
                continue  # bỏ MISC
            surface = " ".join(words[s:e])
            if surface not in out[TYPE_NAME[etype]]:
                out[TYPE_NAME[etype]].append(surface)
            spans.append((etype, s, e, surface))
        return out, spans, words, tags


# 30 câu + gold (cụm từ, loại). Gán theo quy ước CoNLL: tên đội/CLB = ORG, quốc gia (kể cả đội tuyển) = LOC.
GOLD = [
    ("Barack Obama visited Paris on Monday .", [("Barack Obama", "PER"), ("Paris", "LOC")]),
    ("Apple chief executive Tim Cook announced new products in Cupertino .", [("Apple", "ORG"), ("Tim Cook", "PER"), ("Cupertino", "LOC")]),
    ("Angela Merkel met Emmanuel Macron in Berlin to discuss the European Union budget .",
     [("Angela Merkel", "PER"), ("Emmanuel Macron", "PER"), ("Berlin", "LOC"), ("European Union", "ORG")]),
    ("Microsoft said on Tuesday it would open a new office in Dublin .", [("Microsoft", "ORG"), ("Dublin", "LOC")]),
    ("The United Nations Security Council held an emergency meeting in New York .",
     [("United Nations Security Council", "ORG"), ("New York", "LOC")]),
    ("Manchester United beat Chelsea 2-1 on Saturday .", [("Manchester United", "ORG"), ("Chelsea", "ORG")]),
    ("Prime Minister Narendra Modi arrived in Tokyo for talks with Fumio Kishida .",
     [("Narendra Modi", "PER"), ("Tokyo", "LOC"), ("Fumio Kishida", "PER")]),
    ("Reuters reported that Toyota will build a factory in Texas .", [("Reuters", "ORG"), ("Toyota", "ORG"), ("Texas", "LOC")]),
    ("Elon Musk founded SpaceX in California .", [("Elon Musk", "PER"), ("SpaceX", "ORG"), ("California", "LOC")]),
    ("Researchers at Stanford University and the Massachusetts Institute of Technology published the study last week .",
     [("Stanford University", "ORG"), ("Massachusetts Institute of Technology", "ORG")]),
    ("Pele was born in Brazil and later played for Santos .", [("Pele", "PER"), ("Brazil", "LOC"), ("Santos", "ORG")]),
    ("The World Health Organization warned that cases were rising in Africa .", [("World Health Organization", "ORG"), ("Africa", "LOC")]),
    ("Samsung Electronics and Google signed an agreement in Seoul .",
     [("Samsung Electronics", "ORG"), ("Google", "ORG"), ("Seoul", "LOC")]),
    ("President Joe Biden spoke with Volodymyr Zelensky from Washington .",
     [("Joe Biden", "PER"), ("Volodymyr Zelensky", "PER"), ("Washington", "LOC")]),
    ("Lionel Messi scored twice as Argentina defeated France in Qatar .",
     [("Lionel Messi", "PER"), ("Argentina", "LOC"), ("France", "LOC"), ("Qatar", "LOC")]),
    ("Goldman Sachs analysts expect the Federal Reserve to cut rates in March .", [("Goldman Sachs", "ORG"), ("Federal Reserve", "ORG")]),
    ("Marie Curie worked at the University of Paris .", [("Marie Curie", "PER"), ("University of Paris", "ORG")]),
    ("Amazon opened a warehouse near Mumbai , while Flipkart expanded in Bangalore .",
     [("Amazon", "ORG"), ("Mumbai", "LOC"), ("Flipkart", "ORG"), ("Bangalore", "LOC")]),
    ("Vietnam Airlines launched a new route between Hanoi and Singapore .",
     [("Vietnam Airlines", "ORG"), ("Hanoi", "LOC"), ("Singapore", "LOC")]),
    ("The European Commission fined Meta 1.2 billion euros .", [("European Commission", "ORG"), ("Meta", "ORG")]),
    ("Novak Djokovic won the Australian Open final in Melbourne .", [("Novak Djokovic", "PER"), ("Melbourne", "LOC")]),
    ("Tesla shares fell after Elon Musk posted on Twitter .", [("Tesla", "ORG"), ("Elon Musk", "PER"), ("Twitter", "ORG")]),
    ("A spokesman for the Pentagon said troops would leave Iraq by December .", [("Pentagon", "ORG"), ("Iraq", "LOC")]),
    ("Volkswagen plans to close a plant in Dresden , the company told Reuters .",
     [("Volkswagen", "ORG"), ("Dresden", "LOC"), ("Reuters", "ORG")]),
    ("Sundar Pichai met Indian officials in Delhi .", [("Sundar Pichai", "PER"), ("Delhi", "LOC")]),
    ("The Bank of England raised interest rates , Governor Andrew Bailey said in London .",
     [("Bank of England", "ORG"), ("Andrew Bailey", "PER"), ("London", "LOC")]),
    ("Mount Everest lies on the border between Nepal and China .", [("Mount Everest", "LOC"), ("Nepal", "LOC"), ("China", "LOC")]),
    ("Jack Ma stepped down as chairman of Alibaba in Hangzhou .", [("Jack Ma", "PER"), ("Alibaba", "ORG"), ("Hangzhou", "LOC")]),
    ("Liverpool signed Mohamed Salah from Roma in 2017 .", [("Liverpool", "ORG"), ("Mohamed Salah", "PER"), ("Roma", "ORG")]),
    ("Greenpeace activists protested outside the Shell headquarters in The Hague .",
     [("Greenpeace", "ORG"), ("Shell", "ORG"), ("The Hague", "LOC")]),
]


def evaluate(extractor):
    gold_seqs, pred_seqs, rows, examples = [], [], [], []
    for text, ents in GOLD:
        words = text.split()  # câu gold đã tách sẵn bằng khoảng trắng
        gold = words_to_bio(words, ents)
        tags, _ = extractor.tag(words)
        tags = [t if t == "O" or t[2:] in TYPE_NAME else "O" for t in tags]  # bỏ MISC
        gold_seqs.append(gold)
        pred_seqs.append(tags)
        g, p = spans_from_tags(gold), spans_from_tags(tags)
        for e in sorted(p - g, key=lambda x: x[1]):
            rows.append((text, "FP", e[0], " ".join(words[e[1]:e[2]])))
        for e in sorted(g - p, key=lambda x: x[1]):
            rows.append((text, "FN", e[0], " ".join(words[e[1]:e[2]])))
        raw = " ".join(words).replace(" .", ".").replace(" ,", ",")
        examples.append({"text": raw, "extracted": extractor.extract(text)[0],
                         "gold": {TYPE_NAME[t]: [ph for ph, tt in ents if tt == t] for t in TYPE_NAME}})
    return entity_metrics(gold_seqs, pred_seqs), rows, examples


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ex = NERExtractor(device=device)
    demo, _, _, _ = ex.extract("Barack Obama visited Paris.")
    print("Demo:", json.dumps(demo, ensure_ascii=False))

    m, err_rows, examples = evaluate(ex)
    out = [{"entity": "OVERALL", **{k: m["overall"][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "support")}}]
    for t in ("PER", "ORG", "LOC"):
        out.append({"entity": TYPE_NAME[t], **{k: m["per_type"][t][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "support")}})
    df = pd.DataFrame(out)
    print(df.round(4).to_string(index=False))
    df.to_csv("ie_results.csv", index=False)
    pd.DataFrame(err_rows, columns=["sentence", "error", "type", "entity"]).to_csv("ie_errors.csv", index=False)
    with open("ie_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=1, ensure_ascii=False)
    print("Errors:")
    for r in err_rows:
        print("  ", r)
    print("\nVí dụ pipeline (3 câu đầu):")
    for e in examples[:3]:
        print(json.dumps(e["extracted"], ensure_ascii=False), "<-", e["text"])
