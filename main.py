import os
import zipfile
import urllib.request
import io
import re
import math
import random
from collections import Counter, defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------
UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
LOCAL_TXT = "SMSSpamCollection"


def download_and_extract_ucismsspam(dest_folder="."):
    print("Preuzimam dataset sa UCI...")
    resp = urllib.request.urlopen(UCI_ZIP_URL)
    data = resp.read()
    z = zipfile.ZipFile(io.BytesIO(data))
    for name in z.namelist():
        if name.lower().endswith("smsspamcollection"):
            content = z.read(name).decode('utf-8', errors='ignore')
            path = os.path.join(dest_folder, LOCAL_TXT)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print("Raspakovan:", path)
            return path
    raise FileNotFoundError("SMSSpamCollection nije pronadjen u zip-u.")


# --------------------
def load_smsspam(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['label', 'text'], encoding='utf-8', quoting=3)
    return df


# --------------------
RE_URL = re.compile(r'(http[s]?://|www\.)', re.IGNORECASE)
RE_PHONE = re.compile(r'(\+?\d[\d\-\s]{5,}\d)')

KEYWORDS = [
    'free', 'win', 'winner', 'prize', 'congrat', 'cash', 'urgent', 'offer', 'buy', 'cheap',
    'click', 'visit', 'promo', 'loan', 'money', 'claim', 'limited', 'unsubscribe', 'reply', 'reward',
    'call', 'txt', 'now', 'guarantee', 'easy'
]

SPAM_KEYWORD_COMBO = [
    'free', 'win', 'winner', 'prize', 'congrat', 'cash', 'urgent', 'offer', 'buy', 'cheap', 'money', 'claim'
]


def text_to_binary_features(text):
    t = text.strip()
    t_lower = t.lower()
    words = t_lower.split()
    words_set = set(words)

    feats = {}

    for kw in KEYWORDS:
        feats[f'has_{kw}'] = 1 if kw in words_set else 0

    feats['has_link'] = 1 if RE_URL.search(t) else 0
    feats['has_digits'] = 1 if any(ch.isdigit() for ch in t) else 0
    feats['has_phone_like'] = 1 if RE_PHONE.search(t) else 0

    feats['len_words_gt_7'] = 1 if len(words) > 7 else 0
    feats['len_words_gt_12'] = 1 if len(words) > 12 else 0

    num_caps = sum(1 for ch in t if ch.isupper())
    feats['many_caps'] = 1 if num_caps > 5 else 0
    feats['has_emoticon'] = 1 if (':)' in t_lower or ':-)' in t_lower or ':(' in t_lower) else 0

    feats['has_unsubscribe'] = 1 if 'unsubscribe' in t_lower or 'stop' in t_lower else 0


    feats['spam_keyword_present'] = 1 if words_set & set(SPAM_KEYWORD_COMBO) else 0
    feats['heuristic_spam_combo'] = 1 if sum(feats[f'has_{k}'] for k in SPAM_KEYWORD_COMBO) >= 2 else 0

    # kratke poruke koje su spam
    feats['short_spam'] = 1 if len(words) <= 3 and feats['spam_keyword_present'] else 0
    feats['short_message_spam'] = 1 if len(words) <= 5 and feats['spam_keyword_present'] else 0

    # specifične kombinacije
    feats['free_money_combo'] = 1 if 'free' in words_set and 'money' in words_set else 0
    feats['win_prize_combo'] = 1 if 'win' in words_set and 'prize' in words_set else 0
    feats['urgent_offer_combo'] = 1 if 'urgent' in words_set and 'offer' in words_set else 0

    return feats


# --------------------
# ID3
def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0.0
    counts = Counter(labels)
    ent = 0.0
    for cnt in counts.values():
        p = cnt / total
        ent -= p * math.log2(p) if p > 0 else 0.0
    return ent


def information_gain(parent_labels, splits_labels):
    parent_ent = entropy(parent_labels)
    total = len(parent_labels)
    weighted = 0.0
    for s in splits_labels:
        weighted += (len(s) / total) * entropy(s)
    return parent_ent - weighted


class ID3Node:
    def __init__(self, attribute=None, is_leaf=False, prediction=None):
        self.attribute = attribute
        self.children = {}
        self.is_leaf = is_leaf
        self.prediction = prediction


def majority(labels):
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def id3(examples, attributes, labels, depth=0, max_depth=None):
    if len(set(labels)) == 1:
        return ID3Node(is_leaf=True, prediction=labels[0])
    if not attributes or (max_depth is not None and depth >= max_depth):
        return ID3Node(is_leaf=True, prediction=majority(labels))
    best_attr = None
    best_ig = -1
    for attr in attributes:
        val_to_labels = defaultdict(list)
        for ex, lab in zip(examples, labels):
            val = ex.get(attr, '0')
            val_to_labels[val].append(lab)
        splits = list(val_to_labels.values())
        ig = information_gain(labels, splits)
        if ig > best_ig:
            best_ig = ig
            best_attr = attr
    if best_attr is None:
        return ID3Node(is_leaf=True, prediction=majority(labels))
    node = ID3Node(attribute=best_attr)
    values = set(ex.get(best_attr, '0') for ex in examples)
    for val in values:
        sub_examples = [ex for ex in examples if ex.get(best_attr, '0') == val]
        sub_labels = [lab for ex, lab in zip(examples, labels) if ex.get(best_attr, '0') == val]
        if not sub_examples:
            node.children[val] = ID3Node(is_leaf=True, prediction=majority(labels))
        else:
            remaining_attrs = [a for a in attributes if a != best_attr]
            node.children[val] = id3(sub_examples, remaining_attrs, sub_labels, depth + 1, max_depth)
    return node


def predict_single(node, example):
    while not node.is_leaf:
        attr = node.attribute
        val = example.get(attr, '0')
        if val not in node.children:
            child_preds = [child.prediction for child in node.children.values() if child.is_leaf]
            if child_preds:
                return majority(child_preds)
            else:
                return random.choice(list(node.children.values())).prediction
        node = node.children[val]
    return node.prediction


# Heuristika nakon predikcije
def predict_single_with_heuristic(node, example):
    label = predict_single(node, example)
    if int(example.get('short_message_spam', '0')) == 1:
        return 'spam'
    if int(example.get('free_money_combo', '0')) == 1:
        return 'spam'
    if int(example.get('win_prize_combo', '0')) == 1:
        return 'spam'
    if int(example.get('urgent_offer_combo', '0')) == 1:
        return 'spam'
    return label


def predict(node, examples):
    return [predict_single_with_heuristic(node, ex) for ex in examples]


# -------------------
def train_tree():
    if not os.path.exists(LOCAL_TXT):
        path = download_and_extract_ucismsspam(dest_folder=".")
    else:
        path = LOCAL_TXT

    df = load_smsspam(path)

    # Filter: uklanjanje klasa sa samo 1 primerkom
    feat_dicts = []
    labels = []
    counts = Counter(df['label'])
    valid_labels = {lab for lab, cnt in counts.items() if cnt > 1}

    for t, lab in zip(df['text'].astype(str), df['label']):
        if lab not in valid_labels:
            continue
        feats = text_to_binary_features(t)
        feats = {k: str(v) for k, v in feats.items()}
        feat_dicts.append(feats)
        labels.append(lab)

    attributes = list(feat_dicts[0].keys())

    X_train, X_test, y_train, y_test = train_test_split(
        feat_dicts, labels, test_size=0.30, random_state=42, stratify=labels
    )

    tree = id3(X_train, attributes, y_train, max_depth=None)
    return tree
