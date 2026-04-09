import os
import re
import email
import email.policy
import logging
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from email import policy as email_policy
from email.parser import BytesParser, Parser
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack, csr_matrix

from imblearn.over_sampling import SMOTE
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Dataset Paths ──────────────────────────────────────────────────────
    # SpamAssassin: point to the root folder that contains "ham" and "spam" sub-dirs
    "spamassassin_dir": "./datasets/spamassassin",

    # Enron: two options —
    #   Option A (CSV)    : single CSV with columns "text" and "label" (0=ham, 1=spam)
    #   Option B (maildir): root dir that contains per-user mail directories
    "enron_csv":    "./datasets/enron/emails.csv",
    "enron_maildir":"./datasets/enron/maildir",

    # Kaggle Phishing: CSV with columns "Email Text" and "Email Type"
    #   Email Type values: "Phishing Email" | "Safe Email"
    "kaggle_csv": "./datasets/kaggle_phishing/Phishing_Email.csv",

    # ── Output ─────────────────────────────────────────────────────────────
    "model_dir":   "./models",
    "model_name":  "phishguard_rf.joblib",
    "report_path": "./models/evaluation_report.txt",

    # ── Feature Engineering ────────────────────────────────────────────────
    "tfidf_max_features": 15000,
    "tfidf_ngram_range":  (1, 3),
    "tfidf_sublinear_tf": True,

    # ── Random Forest Hyperparameters ──────────────────────────────────────
    "rf_n_estimators":      300,
    "rf_max_depth":         None,   # None = grow full trees
    "rf_min_samples_split": 5,
    "rf_min_samples_leaf":  2,
    "rf_max_features":      "sqrt",
    "rf_class_weight":      "balanced",  # handles class imbalance
    "rf_n_jobs":            -1,
    "rf_random_state":      42,
    "rf_oob_score":         True,

    # ── Training ───────────────────────────────────────────────────────────
    "test_size":        0.20,
    "val_size":         0.10,
    "random_state":     42,
    "use_smote":        False,   # enable if dataset is highly imbalanced
    "cv_folds":         5,

    # ── Misc ───────────────────────────────────────────────────────────────
    "max_samples_per_source": 20000,  # cap per dataset to keep memory in check
    "min_text_length":        20,
}

# Download NLTK data (one-time)
for pkg in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words('english'))
STEMMER   = PorterStemmer()


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lower-case, remove HTML, URLs, special chars, stem tokens."""
    if not isinstance(text, str):
        return ""
    # Strip HTML
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = text.lower()
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' URL_TOKEN ', text)
    # Replace email addresses
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.\w+\b', ' EMAIL_TOKEN ', text)
    # Replace phone numbers
    text = re.sub(r'\b\d[\d\s\-().+]{6,}\b', ' PHONE_TOKEN ', text)
    # Keep only letters and tokens
    text = re.sub(r'[^a-z\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords and stem
    tokens = [STEMMER.stem(t) for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# HAND-CRAFTED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
PHISHING_KEYWORDS = [
    'verify', 'account', 'suspended', 'urgent', 'click here', 'login',
    'password', 'bank', 'paypal', 'credit card', 'ssn', 'social security',
    'confirm', 'update', 'expires', 'limited time', 'act now', 'winner',
    'congratulations', 'claim', 'free', 'prize', 'million dollars', 'lottery',
    'inheritance', 'bitcoin', 'crypto', 'investment', 'validate', 'suspicious',
    'unusual activity', 'locked', 'disabled', 'blocked', 'restricted',
]

URGENCY_PHRASES = [
    'immediately', 'within 24 hours', 'urgent', 'asap', 'right now',
    'expire', 'last chance', 'today only', 'deadline', 'final notice',
]

DECEPTIVE_INDICATORS = [
    'dear customer', 'dear user', 'dear account holder', 'valued customer',
    'dear friend', 'hello dear', 'beloved',
]


def extract_handcrafted_features(text: str, subject: str = "", sender: str = "") -> List[float]:
    """
    Extract 30 hand-crafted features per email:
      - URL counts and patterns
      - Keyword density signals
      - Urgency / deception scores
      - Sender domain anomalies
      - Text statistics
      - HTML/attachment indicators
    """
    raw   = text.lower() if isinstance(text, str) else ""
    subj  = subject.lower() if isinstance(subject, str) else ""
    sendr = sender.lower() if isinstance(sender, str) else ""
    combined = raw + " " + subj

    # ── URL features ──────────────────────────────────────────────────────
    urls           = re.findall(r'https?://\S+|www\.\S+', raw)
    n_urls         = len(urls)
    n_suspicious_url = sum(1 for u in urls if any(s in u for s in
                        ['login','secure','verify','account','update','confirm',
                         'paypal','apple','google','microsoft','bank']))
    has_ip_url     = int(bool(re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', raw)))
    n_url_mismatch = int(bool(re.search(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', raw)
                          and len(re.findall(r'<a[^>]+href=', raw)) > 0))

    # ── Email count ───────────────────────────────────────────────────────
    n_emails       = len(re.findall(r'\b[\w.+-]+@[\w-]+\.\w+\b', raw))

    # ── Keyword signals ───────────────────────────────────────────────────
    word_count     = max(len(raw.split()), 1)
    kw_count       = sum(1 for kw in PHISHING_KEYWORDS if kw in combined)
    urgency_count  = sum(1 for ph in URGENCY_PHRASES    if ph in combined)
    deceptive_count= sum(1 for ph in DECEPTIVE_INDICATORS if ph in combined)
    kw_density     = kw_count / word_count

    # ── Subject features ──────────────────────────────────────────────────
    subj_has_re_fw   = int(bool(re.search(r'\b(re:|fw:|fwd:)', subj)))
    subj_all_caps    = int(bool(re.search(r'\b[A-Z]{4,}\b', subject or "")))
    subj_has_urgency = int(any(p in subj for p in ['urgent','immediately','asap','action required','verify']))
    subj_has_excl    = subject.count('!') if subject else 0
    subj_len         = len(subject) if subject else 0

    # ── Sender anomalies ──────────────────────────────────────────────────
    sender_has_digit    = int(bool(re.search(r'\d{3,}', sendr)))
    sender_looks_legit  = int(any(b in sendr for b in
                            ['paypal','amazon','google','microsoft','apple',
                             'facebook','netflix','bank']))
    sender_domain_typo  = int(bool(re.search(
        r'(paypa1|g00gle|arnazon|micros0ft|faceb00k|netf1ix|app1e)',
        sendr)))
    is_free_domain      = int(any(d in sendr for d in
                            ['gmail','yahoo','hotmail','outlook','aol','protonmail']))
    sender_reply_mismatch = 0   # placeholder; populated if both parsed

    # ── Text statistics ───────────────────────────────────────────────────
    char_count       = len(raw)
    avg_word_len     = sum(len(w) for w in raw.split()) / word_count
    excl_count       = raw.count('!')
    question_count   = raw.count('?')
    dollar_count     = raw.count('$')
    has_html         = int('<html' in raw or '<body' in raw or '<a href' in raw)
    has_attachment   = int(bool(re.search(
        r'content-disposition:\s*attachment|\.exe|\.zip|\.rar|\.pdf attached',
        raw, re.I)))
    n_digits_ratio   = sum(c.isdigit() for c in raw) / max(char_count, 1)
    uppercase_ratio  = sum(c.isupper() for c in (text or "")) / max(len(text or ""), 1)

    return [
        n_urls, n_suspicious_url, has_ip_url, n_url_mismatch,
        n_emails,
        kw_count, kw_density, urgency_count, deceptive_count,
        subj_has_re_fw, subj_all_caps, subj_has_urgency, subj_has_excl, subj_len,
        sender_has_digit, sender_looks_legit, sender_domain_typo,
        is_free_domain, sender_reply_mismatch,
        char_count, word_count, avg_word_len,
        excl_count, question_count, dollar_count,
        has_html, has_attachment,
        n_digits_ratio, uppercase_ratio,
        float(kw_count > 3),   # binary "many phishing keywords" flag
    ]


FEATURE_NAMES = [
    'n_urls','n_suspicious_url','has_ip_url','n_url_mismatch',
    'n_emails',
    'kw_count','kw_density','urgency_count','deceptive_count',
    'subj_re_fw','subj_all_caps','subj_urgency','subj_excl','subj_len',
    'sender_digit','sender_legit_brand','sender_domain_typo',
    'free_domain','sender_reply_mismatch',
    'char_count','word_count','avg_word_len',
    'excl_count','question_count','dollar_count',
    'has_html','has_attachment',
    'digit_ratio','uppercase_ratio',
    'many_kw_flag',
]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────

# def load_spamassassin(root_dir: str, max_samples: int = 20000) -> pd.DataFrame:
#     """
#     Load SpamAssassin Public Corpus.
#     Expected directory layout:
#         <root_dir>/
#             ham/   (or easy_ham/, hard_ham/)
#             spam/  (or spam_2/)
#     Download: https://spamassassin.apache.org/old/publiccorpus/
#     """
#     root = Path(root_dir)
#     if not root.exists():
#         log.warning(f"SpamAssassin dir not found: {root}")
#         return pd.DataFrame()

#     records = []
#     # Support multiple sub-folder naming conventions
#     ham_dirs  = [d for d in root.iterdir() if d.is_dir() and 'ham'  in d.name.lower()]
#     spam_dirs = [d for d in root.iterdir() if d.is_dir() and 'spam' in d.name.lower()]

#     def read_raw_email(path: Path) -> str:
#         for enc in ('utf-8','latin-1','cp1252'):
#             try:
#                 return path.read_text(encoding=enc, errors='replace')
#             except Exception:
#                 pass
#         return ""

#     def parse_sa_email(path: Path, label: int) -> Optional[dict]:
#         content = read_raw_email(path)
#         if not content:
#             return None
#         try:
#             msg = Parser(policy=email_policy.default).parsestr(content)
#         except Exception:
#             return {'text': content, 'subject': '', 'sender': '', 'label': label}
#         body = extract_body(msg)
#         return {
#             'text':    body,
#             'subject': str(msg.get('Subject', '') or ''),
#             'sender':  str(msg.get('From', '')    or ''),
#             'label':   label,
#         }

#     log.info("Loading SpamAssassin corpus…")
#     for ddir in ham_dirs:
#         for fp in tqdm(list(ddir.iterdir())[:max_samples//2], desc=f"  ham/{ddir.name}"):
#             if fp.is_file():
#                 r = parse_sa_email(fp, 0)
#                 if r: records.append(r)

#     for ddir in spam_dirs:
#         for fp in tqdm(list(ddir.iterdir())[:max_samples//2], desc=f"  spam/{ddir.name}"):
#             if fp.is_file():
#                 r = parse_sa_email(fp, 1)
#                 if r: records.append(r)

#     df = pd.DataFrame(records)
#     print(df.columns)   #test
#     print(df.head())    #test
#     log.info(f"SpamAssassin loaded: {len(df)} emails  "
#              f"(ham={len(df[df['label']==0])}, spam={len(df[df.label==1])})")
#     return df


# def extract_body(msg) -> str:
#     """Recursively extract plain-text body from an email.Message."""
#     body_parts = []
#     if msg.is_multipart():
#         for part in msg.walk():
#             ct = part.get_content_type()
#             cd = str(part.get('Content-Disposition', ''))
#             if 'attachment' in cd:
#                 continue
#             if ct == 'text/plain':
#                 try:
#                     body_parts.append(part.get_content())
#                 except Exception:
#                     payload = part.get_payload(decode=True)
#                     if payload:
#                         body_parts.append(payload.decode('utf-8','replace'))
#             elif ct == 'text/html':
#                 try:
#                     html = part.get_content()
#                 except Exception:
#                     payload = part.get_payload(decode=True)
#                     html = payload.decode('utf-8','replace') if payload else ''
#                 body_parts.append(BeautifulSoup(html,'html.parser').get_text(' '))
#     else:
#         try:
#             body_parts.append(msg.get_content())
#         except Exception:
#             payload = msg.get_payload(decode=True)
#             if payload:
#                 body_parts.append(payload.decode('utf-8','replace'))
#     return ' '.join(body_parts)


def load_enron(csv_path: str = None, maildir_path: str = None,
               max_samples: int = 20000) -> pd.DataFrame:
    """
    Load Enron Email Dataset.

    Option A — CSV (preferred):
        Kaggle: "Email Spam Classification Dataset CSV"
        Columns: "text" (or "message") + "label" (spam/ham or 1/0)
        URL: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

    Option B — raw maildir:
        Download from CMU: https://www.cs.cmu.edu/~./enron/
        Untar to a folder; each sub-folder is a user, inside are mail folders.
        Phishing/spam labels come from the enron-spam dataset variant.
    """
    if csv_path and Path(csv_path).exists():
        log.info(f"Loading Enron CSV from {csv_path}…")
        df = pd.read_csv(csv_path)
        # normalise column names
        df.columns = [c.lower().strip() for c in df.columns]
        text_col  = next((c for c in df.columns if c in ('text','message','body','email text')), None)
        label_col = next((c for c in df.columns if c in ('label','spam','category','class')), None)
        if not text_col or not label_col:
            log.warning(f"Could not identify text/label columns. Found: {df.columns.tolist()}")
            return pd.DataFrame()
        df = df[[text_col, label_col]].rename(columns={text_col:'text', label_col:'label'})
        # normalise labels to 0/1
        if df['label'].dtype == object:
            df['label'] = df['label'].str.lower().map(
                {'spam':1,'ham':0,'phishing':1,'safe':0,'1':1,'0':0})
        df['label']   = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        df['subject'] = ''
        df['sender']  = ''
        df = df.dropna(subset=['text']).head(max_samples)
        log.info(f"Enron CSV loaded: {len(df)} emails  "
                 f"(ham={len(df[df.label==0])}, spam={len(df[df.label==1])})")
        return df

    if maildir_path and Path(maildir_path).exists():
        log.info(f"Loading Enron maildir from {maildir_path}…")
        records = []
        root = Path(maildir_path)
        # The spam/ham split must come from external labels file if using raw maildir.
        # Here we demonstrate structure traversal; provide a labels dict if available.
        for user_dir in sorted(root.iterdir()):
            if not user_dir.is_dir(): continue
            for mail_folder in user_dir.iterdir():
                if not mail_folder.is_dir(): continue
                label = 1 if 'spam' in mail_folder.name.lower() else 0
                for fp in list(mail_folder.iterdir())[:500]:
                    if not fp.is_file(): continue
                    for enc in ('utf-8','latin-1'):
                        try:
                            content = fp.read_text(encoding=enc, errors='replace')
                            msg = Parser(policy=email_policy.default).parsestr(content)
                            records.append({
                                'text':    extract_body(msg),
                                'subject': str(msg.get('Subject','') or ''),
                                'sender':  str(msg.get('From','')    or ''),
                                'label':   label,
                            })
                            break
                        except Exception:
                            pass
            if len(records) >= max_samples: break
        df = pd.DataFrame(records)
        log.info(f"Enron maildir loaded: {len(df)} emails")
        return df

    log.warning("Enron dataset not found — skipping.")
    return pd.DataFrame()


def load_kaggle_phishing(csv_path: str, max_samples: int = 20000) -> pd.DataFrame:
    """
    Load Kaggle Phishing Email Dataset.
    Expected columns: "Email Text", "Email Type"
    Email Type: "Phishing Email" | "Safe Email"
    Download: https://www.kaggle.com/datasets/subhajournal/phishingemails
    """
    path = Path(csv_path)
    if not path.exists():
        log.warning(f"Kaggle phishing CSV not found: {path}")
        return pd.DataFrame()

    log.info(f"Loading Kaggle phishing dataset from {csv_path}…")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Try common column naming conventions
    text_col  = next((c for c in df.columns if 'text'    in c.lower() or 'body'  in c.lower()), None)
    label_col = next((c for c in df.columns if 'type'    in c.lower() or 'label' in c.lower()
                                            or 'class'   in c.lower()), None)

    if not text_col or not label_col:
        log.warning(f"Columns not recognised: {df.columns.tolist()}")
        return pd.DataFrame()

    df = df[[text_col, label_col]].rename(columns={text_col:'text', label_col:'label_str'})
    df['label'] = df['label_str'].str.lower().map(
        lambda x: 1 if 'phishing' in str(x) or 'spam' in str(x) else 0)
    df['subject'] = ''
    df['sender']  = ''
    df = df.dropna(subset=['text','label']).head(max_samples)

    log.info(f"Kaggle dataset loaded: {len(df)} emails  "
             f"(safe={len(df[df.label==0])}, phishing={len(df[df.label==1])})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class EmailFeatureBuilder:
    """
    Combines TF-IDF vectorizer with 30 hand-crafted features.
    fit_transform() returns a sparse matrix (TF-IDF) hstacked with dense array.
    """
    def __init__(self, tfidf_params: dict):
        self.tfidf = TfidfVectorizer(**tfidf_params)
        self.scaler = StandardScaler(with_mean=False)  # sparse-safe
        self._fitted = False

    def _handcrafted(self, df: pd.DataFrame) -> np.ndarray:
        feats = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  hand-crafted features"):
            feats.append(extract_handcrafted_features(
                row.get('text',''),
                row.get('subject',''),
                row.get('sender',''),
            ))
        return np.array(feats, dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame):
        log.info("Cleaning text…")
        df['clean'] = df['text'].apply(clean_text)
        log.info("Fitting TF-IDF…")
        tfidf_mat   = self.tfidf.fit_transform(df['clean'])
        log.info("Extracting hand-crafted features…")
        hc_mat      = self._handcrafted(df)
        hc_scaled   = self.scaler.fit_transform(hc_mat)
        self._fitted = True
        return hstack([tfidf_mat, csr_matrix(hc_scaled)])

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        df = df.copy()
        df['clean'] = df['text'].apply(clean_text)
        tfidf_mat   = self.tfidf.transform(df['clean'])
        hc_mat      = self._handcrafted(df)
        hc_scaled   = self.scaler.transform(hc_mat)
        return hstack([tfidf_mat, csr_matrix(hc_scaled)])


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def build_random_forest(cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators      = cfg['rf_n_estimators'],
        max_depth         = cfg['rf_max_depth'],
        min_samples_split = cfg['rf_min_samples_split'],
        min_samples_leaf  = cfg['rf_min_samples_leaf'],
        max_features      = cfg['rf_max_features'],
        class_weight      = cfg['rf_class_weight'],
        n_jobs            = cfg['rf_n_jobs'],
        random_state      = cfg['rf_random_state'],
        oob_score         = cfg['rf_oob_score'],
        verbose           = 0,
    )


def evaluate_model(clf, X_test, y_test, feature_builder: EmailFeatureBuilder,
                   report_path: str = None) -> dict:
    y_pred     = clf.predict(X_test)
    y_proba    = clf.predict_proba(X_test)[:, 1]
    roc_auc    = roc_auc_score(y_test, y_proba)
    avg_prec   = average_precision_score(y_test, y_proba)
    cm         = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])

    print("\n" + "═"*60)
    print("  MODEL EVALUATION REPORT")
    print("═"*60)
    print(report_str)
    print(f"  ROC-AUC Score        : {roc_auc:.4f}")
    print(f"  Average Precision    : {avg_prec:.4f}")
    if hasattr(clf, 'oob_score_'):
        print(f"  OOB Score            : {clf.oob_score_:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

    # Feature importance (top-20)
    n_tfidf = len(feature_builder.tfidf.vocabulary_)
    importances = clf.feature_importances_
    hc_importances = importances[n_tfidf:]
    hc_sorted = sorted(zip(FEATURE_NAMES, hc_importances), key=lambda x: x[1], reverse=True)

    print("\n  Top 10 Hand-Crafted Features by Importance:")
    for name, imp in hc_sorted[:10]:
        bar = '█' * int(imp * 400)
        print(f"    {name:<28} {bar}  ({imp:.5f})")

    tfidf_imps = importances[:n_tfidf]
    vocab_inv  = {v: k for k, v in feature_builder.tfidf.vocabulary_.items()}
    top_tfidf  = sorted(enumerate(tfidf_imps), key=lambda x: x[1], reverse=True)[:15]
    print("\n  Top 15 TF-IDF Tokens by Importance:")
    for idx, imp in top_tfidf:
        bar = '█' * int(imp * 600)
        print(f"    {vocab_inv.get(idx,'?'):<28} {bar}  ({imp:.5f})")
    print("═"*60)

    metrics = {
        'roc_auc': roc_auc, 'avg_precision': avg_prec,
        'confusion_matrix': cm.tolist(),
        'classification_report': report_str,
    }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_str + f"\nROC-AUC: {roc_auc:.4f}\nAvg Precision: {avg_prec:.4f}\n")
        log.info(f"Report saved → {report_path}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER  (for integration into PhishGuard app)
# ─────────────────────────────────────────────────────────────────────────────
class PhishGuardPredictor:
    """
    Load a saved model bundle and predict on raw email data.

    Usage:
        predictor = PhishGuardPredictor("./models/phishguard_rf.joblib")
        result = predictor.predict(
            text    = "Dear customer, verify your account immediately…",
            subject = "URGENT: Account suspended",
            sender  = "noreply@paypa1-secure.com",
        )
        print(result)
        # {'verdict': 'PHISHING', 'threat_score': 92, 'confidence': 0.94,
        #  'top_features': [...]}
    """
    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.clf             = bundle['clf']
        self.feature_builder = bundle['feature_builder']
        log.info(f"PhishGuardPredictor loaded from {model_path}")

    def predict(self, text: str, subject: str = "", sender: str = "") -> dict:
        df = pd.DataFrame([{'text': text, 'subject': subject, 'sender': sender}])
        X  = self.feature_builder.transform(df)
        proba   = self.clf.predict_proba(X)[0]
        phish_p = float(proba[1])
        verdict = ('PHISHING'   if phish_p >= 0.70 else
                   'SUSPICIOUS' if phish_p >= 0.40 else
                   'SAFE')
        hc = extract_handcrafted_features(text, subject, sender)
        top_feat = sorted(zip(FEATURE_NAMES, hc), key=lambda x: abs(x[1]), reverse=True)[:5]
        return {
            'verdict':      verdict,
            'threat_score': round(phish_p * 100),
            'confidence':   round(phish_p, 4),
            'safe_prob':    round(float(proba[0]), 4),
            'top_features': {k: round(v, 4) for k, v in top_feat},
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("""
╔══════════════════════════════════════════════╗
║   PhishGuard AI — Random Forest Training     ║
╚══════════════════════════════════════════════╝
    """)

    # ── 1. Load datasets ──────────────────────────────────────────────────
    dfs = []
    log.info("Loading datasets…")

    # df_sa = load_spamassassin(CFG['spamassassin_dir'], CFG['max_samples_per_source'])
    # if not df_sa.empty: dfs.append(df_sa)

    df_en = load_enron(
        csv_path    = CFG['enron_csv']    if Path(CFG['enron_csv']).exists()    else None,
        maildir_path= CFG['enron_maildir'] if Path(CFG['enron_maildir']).exists() else None,
        max_samples = CFG['max_samples_per_source'],
    )
    if not df_en.empty: dfs.append(df_en)

    df_kg = load_kaggle_phishing(CFG['kaggle_csv'], CFG['max_samples_per_source'])
    if not df_kg.empty: dfs.append(df_kg)

    if not dfs:
        log.error("No datasets loaded! Check your paths in CFG and download the datasets.")
        log.error("""
  Dataset download links:
  ─ SpamAssassin : https://spamassassin.apache.org/old/publiccorpus/
  ─ Enron (CSV)  : https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
  ─ Kaggle Phish : https://www.kaggle.com/datasets/subhajournal/phishingemails
        """)
        return

    # ── 2. Combine & sanity-check ─────────────────────────────────────────
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['text'] = all_df['text'].fillna('').astype(str)
    all_df = all_df[all_df['text'].str.len() >= CFG['min_text_length']].reset_index(drop=True)

    log.info(f"Combined dataset: {len(all_df)} emails  "
             f"(legitimate={len(all_df[all_df.label==0])}, phishing/spam={len(all_df[all_df.label==1])})")

    # ── 3. Train / validation / test split ────────────────────────────────
    X_raw, X_test_raw, y_train_all, y_test = train_test_split(
        all_df, all_df['label'],
        test_size    = CFG['test_size'],
        random_state = CFG['random_state'],
        stratify     = all_df['label'],
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y_train_all,
        test_size    = CFG['val_size'] / (1 - CFG['test_size']),
        random_state = CFG['random_state'],
        stratify     = y_train_all,
    )
    log.info(f"Split sizes — train:{len(X_train_raw)}  val:{len(X_val_raw)}  test:{len(X_test_raw)}")

    # ── 4. Build features ─────────────────────────────────────────────────
    tfidf_params = {
        'max_features': CFG['tfidf_max_features'],
        'ngram_range':  CFG['tfidf_ngram_range'],
        'sublinear_tf': CFG['tfidf_sublinear_tf'],
        'min_df': 3,
        'max_df': 0.95,
        'analyzer': 'word',
        'token_pattern': r'\b[a-z_]{2,}\b',
    }
    fb = EmailFeatureBuilder(tfidf_params)
    log.info("Building training features…")
    X_train = fb.fit_transform(X_train_raw)
    log.info("Building validation features…")
    X_val   = fb.transform(X_val_raw)
    log.info("Building test features…")
    X_test  = fb.transform(X_test_raw)

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    # ── 5. Optional SMOTE oversampling ────────────────────────────────────
    if CFG['use_smote']:
        log.info("Applying SMOTE oversampling…")
        sm = SMOTE(random_state=CFG['random_state'], k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        log.info(f"After SMOTE: {X_train.shape[0]} samples")

    # ── 6. Train Random Forest ────────────────────────────────────────────
    log.info(f"Training Random Forest ({CFG['rf_n_estimators']} trees)…")
    clf = build_random_forest(CFG)
    clf.fit(X_train, y_train)
    log.info("Training complete.")

    # ── 7. Validation check ───────────────────────────────────────────────
    val_pred  = clf.predict(X_val)
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_auc   = roc_auc_score(y_val, val_proba)
    log.info(f"Validation ROC-AUC: {val_auc:.4f}")

    # ── 8. Cross-validation ───────────────────────────────────────────────
    log.info(f"Running {CFG['cv_folds']}-fold cross-validation on training set…")
    from scipy.sparse import vstack as sp_vstack
    X_cv = sp_vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])
    cv_scores = cross_val_score(
        build_random_forest(CFG), X_cv, y_cv,
        cv=StratifiedKFold(n_splits=CFG['cv_folds'], shuffle=True,
                           random_state=CFG['random_state']),
        scoring='roc_auc', n_jobs=-1,
    )
    log.info(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 9. Evaluate on test set ───────────────────────────────────────────
    evaluate_model(clf, X_test, y_test, fb, CFG['report_path'])

    # ── 10. Save model bundle ─────────────────────────────────────────────
    model_dir  = Path(CFG['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / CFG['model_name']

    bundle = {
        'clf':             clf,
        'feature_builder': fb,
        'config':          CFG,
        'cv_auc_mean':     cv_scores.mean(),
        'cv_auc_std':      cv_scores.std(),
        'val_auc':         val_auc,
    }
    joblib.dump(bundle, model_path, compress=3)
    log.info(f"Model bundle saved → {model_path}  ({model_path.stat().st_size / 1e6:.1f} MB)")

    # ── 11. Quick smoke test ──────────────────────────────────────────────
    print("\n── Smoke Test ──────────────────────────────────────────────")
    predictor = PhishGuardPredictor(str(model_path))

    examples = [
        {
            "text":    "Dear customer, your PayPal account has been limited. Click here immediately to verify your information and avoid account suspension. Act now!",
            "subject": "URGENT: PayPal account suspended",
            "sender":  "noreply@paypa1-secure.net",
            "expected":"PHISHING",
        },
        {
            "text":    "Hi team, the Q3 budget meeting has been rescheduled to Thursday at 2pm. Please bring your reports. Let me know if you have conflicts.",
            "subject": "Meeting rescheduled",
            "sender":  "sarah.johnson@company.com",
            "expected":"SAFE",
        },
        {
            "text":    "Congratulations! You've been selected to receive a $1000 Amazon gift card. Click the link below to claim your prize within 24 hours.",
            "subject": "You're a winner!!!",
            "sender":  "prizes@free-rewards123.xyz",
            "expected":"PHISHING",
        },
    ]

    for ex in examples:
        r = predictor.predict(ex['text'], ex['subject'], ex['sender'])
        status = "✓" if r['verdict'] == ex['expected'] else "✗"
        print(f"  {status} [{r['verdict']:^11}] score={r['threat_score']:>3}/100  "
              f"conf={r['confidence']:.2f}  → {ex['subject'][:50]}")
    print("────────────────────────────────────────────────────────────\n")

    print(f"\n Training complete! Model ready at: {model_path}")
    print(f"   Use PhishGuardPredictor to integrate into your app.\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()