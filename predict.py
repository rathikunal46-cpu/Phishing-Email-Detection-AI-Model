import sys
import joblib
from scipy.sparse import hstack

# 🔥 IMPORTANT (fix class loading issue)
from train import EmailFeatureBuilder   # or feature_builder


def load_model(model_path):
    bundle = joblib.load(model_path)

    # 🔥 Correct keys based on your output
    model = bundle["clf"]
    feature_builder = bundle["feature_builder"]

    return model, feature_builder


def predict_email(text, model, feature_builder):
    import pandas as pd

    df = pd.DataFrame({"text": [text]})
    X = feature_builder.transform(df)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    label = "PHISHING" if pred == 1 else "SAFE"
    score = int(prob * 100)

    return label, score, prob


def main():
    if len(sys.argv) < 2:
        print('\nUsage: python predict.py "Your email text here"\n')
        return

    email_text = sys.argv[1]

    print("\n🔍 Analyzing email...\n")

    model, feature_builder = load_model("models/phishguard_rf.joblib")

    label, score, prob = predict_email(email_text, model, feature_builder)

    print("===================================")
    print(f"[ {label} ]")
    print(f"Score      : {score}/100")
    print(f"Confidence : {prob:.2f}")
    print("===================================\n")


if __name__ == "__main__":
    main()