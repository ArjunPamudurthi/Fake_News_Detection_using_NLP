import joblib, re

pipe = joblib.load("outputs/pipeline.joblib")

def clean(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

article = (
    "Global renewable energy capacity has increased significantly in recent years, "
    "with solar and wind power leading the transition away from fossil fuels. "
    "According to the International Energy Agency, renewable sources now account "
    "for nearly 30 percent of global electricity generation. Governments worldwide "
    "are implementing policies to accelerate this shift, including tax incentives "
    "for clean energy investment and stricter emissions standards for power plants."
)

cleaned = clean(article)
proba = pipe.predict_proba([cleaned])[0]
label = "REAL" if proba[0] > proba[1] else "FAKE"
print(f"REAL prob: {proba[0]:.4f}")
print(f"FAKE prob: {proba[1]:.4f}")
print(f"Prediction: {label}")
