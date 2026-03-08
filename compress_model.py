import joblib, os

pipe = joblib.load("outputs/pipeline.joblib")
old_size = os.path.getsize("outputs/pipeline.joblib")

joblib.dump(pipe, "outputs/pipeline.joblib", compress=3)
new_size = os.path.getsize("outputs/pipeline.joblib")

print(f"Before: {old_size/1024/1024:.1f} MB")
print(f"After:  {new_size/1024/1024:.1f} MB")

# Also compress the separate parts
clf = joblib.load("outputs/model.joblib")
vec = joblib.load("outputs/vectorizer.joblib")
joblib.dump(clf, "outputs/model.joblib", compress=3)
joblib.dump(vec, "outputs/vectorizer.joblib", compress=3)
print("All artifacts compressed.")
