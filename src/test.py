# # Analyze misclassified predictions to identify total errors, class-wise mistakes, 
# commonly confused classes, and average prediction confidence.
import pandas as pd

df = pd.read_csv(
    "artifacts/evaluation/misclassified_predictions.csv"
)

print("Total errors:", len(df))

print("\nErrors by actual class:")
print(df["actual_class"].value_counts())

print("\nActual class → Predicted class:")
print(
    df.groupby(["actual_class", "predicted_class"])
      .size()
      .sort_values(ascending=False)
)

print("\nAverage confidence of incorrect predictions:")
print(df["confidence"].mean())