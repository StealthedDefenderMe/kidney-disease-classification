from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

csv_path = Path("artifacts/evaluation/misclassified_predictions.csv")
image_dir = Path("artifacts/evaluation/misclassified_images")

# Code to get the sample of 12 pictures predicted confidently wrong as cyst from tumor

df = pd.read_csv(csv_path)

# Inspect Tumor images predicted as Cyst
errors = df[
    (df["actual_class"] == "Tumor") &
    (df["predicted_class"] == "Cyst")
].head(12)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for ax, (_, row) in zip(axes.flat, errors.iterrows()):
    image_path = image_dir / row["image_name"]

    if image_path.exists():
        ax.imshow(Image.open(image_path))
        ax.set_title(f'{row["actual_class"]} → {row["predicted_class"]}\n'
                     f'Confidence: {row["confidence"]}%')
    else:
        ax.set_title(f'Image not found:\n{row["image_name"]}')

    ax.axis("off")

for ax in list(axes.flat)[len(errors):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("artifacts/evaluation/tumor_to_cyst_contact_sheet.png")
plt.show()