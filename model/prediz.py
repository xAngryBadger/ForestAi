import torch
from deepforest import main
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import pandas as pd
import numpy as np # We need numpy for the fix

# 1. Define the path to your best trained model
model_path = r"E:\deepbug ai\model\checkpoints\best_model-epoch=30-val_bbox_regression=0.49.ckpt"

# 2. Define the path to an image you want to predict
image_path = r"F:\ForestsAI\ForestsAi img\img\inúteis\senna_alata (88).JPG"

# Check if the image exists
image_path_obj = Path(image_path)
if not image_path_obj.exists():
    raise FileNotFoundError(f"Image not found at: {image_path}")

# 3. Load the model from the checkpoint
model = main.deepforest.load_from_checkpoint(model_path)

# Ensure the model is in evaluation mode
model.eval()

# 4. Set the prediction threshold on the model's configuration
model.config['score_thresh'] = 0.3

# --- CUSTOM ANNOTATION WORKFLOW ---

# 5. Get RAW PREDICTIONS instead of a plotted image
print(f"Predicting on image: {image_path}")
predictions_df = model.predict_image(
    path=image_path,
    return_plot=False
)

### THE FIX IS HERE ###
# 6. Load the image using a method that supports non-ASCII characters
# This reads the file as bytes, then decodes it with OpenCV from memory,
# which avoids issues with special characters in the file path.
try:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_to_draw = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
except Exception as e:
    print(f"Error loading image with cv2.imdecode: {e}")
    # Fallback to the original method, which might fail
    image_to_draw = cv2.imread(image_path)

if image_to_draw is None:
    raise IOError(f"Could not load the image file at: {image_path}. Check the file path and integrity.")

# 7. Check if any predictions were made
if predictions_df is not None and not predictions_df.empty:
    print(f"Found {len(predictions_df)} objects. Drawing annotations...")
    for index, row in predictions_df.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label, score = row['label'], row['score']

        box_color = (0, 0, 0)
        box_thickness = 3
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)

        cv2.rectangle(image_to_draw, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
        text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image_to_draw, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), box_color, cv2.FILLED)
        cv2.putText(image_to_draw, text, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
else:
    print(f"No objects found with a confidence score above {model.config['score_thresh']}.")

# 8. Display the final image with our custom drawings
final_image_rgb = cv2.cvtColor(image_to_draw, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(final_image_rgb)
plt.title(f"Custom Predictions on {image_path_obj.name}")
plt.axis('off')
plt.show()